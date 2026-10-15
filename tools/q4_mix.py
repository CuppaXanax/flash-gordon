#!/usr/bin/env python3
"""Splice an explicit family/tensor Q4/Q5/Q6/Q8 plan without requantizing payloads."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from q4_source import header, string, SIZES


def family(name):
    if ".attn_qkv.weight" in name: return "gdn-qkv"
    if ".ssm_out.weight" in name: return "gdn-out"
    if ".attn_gate.weight" in name: return "gdn-other"
    if any(f".attn_{x}.weight" in name for x in ("q","k","v","output")): return "qsa"
    for suffix in ("gate","up","down"):
        if f".ffn_{suffix}_shexp.weight" in name: return f"shared-{suffix}"
    if ".hc_" in name or name.startswith("output_hc_"): return "hc"
    return "other"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def mix(paths, plan, output, changed_families_only=False):
    if "q8" not in paths:
        raise ValueError("missing deployed Q8 reference")
    if set(plan)-{"families","tensors"}:
        raise ValueError("unknown plan field")
    policies, exact = plan.get("families",{}), plan.get("tensors",{})
    families = {"gdn-qkv","gdn-out","gdn-other","qsa","shared-gate",
                "shared-up","shared-down","hc","other"}
    if set(policies)-families or any(v not in ("q4","q5","q6","q8")
                                   for v in list(policies.values())+list(exact.values())):
        raise ValueError("unknown family or precision")
    source, ledgers = {}, {}
    for key, path in paths.items():
        md, ts, base, alignment = header(str(path))
        source[key] = dict(base=base,tensors={t["name"]:t for t in ts})
        ledgers[key] = dict(path=str(path),sha256=sha256(path))
        if key == "q8":
            metadata, reference, align = md, ts, alignment
    if set(exact)-{t["name"] for t in reference}:
        raise ValueError("unknown tensor override")
    chosen, cursor = [], 0
    for ref in reference:
        name = ref["name"]
        if ref["type"] != 8 or len(ref["shape"]) != 2 or name in ("output.weight","token_embd.weight"):
            raise ValueError("reference must contain nonhead deployed Q8 matrices only")
        group = family(name)
        policy = exact.get(name,policies.get(group,"q8"))
        # K formats cannot encode K=320/640. Never label IQ4_NL as Q5/Q6.
        if policy in ("q5","q6") and ref["shape"][0] % 256: policy = "q8"
        if policy not in source:
            raise ValueError(f"missing {policy} source")
        t = source[policy]["tensors"].get(name)
        expected = {"q8":8,"q5":13,"q6":14}.get(policy,20 if ref["shape"][0]%256 else 12)
        if not t or t["shape"] != ref["shape"] or t["type"] != expected:
            raise ValueError(f"source type/shape mismatch: {name}")
        item = dict(t,policy=policy,family=group,source_offset=source[policy]["base"]+t["offset"])
        block, size = SIZES[t["type"]]
        item["bytes"] = math.prod(t["shape"])//block*size
        cursor = (cursor+align-1)//align*align
        item["offset"], cursor = cursor, cursor+item["bytes"]
        chosen.append(item)
    changed = {t["family"] for t in chosen if t["type"] != 8}
    if changed_families_only:
        if not changed: raise ValueError("no changed families")
        chosen = [t for t in chosen if t["family"] in changed]
    routers = []
    if "routers" in source:
        routers = [t for t in source["routers"]["tensors"].values()
                   if t["name"].endswith(".ffn_gate_inp.weight") and t["type"] == 30]
        if not routers or any(len(t["shape"]) != 2 for t in routers):
            raise ValueError("missing/invalid BF16 routers; native repacker must prove losslessness")
        for t in routers:
            chosen.append(dict(t,policy="routers",family="router",
                source_offset=source["routers"]["base"]+t["offset"],
                bytes=math.prod(t["shape"])*2))
    cursor = 0
    for t in chosen:
        cursor = (cursor+align-1)//align*align
        t["offset"], cursor = cursor, cursor+t["bytes"]
    h = b"GGUF"+struct.pack("<IQQ",3,len(chosen),len(metadata))+b"".join(metadata)
    for t in chosen:
        h += string(t["name"])+struct.pack("<IQQIQ",2,*t["shape"],t["type"],t["offset"])
    h += bytes((-len(h))%align)
    with ExitStack() as stack:
        files = {k:stack.enter_context(p.open("rb")) for k,p in paths.items()}
        dst = stack.enter_context(output.open("xb"))
        dst.write(h)
        for t in chosen:
            src = files[t["policy"]]
            src.seek(t["source_offset"]);dst.seek(len(h)+t["offset"])
            digest, remaining = hashlib.sha256(), t["bytes"]
            while remaining:
                raw = src.read(min(remaining,8<<20))
                if not raw: raise ValueError("truncated mix source")
                dst.write(raw);digest.update(raw);remaining -= len(raw)
            t["sha256"] = digest.hexdigest()
        dst.flush();os.fsync(dst.fileno())
    result = dict(plan=plan,sources=ledgers,sha256=sha256(output),tensors=chosen,
                  changed_families=sorted(changed),changed_families_only=changed_families_only,
                  bf16_router_count=len(routers))
    output.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q4",type=Path)
    p.add_argument("--q5",type=Path)
    p.add_argument("--q6",type=Path)
    p.add_argument("--q8",type=Path,required=True)
    p.add_argument("--routers",type=Path,
                   help="append only BF16 MoE routers; repacker must prove every value against deployed F32")
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--changed-families-only",action="store_true",
                   help="omit unchanged families; retain Q8 fallback tensors inside changed families")
    args=p.parse_args()
    paths={k:getattr(args,k) for k in ("q4","q5","q6","q8","routers") if getattr(args,k)}
    result=mix(paths,json.loads(args.plan.read_text()),args.output,args.changed_families_only)
    print(result["sha256"],"families="+",".join(result["changed_families"]))


if __name__=="__main__":
    main()
