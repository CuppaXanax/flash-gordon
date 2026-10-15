#!/usr/bin/env python3
"""Check BF16 name/shape/layout against Q8 dense rows in the published source."""
import argparse
import json
from pathlib import Path
import struct
from q4_source import header, fetch, REPO, REV


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    _, tensors, base, _ = header(str(args.bf16))
    bf16 = {t["name"]: t for t in tensors}
    reference = {}
    for shard in range(1, 5):
        name = f"unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-{shard:05}-of-00004.gguf"
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
        _, ts, start, _ = header(url)
        for t in ts:
            if t["name"] in bf16 and t["type"] == 8:
                reference[t["name"]] = (t, url, start)
    names = ["output.weight", "output_hc_down.weight", "output_hc_up.weight",
             "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
             "blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight",
             "blk.0.hc_ffn_down.weight", "blk.0.hc_ffn_up.weight",
             "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight",
             "blk.0.ffn_down_shexp.weight", "blk.3.attn_q.weight",
             "blk.3.attn_k.weight", "blk.3.attn_v.weight",
             "blk.3.attn_output.weight", "blk.1.ple_key.weight",
             "blk.1.ple_value.weight", "blk.47.attn_q.weight"]
    results = []
    with args.bf16.open("rb") as src:
        for name in names:
            t, url, reference_base = reference[name]
            b = bf16[name]
            if b["type"] != 30 or t["shape"] != b["shape"]:
                raise ValueError(f"mapping type/shape mismatch: {name}")
            count = t["shape"][0] * t["shape"][1]
            max_ratio = 0.0
            for block in (0, count//64, count//32-1):
                raw = fetch(url, reference_base+t["offset"]+block*34, 34)
                d, = struct.unpack("<e", raw[:2])
                q = struct.unpack("<32b", raw[2:])
                src.seek(base+b["offset"]+block*64)
                words = struct.unpack("<32H", src.read(64))
                vals = [struct.unpack("<f", struct.pack("<I", w << 16))[0] for w in words]
                # Q8 uses rounded FP16 scales; allow scale rounding across 127 steps.
                tolerance = abs(d)*(.51+127/1024)+1e-9
                max_ratio = max(max_ratio, max(abs(v-d*a)/tolerance for v, a in zip(vals, q)))
            row = dict(name=name, shape=b["shape"], sampled_blocks=3,
                       max_error_over_q8_bound=max_ratio, passed=max_ratio <= 1)
            print(json.dumps(row), flush=True)
            results.append(row)
    report = dict(repo=REPO, revision=REV, baseline="published abliterated UD-Q4_K_XL",
                  current_pack_identity_verified=False, passed=all(r["passed"] for r in results),
                  tensors=results)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
