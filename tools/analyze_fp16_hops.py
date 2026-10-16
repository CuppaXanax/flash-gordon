#!/usr/bin/env python3
"""Cumulative Q8-reference KL and causal single-hop attribution, never summed KL."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_bf16_hops import ci, quality, transport
from q4_quality import compare, contexts, read


def metrics(rows):
    values=[r["kl"] for r in rows]
    agreement=statistics.mean(r["top1_agrees"] for r in rows)
    mean=statistics.mean(values)
    return dict(positions=len(rows),kl_mean=mean,kl_max=max(values),
                top1_agreement=agreement,passed=mean<=.005 and agreement>=.985)


def scaled_transport(receipt, codec):
    """Normalize only validated scale overhead; reuse the original paired math."""
    normalized=[]
    for item in receipt:
        copy=dict(item);lines=[]
        for line in item["Stdout"].splitlines():
            row=json.loads(line);compressed=row["compressed"]
            assert row["codec"]==(codec if compressed else "f32")
            assert row["bf16"] is False
            count=10240+(2560 if row["ngram"] else 0)
            overhead=(4*((count+127)//128) if codec=="fp16-128" else 4) if compressed else 0
            row["payload_bytes"]-=overhead;row["frame_bytes"]-=overhead
            row["bf16"]=compressed
            lines.append(json.dumps(row))
        copy["Stdout"]="\n".join(lines);normalized.append(copy)
    result=transport(normalized)
    overhead=2640 if codec=="fp16-128" else 32
    result["work_frames_bf16_bytes"]+=overhead
    result["physical_payload_serialization_saving_us_at_1Gbps"]=(338368-result["work_frames_bf16_bytes"])/125
    # The original API names the smaller mode BF16; do not mislabel the new codec.
    def rename(value):
        if isinstance(value,dict):
            return {k.replace("bf16","fp16"):rename(v) for k,v in value.items()}
        if isinstance(value,list):return [rename(v) for v in value]
        return value
    result=rename(result)
    result["codec"]=codec
    result["raw_legs"]=[json.loads(line) for item in receipt for line in item["Stdout"].splitlines()]
    if codec=="fp16":
        result["mixed_first_f32_model"]=dict(
            saving_us=ci([7*r["ordinary"]["saving_us"] for r in result["rounds"]]),
            candidate_us=ci([7*r["ordinary"]["fp16_hop_us"]+r["initial"]["f32_hop_us"] for r in result["rounds"]]),
            method="Initial work remains f32; seven ordinary scaled-FP16 transfers. Modeled, not ring TPS.")
        result["mixed_no_rank0_model"]=dict(
            saving_us=ci([6*r["ordinary"]["saving_us"]+r["initial"]["saving_us"] for r in result["rounds"]]),
            candidate_us=ci([6*r["ordinary"]["fp16_hop_us"]+r["ordinary"]["f32_hop_us"]+r["initial"]["fp16_hop_us"] for r in result["rounds"]]),
            method="Destination0 ordinary work remains f32; initial and six ordinary transfers scaled FP16. Modeled, not ring TPS.")
    return result


def matched_q8(control):
    manifest=control/"original-q8.fgm"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=="35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb"
    summaries=[]
    for rank in range(8):
        rows=[json.loads(s) for s in (control/f"q8-rank-{rank}-restored.jsonl").read_text().splitlines()]
        summary=[r for r in rows if r.get("kind")=="q8_rank_restored"]
        assert len(summary)==1 and summary[0]["rank"]==rank
        assert summary[0]["trained_spans_sha_exact"]
        summaries.extend(summary)
    assert sum(r["spans"] for r in summaries)==1223
    assert sum(r["copied"] for r in summaries)==824
    assert sum(r["cooked"] for r in summaries)==351
    assert sum(r["lossless_routers"] for r in summaries)==48
    numerical=[json.loads(s) for s in (control/"q8-quality.jsonl").read_text().splitlines()]
    assert len(numerical)==320 and all(abs(r["kl"])<=1e-12 for r in numerical)
    final={r["case"]:r for r in numerical if r["final"]}
    assert set(final)==set(range(80))
    logits=[]
    for case in range(80):
        values=read(control/"logits"/"q8"/f"{case:03d}.fgl")
        assert hashlib.sha256(values.tobytes()).hexdigest()==final[case]["bf16_logits_sha256"]
        logits.append(values)
    binary=(control/"q8-head.sha256").read_text().split()[0]
    assert binary=="11ce46e6cb662c4d70509f28301c50a1bb3ff1dd9be49b1516cd9cd453f57239"
    for first in range(0,48,6):
        stage=[json.loads(s) for s in (control/f"q8-stage-{first}.jsonl").read_text().splitlines()]
        assert len(stage)==81 and stage[-1]["prefix_max_relative"]<=1e-4
        assert stage[-1]["codec"]=="f32"
    return logits,dict(method="Original-Q8 trained spans restored SHA-exact; identical frozen decode geometry, compiler binary, kernels, supplied tokens and prepared embedding/ngram inputs. No historical-prefill geometry substitution.",
                       numerical_binary_sha256=binary,rank_reconstruction=summaries)


def analyze(root, reference, previous, control=None):
    corpus=json.loads((reference/"corpus.json").read_text(encoding="utf-8-sig"))
    assert corpus==[list(r) for r in contexts()]
    original_manifest=reference.parent/"baseline.fgm"
    capture=[json.loads(s) for s in (reference/"fleet.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    assert capture and all(r["ExitCode"]==0 for r in capture)
    manifest_sha=hashlib.sha256(original_manifest.read_bytes()).hexdigest()
    assert manifest_sha=="35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb"
    assert bytes.fromhex("4708a6600142d094") in original_manifest.read_bytes()
    assert any("manifest_sha=4708a6600142d094" in r["Stderr"] for r in capture)
    previous_rows=[json.loads(s) for s in previous.read_text().splitlines()]
    previous_final={r["case"]:r for r in previous_rows if r["final"]}
    historical_q8=[read(reference/f"{key}.fgl") for key,_ in corpus]
    q8,control_identity=matched_q8(control) if control else (historical_q8,None)
    baseline=[read(root/"logits"/"f32"/f"{c:03d}.fgl") for c in range(80)]
    identities={}
    for c,(key,_) in enumerate(corpus):
        digest=hashlib.sha256(baseline[c].tobytes()).hexdigest()
        assert digest==previous_final[c]["teacher_logits_sha256"],"f32 baseline drift"
        identities[key]=dict(q8_file_sha256=hashlib.sha256((reference/f"{key}.fgl").read_bytes()).hexdigest(),
                             mix04_logits_sha256=digest)
    result=dict(reference=dict(corpus_sha256=hashlib.sha256((reference/"corpus.json").read_bytes()).hexdigest(),
                    manifest_sha256=manifest_sha,
                    capture_receipt_sha256=hashlib.sha256((reference/"fleet.jsonl").read_bytes()).hexdigest(),
                    method="Retained original-Q8 full-vocabulary captures on byte/text-identical frozen80 contexts; historical serving geometry. Direct KL(Q8||mix04+codec), not KL addition. New f32 mix04 baseline reproduces prior teacher logits SHA-exact.",
                    captures=identities),policies={})
    if control_identity:
        result["matched_q8_reference"]=control_identity
        result["reference_geometry_comparison"]=metrics([
            dict(kl=compare(historical_q8[c],q8[c])[0],
                 top1_agrees=compare(historical_q8[c],q8[c])[1]) for c in range(80)])
        result["reference"]["method"]="cumulative_vs_q8 retains the historical original-Q8 captures; matched_decode_cumulative_vs_q8 is a separate authentic-Q8 matched-geometry control. Neither silently replaces the other. Direct KL, never added/subtracted as error shares."
    result["screened_policy"]=json.loads((root/"fp16-screened-policy.json").read_text())
    names=["f32","fp16","fp16-128","bf16","mixed-first-f32","mixed-no-rank0","mixed-screened"]+[f"single-{r}" for r in range(8)]
    for name in names:
        numerical=[json.loads(s) for s in (root/f"{name}-quality.jsonl").read_text().splitlines()]
        incremental=quality(numerical)
        incremental["method"]=f"Frozen80 contexts/four trailing positions, causal trained layers/state; policy {name}. Actual codec at selected work boundaries, f32 common prefix."
        final={r["case"]:r for r in numerical if r["final"]}
        observations=[];matched_observations=[]
        for c in range(80):
            logits=read(root/"logits"/name/f"{c:03d}.fgl")
            assert hashlib.sha256(logits.tobytes()).hexdigest()==final[c]["bf16_logits_sha256"]
            kld,agree=compare(q8[c],logits)
            delta,agrees=compare(baseline[c],logits)
            assert math.isclose(delta,final[c]["kl"],abs_tol=1e-10)
            assert agrees==(final[c]["teacher_top1"]==final[c]["bf16_top1"])
            historical_delta,historical_agrees=compare(historical_q8[c],logits) if control else (kld,agree)
            observations.append(dict(case=c,kl=historical_delta,top1_agrees=historical_agrees,
                matched_decode_kl=kld if control else None,
                matched_decode_top1_agrees=agree if control else None,
                incremental_kl=delta,incremental_top1_agrees=agrees))
            matched_observations.append(dict(case=c,kl=kld,top1_agrees=agree))
        result["policies"][name]=dict(cumulative_vs_q8=metrics(observations),
            matched_decode_cumulative_vs_q8=metrics(matched_observations) if control else None,
            incremental_vs_mix04={k:v for k,v in incremental.items() if k!="observations"},
            observations=observations)
    order=[1,0,2,3,4,5,6,7]
    ranked=sorted(range(8),key=lambda rank:result["policies"][f"single-{rank}"]["incremental_vs_mix04"]["primary_80"]["kl_mean"],reverse=True)
    result["attribution"]=dict(method="Each single-rank policy compresses only that destination's work boundary; all other boundaries remain f32. Complete subsequent trained layers/state/head propagate the perturbation. Isolated KL values are not additive.",
        ranked_destinations=ranked,rows=[dict(destination_rank=r,hop_index=order.index(r),
            layers_first=order.index(r)*6,layers_last=order.index(r)*6+5,
            isolated_kl_mean=result["policies"][f"single-{r}"]["incremental_vs_mix04"]["primary_80"]["kl_mean"]) for r in ranked])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--q8-control",type=Path)
    parser.add_argument("--pair",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root,args.reference,args.previous,args.q8_control)
    if args.pair:
        pair=json.loads(args.pair.read_text(encoding="utf-8-sig"));result["transport"]={}
        for codec in ("fp16","fp16-128"):
            selected=[]
            for item in pair:
                copy=dict(item)
                # Each two-codec recording contains f32 legs for both: use round ordering.
                all_rows=[json.loads(s) for s in item["Stdout"].splitlines()]
                start=0 if codec=="fp16" else 30
                copy["Stdout"]="\n".join(json.dumps(r) for r in all_rows[start:start+30])
                selected.append(copy)
            result["transport"][codec]=scaled_transport(selected,codec)
        mask=result["screened_policy"]["mask"]
        initial=bool(mask&2);ordinary=mask.bit_count()-initial
        rounds=result["transport"]["fp16"]["rounds"]
        savings=[ordinary*r["ordinary"]["saving_us"]+initial*r["initial"]["saving_us"] for r in rounds]
        candidates=[(7-ordinary)*r["ordinary"]["f32_hop_us"]+ordinary*r["ordinary"]["fp16_hop_us"]+
                    (0 if initial else 1)*r["initial"]["f32_hop_us"]+initial*r["initial"]["fp16_hop_us"] for r in rounds]
        result["transport"]["fp16"]["mixed_screened_model"]=dict(
            mask=mask,saving_us=ci(savings),candidate_us=ci(candidates),
            method="Same paired-round model for the replayed, in-corpus screened destination mask; not serving timing.")
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
