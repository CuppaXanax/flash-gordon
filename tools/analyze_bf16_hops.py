#!/usr/bin/env python3
"""Analyze five actual-codec paired hop rounds and offline full-vocabulary quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def ci(values):
    if len(values) != 5 or not all(math.isfinite(x) for x in values):
        raise ValueError("requires five finite round observations")
    return dict(mean=statistics.mean(values),
                ci95_half=2.776445105*statistics.stdev(values)/math.sqrt(5))


def transport(receipt):
    if {r["IP"] for r in receipt} != {"192.168.42.52", "192.168.42.53"}:
        raise ValueError("wrong measurement pair")
    if any(r["ExitCode"] or r["Stderr"] for r in receipt):
        raise ValueError("failed pair recording")
    rows = [json.loads(line) for r in receipt for line in r["Stdout"].splitlines() if line]
    by_key = {(r["round"], r["leg"], r["ngram"]): r for r in rows}
    if len(rows) != 30 or len(by_key) != 30:
        raise ValueError("incomplete/duplicate five-round A/B/A recording")
    rounds = []
    shapes = {False: [], True: []}
    for number in range(5):
        result = dict(round=number)
        for ngram in (False, True):
            legs = [by_key[(number, leg, ngram)] for leg in range(3)]
            for leg, r in enumerate(legs):
                expected = leg != 1 if number % 2 else leg == 1
                values = 10240 + (2560 if ngram else 0)
                if (r["bf16"] != expected or r["samples"] != 500 or
                    r["payload_bytes"] != 24+values*(2 if expected else 4) or
                    r["frame_bytes"] != r["payload_bytes"]+32 or
                    not r["codec_validated"]):
                    raise ValueError("unexpected shape, mode, sample count or codec gate")
                if not all(math.isfinite(r[k]) and r[k] > 0 for k in
                           ("rtt_mean_us", "rtt_p50_us", "rtt_p99_us")):
                    raise ValueError("invalid timings")
            a = (legs[0]["rtt_p50_us"]+legs[2]["rtt_p50_us"])/4
            b = legs[1]["rtt_p50_us"]/2
            f32, bf16 = (b, a) if number % 2 else (a, b)
            row = dict(round=number, f32_hop_us=f32, bf16_hop_us=bf16,
                       saving_us=f32-bf16)
            shapes[ngram].append(row)
            result["initial" if ngram else "ordinary"] = row
        result["modeled_f32_us"] = 7*result["ordinary"]["f32_hop_us"]+result["initial"]["f32_hop_us"]
        result["modeled_bf16_us"] = 7*result["ordinary"]["bf16_hop_us"]+result["initial"]["bf16_hop_us"]
        result["modeled_saving_us"] = result["modeled_f32_us"]-result["modeled_bf16_us"]
        rounds.append(result)
    return dict(method="RTT/2 with real work encode/decode, framing, CRC and validation; not synchronized one-way timing",
                paired_rounds=5, samples_per_leg=500, warmup_per_leg=50,
                raw_legs=rows, rounds=rounds,
                ordinary={k: ci([r[k] for r in shapes[False]]) for k in
                          ("f32_hop_us", "bf16_hop_us", "saving_us")},
                initial={k: ci([r[k] for r in shapes[True]]) for k in
                         ("f32_hop_us", "bf16_hop_us", "saving_us")},
                eight_work_hop_model={k: ci([r[k] for r in rounds]) for k in
                                     ("modeled_f32_us", "modeled_bf16_us", "modeled_saving_us")},
                work_frames_f32_bytes=338368, work_frames_bf16_bytes=169408,
                physical_payload_serialization_saving_us_at_1Gbps=1351.68,
                scope="Eight sequential work transfers only; unchanged output/head, prefill, MTP verification and other work excluded. Not measured serving TPS.")


def quality(rows):
    if len(rows) != 320 or len({(r["case"], r["position"]) for r in rows}) != 320:
        raise ValueError("expected 80 contexts x four teacher-forced positions")
    if {r["case"] for r in rows} != set(range(80)):
        raise ValueError("wrong frozen corpus cases")
    for r in rows:
        if r["vocab"] != 248320 or not math.isfinite(r["kl"]) or r["kl"] < 0:
            raise ValueError("invalid full-vocabulary observation")
        if not all(isinstance(r[k], int) and 0 <= r[k] < r["vocab"]
                   for k in ("teacher_top1", "bf16_top1")):
            raise ValueError("invalid top-1 token")
    for case in range(80):
        items = sorted((r for r in rows if r["case"] == case), key=lambda r: r["position"])
        if (len(items) != 4 or
            [r["position"] for r in items] != list(range(items[0]["position"], items[0]["position"]+4)) or
            [r["final"] for r in items] != [False, False, False, True]):
            raise ValueError("requires four consecutive trailing positions per context")
    primary = [r for r in rows if r["final"]]
    if len(primary) != 80 or len({r["case"] for r in primary}) != 80:
        raise ValueError("missing final-position quality gate")
    result = {}
    for name, items in (("primary_80", primary), ("all_320", rows)):
        kl = statistics.mean(r["kl"] for r in items)
        agreement = statistics.mean(r["teacher_top1"] == r["bf16_top1"] for r in items)
        result[name] = dict(positions=len(items), kl_mean=kl,
                            kl_max=max(r["kl"] for r in items), top1_agreement=agreement,
                            passed=kl <= .005 and agreement >= .985)
    result["method"] = ("Sealed trained pack; frozen Q4 textual contexts. Both paths decode the common prefix in f32; "
                        "only the last four supplied tokens use BF16 at all eight work boundaries. "
                        "Independent causal owner state, unchanged f32 kernels/accumulation, full-vocabulary KL(P_f32||P_bf16).")
    result["observations"] = rows
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", type=Path, required=True)
    p.add_argument("--quality", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = dict(transport=transport(json.loads(a.pair.read_text(encoding="utf-8-sig"))))
    if a.quality:
        data["quality"] = quality([json.loads(s) for s in a.quality.read_text().splitlines() if s])
    data["input_sha256"] = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (a.pair, a.quality) if path}
    a.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
