#!/usr/bin/env python3
"""Fail-closed MTP pass-local correctness and original-Q8 model-quality gate."""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

TOP2_MARGIN_MAX = .10
MEAN_KL_MAX = .005
TOP1_MIN = .985
Q8_MANIFEST = "35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb"
MIX04_MANIFEST = "0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252"
CORPUS80 = "ad99ce71c9f87edc6e876fa956223b058036ebb718c9a198f9c908023011446c"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def evaluate(events, quality):
    rows = events.get("rows", [])
    requests = events.get("requests", [])
    expected, seen, counts, tokens = {}, set(), {}, {}
    coverage = bool(rows and requests)
    for request in requests:
        key = request.get("id")
        count = request.get("tokens")
        if not isinstance(key, str) or key in expected or type(count) is not int or count <= 0:
            coverage = False
            continue
        expected[key] = request
        counts[key] = 0
        tokens[key] = []
    consistent = margins = True
    divergences = 0
    for row in rows:
        key, position = row.get("id"), row.get("position")
        if (key not in expected or type(position) is not int or
                (key, position) in seen or position != counts.get(key)):
            coverage = False
            continue
        seen.add((key, position))
        counts[key] += 1
        token, target, off = row.get("emitted"), row.get("target_top1"), row.get("off_top1")
        valid = all(type(x) is int and 0 <= x < 248320 for x in (token, target, off))
        consistent &= valid and token == target
        tokens[key].append(token if valid else 0)
        prefix = row.get("target_prefix_sha256")
        margins &= (isinstance(prefix, str) and bool(re.fullmatch("[0-9a-f]{64}", prefix)) and
                    prefix == row.get("off_prefix_sha256"))
        if valid and target != off:
            divergences += 1
            values = (row.get("off_margin"), row.get("target_margin"))
            margins &= (all(finite(x) and 0 <= x < TOP2_MARGIN_MAX for x in values)
                        and row.get("off_top2") == target and row.get("target_top2") == off)
    for key, request in expected.items():
        coverage &= counts[key] == request["tokens"]
        digest = hashlib.sha256(b"".join(struct.pack("<I", x) for x in tokens[key])).hexdigest()
        coverage &= digest == request.get("tokens_sha256")
    mean, agreement = quality.get("mean_kld"), quality.get("top1_agreement")
    quality_ok = (quality.get("full_vocab") == 248320 and
                  type(quality.get("samples")) is int and quality["samples"] == 80 and
                  quality.get("candidate_columns") == 4 and
                  quality.get("reference_manifest_sha256") == Q8_MANIFEST and
                  quality.get("candidate_manifest_sha256") == MIX04_MANIFEST and
                  quality.get("corpus_sha256") == CORPUS80 and
                  quality.get("reference_kind") == "original-Q8" and
                  quality.get("same_prefix_geometry") is True and
                  quality.get("frozen_corpus_matched") is True and
                  finite(mean) and 0 <= mean <= MEAN_KL_MAX and
                  finite(agreement) and TOP1_MIN <= agreement <= 1 and
                  abs(agreement*80-round(agreement*80)) < 1e-9)
    return dict(passed=bool(coverage and consistent and margins and quality_ok),
                self_consistency=bool(coverage and consistent),
                small_margin_divergences=bool(coverage and margins),
                quality_passed=bool(quality_ok), compared_emissions=len(seen),
                divergences=divergences, margin_threshold_exclusive=TOP2_MARGIN_MAX,
                mean_kl_max=MEAN_KL_MAX, top1_min=TOP1_MIN,
                bitwise_n1_n4_identity_required=False,
                scope="every declared emission at the same prefix; full-vocabulary frozen-Q8 quality separately")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "quality", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.events.read_text()), json.loads(args.quality.read_text()))
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
