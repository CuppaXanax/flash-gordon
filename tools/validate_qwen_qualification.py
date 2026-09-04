#!/usr/bin/env python3
"""Validate a native-context client curve and its optional server join."""
import argparse
import json
import statistics
import sys
from collections import defaultdict

SCHEMA = "fg.qwen.qual.validation.v1"
REQUIRED_CHECKPOINTS = (4096, 8192, 16384, 32768, 65536, 131072, 196608, 261888)


def load(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def median(values):
    return statistics.median(values) if values else None


def sampling_matches(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return (left.get("top_k") == right.get("top_k") and
            all(isinstance(left.get(field), (int, float)) and
                isinstance(right.get(field), (int, float)) and
                abs(float(left[field]) - float(right[field])) <= 1e-6
                for field in ("temperature", "top_p")))


def validate(client, server=None, stage=None, tolerance_ratio=0.05, tolerance_tokens=512,
             min_prefill=250.0, min_decode=50.0, drop_ratio=0.35,
             discontinuity_ratio=2.0):
    errors, warnings = [], []
    client = [r for r in client if r.get("schema") == "fg.qwen.qual.client.v1"]
    server_records = [r for r in (server or [])
                      if r.get("schema") == "fg.qwen.qual.server.v1"]
    client_ids = [r.get("request_id") for r in client]
    server_ids = [r.get("request_id") for r in server_records]
    for request_id in sorted({value for value in client_ids if value is not None}):
        if client_ids.count(request_id) > 1:
            errors.append("duplicate client request ID %s" % request_id)
    for request_id in sorted({value for value in server_ids if value is not None}):
        if server_ids.count(request_id) > 1:
            errors.append("duplicate server request ID %s" % request_id)
    if any(value is None for value in client_ids):
        errors.append("client record missing request ID")
    if server is None:
        errors.append("server summaries are required for qualification")
    server_by_id = {r.get("request_id"): r for r in server_records}
    joined = []
    ranges = set()
    previous_end = 0
    for record in client:
        request_id = record.get("request_id")
        if record.get("http_status") != 200 or record.get("error"):
            errors.append("client request failed for %s" % request_id)
            continue
        if not record.get("finish_reason"):
            errors.append("client request has no finish reason for %s" % request_id)
        if request_id in server_by_id:
            merged = dict(record)
            merged["server"] = server_by_id[request_id]
            joined.append(merged)
            s = server_by_id[request_id]
            if s.get("http_status") != 200 or s.get("status") != "ok":
                errors.append("server request failed for %s" % request_id)
            if s.get("prompt_tokens") != record.get("prompt_tokens"):
                errors.append("usage mismatch for %s" % request_id)
            if (s.get("generated_tokens") is not None and
                    s.get("generated_tokens") != record.get("completion_tokens")):
                errors.append("completion usage mismatch for %s" % request_id)
            if s.get("prompt_tokens", 0) != s.get("reused_tokens", 0) + s.get("prefilled_tokens", 0):
                errors.append("reused+prefilled accounting mismatch for %s" % request_id)
        elif server is not None:
            errors.append("missing server summary for %s" % request_id)
        sampling = record.get("sampling") or {}
        if sampling.get("temperature", 0) <= 0 or not 0 < sampling.get("top_p", 0) < 1 or sampling.get("top_k") != 20:
            errors.append("sampling policy violation for %s" % request_id)
        if (request_id in server_by_id and
                not sampling_matches(server_by_id[request_id].get("sampling"), sampling)):
            errors.append("client/server sampling mismatch for %s" % request_id)
        start, end = record.get("corpus_char_start"), record.get("corpus_char_end")
        if start is None or end is None or end < start:
            errors.append("invalid corpus range for %s" % request_id)
        elif end > start:
            if (start, end) in ranges:
                errors.append("duplicate corpus range %d:%d" % (start, end))
            ranges.add((start, end))
            if start != previous_end:
                errors.append("non-contiguous corpus range %d:%d after %d" % (start, end, previous_end))
            previous_end = end
    if not client:
        errors.append("no client records")
    contexts = [(r.get("prompt_tokens"), r) for r in client
                if r.get("http_status") == 200 and isinstance(r.get("prompt_tokens"), int)]
    if any(contexts[i][0] < contexts[i - 1][0] for i in range(1, len(contexts))):
        errors.append("non-monotonic prompt context")
    successful = [r for r in joined if r.get("server")]
    for index, record in enumerate(successful[1:], 1):
        server_record = record["server"]
        if (server_record.get("reset_reason") in ("cold-start", "explicit", "failure") or
                (server_record.get("reused_tokens") == 0 and
                 (record.get("prompt_tokens") or 0) > 4096)):
            errors.append("cold prefix reset after first request at %s" % record.get("request_id"))
    by_target = defaultdict(list)
    for r in successful:
        by_target[r.get("checkpoint_target")].append(r)
    coverage = []
    for target in REQUIRED_CHECKPOINTS:
        eligible = [r for r in successful if r.get("prompt_tokens", 0) >= target * (1 - tolerance_ratio)
                    and r.get("prompt_tokens", 0) <= target + tolerance_tokens]
        nearest = min(successful, key=lambda r: abs((r.get("prompt_tokens") or 0) - target),
                      default=None)
        coverage.append({"target": target,
                         "actual": None if nearest is None else nearest.get("prompt_tokens"),
                         "delta": None if nearest is None else (nearest.get("prompt_tokens") - target),
                         "tolerance_tokens": max(tolerance_tokens, int(target * tolerance_ratio)),
                         "covered": bool(eligible)})
        if not eligible:
            errors.append("missing coverage near %d" % target)
        elif target == 4096:
            rates = [r["server"].get("prefill_tps", 0) for r in eligible
                     if r.get("corpus_char_end", 0) > r.get("corpus_char_start", 0)
                     and r["server"].get("prefilled_tokens", 0) > 0
                     and r["server"].get("prefill_tps") is not None]
            if not rates:
                errors.append("missing incremental prefill sample near 4096")
            if rates and median(rates) < min_prefill:
                errors.append("4K incremental prefill median %.3f < %.3f" % (median(rates), min_prefill))
    groups = []
    for target, rows in by_target.items():
        checkpoint_rows = [r for r in rows if r.get("checkpoint_crossed")]
        rates = [r["server"].get("decode_tps") for r in checkpoint_rows
                 if r["server"].get("decode_tps") is not None]
        if not rates:
            errors.append("missing decode samples at target %s" % target)
            continue
        value = median(rates)
        groups.append((min((r.get("prompt_tokens") or 0) for r in checkpoint_rows),
                       value, target, len(rates)))
        if value < min_decode:
            errors.append("decode median %.3f < %.3f at target %s" % (value, min_decode, target))
        if len(rates) < 3:
            errors.append("only %d decode samples at target %s" % (len(rates), target))
    groups.sort()
    adjacent = []
    for left, right in zip(groups, groups[1:]):
        if right[0] > left[0] and right[0] <= left[0] * 2 and right[1] < left[1] * (1 - drop_ratio):
            errors.append("adjacent decode drop %.1f%% from %s to %s" %
                          ((1 - right[1] / left[1]) * 100, left[2], right[2]))
        adjacent.append({"from_context": left[0], "to_context": right[0],
                         "from_tps": left[1], "to_tps": right[1],
                         "ratio": right[1] / left[1] if left[1] else None})
    for index in range(1, len(groups) - 1):
        before, current, after = groups[index - 1:index + 2]
        if current[1] > discontinuity_ratio * before[1] and current[1] > discontinuity_ratio * after[1]:
            errors.append("two-neighbor discontinuity at target %s" % current[2])
    prefill_groups = []
    for target, rows in by_target.items():
        rates = [r["server"].get("prefill_tps") for r in rows
                 if r.get("checkpoint_crossed")
                 and r.get("corpus_char_end", 0) > r.get("corpus_char_start", 0)
                 and r["server"].get("prefilled_tokens", 0) > 0
                 and r["server"].get("prefill_tps") is not None]
        if rates:
            crossed = [r for r in rows if r.get("checkpoint_crossed")]
            prefill_groups.append((min(r.get("prompt_tokens") or 0 for r in crossed),
                                   median(rates), target, len(rates)))
    prefill_groups.sort()
    for left, right in zip(prefill_groups, prefill_groups[1:]):
        if (right[0] > left[0] and right[0] <= left[0] * 2 and
                right[1] < left[1] * (1 - drop_ratio)):
            errors.append("adjacent prefill drop %.1f%% from %s to %s" %
                          ((1 - right[1] / left[1]) * 100, left[2], right[2]))
    for index in range(1, len(prefill_groups) - 1):
        before, current, after = prefill_groups[index - 1:index + 2]
        if ((current[1] > discontinuity_ratio * before[1] and
             current[1] > discontinuity_ratio * after[1]) or
                (current[1] * discontinuity_ratio < before[1] and
                 current[1] * discontinuity_ratio < after[1])):
            errors.append("two-neighbor prefill discontinuity at target %s" % current[2])
        if current[1] * discontinuity_ratio < before[1] and current[1] * discontinuity_ratio < after[1]:
            errors.append("two-neighbor discontinuity at target %s" % current[2])
    if stage is not None:
        warnings.append("stage records are diagnostic only; no timing is joined without a request key")
    report = {"schema": SCHEMA, "pass": not errors, "errors": errors, "warnings": warnings,
              "records": len(client), "joined_records": len(successful),
              "stage_records": 0 if stage is None else len(stage),
              "coverage": coverage, "decode_curve": groups,
              "prefill_curve": prefill_groups, "adjacent_ratios": adjacent,
              "policy": {"min_prefill_tps_at_4k": min_prefill, "min_decode_tps": min_decode,
                         "aspirational_prefill_tps": 600.0, "aspirational_decode_tps": 100.0,
                         "adjacent_drop_ratio": drop_ratio,
                         "discontinuity_ratio": discontinuity_ratio}}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client_jsonl")
    parser.add_argument("--server-jsonl")
    parser.add_argument("--stage-jsonl", help="optional P5C stage JSONL for diagnostic counts")
    parser.add_argument("--output")
    parser.add_argument("--min-prefill-tps", type=float, default=250.0)
    parser.add_argument("--min-decode-tps", type=float, default=50.0)
    args = parser.parse_args(argv)
    client = load(args.client_jsonl)
    server = load(args.server_jsonl) if args.server_jsonl else None
    stage = load(args.stage_jsonl) if args.stage_jsonl else None
    report = validate(client, server, min_prefill=args.min_prefill_tps,
                      min_decode=args.min_decode_tps, stage=stage)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
