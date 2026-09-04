#!/usr/bin/env python3
"""Analyze Flash Gordon BC250/Qwen JSONL measurements.

The tool is intentionally offline and dependency-free.  It reports measured
service facts and mathematical bounds; it does not turn a bound into an
end-to-end TPS claim.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

RANK_COUNT = 8
KNOWN_SCHEMAS = {
    "fg.bc250.roofline.v1",
    "fg.bc250.qsa_curve.v1",
    "flash-gordon.profile",
}
STAGE_FIELDS = ("rank", "stage", "sequence", "execution_kind", "first_token",
                "tokens", "wall_ms", "gpu_ms", "kernel_ms", "submissions",
                "dispatches", "boundary_bytes", "terminal", "status")
FABRIC_FIELDS = ("local_rank", "peer", "fabric_class", "direction",
                  "message_type", "payload_bytes", "framed_bytes", "wall_ms",
                  "mode", "status")
KERNEL_FIELDS = ("rank", "stage", "sequence", "execution_kind", "first_token",
                 "tokens", "scope", "kernel", "calls", "gpu_ms", "status")


class ProfileError(ValueError):
    """Raised for malformed, incomplete, or ambiguous measurement input."""


def _number(value: Any, name: str, line: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"line {line}: {name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ProfileError(f"line {line}: {name} must be finite")
    return value


def _required(record: dict[str, Any], fields: Iterable[str], line: int) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ProfileError(f"line {line}: missing {', '.join(missing)}")


def _validate_record(record: Any, line: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ProfileError(f"line {line}: record must be a JSON object")
    schema = record.get("schema")
    if schema not in KNOWN_SCHEMAS:
        raise ProfileError(f"line {line}: unsupported or missing schema {schema!r}")
    if schema == "flash-gordon.profile":
        if record.get("version") != 1:
            raise ProfileError(f"line {line}: unsupported profile schema version")
        kind = record.get("record_type")
        fields = STAGE_FIELDS if kind == "stage_service" else FABRIC_FIELDS if kind == "fabric_service" else KERNEL_FIELDS if kind == "kernel_service" else ()
        if not fields:
            raise ProfileError(f"line {line}: unsupported profile record_type {kind!r}")
        _required(record, fields, line)
        if not isinstance(record["status"], int) or isinstance(record["status"], bool):
            raise ProfileError(f"line {line}: status must be an integer")
        rank = record.get("rank", record.get("local_rank"))
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank >= RANK_COUNT:
            raise ProfileError(f"line {line}: rank is outside 0..{RANK_COUNT - 1}")
        for field in ("wall_ms", "gpu_ms", "kernel_ms") if kind == "stage_service" else ("wall_ms",) if kind == "fabric_service" else ("gpu_ms",):
            if _number(record[field], field, line) < 0:
                raise ProfileError(f"line {line}: {field} must be nonnegative")
        if kind == "stage_service":
            if record["execution_kind"] not in ("decode", "prefill"):
                raise ProfileError(f"line {line}: invalid execution_kind")
            if not isinstance(record["tokens"], int) or record["tokens"] <= 0:
                raise ProfileError(f"line {line}: tokens must be a positive integer")
            if not isinstance(record["sequence"], int) or record["sequence"] < 0:
                raise ProfileError(f"line {line}: sequence must be a nonnegative integer")
            if not isinstance(record["stage"], int) or record["stage"] < 0 or record["stage"] >= RANK_COUNT:
                raise ProfileError(f"line {line}: stage is outside 0..{RANK_COUNT - 1}")
            if not isinstance(record["first_token"], int) or record["first_token"] < 0:
                raise ProfileError(f"line {line}: first_token must be a nonnegative integer")
            if not isinstance(record["terminal"], bool):
                raise ProfileError(f"line {line}: terminal must be boolean")
        elif kind == "fabric_service":
            if record["direction"] not in ("send", "receive"):
                raise ProfileError(f"line {line}: invalid fabric direction")
            if record["mode"] not in ("direct", "io_uring", "io_uring_batch"):
                raise ProfileError(f"line {line}: invalid fabric mode")
            for field in ("payload_bytes", "framed_bytes", "peer", "message_type"):
                if not isinstance(record[field], int) or record[field] < 0:
                    raise ProfileError(f"line {line}: {field} must be a nonnegative integer")
        else:
            if record["execution_kind"] not in ("decode", "prefill"):
                raise ProfileError(f"line {line}: invalid kernel execution_kind")
            if not isinstance(record["sequence"], int) or record["sequence"] < 0:
                raise ProfileError(f"line {line}: sequence must be a nonnegative integer")
            if not isinstance(record["calls"], int) or record["calls"] < 0:
                raise ProfileError(f"line {line}: calls must be a nonnegative integer")
    return record


def load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        try:
            stream = path.open(encoding="utf-8")
        except OSError as exc:
            raise ProfileError(f"cannot read {path}: {exc}") from exc
        with stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ProfileError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
                records.append(_validate_record(parsed, line_number))
    if not records:
        raise ProfileError("no JSONL records supplied")
    return records


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "min_ms": min(values),
            "p50_ms": statistics.median(ordered), "p95_ms": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .95) - 1)],
            "max_ms": max(values), "mean_ms": statistics.fmean(values)}


def _stage_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    stages = [record for record in records if record.get("record_type") == "stage_service"
              and record["status"] == 0]
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in stages:
        key = (record["execution_kind"], record["tokens"], record["first_token"],
               record.get("sequence_family", "default"))
        by_group[key].append(record)
    groups: list[dict[str, Any]] = []
    for (kind, tokens, first_token, family), group in sorted(by_group.items(), key=str):
        by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            by_stage[record["stage"]].append(record)
        stage_rows = []
        representative: dict[int, float] = {}
        for stage, rows in sorted(by_stage.items()):
            values = [float(row["wall_ms"]) for row in rows]
            row = {"stage": stage, "ranks": sorted({row["rank"] for row in rows}),
                   "service": _summary(values)}
            stage_rows.append(row)
            representative[stage] = statistics.median(values)
        complete = set(representative) == set(range(RANK_COUNT))
        stage_values = list(representative.values())
        max_service = max(stage_values) if stage_values else None
        total_service = sum(stage_values) if stage_values else None
        common = {
            "execution_kind": kind, "tokens": tokens, "first_token": first_token,
            "sequence_family": family,
            "records": len(group), "stages": stage_rows, "complete_stage_coverage": complete,
            "slowest_stage": max(representative, key=representative.get) if complete else None,
            "slowest_stage_service_ms": max_service if complete else None,
            "stage_service_sum_ms": total_service if complete else None,
            "ideal_balanced_stage_time_ms": total_service / len(stage_values) if complete else None,
            "imbalance_ratio": max_service / (total_service / len(stage_values)) if complete and total_service else None,
            "recoverable_imbalance_fraction": 1.0 - (total_service / len(stage_values)) / max_service if complete and max_service else None,
        }
        if kind == "prefill":
            common["prefill_pipeline_steady_state_upper_bound_tps"] = tokens * 1000.0 / max_service if complete and max_service else None
            common["pipeline_fill_ttft_service_ms"] = total_service
        else:
            by_sequence: dict[int, dict[int, float]] = defaultdict(dict)
            for record in group:
                sequence_stages=by_sequence[record["sequence"]]
                if record["stage"] in sequence_stages:
                    raise ProfileError("duplicate stage record in a decode sequence")
                sequence_stages[record["stage"]]=float(record["wall_ms"])
            sums = [sum(values.values()) for values in by_sequence.values()
                    if complete and set(values)==set(range(RANK_COUNT))]
            common["decode_comparable_sequences"] = len(sums)
            common["decode_observed_critical_path_sum_ms"] = _summary(sums) if sums else None
            common["decode_steady_state_max_stage_tps"] = 1000.0 / max_service if complete and max_service else None
            common["decode_sum_stage_tps_for_comparison"] = 1000.0 / statistics.median(sums) if sums else None
        groups.append(common)
    return {"groups": groups}


def _fabric_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    fabric = [record for record in records if record.get("record_type") == "fabric_service" and record.get("status", 0) == 0]
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in fabric:
        buckets[(record["fabric_class"], record["direction"], record["mode"], record["payload_bytes"])].append(record)
    rows = []
    for key, bucket in sorted(buckets.items(), key=str):
        cls, direction, mode, size = key
        walls = [float(record["wall_ms"]) for record in bucket]
        # A batch record's wall time covers the whole batch.  Without a shared
        # batch identifier and total byte count, per-item rates would be false.
        payload_rates = [] if mode == "io_uring_batch" else [size / wall / 1e6 for wall in walls if wall > 0]
        framed_rates = [] if mode == "io_uring_batch" else [record["framed_bytes"] / wall / 1e6 for record, wall in zip(bucket, walls) if wall > 0]
        rows.append({"fabric_class": cls, "direction": direction, "mode": mode,
                     "payload_bytes": size, "records": len(bucket),
                     "latency": _summary(walls),
                     "payload_gbps": statistics.fmean(payload_rates) if payload_rates else None,
                     "framed_gbps": statistics.fmean(framed_rates) if framed_rates else None,
                     "rate_status": "unknown: batch record lacks aggregate byte correlation" if mode == "io_uring_batch" else "measured"})
    correlated = all("request_id" in record for record in fabric) and any(record.get("request_id") is not None for record in records if record.get("record_type") == "stage_service")
    return {"groups": rows, "fabric_share_of_stage_wall": None,
            "fabric_share_status": "measurable correlation present" if correlated else "unknown: records have no shared request correlation key"}


def _primitive_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [record for record in records if record["schema"].startswith("fg.bc250.")]
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        by_benchmark[record.get("benchmark", record.get("component", "unknown"))].append(record)
    return {"record_count": len(rows), "benchmarks": {key: len(value) for key, value in sorted(by_benchmark.items())}}


def _expert_counterfactual(records: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    kernels = [record for record in records if record.get("record_type") == "kernel_service" and
               record["status"] == 0 and
               "expert" in str(record.get("scope", "")).lower()]
    if not kernels:
        return None
    grouped: dict[tuple[Any, ...], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for record in kernels:
        key=(record["execution_kind"],record["tokens"],record["first_token"],
             record["sequence"])
        grouped[key][record["rank"]] += float(record["gpu_ms"])
    result=[]
    for (kind,tokens,first_token,sequence),by_rank in sorted(grouped.items(),key=str):
        values=list(by_rank.values());maximum=max(values);mean=statistics.fmean(values)
        result.append({"scope":"expert kernel aggregates (not per-expert placement)",
            "execution_kind":kind,"tokens":tokens,"first_token":first_token,
            "sequence":sequence,"rank_gpu_ms":dict(sorted(by_rank.items())),
            "max_gpu_ms":maximum,"mean_gpu_ms":mean,
            "ideal_balanced_gpu_ms":sum(values)/len(values),
            "max_vs_mean_ratio":maximum/mean if mean else None,
            "counterfactual_recoverable_fraction":1.0-mean/maximum if maximum else None})
    return result


def analyze(records: list[dict[str, Any]], allow_partial: bool = False,
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    stage_records = [record for record in records if record.get("record_type") == "stage_service"]
    rank_ids = sorted({record["rank"] for record in stage_records} | {record["local_rank"] for record in records if record.get("record_type") == "fabric_service"})
    missing = [rank for rank in range(RANK_COUNT) if rank not in rank_ids]
    if missing and not allow_partial:
        raise ProfileError(f"fleet-incomplete: missing rank IDs {missing}; pass --allow-partial to inspect partial data")
    stage = _stage_analysis(records)
    expert_balance=_expert_counterfactual(records)
    straggler_ranks=sorted({rank for group in stage["groups"]
        if group["slowest_stage"] is not None for row in group["stages"]
        if row["stage"]==group["slowest_stage"] for rank in row["ranks"]})
    return {
        "schema": "fg.profile.analysis.v1", "partial": bool(missing),
        "warning": f"partial fleet: missing rank IDs {missing}" if missing else None,
        "rank_ids": rank_ids, "missing_rank_ids": missing, "metadata": metadata or {},
        "stage_service": stage, "fabric_service": _fabric_analysis(records),
        "primitive_measurements": _primitive_analysis(records),
        "counterfactual": {
            "label": "counterfactual",
            "stage_balance": "ideal_balanced_stage_time_ms and recoverable_imbalance_fraction are mathematical balance headroom only",
            "expert_service_balance": expert_balance,
            "expert_service_status": "measured kernel scope aggregate; not per-expert placement" if expert_balance else "unknown: no expert kernel scope records supplied",
            "idle_or_straggler_ranks": straggler_ranks,
        },
    }


def _table(result: dict[str, Any]) -> str:
    lines = ["FG profile analysis (measured facts; bounds are not achieved E2E TPS)",
             f"fleet ranks: {','.join(map(str, result['rank_ids'])) or 'none'}" + (" [PARTIAL]" if result["partial"] else "")]
    for group in result["stage_service"]["groups"]:
        bound = group.get("prefill_pipeline_steady_state_upper_bound_tps", group.get("decode_steady_state_max_stage_tps"))
        lines.append(f"{group['execution_kind']:7} first={group['first_token']:6} batch={group['tokens']:3} slowest=stage {group['slowest_stage']} "
                     f"service={group['slowest_stage_service_ms']:.3f} ms bound={bound:.3f} tok/s" if bound is not None else
                     f"{group['execution_kind']:7} batch={group['tokens']:3} no complete stage bound")
    for row in result["fabric_service"]["groups"]:
        payload_rate = "unknown" if row["payload_gbps"] is None else f"{row['payload_gbps']:.3f} GB/s"
        lines.append(f"fabric {row['fabric_class']}/{row['direction']}/{row['mode']} {row['payload_bytes']} B "
                     f"latency p50={row['latency']['p50_ms']:.3f} ms payload={payload_rate}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="JSONL measurement files")
    parser.add_argument("--allow-partial", action="store_true", help="permit reports missing one or more of the eight ranks")
    parser.add_argument("--metadata", help="optional JSON metadata object or file")
    args = parser.parse_args(argv)
    try:
        metadata: dict[str, Any] = {}
        if args.metadata:
            candidate = Path(args.metadata)
            raw = candidate.read_text(encoding="utf-8") if candidate.exists() else args.metadata
            metadata = json.loads(raw)
            if not isinstance(metadata, dict):
                raise ProfileError("metadata must be a JSON object")
        result = analyze(load_jsonl(args.files), args.allow_partial, metadata)
    except (OSError, ProfileError, json.JSONDecodeError) as exc:
        print(f"fg profile analysis error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    print(_table(result), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
