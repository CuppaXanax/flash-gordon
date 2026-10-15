#!/usr/bin/env python3
"""Summarize a matched final decode window from rank stderr captures."""
import argparse
import json
import re
import statistics
from pathlib import Path


def fields(line):
    result = dict(re.findall(r"(\w+)=([^\s]+)", line))
    for key, value in result.items():
        try:
            result[key] = float(value)
        except ValueError:
            pass
    return result


def mean(rows):
    if not rows:
        return {}
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key, value in rows[0].items()
        if isinstance(value, (float, int))
        and all(isinstance(row.get(key), (float, int)) for row in rows)
    }


def summarize(directory, suffix, count, legacy_embed=False):
    result = {"window": count, "suffix": suffix, "ranks": []}
    window_tokens = None
    for rank in range(8):
        lines = (directory / f"rank-{rank}-{suffix}.log").read_text().splitlines()
        block_prefix = "RING_DECODE token=" if rank == 0 else "RING_DECODE_BLOCK "
        blocks = [fields(line) for line in lines if line.startswith(block_prefix)][-count:]
        tokens = [row["token"] for row in blocks]
        if len(tokens) != count or tokens != list(range(int(tokens[0]), int(tokens[0]) + count)):
            raise ValueError(f"rank {rank}: expected {count} consecutive decode tokens")
        if window_tokens is not None and tokens != window_tokens:
            raise ValueError(f"rank {rank}: decode window differs from coordinator")
        window_tokens = tokens
        profiles = [fields(line) for line in lines if line.startswith("DECODE_PROFILE rank=")][-count:]
        receives, sends, outputs = [], [], []
        for line in lines:
            if line.startswith('{"schema":"flash-gordon.profile"'):
                row = json.loads(line)
                if row.get("record_type") == "fabric_service" and row["message_type"] == 44:
                    (receives if row["direction"] == "receive" else sends).append(row)
            elif line.startswith("OUTPUT_SERVICE "):
                outputs.append(fields(line))
        kernels, current = [], None
        for line in lines:
            if line.startswith("DECODE_PROFILE rank="):
                current = {}
                kernels.append(current)
            elif line.startswith("DECODE_PROFILE_KERNEL ") and current is not None:
                row = fields(line)
                scope = row["scope"]
                current[scope] = current.get(scope, 0.0) + row["gpu_ms"]
        result["ranks"].append({
            "rank": rank, "samples": len(blocks), "block": mean(blocks),
            "profile": mean(profiles), "kernel_scopes_ms": mean(kernels[-count:]),
            "work_receive": mean(receives[-count:]), "work_send": mean(sends[-count:]),
            "work_receive_wait_p50_ms": statistics.median(
                row["wait_ms"] for row in receives[-count:]) if receives else None,
            "output": mean(outputs[-count:]),
        })
    result["token_start"], result["token_end"] = window_tokens[0], window_tokens[-1]
    ranks = result["ranks"]
    coordinator = ranks[0]["block"]
    services = coordinator["own_ms"] + sum(row["block"]["total_ms"] for row in ranks[1:])
    # Original trace's embed_ms includes n-gram and send preparation.
    prepare = coordinator["embed_ms"]
    if not legacy_embed:
        prepare += coordinator["ngram_ms"] + coordinator["send_ms"]
    result["serial_ledger_ms"] = {
        "coordinator_prepare": prepare,
        "rank_services_including_enqueue_hc": services,
        "transport_head_completion_and_unattributed": coordinator["total_ms"] - prepare - services,
        "observed": coordinator["total_ms"],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--legacy-embed", action="store_true",
                        help="Old trace's embed_ms includes all coordinator preparation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    summary = summarize(args.directory, args.suffix, args.count, args.legacy_embed)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["serial_ledger_ms"], indent=2))
