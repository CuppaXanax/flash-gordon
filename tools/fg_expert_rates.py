#!/usr/bin/env python3
"""Build the decode expert-chain achieved-GB/s table from FG_DECODE_PROFILE logs.

Reads DECODE_PROFILE_KERNEL lines (rank, scope, kernel, calls, gpu_ms) and
prices every expert dispatch with the byte model from
PERFORMANCE_BYTE_BUDGET_2026-09-15.md: the manifest-exact top-10 expert
weights of the rank's own six layers.  The fused decode pair reads exactly the
selected experts, so a call is priced as ten times the per-expert tensor bytes
of the layer; a scope call is priced as the rank's six-layer weight total
divided by six.

Usage:
    python3 tools/fg_expert_rates.py profile.log [more.log ...] [--json]

Only device-visible weight bytes are counted, matching the audit's
"experts (top-10)" row; the activation, mid and reduce traffic is small and
L2-resident and is deliberately not folded in.  Kernels the model does not
know are reported unpriced instead of guessed at.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any

# ggml type codes used by the deployed manifest.
Q4_K = 12
Q5_K = 13
Q5_1 = 7
Q8_0 = 8

EXPERT_GATE = 640
EXPERT_IN = 2560
EXPERT_DOWN_IN = 640

BLOCKS_256 = EXPERT_IN // 256
BLOCKS_32 = EXPERT_DOWN_IN // 32


def k_quant_expert(ggml_type: int) -> int:
    return EXPERT_GATE * BLOCKS_256 * (144 if ggml_type == Q4_K else 176)


def q5_1_expert() -> int:
    return EXPERT_IN * BLOCKS_32 * 24


def q8_0_cooked_expert() -> int:
    return EXPERT_IN * BLOCKS_32 * 34


# layer -> (gate, up, down) ggml types, from the deployed manifest.
LAYER_TYPES: dict[int, tuple[int, int, int]] = {}
for _layer in range(48):
    LAYER_TYPES[_layer] = (Q4_K, Q4_K, Q5_1)
LAYER_TYPES[2] = (Q5_K, Q5_K, Q8_0)
for _layer in (4, 30, 46, 47):
    LAYER_TYPES[_layer] = (Q4_K, Q4_K, Q8_0)


def expert_layer_bytes(layer: int, projection: str) -> int:
    gate_type, up_type, down_type = LAYER_TYPES[layer]
    if projection == "gate":
        return 10 * k_quant_expert(gate_type)
    if projection == "up":
        return 10 * k_quant_expert(up_type)
    if projection == "gate_up":
        return expert_layer_bytes(layer, "gate") + expert_layer_bytes(layer, "up")
    if projection == "down":
        bytes_per_expert = q5_1_expert() if down_type == Q5_1 else q8_0_cooked_expert()
        return 10 * bytes_per_expert
    raise KeyError(projection)


# rank -> layers owned, from the audit's layer_owner table (rank 1 owns 0-5,
# rank 0 owns 6-11, every other rank owns the next six).
RANK_LAYERS: dict[int, list[int]] = {0: list(range(6, 12)), 1: list(range(0, 6))}
for _rank in range(2, 8):
    RANK_LAYERS[_rank] = list(range(6 * _rank, 6 * _rank + 6))

# rank -> per-call weight bytes for the fused pair and the legacy reduce steps.
RANK_GATE_UP = {rank: sum(expert_layer_bytes(layer, "gate_up") for layer in layers)
                for rank, layers in RANK_LAYERS.items()}
RANK_DOWN = {rank: sum(expert_layer_bytes(layer, "down") for layer in layers)
             for rank, layers in RANK_LAYERS.items()}
RANK_GATE = {rank: sum(expert_layer_bytes(layer, "gate") for layer in layers)
             for rank, layers in RANK_LAYERS.items()}
RANK_UP = {rank: sum(expert_layer_bytes(layer, "up") for layer in layers)
           for rank, layers in RANK_LAYERS.items()}

MID_BYTES = 10 * EXPERT_GATE * 4
REDUCED_BYTES = EXPERT_IN * 4

# scope -> kernel basename -> (projection key, per-call byte model)
MODEL: dict[str, dict[str, tuple[str, str]]] = {
    "expert_decode": {
        "fg_moe_decode_gate_up.spv": ("gate_up", "layer_mean"),
        "fg_moe_decode_down_reduce.spv": ("down", "layer_mean"),
    },
    "expert_gate_up": {"fg_moe_decode_gate_up.spv": ("gate_up", "layer_mean")},
    "expert_down_reduce": {"fg_moe_decode_down_reduce.spv": ("down", "layer_mean")},
    "expert_activation": {"fg_swiglu.spv": ("gate_up", "swiglu")},
    "expert_gate": {"fg_moe_kquant.spv": ("gate_up", "half_gate")},
    "expert_up": {"fg_moe_kquant.spv": ("gate_up", "half_up")},
    "expert_reduce": {"fg_moe_reduce.spv": ("down", "reduce")},
}
# The unchained decode path names its scopes per projection; the chained path
# leaves the profile scope at its default.
MODEL["unscoped"] = dict(MODEL["expert_decode"])


def price(rank: int, scope: str, kernel: str, calls: int) -> tuple[float, bool]:
    entry = MODEL.get(scope, {}).get(kernel)
    if entry is None:
        return 0.0, False
    projection, shape = entry
    if shape == "swiglu":
        return float(3 * MID_BYTES) * calls, True
    if shape == "half_gate":
        return float(RANK_GATE[rank]) / 6.0 * calls, True
    if shape == "half_up":
        return float(RANK_UP[rank]) / 6.0 * calls, True
    if shape == "reduce":
        return float(11 * REDUCED_BYTES + 10 * 4) * calls, True
    per_call = RANK_GATE_UP[rank] if projection == "gate_up" else RANK_DOWN[rank]
    return float(per_call) / 6.0 * calls, True


LINE = re.compile(
    r"DECODE_PROFILE_KERNEL rank=(\d+) scope=(\S+) kernel=(\S+) calls=(\d+) gpu_ms=([0-9.]+)"
)


class Row:
    __slots__ = ("rank", "scope", "kernel", "bytes", "calls", "ms", "priced")

    def __init__(self, rank: int, scope: str, kernel: str) -> None:
        self.rank = rank
        self.scope = scope
        self.kernel = kernel
        self.bytes = 0.0
        self.calls = 0
        self.ms = 0.0
        self.priced = True


def rate_gbps(bytes_count: float, ms: float) -> float | None:
    if ms <= 0.0:
        return None
    return bytes_count / ms / 1e6


def load(paths: list[str]) -> dict[tuple[int, str, str], Row]:
    rows: dict[tuple[int, str, str], Row] = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = LINE.search(line)
                if not match:
                    continue
                rank, scope, kernel, calls, ms = match.groups()
                key = (int(rank), scope, kernel)
                row = rows.get(key)
                if row is None:
                    row = rows[key] = Row(*key)
                price_bytes, priced = price(*key, int(calls))
                row.bytes += price_bytes
                row.calls += int(calls)
                row.ms += float(ms)
                row.priced = row.priced and priced
    return rows


def build(paths: list[str]) -> dict[str, Any]:
    rows = load(paths)
    kernels = []
    scopes: dict[tuple[int, str], dict[str, Any]] = {}
    for row in sorted(rows.values(), key=lambda item: (item.rank, item.scope, item.kernel)):
        kernels.append({
            "rank": row.rank,
            "scope": row.scope,
            "kernel": row.kernel,
            "calls": row.calls,
            "gpu_ms": round(row.ms, 4),
            "bytes": int(row.bytes),
            "priced": row.priced,
            "achieved_gbps": None if not row.priced or row.ms <= 0.0
            else round(rate_gbps(row.bytes, row.ms) or 0.0, 1),
        })
        key = (row.rank, row.scope)
        scope = scopes.setdefault(key, {"rank": row.rank, "scope": row.scope, "calls": 0,
                                        "gpu_ms": 0.0, "bytes": 0, "priced": True})
        scope["calls"] += row.calls
        scope["gpu_ms"] += row.ms
        scope["bytes"] += row.bytes
        scope["priced"] = scope["priced"] and row.priced
    for scope in scopes.values():
        scope["gpu_ms"] = round(scope["gpu_ms"], 4)
        scope["achieved_gbps"] = (None if not scope["priced"]
                                  else round(rate_gbps(scope["bytes"], scope["gpu_ms"]) or 0.0, 1))
    return {"kernels": kernels,
            "scopes": sorted(scopes.values(), key=lambda item: (item["rank"], item["scope"]))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="logs with DECODE_PROFILE_KERNEL lines")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    result = build(args.logs)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"{'rank':>4} {'scope':<20} {'kernel':<34} {'calls':>5} {'ms':>8} "
          f"{'MB':>8} {'GB/s':>7}")
    for entry in result["kernels"]:
        gbps = "-" if entry["achieved_gbps"] is None else f"{entry['achieved_gbps']:.1f}"
        print(f"{entry['rank']:>4} {entry['scope']:<20} {entry['kernel']:<34} "
              f"{entry['calls']:>5} {entry['gpu_ms']:>8.3f} {entry['bytes'] / 1e6:>8.2f} "
              f"{gbps:>7}")
    print()
    print(f"{'rank':>4} {'scope':<20} {'calls':>5} {'ms':>8} {'MB':>8} {'GB/s':>7}")
    for entry in result["scopes"]:
        gbps = "-" if entry["achieved_gbps"] is None else f"{entry['achieved_gbps']:.1f}"
        print(f"{entry['rank']:>4} {entry['scope']:<20} {entry['calls']:>5} "
              f"{entry['gpu_ms']:>8.3f} {entry['bytes'] / 1e6:>8.2f} {gbps:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
