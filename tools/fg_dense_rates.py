#!/usr/bin/env python3
"""Build the dense-chain achieved-GB/s table from FG_DECODE_PROFILE logs.

Reads DECODE_PROFILE_KERNEL lines (rank, scope, kernel, calls, gpu_ms) and
prices every dispatch with the byte model below, which follows
PERFORMANCE_BYTE_BUDGET_2026-09-15.md (manifest-exact tensor bytes) plus the
dense kernel tile sizes from src/quant.c.  The output is one row per
(rank, scope, kernel) with bytes, milliseconds and achieved GB/s, plus a
per-scope and fleet total.

Usage:
    python3 tools/fg_dense_rates.py profile.log [more.log ...] [--json]

The byte model is keyed by (scope, kernel basename); when a scope dispatches
two shapes through the same kernel name (GDN qkv and z are both
fg_dense_q8_0_cooked_r8.spv) the model carries a per-call mixture and reports
the blended rate.  L2 re-read traffic (the r8 input row re-read per workgroup)
is reported separately as its own column; only device-visible buffer traffic
is counted in `bytes`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any

# cooked Q8_0 tile bytes: align64(16*blocks*2) + 16*blocks*32 per 16-row tile.
def cooked_tile(blocks: int) -> int:
    quant = (16 * blocks * 2 + 63) & ~63
    return quant + 16 * blocks * 32


def cooked_matrix(input_width: int, rows: int) -> int:
    blocks = input_width // 32
    return ((rows + 15) // 16) * cooked_tile(blocks)


QKV = cooked_matrix(2560, 10240)
Z = cooked_matrix(2560, 6144)
OUT = cooked_matrix(6144, 2560)
UP = cooked_matrix(320, 10240)
DOWN = cooked_matrix(10240, 320)
SGATE = cooked_matrix(2560, 640)
SDOWN = cooked_matrix(640, 2560)
PKEY = cooked_matrix(2560, 10240)
PVAL = cooked_matrix(2560, 2560)
HEAD = 675430400

HIDDEN_B = 2560 * 4
HYPER_B = 10240 * 4

CONV_STATE = 10240 * 4 * 4
RECURRENT_STATE = 48 * 128 * 128 * 4

# scope -> kernel basename -> list of (call share, bytes per call, extra L2 bytes per call)
MODEL: dict[str, dict[str, list[tuple[float, int, int]]]] = {
    "gdn_projection": {
        "fg_dense_q8_0_cooked_r8.spv": [
            (0.5, QKV + HIDDEN_B + 40960, 0),   # qkv 27.85 MB of cooked weights
            (0.5, Z + HIDDEN_B + 24576, 0),     # z 16.71 MB of cooked weights
        ],
        "fg_dense_f32.spv": [(1.0, 48 * 2560 * 4 + HIDDEN_B + 192, 0)],
    },
    "gdn_recurrent": {
        "fg_gdn_conv_decode.spv": [(1.0, CONV_STATE * 2 + 163840 + 40960 + 40960, 0)],
        "fg_gdn_recurrent_algebraic.spv": [(1.0, RECURRENT_STATE * 3 + 40960 + 24576 + 24576, 0)],
    },
    "gdn_output": {"fg_dense_q8_0_cooked_r8.spv": [(1.0, OUT + 6144 * 4 + HIDDEN_B, 0)]},
    "gr_attn_read": {
        "fg_group_rms_norm.spv": [(1.0, HYPER_B * 3, 0)],
        "fg_dense_q8_0_cooked_split.spv": [(1.0, DOWN + HYPER_B + 320 * 8 * 4, 0)],
        "fg_dense_q8_0_cooked_split_reduce.spv": [(1.0, 320 * 8 * 4 + 320 * 4 * 2, 0)],
        "fg_dense_q8_0_cooked_split_reduce_silu.spv": [(1.0, 320 * 8 * 4 + 320 * 4 * 2, 0)],
        "fg_hc_inject_partial.spv": [(1.0, HYPER_B + 4 * HYPER_B + 24 * 4 * 4, 0)],
        "fg_silu_scaled.spv": [(1.0, 320 * 4 * 2, 0)],
        "fg_dense_q8_0_cooked_r8.spv": [(1.0, UP + 320 * 4 + HIDDEN_B, 1280 * 320 * 4)],
        "fg_gr_mix_partial.spv": [(1.0, HYPER_B * 2 + 24 * 4 * 4 + HIDDEN_B + 16, 0)],
    },
    "gr_ffn_read": {},  # mirrored from gr_attn_read at load time
    "ple": {
        "fg_dense_q8_0_cooked_r8.spv": [
            (0.5, PKEY + HYPER_B + 40960, 0),
            (0.5, PVAL + HYPER_B + 10240, 0),
        ],
        "fg_group_rms_norm.spv": [(1.0, HYPER_B * 3, 0)],
        "fg_ple_gate.spv": [(1.0, HYPER_B * 3 + HIDDEN_B, 0)],
        "fg_ple_conv_decode_add.spv": [(1.0, 40960 * 3 + 163840 + 368640 * 2 + 40960, 0)],
        "fg_ple_conv_decode.spv": [(1.0, 40960 * 2 + 163840 + 368640 * 2, 0)],
        "fg_add_f32.spv": [(1.0, HYPER_B * 3, 0)],
    },
    "shared_expert": {
        "fg_dense_q8_0_cooked_r8.spv": [
            (2.0 / 3.0, SGATE + HIDDEN_B + 2560, 0),
            (1.0 / 3.0, SDOWN + 2560 + HIDDEN_B, 0),
        ],
        "fg_dense_f32.spv": [(1.0, HIDDEN_B * 2 + 4, 0)],
        "fg_swiglu.spv": [(1.0, 2560 * 4 * 3, 0)],
    },
    "router": {"fg_dense_f32.spv": [(1.0, 512 * 2560 * 4 + HIDDEN_B + 2048, 0)]},
    "router_quantization": {"fg_quantize_q8_k.spv": [(1.0, HIDDEN_B + 10 * 292, 0)]},
    "output": {"fg_dense_q8_0_cooked_r8.spv": [(1.0, HEAD)]},
}
# QSA projections and the record machinery are deliberately unpriced here: the
# audit's QSA numbers are a model, not manifest-exact, and their rates land
# above the ceiling when priced with the cooked tile sizes.
MODEL["gr_ffn_read"] = MODEL["gr_attn_read"]

LINE = re.compile(
    r"DECODE_PROFILE_KERNEL rank=(\d+) scope=(\S+) kernel=(\S+) calls=(\d+) gpu_ms=([0-9.]+)"
)


class Row:
    __slots__ = ("bytes", "l2", "calls", "ms", "priced")

    def __init__(self) -> None:
        self.bytes = 0.0
        self.l2 = 0.0
        self.calls = 0
        self.ms = 0.0
        self.priced = True


def price(scope: str, kernel: str, calls: int) -> tuple[float, float, bool]:
    entry = MODEL.get(scope, {}).get(kernel)
    if entry is None:
        return 0.0, 0.0, False
    total_share = sum(share for share, _, _ in entry)
    bytes_per_call = sum(share * b for share, b, _ in entry) / total_share
    l2_per_call = sum(share * l for share, _, l in entry) / total_share
    return bytes_per_call * calls, l2_per_call * calls, True


def rate_gbps(bytes_count: float, ms: float) -> float | None:
    if ms <= 0.0:
        return None
    return bytes_count / ms / 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="logs with DECODE_PROFILE_KERNEL lines")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    rows: dict[tuple[int, str, str], Row] = defaultdict(Row)
    for path in args.logs:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = LINE.search(line)
                if not match:
                    continue
                rank, scope, kernel, calls, ms = match.groups()
                key = (int(rank), scope, kernel)
                price_bytes, l2_bytes, priced = price(scope, kernel, int(calls))
                row = rows[key]
                row.bytes += price_bytes
                row.l2 += l2_bytes
                row.calls += int(calls)
                row.ms += float(ms)
                row.priced = row.priced and priced

    entries = []
    for (rank, scope, kernel), row in sorted(rows.items()):
        entries.append({
            "rank": rank,
            "scope": scope,
            "kernel": kernel,
            "calls": row.calls,
            "gpu_ms": round(row.ms, 4),
            "bytes": int(row.bytes),
            "priced": row.priced,
            "achieved_gbps": None if not row.priced else round(rate_gbps(row.bytes, row.ms) or 0.0, 1),
            "l2_bytes": int(row.l2),
        })
    scopes: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in entries:
        key = (entry["rank"], entry["scope"])
        scope = scopes.setdefault(key, {"rank": entry["rank"], "scope": entry["scope"],
                                        "calls": 0, "gpu_ms": 0.0, "bytes": 0,
                                        "priced": True, "l2_bytes": 0})
        scope["calls"] += entry["calls"]
        scope["gpu_ms"] += entry["gpu_ms"]
        scope["bytes"] += entry["bytes"]
        scope["l2_bytes"] += entry["l2_bytes"]
        scope["priced"] = scope["priced"] and entry["priced"]
    for scope in scopes.values():
        scope["gpu_ms"] = round(scope["gpu_ms"], 4)
        scope["achieved_gbps"] = (None if not scope["priced"]
                                  else round(rate_gbps(scope["bytes"], scope["gpu_ms"]) or 0.0, 1))
    result = {"kernels": entries,
              "scopes": sorted(scopes.values(), key=lambda item: (item["rank"], item["scope"]))}

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"{'rank':>4} {'scope':<22} {'kernel':<45} {'calls':>5} {'ms':>9} "
          f"{'MB':>9} {'GB/s':>7} {'L2 MB':>7}")
    for entry in entries:
        gbps = "-" if entry["achieved_gbps"] is None else f"{entry['achieved_gbps']:.1f}"
        print(f"{entry['rank']:>4} {entry['scope']:<22} {entry['kernel']:<45} "
              f"{entry['calls']:>5} {entry['gpu_ms']:>9.3f} {entry['bytes'] / 1e6:>9.2f} "
              f"{gbps:>7} {entry['l2_bytes'] / 1e6:>7.2f}")
    print()
    print(f"{'rank':>4} {'scope':<22} {'calls':>5} {'ms':>9} {'MB':>9} {'GB/s':>7}")
    for scope in result["scopes"]:
        gbps = "-" if scope["achieved_gbps"] is None else f"{scope['achieved_gbps']:.1f}"
        print(f"{scope['rank']:>4} {scope['scope']:<22} {scope['calls']:>5} "
              f"{scope['gpu_ms']:>9.3f} {scope['bytes'] / 1e6:>9.2f} {gbps:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
