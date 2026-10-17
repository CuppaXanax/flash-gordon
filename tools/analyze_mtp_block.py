#!/usr/bin/env python3
"""Attribute intrusive N1/N4 profiles; do not interpret them as replay timings."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def category(scope, name):
    if scope.startswith("gr_"):
        return "HC/projections/small ops"
    if "gate_up" in name or name.startswith("fg_roof_expert"):
        return "Expert gate/up"
    if name.startswith(("fg_small_down", "fg_roof_down", "fg_moe_decode_down",
                        "fg_moe_reduce")):
        return "Expert down/reduce"
    if ("router" in name or "union" in name or "schedule" in name
            or scope.startswith("router") or scope.startswith("moe_reduce")):
        return "Expert routing/union"
    if name.startswith(("fg_gdn_", "fg_roof_recurrent")):
        return "GDN convolution/recurrence/normalization"
    if ("qsa_" in name or "index_" in name
            or ("qsa" in scope and name == "fg_dense_bf16_f32.spv")
            or ("qsa" in scope and name == "fg_quantize_q8_0.spv")):
        return "QSA attention/cache/index"
    if name.startswith(("fg_small_q", "fg_small_iq", "fg_small_reuse",
                        "fg_dense_", "fg_small_f32", "fg_roof_dense",
                        "fg_roof_gdn_gate")) or scope.startswith("shared_expert"):
        return "Dense projections/shared"
    return "Other"


def read_profile(path, small=False):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.startswith("{")]
    kernels = [r for r in rows if r.get("kind") == "kernel"]
    totals = [r for r in rows if r.get("kind") == "profile"]
    if not kernels or len(totals) != 1:
        raise ValueError(f"{path}: expected one complete intrusive profile")
    if small and not {"gr_attn_read_small", "gr_ffn_write_small"}.issubset(
            {r["scope"] for r in kernels}):
        raise ValueError(f"{path}: stale small-N scopes; reprofile with scope instrumentation")
    grouped = defaultdict(float)
    for row in kernels:
        us = float(row["gpu_us"])
        if not math.isfinite(us) or us < 0 or row["count"] < 1:
            raise ValueError(f"{path}: invalid kernel timing")
        grouped[category(row["scope"], row["name"])] += us
    gaps = float(totals[0].get("gaps_us", 0))
    if not math.isfinite(gaps) or gaps < 0:
        raise ValueError(f"{path}: invalid inter-dispatch gaps")
    if gaps:
        grouped["Inter-dispatch gaps"] = gaps
    if not math.isclose(sum(grouped.values()), totals[0]["gpu_us"], abs_tol=0.05):
        raise ValueError(f"{path}: kernel timings do not close against GPU total")
    return grouped, totals[0]


def compare(n1, n4):
    return sorted(
        ({"operation": op, "n1_us": n1.get(op, 0), "n4_us": n4.get(op, 0),
          "ratio": n4.get(op, 0) / n1[op] if n1.get(op, 0) else None,
          "excess_vs_1_5_us": n4.get(op, 0) - 1.5 * n1.get(op, 0)}
         for op in n1.keys() | n4.keys()),
        key=lambda row: -row["excess_vs_1_5_us"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n1", type=Path)
    parser.add_argument("n4", type=Path)
    args = parser.parse_args()
    n1, total1 = read_profile(args.n1)
    n4, total4 = read_profile(args.n4, small=True)
    print(json.dumps({"intrusive_attribution_only": True,
                      "operations": compare(n1, n4),
                      "n1_profile": total1, "n4_profile": total4}, indent=2))


if __name__ == "__main__":
    main()
