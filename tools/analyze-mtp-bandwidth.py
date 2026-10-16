#!/usr/bin/env python3
"""Summarize paired stock-clock small-N research receipts."""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import statistics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith("{")]


def summary(values):
    mean = statistics.mean(values)
    ci = 2.776445105 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) == 5 else None
    return {"mean": mean, "ci95_half": ci, "rounds": values}


def median(path):
    rows = records(path)
    samples = [r["wall_us"] for r in rows if r["kind"] == "sample"]
    configs = [r for r in rows if r["kind"] == "config"]
    if configs:
        assert len(samples) == configs[0]["iterations"] * configs[0]["windows"], path
    assert all(math.isfinite(v) and v > 0 for v in samples), path
    return statistics.median(samples)


def family(name):
    if "gate_up" in name:
        return "expert_gate_up"
    if "small_down" in name or "moe_decode_down" in name:
        return "expert_down"
    if any(s in name for s in ("dense", "reuse", "small_q4", "small_q5",
                               "small_iq4", "small_q8", "small_f32",
                               "small_index_bf16", "small_router_bf16")):
        return "dense"
    if "qsa" in name or "topk_reduce" in name:
        return "qsa"
    if "gdn_small" in name or "gdn_recurrent" in name:
        return "recurrence"
    return "other"


def analyze(directory):
    result = {"paired": {}, "profiles": {}}
    for kind in ("short", "long", "layout", "mixed", "combined-long"):
        root = directory / ("paired-" + kind)
        groups = collections.defaultdict(list)
        for path in sorted(root.glob("r*-n*-b*-f*.jsonl")):
            match = re.fullmatch(r"r(\d+)-n(\d+)-b(\d+)-f(\d+)\.jsonl", path.name)
            if match is None:
                continue
            round_, n, index, flags = map(int, match.groups())
            before = root / f"r{round_}-n{n}-a{index}-f330.jsonl"
            after = root / f"r{round_}-n{n}-a{index+1}-f330.jsonl"
            ref = (median(before) + median(after)) / 2
            base = (median(root / f"r{round_}-n1-before-f0.jsonl") +
                    median(root / f"r{round_}-n1-after-f0.jsonl")) / 2
            candidate = median(path)
            groups[(n, flags)].append({
                "round": round_, "baseline_us": ref, "candidate_us": candidate,
                "n1_us": base, "baseline_ratio": ref / base,
                "candidate_ratio": candidate / base,
                "reduction_percent": 100 * (1 - candidate / ref)})
        for rows in groups.values():
            assert len(rows) == 5 and {r["round"] for r in rows} == set(range(1, 6))
        result["paired"][kind] = [
            {"n": n, "flags": flags, "samples": rows,
             **{key: summary([r[key] for r in rows])
                for key in ("baseline_us", "candidate_us", "n1_us",
                            "baseline_ratio", "candidate_ratio", "reduction_percent")}}
            for (n, flags), rows in sorted(groups.items())]
        profile_groups = collections.defaultdict(list)
        for path in sorted(root.glob("profile-r*-n*-f*.jsonl")):
            match = re.fullmatch(r"profile-r(\d+)-n(\d+)-f(\d+)\.jsonl", path.name)
            round_, n, flags = map(int, match.groups())
            times = collections.defaultdict(float)
            kernels = {}
            counts = {}
            q8_us = 0
            for row in records(path):
                if row["kind"] == "kernel":
                    times[family(row["name"])] += row["gpu_us"]
                    kernels[row["scope"] + "/" + row["name"]] = row["gpu_us"]
                    counts[row["scope"] + "/" + row["name"]] = row.get("count")
                    if "down_q8" in row["name"]:
                        q8_us += row["gpu_us"]
            routes = root / f"routes-r{round_}-n{n}-f{flags}.jsonl"
            route_rows = records(routes)
            ids = collections.defaultdict(set)
            for row in route_rows:
                ids[row["layer"]].update(row["experts"])
            union = sum(len(value) for value in ids.values())
            ledger = root / "ledger.jsonl"
            if ledger.exists():
                weights = records(ledger)
                weight_bytes = {"dense": sum(r["bytes_per_copy"] for r in weights
                                             if r.get("copies") == 1)}
                for name, substring in (("expert_gate_up", ("gate_exps", "up_exps")),
                                         ("expert_down", ("down_exps",))):
                    weight_bytes[name] = sum(
                        len(ids[r["layer"]]) * r["bytes_per_copy"] for r in weights
                        if r.get("copies") == 512 and any(s in r["name"] for s in substring))
            else:
                weight_bytes = {"dense": 302110208,
                                "expert_gate_up": union * 2 * 921600,
                                "expert_down": union * 1228800}
            q8 = {}
            if ledger.exists() and q8_us:
                q8_layers = {r["layer"]: r["bytes_per_copy"] for r in weights
                             if r.get("type") == 8 and r.get("copies") == 512
                             and "down_exps" in r["name"]}
                unique_bytes = sum(len(ids[layer]) * size for layer, size in q8_layers.items())
                slot_bytes = sum(len(r["experts"]) * q8_layers[r["layer"]]
                                 for r in route_rows if r["layer"] in q8_layers)
                q8 = {"gpu_us": q8_us, "unique_bytes": unique_bytes,
                      "slot_bytes_without_cross_query_reuse": slot_bytes,
                      "unique_weight_GBs": unique_bytes / q8_us / 1000}
            profile_groups[(n, flags)].append({
                "round": round_, "gpu_us": dict(times), "kernels": kernels,
                "kernel_dispatches": counts, "q8_down": q8,
                "expert_union_per_layer": {str(k): len(v) for k, v in ids.items()},
                "compulsory_weight_bytes": weight_bytes,
                "effective_weight_GBs": {k: v / times[k] / 1000
                                          for k, v in weight_bytes.items() if times[k]}})
        for rows in profile_groups.values():
            assert len(rows) == 5 and {r["round"] for r in rows} == set(range(1, 6))
        result["profiles"][kind] = [
            {"n": n, "flags": flags, "samples": rows,
             "gpu_us": {k: summary([r["gpu_us"].get(k, 0) for r in rows])
                        for k in sorted({k for r in rows for k in r["gpu_us"]})},
             "effective_weight_GBs": {
                 k: summary([r["effective_weight_GBs"][k] for r in rows])
                 for k in rows[0]["effective_weight_GBs"]}}
            for (n, flags), rows in sorted(profile_groups.items())]
    shape_groups = collections.defaultdict(list)
    root = directory / "paired-layout"
    for path in sorted(root.glob("shape-r*-k*-m*-t*-b-f*.jsonl")):
        match = re.fullmatch(r"shape-r(\d+)-k(\d+)-m(\d+)-t(\d+)-b-f(\d+)\.jsonl", path.name)
        round_, k, m, type_, flags = map(int, match.groups())
        before = root / f"shape-r{round_}-k{k}-m{m}-t{type_}-a0-f330.jsonl"
        after = root / f"shape-r{round_}-k{k}-m{m}-t{type_}-a1-f330.jsonl"
        costs = [{(r["compact"], r["n"]): r["us"] for r in records(p)
                  if r.get("kind") == "dense_cost"} for p in (before, path, after)]
        for (compact, n), value in costs[1].items():
            padded = (m + 7) & ~7
            weight_bytes = padded * (k // 32) * (18 if type_ == 20 else 24 if type_ == 12 else 28)
            if compact and type_ != 20:
                weight_bytes = padded * (k // 256) * (148 if type_ == 12 else 180)
            base = (costs[0][compact, n] + costs[2][compact, n]) / 2
            shape_groups[k, m, type_, compact, n, flags].append({
                "round": round_, "baseline_us": base, "candidate_us": value,
                "weight_bytes": weight_bytes, "baseline_GBs": weight_bytes / base / 1000,
                "candidate_GBs": weight_bytes / value / 1000,
                "reduction_percent": 100 * (1 - value / base)})
    result["shapes"] = []
    for (k, m, type_, compact, n, flags), rows in sorted(shape_groups.items()):
        assert len(rows) == 5 and {r["round"] for r in rows} == set(range(1, 6))
        result["shapes"].append({
            "k": k, "m": m, "type": type_, "compact": compact, "n": n, "flags": flags,
            "scope": "Repeated static GEMV: cacheable weights, not measured bus traffic",
            "samples": rows, **{key: summary([r[key] for r in rows])
                               for key in ("baseline_us", "candidate_us", "baseline_GBs",
                                           "candidate_GBs", "reduction_percent")}})
    result["pilots"] = {}
    result["oracles"] = {}
    for root in sorted(directory.glob("round*")):
        for path in sorted(root.glob("*.jsonl")):
            rows = records(path)
            kernels = [r for r in rows if r.get("kind") == "kernel"]
            if kernels:
                times = collections.defaultdict(float)
                for r in kernels:
                    times[family(r["name"])] += r["gpu_us"]
                result["pilots"][path.relative_to(directory).as_posix()] = {
                    "scope": "single intrusive/unpaired diagnostic, not wall qualification",
                    "config": next((r for r in rows if r["kind"] == "config"), None),
                    "median_wall_us": median(path), "family_gpu_us": dict(times)}
            dense = [r for r in rows if r.get("kind") == "dense_small_test"]
            gates = [r for r in rows if r.get("kind", "").startswith("small_") or r.get("kind") == "rollback"]
            if dense or gates:
                result["oracles"][path.relative_to(directory).as_posix()] = {
                    "dense_cases": len(dense),
                    "dense_max_abs": max((r["max_abs"] for r in dense), default=None),
                    "dense_tolerance_ok": all(r["tolerance_ok"] for r in dense),
                    "gates": gates}
        for path in sorted(root.glob("*.txt")):
            if any(s in path.name for s in ("cpu", "residency", "exact-q8")):
                text = path.read_text()
                errors = [float(v) for v in re.findall(r"max_rel=([0-9.eE+-]+)", text)]
                result["oracles"][path.relative_to(directory).as_posix()] = {
                    "passed": text.rstrip().endswith("PASS"),
                    "max_rel": max(errors, default=None),
                    "last_lines": text.splitlines()[-4:]}
    result["build_receipts"] = {
        path.relative_to(directory).as_posix(): path.read_text()
        for path in directory.glob("**/*-sha256.txt")}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "Opt-in synthetic six-layer research; no real-model or serving promotion",
        "method": "Five seed-matched alternating candidate orders; each B bracketed by A=330; N1 bracket per round",
        "operating_point": "Stock dynamic governor, unchanged clocks/CU configuration",
        "roof_GBs": 450,
        "byte_metric": "Compulsory selected cooked-weight bytes, not measured bus traffic; activation/state traffic excluded",
        "blades": {name: analyze(args.input / name) for name in ("50", "53")},
    }
    report["archives"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.input.glob("*.tar.gz"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
