#!/usr/bin/env python3
"""Summarize stock small-N paired A/B/A runs without pooling correlated samples."""
import argparse
import json
import math
from pathlib import Path
import statistics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith("{")]


def estimate(values):
    return {"mean": statistics.mean(values),
            "ci95": 2.776445105 * statistics.stdev(values) / math.sqrt(len(values))}


def run(path, n, flags, seed):
    data = records(path)
    config = next(x for x in data if x["kind"] == "config")
    assert config["batch"] == n and config["small_kernels"] == flags
    assert config["seed"] == seed and config["warmup"] * n == 264
    assert not config["clocks"] and not config["intrusive"]
    values = [x["wall_us"] for x in data if x["kind"] == "sample"]
    assert len(values) == 216 // n and all(math.isfinite(x) and x > 0 for x in values)
    return {"median_us": statistics.median(values),
            "mean_us": statistics.mean(values), "sample_count": len(values)}


def analyze(directory):
    directory = Path(directory) / "qualify"
    results = {"n1_rounds": [], "candidates": {}}
    for seed in range(1, 6):
        before = run(directory / f"r{seed}-n1-before-f0.jsonl", 1, 0, seed)
        after = run(directory / f"r{seed}-n1-after-f0.jsonl", 1, 0, seed)
        results["n1_rounds"].append({"seed": seed, "before": before, "after": after,
                                    "reference_us": (before["median_us"] + after["median_us"]) / 2})
    results["n1_us"] = estimate([x["reference_us"] for x in results["n1_rounds"]])
    for flags in (74, 330):
        candidate = {}
        for n in (2, 3, 4):
            rounds = []
            for seed in range(1, 6):
                order = (74, 330) if seed % 2 else (330, 74)
                index = order.index(flags)
                a = run(directory / f"r{seed}-n{n}-a{index}-f0.jsonl", n, 0, seed)
                b = run(directory / f"r{seed}-n{n}-b{index}-f{flags}.jsonl", n, flags, seed)
                c = run(directory / f"r{seed}-n{n}-a{index+1}-f0.jsonl", n, 0, seed)
                baseline = (a["median_us"] + c["median_us"]) / 2
                optimized = b["median_us"]
                reference = results["n1_rounds"][seed - 1]["reference_us"]
                rounds.append({"seed": seed, "a_before": a, "b": b, "a_after": c,
                               "baseline_us": baseline, "candidate_us": optimized,
                               "baseline_ratio": baseline / reference,
                               "candidate_ratio": optimized / reference,
                               "delta_us": optimized - baseline,
                               "speedup_percent": 100 * (1 - optimized / baseline)})
            candidate[str(n)] = {key: estimate([x[key] for x in rounds]) for key in
                                 ("baseline_us", "candidate_us", "baseline_ratio",
                                  "candidate_ratio", "delta_us", "speedup_percent")}
            candidate[str(n)]["rounds"] = rounds
        results["candidates"][str(flags)] = candidate
    results["oracles"] = {}
    for flags in (74, 330):
        data = records(directory / f"oracles-f{flags}.jsonl")
        tests = [x for x in data if x["kind"].endswith("test") or x["kind"] == "rollback"]
        block = [x for x in tests if x["kind"] == "small_block_test"]
        assert len(block) == 3 and all(x["tolerance_ok"] for x in block)
        results["oracles"][str(flags)] = tests
    results["profile"] = {}
    for flags in (0, 74, 330):
        results["profile"][str(flags)] = {}
        for n in (2, 3, 4):
            data = records(directory / f"profile-f{flags}-n{n}.jsonl")
            results["profile"][str(flags)][str(n)] = [
                x for x in data if x["kind"] in ("kernel", "profile", "expert_union")]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blade50", required=True)
    parser.add_argument("--blade53", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"method": "Five seeds; per-run median; adjacent A/B/A baseline; paired t4 CI95",
              "prefix": 264, "measured_tokens": 216, "stock_governor": True,
              "blade50": analyze(args.blade50), "blade53": analyze(args.blade53)}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for blade in ("blade50", "blade53"):
        for flags, candidate in result[blade]["candidates"].items():
            print(blade, flags, [(n, round(x["candidate_ratio"]["mean"], 4),
                                round(x["candidate_ratio"]["ci95"], 4),
                                round(x["speedup_percent"]["mean"], 2))
                               for n, x in candidate.items()])


if __name__ == "__main__":
    main()
