#!/usr/bin/env python3
"""Analyze round-paired roofline replay evidence, not intrusive token timings."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import random
import statistics

spec = importlib.util.spec_from_file_location("fusion", Path(__file__).with_name("fusion-ledger.py"))
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)
mean = statistics.mean


def canonical(name):
    if name.startswith("fg_roof_dense"):
        return "fg_dense_compact.spv"
    if name.startswith("fg_roof_qsa"):
        return "fg_qsa_decode_attention_split.spv"
    return {"fg_roof_router.spv": "fg_router_top10.spv",
            "fg_roof_hc_inject.spv": "fg_hc_inject_partial.spv",
            "fg_roof_hc_mix.spv": "fg_gr_mix_partial.spv"}.get(name, name)


def interval(rounds, key, seed=5817):
    rng = random.Random(seed)
    values = sorted(mean(rng.choice(rounds)[key] for _ in rounds) for _ in range(10000))
    return [values[250], values[9749]]


def paired(directory, cu):
    rounds = []
    for seed in range(1, 6):
        captures = [fusion.load(directory / f"cu{cu}-{seed}-{phase}.jsonl") for phase in ("a", "b", "a2")]
        if any(c[3] != captures[0][3] for c in captures):
            raise ValueError(f"CU{cu} seed{seed}: byte-exact hash gate failed")
        configs = [c[1] for c in captures]
        for key in ("weights", "seed", "warmup", "iterations", "windows", "clocks", "intrusive", "fusion_mask"):
            if len({c[key] for c in configs}) != 1:
                raise ValueError(f"mismatched {key}")
        flags = [next(r for r in c[0] if r["kind"] == "roofline") for c in captures]
        if [r["mask"] for r in flags] != ["0", "27", "0"]:
            raise ValueError("not the qualified A/B/A")
        if any(r["dense_variant"] != "7" or r["subgroup_size"] != 64 for r in flags):
            raise ValueError("wrong dense geometry or subgroup size")
        means = [fusion.mean_sample(c[2]) for c in captures]
        a = (means[0]["wall_us"] + means[2]["wall_us"]) / 2
        b = means[1]["wall_us"]
        layers_a = [(x + y) / 2 for x, y in zip(means[0]["layers_us"], means[2]["layers_us"])]
        layers_b = means[1]["layers_us"]
        delta = [x - y for x, y in zip(layers_a, layers_b)]
        gdn = mean(delta[i] for i in (0, 2, 3, 4))
        qsa = mean(delta[i] for i in (1, 5))
        parity = json.loads((directory / f"cu{cu}-{seed}-parity.json").read_text())
        if not all(r["bit_exact"] and r["changed"] == 0 for r in parity.values()):
            raise ValueError("all-element dump parity failed")
        rounds.append({"seed": seed, "baseline_wall_us": a, "candidate_wall_us": b,
                       "saving_us": a - b, "saving_percent": (a - b) / a * 100,
                       "speedup": a / b, "baseline_layers_us": layers_a, "candidate_layers_us": layers_b,
                       "gdn_saving_us": gdn, "qsa_saving_us": qsa,
                       "projected_48_layer_saving_ms": (36 * gdn + 12 * qsa) / 1000,
                       "windows": means, "parity": parity})
    a, b = mean(r["baseline_wall_us"] for r in rounds), mean(r["candidate_wall_us"] for r in rounds)
    saving = mean(r["projected_48_layer_saving_ms"] for r in rounds)
    projection_ci = interval(rounds, "projected_48_layer_saving_ms")
    result = {
        "rounds": rounds, "baseline_wall_us": a, "candidate_wall_us": b,
        "saving_us": a - b, "saving_percent": (a - b) / a * 100,
        "saving_us_95_ci": interval(rounds, "saving_us"),
        "saving_percent_95_ci": interval(rounds, "saving_percent"),
        "speedup_95_ci": interval(rounds, "speedup"),
        "baseline_layers_us": [mean(r["baseline_layers_us"][i] for r in rounds) for i in range(6)],
        "candidate_layers_us": [mean(r["candidate_layers_us"][i] for r in rounds) for i in range(6)],
        "projection_not_measurement": {
            "gdn_saving_us": mean(r["gdn_saving_us"] for r in rounds),
            "qsa_saving_us": mean(r["qsa_saving_us"] for r in rounds),
            "48_layer_saving_ms": saving, "48_layer_saving_ms_95_ci": projection_ci,
            "historical_anchor_ms_token": 37.437, "anchored_ms_token": 37.437 - saving,
            "anchored_ms_token_95_ci": [37.437 - projection_ci[1], 37.437 - projection_ci[0]],
            "anchored_tps": 1000 / (37.437 - saving),
            "warning": "Different weight/context/clock operating points; not eight-rank measurement or deployment.",
        },
    }
    result["controls"] = {}
    for label in ("no-clocks", "long"):
        controls = [fusion.load(directory / f"cu{cu}-{label}-{mask}.jsonl") for mask in (0, 27)]
        if controls[0][3] != controls[1][3]:
            raise ValueError(f"{label}: parity failed")
        result["controls"][label] = [fusion.mean_sample(c[2]) for c in controls]
    return result


def profile(directory, cu, mask):
    groups, envelopes = {}, []
    for seed in range(1, 6):
        rows, _, _, parity = fusion.load(directory / f"cu{cu}-profile-{seed}-{mask}.jsonl")
        other = fusion.load(directory / f"cu{cu}-profile-{seed}-{27-mask}.jsonl")[3]
        if parity != other:
            raise ValueError("intrusive profile parity failed")
        envelopes.append(next(r for r in rows if r["kind"] == "profile"))
        for r in rows:
            if r["kind"] == "kernel":
                groups.setdefault((r["scope"], canonical(r["name"])), []).append(r["gpu_us"] / r["count"])
    return {key: mean(values) for key, values in groups.items()}, envelopes


def dispatch_rows(rows):
    normalized = [dict(r, name=canonical(r["name"])) if r["kind"] == "dispatch" else r for r in rows]
    for r, scope in fusion.scoped_dispatches(normalized):
        if scope == "gdn_projection":
            scope = {"15155200": "gdn_qkv", "9093120": "gdn_gate"}.get(str(r["binding_bytes"][0]), "gdn_controls")
        yield r, scope


def ledger(directory, cu):
    times_a, envelopes_a = profile(directory, cu, 0)
    times_b, envelopes_b = profile(directory, cu, 27)
    rows, _, _, _ = fusion.load(directory / f"cu{cu}-1-a.jsonl")
    estimates, ranking = [], {}
    for dispatch, scope in dispatch_rows(rows):
        key = scope, dispatch["name"]
        if key not in times_a or key not in times_b:
            raise ValueError(f"unmatched scope/shader {key}")
        byte_count, basis = fusion.traffic(dispatch)
        if byte_count is None:
            raise ValueError(f"unestimated traffic {key}")
        a, b = times_a[key], times_b[key]
        ideal = byte_count / 450000
        weight = 6 if dispatch["layer"] % 4 == 3 else 9
        excess = max(0, a - max(ideal, 0.934)) * weight
        row = {"cu": cu, "layer": dispatch["layer"], "index": dispatch["index"], "scope": scope,
               "shader": dispatch["name"], "grid": json.dumps(dispatch["grid"]), "reference_bytes": byte_count,
               "bytes_basis": basis, "ideal_us_450GBs": ideal, "baseline_scope_mean_us": a,
               "candidate_scope_mean_us": b, "baseline_pct_450GBs": ideal / a * 100,
               "candidate_pct_450GBs": ideal / b * 100, "candidate_x_dependency_floor": b / 0.934,
               "model_excess_us_over_max_byte_or_floor": excess}
        estimates.append(row)
        group = ranking.setdefault(key, {"scope": scope, "shader": dispatch["name"], "model_occurrences": 0,
                                        "model_excess_us": 0, "baseline_us": a, "candidate_us": b,
                                        "reference_bytes": byte_count})
        group["model_occurrences"] += weight
        group["model_excess_us"] += excess
    return estimates, sorted(ranking.values(), key=lambda r: r["model_excess_us"], reverse=True), [envelopes_a, envelopes_b]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dispatch-csv", required=True, type=Path)
    args = parser.parse_args()
    result, all_dispatches = {}, []
    for cu in (40, 24):
        value = paired(args.captures, cu)
        dispatches, ranking, envelopes = ledger(args.captures, cu)
        value["ranked_baseline_excess"] = ranking
        value["intrusive_envelopes"] = envelopes
        value["dispatch_count"] = len(dispatches)
        result[cu] = value
        all_dispatches.extend(dispatches)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.dispatch_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_dispatches[0]))
        writer.writeheader()
        writer.writerows(all_dispatches)
    for cu, r in result.items():
        print(json.dumps({"cu": cu, **{k: r[k] for k in ("baseline_wall_us", "candidate_wall_us",
            "saving_percent", "saving_us_95_ci", "projection_not_measurement")}}, indent=2))


if __name__ == "__main__":
    main()
