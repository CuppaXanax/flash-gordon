#!/usr/bin/env python3
"""Summarize paired replay captures; detailed kernel times are a separate run."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    config = next(row for row in rows if row["kind"] == "config")
    samples = [row for row in rows if row["kind"] == "sample"]
    expected = config["iterations"] * config["windows"]
    if len(samples) != expected:
        raise ValueError(f"{path}: {len(samples)} samples, expected {expected}")
    if [row["token"] for row in samples] != list(range(config["warmup"], config["warmup"] + expected)):
        raise ValueError(f"{path}: nonconsecutive token window")
    if not config["intrusive"] and any(row["static_slots"] != 2 for row in samples):
        raise ValueError(f"{path}: static replay is not active")
    return rows, config, samples, next(row for row in rows if row["kind"] == "parity")


def mean_sample(samples):
    return {
        "wall_us": statistics.mean(row["wall_us"] for row in samples),
        "layers_us": [statistics.mean(row["layers_us"][i] for row in samples) for i in range(6)],
        "windows_wall_us": [
            statistics.mean(row["wall_us"] for row in samples if row["window"] == window)
            for window in sorted({row["window"] for row in samples})
        ],
    }


def paired(directory):
    rounds = []
    for seed in (1, 2, 3):
        captures = [load(directory / f"q4-{seed}-{phase}.jsonl") for phase in ("a", "b", "a2")]
        if not all(capture[3] == captures[0][3] for capture in captures):
            raise ValueError(f"seed {seed}: output or GDN state parity failed")
        configs = [capture[1] for capture in captures]
        if [config["fusion_mask"] for config in configs] != [0, 7, 0]:
            raise ValueError(f"seed {seed}: expected mask 0/7/0")
        for key in ("weights", "seed", "warmup", "iterations", "windows", "clocks", "intrusive"):
            if len({config[key] for config in configs}) != 1:
                raise ValueError(f"seed {seed}: mismatched {key}")
        means = [mean_sample(capture[2]) for capture in captures]
        baseline_wall = (means[0]["wall_us"] + means[2]["wall_us"]) / 2
        baseline_layers = [(a + b) / 2 for a, b in zip(means[0]["layers_us"], means[2]["layers_us"])]
        rounds.append({
            "seed": seed, "baseline_wall_us": baseline_wall, "candidate_wall_us": means[1]["wall_us"],
            "saving_us": baseline_wall - means[1]["wall_us"],
            "baseline_layers_us": baseline_layers, "candidate_layers_us": means[1]["layers_us"],
            "a": means[0], "b": means[1], "a2": means[2], "parity": captures[0][3],
        })
    rng = random.Random(5817)
    bootstrap, speedup = [], []
    for _ in range(10000):
        resample = [rng.choice(rounds) for _ in rounds]
        bootstrap.append(statistics.mean(row["saving_us"] for row in resample))
        speedup.append(sum(row["baseline_wall_us"] for row in resample) /
                       sum(row["candidate_wall_us"] for row in resample))
    bootstrap.sort()
    speedup.sort()
    baseline = statistics.mean(row["baseline_wall_us"] for row in rounds)
    candidate = statistics.mean(row["candidate_wall_us"] for row in rounds)
    return {
        "rounds": rounds, "baseline_wall_us": baseline, "candidate_wall_us": candidate,
        "saving_us": baseline - candidate, "saving_percent": (baseline - candidate) / baseline * 100,
        "round_bootstrap_saving_us_95": [bootstrap[250], bootstrap[9749]],
        "round_bootstrap_speedup_95": [speedup[250], speedup[9749]],
        "baseline_layers_us": [statistics.mean(row["baseline_layers_us"][i] for row in rounds) for i in range(6)],
        "candidate_layers_us": [statistics.mean(row["candidate_layers_us"][i] for row in rounds) for i in range(6)],
    }


def traffic(dispatch):
    """Unique logical traffic, not DRAM counters or full scratch descriptor spans."""
    name = dispatch["name"]
    b, p = dispatch["binding_bytes"], dispatch["push"]
    if name.startswith("fg_shared_swiglu"):
        return b[0] + b[3], "exact weight storage"
    if name.startswith("fg_dense"):
        return b[0], "exact weight storage; activation traffic omitted"
    if name == "fg_moe_decode_gate_up.spv":
        return (b[0] + b[1]) // 512 * 10, "10 selected expert weight matrices"
    if name.startswith("fg_moe_decode_down"):
        return b[0] // 512 * 10, "10 selected expert weight matrices"
    if name == "fg_group_rms_norm.spv":
        return p[0] * p[1] * p[2] * 12, "logical input + norm weights + output"
    if name == "fg_hc_inject_partial.spv":
        return 10240 * 4 * 5 + p[3] * 16, "logical unique norm + inject weights + partials"
    if name == "fg_gr_mix_partial.spv":
        return 10240 * 8 + p[3] * 16 + 2560 * 4 + 16, "logical norm + logits + partials + outputs"
    if name == "fg_gr_write.spv":
        return 40960 * 2 + 10240 + 16, "logical hyper read/write + block + gates"
    if name == "fg_moe_shared_gr_write.spv":
        return 40960 * 2 + 10240 * 2 + 20, "logical unique inputs + output; cache reuse assumed"
    if name == "fg_moe_shared_add.spv":
        return 10240 * 3 + 4, "logical two inputs + output + scalar"
    if name == "fg_silu_scaled.spv":
        return p[0] * 8, "logical input + output"
    if name == "fg_swiglu.spv":
        return p[0] * 12, "logical two inputs + output"
    if name == "fg_quantize_q8_k.spv":
        return p[0] * 4 + p[1] * 296, "logical activation read + packed output"
    if name == "fg_quantize_q8_0.spv":
        return p[0] * 4 + p[1] * 34, "logical activation read + packed output"
    if name.startswith("fg_router_top10"):
        return 2048 + 80 + (360 if "tiles" in name else 0), "logical logits + routes/gates/schedule"
    if name == "fg_decode_tile_schedule.spv":
        return 400, "logical selected IDs + tile schedule"
    if name == "fg_gdn_conv_decode.spv":
        return 10240 * (4 + 32 + 16 + 4), "logical projection + state R/W + weights + output"
    if name == "fg_gdn_recurrent_algebraic.spv":
        return 3145728 * 2 + 40960 + 24576 * 2 + 48 * 16 + 128 * 4, "logical recurrent state R/W + small tensors"
    if name == "fg_qsa_record_commit.spv":
        return 1236 * 2 + 136, "logical QSA record read/write + index history write"
    if name == "fg_qsa_record_gather.spv":
        return (p[2] * 4 + p[4]) * 1236 * 2, "logical selected records read/write"
    if name in ("fg_qsa_attention.spv", "fg_qsa_decode_attention_split.spv"):
        return p[0] * 1088 + 6144 * 12, "unique KV + query/gate/output; per-head reloads omitted"
    if name == "fg_qsa_attention_merge.spv":
        return p[0] * (6144 + 48) * 4 + 6144 * 8, "logical split partials + gate/output"
    if name == "fg_qsa_prepare.spv":
        return 104460, "logical unique Q/G/K + norms + position + outputs"
    if name == "fg_qsa_index_prepare.spv":
        return 4620, "logical unique raw index query + norm + position + output"
    return None, "not estimated"


def scoped_dispatches(rows):
    phase = {}
    for row in rows:
        if row["kind"] != "dispatch":
            continue
        layer, index, name = row["layer"], row["index"], row["name"]
        current = phase.get(layer, "gr_attn_read")
        if index == 6:
            current = "qsa_projection" if layer % 4 == 3 else "gdn_projection"
        if name in ("fg_gdn_conv_decode.spv", "fg_gdn_recurrent_algebraic.spv"):
            current = "gdn_recurrent"
        elif current == "gdn_recurrent" and name.startswith("fg_dense"):
            current = "gdn_output"
        elif name == "fg_qsa_record_commit.spv":
            current = "qsa_state_attention"
        elif current == "qsa_state_attention" and name.startswith("fg_dense"):
            current = "qsa_output"
        elif name == "fg_gr_write.spv" and current in ("gdn_output", "qsa_output"):
            current = "gr_attn_write"
        elif current == "gr_attn_write":
            current = "gr_ffn_read"
        if current == "gr_ffn_read" and name == "fg_dense_f32.spv" and row["binding_bytes"][0] == 5242880:
            current = "router"
        elif name == "fg_quantize_q8_k.spv":
            current = "router_quantization"
        elif current == "router_quantization":
            current = "shared_expert"
        if name.startswith("fg_router_top10"):
            current = "expert_decode"
        if name == "fg_moe_shared_add.spv":
            current = "moe_reduce"
        if name == "fg_moe_shared_gr_write.spv" or (name == "fg_gr_write.spv" and current == "moe_reduce"):
            current = "gr_ffn_write"
        phase[layer] = current
        yield row, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch-csv", type=Path, required=True)
    args = parser.parse_args()
    result = paired(args.captures)
    result["scope"] = "synthetic layers 6..11, static GDN replay, short committed contexts"
    result["dispatches"] = {}
    result["profiles"] = {}
    with args.dispatch_csv.open("w", newline="") as stream:
        fields = ["mask", "layer", "index", "scope", "shader", "barrier", "grid", "push",
                  "reference_bytes", "ideal_us_at_450GBs", "bytes_basis", "intrusive_scope_mean_us"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mask, phase in ((0, "a"), (7, "b")):
            rows, _, _, _ = load(args.captures / f"q4-1-{phase}.jsonl")
            profile_rows, _, _, profile_parity = load(args.captures / f"profile-{mask}.jsonl")
            kernels = [row for row in profile_rows if row["kind"] == "kernel"]
            times = {(row["scope"], row["name"]): row["gpu_us"] / row["count"] for row in kernels}
            result["profiles"][mask] = {"kernels": kernels, "parity": profile_parity,
                **next(row for row in profile_rows if row["kind"] == "profile")}
            counts = {layer: 0 for layer in range(6, 12)}
            for dispatch, scope in scoped_dispatches(rows):
                counts[dispatch["layer"]] += 1
                byte_count, basis = traffic(dispatch)
                writer.writerow({
                    "mask": mask, "layer": dispatch["layer"], "index": dispatch["index"],
                    "scope": scope, "shader": dispatch["name"], "barrier": dispatch["barrier"],
                    "grid": json.dumps(dispatch["grid"]), "push": json.dumps(dispatch["push"]),
                    "reference_bytes": byte_count, "ideal_us_at_450GBs": byte_count / 450000 if byte_count else "",
                    "bytes_basis": basis, "intrusive_scope_mean_us": times.get((scope, dispatch["name"]), ""),
                })
            result["dispatches"][mask] = counts
    if result["profiles"][0]["parity"] != result["profiles"][7]["parity"]:
        raise ValueError("intrusive profile output or GDN state parity failed")
    result["ablations"] = {}
    for path in sorted(args.captures.glob("ablation-*.jsonl")):
        _, config, samples, parity = load(path)
        result["ablations"][config["fusion_mask"]] = {**mean_sample(samples), "parity": parity}
    if len({json.dumps(row["parity"], sort_keys=True) for row in result["ablations"].values()}) > 1:
        raise ValueError("ablation output or GDN state parity failed")
    result["controls"] = {}
    for path in sorted(args.captures.glob("no-clocks-*.jsonl")) + sorted(args.captures.glob("q8-*.jsonl")):
        _, _, samples, parity = load(path)
        result["controls"][path.stem] = {**mean_sample(samples), "parity": parity}
    for prefix in ("no-clocks", "q8"):
        if result["controls"][f"{prefix}-0"]["parity"] != result["controls"][f"{prefix}-7"]["parity"]:
            raise ValueError(f"{prefix} output or GDN state parity failed")
    delta = [a - b for a, b in zip(result["baseline_layers_us"], result["candidate_layers_us"])]
    gdn_saving = statistics.mean(delta[i] for i in (0, 2, 3, 4))
    qsa_saving = statistics.mean(delta[i] for i in (1, 5))
    fleet_saving_ms = (36 * gdn_saving + 12 * qsa_saving) / 1000
    result["projection_not_measurement"] = {
        "gdn_saving_us": gdn_saving, "qsa_saving_us": qsa_saving,
        "48_layer_saving_ms": fleet_saving_ms,
        "anchored_ms_token": 37.437 - fleet_saving_ms,
        "anchored_tps": 1000 / (37.437 - fleet_saving_ms),
        "warning": "Different clock/weight/context operating points; no fleet promotion or TPS claim.",
    }
    for phase in ("baseline", "candidate"):
        values = result[f"{phase}_layers_us"]
        result["projection_not_measurement"][f"raw_{phase}_48_layer_ms"] = (
            36 * statistics.mean(values[i] for i in (0, 2, 3, 4)) +
            12 * statistics.mean(values[i] for i in (1, 5))) / 1000
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("baseline_wall_us", "candidate_wall_us",
        "saving_percent", "round_bootstrap_saving_us_95", "projection_not_measurement")}, indent=2))


if __name__ == "__main__":
    main()
