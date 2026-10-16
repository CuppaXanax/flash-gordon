"""Packed-byte lower bounds; routing observations must identify their provenance."""
import argparse
import collections
import json
from pathlib import Path
import statistics
import struct


def routing_unions(rows):
    streams = collections.defaultdict(dict)
    for row in rows:
        ids = row["experts"]
        if len(ids) != 10 or len(set(ids)) != 10 or any(
                not isinstance(e, int) or not 0 <= e < 512 for e in ids):
            raise ValueError("routing row requires ten distinct global experts")
        key = (row["category"], row["case"], row["layer"])
        token = row["token"]
        if token in streams[key]:
            raise ValueError("duplicate routing position")
        streams[key][token] = (set(ids), row.get("anchor", True))
    result = []
    for n in range(1, 5):
        categories = collections.defaultdict(list)
        for (category, _, layer), stream in streams.items():
            for first in sorted(stream):
                if stream[first][1] and all(first + t in stream for t in range(n)):
                    categories[category, layer].append(len(set().union(
                        *(stream[first + t][0] for t in range(n)))))
        for (category, layer), values in sorted(categories.items()):
            result.append(dict(n=n, category=category, layer=layer, windows=len(values),
                               mean_unique=statistics.mean(values),
                               min_unique=min(values), max_unique=max(values)))
    return result


def captured_routes(directory):
    rows = []
    for line in (directory / "cases.tsv").read_text().splitlines():
        case, category, session, prompt, generated, _ = line.split()
        prompt, generated = int(prompt), int(generated)
        if category not in ("chat", "code") or prompt < 1 or generated < 259:
            raise ValueError("invalid real-text routing case")
        for layer in range(48):
            stem = directory / f"routes-{int(session)}-{layer}"
            if not stem.with_suffix(".coverage").exists():
                continue
            coverage = stem.with_suffix(".coverage").read_bytes()
            data = stem.with_suffix(".u32").read_bytes()
            for token in range(prompt, prompt + 259):
                if token >= len(coverage) or coverage[token] != 1:
                    raise ValueError("hole in generated-token routing coverage")
                offset = token * 10 * 4
                if offset + 40 > len(data):
                    raise ValueError("truncated routing IDs")
                rows.append(dict(category=category, case=case, layer=layer,
                                 token=token, anchor=token < prompt + 256,
                                 experts=list(struct.unpack_from("<10I", data, offset))))
    if not rows:
        raise ValueError("capture contains no per-layer routing; final HC is insufficient")
    return rows


def family(name):
    if "gate_up" in name:
        return "expert_gate_up"
    if "down_reduce" in name or name == "fg_small_down.spv":
        return "expert_down"
    if name.startswith(("fg_dense_", "fg_small_q", "fg_small_iq",
                        "fg_small_f32", "fg_small_index", "fg_small_router")):
        return "dense"
    if "gdn" in name:
        return "gdn_sequential"
    if "qsa" in name:
        return "qsa_sequential"
    return "other"


def summarize(weights, profiles, routes=None, prefix=256, routes_origin=None):
    dense = sum(w["bytes_per_copy"] for w in weights if w["copies"] == 1)
    gu = sum(w["bytes_per_copy"] for w in weights
             if w["name"] in ("ffn_gate_exps.weight", "ffn_up_exps.weight"))
    down = sum(w["bytes_per_copy"] for w in weights
               if w["name"] == "ffn_down_exps.weight")
    gdn_layers = len({w["layer"] for w in weights if w["name"] == "ssm_a"})
    qsa_layers = len({w["layer"] for w in weights if w["name"] == "attn_q.weight"})
    state = gdn_layers * 48 * 128 * 128 * 4
    expert_bytes = collections.defaultdict(int)
    for weight in weights:
        if weight["copies"] == 512:
            expert_bytes[weight["layer"]] += weight["bytes_per_copy"]
    result = dict(
        status="optimistic byte lower bounds, not a performance prediction",
        real_target_union_available=routes is not None and routes_origin == "target-capture",
        routing_origin=routes_origin,
        dense_shared_bytes=dense, expert_gate_up_bytes_per_union_expert=gu,
        expert_down_bytes_per_union_expert=down, recurrent_state_bytes=state,
        assumptions=dict(theoretical_gbps=448, sensitivity_gbps=340,
                         sensitivity_is_not_a_measured_bandwidth=True,
                         logical_prefix_tokens=prefix,
                         excludes="activation/intermediate traffic, dispatch/submit, "
                                  "dequantization, synchronization and disk/network"),
        routing=[] if routes is None else routing_unions(routes),
        shapes=[])
    for n in range(1, 5):
        # Recurrence arithmetic is sequential, but a cached implementation can
        # retain state across all N updates and transfer it only at the edges.
        mandatory_state = 2 * state
        sequential = 2 * state * n
        scenarios = []
        for label, unique in (("perfect_overlap", 10), ("disjoint", 10 * n)):
            total = dense + (gu + down) * unique + mandatory_state
            scenarios.append(dict(
                label=label, unique_per_layer=unique, bytes=total,
                per_token_read_write_bound_bytes=total + sequential - mandatory_state,
                theoretical_us=total / 448000,
                sensitivity_us=total / 340000))
        logical_qsa = qsa_layers * sum(prefix + t + 1 for t in range(n)) * 1236
        grouped = collections.defaultdict(lambda: dict(gpu_us=0., dispatches=0))
        for entry in profiles[n]:
            if entry["kind"] == "kernel":
                item = grouped[family(entry["name"])]
                item["gpu_us"] += entry["gpu_us"]
                item["dispatches"] += entry["count"]
        observed = []
        for category in sorted({row["category"] for row in result["routing"]}):
            layer_unions = {row["layer"]: row["mean_unique"]
                            for row in result["routing"]
                            if row["n"] == n and row["category"] == category}
            complete = set(expert_bytes).issubset(layer_unions)
            if complete:
                total = dense + mandatory_state + sum(
                    count * layer_unions[layer] for layer, count in expert_bytes.items())
                observed.append(dict(category=category, bytes=total,
                                     theoretical_us=total / 448000,
                                     sensitivity_us=total / 340000))
        result["shapes"].append(dict(
            n=n, bounds=scenarios, recurrent_read_write_bytes=sequential,
            cached_recurrent_minimum_bytes=mandatory_state,
            unretained_shader_logical_state_bytes=3 * state * n,
            observed_routing_bounds=observed,
            qsa_logical_record_read_bytes=logical_qsa,
            qsa_note="sequential query work; not added to DRAM bound because "
                     "cross-query/head cache reuse is not measured",
            families=dict(grouped)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    route_args = parser.add_mutually_exclusive_group()
    route_args.add_argument("--routes", type=Path)
    route_args.add_argument("--captured-routes", type=Path)
    parser.add_argument("--routes-origin", choices=("synthetic", "target-capture", "native-draft"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    weights = read(args.directory / "physics-weights.jsonl")
    profiles = {n: read(args.directory / f"physics-profile-{n}.jsonl")
                for n in range(1, 5)}
    routes = None if args.routes is None else read(args.routes)
    if args.routes and not args.routes_origin:
        parser.error("--routes requires an explicit --routes-origin")
    if args.captured_routes:
        routes = captured_routes(args.captured_routes)
        args.routes_origin = "target-capture"
    args.output.write_text(json.dumps(summarize(
        weights, profiles, routes, routes_origin=args.routes_origin), indent=2) + "\n")


if __name__ == "__main__":
    main()
