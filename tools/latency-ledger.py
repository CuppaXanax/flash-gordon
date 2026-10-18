"""Classify the replay ledger using measured floors, not average dispatch time."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def classify(row):
    ideal = float(row["ideal_us_at_450GBs"])
    measured = float(row["intrusive_scope_mean_us"])
    # Within 1.6x of the ideal byte roof is 281-450 GB/s logical payload.
    if ideal >= 3.0 and measured <= 1.6 * ideal:
        kind = "bandwidth"
    elif ideal < 3.0 and measured <= 2.52:
        kind = "floor"
    else:
        kind = "excess"
    envelope = max(ideal, .934)
    return kind, envelope, max(0, measured - envelope)


def summarize(rows):
    groups = defaultdict(lambda: {"dispatches": 0, "diagnostic_us": 0,
                                  "ideal_us": 0, "envelope_us": 0})
    shaders = defaultdict(lambda: {"dispatches": 0, "diagnostic_us": 0,
                                   "ideal_us": 0, "envelope_us": 0})
    for row in rows:
        kind, floor, _ = classify(row)
        for dest in (groups[kind], shaders[row["shader"]]):
            dest["dispatches"] += 1
            dest["diagnostic_us"] += float(row["intrusive_scope_mean_us"])
            dest["ideal_us"] += float(row["ideal_us_at_450GBs"])
            dest["envelope_us"] += floor
    return {"classes": dict(groups), "shaders": dict(shaders)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    with args.ledger.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("empty dispatch ledger")
    fields = list(rows[0]) + ["classification", "lower_envelope_us",
                            "excess_over_envelope_us", "diagnostic_effective_GBs"]
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        for row in rows:
            kind, envelope, excess = classify(row)
            measured = float(row["intrusive_scope_mean_us"])
            writer.writerow(dict(row, classification=kind,
                                 lower_envelope_us=envelope,
                                 excess_over_envelope_us=excess,
                                 diagnostic_effective_GBs=float(row["reference_bytes"]) / measured / 1000))
    baseline = [r for r in rows if r["mask"] == "0"]
    layers = sorted(set(r["layer"] for r in baseline), key=int)
    if len(layers) != 6:
        raise ValueError("expected fusion experiment's six-layer ledger")
    by_type = {}
    for name, layer_count in [("GDN", 36), ("QSA", 12)]:
        selected = [r for r in baseline if (int(r["layer"]) % 4 == 3) == (name == "QSA")]
        count = len(set(r["layer"] for r in selected))
        per = summarize(selected)
        for group in per["classes"].values():
            for key in group:
                group[key] = group[key] / count * layer_count
        per["model_layer_count"] = layer_count
        by_type[name] = per
    summary = {
        "definitions": {
            "bandwidth": "ideal >= 3 us and diagnostic <= 1.6 * ideal (logical >= 281.25 GB/s)",
            "floor": "ideal < 3 us and diagnostic <= 2.52 us (1.5x measured 1.68 us dependent launch)",
            "excess": "above both operational envelopes; not proof all excess is removable",
            "lower_envelope_us": "max(bytes/450GBs, measured 0.934 us dependent-chain step); not an achieved kernel target",
            "warning": "Intrusive scope averages are attached to each recording, not per-dispatch replay timestamps; summing mixed-shape means can differ from layer clocks.",
        },
        "counts_by_mask": dict(Counter(r["mask"] for r in rows)),
        "baseline_block": summarize(baseline),
        "type_weighted_model_diagnostic": by_type,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"baseline": summary["baseline_block"]["classes"],
                      "counts": summary["counts_by_mask"]}, indent=2))


if __name__ == "__main__":
    main()
