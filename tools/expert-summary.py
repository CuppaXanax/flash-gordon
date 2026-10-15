"""Summarize the three full-slab paired GPU timestamp captures."""
import collections
import json
import re
import statistics
from pathlib import Path

root = Path("results") / "hillclimb"
pattern = re.compile(
    r"gate=(\d+) down=(\d+) rep=(\d+) reference=(\d+) gu_us=([\d.]+) dr_us=([\d.]+)"
)
cases = collections.defaultdict(lambda: collections.defaultdict(list))
runs = []
for number in range(1, 4):
    data = json.loads((root / f"expert-512-{number}.json").read_text(encoding="utf-8-sig"))
    assert data["ExitCode"] == 0
    assert "BASELINE_PARITY" in data["Stdout"]
    samples = collections.defaultdict(dict)
    for gate, down, rep, reference, gu, dr in pattern.findall(data["Stdout"]):
        key = (int(gate), int(down))
        variant = "before" if reference == "1" else "after"
        assert (key, variant, int(rep)) not in samples
        samples[key, variant, int(rep)] = (float(gu), float(dr))
        cases[key][variant].append((float(gu), float(dr)))
    assert len(samples) == 180
    run = {"run": number, "cases": []}
    for gate, down in [(12, 7), (13, 8), (12, 8)]:
        entry = {"gate": gate, "down": down}
        for variant in ["before", "after"]:
            points = [samples[(gate, down), variant, rep] for rep in range(30)]
            entry[variant] = {
                "gate_up_us": statistics.mean(p[0] for p in points),
                "down_us": statistics.mean(p[1] for p in points),
                "pair_us": statistics.mean(sum(p) for p in points),
            }
        run["cases"].append(entry)
    runs.append(run)

result = {
    "device": {"ip": "192.168.42.50", "reported_active_CU": 24, "subgroup": 64},
    "method": "Three runs, 30 matched selections/run/format, ABBA order, 512-expert slabs",
    "samples_per_variant_per_format": 90,
    "deployed": False,
    "runs": runs,
    "cases": [],
}
for gate, down, layers, gu_bytes, dr_bytes in [
    (12, 7, 43, 18432000, 12288000),
    (13, 8, 1, 22528000, 17408000),
    (12, 8, 4, 18432000, 17408000),
]:
    entry = {"gate_type": gate, "down_type": down, "layers": layers, "bytes": gu_bytes + dr_bytes}
    for variant in ["before", "after"]:
        points = cases[gate, down][variant]
        assert len(points) == 90
        gu = statistics.mean(p[0] for p in points)
        dr = statistics.mean(p[1] for p in points)
        entry[variant] = {
            "gate_up_us": gu,
            "down_us": dr,
            "pair_us": gu + dr,
            "gate_up_GBps": gu_bytes / gu / 1000,
            "down_GBps": dr_bytes / dr / 1000,
            "pair_GBps": (gu_bytes + dr_bytes) / (gu + dr) / 1000,
        }
    entry["pair_speedup"] = entry["before"]["pair_us"] / entry["after"]["pair_us"]
    result["cases"].append(entry)

for variant in ["before", "after"]:
    result[f"synthetic_48_layer_{variant}_us"] = sum(
        c["layers"] * c[variant]["pair_us"] for c in result["cases"]
    )
result["synthetic_saving_us"] = result["synthetic_48_layer_before_us"] - result["synthetic_48_layer_after_us"]
(root / "expert-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for case in result["cases"]:
    before, after = case["before"], case["after"]
    print(case["gate_type"], case["down_type"],
          f'{before["gate_up_us"]:.3f}/{before["down_us"]:.3f} -> {after["gate_up_us"]:.3f}/{after["down_us"]:.3f}',
          f'pair {before["pair_us"]:.3f} -> {after["pair_us"]:.3f} us',
          f'GB/s {before["pair_GBps"]:.1f} -> {after["pair_GBps"]:.1f}')
print("Synthetic 48-layer saving:", result["synthetic_saving_us"], "us; not a serving measurement")
