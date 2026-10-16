import json
import math
import random
import statistics
from pathlib import Path

root = Path("results/hillclimb/multisess-fleet-20261016")
pairs = []
for index in range(5):
    pair = {}
    for arm in ("A", "B"):
        path = root / f"pair-{index}-{arm}" / "battery.json"
        if not path.exists():
            break
        battery = json.loads(path.read_text(encoding="utf-8-sig"))
        if not battery["complete"]:
            break
        pair[arm] = battery
    if len(pair) == 2:
        pairs.append(pair)
if len(pairs) < 5:
    raise SystemExit(f"Only {len(pairs)} complete pairs; no decision.")

def quantile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]

rows = []
for name in pairs[0]["A"]["metrics"]:
    a = [pair["A"]["metrics"][name]["value"] for pair in pairs]
    b = [pair["B"]["metrics"][name]["value"] for pair in pairs]
    logs = [math.log(y / x) for x, y in zip(a, b)]
    rng = random.Random(1729)
    draws = [math.exp(statistics.median(rng.choices(logs, k=5))) for _ in range(20000)]
    bootstrap = [quantile(draws, .025), quantile(draws, .975)]
    se = statistics.stdev(logs) / math.sqrt(5)
    mean = statistics.mean(logs)
    ci = [min(bootstrap[0], math.exp(mean - 2.776445105 * se)),
          max(bootstrap[1], math.exp(mean + 2.776445105 * se))]
    higher = pairs[0]["A"]["metrics"][name]["higher_is_better"]
    median_ratio = statistics.median(b) / statistics.median(a)
    paired_ratio = math.exp(statistics.median(logs))
    operator_pass = (median_ratio >= .97 and paired_ratio >= .97 and ci[1] >= .97
                     if higher else
                     median_ratio <= 1.03 and paired_ratio <= 1.03 and ci[0] <= 1.03)
    rows.append(dict(metric=name, A=statistics.median(a), B=statistics.median(b),
                     B_over_A=median_ratio, paired_median_ratio=paired_ratio,
                     bootstrap_ci95=bootstrap, ci95=ci, higher_is_better=higher,
                     operator_pass=operator_pass,
                     strict_noninferior=(ci[0] >= .97 if higher else ci[1] <= 1.03),
                     A_samples=a, B_samples=b))

protected = [r for r in rows if r["metric"].startswith("solo-") or r["metric"] == "chat.ttft_ms"]
aggregate = next(r for r in rows if r["metric"] == "concurrency.aggregate_wall_tps")
hashes = []
concurrent_hashes = []
for index, pair in enumerate(pairs):
    for context in (4096, 32768):
        a = next(x for x in pair["A"]["solo"] if x["target"] == context)
        b = next(x for x in pair["B"]["solo"] if x["target"] == context)
        hashes.append(dict(pair=index, context=context,
                           cold_equal=a["cold"]["Content"] == b["cold"]["Content"],
                           decode_equal=a["decode"]["Content"] == b["decode"]["Content"]))
    for mode in ("concurrency", "mixed"):
        for trial in pair["B"][mode]["concurrency"]["trials"]:
            for stream in trial["load"]["streams"]:
                if stream["same_prompt_hash_match"] is not None:
                    concurrent_hashes.append(dict(pair=index, mode=mode, stream=stream["name"],
                                                   equal=stream["same_prompt_hash_match"]))
result = dict(pairs=5, metrics=rows, protected_pass=all(r["operator_pass"] for r in protected),
              aggregate_median_increases=aggregate["B_over_A"] > 1,
              aggregate_improves=aggregate["B_over_A"] > 1 and aggregate["ci95"][0] > 1,
              aggregate_clear_win=aggregate["ci95"][0] > 1,
              solo_hashes=hashes, solo_hash_identity=all(
                  h["cold_equal"] and h["decode_equal"] for h in hashes),
              concurrent_hashes=concurrent_hashes,
              concurrent_hash_identity=bool(concurrent_hashes) and all(h["equal"] for h in concurrent_hashes),
              method="Paired median log-ratio percentile bootstrap, 20000 fixed-seed resamples; "
                     "CI expanded by df=4 Student-t 95% interval for mean log-ratio. "
                     "Operator gate is the user's relaxed median/no-clear-regression rule; "
                     "strict noninferiority reported separately.")
(root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
for row in rows:
    print(f'{row["metric"]}: {row["A"]:.3f} -> {row["B"]:.3f} '
          f'ratio={row["B_over_A"]:.4f} CI={row["ci95"]} pass={row["operator_pass"]}')
print("DECISION INPUTS", {k: v for k, v in result.items()
                           if k not in ("metrics", "solo_hashes", "concurrent_hashes", "method")})
