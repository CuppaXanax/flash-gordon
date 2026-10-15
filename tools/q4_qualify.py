#!/usr/bin/env python3
"""Apply every comparable BASELINE harness metric and the independent quality gate."""
import argparse
import json
import math
from pathlib import Path


def evaluate(baseline, run, quality):
    metrics = []

    def check(name, before, after, higher):
        valid = (type(before) in (float, int) and type(after) in (float, int)
                 and math.isfinite(before) and math.isfinite(after)
                 and before > 0 and after > 0)
        ratio = after / before if valid else None
        worse = (1-ratio if higher else ratio-1)*100 if valid else None
        metrics.append(dict(name=name, baseline=before, candidate=after,
                            higher_is_better=higher, worse_percent=worse,
                            passed=valid and worse <= 3.00000001))

    prefill = {x["target_prompt_tokens"]: x for x in run.get("prefill", [])}
    for before in baseline["prefill"]:
        target = {"4K": 4096, "32K": 32768, "128K": 131072}[before["target"]]
        after = prefill.get(target, {})
        for old, new, higher in [
            ("client_prompt_tokens_over_cold_stream_ttft_tps", "client_prefill_tps", True),
            ("engine_prefill_tps", "engine_prefill_tps", True),
            ("client_ttft_ms", "stream_first_content_ms", False),
        ]:
            check(f"prefill/{before['target']}/{new}", before[old], after.get(new), higher)
        if after and after.get("cold_measurement") is not True:
            metrics.append(dict(name=f"prefill/{target}/cold", passed=False))
    decode = {x["name"]: x for x in run.get("decode", [])}
    for before in baseline["decode"]:
        if before["case"] == "short-context-quick":
            continue
        name = {"short-context-full": "short-context", "32K-context": "context-32k",
                "128K-context": "context-128k"}[before["case"]]
        after = decode.get(name, {})
        for old, new, higher in [
            ("client_sse_interval_tps_excluding_first", "client_sse_inter_event_tps_excluding_first", True),
            ("engine_decode_tps", "engine_decode_tps", True),
            ("inter_event_p50_ms", "inter_event_p50_ms", False),
            ("inter_event_p90_ms", "inter_event_p90_ms", False),
        ]:
            check(f"decode/{name}/{new}", before[old], after.get(new), higher)
    ttft = {x["name"]: x for x in run.get("ttft", [])}
    for before in baseline["ttft"]:
        check(f"ttft/{before['case']}", before["ttft_ms"],
              ttft.get(before["case"], {}).get("ttft_ms"), False)
    concurrent = run.get("concurrency") or {}
    streams = {x["name"]: x for x in concurrent.get("streams", [])}
    for key, name in [("chat_style", "chat-style"), ("agent_style", "agent-style")]:
        stream = streams.get(name, {})
        rate = stream.get("client_end_to_end_delta_events_per_second")
        if rate is None and stream.get("wall_seconds", 0) > 0 and "visible_delta_events" in stream:
            rate = stream["visible_delta_events"]/stream["wall_seconds"]
        check(f"concurrency/{name}", baseline["concurrency"][key]["end_to_end_visible_delta_events_per_second"],
              rate, True)
        check(f"concurrency/{name}/wall_seconds", baseline["concurrency"][key]["wall_seconds"],
              streams.get(name, {}).get("wall_seconds"), False)
    check("concurrency/aggregate", baseline["concurrency"]["aggregate_visible_delta_events_per_second"],
          concurrent.get("aggregate_end_to_end_delta_events_per_second", concurrent.get("aggregate_visible_event_tps")), True)
    check("full/elapsed_seconds", baseline["runs"]["full"]["elapsed_seconds"],
          run.get("metadata", {}).get("elapsed_seconds"), False)
    performance = all(x["passed"] for x in metrics)
    correctness = (not run.get("errors") and len(run.get("correctness", [])) == 5
                   and all(x.get("reference_match") is True for x in run["correctness"]))
    quality_passed = quality.get("passed") is True
    return dict(passed=bool(performance and correctness and quality_passed),
                quality_passed=quality_passed, greedy_correctness_passed=correctness,
                performance_passed=performance, threshold_percent=3,
                metrics=metrics, failed_metrics=[x["name"] for x in metrics if not x["passed"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "run", "quality", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    result = evaluate(load(args.baseline), load(args.run), load(args.quality))
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
