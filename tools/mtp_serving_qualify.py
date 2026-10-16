#!/usr/bin/env python3
"""Bounded HTTP correctness/performance client; never starts or stops a ring."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import re
import statistics
import time
import urllib.request

MODEL = "Qwen3.8-Flash-Next-Uncensored"
CASES = [
    ("chat-0", "chat", "/no_think Explain why leaves look green in three sentences."),
    ("chat-1", "chat", "/no_think Plan a quiet two-day visit to Kyoto with one vegetarian meal daily."),
    ("chat-2", "chat", "/no_think Explain TCP retransmission and why latency differs from bandwidth."),
    ("chat-3", "chat", "/no_think Give a numbered checklist for moving house without losing documents."),
    ("code-0", "code", "/no_think Write a C17 binary search over sorted uint32_t values; include edge tests."),
    ("code-1", "code", "/no_think Write a Python bounded LRU cache using OrderedDict, with three tests."),
    ("code-2", "code", "/no_think Write a TypeScript async retry helper with exponential backoff and cancellation."),
    ("code-3", "code", "/no_think Write a C17 function that decodes one UTF-8 scalar and rejects malformed input."),
]


def semantic(response):
    choice = response["choices"][0]
    return {"message": choice["message"], "finish_reason": choice["finish_reason"],
            "completion_tokens": response["usage"]["completion_tokens"]}


def decode_sse(raw):
    events = []
    done = False
    for frame in raw.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        assert frame.startswith("data: "), f"invalid SSE frame: {frame[:100]}"
        payload = frame[6:]
        if payload == "[DONE]":
            assert not done
            done = True
        else:
            assert not done, "event after DONE"
            events.append(json.loads(payload))
    assert done and events, "missing DONE or completion events"
    choices = [event["choices"][0] for event in events if event.get("choices")]
    assert choices[0]["delta"].get("role") == "assistant"
    assert choices[-1]["finish_reason"] in ("length", "stop", "tool_calls")
    content = "".join(choice["delta"].get("content", "") for choice in choices)
    reasoning = "".join(choice["delta"].get("reasoning_content", "") for choice in choices)
    return content, reasoning, choices[-1]["finish_reason"]


def request(base, prompt, user, limit=128, stream=False, stop=None):
    body = {"model": MODEL, "temperature": 0, "max_tokens": limit,
            "user": user, "stream": stream, "messages": [{"role": "user", "content": prompt}]}
    if stop is not None:
        body["stop"] = stop
    started = time.perf_counter()
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as response:
        headers = dict(response.headers)
        raw = response.read()
    elapsed = time.perf_counter() - started
    if stream:
        content, reasoning, finish = decode_sse(raw)
        return {"content": content, "reasoning": reasoning, "finish_reason": finish,
                "wall_seconds": elapsed, "raw_sha256": hashlib.sha256(raw).hexdigest()}
    result = json.loads(raw)
    decode = float(headers["X-Flash-Gordon-Decode-Seconds"])
    tokens = result["usage"]["completion_tokens"]
    assert tokens > 0 and decode > 0, "missing decode accounting"
    return {"id": result["id"], "semantic": semantic(result), "tokens": tokens,
            "decode_seconds": decode, "decode_tps": tokens / decode,
            "wall_seconds": elapsed, "wall_tps": tokens / elapsed, "headers": headers}


def run_suite(args):
    rows = []
    budget_path = pathlib.Path(args.output).parent / "diagnostic-budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else None
    repeats = 3
    if budget is not None:
        assert getattr(args, "diagnostic", False), "budget override requires explicit diagnostic mode"
        assert budget.get("diagnostic_only") is True
        repeats = budget["perf_repeats"]
        assert type(repeats) is int and 1 <= repeats <= 3
    if args.suite == "greedy":
        for name, category, prompt in CASES:
            rows.append({"case": name, "category": category,
                         **request(args.base, prompt, f"greedy-{name}", 96)})
    elif args.suite == "perf":
        for repeat in range(repeats):
            for name, category, prompt in CASES:
                if not name.endswith(("-0", "-1")):
                    continue
                rows.append({"case": name, "category": category, "repeat": repeat,
                             **request(args.base, prompt, f"perf-{name}", 128)})
        for repeat in range(0 if args.single_session else 3):
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = [pool.submit(request, args.base, case[2], f"coexist-{case[1]}", 128)
                           for case in (CASES[0], CASES[4])]
                pair = [future.result() for future in pending]
            wall = time.perf_counter() - started
            rows.append({"case": "coexist", "repeat": repeat, "requests": pair,
                         "aggregate_wall_tps": sum(row["tokens"] for row in pair) / wall})
    else:
        prompt = CASES[0][2]
        plain = request(args.base, prompt, "stream-plain", 32)
        streamed = request(args.base, prompt, "stream-sse", 32, stream=True)
        assert streamed["content"] == plain["semantic"]["message"].get("content", "")
        assert streamed["reasoning"] == plain["semantic"]["message"].get("reasoning_content", "")
        assert streamed["finish_reason"] == plain["semantic"]["finish_reason"]
        text = streamed["content"]
        assert len(text) >= 12, "insufficient visible text for stop test"
        stop = text[5:11]
        for stream in (False, True):
            stopped = request(args.base, prompt, f"custom-stop-{stream}", 32, stream=stream, stop=stop)
            visible = stopped["content"] if stream else stopped["semantic"]["message"]["content"]
            finish = stopped["finish_reason"] if stream else stopped["semantic"]["finish_reason"]
            assert visible == text[:text.index(stop)] and finish == "stop"
            rows.append({"case": f"custom-stop-{stream}", "result": stopped})
        # Closing the connection during a committed burst must not poison either slot.
        body = {"model": MODEL, "temperature": 0, "max_tokens": 256, "stream": True,
                "user": "cancel-chat", "messages": [{"role": "user", "content": CASES[4][2]}]}
        req = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as response:
            while True:
                line = response.readline()
                assert line, "stream ended before cancellation point"
                if line.startswith(b"data: ") and b'"content"' in line:
                    break
        def after_cancel(case):
            return request(args.base, case[2], f"after-cancel-{case[1]}", 32)
        if args.single_session:
            pair = [after_cancel(case) for case in (CASES[0], CASES[4])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pair = list(pool.map(after_cancel, (CASES[0], CASES[4])))
        rows.extend([{"case": "stream-parity", "result": streamed},
                     {"case": "post-cancel-coexist", "requests": pair}])
    output = pathlib.Path(args.output)
    output.write_text(json.dumps({"phase": args.phase, "suite": args.suite, "rows": rows,
                                 "diagnostic_budget": budget},
                                 indent=2) + "\n", encoding="utf-8")
    print(f"{args.phase} {args.suite}: {len(rows)} cases PASS")


def token_receipts(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    return {match[0]: (int(match[1]), match[2]) for match in
            re.findall(r"API_TOKEN_HASH id=(\S+) count=(\d+) sha256=([0-9a-f]{64})", text)}


def verify(args):
    candidate = json.loads(pathlib.Path(args.candidate).read_text(encoding="utf-8"))
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    current_hashes, baseline_hashes = token_receipts(args.trace), token_receipts(args.baseline_trace)
    assert len(candidate["rows"]) == len(baseline["rows"]) == 8
    failures = []
    for current, expected in zip(candidate["rows"], baseline["rows"]):
        assert current["case"] == expected["case"]
        if (current["semantic"] != expected["semantic"] or
                current_hashes[current["id"]] != baseline_hashes[expected["id"]]):
            failures.append(current["case"])
        assert current_hashes[current["id"]][0] == current["tokens"], "hash count differs"
    events_path = getattr(args, "correctness_events", None)
    quality_path = getattr(args, "quality", None)
    if events_path or quality_path:
        from mtp_correctness import evaluate
        assert events_path and quality_path and args.report_output, "new MTP gate needs events, Q8 quality and report"
        events = json.loads(pathlib.Path(events_path).read_text())
        assert {r["id"] for r in events["requests"]} == {r["id"] for r in candidate["rows"]}
        for row in events["requests"]:
            assert (row["tokens"], row["tokens_sha256"]) == current_hashes[row["id"]]
        gate = evaluate(events, json.loads(pathlib.Path(quality_path).read_text()))
        gate["historical_identity_mismatch_cases"] = failures
        pathlib.Path(args.report_output).write_text(json.dumps(gate, indent=2)+"\n")
        assert gate["passed"], gate
        print("Pass-local argmax, small same-prefix margins and original-Q8 KL/top1: PASS")
        return
    if getattr(args, "diagnostic", False):
        assert args.report_output, "diagnostic comparison requires a persistent report"
        pathlib.Path(args.report_output).write_text(json.dumps({
            "diagnostic_only": True, "qualified": False, "identity_passed": not failures,
            "passed": 8-len(failures), "failed_cases": failures}, indent=2) + "\n")
        print(f"DIAGNOSTIC greedy identity: {8-len(failures)}/8; not a qualification")
    else:
        assert not failures, failures
        print("Same-build target token-ID SHA256/count and public completion identity: 8/8 PASS")


def summarize(args):
    directory = pathlib.Path(args.directory)
    phases = ["A0", "K1", "A1", "K2", "A2", "K3", "A3"]
    expanded = (directory / "B0-perf.json").exists()
    single = (directory / "D3-perf.json").exists()
    n4_only = (directory / "F3-perf.json").exists() and not single
    if n4_only:
        single = True
    if expanded:
        phases = ["A0", "B0", "K1", "S1", "A1", "K2", "S2", "A2", "K3", "S3", "A3"]
    if single:
        phases = ["A0", "D3", "A1", "F1", "A2", "F2", "A3", "F3", "A4"]
    if n4_only:
        phases = ["A0", "F3", "A1"]
    data = {phase: json.loads((directory / f"{phase}-perf.json").read_text(encoding="utf-8"))
            for phase in phases}
    comparisons = []
    arms = [(1, 0, 2), (3, 2, 4), (5, 4, 6)]
    if expanded:
        arms = [(2, 0, 4), (3, 0, 4), (5, 4, 7), (6, 4, 7), (8, 7, 10), (9, 7, 10)]
    if single:
        arms = [(1, 0, 2), (3, 2, 4), (5, 4, 6), (7, 6, 8)]
    if n4_only:
        arms = [(1, 0, 2)]
    for index, left, right in arms:
        phase = phases[index]
        before, candidate, after = (data[phases[i]]["rows"] for i in (left, index, right))
        grouped = {}
        for a, b, c in zip(before, candidate, after):
            assert (a["case"], a["repeat"]) == (b["case"], b["repeat"]) == (c["case"], c["repeat"])
            key = "coexist" if b["case"] == "coexist" else b["category"]
            field = "aggregate_wall_tps" if key == "coexist" else "decode_tps"
            ratio = b[field] / statistics.mean([a[field], c[field]])
            grouped.setdefault(key, []).append(ratio)
        comparisons.append({"phase": phase, "bracketed_tps_ratios": {
            key: {"paired_median": statistics.median(values), "pairs": values}
            for key, values in grouped.items()}})
    native = {}
    for phase in phases:
        trace = directory / f"{phase}-rank-0.log"
        if not trace.exists():
            continue
        grouped = {}
        pattern = (r"MTP_ROUND slot=(\d+) first=\d+ k=(\d+) accepted=(\d+) "
                   r"emitted=(\d+) consumed=(\d+) ms=([0-9.]+)")
        for slot, k, accepted, emitted, consumed, milliseconds in re.findall(
                pattern, trace.read_text(encoding="utf-8")):
            k, accepted = int(k), int(accepted)
            key = f"slot{slot}-k{k}"
            group = grouped.setdefault(key, {"rounds": 0, "accepted": 0, "drafts": 0,
                                            "emitted": 0, "consumed": 0, "milliseconds": 0.0,
                                            "conditional_successes": [0] * k,
                                            "conditional_trials": [0] * k})
            assert 0 <= accepted <= k <= 3
            group["rounds"] += 1
            group["accepted"] += accepted
            group["drafts"] += k
            group["emitted"] += int(emitted)
            group["consumed"] += int(consumed)
            group["milliseconds"] += float(milliseconds)
            for position in range(min(accepted + 1, k)):
                group["conditional_trials"][position] += 1
                group["conditional_successes"][position] += int(position < accepted)
        for group in grouped.values():
            group["emitted_per_pass"] = group["emitted"] / group["rounds"]
            group["mean_round_ms"] = group["milliseconds"] / group["rounds"]
        native[phase] = grouped
    result = {"instrumentation": "stock recipe plus existing FG_RING_TRACE request token-ID hashes",
              "qualification": ("single-session " if single else "") +
                               "short-context serving only; no promotion or long-context claim",
              "comparisons": comparisons, "native_rounds_all_phase_requests": native}
    (directory / "serving-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--base", default="http://192.168.42.42:8080")
    run.add_argument("--phase", required=True)
    run.add_argument("--suite", choices=("greedy", "perf", "stream"), required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--single-session", action="store_true")
    run.add_argument("--diagnostic", action="store_true")
    check = commands.add_parser("verify")
    for name in ("candidate", "baseline", "trace", "baseline-trace"):
        check.add_argument("--" + name, required=True)
    check.add_argument("--diagnostic", action="store_true")
    check.add_argument("--report-output")
    check.add_argument("--correctness-events", help="receipt-complete pass-local choices and same-prefix top2 comparisons")
    check.add_argument("--quality", help="full-vocabulary original-Q8 model-quality result")
    summary = commands.add_parser("summarize")
    summary.add_argument("--directory", required=True)
    args = parser.parse_args()
    {"run": run_suite, "verify": verify, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
