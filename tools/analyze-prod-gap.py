"""Descriptive production ledger; token observations are not independent trials."""
import argparse
import gzip
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p95": values[min(len(values)-1, int(len(values)*.95))]}


def union_ms(intervals):
    end = None
    total = 0
    for begin, finish in sorted(intervals):
        if end is None or begin > end:
            total += finish-begin
        elif finish > end:
            total += finish-end
        end = finish if end is None else max(end, finish)
    return total / 1e6


def cpu_calls(calls):
    if not calls:
        return {"counts": {}}
    totals = defaultdict(float)
    counts = defaultdict(int)
    intervals = []
    recording = {}
    record_spans = []
    for call in calls:
        begin, end = int(call["begin_ns"]), int(call["end_ns"])
        assert end >= begin and int(call["result"]) == 0
        op = call["op"]
        totals[op+"_ms"] += (end-begin)/1e6
        counts[op] += 1
        intervals.append((begin, end))
        if op == "begin":
            recording[call["object"]] = begin
        elif op == "end" and call["object"] in recording:
            record_spans.append((recording.pop(call["object"]), end))
    totals["record_span_ms"] = sum(b-a for a, b in record_spans)/1e6
    totals["accounted_union_ms"] = union_ms(intervals + record_spans)
    totals["call_window_ms"] = (int(calls[-1]["end_ns"])-int(calls[0]["begin_ns"]))/1e6
    before = [int(c.get("gpu_before_ns", 0)) for c in calls if c["op"] == "submit"]
    after = [int(c.get("gpu_after_ns", 0)) for c in calls if c["op"] == "wait"]
    if before and after and 0 < before[0] < 2**64-1 and 0 < after[-1] < 2**64-1:
        assert before[0] <= after[-1]
        totals["gpu_engine_ms"] = (after[-1]-before[0])/1e6
    return {**totals, "counts": dict(counts)}


def parse_lines(lines, rank):
    rings, blocks, requests, heads = [], [], [], []
    pending_request = []
    active = None
    calls, own_calls, hc_calls = [], [], []
    receive, send, own_receive, initial_send = None, None, None, None
    head_calls, collect_head = [], False
    for line in lines:
        if line.startswith("VKWALL "):
            if active == "block":
                calls.append(fields(line))
            elif collect_head:
                head_calls.append(fields(line))
            continue
        if line.startswith('{"schema":"flash-gordon.profile"'):
            event = json.loads(line)
            if event.get("record_type") != "fabric_service":
                continue
            if event["message_type"] == 44:
                if event["direction"] == "receive":
                    active = "block"
                    calls, hc_calls, send = [], [], None
                    head_calls, collect_head = [], False
                    receive = event
                elif active == "block":
                    send = event
                    if rank == 0:
                        own_calls, own_receive = calls, receive
                        calls = []
                        active = None
                        head_calls, collect_head = [], True
                elif rank == 0 and event["direction"] == "send":
                    initial_send = event
            continue
        if line.startswith("OUTPUT_SPLIT_STATE "):
            data = fields(line)
            if rank == 7 and data["what"] == "hc":
                begins = [i for i, c in enumerate(calls) if c["op"] == "begin"]
                if begins:
                    hc_calls = calls[begins[-1]:]
            elif (rank in (0, 1, 2) and data["what"] == "slice") or (
                    rank == 4 and data["what"] == "local"):
                assert active is None and collect_head
                heads.append({"token": int(data["token"]), "cpu": cpu_calls(head_calls)})
                head_calls, collect_head = [], False
        elif line.startswith("RING_DECODE_BLOCK "):
            data = fields(line)
            blocks.append({"native": data, "cpu": cpu_calls(calls),
                           "hc_cpu": cpu_calls(hc_calls), "receive": receive, "send": send})
            calls, hc_calls, active = [], [], None
            head_calls, collect_head = [], rank in (1, 2, 4)
        elif line.startswith("RING_DECODE "):
            data = fields(line)
            row = {"native": data, "cpu": cpu_calls(own_calls),
                   "receive": own_receive, "send": send, "initial_send": initial_send}
            rings.append(row)
            pending_request.append(row)
            own_calls, own_receive, send, initial_send = [], None, None, None
        elif line.startswith("GENERATE_TRACE "):
            data = fields(line)
            assert len(pending_request) == int(data["tokens"]), (
                len(pending_request), data["tokens"])
            requests.append({"native": data, "rings": pending_request})
            pending_request = []
    assert not pending_request
    return {"rings": rings, "blocks": blocks, "requests": requests, "heads": heads}


def header(record, name):
    return float(next(v for k, v in record["headers"].items() if k.lower() == name.lower()))


def replay_round(wall_us, layers_us):
    gdn = statistics.mean(layers_us[i] for i in (0, 2, 3, 4))
    qsa = statistics.mean(layers_us[i] for i in (1, 5))
    remainder = wall_us-sum(layers_us)
    weighted = 36*gdn+12*qsa
    return {"gdn_layer_us": gdn, "qsa_layer_us": qsa,
            "type_weighted_layer_ms": weighted/1000,
            "unassigned_replay_remainder_per_block_us": remainder,
            "type_weighted_wall_reference_ms": (weighted+8*remainder)/1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    base = args.root / "results" / "hillclimb"
    directory = base / "prod-gap"
    result = {"inferential": False, "sanity": {}, "phases": {}, "traces": {},
              "gpu_timestamp_queries": False,
              "gpu_accounting": "Read-only per-client DRM engine busy counters. Whole-request snapshots include cached-prefix prefill and heads; engine-phase submit/wait brackets isolate block jobs. Not hardware per-kernel timestamps."}
    reference = directory/"references"/"roofline-summary.json"
    roofline = read(reference)["24"]
    result["replay_reference"] = {
        "source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "caveat": "Synthetic Q4, fixed 2000 MHz, short context. Candidate is unmerged and model-quality unqualified. Wall-minus-layer remainder is not pure host time."}
    for name in ("baseline", "candidate"):
        rounds = [replay_round(r[name+"_wall_us"], r[name+"_layers_us"]) for r in roofline["rounds"]]
        result["replay_reference"][name] = {
            "flags": roofline[name+"_flags"], "rounds": rounds,
            "means": {k: statistics.mean(r[k] for r in rounds) for k in rounds[0]}}
    historical = {r["metric"]: r for r in read(base/"ab-mix"/"ab-result.json")["metrics"]}
    sanity = read(directory/"sanity.json")["rows"]
    for context in (4096, 32768):
        rows = [r for r in sanity if r["context"] == context]
        out = {"actual_completion_tokens": [r["decode"]["usage"]["completion_tokens"] for r in rows]}
        for kind in ("engine", "client"):
            values = [header(r["decode"], "X-Flash-Gordon-Decode-TPS") if kind == "engine"
                      else r["decode"]["client_wall_tps"] for r in rows]
            reference = historical[f"decode-{context}.{kind}_tps"]["A_median"]
            out[kind] = {"current": summary(values), "historical_original_median": reference,
                         "ratio_of_medians": statistics.median(values)/reference}
        if context == 4096:
            values = [header(r["cold"], "X-Flash-Gordon-Prefill-TPS") for r in rows]
            reference = historical["prefill-4096.engine_tps"]["A_median"]
            out["prefill_engine"] = {"current": summary(values),
                "historical_original_median": reference,
                "ratio_of_medians": statistics.median(values)/reference}
        result["sanity"][str(context)] = out
    expected_hashes, expected_requests = {}, {}
    phases = ["normal", "trace", "wall", "post"]
    if (directory/"engine").exists():
        phases.append("engine")
    for phase in phases:
        cases = []
        for path in sorted((directory/phase).glob("*-decode.json")):
            case = int(path.name.split("-")[0])
            row = read(path)["decode"]
            prime = read(directory/phase/f"{case}-prime.json")["rows"][0]
            if phase == "normal":
                expected_hashes[case] = row["sha256"]
                expected_requests[case] = (row["request_sha256"],
                    prime["cold"]["request_sha256"], prime["cold"]["sha256"])
            assert row["sha256"] == expected_hashes[case], f"{phase}/{case} changed greedy output"
            assert (row["request_sha256"], prime["cold"]["request_sha256"],
                    prime["cold"]["sha256"]) == expected_requests[case], f"{phase}/{case} changed workload"
            snapshots = {}
            before = {json.loads(r["Stdout"])["rank"]: json.loads(r["Stdout"])
                      for r in read(directory/f"{phase}-{case}-before.json")}
            after = {json.loads(r["Stdout"])["rank"]: json.loads(r["Stdout"])
                     for r in read(directory/f"{phase}-{case}-after.json")}
            count = row["usage"]["completion_tokens"]
            assert count == 128 and prime["cold"]["usage"]["completion_tokens"] == 1
            for rank in range(8):
                a, b = before[rank], after[rank]
                assert a["pid"] == b["pid"]
                assert b["gpu_engine_ns"] >= a["gpu_engine_ns"]
                scale = 1000/a["clock_ticks_per_second"]/count
                snapshots[str(rank)] = {
                    "gpu_engine_ms_per_output_token": (b["gpu_engine_ns"]-a["gpu_engine_ns"])/1e6/count,
                    "cpu_user_ms_per_output_token": (b["cpu_user_ticks"]-a["cpu_user_ticks"])*scale,
                    "cpu_system_ms_per_output_token": (b["cpu_system_ticks"]-a["cpu_system_ticks"])*scale,
                }
            cases.append({"case": case, "context": prime["context"], "completion_tokens": count,
                "prefilled_tokens": header(row, "X-Flash-Gordon-Prefilled-Tokens"),
                "engine_tps": header(row, "X-Flash-Gordon-Decode-TPS"),
                "client_wall_tps": row["client_wall_tps"], "sha256": row["sha256"],
                "request_sha256": row["request_sha256"], "matched_requests": True,
                "rank_accounting": snapshots})
        result["phases"][phase] = cases
    reference_tokens = None
    for phase in [p for p in phases if p in ("trace", "wall", "engine")]:
        parsed = {}
        for rank in range(8):
            path = directory/"logs"/f"{phase}-192.168.42.{42+rank}.log.gz"
            with gzip.open(path, "rt") as source:
                parsed[rank] = parse_lines(source, rank)
        master = parsed[0]
        tokens = [int(r["native"]["token"]) for r in master["rings"]]
        if reference_tokens is None:
            reference_tokens = tokens
        assert tokens == reference_tokens
        assert len(master["requests"]) == 8
        for rank in range(1, 8):
            assert [int(r["native"]["token"]) for r in parsed[rank]["blocks"]] == tokens
        for rank in (0, 1, 2, 4):
            assert [r["token"] for r in parsed[rank]["heads"]] == tokens
        case_summaries = []
        offset = 0
        for request_index, request in enumerate(master["requests"]):
            n = len(request["rings"])
            if request_index % 2 == 0:
                offset += n
                continue
            case = request_index//2
            selected = list(range(offset+min(8, n), offset+n))
            assert len(selected) >= 24, "Insufficient sustained trace"
            ranks = {}
            for rank in range(8):
                rows = [parsed[rank]["blocks"][i] if rank else master["rings"][i] for i in selected]
                value = {"layers": rows[0]["native"].get("layers", rows[0]["native"].get("own_layers"))}
                names = ("write_ms", "block_ms", "read_ms", "publish_ms", "egress_ms", "total_ms") if rank else (
                    "own_ms", "own_run_ms", "own_read_ms")
                for name in names:
                    value[name] = summary([float(r["native"][name]) for r in rows])
                for name in ("wait_ms", "submit_ms", "record_span_ms", "reset_ms", "accounted_union_ms", "gpu_engine_ms"):
                    value["vk_"+name] = summary([r["cpu"][name] for r in rows if name in r["cpu"]])
                value["vk_counts_per_token"] = {
                    name: statistics.mean(r["cpu"]["counts"].get(name, 0) for r in rows)
                    for name in ("wait", "submit", "begin", "end", "reset")} if any(r["cpu"]["counts"] for r in rows) else None
                native_total = "total_ms" if rank else "own_ms"
                value["cpu_other_wall_ms"] = summary([
                    float(r["native"][native_total])-r["cpu"]["accounted_union_ms"]
                    for r in rows if "accounted_union_ms" in r["cpu"]])
                if rank == 7:
                    for name in ("wait_ms", "submit_ms", "record_span_ms", "gpu_engine_ms"):
                        value["hc_vk_"+name] = summary([r["hc_cpu"][name] for r in rows if name in r["hc_cpu"]])
                for name in ("wall_ms", "header_ms", "payload_ms", "validation_ms", "wait_ms"):
                    value["receive_"+name] = summary([r["receive"][name] for r in rows if r["receive"]])
                value["send_wall_ms"] = summary([r["send"]["wall_ms"] for r in rows if r.get("send")])
                if rank in (0, 1, 2, 4):
                    for name in ("wait_ms", "submit_ms", "record_span_ms", "call_window_ms", "gpu_engine_ms"):
                        value["head_vk_"+name] = summary([
                            parsed[rank]["heads"][i]["cpu"][name] for i in selected
                            if name in parsed[rank]["heads"][i]["cpu"]])
                ranks[str(rank)] = value
            cycles, services, residuals, grams, embeddings, receives, sends = [], [], [], [], [], [], []
            for i in selected:
                co = master["rings"][i]["native"]
                cycle = float(co["total_ms"])
                service = float(co["own_ms"]) + sum(
                    float(parsed[r]["blocks"][i]["native"]["total_ms"]) for r in range(1, 8))
                prep = float(co["embed_ms"])
                cycles.append(cycle)
                services.append(service)
                residuals.append(cycle-prep-service)
                grams.append(float(co["ngram_ms"]))
                embeddings.append(prep-float(co["ngram_ms"])-float(co["send_ms"]))
                rowset = [master["rings"][i]] + [parsed[r]["blocks"][i] for r in range(1, 8)]
                receives.append(sum(r["receive"]["wall_ms"] for r in rowset if r["receive"]))
                sends.append(sum(r["send"]["wall_ms"] for r in rowset if r.get("send"))
                             + master["rings"][i]["initial_send"]["wall_ms"])
            case_summaries.append({"case": case, "context": result["phases"][phase][case]["context"],
                "kept_tokens": len(selected), "discarded_initial_tokens": min(8, n),
                "cycle_ms": summary(cycles), "rank_service_sum_ms": summary(services),
                "residual_after_prep_and_services_ms": summary(residuals),
                "ngram_ms": summary(grams), "embedding_call_ms": summary(embeddings),
                "work_receive_wall_sum_ms": summary(receives),
                "work_send_wall_sum_ms": summary(sends),
                "max_head_vk_window_ms": summary([
                    max(parsed[r]["heads"][i]["cpu"]["call_window_ms"] for r in (0, 1, 2, 4))
                    for i in selected if all("call_window_ms" in parsed[r]["heads"][i]["cpu"]
                                             for r in (0, 1, 2, 4))]),
                "coordinator_submissions_per_token": summary([
                    int(master["rings"][i]["native"]["submissions_total"])
                    - int(master["rings"][i-1]["native"]["submissions_total"]) for i in selected]),
                "coordinator_recorded_dispatches_per_token": summary([
                    int(master["rings"][i]["native"]["dispatches_total"])
                    - int(master["rings"][i-1]["native"]["dispatches_total"]) for i in selected]),
                "generate": request["native"], "ranks": ranks})
            offset += n
        result["traces"][phase] = case_summaries
        if phase != "trace":
            for case, control in zip(case_summaries, result["traces"]["trace"]):
                for name in ("coordinator_submissions_per_token", "coordinator_recorded_dispatches_per_token"):
                    assert case[name] == control[name], f"{phase} changed {name}"
        if phase == "engine":
            for case, control in zip(case_summaries, result["traces"]["wall"]):
                for rank in range(8):
                    assert case["ranks"][str(rank)]["vk_counts_per_token"] == control["ranks"][str(rank)]["vk_counts_per_token"]
    (base/"PROD_GAP.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
    print(json.dumps(result["sanity"], indent=2))
    for phase, cases in result["traces"].items():
        for case in cases:
            print(phase, case["case"], case["context"], "cycle", case["cycle_ms"]["mean"],
                  "services", case["rank_service_sum_ms"]["mean"],
                  "ngram", case["ngram_ms"]["mean"],
                  "residual", case["residual_after_prep_and_services_ms"]["mean"])


if __name__ == "__main__":
    main()
