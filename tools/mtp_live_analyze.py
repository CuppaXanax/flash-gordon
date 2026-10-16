"""Summarize diagnostic off/N4/off receipts without claiming qualification."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import tarfile

from mtp_diagnose import parse_trace, validate_stage_ledgers


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(samples=len(values), mean=st.mean(values), median=st.median(values),
                p95=values[math.ceil(.95*len(values))-1], maximum=max(values))


def fields(line):
    return {key: value for key, value in
            re.findall(r"(\w+)=([^\s]+)", line)}


def transport_summary(transport_receipt):
    receipt = json.loads(transport_receipt.read_text(encoding="utf-8-sig"))
    transport = [json.loads(line) for r in receipt for line in r["Stdout"].splitlines()
                 if line.startswith("{")]
    assert len(transport) == 40 and all(r["validated"] for r in transport)
    wire = {}
    for n in range(1,5):
        ordinary = [r["rtt_mean_us"]/2000 for r in transport if r["n"]==n and not r["initial"]]
        initial = [r["rtt_mean_us"]/2000 for r in transport if r["n"]==n and r["initial"]]
        assert len(ordinary) == len(initial) == 5
        models = [7*a+b for a,b in zip(ordinary,initial)]
        wire[str(n)] = dict(ordinary_rtt_half_ms=stats(ordinary), initial_rtt_half_ms=stats(initial),
            eight_work_hop_pair_transfer_model_ms=st.mean(models),
            model_ci95_ms=2.7764451051977987*st.stdev(models)/math.sqrt(5),
            ordinary_frame_bytes=next(r["frame_bytes"] for r in transport if r["n"]==n and not r["initial"]),
            initial_frame_bytes=next(r["frame_bytes"] for r in transport if r["n"]==n and r["initial"]))
    return wire


def analyze_incomplete(run, transport_receipt):
    outcome = json.loads((run/"outcome.json").read_text(encoding="utf-8-sig"))
    assert outcome["productionVerified"] and outcome["scratchRemoved"], outcome
    assert not outcome["complete"] and outcome["diagnosticOnly"], outcome
    rows = json.loads((run/"A0-perf.json").read_text())["rows"]
    ranks = {}
    for rank in range(8):
        with tarfile.open(run/f"evidence-192.168.42.{rank+42}.tar.gz") as archive:
            log = archive.extractfile(f"serving-F3-rank-{rank}.log").read().decode()
        ranks[str(rank)] = dict(
            verify=[fields(line) for line in log.splitlines() if line.startswith("MTP_VERIFY_RANK ")],
            vocab=[fields(line) for line in log.splitlines() if line.startswith("MTP_VOCAB_STAGE ")],
            native=[fields(line) for line in log.splitlines() if line.startswith("MTP_LOCAL_CHAIN ")],
            memory_lines=[line for line in log.splitlines() if "Vulkan memory:" in line],
            completed_rounds=sum(line.startswith("MTP_ROUND ") for line in log.splitlines()),
            tail=log.splitlines()[-8:])
    assert all(len(r["verify"]) == 1 and r["completed_rounds"] == 0 for r in ranks.values())
    verify = [r["verify"][0] for r in ranks.values()]
    assert all((r["first"],r["n"],r["status"]) ==
               (verify[0]["first"],"4","0") for r in verify)
    return dict(
        scope="Incomplete diagnostic: one cold N4 pass, zero completed MTP rounds; no on/off comparison",
        off={c:dict(requests=len(v),decode_tps=stats([r["decode_tps"] for r in v]),
                    wall_tps=stats([r["wall_tps"] for r in v]))
             for c in ("chat","code") if (v:=[r for r in rows if r["category"] == c])},
        mtp_decode_tps=None, live_acceptance=None, live_round_ms=None,
        cold_verify_compute_sum_ms=sum(float(r["compute_ms"]) for r in verify),
        cold_verify_service_sum_ms=sum(float(r[k]) for r in verify
                                      for k in ("input_ms","compute_ms","read_ms")),
        ranks=ranks, direct_codec_crc_tcp_pair=transport_summary(transport_receipt),
        transport_scope="Measured actual application-frame codec/CRC TCP RTT/2 on .48/.49; pair-to-eight-hop transfer remains a model, not synchronized one-way or wire-only latency",
        gates=dict(full_same_prefix_divergence_capture=False, original_q8_n4_quality=False,
                   promotion=False), outcome=outcome)


def analyze(run, transport_receipt):
    outcome = json.loads((run/"outcome.json").read_text(encoding="utf-8-sig"))
    assert outcome["productionVerified"] and outcome["scratchRemoved"], outcome
    assert outcome["complete"] and outcome["diagnosticOnly"], outcome
    rows = {p: json.loads((run/f"{p}-perf.json").read_text())["rows"]
            for p in ("A0", "F3", "A1")}
    text = (run/"F3-rank-0.log").read_text()
    requests = parse_trace(text)
    ledgers, pending = {}, []
    for line in text.splitlines():
        if line.startswith("MTP_STAGE "):
            pending.append(fields(line))
        match = re.search(r"API_TOKEN_HASH id=(\S+)", line)
        if match:
            ledgers[match[1]], pending = pending, []
    ranks = {}
    for rank in range(8):
        with tarfile.open(run/f"evidence-192.168.42.{rank+42}.tar.gz") as archive:
            log = archive.extractfile(f"serving-F3-rank-{rank}.log").read().decode()
        ranks[rank] = [fields(line) for line in log.splitlines()
                       if line.startswith("MTP_VERIFY_RANK ")]
        assert all(r["status"] == "0" for r in ranks[rank])
        validate_stage_ledgers(log)
    all_stages = [fields(line) for line in text.splitlines()
                  if line.startswith("MTP_STAGE ")]
    owner_rounds = collections.defaultdict(list)
    for index, stage in enumerate(all_stages):
        for rank in range(8):
            assert index < len(ranks[rank]), (index, rank)
            row = ranks[rank][index]
            assert (row["first"], int(row["n"])) == (
                stage["first"], int(stage["k"])+1), (index, rank, row, stage)
            owner_rounds[id(stage)].append(row)
    # Reassociate parsed ledger objects by their global order, not first-token
    # numbers: prompts repeat positions across requests.
    cursor = 0
    owners_by_request = {}
    for line in text.splitlines():
        if line.startswith("MTP_STAGE "):
            cursor += 1
        match = re.search(r"API_TOKEN_HASH id=(\S+)", line)
        if match:
            count = len(ledgers[match[1]])
            owners_by_request[match[1]] = [
                owner_rounds[id(s)] for s in all_stages[cursor-count:cursor]]
    comparisons = {}
    before = {(r["case"], r["repeat"]): r for r in rows["A0"]}
    after = {(r["case"], r["repeat"]): r for r in rows["A1"]}
    stage_keys = ("draft_ms", "prepare_ms", "verify_ms", "target_repair_ms",
                  "native_repair_ms", "commit_ms", "host_ms", "overlap_ms", "total_ms")
    for category in ("chat", "code"):
        candidates = [r for r in rows["F3"] if r["category"] == category]
        off = [st.mean([before[r["case"], r["repeat"]]["decode_tps"],
                        after[r["case"], r["repeat"]]["decode_tps"]]) for r in candidates]
        rounds = [v for r in candidates for v in requests[r["id"]]["rounds"]]
        stages = [v for r in candidates for v in ledgers[r["id"]]]
        owner_groups = [v for r in candidates for v in owners_by_request[r["id"]]]
        assert len(rounds) == len(stages) == len(owner_groups)
        n4_groups = [g for g in owner_groups if all(int(x["n"]) == 4 for x in g)]
        n4_stages = [s for s, g in zip(stages, owner_groups) if int(g[0]["n"]) == 4]
        owner_service = [sum(float(x[k]) for x in g for k in
                             ("input_ms", "compute_ms", "read_ms")) for g in n4_groups]
        comparisons[category] = dict(requests=len(candidates),
            off_decode_tps=st.median(off),
            mtp_decode_tps=st.median(r["decode_tps"] for r in candidates),
            paired_ratio=st.median(r["decode_tps"]/v for r,v in zip(candidates,off)),
            mtp_wall_tps=stats([r["wall_tps"] for r in candidates]),
            tokens_per_verify=sum(r["emitted"] for r in rounds)/len(rounds),
            accepted_fraction=sum(r["accepted"] for r in rounds)/sum(r["k"] for r in rounds),
            round_ms=stats([r["milliseconds"] for r in rounds]),
            keep_counts=dict(collections.Counter(r["consumed"] for r in rounds)),
            stages_ms={k: stats([float(s[k]) for s in stages]) for k in stage_keys},
            n4_owner_compute_sum_ms=stats([sum(float(x["compute_ms"]) for x in g)
                                          for g in n4_groups]),
            n4_owner_service_sum_ms=stats(owner_service),
            n4_transport_head_host_residual_ms=stats([
                float(s["verify_ms"])-service for s, service in zip(n4_stages, owner_service)]),
            n4_owner_compute_ms={str(rank): stats([float(g[rank]["compute_ms"])
                                                 for g in n4_groups]) for rank in range(8)})
    choices = re.findall(r"MTP_CHOICE .*?emitted=(\d+) target_argmax=(\d+)",text)
    assert choices and all(a == b for a,b in choices)
    return dict(scope="Diagnostic short-context single-session, no promotion",
        comparisons=comparisons, direct_codec_crc_tcp_pair=transport_summary(transport_receipt),
        transport_scope="Actual N1..4 protocol codec/CRC and two-send TCP RTT/2 on .48/.49; eight-hop value remains a pair-to-ring transfer model, not wire-only or synchronized one-way timing",
        live_residual_scope="Actual verify wall minus same-round owner input/compute/read sums; includes head, transport, host and overlap, not isolated wire latency",
        validation=dict(stage_ledgers=validate_stage_ledgers(text),
                        pass_local_argmax_checks=len(choices), owner_round_alignment=True),
        gates=dict(full_same_prefix_divergence_capture=False, original_q8_n4_quality=False,
                   promotion=False), outcome=outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--transport-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Report the first cold failed pass, never infer MTP TPS or acceptance")
    args = parser.parse_args()
    report = (analyze_incomplete if args.allow_incomplete else analyze)(
        args.run,args.transport_receipt)
    report["inputs_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.run.iterdir() if p.is_file() and p.suffix in (".json",".log",".gz")}
    report["transport_receipt_sha256"] = hashlib.sha256(args.transport_receipt.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report.get("comparisons",report.get("off")),indent=2))


if __name__ == "__main__":
    main()
