"""Summarize offline replay; CPU-hook wall is never a GPU-time estimate."""
import argparse
import gzip
import hashlib
import json
import random
import re
import statistics
import tarfile
from pathlib import Path


def samples(text):
    result = []
    for line in re.findall(r"HOST_GROUP_BENCH ([^\r\n]+)", text):
        fields = dict(word.split("=", 1) for word in line.split())
        assert fields.pop("parity") == "PASS"
        row = {key: float(value) if "." in value else int(value)
               for key, value in fields.items()}
        assert row["submits_per_token"] == (7, 1, 2)[row["grouped"]]
        result.append(row)
    return result


def interval_union(intervals):
    total = 0
    last = 0
    for begin, end in sorted(intervals):
        total += max(0, end - max(begin, last))
        last = max(last, end)
    return total


def paired(base, candidate):
    rounds = sorted(base)
    assert rounds == sorted(candidate) and len(rounds) >= 3
    rng = random.Random(1729)
    values = sorted(
        statistics.mean(base[i] / candidate[i] for i in rng.choices(rounds, k=len(rounds)))
        for _ in range(50000)
    )
    lower, upper = values[1250], values[48749]
    return {
        "mean_speedup": statistics.mean(base[i] / candidate[i] for i in rounds),
        "paired_round_bootstrap_95pct": [lower, upper],
        "noninferior_lower_ge_0_97": lower >= 0.97,
        "win_lower_gt_1": lower > 1,
        "scope": "synthetic warm replay only; not a production acceptance gate",
    }


def analyze(native, archive):
    devices = {}
    records = json.loads(native.read_text(encoding="utf-8-sig"))
    for record in records if isinstance(records, list) else [records]:
        assert record["ExitCode"] == 0
        rows = samples(record["Stdout"] + "\n" + record["Stderr"])
        assert len(rows) == 15
        modes = {}
        for mode in range(3):
            selected = [row for row in rows if row["grouped"] == mode]
            assert len(selected) == 5 and all(row["tokens"] == 128 for row in selected)
            modes[mode] = {row["round"]: row["wall_ms"] for row in selected}
        devices[record["IP"]] = {
            "mean_ms_per_synthetic_block": {
                mode: statistics.mean(values.values()) for mode, values in modes.items()
            },
            "one_submit_vs_baseline": paired(modes[0], modes[1]),
            "early_publish_vs_baseline": paired(modes[0], modes[2]),
            "samples": rows,
        }
    with tarfile.open(archive, "r:gz") as contents:
        trace = gzip.decompress(contents.extractfile("cpu-wall.log.gz").read()).decode()
        binary_hashes = contents.extractfile("binary-sha256.txt").read().decode()
        kernel = contents.extractfile("kernel.txt").read().decode().strip()
    windows = samples(trace)
    assert len(windows) == 15
    events = [
        (op, int(begin), int(end), obj, int(status))
        for op, begin, end, obj, status in re.findall(
            r"VKWALL op=(\w+) begin_ns=(\d+) end_ns=(\d+) object=([0-9a-f]+) result=(-?\d+)",
            trace,
        )
    ]
    assert events and all(event[-1] == 0 for event in events)
    ledgers = []
    for window in windows:
        totals = dict.fromkeys(("submit", "wait", "reset"), 0)
        counts = dict.fromkeys(totals, 0)
        open_commands, recording = {}, []
        for op, begin, end, obj, _ in events:
            if begin < window["begin_ns"] or end > window["end_ns"]:
                continue
            if op in totals:
                totals[op] += end - begin
                counts[op] += 1
            elif op == "begin":
                assert obj not in open_commands
                open_commands[obj] = begin
            elif op == "end":
                recording.append((open_commands.pop(obj), end))
        assert not open_commands
        scale = window["tokens"] * 1e6
        ledger = {op + "_ms": value / scale for op, value in totals.items()}
        ledger["recording_ms"] = interval_union(recording) / scale
        ledger["wall_ms"] = (window["end_ns"] - window["begin_ns"]) / scale
        ledger["other_ms"] = ledger["wall_ms"] - sum(
            ledger[key] for key in ("submit_ms", "wait_ms", "reset_ms", "recording_ms")
        )
        assert ledger["other_ms"] >= 0
        assert counts["submit"] / window["tokens"] == window["submits_per_token"]
        ledger["host_outside_fence_ms"] = ledger["wall_ms"] - ledger["wait_ms"]
        ledger.update({op + "_calls": count / window["tokens"] for op, count in counts.items()})
        ledger.update(mode=window["grouped"], round=window["round"])
        ledgers.append(ledger)
    means = {
        mode: {
            key: statistics.mean(row[key] for row in ledgers if row["mode"] == mode)
            for key in ledgers[0] if key not in ("mode", "round")
        } for mode in range(3)
    }
    return {
        "native": devices, "cpu_hook_means": means, "cpu_hook_samples": ledgers,
        "binary_hashes": binary_hashes, "kernel": kernel,
        "trace_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "limitations": [
            "No production access, pack, model-level greedy hashes or fleet TPS measurements.",
            "Unchanged arithmetic kernels; synthetic outputs compared byte-for-byte.",
            "CPU-only preload hook perturbs recording and scheduling; native timing is separate.",
            "Fence-call blocking is not measured GPU busy time.",
            "vkResetFences is unhooked and included in residual other_ms.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.native, args.trace), indent=2) + "\n")
