#!/usr/bin/env python3
"""Join isolated cross-runs to original full-ring captures without mixing cohorts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

from reconcile_quality import compare


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(destination, filter="data")


def run(root, q4, stack, previous):
    unpack(root / "evidence.tar.gz", root / "raw")
    unpack(root / "logits.tar.gz", root / "raw")
    raw = root / "raw"
    corpus = json.loads((raw / "corpus.json").read_text(encoding="utf-8"))
    positions = json.loads((raw / "evidence" / "positions.json").read_text())
    assert len(corpus) == len(positions) == 560
    for filename, offset, count in (("r3-teacher-native-positions.json", 0, 80),
                                    ("teacher480r2-native-positions.json", 80, 480)):
        historical = json.loads((root / filename).read_text())
        assert len(historical) == count
        for index, (old_index, tokens, _) in enumerate(historical):
            assert int(old_index) == index
            assert int(tokens) == positions[offset + index]["tokens"]
    policies = [f"{w}-{m}" for w in ("mixed", "q8") for m in ("decode", "prefill", "legacy")]
    stages = []
    for policy in policies:
        for first in range(0, 48, 6):
            path = raw / "evidence" / f"{policy}-stage-{first}"
            receipts = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
            receipt = receipts[-1]
            assert receipt["kind"] == "stage_complete"
            assert receipt["first"] == first and receipt["last"] == first + 5
            assert receipt["rounded_boundaries"] == 0 and receipt["prefix_max_relative"] == 0
            method = "decode" if policy.endswith("-decode") else "prefill"
            assert receipt["arithmetic_path"] == method
            assert receipt["microbatch"] == (2 if method == "decode" else 128)
            hashes = [line.split()[0] for line in path.with_suffix(".sha256").read_text().splitlines()]
            assert len(hashes) == 2 and hashes[0] == hashes[1], path
            stages.append(dict(policy=policy, **receipt, independent_slot_frames_sha256=hashes[0]))
    restored = []
    for rank in range(8):
        path = raw / "evidence" / f"q8-rank-{rank}-restored.jsonl"
        receipt = json.loads(path.read_text().splitlines()[-1])
        assert receipt["kind"] == "q8_rank_restored" and receipt["rank"] == rank
        assert receipt["trained_spans_sha_exact"]
        restored.append(receipt)
    assert sum(row["spans"] for row in restored) == 1223
    captures = {}
    repeats = {}
    for policy in policies:
        captures[policy] = {}
        for cohort, count in (("forced", 80), ("natural", 480)):
            items = [(i, key[len(cohort) + 1:], text) for i, (key, text) in enumerate(corpus)
                     if key.startswith(cohort + "-")]
            assert len(items) == count
            target = root / "captures" / policy / cohort
            target.mkdir(parents=True, exist_ok=True)
            for i, key, text in items:
                assert positions[i]["id"] == f"{cohort}-{key}"
                assert positions[i]["final_position"] == positions[i]["tokens"] - 1
                assert positions[i]["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
                source = raw / "logits" / policy / f"{i:03d}.fgl"
                output = target / f"{key}.fgl"
                if not output.exists():
                    os.link(source, output)
                elif not os.path.samefile(source, output):
                    raise ValueError(f"stale capture link: {output}")
            (target / "corpus.json").write_text(
                json.dumps([[k, t] for _, k, t in items], ensure_ascii=False) + "\n", encoding="utf-8")
            captures[policy][cohort] = target
        by_text = {}
        agreements = []
        for i, (_, text) in enumerate(corpus):
            sha = hashlib.sha256((raw / "logits" / policy / f"{i:03d}.fgl").read_bytes()).hexdigest()
            if text in by_text:
                agreements.append(sha == by_text[text])
            by_text[text] = sha
        repeats[policy] = dict(repeated_frontiers=len(agreements),
                               byte_exact=sum(agreements), all_exact=all(agreements))
    result = dict(samples=560, historical_native_token_position_agreements=560,
                  repeated_context_reset_controls=repeats,
                  independent_slot_stage_controls=stages, original_q8_restoration=restored,
                  comparisons={})

    def record(name, reference, candidate):
        result["comparisons"][name] = compare(reference, candidate, candidate / "corpus.json")
        print(name, {k: v for k, v in result["comparisons"][name].items()
                     if k not in ("rows", "reference", "candidate")}, flush=True)

    for cohort in ("forced", "natural"):
        for method in ("decode", "prefill", "legacy"):
            record(f"{cohort}-cumulative-{method}", captures[f"q8-{method}"][cohort],
                   captures[f"mixed-{method}"][cohort])
        for weights in ("q8", "mixed"):
            record(f"{cohort}-{weights}-prefill-vs-decode", captures[f"{weights}-prefill"][cohort],
                   captures[f"{weights}-decode"][cohort])
            record(f"{cohort}-{weights}-legacy-vs-prefill", captures[f"{weights}-legacy"][cohort],
                   captures[f"{weights}-prefill"][cohort])
    record("natural-originalQ8-vs-fresh-legacyQ8", q4 / "teacher480r2",
           captures["q8-legacy"]["natural"])
    record("natural-originalMix04-vs-fresh-legacyMix04", q4 / "mix480-04",
           captures["mixed-legacy"]["natural"])
    record("natural-stackB-vs-fresh-prefillMix04", stack / "full-wide" / "capture",
           captures["mixed-prefill"]["natural"])
    record("forced-originalQ8-vs-fresh-legacyQ8", q4.parent / "q4-qualification" / "r3-teacher",
           captures["q8-legacy"]["forced"])
    record("natural-originalQ8-vs-fresh-prefillMix04", q4 / "teacher480r2",
           captures["mixed-prefill"]["natural"])
    record("natural-originalQ8-vs-fresh-decodeMix04", q4 / "teacher480r2",
           captures["mixed-decode"]["natural"])
    for weights, old in (("mixed", previous / "fp16-quality" / "logits" / "f32"),
                         ("q8", previous / "fp16-q8-control" / "logits" / "q8")):
        equal = []
        for i in range(80):
            before = (old / f"{i:03d}.fgl").read_bytes()
            after = (raw / "logits" / f"{weights}-decode" / f"{i:03d}.fgl").read_bytes()
            equal.append(before == after)
        result[f"previous_forced80_{weights}_decode_identity"] = dict(
            samples=80, byte_exact=sum(equal), all_exact=all(equal))
    for name in ("old-mix04-480", "old-stack-incremental-480", "old-stack-cumulative-480",
                 "old-build-drift-480"):
        result[name] = json.loads((root / f"{name}.json").read_text())
    result["forced_prefix_breakdown"] = {}
    result["union560"] = {}
    for method in ("decode", "prefill", "legacy"):
        forced = result["comparisons"][f"forced-cumulative-{method}"]["rows"]
        natural = result["comparisons"][f"natural-cumulative-{method}"]["rows"]
        result["forced_prefix_breakdown"][method] = {}
        for suffix, text in enumerate(("", "First", "First, consider", "First, consider the")):
            rows = [row for row in forced if row["id"].endswith(f"-{suffix}")]
            assert len(rows) == 20
            result["forced_prefix_breakdown"][method][text] = dict(
                samples=20, kl_mean=sum(row["kl"] for row in rows) / 20,
                top1_agreement=sum(row["top1"] for row in rows) / 20)
        rows = forced + natural
        mean = sum(row["kl"] for row in rows) / 560
        agreement = sum(row["top1"] for row in rows) / 560
        result["union560"][method] = dict(
            samples=560, kl_mean=mean, top1_agreement=agreement,
            strict_gate=mean <= .005 and agreement >= .985)
    result["artifact_sha256"] = {}
    for name in ("evidence.tar.gz", "logits.tar.gz"):
        with (root / name).open("rb") as f:
            result["artifact_sha256"][name] = hashlib.file_digest(f, "sha256").hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "q4", "stack", "previous", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = run(a.root, a.q4, a.stack, a.previous)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
