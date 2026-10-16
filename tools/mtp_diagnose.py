"""Analyze sealed MTP traces without treating missing logits as measurements."""

import argparse
from array import array
import collections
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys


ROUND = re.compile(
    r"MTP_ROUND slot=(\d+) first=(\d+) k=(\d+) accepted=(\d+) "
    r"emitted=(\d+) consumed=(\d+) ms=([0-9.]+)")
TOKEN_HASH = re.compile(r"API_TOKEN_HASH id=(\S+) count=(\d+) sha256=([0-9a-f]{64})")
ARMS = {"D3": ("A0", "A1"), "F1": ("A1", "A2"),
        "F2": ("A2", "A3"), "F3": ("A3", "A4")}


def parse_trace(text):
    requests, pending = {}, []
    for line in text.splitlines():
        match = ROUND.search(line)
        if match:
            slot, first, k, accepted, emitted, consumed, milliseconds = match.groups()
            assert slot == "0", "single-session trace required"
            pending.append(dict(first=int(first), k=int(k), accepted=int(accepted),
                                emitted=int(emitted), consumed=int(consumed),
                                milliseconds=float(milliseconds)))
        match = TOKEN_HASH.search(line)
        if match:
            assert match[1] not in requests, "duplicate request receipt"
            requests[match[1]] = dict(count=int(match[2]), sha256=match[3], rounds=pending)
            pending = []
    return requests


def token_digest(tokens):
    return hashlib.sha256(b"".join(struct.pack("<I", t) for t in tokens)).hexdigest()


def read_vocab(path):
    data = path.read_bytes()
    vocab, inverse = {}, collections.defaultdict(list)
    offset = 0
    while offset < len(data):
        token, size = struct.unpack_from("<II", data, offset)
        offset += 8
        assert size <= len(data)-offset
        raw = data[offset:offset+size]
        offset += size
        vocab[token] = raw
        if raw:
            inverse[raw].append(token)
    return vocab, inverse


def recover_tokens(encoded, receipt, vocabulary=None):
    if not encoded or not receipt:
        return None
    for tokens in (encoded["tokens"], encoded["tokens"] + [encoded["eos"]]):
        if len(tokens) == receipt["count"] and token_digest(tokens) == receipt["sha256"]:
            return tokens
    if not vocabulary:
        return None
    vocab, inverse = vocabulary
    budget = 250000
    for original in (encoded["tokens"], encoded["tokens"] + [encoded["eos"]]):
        if len(original)+1 != receipt["count"]:
            continue
        variants = [original]
        for candidate, raw in vocab.items():
            if raw and all(byte in b" \t\r\n\v\f" for byte in raw):
                tokens = original+[candidate]
                if token_digest(tokens) == receipt["sha256"]:
                    return tokens
        for i, token in enumerate(original):
            raw = vocab[token]
            if b"\xef\xbf\xbd" not in raw:
                continue
            visible = raw.decode("utf-8", errors="replace")
            for candidate, value in vocab.items():
                if value != raw and value.decode("utf-8", errors="replace") == visible:
                    variants.append(original[:i]+[candidate]+original[i+1:])
        for tokens in variants:
            for i, token in enumerate(tokens):
                raw = vocab[token]
                for split in range(1, len(raw)):
                    for left in inverse.get(raw[:split], []):
                        for right in inverse.get(raw[split:], []):
                            candidate = tokens[:i]+[left, right]+tokens[i+1:]
                            budget -= 1
                            if token_digest(candidate) == receipt["sha256"]:
                                return candidate
                            if budget <= 0:
                                return None
    return None


def first_difference(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)))


def validate_stage_ledgers(text):
    count = 0
    keys = ("draft_ms", "prepare_ms", "verify_ms", "target_repair_ms",
            "native_repair_ms", "commit_ms", "host_ms")
    for match in re.finditer(r"MTP_STAGE ([^\n]+)", text):
        fields = dict(part.split("=", 1) for part in match[1].split())
        assert fields["target_replay"] == "0"
        values = [float(fields[key]) for key in keys]
        assert min(values) >= -0.001
        overlap = float(fields.get("overlap_ms", 0))
        assert 0 <= overlap <= min(values[0], values[1])+0.002
        assert abs(sum(values)-overlap-float(fields["total_ms"])) <= 0.006
        count += 1
    for match in re.finditer(r"MTP_VERIFY_RANK ([^\n]+)", text):
        fields = dict(part.split("=", 1) for part in match[1].split())
        values = [float(fields[key]) for key in
                  ("input_ms", "compute_ms", "read_ms", "forward_or_head_ms")]
        assert min(values) >= 0
        assert abs(sum(values)-float(fields["total_ms"])) <= 0.004
    return count


def analyze(directory, encoded, vocabulary=None):
    phases = list(dict.fromkeys(["A0"] + [x for arm, bracket in ARMS.items()
                                        for x in (arm, *bracket)]))
    traces = {p: parse_trace((directory / f"{p}-rank-0.log").read_text())
              for p in phases}
    greedy = {p: json.loads((directory / f"{p}-greedy.json").read_text(
        encoding="utf-8-sig"))["rows"] for p in phases}
    perf = {p: json.loads((directory / f"{p}-perf.json").read_text(
        encoding="utf-8-sig"))["rows"] for p in phases}
    baseline = {r["case"]: r for r in greedy["A0"]}
    mismatches, recovery = [], {}
    for phase in phases:
        for row in greedy[phase]:
            key = f'{phase}-{row["case"]}'
            receipt = traces[phase].get(row["id"])
            tokens = recover_tokens(encoded.get(key), receipt, vocabulary)
            recovery[key] = dict(hash_verified=tokens is not None,
                                 encoded_supplied=key in encoded,
                                 receipt_count=receipt["count"] if receipt else None)
            base = baseline[row["case"]]
            left = base["semantic"]["message"]["content"]
            right = row["semantic"]["message"]["content"]
            if phase not in ARMS or left == right:
                continue
            index = first_difference(left, right)
            off = recover_tokens(encoded.get(f'A0-{row["case"]}'),
                                 traces["A0"].get(base["id"]), vocabulary)
            exact = tokens is not None and off is not None
            token_index = first_difference(off, tokens) if exact else None
            item = dict(arm=phase, case=row["case"], character_index=index,
                        off_excerpt=left[index:index+60], on_excerpt=right[index:index+60],
                        token_hash_verified=exact, token_index=token_index,
                        off_token=off[token_index] if exact and token_index < len(off) else None,
                        on_token=tokens[token_index] if exact and token_index < len(tokens) else None,
                        off_top2_margin=None, on_top2_margin=None,
                        classification="unresolved: historical logits were not recorded")
            if exact:
                item["common_output_prefix"] = off[:token_index]
            mismatches.append(item)
    ledger = {}
    for arm, (left, right) in ARMS.items():
        before = {(r["case"], r["repeat"]): r for r in perf[left]}
        after = {(r["case"], r["repeat"]): r for r in perf[right]}
        ledger[arm] = {}
        for category in ("chat", "code"):
            rows = [r for r in perf[arm] if r["category"] == category and
                    (r["case"], r["repeat"]) in before and
                    (r["case"], r["repeat"]) in after]
            rounds = [x for r in rows for x in traces[arm][r["id"]]["rounds"]]
            assert rounds, f"missing round evidence for {arm}/{category}"
            repairs = sum(r["consumed"] < r["k"]+1 for r in rounds)
            emitted = sum(r["emitted"] for r in rounds)
            grouped = collections.defaultdict(list)
            for r in rounds:
                grouped[f'k{r["k"]}-accepted{r["accepted"]}-keep{r["consumed"]}'].append(
                    r["milliseconds"])
            ledger[arm][category] = dict(
                matched_requests=len(rows), rounds=len(rounds), repair_passes=repairs,
                physical_target_passes=len(rounds)+repairs,
                emitted=emitted, tokens_per_verification_round=emitted/len(rounds),
                tokens_per_verify_or_repair_pass=emitted/(len(rounds)+repairs),
                measured_round_ms_mean=statistics.mean(r["milliseconds"] for r in rounds),
                measured_round_ms_median=statistics.median(r["milliseconds"] for r in rounds),
                amortized_total_round_ms_per_physical_target_pass=(
                    sum(r["milliseconds"] for r in rounds)/(len(rounds)+repairs)),
                grouped_round_ms={k: dict(count=len(v), mean=statistics.mean(v),
                    median=statistics.median(v)) for k, v in sorted(grouped.items())},
                historical_component_ms=None)
    return dict(scope="offline sealed-lease diagnosis; no new performance run",
                token_hash_encoding="little-endian uint32",
                token_recovery=recovery, mismatches=mismatches, ledger=ledger,
                offline_fixed_k3_tokens_per_verification_round=dict(chat=2.3899, code=3.2422),
                historical_components="not recorded; do not infer draft/kernel/transport time from TPS",
                greedy_gate="failed; no model-level equivalence claim")


def write_plan(directory, report, encoded, vocabulary, output):
    from mtp_serving_qualify import CASES
    prompts = {name: prompt for name, _, prompt in CASES}
    baseline = {r["case"]: r for r in json.loads(
        (directory / "A0-greedy.json").read_text(encoding="utf-8-sig"))["rows"]}
    before = parse_trace((directory / "A0-rank-0.log").read_text())
    with output.open("wb") as f:
        f.write(b"FGMDIAG1"+struct.pack("<I", len(report["mismatches"])))
        for mismatch in report["mismatches"]:
            assert mismatch["token_hash_verified"]
            arm, case = mismatch["arm"], mismatch["case"]
            row = next(r for r in json.loads((directory / f"{arm}-greedy.json").read_text(
                encoding="utf-8-sig"))["rows"] if r["case"] == case)
            prompt = int(row["headers"]["X-Flash-Gordon-Prompt-Tokens"])
            assert row["headers"]["X-Flash-Gordon-Reused-Tokens"] == "0"
            assert int(row["headers"]["X-Flash-Gordon-Prefilled-Tokens"]) == prompt
            tokens = recover_tokens(encoded[f"A0-{case}"], before[baseline[case]["id"]], vocabulary)
            assert tokens is not None
            rounds = []
            traces = parse_trace((directory / f"{arm}-rank-0.log").read_text())
            for r in traces[row["id"]]["rounds"]:
                rounds.append((r["first"], r["k"]+1, r["consumed"]))
                if r["first"]+r["consumed"] >= prompt+mismatch["token_index"]:
                    break
            assert rounds
            for text in (f"{arm}-{case}", prompts[case]):
                data = text.encode()
                f.write(struct.pack("<I", len(data))+data)
            f.write(struct.pack("<IIII", prompt, mismatch["token_index"],
                                mismatch["on_token"], len(tokens)))
            f.write(struct.pack("<"+"I"*len(tokens), *tokens))
            f.write(struct.pack("<I", len(rounds)))
            for r in rounds:
                f.write(struct.pack("<III", *r))


def validate_logits(rows, path, vocab_size):
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == len(rows)*2*vocab_size
    for index, row in enumerate(rows):
        off = values[index*2*vocab_size:(index*2+1)*vocab_size]
        on = values[(index*2+1)*vocab_size:(index*2+2)*vocab_size]
        for label, scores in (("off", off), ("on", on)):
            assert all(math.isfinite(x) for x in scores)
            ids = heapq.nlargest(2, range(vocab_size), key=scores.__getitem__)
            assert ids == [row[f"{label}_top1"], row[f"{label}_top2"]]
            assert math.isclose(scores[ids[0]]-scores[ids[1]], row[f"{label}_margin"],
                                rel_tol=1e-6, abs_tol=1e-7)
        delta = max(abs(a-b) for a, b in zip(off, on))
        assert math.isclose(delta, row["max_logit_delta"], rel_tol=1e-6, abs_tol=1e-7)
        a, b = max(off), max(on)
        za = math.log(math.fsum(math.exp(x-a) for x in off))
        zb = math.log(math.fsum(math.exp(x-b) for x in on))
        row["diagnostic_full_vocab_kl_off_on"] = max(0., math.fsum(
            math.exp(x-a-za)*((x-a-za)-(y-b-zb)) for x, y in zip(off, on)))


def integrate_oracles(report, directory, model_directory):
    expected = {f'{r["arm"]}-{r["case"]}' for r in report["mismatches"]}
    records = []
    for index in range(8):
        rows = [json.loads(line) for line in
                (directory / f"oracle-stage{index}.jsonl").read_text().splitlines()]
        proofs = [r for r in rows if r["kind"] == "trained_rollback_oracle"]
        assert len(proofs) == len(expected)
        assert {r["case"] for r in proofs} == expected
        for row in proofs:
            assert row["first_layer"] == index*6
            assert all(row[key] for key in
                       ("full_journal_exact", "rollback_hc_exact", "rollback_state_exact"))
            assert all(math.isfinite(row[key]) and row[key] >= 0 for key in
                       ("future_changed_prefix_max_abs", "future_changed_prefix_max_relative"))
        records.extend(proofs)
    original = {line.split()[1]: line.split()[0] for line in
                (model_directory / "offline-frames.sha256").read_text().splitlines()}
    repeated = {line.split()[1]: line.split()[0] for line in
                (directory / "oracle-frames.sha256").read_text().splitlines()}
    for index in range(8):
        for suffix in ("off", "on"):
            assert original[f"offline-stage{index}.{suffix}"] == repeated[f"oracle-stage{index}.{suffix}"]
    exact = all(r["future_changed_prefix_max_abs"] == 0 for r in records)
    report["trained_state_oracles"] = dict(
        cases=len(records), full_journal_exact=True, rollback_hc_exact=True,
        rollback_state_exact=True, original_streams_sha_matched=16,
        first_row_future_perturbation_bit_exact=exact, records=records,
        scope="nine frozen failure prefixes per owner; logical context256, not a full-context serving gate")
    if exact:
        for row in report["mismatches"]:
            if row["offline_model"]["historical_choices_reproduced"]:
                row["classification"] = (
                    "batch-dependent numerical drift swaps top2; trained owner rollback/"
                    "reconstruction and first-row future-column causality are bit-exact")


def integrate_offline(report, directory, vocab_size=248320, oracle_directory=None):
    expected = {f'{r["arm"]}-{r["case"]}': r for r in report["mismatches"]}
    rows = [json.loads(line) for line in
            (directory / "offline-result.jsonl").read_text().splitlines()]
    assert len(rows) == len(expected)
    validate_logits(rows, directory / "offline-result.logits", vocab_size)
    measured = {}
    for row in rows:
        assert row["case"] in expected and row["case"] not in measured
        assert row["prefix_predictions_checked"] > 0
        assert all(math.isfinite(row[key]) and row[key] >= 0 for key in
                   ("off_margin", "on_margin", "max_logit_delta"))
        mismatch = expected[row["case"]]
        reproduced = (not row["prefix_predictions_mismatched"] and
                      row["off_top1"] == mismatch["off_token"] and
                      row["on_top1"] == mismatch["on_token"])
        assert reproduced == row["historical_choices_reproduced"]
        mismatch["offline_model"] = row
        if reproduced:
            mismatch.update(off_top2_margin=row["off_margin"],
                            on_top2_margin=row["on_margin"],
                            classification="historical logit swap reproduced; state cause requires separate evidence")
        measured[row["case"]] = row
    stages = []
    for first in range(0, 48, 6):
        records = [json.loads(line) for line in
                   (directory / f"offline-stage{first//6}.jsonl").read_text().splitlines()]
        complete = [r for r in records if r["kind"] == "stage"]
        states = [r for r in records if r["kind"] == "state_comparison"]
        assert len(complete) == len(expected) == len(states)
        assert {r["case"] for r in complete} == expected.keys()
        assert {r["case"] for r in states} == expected.keys()
        assert all(r["first"] == first and r["last"] == first+5 for r in complete)
        blocks = [r for r in records if r["kind"] == "block_ledger"]
        for block in blocks:
            assert min(block[k] for k in ("input_ms", "compute_ms", "read_ms")) >= 0
            assert abs(sum(block[k] for k in ("input_ms", "compute_ms", "read_ms")) -
                       block["total_ms"]) <= 0.00001
        service = {}
        for n in range(1, 5):
            selected = [r["compute_ms"] for r in blocks if
                        r["operation"] == ("off" if n == 1 else "verify") and r["n"] == n]
            if selected:
                service[str(n)] = dict(count=len(selected), median=statistics.median(selected))
        stages.append(dict(first_layer=first, cases=len(complete), states=states,
                           block_ledger_records=len(blocks), owner_compute_ms_by_n=service))
    summed_service = None
    if all(len(stage["owner_compute_ms_by_n"]) == 4 for stage in stages):
        summed_service = {str(n): sum(stage["owner_compute_ms_by_n"][str(n)]["median"]
                                     for stage in stages) for n in range(1, 5)}
    report["offline_model"] = dict(
        historical_cases_reproduced=sum(r["historical_choices_reproduced"] for r in rows),
        cases=len(expected), stages=stages,
        summed_median_owner_compute_ms_by_n=summed_service,
        scope="rank-major authentic .51 model replay; not historical component timings",
        service_scope="mixed diagnostic failure cohorts on one blade, not a matched performance gate or ring time",
        kl_scope="selected first-divergence contexts, not an unbiased model-quality gate",
        future_inputs="unlogged discarded drafts replaced by teacher-forced future inputs")
    if oracle_directory:
        integrate_oracles(report, oracle_directory, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--offline", type=Path)
    parser.add_argument("--oracles", type=Path)
    args = parser.parse_args()
    encoded = {}
    if args.tokens:
        encoded = {r["name"]: r for r in map(json.loads, args.tokens.read_text().splitlines())}
    vocabulary = read_vocab(args.vocab) if args.vocab else None
    report = analyze(args.directory, encoded, vocabulary)
    if args.oracles and not args.offline:
        parser.error("--oracles requires --offline")
    if args.offline:
        integrate_offline(report, args.offline, oracle_directory=args.oracles)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if args.plan:
        write_plan(args.directory, report, encoded, vocabulary, args.plan)


if __name__ == "__main__":
    main()
