import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mtp_diagnose import first_difference, integrate_offline, integrate_oracles, parse_trace, recover_tokens, token_digest, validate_stage_ledgers


class DiagnosisTests(unittest.TestCase):
    def test_token_receipt_encoding(self):
        tokens = [1, 0x12345678, 9]
        digest = hashlib.sha256(struct.pack("<III", *tokens)).hexdigest()
        self.assertEqual(token_digest(tokens), digest)
        receipt = dict(count=3, sha256=digest)
        self.assertEqual(recover_tokens(dict(tokens=tokens, eos=9), receipt), tokens)
        self.assertEqual(recover_tokens(dict(tokens=tokens[:-1], eos=9), receipt), tokens)
        self.assertIsNone(recover_tokens(dict(tokens=tokens[::-1], eos=9), receipt))

    def test_request_round_attribution(self):
        digest = "a"*64
        trace = ("MTP_ROUND slot=0 first=26 k=3 accepted=1 emitted=2 consumed=2 ms=121.250\n"
                 f"API_TOKEN_HASH id=request-one count=4 sha256={digest}\n"
                 f"API_TOKEN_HASH id=request-two count=2 sha256={digest}\n")
        parsed = parse_trace(trace)
        self.assertEqual(parsed["request-one"]["rounds"][0]["milliseconds"], 121.25)
        self.assertEqual(parsed["request-two"]["rounds"], [])
        with self.assertRaises(AssertionError):
            parse_trace(trace+trace)

    def test_noncanonical_token_split_requires_receipt(self):
        vocabulary = ({1: b"ab", 2: b"a", 3: b"b"}, {b"a": [2], b"b": [3]})
        receipt = dict(count=2, sha256=token_digest([2, 3]))
        self.assertEqual(recover_tokens(dict(tokens=[1], eos=9), receipt, vocabulary), [2, 3])
        receipt["sha256"] = "f"*64
        self.assertIsNone(recover_tokens(dict(tokens=[1], eos=9), receipt, vocabulary))

    def test_public_response_strips_trailing_whitespace(self):
        vocabulary = ({1: b"yes", 2: b"\n"}, {b"yes": [1], b"\n": [2]})
        receipt = dict(count=2, sha256=token_digest([1, 2]))
        self.assertEqual(recover_tokens(dict(tokens=[1], eos=9), receipt, vocabulary), [1, 2])

    def test_first_difference(self):
        self.assertEqual(first_difference([2, 4], [2, 5]), 1)
        self.assertEqual(first_difference([2], [2, 5]), 1)
        self.assertEqual(first_difference([], []), 0)

    def test_stage_ledger_is_disjoint(self):
        trace = ("MTP_STAGE slot=0 first=26 k=3 draft_ms=10 prepare_ms=1 verify_ms=80 "
                 "target_repair_ms=2 native_repair_ms=3 commit_ms=1 host_ms=4 "
                 "total_ms=101 target_replay=0\n")
        self.assertEqual(validate_stage_ledgers(trace), 1)
        with self.assertRaises(AssertionError):
            validate_stage_ledgers(trace.replace("total_ms=101", "total_ms=120"))
        with self.assertRaises(AssertionError):
            validate_stage_ledgers(trace.replace("target_replay=0", "target_replay=1"))
        rank = ("MTP_VERIFY_RANK rank=7 first=26 n=4 replay=0 input_ms=1 compute_ms=2 "
                "read_ms=3 forward_or_head_ms=4 total_ms=10 status=0\n")
        self.assertEqual(validate_stage_ledgers(trace+rank), 1)
        with self.assertRaises(AssertionError):
            validate_stage_ledgers(trace+rank.replace("total_ms=10", "total_ms=11"))

    def test_model_margins_require_historical_reproduction(self):
        row = dict(case="F3-chat-1", prefix_predictions_checked=24,
                   prefix_predictions_mismatched=0, off_top1=10, on_top1=12,
                   off_top2=12, on_top2=10,
                   off_margin=0.001, on_margin=0.002, max_logit_delta=0.002,
                   historical_choices_reproduced=True)
        off, on = [-20.]*16, [-20.]*16
        off[10], off[12], on[10], on[12] = .001, 0., 0., .002
        raw = struct.pack("<32f", *(off+on))
        texts = {"offline-result.jsonl": json.dumps(row)}
        for index in range(8):
            records = [dict(kind="stage", case=row["case"], first=index*6, last=index*6+5),
                       dict(kind="state_comparison", case=row["case"])]
            texts[f"offline-stage{index}.jsonl"] = "\n".join(map(json.dumps, records))
        report = dict(mismatches=[dict(arm="F3", case="chat-1", off_token=10, on_token=12)])
        with patch.object(Path, "read_text", lambda path: texts[path.name]), \
                patch.object(Path, "read_bytes", lambda path: raw):
            integrate_offline(report, Path("evidence"), vocab_size=16)
        self.assertEqual(report["mismatches"][0]["off_top2_margin"], 0.001)
        self.assertEqual(report["offline_model"]["historical_cases_reproduced"], 1)
        row["on_top1"] = 13
        texts["offline-result.jsonl"] = json.dumps(row)
        with patch.object(Path, "read_text", lambda path: texts[path.name]), \
                patch.object(Path, "read_bytes", lambda path: raw):
            with self.assertRaises(AssertionError):
                integrate_offline(report, Path("evidence"), vocab_size=16)

    def test_trained_oracles_require_matching_streams(self):
        texts, original, repeated = {}, [], []
        for index in range(8):
            row = dict(kind="trained_rollback_oracle", case="F3-chat-1", first_layer=index*6,
                       full_journal_exact=True, rollback_hc_exact=True, rollback_state_exact=True,
                       future_changed_prefix_max_abs=0., future_changed_prefix_max_relative=0.)
            texts[f"oracle-stage{index}.jsonl"] = json.dumps(row)
            for suffix in ("off", "on"):
                original.append(f'{"a"*64}  offline-stage{index}.{suffix}')
                repeated.append(f'{"a"*64}  oracle-stage{index}.{suffix}')
        texts["offline-frames.sha256"] = "\n".join(original)
        texts["oracle-frames.sha256"] = "\n".join(repeated)
        report = dict(mismatches=[dict(arm="F3", case="chat-1",
                                      offline_model=dict(historical_choices_reproduced=True))])
        with patch.object(Path, "read_text", lambda path: texts[path.name]):
            integrate_oracles(report, Path("oracles"), Path("model"))
        self.assertEqual(report["trained_state_oracles"]["cases"], 8)
        self.assertTrue(report["trained_state_oracles"]["first_row_future_perturbation_bit_exact"])
        texts["oracle-frames.sha256"] = texts["oracle-frames.sha256"].replace("a"*64, "b"*64, 1)
        with patch.object(Path, "read_text", lambda path: texts[path.name]):
            with self.assertRaises(AssertionError):
                integrate_oracles(report, Path("oracles"), Path("model"))


if __name__ == "__main__":
    unittest.main()
