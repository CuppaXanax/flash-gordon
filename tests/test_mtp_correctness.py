import copy
import hashlib
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mtp_correctness import evaluate, Q8_MANIFEST, MIX04_MANIFEST, CORPUS80


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.events = dict(requests=[dict(id="case", tokens=1,
            tokens_sha256=hashlib.sha256(struct.pack("<I", 9)).hexdigest())],
            rows=[dict(id="case", position=0, emitted=9, target_top1=9,
                off_top1=7, off_top2=9, target_top2=7, off_margin=.05,
                target_margin=.04, target_prefix_sha256="a"*64, off_prefix_sha256="a"*64)])
        self.quality = dict(full_vocab=248320, samples=80, mean_kld=.004,
            top1_agreement=.9875, reference_kind="original-Q8",
            same_prefix_geometry=True, frozen_corpus_matched=True, candidate_columns=4,
            reference_manifest_sha256=Q8_MANIFEST, candidate_manifest_sha256=MIX04_MANIFEST,
            corpus_sha256=CORPUS80)

    def test_low_margin_not_bitwise_identity(self):
        self.assertTrue(evaluate(self.events, self.quality)["passed"])

    def test_emission_must_match_actual_target(self):
        self.events["rows"][0]["emitted"] = 7
        self.assertFalse(evaluate(self.events, self.quality)["self_consistency"])

    def test_high_margin_or_non_top2_swap_fails(self):
        for field, value in (("off_margin", .1), ("target_margin", .11),
                             ("off_top2", 10), ("target_top2", 11)):
            events = copy.deepcopy(self.events)
            events["rows"][0][field] = value
            self.assertFalse(evaluate(events, self.quality)["passed"])

    def test_all_emissions_required(self):
        self.events["requests"][0]["tokens"] = 2
        self.assertFalse(evaluate(self.events, self.quality)["passed"])

    def test_receipt_and_prefix_bindings(self):
        self.events["rows"][0]["off_prefix_sha256"] = "b"*64
        self.assertFalse(evaluate(self.events, self.quality)["passed"])
        self.events["rows"][0]["off_prefix_sha256"] = "a"*64
        self.events["requests"][0]["tokens_sha256"] = "0"*64
        self.assertFalse(evaluate(self.events, self.quality)["passed"])

    def test_strict_original_q8_quality(self):
        for field, value in (("mean_kld", .0051), ("top1_agreement", .975),
                             ("top1_agreement", .986),
                             ("reference_kind", "mix04"), ("samples", 9),
                             ("samples", 480), ("candidate_columns", 1),
                             ("reference_manifest_sha256", "0"*64),
                             ("corpus_sha256", "0"*64),
                             ("same_prefix_geometry", False), ("mean_kld", float("nan"))):
            quality = dict(self.quality)
            quality[field] = value
            self.assertFalse(evaluate(self.events, quality)["quality_passed"])

    def test_duplicate_and_missing_evidence_fail(self):
        self.events["rows"] *= 2
        self.assertFalse(evaluate(self.events, self.quality)["passed"])
        self.assertFalse(evaluate({}, {})["passed"])


if __name__ == "__main__":
    unittest.main()
