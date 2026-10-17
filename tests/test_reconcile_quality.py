import json
import os
from pathlib import Path
import shutil
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reconcile_quality import compare, logits


class ReconcileQualityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ".build" / f"reconcile-test-{os.getpid()}"
        self.root.mkdir()
        for name in ("reference", "candidate"):
            (self.root / name).mkdir()
            (self.root / name / "corpus.json").write_text(
                json.dumps([["case", "exact rendered text"]]), encoding="utf-8")
        self.corpus = self.root / "reference" / "corpus.json"

    def tearDown(self):
        shutil.rmtree(self.root)

    def write(self, name, values):
        path = self.root / name / "case.fgl"
        path.write_bytes(b"FGLOG01\n" + struct.pack("<I", 248320)
                         + np.asarray(values, dtype="<f4").tobytes())
        return path

    def test_identity_and_known_uniform_kl(self):
        zero = np.zeros(248320, dtype=np.float32)
        self.write("reference", zero)
        self.write("candidate", zero)
        result = compare(self.root / "reference", self.root / "candidate", self.corpus)
        self.assertEqual(result["kl_mean"], 0)
        self.assertEqual(result["top1_agreement"], 1)
        changed = zero.copy()
        changed[9] = 2
        self.write("candidate", changed)
        result = compare(self.root / "reference", self.root / "candidate", self.corpus)
        expected = np.log((248319 + np.exp(2)) / 248320) - 2 / 248320
        self.assertAlmostEqual(result["kl_mean"], expected, places=13)
        self.assertEqual(result["top1_agreement"], 0)

    def test_corresponding_text_is_required(self):
        (self.root / "candidate" / "corpus.json").write_text(
            json.dumps([["case", "different text"]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corpus mismatch"):
            compare(self.root / "reference", self.root / "candidate", self.corpus)

    def test_incomplete_and_nonfinite_are_rejected(self):
        path = self.write("reference", np.zeros(248320))
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "length"):
            logits(path)
        for raw in (b"", b"FGLOG01\n", b"FGLOG01\n\0\0"):
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "header"):
                logits(path)
        values = np.zeros(248320)
        values[1] = np.inf
        path = self.write("reference", values)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            logits(path)


if __name__ == "__main__":
    unittest.main()
