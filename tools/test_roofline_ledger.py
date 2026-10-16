import importlib.util
import math
from pathlib import Path
import struct
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load("roofline-compare")
ledger = load("roofline-ledger")


class RooflineAnalysisTests(unittest.TestCase):
    def compare(self, left, right, tolerance=1e-4):
        with patch.object(Path, "read_bytes", side_effect=[struct.pack("f" * len(left), *left),
                                                          struct.pack("f" * len(right), *right)]):
            return compare.compare("a", "b", tolerance, True)

    def test_exact(self):
        self.assertTrue(self.compare([1.0, -2.0, 0.0], [1.0, -2.0, 0.0])["bit_exact"])

    def test_tolerance(self):
        self.assertEqual(self.compare([1.0], [1.000003])["changed"], 1)
        with self.assertRaises(ValueError):
            self.compare([1.0], [1.001])

    def test_block_drift_is_diagnostic(self):
        self.assertGreater(self.compare([1.0], [1.001], None)["max_scaled"], 1e-4)

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            self.compare([1.0], [math.nan])

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            self.compare([1.0], [1.0, 2.0])

    def test_round_resampling(self):
        self.assertEqual(ledger.interval([{"saving": 12.0}] * 5, "saving"), [12.0, 12.0])
        self.assertEqual(ledger.interval([{"saving": float(i)} for i in range(5)], "saving"),
                         ledger.interval([{"saving": float(i)} for i in range(5)], "saving"))

    def test_aliases(self):
        self.assertEqual(ledger.canonical("fg_roof_qsa_h6.spv"), "fg_qsa_decode_attention_split.spv")
        self.assertEqual(ledger.canonical("fg_roof_hc_mix.spv"), "fg_gr_mix_partial.spv")
        self.assertEqual(ledger.canonical("fg_roof_control.spv"), "fg_dense_f32.spv")
        self.assertEqual(ledger.canonical("fg_dense_compact.spv"), "fg_dense_compact.spv")
        self.assertEqual(ledger.canonical("fg_roof_dense_iq4_w32.spv"), "fg_dense_compact_iq4.spv")

    def test_split_profile_includes_merge(self):
        rows = [{"kind": "profile"},
                {"kind": "kernel", "scope": "gr_attn_read", "name": "fg_roof_stream_q4_split4.spv",
                 "gpu_us": 40.0, "count": 2},
                {"kind": "kernel", "scope": "gr_attn_read", "name": "fg_roof_split_merge.spv",
                 "gpu_us": 2.0, "count": 2}]
        with patch.object(ledger.fusion, "load", return_value=(rows, {}, [], {})):
            values, _ = ledger.profile(Path("."), 40, 251)
        self.assertEqual(values[("gr_attn_read", "fg_dense_compact_long.spv")], 21.0)


if __name__ == "__main__":
    unittest.main()
