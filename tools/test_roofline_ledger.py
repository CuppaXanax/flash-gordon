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

    def test_expert_split_and_recurrent_normalization_are_not_free(self):
        rows = [{"kind": "profile"}]
        for name, scope, duration in (
                ("fg_roof_down_split.spv", "expert_decode", 36),
                ("fg_roof_down_merge.spv", "expert_decode", 4),
                ("fg_roof_recurrent_vec.spv", "gdn_recurrent", 18),
                ("fg_roof_recurrent_norm.spv", "gdn_recurrent", 2)):
            rows.append({"kind": "kernel", "name": name, "scope": scope, "gpu_us": duration, "count": 1})
        with patch.object(ledger.fusion, "load", return_value=(rows, {}, [], {})):
            values, _ = ledger.profile(Path("."), 40, 2047, True)
        self.assertEqual(values[("expert_decode", "fg_moe_decode_down_reduce.spv")], 40)
        self.assertEqual(values[("gdn_recurrent", "fg_gdn_recurrent_algebraic.spv")], 20)

    def test_grouped_controls_count_both_alpha_and_beta(self):
        rows = [{"kind": "profile"},
                {"kind": "kernel", "scope": "gdn_gate", "name": "fg_roof_dense_l1_w32.spv",
                 "gpu_us": 120, "count": 4},
                {"kind": "kernel", "scope": "gdn_controls", "name": "fg_roof_control.spv",
                 "gpu_us": 24, "count": 8}]
        with patch.object(ledger.fusion, "load", return_value=(rows, {}, [], {})):
            values, _ = ledger.profile(Path("."), 40, 251, True)
        self.assertEqual(values[("gdn_gate_controls", "gdn_gate_controls")], 36)

    def test_hc_fusion_groups_do_not_drop_removed_dispatches(self):
        for name in ("fg_dense_compact_long.spv", "fg_roof_hc_inject.spv",
                     "fg_silu_scaled.spv", "fg_roof_hc_down_merge.spv"):
            self.assertEqual(ledger.logical_key("gr_attn_read", name, True),
                             ("gr_attn_read", "hc_down_inject"))
        for name in ("fg_roof_hc_up_mix_iq4.spv", "fg_dense_compact_iq4.spv",
                     "fg_roof_hc_mix.spv"):
            self.assertEqual(ledger.logical_key("gr_ffn_read", name, True),
                             ("gr_ffn_read", "hc_up_mix"))

    def test_incremental_dense_dispatch_scope(self):
        self.assertEqual(ledger.logical_key("gdn_gate", "fg_roof_dense_l1_w32.spv"),
                         ("gdn_gate", "fg_dense_compact_mid.spv"))

    def test_grouping_preserves_unrelated_repeated_dispatches(self):
        name="fg_dense_f32.spv"
        dispatches=[({"layer":7,"index":i,"name":name,"grid":[1,1,1],"binding_bytes":[128],"push":[]},
                     "qsa_projection") for i in (0,1)]
        with patch.object(ledger,"profile",return_value=({("qsa_projection",name):2.0},[])), \
             patch.object(ledger.fusion,"load",return_value=([],{},[],{})), \
             patch.object(ledger,"dispatch_rows",return_value=iter(dispatches)):
            rows,ranking,_=ledger.ledger(Path("."),40,1023,251,True)
        self.assertEqual(len(rows),2)
        self.assertEqual(ranking[0]["model_occurrences"],12)


if __name__ == "__main__":
    unittest.main()
