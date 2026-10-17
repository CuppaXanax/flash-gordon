import unittest
import json
from pathlib import Path
from unittest.mock import patch

from analyze_mtp_block import category, compare, read_profile


class BlockAttributionTests(unittest.TestCase):
    def test_hc_dense_is_not_counted_as_unrelated_dense(self):
        self.assertEqual(category("gr_ffn_read_small", "fg_small_q4_n4.spv"),
                         "HC/projections/small ops")

    def test_expert_names_override_inherited_scope(self):
        self.assertEqual(category("gdn_prefill_output", "fg_small_gate_up_four_fma.spv"),
                         "Expert gate/up")
        self.assertEqual(category("expert_decode", "fg_moe_decode_down_q8.spv"),
                         "Expert down/reduce")

    def test_gdn_and_qsa_nonprojection_work(self):
        self.assertEqual(category("gdn_recurrent_prefill", "fg_gdn_small_cached.spv"),
                         "GDN convolution/recurrence/normalization")
        self.assertEqual(category("qsa_small", "fg_small_index_bf16_n4.spv"),
                         "QSA attention/cache/index")
        self.assertEqual(category("qsa_small", "fg_small_reuse8_q4_n4.spv"),
                         "Dense projections/shared")

    def test_qsa_bf16_index_classification_matches_n1(self):
        self.assertEqual(category("qsa_projection", "fg_dense_bf16_f32.spv"),
                         category("qsa_small", "fg_small_index_bf16_n4.spv"))

    def test_relative_excess_ranking_and_missing_n1(self):
        rows = compare({"HC": 10, "dense": 100}, {"dense": 180, "HC": 90, "new": 20})
        self.assertEqual([r["operation"] for r in rows], ["HC", "dense", "new"])
        self.assertEqual(sum(r["n4_us"] for r in rows), 290)
        self.assertEqual(rows[0]["excess_vs_1_5_us"], 75)
        self.assertIsNone(rows[-1]["ratio"])

    def test_scoped_profile_closes(self):
        rows = [
            {"kind": "kernel", "scope": scope, "name": "fg_small_q4_n4.spv",
             "count": 1, "gpu_us": us}
            for scope, us in (("gr_attn_read_small", 2), ("gr_ffn_write_small", 3))
        ] + [{"kind": "profile", "gpu_us": 5}]
        with patch.object(Path, "read_text", return_value="\n".join(map(json.dumps, rows))):
            grouped, _ = read_profile("unused", small=True)
        self.assertEqual(grouped, {"HC/projections/small ops": 5})

    def test_stale_profile_is_rejected(self):
        rows = [{"kind": "kernel", "scope": "gdn_prefill_output",
                 "name": "fg_small_gate_up_four_fma.spv", "count": 6, "gpu_us": 3},
                {"kind": "profile", "gpu_us": 3}]
        with patch.object(Path, "read_text", return_value="\n".join(map(json.dumps, rows))):
            with self.assertRaisesRegex(ValueError, "stale small-N scopes"):
                read_profile("unused", small=True)

    def test_total_includes_timestamp_gaps(self):
        rows = [{"kind": "kernel", "scope": "router", "name": "fg_dense_f32.spv",
                 "count": 1, "gpu_us": 5},
                {"kind": "profile", "gpu_us": 7, "gaps_us": 2}]
        with patch.object(Path, "read_text", return_value="\n".join(map(json.dumps, rows))):
            grouped, _ = read_profile("unused")
        self.assertEqual(grouped, {"Expert routing/union": 5, "Inter-dispatch gaps": 2})


if __name__ == "__main__":
    unittest.main()
