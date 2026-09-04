import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze_fg_profile", HERE / "analyze_fg_profile.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def stage(rank, stage_number, kind="prefill", tokens=8, sequence=1, wall=2.0):
    return {"schema": "flash-gordon.profile", "version": 1,
            "record_type": "stage_service", "rank": rank, "stage": stage_number,
            "sequence": sequence, "execution_kind": kind, "first_token": 0,
            "tokens": tokens, "wall_ms": wall, "gpu_ms": wall / 2,
            "kernel_ms": wall / 3, "submissions": 1, "dispatches": 2,
            "boundary_bytes": 40960, "terminal": stage_number == 7,
            "status": 0}


def fabric(rank=0, size=1000, wall=1.0):
    return {"schema": "flash-gordon.profile", "version": 1,
            "record_type": "fabric_service", "local_rank": rank, "peer": 1,
            "fabric_class": "bulk", "direction": "send", "message_type": 34,
            "payload_bytes": size, "framed_bytes": size + 36, "wall_ms": wall,
            "mode": "io_uring", "status": 0}


class AnalyzerTests(unittest.TestCase):
    def complete(self, kind="prefill", tokens=8):
        return [stage(rank, rank, kind, tokens, wall=float(rank + 1)) for rank in range(8)]

    def test_json_escaping_and_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(json.dumps(stage(0, 0)) + "\n{broken\n", encoding="utf-8")
            with self.assertRaises(MOD.ProfileError):
                MOD.load_jsonl([path])

    def test_completeness_and_partial_warning(self):
        with self.assertRaises(MOD.ProfileError):
            MOD.analyze(self.complete()[:7])
        result = MOD.analyze(self.complete()[:7], allow_partial=True)
        self.assertTrue(result["partial"])
        self.assertEqual(result["missing_rank_ids"], [7])

    def test_prefill_max_stage_formula_and_ttft(self):
        result = MOD.analyze(self.complete())
        group = result["stage_service"]["groups"][0]
        self.assertEqual(group["slowest_stage_service_ms"], 8.0)
        self.assertAlmostEqual(group["prefill_pipeline_steady_state_upper_bound_tps"], 1000.0)
        self.assertAlmostEqual(group["pipeline_fill_ttft_service_ms"], 36.0)

    def test_decode_sum_and_max_are_distinct(self):
        result = MOD.analyze(self.complete("decode", 1))
        group = result["stage_service"]["groups"][0]
        self.assertAlmostEqual(group["decode_observed_critical_path_sum_ms"]["p50_ms"], 36.0)
        self.assertAlmostEqual(group["decode_steady_state_max_stage_tps"], 125.0)
        self.assertAlmostEqual(group["decode_sum_stage_tps_for_comparison"], 1000.0 / 36.0)

    def test_mixed_batches_are_grouped(self):
        records = self.complete() + self.complete(tokens=128)
        result = MOD.analyze(records)
        self.assertEqual({group["tokens"] for group in result["stage_service"]["groups"]}, {8, 128})

    def test_different_context_positions_are_not_averaged(self):
        records = self.complete()
        later = self.complete()
        for record in later:
            record["first_token"] = 65536
            record["sequence"] = 2
        result = MOD.analyze(records + later)
        self.assertEqual({group["first_token"] for group in
                          result["stage_service"]["groups"]}, {0, 65536})

    def test_failed_stage_is_not_used_as_measurement(self):
        records = self.complete()
        records[-1]["status"] = 1
        result = MOD.analyze(records)
        self.assertFalse(result["stage_service"]["groups"][0]
                         ["complete_stage_coverage"])

    def test_fabric_aggregation_and_unknown_correlation(self):
        result = MOD.analyze(self.complete() + [fabric(0), fabric(0, 2000, 2.0)])
        row = result["fabric_service"]["groups"]
        self.assertEqual(len(row), 2)
        self.assertIsNone(result["fabric_service"]["fabric_share_of_stage_wall"])

    def test_counterfactual_label(self):
        result = MOD.analyze(self.complete())
        self.assertEqual(result["counterfactual"]["label"], "counterfactual")
        self.assertIsNone(result["counterfactual"]["expert_service_balance"])

    def test_invalid_units_and_mode_rejected(self):
        bad = fabric()
        bad["wall_ms"] = -1
        with self.assertRaises(MOD.ProfileError):
            MOD._validate_record(bad, 1)
        bad = fabric()
        bad["mode"] = "other"
        with self.assertRaises(MOD.ProfileError):
            MOD._validate_record(bad, 1)


if __name__ == "__main__":
    unittest.main()
