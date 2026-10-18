import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("fusion_ledger", Path(__file__).with_name("fusion-ledger.py"))
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


class FusionLedgerTests(unittest.TestCase):
    def test_expert_traffic_is_selected_not_full_slab(self):
        count, _ = ledger.traffic({"name": "fg_moe_decode_gate_up.spv",
            "binding_bytes": [471859200, 471859200], "push": []})
        self.assertEqual(count, 18432000)

    def test_norm_traffic_is_logical_not_scratch_capacity(self):
        count, _ = ledger.traffic({"name": "fg_group_rms_norm.spv",
            "binding_bytes": [5242880, 40960, 5242880], "push": [2560, 4, 1, 0]})
        self.assertEqual(count, 122880)

    def test_unknown_traffic_is_not_zero(self):
        self.assertIsNone(ledger.traffic({"name": "unknown.spv", "binding_bytes": [], "push": []})[0])

    def test_pairing_averages_both_baselines_and_rejects_parity(self):
        def capture(path):
            seed = int(path.stem.split("-")[1])
            phase = path.stem.split("-")[2]
            wall = {"a": 100, "b": 90, "a2": 110}[phase]
            config = dict(weights="synthetic", seed=seed, warmup=1, iterations=1,
                          windows=1, clocks=True, intrusive=False, fusion_mask=7 if phase == "b" else 0)
            sample = dict(wall_us=wall, layers_us=[wall / 6] * 6, window=0)
            return [], config, [sample], {"output": seed, "state": seed}
        with patch.object(ledger, "load", side_effect=capture):
            result = ledger.paired(Path("."))
        self.assertEqual(result["baseline_wall_us"], 105)
        self.assertEqual(result["candidate_wall_us"], 90)
        self.assertEqual(result["round_bootstrap_saving_us_95"], [15, 15])
        def corrupt(path):
            rows, config, samples, parity = capture(path)
            if path.stem.endswith("-b"):
                parity["state"] += 1
            return rows, config, samples, parity
        with patch.object(ledger, "load", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "parity failed"):
                ledger.paired(Path("."))

    def test_load_rejects_missing_static_replay(self):
        text = ('{"kind":"config","warmup":1,"iterations":1,"windows":1,"intrusive":false}\n'
                '{"kind":"sample","token":1,"static_slots":0}\n'
                '{"kind":"parity"}\n')
        with patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "static replay is not active"):
                ledger.load(Path("capture.jsonl"))


if __name__ == "__main__":
    unittest.main()
