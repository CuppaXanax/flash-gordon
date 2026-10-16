import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_mtp_bandwidth", PROJECT / "tools" / "analyze-mtp-bandwidth.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class BandwidthAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / ".build" / ("mtp-analysis-test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def write(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def wall(self, name, value):
        self.write("paired-layout/" + name + ".jsonl", [
            {"kind": "config", "iterations": 2, "windows": 1},
            {"kind": "sample", "wall_us": value},
            {"kind": "sample", "wall_us": value}])

    def test_brackets_and_n1_before_glob(self):
        for r in range(1, 6):
            self.wall(f"r{r}-n1-before-f0", 80)
            self.wall(f"r{r}-n1-after-f0", 80)
            self.wall(f"r{r}-n2-a0-f330", 90)
            self.wall(f"r{r}-n2-a1-f330", 110)
            self.wall(f"r{r}-n2-b0-f2097482", 80)
        row = ANALYSIS.analyze(self.root)["paired"]["layout"][0]
        self.assertEqual(row["baseline_us"]["mean"], 100)
        self.assertEqual(row["candidate_ratio"]["mean"], 1)
        self.assertAlmostEqual(row["reduction_percent"]["mean"], 20)
        self.assertEqual(row["reduction_percent"]["ci95_half"], 0)

    def test_shape_payload_and_cacheable_bandwidth(self):
        for r in range(1, 6):
            for suffix, value in (("a0-f330", 100), ("a1-f330", 100),
                                  ("b-f2097482", 80)):
                self.write(f"paired-layout/shape-r{r}-k2560-m10240-t12-{suffix}.jsonl", [
                    {"kind": "dense_cost", "compact": 1, "n": 4, "us": value}])
        row = ANALYSIS.analyze(self.root)["shapes"][0]
        self.assertEqual(row["samples"][0]["weight_bytes"], 15155200)
        self.assertAlmostEqual(row["candidate_GBs"]["mean"], 189.44)
        self.assertAlmostEqual(row["reduction_percent"]["mean"], 20)

    def test_all_layer_union_and_pooled_scope(self):
        ledger = [{"copies": 1, "bytes_per_copy": 10000}]
        for layer in (6, 7):
            for name, size in (("gate_exps", 20), ("up_exps", 30), ("down_exps", 40)):
                ledger.append({"layer": layer, "copies": 512,
                               "name": name, "bytes_per_copy": size})
        self.write("paired-layout/ledger.jsonl", ledger)
        for r in range(1, 6):
            self.write(f"paired-layout/profile-r{r}-n4-f2097482.jsonl", [
                {"kind": "kernel", "scope": "pooled", "name": name, "gpu_us": value}
                for name, value in (("fg_small_reuse_q4", 100),
                                    ("fg_small_gate_up", 20), ("fg_small_down", 10))])
            self.write(f"paired-layout/routes-r{r}-n4-f2097482.jsonl", [
                {"layer": 6, "experts": [0, 1]},
                {"layer": 6, "experts": [1]},
                {"layer": 7, "experts": [2]}])
        row = ANALYSIS.analyze(self.root)["profiles"]["layout"][0]
        self.assertEqual(row["samples"][0]["compulsory_weight_bytes"],
                         {"dense": 10000, "expert_gate_up": 150, "expert_down": 120})
        self.assertAlmostEqual(row["effective_weight_GBs"]["expert_gate_up"]["mean"], .0075)
        self.assertEqual(len(row["samples"][0]["kernels"]), 3)

    def test_pilot_route_without_kind_is_not_an_oracle(self):
        self.write("round7/routes.jsonl", [{"layer": 6, "experts": [0, 1]}])
        result = ANALYSIS.analyze(self.root)
        self.assertFalse(result["pilots"])
        self.assertFalse(result["oracles"])


if __name__ == "__main__":
    unittest.main()
