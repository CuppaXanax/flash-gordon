import importlib.util
import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

spec = importlib.util.spec_from_file_location("capture", Path(__file__).parents[1] / "tools" / "mtp_capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def fixture(reject=False):
    return [dict(case=f"{category}-{i:02d}", category=category, position=p,
                 draft=[1, 2, 3] if not reject else [9, 2, 3], target=[1, 2, 3],
                 match=[1, 1, 1] if not reject else [0, 1, 1],
                 prefix=[1, 1, 1] if not reject else [0, 0, 0])
            for category in ("chat", "code") for i in range(20) for p in range(256)]


class CaptureTests(unittest.TestCase):
    def test_corpus(self):
        # Project-local scratch, never the system temporary directory.
        root = Path(__file__).parents[1] / ".build"
        root.mkdir(exist_ok=True)
        directory=root / ("capture-test-" + uuid.uuid4().hex)
        try:
            capture.corpus(directory)
            data = json.loads((directory / "corpus.json").read_text())
            self.assertEqual(len(data["cases"]), 48)
            self.assertEqual(len({c["sha256"] for c in data["cases"]}), 48)
            for case in data["cases"]:
                raw = (directory / (case["id"] + ".txt")).read_bytes()
                self.assertNotIn(b"\r", raw)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), case["sha256"])
            self.assertIn("<think>\n\n</think>", (directory / "chat-00.txt").read_text())
        finally:
            shutil.rmtree(directory)

    def test_all_accepted(self):
        result = capture.summarize(fixture())
        for k in (1, 2, 3):
            self.assertEqual(result["acceptance"]["all"][f"k{k}"]["anchored_committed_tokens_per_pass"], k + 1)
        self.assertEqual(result["acceptance"]["all"]["k3"]["fixed_k_policy_walk"]["tokens_per_pass"], 255 / 64)

    def test_conditional_denominators(self):
        rows = fixture()
        for row in rows:
            case = int(row["case"][-2:])
            if case % 2:
                row.update(draft=[9, 2, 3], match=[0, 1, 1], prefix=[0, 0, 0])
            elif case % 4 == 0:
                row.update(draft=[1, 9, 3], match=[1, 0, 1], prefix=[1, 0, 0])
        positions = capture.summarize(rows)["acceptance"]["all"]["k3"]["draft_positions"]
        self.assertEqual([p["conditional_acceptance"] for p in positions], [.5, .5, 1])
        self.assertEqual([p["attempts"] for p in positions], [10240, 5120, 2560])

    def test_rejected_prefix(self):
        result = capture.summarize(fixture(True))
        self.assertEqual(result["acceptance"]["all"]["k3"]["anchored_committed_tokens_per_pass"], 1)
        self.assertIsNone(result["acceptance"]["all"]["k3"]["draft_positions"][1]["conditional_acceptance"])

    def test_incomplete(self):
        with self.assertRaises(ValueError):
            capture.summarize(fixture()[:-1])

    def test_duplicate(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            capture.summarize(rows + rows[:1])

    def test_greedy_coverage(self):
        receipts = [dict(case=f"{category}-{i:02d}", k=k, equal=True,
                         tokens=256, rollbacks=1, verifier="causal-N1-reference")
                    for category in ("chat", "code") for i in range(4) for k in (1, 2, 3)]
        self.assertTrue(capture.summarize(fixture(), receipts)["greedy"]["passed"])
        receipts[0]["equal"] = False
        with self.assertRaises(ValueError):
            capture.summarize(fixture(), receipts)


if __name__ == "__main__":
    unittest.main()
