import importlib.util
import json
import os
import pathlib
import shutil
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "mtp_serving_qualify", pathlib.Path(__file__).parents[1] / "tools" / "mtp_serving_qualify.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


class ServingClientTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(".build") / f"mtp-serving-client-tests-{os.getpid()}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_incremental_sse(self):
        deltas = [{"role": "assistant"}, {"content": "α"}, {"content": "β"}, {}]
        frames = [": keepalive\n\n"]
        for index, delta in enumerate(deltas):
            event = {"choices": [{"delta": delta,
                                  "finish_reason": "stop" if index == 3 else None}]}
            frames.append("data: " + json.dumps(event) + "\n\n")
        raw = ("".join(frames) + "data: [DONE]\n\n").encode()
        self.assertEqual(CLIENT.decode_sse(raw), ("αβ", "", "stop"))
        with self.assertRaises(AssertionError):
            CLIENT.decode_sse(raw.replace(b"data: [DONE]\n\n", b""))
        with self.assertRaises(AssertionError):
            CLIENT.decode_sse(raw + b'data: {"choices":[]}\n\n')

    def test_exact_token_hash_gate(self):
        paths = {key: self.root / key for key in ("candidate", "baseline", "trace", "baseline_trace")}
        rows = [{"case": str(i), "semantic": {"message": {"content": str(i)}},
                 "id": f"request-{i}", "tokens": 96} for i in range(8)]
        for key in ("candidate", "baseline"):
            paths[key].write_text(json.dumps({"rows": rows}), encoding="utf-8")
        trace = "".join(f"API_TOKEN_HASH id=request-{i} count=96 sha256={'a' * 64}\n" for i in range(8))
        for key in ("trace", "baseline_trace"):
            paths[key].write_text(trace, encoding="utf-8")
        args = types.SimpleNamespace(**{key: str(value) for key, value in paths.items()})
        CLIENT.verify(args)
        paths["trace"].write_text(trace.replace("a" * 64, "b" * 64, 1), encoding="utf-8")
        with self.assertRaises(AssertionError):
            CLIENT.verify(args)

    def test_bracketed_pairing(self):
        phases = ["A0", "K1", "A1", "K2", "A2", "K3", "A3"]
        for phase in phases:
            scale = 2 if phase.startswith("K") else 1
            rows = [{"case": "chat-0", "repeat": 0, "category": "chat", "decode_tps": 30 * scale},
                    {"case": "code-0", "repeat": 0, "category": "code", "decode_tps": 30 * scale},
                    {"case": "coexist", "repeat": 0, "aggregate_wall_tps": 40 * scale}]
            (self.root / f"{phase}-perf.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
        (self.root / "K3-rank-0.log").write_text(
            "MTP_ROUND slot=1 first=1 k=3 accepted=1 emitted=2 consumed=2 ms=10.0\n"
            "MTP_ROUND slot=1 first=3 k=3 accepted=3 emitted=4 consumed=4 ms=20.0\n",
            encoding="utf-8")
        CLIENT.summarize(types.SimpleNamespace(directory=str(self.root)))
        summary = json.loads((self.root / "serving-summary.json").read_text(encoding="utf-8"))
        for comparison in summary["comparisons"]:
            for result in comparison["bracketed_tps_ratios"].values():
                self.assertEqual(result["paired_median"], 2)
        rounds = summary["native_rounds_all_phase_requests"]["K3"]["slot1-k3"]
        self.assertEqual(rounds["conditional_trials"], [2, 2, 1])
        self.assertEqual(rounds["conditional_successes"], [2, 1, 1])
        self.assertEqual(rounds["emitted_per_pass"], 3)
        self.assertEqual(rounds["mean_round_ms"], 15)
        shutil.copyfile(self.root / "A0-perf.json", self.root / "B0-perf.json")
        for k in (1, 2, 3):
            shutil.copyfile(self.root / f"K{k}-perf.json", self.root / f"S{k}-perf.json")
        CLIENT.summarize(types.SimpleNamespace(directory=str(self.root)))
        expanded = json.loads((self.root / "serving-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(expanded["comparisons"]), 6)
        for comparison in expanded["comparisons"]:
            for result in comparison["bracketed_tps_ratios"].values():
                self.assertEqual(result["paired_median"], 2)

    def test_single_session_perf_is_serial(self):
        args = types.SimpleNamespace(suite="perf", single_session=True, base="unused",
                                     phase="D3", output=str(self.root / "perf.json"))
        with mock.patch.object(CLIENT, "request", return_value={"decode_tps": 30}) as call:
            with mock.patch.object(CLIENT.concurrent.futures, "ThreadPoolExecutor") as pool:
                CLIENT.run_suite(args)
        self.assertEqual(call.call_count, 12)
        pool.assert_not_called()
        rows = json.loads((self.root / "perf.json").read_text())["rows"]
        self.assertEqual(len(rows), 12)
        self.assertFalse(any(row["case"] == "coexist" for row in rows))

    def test_single_session_brackets(self):
        phases = ["A0", "D3", "A1", "F1", "A2", "F2", "A3", "F3", "A4"]
        for phase in phases:
            scale = 1 if phase.startswith("A") else 2
            rows = [{"case": "chat-0", "repeat": 0, "category": "chat", "decode_tps": 30 * scale},
                    {"case": "code-0", "repeat": 0, "category": "code", "decode_tps": 30 * scale}]
            (self.root / f"{phase}-perf.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
        CLIENT.summarize(types.SimpleNamespace(directory=str(self.root)))
        summary = json.loads((self.root / "serving-summary.json").read_text())
        self.assertEqual([r["phase"] for r in summary["comparisons"]], ["D3", "F1", "F2", "F3"])
        self.assertTrue(summary["qualification"].startswith("single-session"))
        for comparison in summary["comparisons"]:
            self.assertEqual(set(comparison["bracketed_tps_ratios"]), {"chat", "code"})
            self.assertTrue(all(r["paired_median"] == 2 for r in comparison["bracketed_tps_ratios"].values()))


if __name__ == "__main__":
    unittest.main()
