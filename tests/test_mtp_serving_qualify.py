import importlib.util
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "mtp_serving_qualify", pathlib.Path(__file__).parents[1] / "tools" / "mtp_serving_qualify.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "tools"))
from mtp_correctness import Q8_MANIFEST, MIX04_MANIFEST, CORPUS80


class ServingClientTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(".build") / f"mtp-serving-client-tests-{os.getpid()}"
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root)

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell unavailable")
    def test_local_head_plan_and_prelease_gate(self):
        script = pathlib.Path(__file__).parents[1]/"tools"/"Invoke-MTPServingQualification.ps1"
        command = ["pwsh", "-NoProfile", "-File", str(script)]
        result = subprocess.run(command+["-Action", "Plan", "-Profile", "SingleSessionPerf",
            "-LocalHead", "not-read-by-plan", "-DiagnosticGreedy"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["productionAccess"])
        self.assertTrue(plan["localDraftHead"])
        self.assertTrue(plan["diagnosticOnly"])
        self.assertEqual(plan["candidateSessions"], 1)
        self.assertEqual(plan["candidateQsaPageCacheMiB"], 440)
        result = subprocess.run(command+["-Action", "Run", "-LeaseConfirmed",
            "-LeaseId", "mock-new-policy"], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MTP qualification requires complete pass-local/Q8 artifacts", result.stderr)

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
        args.diagnostic = True
        args.report_output = str(self.root / "greedy-report.json")
        CLIENT.verify(args)
        report = json.loads(pathlib.Path(args.report_output).read_text())
        self.assertEqual(report["passed"], 7)
        self.assertFalse(report["qualified"])
        self.assertEqual(report["failed_cases"], ["0"])
        args.report_output = None
        with self.assertRaises(AssertionError):
            CLIENT.verify(args)

    def test_pass_local_gate_binds_all_http_receipts(self):
        paths = {k: self.root/k for k in ("candidate", "baseline", "trace", "baseline_trace",
                                         "correctness_events", "quality", "report_output")}
        digest = hashlib.sha256(struct.pack("<I", 9)).hexdigest()
        rows = [dict(case=str(i), semantic={"message": {"content": "new"}},
                     id=f"request-{i}", tokens=1) for i in range(8)]
        paths["candidate"].write_text(json.dumps(dict(rows=rows)))
        paths["baseline"].write_text(json.dumps(dict(rows=[
            dict(row, semantic={"message": {"content": "old"}}) for row in rows])))
        trace = "".join(f"API_TOKEN_HASH id={r['id']} count=1 sha256={digest}\n" for r in rows)
        paths["trace"].write_text(trace)
        paths["baseline_trace"].write_text(trace.replace(digest, "b"*64))
        events = dict(requests=[dict(id=r["id"], tokens=1, tokens_sha256=digest) for r in rows],
            rows=[dict(id=r["id"], position=0, emitted=9, target_top1=9, off_top1=7,
                       off_top2=9, target_top2=7, off_margin=.05, target_margin=.04,
                       target_prefix_sha256="a"*64, off_prefix_sha256="a"*64) for r in rows])
        quality = dict(full_vocab=248320, samples=80, mean_kld=.004,
                       top1_agreement=.9875, reference_kind="original-Q8",
                       same_prefix_geometry=True, frozen_corpus_matched=True, candidate_columns=4,
                       reference_manifest_sha256=Q8_MANIFEST, candidate_manifest_sha256=MIX04_MANIFEST,
                       corpus_sha256=CORPUS80)
        paths["correctness_events"].write_text(json.dumps(events))
        paths["quality"].write_text(json.dumps(quality))
        args = types.SimpleNamespace(**{k: str(v) for k, v in paths.items()})
        CLIENT.verify(args)
        report = json.loads(paths["report_output"].read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["historical_identity_mismatch_cases"]), 8)
        quality["mean_kld"] = .01
        paths["quality"].write_text(json.dumps(quality))
        with self.assertRaises(AssertionError):
            CLIENT.verify(args)
        events["requests"][0]["tokens_sha256"] = "0"*64
        paths["correctness_events"].write_text(json.dumps(events))
        with self.assertRaises(AssertionError):
            CLIENT.verify(args)
        args.quality = None
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

    def test_diagnostic_budget_is_explicit_and_bounded(self):
        args = types.SimpleNamespace(suite="perf", single_session=True, base="unused",
                                     phase="F1", diagnostic=False,
                                     output=str(self.root / "perf.json"))
        budget_path = self.root / "diagnostic-budget.json"
        budget = {"diagnostic_only": True, "perf_repeats": 1}
        budget_path.write_text(json.dumps(budget))
        with self.assertRaises(AssertionError):
            CLIENT.run_suite(args)
        args.diagnostic = True
        with mock.patch.object(CLIENT, "request", return_value={"decode_tps": 30}) as call:
            CLIENT.run_suite(args)
        self.assertEqual(call.call_count, 4)
        result = json.loads(pathlib.Path(args.output).read_text())
        self.assertEqual(result["diagnostic_budget"], budget)
        for repeat in (0, 4, True):
            budget["perf_repeats"] = repeat
            budget_path.write_text(json.dumps(budget))
            with self.assertRaises(AssertionError):
                CLIENT.run_suite(args)
        budget = {"diagnostic_only": False, "perf_repeats": 1}
        budget_path.write_text(json.dumps(budget))
        with self.assertRaises(AssertionError):
            CLIENT.run_suite(args)


if __name__ == "__main__":
    unittest.main()
