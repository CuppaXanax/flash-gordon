import copy
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from q4_qualify import evaluate
import q4_quality


class Gates(unittest.TestCase):
    def fixture(self):
        baseline = dict(prefill=[], decode=[], ttft=[], concurrency={}, runs={"full": {"elapsed_seconds": 100}})
        run = dict(prefill=[], decode=[], ttft=[], concurrency={}, metadata={"elapsed_seconds": 100},
                   correctness=[{"reference_match": True} for _ in range(5)], errors=[])
        for label, tokens in [("4K", 4096), ("32K", 32768), ("128K", 131072)]:
            baseline["prefill"].append(dict(target=label, client_prompt_tokens_over_cold_stream_ttft_tps=100,
                engine_prefill_tps=100, client_ttft_ms=100))
            run["prefill"].append(dict(target_prompt_tokens=tokens, client_prefill_tps=100,
                engine_prefill_tps=100, stream_first_content_ms=100, cold_measurement=True))
        for label, name in [("short-context-full", "short-context"), ("32K-context", "context-32k"), ("128K-context", "context-128k")]:
            baseline["decode"].append(dict(case=label, client_sse_interval_tps_excluding_first=100,
                engine_decode_tps=100, inter_event_p50_ms=100, inter_event_p90_ms=100))
            run["decode"].append(dict(name=name, client_sse_inter_event_tps_excluding_first=100,
                engine_decode_tps=100, inter_event_p50_ms=100, inter_event_p90_ms=100))
        for name in ["chat-30-token-ceiling", "4k-cold", "agent-style-cached-3.9k-prefix"]:
            baseline["ttft"].append(dict(case=name, ttft_ms=100))
            run["ttft"].append(dict(name=name, ttft_ms=100))
        for key in ["chat_style", "agent_style"]:
            baseline["concurrency"][key] = dict(end_to_end_visible_delta_events_per_second=100, wall_seconds=100)
        baseline["concurrency"]["aggregate_visible_delta_events_per_second"] = 100
        run["concurrency"] = dict(streams=[dict(name=n, client_end_to_end_delta_events_per_second=100,
            wall_seconds=100) for n in ["chat-style", "agent-style"]],
            aggregate_end_to_end_delta_events_per_second=100)
        return baseline, run

    def test_all_metrics_and_boundary(self):
        b, r = self.fixture()
        x = evaluate(b, r, {"passed": True})
        self.assertTrue(x["passed"])
        self.assertEqual(len(x["metrics"]), 30)
        r["prefill"][0]["client_prefill_tps"] = 97
        self.assertTrue(evaluate(b, r, {"passed": True})["passed"])
        r["prefill"][0]["client_prefill_tps"] = 96.99
        self.assertFalse(evaluate(b, r, {"passed": True})["passed"])

    def test_missing_quality_and_greedy(self):
        b, r = self.fixture()
        self.assertFalse(evaluate(b, r, {"passed": False})["passed"])
        r["correctness"][0]["reference_match"] = False
        self.assertFalse(evaluate(b, r, {"passed": True})["passed"])
        r = copy.deepcopy(r)
        r["prefill"] = []
        self.assertFalse(evaluate(b, r, {"passed": True})["performance_passed"])

    def test_invalid_measurements_fail_closed(self):
        b, r = self.fixture()
        for value in [0, -1, float("nan"), float("inf"), True, None]:
            with self.subTest(value=value):
                r["prefill"][0]["stream_first_content_ms"] = value
                self.assertFalse(evaluate(b, r, {"passed": True})["passed"])
        b, r = self.fixture()
        self.assertFalse(evaluate(b, r, {})["passed"])
        r["prefill"][0]["cold_measurement"] = "false"
        self.assertFalse(evaluate(b, r, {"passed": True})["passed"])

    def test_framed_corpus_exact_and_exclusive(self):
        path = Path(f".q4-framed-test-{os.getpid()}.fgev")
        self.assertFalse(path.exists())
        try:
            with patch.object(sys, "argv", ["quality", "--framed-contexts", str(path)]):
                q4_quality.main()
                raw = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    q4_quality.main()
            self.assertEqual(raw[:8], b"FGEVAL01")
            offset = 8
            for _, prompt in q4_quality.contexts():
                size, = struct.unpack_from("<I", raw, offset)
                offset += 4
                self.assertEqual(raw[offset:offset+size].decode("utf-8"), prompt)
                offset += size
            self.assertEqual(offset, len(raw))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
