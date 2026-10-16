import copy
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from q4_qualify import evaluate
import q4_quality
import q4_make_contexts
import q4_profile_report


class Gates(unittest.TestCase):
    def test_capture_corpus_binding_and_subsets(self):
        scratch=Path(f".q4-corpus-test-{os.getpid()}")
        scratch.mkdir()
        try:
            path=scratch/"corpus.json"
            path.write_text(json.dumps([["first","中文"],["second","different text"]],
                ensure_ascii=False),encoding="utf-8")
            q4_quality.validate_capture_corpus(scratch,[("first","中文")])
            q4_quality.validate_capture_corpus(scratch,[("second","different text")])
            with self.assertRaisesRegex(ValueError,"frozen text"):
                q4_quality.validate_capture_corpus(scratch,[("second","wrong text")])
            path.write_text(json.dumps([["first","text"],["first","text"]]))
            with self.assertRaises(ValueError):
                q4_quality.validate_capture_corpus(scratch,[("first","text")])
        finally:
            shutil.rmtree(scratch)

    def test_profile_coverage_and_route_geometry(self):
        scratch=Path(f".q4-profile-test-{os.getpid()}")
        scratch.mkdir()
        try:
            text=("PREFILL_ROUTE rank={rank} layer=1 tokens=128 pairs=5 distinct=2 "
                  "tiles=2 width=16 max_count=3\n"
                  "PREFILL_LAYER_KERNEL rank={rank} layer=1 scope=dense "
                  "kernel=test calls=1 gpu_ms=1.0\n"
                  "PREFILL_WORKER rank={rank} layer=1 tokens=128 decode_ms=1.0 "
                  "forward_ms=2.0 compute_ms=3.0 send_ms=4.0 total_ms=10.0\n"
                  "PREFILL_BLOCK_LAYER rank={rank} layer=1 ms=3.0\n"
                  "PREFILL_LAYER_PROFILE rank={rank} layer=1 gpu_ms=2.0 kernel_ms=1.0\n")
            for rank in range(8):
                (scratch/f"test-rank-{rank}.log").write_text(text.format(rank=rank))
            report=q4_profile_report.collect(scratch,"test")
            self.assertEqual(report["route_totals"]["pairs"],40)
            self.assertEqual(report["route_totals"]["tile_occupancy"],40/256)
            self.assertIn("not all expert GPU work",report["coverage"])
            self.assertEqual(report["worker_rpc_timings"]["0:1:N128"]["send_ms"],4)
            for prefix,value in (("test","a"*64),("other","b"*64)):
                (scratch/f"{prefix}-131072.json").write_text(
                    json.dumps({"request_sha256":value,"unicode":"\u200d"},ensure_ascii=False),
                    encoding="utf-8")
            with patch.object(sys,"argv",["profile","--directory",str(scratch),
                    "--baseline","test","--candidate","other","--output",str(scratch/"out.json")]):
                with self.assertRaisesRegex(ValueError,"request body hashes"):
                    q4_profile_report.main()
            (scratch/"test-rank-0.log").write_text(text.format(rank=0).replace("gpu_ms=1.0","gpu_ms=nan"))
            with self.assertRaises(ValueError):
                q4_profile_report.collect(scratch,"test")
        finally:
            shutil.rmtree(scratch)

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

    def test_new_performance_schemas_refuse_historical_gates(self):
        b, r = self.fixture()
        for changed in ({**r, "schema": 2}, {**r, "pairs": []},
                        {**r, "concurrency": {"schema": 2, "streams": []}}):
            result = evaluate(b, changed, {"passed": True})
            self.assertFalse(result["passed"])
            self.assertIsNone(result["threshold_percent"])
            self.assertIn("ABConfig", result["diagnostic"])

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

    def test_frozen_teacher_prefixes_and_utf8(self):
        raw = bytearray(b"FGCONT01")
        pieces = (b"a", b"\xe4", b"\xb8\xad", b"b")
        for prompt in range(20):
            for step, piece in enumerate(pieces):
                raw.extend(struct.pack("<IIII", prompt, step, 100+step, len(piece)))
                raw.extend(piece)
        contexts, positions = q4_make_contexts.expand(bytes(raw), 3)
        self.assertEqual(len(contexts), 60)
        self.assertEqual([p["continuation_step"] for p in positions[:3]], [0, 1, 3])
        self.assertTrue(contexts[2][1].endswith("a中"))
        with self.assertRaises(ValueError):
            q4_make_contexts.expand(bytes(raw[:-1]), 3)
        with self.assertRaises(ValueError):
            q4_make_contexts.expand(bytes(raw), 4)


if __name__ == "__main__":
    unittest.main()
