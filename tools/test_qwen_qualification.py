import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_native_qualification import collect
from validate_qwen_qualification import validate


class FakeQwen(BaseHTTPRequestHandler):
    requests = []
    counter = 0

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.requests.append(body)
        self.__class__.counter += 1
        chars = sum(len(str(m.get("content") or "")) for m in body["messages"])
        prompt = max(1, chars // 4)
        identifier = "chatcmpl-fake-%d" % self.__class__.counter
        events = [
            {"id": identifier, "choices": [{"delta": {"role": "assistant"},
                                                "finish_reason": None}]},
            {"id": identifier, "choices": [{"delta": {"content": "ok"},
                                                "finish_reason": None}]},
            {"id": identifier, "choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                       "total_tokens": prompt + 1}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for index, event in enumerate(events):
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
            if index == 0:
                time.sleep(.025)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args):
        pass


class QualificationTests(unittest.TestCase):
    def test_streaming_driver_is_append_only_and_adaptive(self):
        FakeQwen.requests = []
        FakeQwen.counter = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeQwen)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus = root / "corpus.txt"
                corpus.write_text("abcdefghijklmnopqrstuvwxyz" * 20, encoding="utf-8")
                output = root / "client.jsonl"
                args = mock.Mock(base_url="http://127.0.0.1:%d/v1" % server.server_port,
                                 model="fake", corpus=str(corpus), output=str(output),
                                 wall_budget_minutes=1, checkpoints="4,16", repeats=2,
                                 request_timeout_seconds=10,
                                 chunk_chars=8, max_chunk_chars=32, chunk_safety=.8,
                                 max_tokens=1, native_context=262144, headroom=256,
                                 temperature=1.0, top_p=.95, top_k=20)
                self.assertEqual(collect(args), 0)
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertTrue(all(r["client_ttft_ms"] is not None for r in records))
                self.assertTrue(all(r["client_ttft_ms"] >= 20 for r in records))
                ranges = [(r["corpus_char_start"], r["corpus_char_end"])
                          for r in records if r["corpus_char_end"] > r["corpus_char_start"]]
                self.assertEqual(ranges, sorted(set(ranges)))
                self.assertGreaterEqual(len(FakeQwen.requests), 3)
                for request in FakeQwen.requests:
                    self.assertEqual((request["temperature"], request["top_p"], request["top_k"]),
                                     (1.0, .95, 20))
        finally:
            server.shutdown()
            server.server_close()

    def test_validator_enforces_sampling_join_and_curve(self):
        client, server = [], []
        for index, target in enumerate((4096, 8192, 16384, 32768, 65536,
                                        131072, 196608, 261888)):
            for repeat in range(3):
                identifier = "r-%d-%d" % (index, repeat)
                client.append({"schema": "fg.qwen.qual.client.v1", "request_id": identifier,
                               "checkpoint_target": target, "prompt_tokens": target,
                               "checkpoint_crossed": True,
                               "http_status": 200, "finish_reason": "stop",
                               "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                               "corpus_char_start": index * 1000,
                               "corpus_char_end": (index + 1) * 1000})
                server.append({"schema": "fg.qwen.qual.server.v1", "request_id": identifier,
                               "http_status": 200, "status": "ok",
                               "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                               "prompt_tokens": target, "reused_tokens": target - 100,
                               "prefilled_tokens": 100, "generated_tokens": None,
                               "prefill_tps": 300,
                               "decode_tps": 60})
        report = validate(client, server)
        self.assertFalse(report["pass"], "duplicate ranges/accounting should be rejected")
        self.assertTrue(any("duplicate" in error for error in report["errors"]))
        client[0]["sampling"]["temperature"] = 0
        report = validate(client, server)
        self.assertTrue(any("sampling" in error for error in report["errors"]))

    def test_validator_flags_accounting_and_adjacent_drop(self):
        records, server = [], []
        for index, target in enumerate((4096, 8192, 16384, 32768, 65536, 131072, 196608, 261888)):
            identifier = "drop-%d" % index
            records.append({"schema": "fg.qwen.qual.client.v1", "request_id": identifier,
                            "checkpoint_target": target, "prompt_tokens": target,
                            "checkpoint_crossed": True,
                            "http_status": 200, "finish_reason": "stop",
                            "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                            "corpus_char_start": index * 100, "corpus_char_end": (index + 1) * 100})
            server.append({"schema": "fg.qwen.qual.server.v1", "request_id": identifier,
                           "http_status": 200, "status": "ok",
                           "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                           "prompt_tokens": target, "reused_tokens": target - 100,
                           "prefilled_tokens": 100, "generated_tokens": None,
                           "prefill_tps": 300,
                           "decode_tps": 60 if index < 2 else 30})
        server[0]["reused_tokens"] = 1
        report = validate(records, server)
        self.assertFalse(report["pass"])
        self.assertTrue(any("accounting" in error or "drop" in error for error in report["errors"]))

    def test_validator_accepts_aspirational_curve_and_rejects_partial_curve(self):
        client, server = [], []
        for index, target in enumerate((4096, 8192, 16384, 32768, 65536,
                                        131072, 196608, 261888)):
            for repeat in range(3):
                identifier = "good-%d-%d" % (index, repeat)
                start = (index * 3 + repeat) * 100
                client.append({"schema": "fg.qwen.qual.client.v1", "request_id": identifier,
                               "checkpoint_target": target, "prompt_tokens": target,
                               "checkpoint_crossed": True,
                               "http_status": 200, "finish_reason": "stop",
                               "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                               "corpus_char_start": start, "corpus_char_end": start + 100})
                server.append({"schema": "fg.qwen.qual.server.v1", "request_id": identifier,
                               "http_status": 200, "status": "ok",
                               "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                               "prompt_tokens": target, "reused_tokens": target - 100,
                               "prefilled_tokens": 100, "generated_tokens": None,
                               "prefill_tps": 700,
                               "decode_tps": 120})
        report = validate(client, server)
        self.assertTrue(report["pass"], report["errors"])
        self.assertEqual(report["policy"]["aspirational_prefill_tps"], 600.0)
        self.assertEqual(report["policy"]["aspirational_decode_tps"], 100.0)
        partial = validate(client[:3], server[:3])
        self.assertFalse(partial["pass"])
        self.assertTrue(any("coverage" in error for error in partial["errors"]))

    def test_validator_rejects_cold_reset_and_non_monotonic_context(self):
        client = []
        server = []
        for index, context in enumerate((4096, 8192)):
            identifier = "reset-%d" % index
            client.append({"schema": "fg.qwen.qual.client.v1", "request_id": identifier,
                           "checkpoint_target": context, "prompt_tokens": context,
                           "checkpoint_crossed": True,
                           "http_status": 200, "finish_reason": "stop",
                           "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                           "corpus_char_start": index * 100, "corpus_char_end": (index + 1) * 100})
            server.append({"schema": "fg.qwen.qual.server.v1", "request_id": identifier,
                           "http_status": 200, "status": "ok",
                           "sampling": {"temperature": 1, "top_p": .95, "top_k": 20},
                           "prompt_tokens": context, "reused_tokens": 0 if index else 100,
                           "prefilled_tokens": context if index else context - 100,
                           "generated_tokens": None,
                           "prefill_tps": 300, "decode_tps": 60,
                           "reset_reason": "cold-start" if index else "none"})
        client[1]["prompt_tokens"] = 2048
        report = validate(client, server)
        self.assertTrue(any("non-monotonic" in error for error in report["errors"]))
        self.assertTrue(any("cold prefix" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
