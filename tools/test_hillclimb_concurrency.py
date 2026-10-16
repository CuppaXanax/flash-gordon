"""Exercise the real PowerShell harness against local simultaneous SSE streams."""
import json
import pathlib
import re
import shutil
import subprocess
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    active = 0
    peak = 0
    agent_sse = 0
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = json.dumps({"data": [{"id": "local-test"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream") and body.get("user") == "hillclimb-agent":
            with self.lock:
                Handler.agent_sse += 1
        count = min(body["max_tokens"], 64)
        reused = 32 if len(body["messages"]) > 2 else 0
        usage = {"prompt_tokens": 96, "completion_tokens": count, "total_tokens": 96 + count}
        text = "task " * count
        if not body.get("stream"):
            data = json.dumps({"usage": usage, "choices": [
                {"message": {"content": text}, "finish_reason": "length"}
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        with self.lock:
            Handler.active += 1
            Handler.peak = max(Handler.peak, Handler.active)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            event({"choices": [{"delta": {"role": "assistant"}}]})
            if body.get("user") == "hillclimb-agent" and not reused:
                time.sleep(0.3)
            for _ in range(count):
                time.sleep(0.005)
                event({"choices": [{"delta": {"content": "task "}}]})
            event({"choices": [{"delta": {}, "finish_reason": "length"}], "usage": usage,
                   "flash_gordon": {"reused_tokens": reused, "prefilled_tokens": 96 - reused}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with self.lock:
                Handler.active -= 1


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is required")
class ConcurrentHarness(unittest.TestCase):
    def run_mode(self, mixed, capacity=False):
        Handler.peak = 0
        Handler.agent_sse = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = pathlib.Path(__file__).resolve().parents[1]
        output = root / ".build" / ("harness-" + uuid.uuid4().hex)
        command = [
            "pwsh", "-NoProfile", "-File", str(root / "tools" / "hillclimb.ps1"),
            "-BaseUrl", f"http://127.0.0.1:{server.server_port}/v1",
            "-ConcurrentGeneration", "-ConcurrentTokens", "64", "-ConcurrentRepeats", "1",
            "-ChatContextTokens", "32", "-AgentContextTokens", "96",
            "-ChatDelayMilliseconds", "20", "-RequestTimeoutSeconds", "20",
            "-ResultsDirectory", str(output),
        ]
        if mixed:
            command.append("-MixedPrefill")
        if capacity:
            command.append("-CapacityProbe")
        artifact = None
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=45)
            match = re.search(r"HILLCLIMB result=(.*?) model=", result.stdout)
            if match:
                artifact = pathlib.Path(match[1])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIsNotNone(artifact, result.stdout)
            data = json.loads(artifact.read_text(encoding="utf-8-sig"))
            load = data["concurrency"]["trials"][0]["load"]
            self.assertEqual(Handler.peak, 2)
            self.assertTrue(data["metadata"]["correctness_pass"])
            self.assertEqual([row["completion_tokens"] for row in load["streams"]], [64, 64])
            if capacity:
                self.assertTrue(load["streams"][0]["same_prompt_hash_match"])
                self.assertIsNone(load["streams"][1]["same_prompt_hash_match"])
                self.assertIsNone(load["aggregate_speedup_over_mean_isolated"])
                self.assertEqual(Handler.agent_sse, 1)
            else:
                self.assertTrue(all(row["same_prompt_hash_match"] for row in load["streams"]))
                self.assertEqual(Handler.agent_sse, 2)
            if mixed:
                self.assertTrue(load["chat_first_token_before_agent_first_token"])
                self.assertGreater(load["streams"][1]["ttft_ms"], 250)
            else:
                self.assertGreater(load["overlapping_visible_decode_seconds"], 0.1)
                self.assertGreater(load["aggregate_speedup_over_mean_isolated"], 1.4)
        finally:
            if artifact:
                artifact.unlink(missing_ok=True)
            if output.exists():
                shutil.rmtree(output)
            server.shutdown()
            server.server_close()
            thread.join()

    def test_warm_decode(self):
        self.run_mode(False)

    def test_chat_during_prefill(self):
        self.run_mode(True)

    def test_single_cold_capacity_probe(self):
        self.run_mode(True, capacity=True)


if __name__ == "__main__":
    unittest.main()
