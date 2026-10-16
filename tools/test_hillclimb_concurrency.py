"""Exercise the real PowerShell harness against local simultaneous SSE streams."""
import hashlib
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
    short_loaded_chat = False
    omit_stream_usage = False
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
        with self.lock:
            if (Handler.short_loaded_chat and Handler.active
                    and body.get("stream") and body.get("user") == "hillclimb-chat"):
                count = 1
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

            request_id = "chatcmpl-local-" + uuid.uuid4().hex
            event({"id": request_id, "choices": [{"delta": {"role": "assistant"}}]})
            if body.get("user") == "hillclimb-agent" and not reused:
                time.sleep(0.3)
            for _ in range(count):
                time.sleep(0.005)
                event({"choices": [{"delta": {"content": "task "}}]})
            terminal = {"id": request_id, "choices": [{"delta": {}, "finish_reason": "length"}]}
            if not Handler.omit_stream_usage:
                terminal.update(usage=usage, flash_gordon={
                    "reused_tokens": reused, "prefilled_tokens": 96 - reused})
            event(terminal)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with self.lock:
                Handler.active -= 1


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is required")
class ConcurrentHarness(unittest.TestCase):
    def run_mode(self, mixed, capacity=False, short=False, missing_usage=False):
        Handler.peak = 0
        Handler.agent_sse = 0
        Handler.short_loaded_chat = short
        Handler.omit_stream_usage = missing_usage
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
        if missing_usage:
            command.extend(["-AllowSerializedBaseline", "-PromptSalt", "fixture-paired-salt"])
        artifact = None
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=45)
            match = re.search(r"HILLCLIMB result=(.*?) model=", result.stdout)
            if match:
                artifact = pathlib.Path(match[1])
            if short:
                self.assertNotEqual(result.returncode, 0)
                artifact = next(output.glob("*.json"))
                data = json.loads(artifact.read_text(encoding="utf-8-sig"))
                self.assertFalse(data["concurrency"]["complete"])
                load = data["concurrency"]["trials"][0]["load"]
                self.assertEqual([s["completion_tokens"] for s in load["streams"]], [1, 64])
                self.assertIn("Insufficient sustained generation", str(data["errors"]))
                return
            self.assertEqual(result.returncode, 2 if missing_usage else 0, result.stdout + result.stderr)
            self.assertIsNotNone(artifact, result.stdout)
            data = json.loads(artifact.read_text(encoding="utf-8-sig"))
            load = data["concurrency"]["trials"][0]["load"]
            self.assertGreater(load["batch_wall_seconds"], 0)
            self.assertGreaterEqual(load["batch_wall_seconds"], load["aggregate_window_seconds"])
            for row in load["streams"]:
                self.assertGreater(row["inter_event_p95_ms"], 0)
                self.assertEqual(len(row["visible_event_times_ms"]), row["visible_delta_events"])
            if missing_usage:
                self.assertFalse(data["concurrency"]["complete"])
                self.assertTrue(data["concurrency"]["requires_server_log_accounting"])
                self.assertEqual(data["concurrency"]["prompt_salt"], "fixture-paired-salt")
                self.assertIsNone(load["aggregate_client_decode_tps"])
                self.assertIsNone(load["aggregate_wall_tps"])
                self.assertGreater(load["aggregate_window_seconds"], 0)
                self.assertTrue(all(row["completion_tokens"] is None for row in load["streams"]))
                self.assertTrue(all(row["request_id"].startswith("chatcmpl-local-")
                                    for row in load["streams"]))
                self.assertTrue(all(len(row["visible_event_times_ms"]) == 64
                                    for row in load["streams"]))
                return
            self.assertEqual(Handler.peak, 2)
            self.assertTrue(data["metadata"]["correctness_pass"])
            self.assertEqual([row["completion_tokens"] for row in load["streams"]], [64, 64])
            self.assertAlmostEqual(load["aggregate_wall_tps"], 128 / load["batch_wall_seconds"], places=8)
            if capacity:
                self.assertTrue(load["streams"][0]["same_prompt_hash_match"])
                self.assertIsNone(load["streams"][1]["same_prompt_hash_match"])
                self.assertIsNone(load["aggregate_speedup_over_mean_isolated"])
                self.assertEqual(Handler.agent_sse, 1)
                self.assertGreater(load["post_near_limit_chat"]["completion_tokens"], 0)
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

    def test_short_generation_retains_partial_trial(self):
        self.run_mode(False, short=True)

    def test_baseline_requires_exact_log_accounting(self):
        self.run_mode(False, missing_usage=True)

    def test_paired_salt_is_reproducible_and_counter_is_distinct(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        path = str(root / "tools" / "hillclimb.ps1").replace("'", "''")
        command = (
            "$t=$null;$e=$null;"
            f"$a=[Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$t,[ref]$e);"
            "$f=$a.EndBlock.Statements|Where-Object{"
            "$_ -is [Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -eq 'New-Nonce'};"
            "Invoke-Expression $f.Extent.Text;$PromptSalt='fixture-paired-salt';$script:promptNonce=0;"
            "$x=New-Nonce;$y=New-Nonce;@($x,$y)|ConvertTo-Json -Compress"
        )
        expected = [hashlib.sha256(f"fixture-paired-salt/{i}".encode()).hexdigest()[:32]
                    for i in range(2)]
        for _ in range(2):
            result = subprocess.run(["pwsh", "-NoProfile", "-Command", command],
                                    cwd=root, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)


if __name__ == "__main__":
    unittest.main()
