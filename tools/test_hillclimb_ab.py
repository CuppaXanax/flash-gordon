"""Local API fixture for paired ordering, byte identity, rollback and token accounting."""
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        data = b'{"data":[{"id":"fixture"}]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        Handler.bodies.append(raw)
        body = json.loads(raw)
        count = min(body["max_tokens"], 64)
        usage = {"prompt_tokens": 4100, "completion_tokens": count}
        time.sleep(0.005)
        if body["stream"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
            self.wfile.flush()
            time.sleep(0.005)
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"ready"}}]}\n\ndata: [DONE]\n\n')
            return
        data = json.dumps({"usage": usage, "choices": [
            {"message": {"content": "ready"}, "finish_reason": "length"}
        ]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        for key, value in {
            "Reused-Tokens": 4100 if len(body["messages"]) > 2 else 0,
            "Prefill-TPS": 350, "Prefill-Seconds": 0.001,
            "Decode-TPS": 25, "Decode-Seconds": 0.003,
        }.items():
            self.send_header("X-Flash-Gordon-" + key, str(value))
        self.end_headers()
        self.wfile.write(data)


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 required")
class PairedHarness(unittest.TestCase):
    def test_paired_statistics(self):
        root = Path(__file__).resolve().parents[1]
        module = root / "tools" / "hillclimb-ab.ps1"
        script = (
            f". '{module}'; "
            "$pairs=@(0..2|ForEach-Object{@{"
            "A=@{metrics=@{speed=@{value=100;higher_is_better=$true};"
            "latency=@{value=100;higher_is_better=$false}}};"
            "B=@{metrics=@{speed=@{value=110;higher_is_better=$true};"
            "latency=@{value=110;higher_is_better=$false}}}}});"
            "for($i=0;$i -lt 3;$i++){"
            "$pairs[$i].A.metrics.noise=@{value=100;higher_is_better=$true};"
            "$pairs[$i].B.metrics.noise=@{value=@(101,103,120)[$i];higher_is_better=$true}};"
            "Get-ABSummary $pairs -Replicates 1000 | ConvertTo-Json -Depth 8"
        )
        result = subprocess.run(["pwsh", "-NoProfile", "-Command", script],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = {row["metric"]: row for row in json.loads(result.stdout)}
        self.assertTrue(rows["speed"]["noninferior"])
        self.assertTrue(rows["speed"]["win"])
        self.assertAlmostEqual(rows["speed"]["B_over_A"], 1.1)
        self.assertFalse(rows["latency"]["noninferior"])
        self.assertEqual(rows["latency"]["status"], "regression")
        self.assertGreater(rows["noise"]["bootstrap_ci95"][0], 1)
        self.assertLess(rows["noise"]["ci95"][0], 0.97)
        self.assertFalse(rows["noise"]["noninferior"])

    def run_mode(self, fail=False):
        root = Path(__file__).resolve().parents[1]
        output = root / ".build" / ("ab-fixture-" + uuid.uuid4().hex)
        output.mkdir(parents=True)
        deploy = output / "deploy.ps1"
        restore = output / "restore.ps1"
        events = output / "events.txt"
        failure = (
            f"if (@(Get-Content -LiteralPath '{events}').Count -eq 3) {{ throw 'fixture deployment failed' }}\n"
            if fail == "after-pair" else "if ($Arm -eq 'B') { throw 'fixture deployment failed' }\n"
        ) if fail else ""
        deploy.write_text(
            "param($Deployment,$Arm)\n"
            f"Add-Content -LiteralPath '{events}' -Value $Arm\n" + failure,
            encoding="utf-8")
        restore.write_text(f"Add-Content -LiteralPath '{events}' -Value restore\n", encoding="utf-8")
        config = output / "config.json"
        config.write_text(json.dumps({
            "A": {}, "B": {}, "deploy_script": str(deploy), "restore_script": str(restore),
            "change_kind": "exact", "contexts": [4096]
        }), encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        Handler.bodies = []
        try:
            command = [
                "pwsh", "-NoProfile", "-File", str(root / "tools" / "hillclimb.ps1"),
                "-BaseUrl", f"http://127.0.0.1:{server.server_port}/v1",
                "-ABConfig", str(config), "-ABOutput", str(output / "result"), "-ABRounds", "3"
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            data = json.loads((output / "result" / "ab-result.json").read_text(encoding="utf-8-sig"))
            self.assertTrue(data["restored_original"])
            if fail:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(data["complete"])
                self.assertFalse(data["eligible"])
                self.assertIn("fixture deployment failed", data["error"])
                if fail != "after-pair":
                    self.assertEqual(events.read_text().splitlines(), ["A", "B", "restore"])
                    return
                self.assertEqual(len(data["pairs"]), 1)
                first_salt = data["pairs"][0]["salt"]
                deploy.write_text(
                    "param($Deployment,$Arm)\n"
                    f"Add-Content -LiteralPath '{events}' -Value $Arm\n", encoding="utf-8")
                result = subprocess.run(command + ["-ABResume"], capture_output=True, text=True, timeout=180)
                data = json.loads((output / "result" / "ab-result.json").read_text(encoding="utf-8-sig"))
                self.assertEqual(data["pairs"][0]["salt"], first_salt)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(data["complete"])
            self.assertTrue(data["hash_identity"])
            expected = ["A", "B", "B", "restore", "B", "A", "A", "B", "restore"] if fail else [
                "A", "B", "B", "A", "A", "B", "restore"]
            self.assertEqual(events.read_text().splitlines(), expected)
            self.assertEqual(len(data["pairs"]), 3)
            for pair in data["pairs"]:
                self.assertEqual(pair["A"]["prefill"][0]["request_sha256"],
                                 pair["B"]["prefill"][0]["request_sha256"])
                self.assertEqual(pair["A"]["concurrency"]["common_tokens"], 64)
            half = len(Handler.bodies) // 6
            self.assertGreater(half, 0)
            self.assertEqual(sorted(Handler.bodies[:half]), sorted(Handler.bodies[half:2 * half]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            shutil.rmtree(output)

    def test_order_bodies_and_restore(self):
        self.run_mode()

    def test_failed_deploy_restores_original(self):
        self.run_mode(fail=True)

    def test_resume_keeps_complete_pairs(self):
        self.run_mode(fail="after-pair")


if __name__ == "__main__":
    unittest.main()
