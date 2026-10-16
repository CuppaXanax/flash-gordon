"""Local fake-fleet tests: never execute SSH or contact a production address."""
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import tarfile
import threading
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.respond({"status": "ok"} if self.path == "/health" else
                     {"data": [{"id": "Qwen3.8-Flash-Next-Uncensored"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.respond({"choices": [{"message": {"content": "OK"}}]})


MOCK = r'''
param($ScriptPath,[string[]]$Targets,$UploadPath,$UploadDestination,
      $DownloadSource,$DownloadPath,$CommandTimeout,[switch]$PassThru)
$ErrorActionPreference='Stop'
$name=[IO.Path]::GetFileName($ScriptPath)
if($UploadDestination -and $UploadDestination.EndsWith('/lease.sh')){
    if([IO.File]::ReadAllText($UploadPath).Contains("`r")){throw 'Linux guard contains CRLF'}
}
if($DownloadSource){
    $source=if($DownloadSource.EndsWith('/cases.tsv')){'cases.tsv'}
      elseif($DownloadSource.EndsWith('/proposals.tar')){'proposals.tar'}
      elseif($DownloadSource.EndsWith('/greedy.jsonl')){'greedy.jsonl'}
      elseif($DownloadSource.EndsWith('/final-evidence.tar.gz')){'evidence.tar.gz'}
      else {'baseline.tar'}
    Copy-Item (Join-Path $env:MTP_CAPTURE_FIXTURES $source) $DownloadPath
}
foreach($target in $Targets){
    $failed=$env:MTP_CAPTURE_TEST_FAIL -eq '1' -and $name.Contains('run-real-HC-native-drafts')
    $text=if($name.Contains('preflight-production')){[string][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()}
      elseif($name.Contains('baseline-status')){'baseline-ready'}
      elseif($name.Contains('greedy-status')){'qualification-passed'}else{'mock-only'}
    [pscustomobject]@{IP=$target;ExitCode=$(if($failed){9}else{0});Stdout=$text;Stderr=''}
}
'''


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 unavailable")
class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / ".build" / ("lease-test-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        for name in ("release.tar.gz", "corpus.tar"):
            (self.directory / name).write_bytes(b"fake-fleet fixture, not a deployment")
        prepared = dict(files={
            "release.tar.gz": hashlib.sha256((self.directory / "release.tar.gz").read_bytes()).hexdigest(),
            "native-q8.gguf": "536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b"},
            corpus_sha256=hashlib.sha256((self.directory / "corpus.tar").read_bytes()).hexdigest())
        (self.directory / "prepared.json").write_text(json.dumps(prepared))
        (self.directory / "fake-fleet.ps1").write_text(MOCK)
        lines = []
        rows = []
        receipts = []
        for category in ("chat", "code"):
            for i in range(20):
                case = f"{category}-{i:02d}"
                lines.append(f"{case}\t{category}\t{i+100}\t100\t259\t7\n")
                for p in range(256):
                    rows.append(dict(case=case, category=category, position=p,
                        draft=[1, 2, 3], target=[1, 2, 3], match=[1, 1, 1], prefix=[1, 1, 1]))
                if i < 4:
                    for k in (1, 2, 3):
                        receipts.append(dict(case=case, k=k, equal=True, tokens=256,
                            rollbacks=1, verifier="causal-N1-reference"))
        (self.directory / "cases.tsv").write_text("".join(lines))
        (self.directory / "acceptance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.directory / "acceptance-summary.json").write_text("{}")
        (self.directory / "greedy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in receipts))
        for name, mode in (("proposals.tar", "w"), ("baseline.tar", "w"), ("evidence.tar.gz", "w:gz")):
            with tarfile.open(self.directory / name, mode) as archive:
                if name == "proposals.tar":
                    for file in ("acceptance.jsonl", "acceptance-summary.json"):
                        archive.add(self.directory / file, arcname=file)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        shutil.rmtree(self.directory)

    def run_controller(self, fail):
        env = dict(os.environ, MTP_CAPTURE_FIXTURES=str(self.directory),
                   MTP_CAPTURE_TEST_FAIL="1" if fail else "0")
        return subprocess.run(["pwsh", "-NoProfile", "-File",
            str(ROOT / "tools" / "Invoke-MTPQualification.ps1"),
            "-Action", "Run", "-LeaseConfirmed", "-LeaseId", "mock-lease",
            "-Artifacts", str(self.directory),
            "-ProductionRunner", str(self.directory / "fake-fleet.ps1"),
            "-DraftRunner", str(self.directory / "fake-fleet.ps1"),
            "-ApiBaseUrl", f"http://127.0.0.1:{self.server.server_port}"],
            env=env, cwd=ROOT, capture_output=True, text=True, timeout=150)

    def test_complete_fake_lease(self):
        completed = self.run_controller(False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        run = self.directory / "runs" / "mock-lease"
        result = json.loads((run / "analysis" / "acceptance-summary.json").read_text())
        self.assertTrue(result["greedy"]["passed"])
        self.assertTrue((run / "restored-api.json").is_file())
        self.assertTrue(list(run.glob("*remove-owned-ring-scratch.json")))
        self.assertTrue(list(run.glob("*remove-owned-draft-scratch.json")))

    def test_failure_still_restores(self):
        completed = self.run_controller(True)
        self.assertNotEqual(completed.returncode, 0)
        run = self.directory / "runs" / "mock-lease"
        self.assertTrue((run / "restored-api.json").is_file(), completed.stdout + completed.stderr)
        self.assertFalse(json.loads((run / "outcome.json").read_text(encoding="utf-8-sig"))["complete"])
        self.assertTrue(list(run.glob("*cancel-watchdogs.json")))
        self.assertTrue(list(run.glob("*remove-owned-ring-scratch.json")))

    def test_plan_is_local_only(self):
        completed = subprocess.run(["pwsh", "-NoProfile", "-File",
            str(ROOT / "tools" / "Invoke-MTPQualification.ps1"), "-Action", "Plan",
            "-Artifacts", str(self.directory)], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(json.loads(completed.stdout)["productionAccess"])
        self.assertFalse((self.directory / "runs").exists())

    def test_run_requires_exclusive_lease(self):
        completed = subprocess.run(["pwsh", "-NoProfile", "-File",
            str(ROOT / "tools" / "Invoke-MTPQualification.ps1"), "-Action", "Run",
            "-Artifacts", str(self.directory)], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("confirmed exclusive lease", completed.stderr)
        self.assertFalse((self.directory / "runs").exists())


if __name__ == "__main__":
    unittest.main()
