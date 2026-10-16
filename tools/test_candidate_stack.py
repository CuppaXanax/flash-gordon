"""Offline-only stack planning, mock orchestration and archive safety tests."""
import io
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import unittest
import uuid

from ab_stack_archive import extract


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 required")
class CandidateStack(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.work = self.root / ".build" / ("stack-fixture-" + uuid.uuid4().hex)
        self.work.mkdir(parents=True)
        self.events = self.work / "events"
        self.lease = self.work / "lease.json"
        self.lease.write_text(json.dumps({
            "multisession_released": True, "mtp_released": True,
            "targets": [f"192.168.42.{i}" for i in range(42, 50)],
            "expires_at": "2099-01-01T00:00:00Z",
        }))
        arm = dict(release="/home/xander/flash-gordon-live/20261016-abq-stack-1",
                   manifest="/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm",
                   binary_sha256="a" * 64, manifest_sha256="b" * 64,
                   capture_program="/home/xander/flash-gordon-live/20261016-abq-stack-1/q4-eval-batch",
                   capture_sha256="c" * 64, shader_tree_sha256="f" * 64, sessions=1, wide=0)
        self.config = dict(A=dict(arm, label="teacher"),
                           corpus=str(self.root / "tools" / "corpora" / "stack-480.json"),
                           candidates=[
                               dict(name="combined", B=dict(arm, label="combined", flags={
                                   "FG_ROOFLINE": 2047, "FG_ROOF_DENSE": 19,
                                   "FG_NGRAM_LOCAL_CACHE": 1})),
                               dict(name="reject", B=dict(arm, label="reject")),
                           ])
        self.capture = self.work / "capture.ps1"
        self.capture.write_text(
            "param($Deployment,$Corpus,$OutputDir,$LeaseFile)\n"
            f"Add-Content '{self.events}' ('capture:'+$Deployment.label)\n"
            "if($Deployment.label-eq'explode'){throw 'mock capture failure'}\n"
            "$null=New-Item -ItemType Directory $OutputDir\n"
            "@{shader_tree_sha256=('f'*64)}|ConvertTo-Json|Set-Content (Join-Path $OutputDir 'capture-identity.json')\n")
        self.deploy = self.work / "deploy.ps1"
        self.deploy.write_text(
            "param($Deployment,$Arm)\n"
            f"Add-Content '{self.events}' ('deploy:'+$Arm)\n")
        self.quality = self.work / "quality.py"
        self.quality.write_text(
            "import argparse,json\nfrom pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "for key in ('corpus','teacher','candidate','output'):p.add_argument('--'+key)\n"
            "a=p.parse_args()\nname=Path(a.output).parent.name\n"
            f"with Path({str(self.events)!r}).open('a') as f:f.write('quality:'+name+'\\n')\n"
            "Path(a.output).write_text(json.dumps(dict(passed=name!='reject',"
            "samples=480,full_vocab=248320,mean_kld=0,p99_kld=0,top1_agreement=1,"
            "corpus_sha256='d'*64)))\n"
            "raise SystemExit(1 if name=='reject' else 0)\n")
        self.perf = self.work / "perf.ps1"
        self.perf.write_text(
            "param($ABConfig,$ABRounds,$ABOrder,$ABOutput)\n"
            "$c=Get-Content $ABConfig -Raw|ConvertFrom-Json -AsHashtable\n"
            "if($ABRounds-ne6-or$ABOrder-ne'ABBA'){throw 'incorrect design'}\n"
            f"Add-Content '{self.events}' ('perf:'+(Split-Path (Split-Path $ABOutput) -Leaf))\n"
            "& $c.deploy_script -Deployment $c.B -Arm B\n"
            "$null=New-Item -ItemType Directory $ABOutput\n"
            "@{complete=$true;eligible=$true;same_kernel_hash_identity=$true;config=$c;quality_pass=$true}"
            "|ConvertTo-Json -Depth 30|Set-Content (Join-Path $ABOutput 'ab-result.json')\n")

    def tearDown(self):
        shutil.rmtree(self.work)

    def run_stack(self, execute=False, lease=True):
        config = self.work / "config.json"
        config.write_text(json.dumps(self.config))
        args = ["pwsh", "-NoProfile", "-File", str(self.root / "tools" / "candidate-stack.ps1"),
                "-ConfigPath", str(config), "-Output", str(self.work / "result"),
                "-CaptureScript", str(self.capture), "-PerformanceScript", str(self.perf),
                "-QualityScript", str(self.quality), "-DeploymentScript", str(self.deploy)]
        if execute:
            args += ["-Execute"]
        if lease:
            args += ["-LeaseFile", str(self.lease)]
        return subprocess.run(args, capture_output=True, text=True, timeout=40)

    def test_plan_is_offline(self):
        result = self.run_stack()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.events.exists())
        self.assertFalse(json.loads((self.work / "result" / "stack-result.json").read_text())["executed"])

    def test_execution_requires_lease(self):
        result = self.run_stack(execute=True, lease=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.events.exists())

    def test_expired_lease_rejected(self):
        lease = json.loads(self.lease.read_text())
        lease["expires_at"] = "2000-01-01T00:00:00Z"
        self.lease.write_text(json.dumps(lease))
        self.assertNotEqual(self.run_stack(execute=True).returncode, 0)
        self.assertFalse(self.events.exists())

    def test_flag_injection_rejected(self):
        self.config["candidates"][0]["B"]["flags"]["LD_PRELOAD"] = "1"
        self.assertNotEqual(self.run_stack(execute=True).returncode, 0)
        self.assertFalse(self.events.exists())

    def test_shell_path_injection_rejected(self):
        self.config["A"]["release"] += "';false"
        self.assertNotEqual(self.run_stack(execute=True).returncode, 0)
        self.assertFalse(self.events.exists())

    def test_host_group_requires_same_kernel_control(self):
        self.config["candidates"][0]["B"]["flags"]["FG_DECODE_SUBMIT_GROUP"] = 2
        result = self.run_stack()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash_control", result.stderr)
        self.assertFalse(self.events.exists())

    def test_host_group_carries_exact_hash_reference(self):
        control = self.config["candidates"][0]
        grouped = json.loads(json.dumps(control))
        grouped.update(name="grouped", hash_control="combined")
        grouped["B"]["label"] = "grouped"
        grouped["B"]["flags"]["FG_DECODE_SUBMIT_GROUP"] = 2
        self.config["candidates"] = [control, grouped]
        result = self.run_stack(execute=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        ab = json.loads((self.work / "result" / "grouped" / "ab-config.json").read_text())
        self.assertEqual(Path(ab["exact_hash_reference"]),
                         self.work / "result" / "combined" / "perf" / "ab-result.json")
        rows = json.loads((self.work / "result" / "stack-result.json").read_text())["results"]
        self.assertTrue(rows[1]["same_kernel_hash_identity"])

    def test_one_quality_and_performance_per_combination(self):
        result = self.run_stack(execute=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events.read_text().splitlines(), [
            "capture:teacher", "capture:combined", "quality:combined", "perf:combined",
            "deploy:B", "capture:reject", "quality:reject", "deploy:restore-baseline"])
        report = json.loads((self.work / "result" / "stack-result.json").read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(report["restored_baseline"])
        self.assertTrue(report["results"][0]["eligible"])
        self.assertFalse(report["results"][1]["perf_run"])
        receipt = json.loads((self.work / "result" / "combined" / "quality-receipt.json").read_text())
        self.assertNotEqual(receipt["A_identity"], receipt["B_identity"])

    def test_capture_failure_restores_baseline(self):
        self.config["candidates"][0]["B"]["label"] = "explode"
        result = self.run_stack(execute=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.events.read_text().splitlines()[-1], "deploy:restore-baseline")

    def test_numeric_receipt_binds_flags_corpus_and_report(self):
        quality = self.work / "numeric-quality.json"
        receipt = self.work / "receipt.json"
        config = self.work / "numeric-config.json"
        restore = self.work / "restore.ps1"
        restore.write_text(f"Add-Content '{self.events}' restore\n")
        self.deploy.write_text("param($Deployment,$Arm)\nthrow 'QUALITY_PREFLIGHT_ACCEPTED'\n")
        design = dict(A=self.config["A"], B=self.config["candidates"][0]["B"],
                      change_kind="numeric", quality_file=str(receipt),
                      deploy_script=str(self.deploy), restore_script=str(restore))
        report = dict(passed=True, samples=480, full_vocab=248320, mean_kld=0,
                      p99_kld=0, top1_agreement=1, corpus_sha256="d" * 64)
        for change in ("none", "flags", "corpus", "report", "shaders"):
            with self.subTest(change=change):
                self.events.unlink(missing_ok=True)
                quality.write_text(json.dumps(report))
                config.write_text(json.dumps(design))
                command = (
                    f". '{self.root / 'tools' / 'ab-stack-common.ps1'}';"
                    f"$c=Get-Content '{config}' -Raw|ConvertFrom-Json -AsHashtable;"
                    "@{A_identity=(Get-ABDeploymentIdentity $c.A);"
                    "B_identity=(Get-ABDeploymentIdentity $c.B);"
                    f"manifest_sha256=$c.B.manifest_sha256;quality_path='{quality}';"
                    f"quality_sha256='{hashlib.sha256(quality.read_bytes()).hexdigest()}';"
                    f"corpus_sha256='{report['corpus_sha256']}'"
                    f"}}|ConvertTo-Json|Set-Content '{receipt}'"
                )
                result = subprocess.run(["pwsh", "-NoProfile", "-Command", command],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                altered = json.loads(config.read_text())
                if change == "flags":
                    altered["B"]["flags"]["FG_NGRAM_LOCAL_CACHE"] = 0
                elif change == "shaders":
                    altered["B"]["shader_tree_sha256"] = "e" * 64
                elif change == "corpus":
                    bound = json.loads(receipt.read_text())
                    bound["corpus_sha256"] = "e" * 64
                    receipt.write_text(json.dumps(bound))
                elif change == "report":
                    quality.write_text(json.dumps(dict(report, mean_kld=0.001)))
                config.write_text(json.dumps(altered))
                result = subprocess.run([
                    "pwsh", "-NoProfile", "-File", str(self.root / "tools" / "hillclimb.ps1"),
                    "-BaseUrl", "http://127.0.0.1:9/v1", "-ABConfig", str(config),
                    "-ABOutput", str(self.work / f"numeric-{change}")
                ], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                if change == "none":
                    self.assertIn("QUALITY_PREFLIGHT_ACCEPTED", result.stderr)
                    self.assertEqual(self.events.read_text().splitlines(), ["restore"])
                else:
                    self.assertNotIn("QUALITY_PREFLIGHT_ACCEPTED", result.stderr)
                    self.assertFalse(self.events.exists())

    def test_archive_rejects_links_paths_and_duplicates(self):
        for kind in ("link", "path", "duplicate"):
            archive = self.work / f"{kind}.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                info = tarfile.TarInfo("../bad" if kind == "path" else "0000.fgl")
                if kind == "link":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "bad"
                    target.addfile(info)
                else:
                    info.size = 12 + 248320 * 4
                    target.addfile(info, io.BytesIO(bytes(info.size)))
                    if kind == "duplicate":
                        target.addfile(info, io.BytesIO(bytes(info.size)))
            output = self.work / kind
            output.mkdir()
            with self.assertRaises(ValueError):
                extract(archive, output, 1)
