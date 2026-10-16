"""Check the stdin transport without SSH or credential files."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import uuid


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 is required")
class FleetTransport(unittest.TestCase):
    def run_fixture(self, target="192.168.42.42", valid=True):
        root = Path(__file__).resolve().parents[1]
        build = root / ".build"
        build.mkdir(exist_ok=True)
        fixture = build / ("fleet-fixture-" + uuid.uuid4().hex + ".ps1")
        text = """param($ScriptPath,$Targets,[switch]$PassThru)
$envPath = Join-Path $PSScriptRoot ".env"
$scriptB64 = 'ZWNobyBvaw=='
$remoteEnv = ''
if ($false) {
            $remote = "tmp=fake sudoPwB64"
} else {
            $remote = "tmp=fake"
}
@{IP=$Targets[0];ExitCode=0;Stdout=$remote;EnvPath=$envPath} | ConvertTo-Json -Compress
"""
        if not valid:
            text = text.replace('            $remote = "tmp=fake"\n', "")
        fixture.write_text(text)
        try:
            return subprocess.run(
                ["pwsh", "-NoProfile", "-File", str(root / "tools" / "ab-fleet.ps1"),
                 "-ScriptPath", str(fixture), "-Targets", target, "-PassThru",
                 "-SourceFleetScript", str(fixture)],
                capture_output=True, text=True, timeout=20,
            ), build
        finally:
            fixture.unlink(missing_ok=True)

    def test_preserves_runner_arguments_and_credential_location(self):
        result, build = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["IP"], "192.168.42.42")
        self.assertEqual(Path(data["EnvPath"]), build / ".env")
        self.assertIn("base64 -d", data["Stdout"])
        self.assertIn("bash -s", data["Stdout"])
        self.assertNotIn("tmp=", data["Stdout"])

    def test_upstream_layout_change_fails_closed(self):
        result, _ = self.run_fixture(valid=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported fleet command layout", result.stderr)

    def test_forbidden_owner_rejected_before_execution(self):
        result, _ = self.run_fixture(target="192.168.42.50")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
