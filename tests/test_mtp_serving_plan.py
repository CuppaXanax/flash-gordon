import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 unavailable")
class ServingPlanTest(unittest.TestCase):
    def plan(self, *args):
        script=Path(__file__).parents[1]/"tools"/"Invoke-MTPServingQualification.ps1"
        return subprocess.run(["pwsh","-NoProfile","-File",str(script),"-Action","Plan",*args],
                              capture_output=True,text=True)

    def test_default_explicit_cache(self):
        result=self.plan()
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertFalse(plan["productionAccess"])
        self.assertEqual(plan["candidateSessions"],2)
        self.assertEqual(plan["candidateQsaPageCacheMiB"],640)

    def test_single_session_subset_plan(self):
        result=self.plan("-Profile","SingleSessionPerf","-LocalHead","not-opened-in-plan.fgdv")
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertFalse(plan["productionAccess"])
        self.assertTrue(plan["localDraftHead"])
        self.assertEqual(plan["candidateQsaPageCacheMiB"],440)

    def test_dual_head_stays_fail_closed(self):
        result=self.plan("-LocalHead","not-opened-in-plan.fgdv")
        self.assertNotEqual(result.returncode,0)
        self.assertIn("dual-session residency is unqualified",result.stderr)
