from pathlib import Path
from unittest import TestCase, mock
import sys


class LeaseIdentityTests(TestCase):
    @classmethod
    def setUpClass(cls):
        script = (Path(__file__).resolve().parents[1] /
                  "tools" / "mtp-lease-guard.sh").read_text()
        cls.code = script.split('python3 - "$pid" "$rank" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        cls.shader = "/home/xander/flash-gordon-live/20261016-abq-stack-adaptive/vulkan"

    def check_identity(self, rank, changes=None, sessions="2"):
        env = dict(FG_SHADER_DIR=self.shader, FG_EXPERT_Q8_WIDE="1",
                   FG_ROOFLINE="2047", FG_ROOF_DENSE="19", FG_ROOF_QSA="3",
                   FG_ROOF_EXPERT="33024", FG_ROOF_HC="96", FG_ROOF_RECURRENT="3",
                   FG_ROOF_GDN="1", FG_NGRAM_LOCAL_CACHE="1", FG_DECODE_SUBMIT_GROUP="2")
        options = ["--manifest", "/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm"]
        if rank == 0:
            env["FG_RING_TRACE"] = "1"
            options += ["--model-name", "Qwen3.8-Flash-Next-Uncensored",
                        "--host", "0.0.0.0", "--port", "8080",
                        "--context-tokens", "262144", "--gpu-index-tokens", "262144",
                        "--qsa-hot-tokens", "0", "--qsa-page-cache-mib", "700",
                        "--sessions", sessions]
        else:
            options += ["--rank", str(rank)]
        env.update(changes or {})
        args = ["flash-gordon", "api" if rank == 0 else "rank"] + options
        def read(path):
            if path.name == "environ":
                return b"\0".join(f"{k}={v}".encode() for k, v in env.items()) + b"\0"
            return b"\0".join(x.encode() for x in args) + b"\0"
        with mock.patch.object(Path, "read_bytes", read), \
                mock.patch.object(sys, "argv", ["identity", "123", str(rank)]):
            exec(compile(self.code, "lease-prod-identity", "exec"), {})

    def test_current_coordinator_and_worker(self):
        self.check_identity(0)
        self.check_identity(7)

    def test_old_and_instrumented_configs_rejected(self):
        for changes in (
            dict(FG_EXPERT_Q8_WIDE="0"), dict(FG_ROOFLINE="0"),
            dict(FG_SHADER_DIR=self.shader.replace("stack-adaptive", "multisess")),
            dict(FG_DECODE_PROFILE="1"), dict(LD_PRELOAD="observer.so"),
        ):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                self.check_identity(0, changes)
        with self.assertRaises(AssertionError):
            self.check_identity(0, sessions="1")
