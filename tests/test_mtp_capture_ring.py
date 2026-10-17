import collections
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import unittest
import uuid


@unittest.skipIf(sys.platform == "win32", "native ring fixture is built on Linux")
class RingCaptureTests(unittest.TestCase):
    def test_local_chain_captures_every_owner(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / ".build" / ("ring-capture-" + uuid.uuid4().hex)
        directory.mkdir(parents=True)
        try:
            base = self.reserve_ports()
            with (directory / "fixture.log").open("w") as log:
                result = subprocess.run(
                    [str(root / "tests" / "test_mtp_ring"), "127.0.0.1", "127.0.0.1",
                     str(base), "3", "2", "7", "1", str(directory)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=100,
                )
            self.assertEqual(result.returncode, 0,
                             (directory / "fixture.log").read_text()[-2000:])
            rows = [json.loads(line)
                    for path in directory.glob("verify-routes-*.jsonl")
                    for line in path.read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(any(row["n"] == 4 for row in rows))
            self.assertEqual({row["layer"] for row in rows}, set(range(48)))
            windows = collections.defaultdict(collections.Counter)
            for row in rows:
                self.assertEqual(row["kind"], "mtp_verify_routes")
                self.assertEqual(len(row["experts"]), row["n"])
                for selected in row["experts"]:
                    self.assertEqual(len(selected), 10)
                    self.assertEqual(len(set(selected)), 10)
                    self.assertTrue(all(0 <= expert < 512 for expert in selected))
                key = (row["session"], row["first"], row["n"], row["replay"])
                windows[key][row["layer"]] += 1
            for counts in windows.values():
                self.assertEqual(set(counts), set(range(48)))
                self.assertEqual(len(set(counts.values())), 1)
        finally:
            shutil.rmtree(directory)

    def reserve_ports(self):
        for _ in range(32):
            base = 40000 + uuid.uuid4().int % 18000
            sockets = []
            try:
                for port in range(base, base + 16):
                    listener = socket.socket()
                    sockets.append(listener)
                    listener.bind(("127.0.0.1", port))
                return base
            except OSError:
                pass
            finally:
                for listener in sockets:
                    listener.close()
        self.fail("no free contiguous fixture port range")
