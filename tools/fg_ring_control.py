#!/usr/bin/env python3
"""Restricted SSH entry point for an existing hash-pinned rank launcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def control(operator, action):
    if action not in ("start", "stop", "status"):
        raise ValueError("only start, stop and status are allowed")
    directory = Path(operator).resolve(strict=True)
    rank = int((directory / "supervisor-rank").read_text().strip())
    if rank < 0:
        raise ValueError("invalid local rank")
    if action != "status":
        subprocess.run(["bash", str(directory / "launch-current.sh"), action], check=True)
        return
    identity = json.loads((directory / "current.json").read_text())
    pid = int((directory / f"rank-{rank}.pid").read_text().strip())
    if pid <= 1:
        raise ValueError("invalid rank PID")
    process = Path("/proc") / str(pid)
    expected = str(Path(identity["release"]) / "flash-gordon")
    actual = os.readlink(process / "exe")
    argv = (process / "cmdline").read_bytes().split(b"\0")
    manifest = identity["manifest"].encode()
    if actual != expected or not any(
        value == b"--manifest" and i + 1 < len(argv) and argv[i + 1] == manifest
        for i, value in enumerate(argv)
    ):
        raise ValueError("rank PID does not belong to the pinned deployment")
    print(json.dumps({"rank": rank, "pid": pid, "owned": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--action", choices=("start", "stop", "status"))
    args = parser.parse_args()
    control(args.operator, args.action or os.environ.get("SSH_ORIGINAL_COMMAND", ""))
