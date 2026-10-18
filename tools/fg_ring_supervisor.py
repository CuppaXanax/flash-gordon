#!/usr/bin/env python3
"""Ordered ring supervision. Serving processes remain owned by pinned launchers."""
import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.operator = Path(config["operator"])
        self.nodes = config["nodes"]
        if len(self.nodes) < 2 or sorted(n["rank"] for n in self.nodes) != list(
            range(len(self.nodes))
        ):
            raise ValueError("nodes must have unique contiguous ranks, starting with zero")
        self.nodes = sorted(self.nodes, key=lambda node: node["rank"])
        self.stop = threading.Event()

    def checkpoint(self):
        if self.stop.is_set() or (self.operator / "supervisor.pause").exists():
            raise InterruptedError("supervisor stopped or paused")

    def command(self, node, action):
        if node["rank"] == 0:
            command = [
                "python3", str(self.operator / "fg_ring_control.py"),
                "--operator", str(self.operator), "--action", action,
            ]
        else:
            command = [
                "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={self.config['known_hosts']}",
                "-o", "IdentitiesOnly=yes", "-i", self.config["identity"],
                f"{self.config['user']}@{node['host']}", action,
            ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if completed.returncode:
            raise RuntimeError(
                f"rank {node['rank']} {action}: {completed.stderr.strip()}"
            )
        return completed.stdout

    def batch(self, nodes, action):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            futures = [pool.submit(self.command, node, action) for node in nodes]
            errors = []
            for future in futures:
                try:
                    future.result()
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))

    def health(self):
        try:
            with urllib.request.urlopen(self.config["health"], timeout=2) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                return json.load(error)
            except (ValueError, OSError):
                return {"status": "down"}
        except (OSError, ValueError):
            return {"status": "down"}

    def ready(self, epoch=None):
        deadline = time.monotonic() + self.config.get("startup_timeout", 240)
        while not self.stop.is_set() and time.monotonic() < deadline:
            self.checkpoint()
            state = self.health()
            if state.get("status") == "ok" and (
                epoch is None or state.get("recovery_epoch", -1) >= epoch
            ):
                self.batch(self.nodes, "status")
                return
            self.stop.wait(1)
        self.checkpoint()
        raise RuntimeError("ring readiness deadline exceeded")

    def restart(self, state):
        self.checkpoint()
        recovering = state.get("ring_recovery") and state.get("status") in (
            "ok", "recovering"
        )
        # A live recovery-capable coordinator keeps its HTTP listener and
        # reopens the fabric. Never stop it just to restart disconnected workers.
        if not recovering:
            self.command(self.nodes[0], "stop")
        self.checkpoint()
        self.batch(self.nodes[1:], "stop")
        self.checkpoint()
        self.batch(self.nodes[1:], "start")
        self.checkpoint()
        if not recovering:
            self.command(self.nodes[0], "start")
        epoch = None
        if recovering:
            epoch = state["recovery_epoch"] + (state["status"] == "ok")
        self.ready(epoch)
        print(json.dumps({"event": "ring_ready", "kept_api": bool(recovering),
                          "at": time.time()}), flush=True)

    def tick(self):
        if (self.operator / "supervisor.pause").exists():
            return
        state = self.health()
        if state.get("status") == "ok":
            try:
                self.batch(self.nodes, "status")
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        try:
            self.restart(state)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(json.dumps({"event": "recovery_failed", "error": str(error)}), flush=True)
            # A broken handshake/GPU may require a fresh coordinator too.
            # Do not silently spin on a permanently half-open ring.
            self.checkpoint()
            self.restart({"status": "down"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    supervisor = Supervisor(json.loads(Path(args.config).read_text()))
    lock = (supervisor.operator / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.check:
        supervisor.batch(supervisor.nodes, "status")
        if supervisor.health().get("status") != "ok":
            raise RuntimeError("API is not healthy")
        print("Pinned ring and restricted SSH transport verified", flush=True)
        return
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda *_: supervisor.stop.set())
    while not supervisor.stop.is_set():
        try:
            supervisor.tick()
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            print(json.dumps({"event": "supervisor_retry", "error": str(error)}), flush=True)
        supervisor.stop.wait(supervisor.config.get("interval", 10))


if __name__ == "__main__":
    main()
