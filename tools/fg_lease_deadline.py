#!/usr/bin/env python3
"""Independent deadline rollback for a bounded fleet qualification lease."""
import argparse
import json
from pathlib import Path
import time
from fg_ring_supervisor import Supervisor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    supervisor = Supervisor(json.loads(Path(args.config).read_text()))
    active = supervisor.operator / "lease-active.json"
    if active.is_file():
        (supervisor.operator / "supervisor.pause").touch()
        supervisor.command(supervisor.nodes[0], "restore")
        supervisor.batch(supervisor.nodes[1:], "restore")
        supervisor.batch(supervisor.nodes[1:], "start")
        supervisor.command(supervisor.nodes[0], "start")
        (supervisor.operator / "supervisor.pause").unlink(missing_ok=True)
        supervisor.ready()
        print(json.dumps({"event": "lease_deadline_baseline_restored",
                          "at": time.time()}), flush=True)
