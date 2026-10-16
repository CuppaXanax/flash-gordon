"""Read-only per-client DRM engine accounting around a serving request."""
import json
import os
import re
import time
from pathlib import Path

address = next(a for a in os.popen("ip -4 addr show").read().split()
               if a.startswith("192.168.42.") and "/" in a)
rank = int(address.split(".")[-1].split("/")[0]) - 42
assert 0 <= rank < 8
operator = Path("/home/xander/flash-gordon-live/20261015-abq-operator")
pid = int((operator / f"rank-{rank}.pid").read_text())
proc = Path("/proc") / str(pid)
config = json.loads((operator / "current.json").read_text())
assert proc.stat().st_uid == os.getuid()
assert proc.joinpath("exe").resolve() == Path(config["release"]) / "flash-gordon"
assert config["manifest"].encode() in proc.joinpath("cmdline").read_bytes().split(b"\0")
clients = {}
for path in proc.joinpath("fdinfo").iterdir():
    try:
        text = path.read_text()
    except FileNotFoundError:
        continue
    match = re.search(r"^drm-client-id:\s*(\d+)", text, re.M)
    if match:
        busy = re.search(r"^drm-engine-gfx:\s*(\d+)", text, re.M)
        clients[match[1]] = int(busy[1]) if busy else None
assert clients and all(v is not None for v in clients.values())
stat = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
clocks = {}
for device in Path("/sys/class/drm").glob("card[0-9]*/device"):
    for name in ("pp_dpm_sclk", "pp_dpm_mclk", "power_dpm_force_performance_level"):
        try:
            clocks[str(device / name)] = (device / name).read_text()
        except (OSError, PermissionError):
            pass
print(json.dumps({
    "rank": rank, "pid": pid, "monotonic_ns": time.monotonic_ns(),
    "drm_clients": clients, "gpu_engine_ns": sum(clients.values()),
    "cpu_user_ticks": int(stat[11]), "cpu_system_ticks": int(stat[12]),
    "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
    "config": config, "clock_snapshots": clocks,
}))
