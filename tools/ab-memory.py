"""Read-only memory sampling for a single manifest-scoped serving PID."""
import argparse
import json
from pathlib import Path
import re
import time


def key_values(path):
    result = {}
    for line in path.read_text().splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1]) * (1024 if len(parts) > 2 and parts[2] in ("kB", "KiB") else 1)
    return result


def sample(pid, manifest, release):
    proc = Path("/proc") / str(pid)
    args = proc.joinpath("cmdline").read_bytes().split(b"\0")
    expected = str(manifest).encode()
    if not any(args[i] == b"--manifest" and args[i + 1] == expected for i in range(len(args) - 1)):
        raise RuntimeError("Serving PID manifest changed")
    if proc.joinpath("exe").resolve() != release / "flash-gordon":
        raise RuntimeError("Serving PID executable changed")
    status = key_values(proc / "status")
    memory = key_values(Path("/proc/meminfo"))
    vm = key_values(Path("/proc/vmstat"))
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    clients = {}
    for info in proc.joinpath("fdinfo").iterdir():
        try:
            text = info.read_text()
        except FileNotFoundError:
            continue
        match = re.search(r"^drm-client-id:\s*(\d+)", text, re.MULTILINE)
        if match:
            clients[match[1]] = key_values(info)
    gpu = sum(client.get("drm-memory-gtt", 0) + client.get("drm-memory-vram", 0) for client in clients.values())
    state_files = []
    for path in manifest.parent.glob("qsa-*.state*"):
        info = path.stat()
        state_files.append({"name": path.name, "apparent_bytes": info.st_size, "allocated_bytes": info.st_blocks * 512})
    return {
        "timestamp": time.time(), "pid": pid, "manifest": str(manifest),
        "mem_available": memory["MemAvailable"], "system_swap_used": memory["SwapTotal"] - memory["SwapFree"],
        "rss": status.get("VmRSS", 0), "process_swap": status.get("VmSwap", 0),
        "major_faults": int(stat[9]), "swap_in_pages": vm.get("pswpin", 0),
        "swap_out_pages": vm.get("pswpout", 0), "oom_kills": vm.get("oom_kill", 0),
        "drm_clients": clients, "drm_gtt_vram_bytes": gpu, "qsa_state_files": state_files,
        "reserve_violation": memory["MemAvailable"] < 512 * 1024 * 1024,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=2400)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        print(json.dumps(sample(args.pid, args.manifest, args.release)), flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
