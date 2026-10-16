import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

root = Path("results/hillclimb/multisess-fleet-20261016/capacity")
capacity = json.loads((root / "capacity.json").read_text(encoding="utf-8-sig"))
started = datetime.fromisoformat(capacity["started"]).timestamp()
finished = datetime.fromisoformat(capacity["finished"]).timestamp()
rows = []
for suffix in range(42, 50):
    path = root / f"192.168.42.{suffix}-memory.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing capacity observations: {path}")
    samples = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not samples:
        raise SystemExit(f"No samples: {path}")
    first, last = samples[0], samples[-1]
    row = dict(rank=suffix - 42, samples=len(samples),
               seconds=last["timestamp"] - first["timestamp"],
               min_mem_available=min(x["mem_available"] for x in samples),
               max_rss=max(x["rss"] for x in samples),
               max_drm_bytes=max(x["drm_gtt_vram_bytes"] for x in samples),
               max_process_swap=max(x["process_swap"] for x in samples),
               system_swap_growth=max(x["system_swap_used"] for x in samples) - first["system_swap_used"],
               swap_in_growth=max(x["swap_in_pages"] for x in samples) - first["swap_in_pages"],
               swap_out_growth=max(x["swap_out_pages"] for x in samples) - first["swap_out_pages"],
               oom_kill_growth=max(x["oom_kills"] for x in samples) - first["oom_kills"],
               major_fault_growth=last["major_faults"] - first["major_faults"],
               max_state_apparent_bytes=max(sum(s["apparent_bytes"] for s in x["qsa_state_files"])
                                            for x in samples),
               max_state_allocated_bytes=max(sum(s["allocated_bytes"] for s in x["qsa_state_files"])
                                             for x in samples),
               raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    row["initial_sample_delay_seconds"] = first["timestamp"] - started
    row["last_sample_age_at_completion_seconds"] = finished - last["timestamp"]
    row["safe"] = (row["min_mem_available"] >= 512 * 1024**2
                   and row["samples"] >= 15
                   and row["seconds"] >= max(30, finished - started - 120)
                   and row["samples"] >= row["seconds"] / 3
                   and row["max_process_swap"] == 0
                   and row["system_swap_growth"] <= 0
                   and row["swap_in_growth"] <= 0
                   and row["swap_out_growth"] <= 0
                   and row["oom_kill_growth"] <= 0)
    rows.append(row)
    with path.open("rb") as source, gzip.open(str(path) + ".gz", "wb") as target:
        target.write(source.read())
    row["gzip_sha256"] = hashlib.sha256(Path(str(path) + ".gz").read_bytes()).hexdigest()
summary = dict(safe=all(x["safe"] for x in rows), ranks=rows,
               minimum_available_gib=min(x["min_mem_available"] for x in rows) / 1024**3,
               note="Two-second manifest/executable-scoped PID samples; zero growth in OOM/swap counters "
                    "and >=512 MiB available required. Raw samples retained as gzip.")
(root / "memory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
