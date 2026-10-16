"""Rebuild AB_QUALIFY.json from a worktree's retained qualification evidence."""
import argparse
import collections
import datetime
import hashlib
import json
import re
import statistics
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path.cwd())
ROOT = parser.parse_args().root.resolve()
RESULTS = ROOT / "results" / "hillclimb"
OPS = RESULTS / "ab-ops"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


summary = {
    "generated": datetime.datetime.now().astimezone().isoformat(),
    "decision": "original",
    "reason": "No complete, quality-approved candidate passed every protected metric and a corrected win.",
    "candidates": {},
    "memory": {},
    "power_caveat": "With only three pairs, the df=2 guard and 76-comparison correction have very low power. The original-pack main-binary control is inconclusive on 12/19 metrics. The gate was not changed after results; redesign power before the next run.",
}
for name in ("main", "mix", "wide", "multi", "multi-fixed"):
    if not (RESULTS / f"ab-{name}" / "ab-result.json").exists():
        continue
    result = read(RESULTS / f"ab-{name}" / "ab-result.json")
    metrics = result["metrics"]
    hashes = [h for pair in result["pairs"] for h in pair["hash_match"]]
    counts = collections.Counter(m["status"] for m in metrics)
    summary["candidates"][name] = {
        "started": result["started"], "finished": result.get("finished"),
        "elapsed_minutes": (
            datetime.datetime.fromisoformat(result["finished"])
            - datetime.datetime.fromisoformat(result["started"])
        ).total_seconds() / 60 if result.get("finished") else None,
        "complete": result["complete"], "complete_pairs": len(result["pairs"]),
        "quality_pass": result.get("quality_pass", False),
        "same_pack_exact_hash_identity": result.get("same_pack_exact_hash_identity"),
        "hash_matches": sum(h["match"] for h in hashes), "hash_comparisons": len(hashes),
        "hash_difference_cases": sorted({h["name"] for h in hashes if not h["match"]}),
        "statuses": dict(counts), "metrics": metrics, "deployment": result["config"]["B"],
        "eligible": result["eligible"], "error": result.get("error"),
        "restored_original": result.get("restored_original", False),
        "concurrent_completion_tokens": sorted({
            s["completion_tokens"] for p in result["pairs"]
            for arm in ("A", "B") for s in p[arm].get("concurrency", {}).get("streams", [])
        }),
    }
    if name != "multi":
        assert result["complete"] and len(result["pairs"]) == 3
    assert not result["eligible"], "Reassess the serving decision before using this summary"

reference = {
    h["name"]: h["sha256"]
    for h in read(RESULTS / "ab-mix" / "ab-result.json")["pairs"][0]["B"]["hashes"]
}
wide = read(RESULTS / "ab-wide" / "ab-result.json")
summary["candidates"]["wide"]["same_pack_hash_difference_cases"] = sorted({
    h["name"] for p in wide["pairs"] for h in p["B"]["hashes"]
    if h["sha256"] != reference[h["name"]]
})
summary["best_pack_selection"] = read(OPS / "best-pack.json")
receipt = read(OPS / "mix04-quality-receipt.json")
quality_path = OPS / "mix480-04-quality.json"
assert hashlib.sha256(quality_path.read_bytes()).hexdigest() == receipt["quality_sha256"]
quality = read(quality_path)
summary["quality"] = {
    k: quality[k] for k in ("passed", "samples", "full_vocab", "mean_kld", "p99_kld", "top1_agreement")
}
summary["quality"]["manifest_sha256"] = receipt["manifest_sha256"]
summary["quality"]["quality_sha256"] = receipt["quality_sha256"]

if (RESULTS / "ab-load-diagnostics.json").exists():
    summary["load_diagnostics"] = read(RESULTS / "ab-load-diagnostics.json")
if (RESULTS / "ab-load-fixed-diagnostics.json").exists():
    summary["fixed_load_diagnostics"] = read(RESULTS / "ab-load-fixed-diagnostics.json")
if (RESULTS / "ab-load-final-diagnostics.json").exists():
    summary["final_load_diagnostics"] = read(RESULTS / "ab-load-final-diagnostics.json")
summary["diagnostic_trials"] = {}
for probe in ("warm-forensic", "near262k-final"):
    files = sorted((RESULTS / f"ab-multi-{probe}").glob("20*.json"))
    if files:
        assert len(files) == 1, "Do not pool diagnostic repeats"
        record = read(files[0])
        summary["diagnostic_trials"][probe] = {
            "source": str(files[0].relative_to(RESULTS)),
            "native_commit": "79b0efc",
            "elapsed_seconds": record["metadata"]["elapsed_seconds"],
            "concurrency": record["concurrency"], "errors": record["errors"],
            "inferential": False,
        }
for probe in ("warm", "near262k", "warm-fixed", "near262k-fixed-interrupted", "warm-forensic", "near262k-final"):
    rows = []
    for archive in sorted((RESULTS / "ab-memory" / probe).glob("*.tgz")):
        output = archive.parent / archive.stem
        output.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as source:
            expected = {f"{probe}.jsonl", f"{probe}.err", f"{probe}-rank.log"}
            assert {m.name for m in source.getmembers()} == expected
            for member in source.getmembers():
                assert member.isfile()
                (output / member.name).write_bytes(source.extractfile(member).read())
        samples = [
            json.loads(line) for line in (output / f"{probe}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert samples and len({s["pid"] for s in samples}) == 1
        times = [s["timestamp"] for s in samples]
        row = {
            "ip": archive.stem, "samples": len(samples), "pid": samples[0]["pid"],
            "sample_start_unix": times[0], "sample_end_unix": times[-1],
            "duration_seconds": times[-1] - times[0],
            "max_sample_gap_seconds": max((b-a for a, b in zip(times, times[1:])), default=0),
            "minimum_mem_available_bytes": min(s["mem_available"] for s in samples),
            "peak_rss_bytes": max(s["rss"] for s in samples),
            "peak_drm_gtt_vram_bytes": max(s["drm_gtt_vram_bytes"] for s in samples),
            "peak_process_swap_bytes": max(s["process_swap"] for s in samples),
            "system_swap_growth_bytes": max(s["system_swap_used"] for s in samples)-samples[0]["system_swap_used"],
            "major_faults_delta": samples[-1]["major_faults"]-samples[0]["major_faults"],
            "swap_in_pages_delta": samples[-1]["swap_in_pages"]-samples[0]["swap_in_pages"],
            "swap_out_pages_delta": samples[-1]["swap_out_pages"]-samples[0]["swap_out_pages"],
            "oom_kills_delta": samples[-1]["oom_kills"]-samples[0]["oom_kills"],
            "reserve_violations": sum(s["reserve_violation"] for s in samples),
            "peak_qsa_apparent_bytes": max(sum(f["apparent_bytes"] for f in s["qsa_state_files"]) for s in samples),
            "peak_qsa_allocated_bytes": max(sum(f["allocated_bytes"] for f in s["qsa_state_files"]) for s in samples),
            "sampler_error": (output / f"{probe}.err").read_text(),
        }
        rows.append(row)
    summary["memory"][probe] = rows
current = RESULTS / "ab-current" / "api-health.json"
if current.exists():
    summary["final_api_health"] = read(current)
for label in ("cleanup", "post-cleanup-health", "verify"):
    path = RESULTS / "ab-current" / f"{label}.json"
    if path.exists():
        summary[f"final_{label.replace('-', '_')}"] = read(path)
summary["capacity_outcome"] = read(OPS / "final-capacity-outcome.json")
summary["capacity_cancellation"] = read(OPS / "near-final-cancel.json")
capture = read(OPS / "capture-final.json")
rank_zero = next(r["Stdout"] for r in capture if r["IP"] == "192.168.42.42")
match = re.search(r"prefill interrupted after (\d+)/(\d+) tokens, frontier (\d+), ([\d.]+) s", rank_zero)
assert match, "Retain the authoritative interrupted token frontier"
summary["capacity_outcome"].update({
    "completed_prefill_tokens": int(match[1]), "requested_prompt_tokens": int(match[2]),
    "frontier": int(match[3]), "elapsed_prefill_seconds": float(match[4]),
    "full_context_qualified": False, "terminal_agent_sse_usage": None,
    "source": "ab-ops/capture-final.json",
})
(RESULTS / "AB_QUALIFY.json").write_text(json.dumps(summary, indent=2)+"\n")
for name, c in summary["candidates"].items():
    print(name, c["complete"], c["complete_pairs"], c["quality_pass"], c["statuses"],
          "elapsed", c["elapsed_minutes"], "eligible", c["eligible"])
for probe, rows in summary["memory"].items():
    for r in rows:
        print(probe, r["ip"], "min_GiB", round(r["minimum_mem_available_bytes"]/2**30, 3),
              "RSS_GiB", round(r["peak_rss_bytes"]/2**30, 3),
              "GPU_GiB", round(r["peak_drm_gtt_vram_bytes"]/2**30, 3),
              "major_faults", r["major_faults_delta"], "OOM", r["oom_kills_delta"],
              "sampler_error", bool(r["sampler_error"]))
