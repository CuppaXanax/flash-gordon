import json
import statistics
from pathlib import Path

root = Path("results/hillclimb/multisess-fleet-20261016")
load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
analysis = load(root / "analysis.json")
selection = load(root / "selection.json")
capacity = load(root / "capacity/capacity.json")
memory_path = root / "capacity/memory-summary.json"
memory = load(memory_path) if memory_path.exists() else {
    "safe": False, "minimum_available_gib": None, "ranks": []}
med = statistics.median
metrics = {row["metric"]: row for row in analysis["metrics"]}
protected_rows = [row for row in analysis["metrics"]
                  if row["metric"].startswith("solo-") or row["metric"] == "chat.ttft_ms"]
arms = {arm: [load(root / f"pair-{i}-{arm}/battery.json") for i in range(5)] for arm in ("A", "B")}
promoted = selection["sessions"] == 2
lines = [
    "# Adaptive two-session fleet qualification",
    "",
    f"**Decision: {'PROMOTED — sessions 2' if promoted else 'RETAINED — main/mix-04 sessions 1'}.** "
    "Exclusive .42–.49 qualification, 2026-10-16. Selection was pinned, all eight owners "
    "stopped/restarted through the retained launcher, and real API generation verified.",
    "",
    "## Identities and decision rule",
    "",
    "| | A: previous production | B: candidate |",
    "|---|---|---|",
    "| Engine source | `a63a166` | `2d0f8a7` (already contains `a63a166`) |",
    "| Release | `20261015-abq-main` | `20261016-abq-multisess` |",
    "| Sessions | 1 | 2 |",
    "| Binary SHA256 | `1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789` | "
    "`de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58` |",
    "",
    "Both arms use unchanged `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm`, "
    "SHA256 `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`; "
    "262144 context/index tokens, 700 MiB QSA page cache, zero hot tokens, wide kernel disabled. "
    "No decode profiler and no scheduler overrides. Candidate archive SHA256: "
    "`55404066fa9a902fa7474637a380d8c1ec85caf7324bfe51741f9a1204e8cdbb`.",
    "",
    "The requested practical gate is median solo performance within 3%, with no 95% interval "
    "wholly beyond the 3% regression boundary, and improved warm concurrent aggregate wall TPS "
    "(a win requires the lower CI above 1). "
    "Both marginal and paired medians are screened. Every solo engine/client metric and solo "
    "chat TTFT is protected separately. This is **not** a claim of strict statistical noninferiority: "
    "the latter is reported separately in `analysis.json`. Same-input A/B solo output identity, "
    "candidate loaded/isolated warm output identity, functional capacity and safe memory are "
    "additional safety checks.",
    "The user's revised operator gate is narrower than the original design's full QoS/parity "
    "acceptance battery. Promotion under this rule must not be represented as passing all of "
    "those older targets; loaded-gap limitations and cold mixed-agent hash divergence remain "
    "explicitly disclosed below.",
    "",
    f"- Protected practical gate: **{analysis['protected_pass']}**.",
    f"- Separately proven pointwise strict noninferiority: "
    f"**{sum(row['strict_noninferior'] for row in protected_rows)}/{len(protected_rows)}** "
    "protected metrics (not a simultaneous familywise guarantee).",
    f"- Aggregate improvement: **{analysis['aggregate_improves']}**; lower CI above 1: "
    f"**{analysis['aggregate_clear_win']}**.",
    f"- Solo A/B visible greedy identity: **{analysis['solo_hash_identity']}**.",
    f"- Candidate loaded/isolated visible identity: **{analysis['concurrent_hash_identity']}**.",
    "",
    "## Frozen protocol and accounting",
    "",
    "Five fresh-salt paired rounds, orders **AB, BA, AB, BA, AB**. Each arm restarts the ring. "
    "Prompt salt `multisess-20261016-pair-N` makes corresponding A/B bodies identical. "
    "Solo cold prompts target 4K/32K, with one cold generation followed by a 128-token cached "
    "decode; exact actual counts are retained. Solo chat is a fresh 128-token SSE response. "
    "Warm concurrency uses two independently primed conversations and 256 actual tokens each; "
    "mixed concurrency starts a cold 32K agent, then chat after 200 ms. The lists and hello-padding "
    "are synthetic workload proxies, not real-repository coding or a model-quality evaluation.",
    "",
    "The sustained-generation prime is a complete `Output exactly Ready.` answer with a "
    "32-token ceiling, not a one-token truncation of the list task. The earlier one-token-list "
    "pilot produced only one agent token and is retained under `pilot-*`, excluded from inference. "
    "The first frozen A arm was operationally resumed: **all valid prior solo/chat/warm samples "
    "were retained**, and only its missing mixed mode was generated after restart.",
    "",
    "Production A does not supply SSE usage/prefix metadata. Counts are reconciled from the "
    "**exact SSE request IDs** against its coordinator completion log, with provenance and log "
    "SHA256 in each battery. Event counts are never substituted for tokens. Raw CLI JSON remains "
    "explicitly incomplete pending that reconciliation. B uses usage on those exact SSE responses. "
    "CLI exit 2 signals a cold-versus-cache hash warning or accounting still pending. The driver "
    "retains a performance trial only after it is "
    "otherwise complete; exit 1/incomplete trials are failures, not zero samples. A's FIFO cannot "
    "retain both independent prefixes, so its loaded/cached-isolated hash differences are not "
    "passed off as correctness success.",
    "",
    "Rates below use actual generated tokens. Primary aggregate **wall** TPS divides by first "
    "launch through latest DONE. The secondary decode-window aggregate starts at earliest visible "
    "token; especially in mixed mode this excludes A's entire queued prefill but includes B's "
    "overlapped prefill, so it must not be used as the mixed-workload speedup gate. TTFT and gaps "
    "are client-visible SSE events; hidden/control/partial-UTF8 tokens are not individually visible.",
    "",
    "Inference: 20,000 fixed-seed paired round bootstraps of median log B/A ratios. The nominal "
    "95% bootstrap interval is conservatively enveloped by a df=4 Student-t interval on mean "
    "log-ratios; this small-sample envelope is explicitly not a simultaneous familywise bound. "
    "A crossing interval is inconclusive, not proof of noninferiority.",
    "",
    "## Five-pair measurements",
    "",
    "TPS metrics: higher is better; milliseconds: lower is better. Entries are medians; ratios "
    "and intervals are B/A. Complete samples and bootstrap-only intervals are in `analysis.json`.",
    "",
    "| Metric | A | B | B/A | Paired 95% envelope | Protected solo gate |",
    "|---|---:|---:|---:|---|---|",
]
for row in analysis["metrics"]:
    lo, hi = row["ci95"]
    protected = row["metric"].startswith("solo-") or row["metric"] == "chat.ttft_ms"
    gate = ("pass" if row["operator_pass"] else "fail") if protected else "diagnostic"
    lines.append(f"| `{row['metric']}` | {row['A']:.3f} | {row['B']:.3f} | {row['B_over_A']:.3f} | "
                 f"[{lo:.3f}, {hi:.3f}] | {gate} |")
lines += [
    "",
    "### Individual paired samples",
    "",
    "| Round/order | 4K engine prefill A/B | 32K engine prefill A/B | "
    "4K engine decode A/B | 32K engine decode A/B | Warm aggregate wall A/B |",
    "|---|---|---|---|---|---|",
]
for i in range(5):
    values = []
    for key in ("solo-4096.prefill_engine_tps", "solo-32768.prefill_engine_tps",
                "solo-4096.decode_engine_tps", "solo-32768.decode_engine_tps",
                "concurrency.aggregate_wall_tps"):
        values.append("/".join(f"{arms[arm][i]['metrics'][key]['value']:.2f}" for arm in ("A", "B")))
    lines.append(f"| {i + 1}/{'AB' if i % 2 == 0 else 'BA'} | " + " | ".join(values) + " |")
lines += ["", "## Loaded chat QoS and ring interpretation", ""]
lines.append(
    "Mixed prefill is a real tradeoff, not a hidden overall win: median mixed batch-wall "
    f"throughput is **{metrics['mixed.aggregate_wall_tps']['B_over_A']:.3f}x** A and cold-agent "
    f"TTFT is **{metrics['mixed.agent-style.ttft_ms']['B_over_A']:.3f}x** A, while chat TTFT falls "
    f"from **{metrics['mixed.chat-style.ttft_ms']['A']/1000:.2f} s** to "
    f"**{metrics['mixed.chat-style.ttft_ms']['B']/1000:.2f} s**. "
    "The primary aggregate gate is the predeclared **warm decode** workload, not this mixed "
    "batch. Loaded per-stream warm-rate intervals remain inconclusive despite near-solo medians.")
for arm in ("A", "B"):
    trials = [x["mixed"]["concurrency"]["trials"][0] for x in arms[arm]]
    gap = med(t["load"]["streams"][0]["inter_event_p95_ms"] for t in trials)
    reference = med(t["isolated"][0]["inter_event_p95_ms"] for t in trials)
    ratio = med(t["load"]["streams"][0]["inter_event_p95_ms"] /
                t["isolated"][0]["inter_event_p95_ms"] for t in trials)
    maximum = med(t["load"]["streams"][0]["inter_event_max_ms"] for t in trials)
    lines.append(f"- {arm}, chat during agent 32K prefill: median visible-gap p95 **{gap:.1f} ms**, "
                 f"isolated p95 **{reference:.1f} ms**, median within-round ratio **{ratio:.2f}x**, "
                 f"median per-trial maximum gap **{maximum:.1f} ms**.")
lines += [
    "",
    "**The ~1.5x loaded-gap goal is not guaranteed by this scheduler.** Report the observed "
    "ratio above rather than calling aggregate throughput a QoS pass. A is FIFO: its short gaps "
    "occur only after the long prefill finishes, while B actually progresses chat during prefill. "
    "The default 15 ms service-time slice is per-rank and soft, not an end-to-end 15 ms budget; "
    "already-issued frames and ring traversal add delay. A smaller `FG_MULTI_PREFILL_SLICE_US` "
    "is a future tuning experiment, not a measured or selected configuration here.",
    "",
    "The measured historical ring ledger gives ~37.437 ms token latency, 3.45–4.32 ms rank service, "
    "and ~0.8 ms head work. The ideal independent-stream bound is "
    "`T_stream >= max(L, N*d_max)`: two streams remain latency-limited, about **26–27 TPS each, "
    "~53 TPS aggregate**, not half the solo rate. These are old-profile ideal estimates, not "
    "a mix-04 performance guarantee. Actual warm per-stream rates are in the table. Real overlapping "
    "visible decode and aggregate improvement rule out generation-long FIFO serialization; "
    "remaining slowdown versus the ideal is not localized to a particular lock by this battery.",
    "Code inspection also confirms `fg_owner_set_active_session` only changes the active-state "
    "selector, and owner static replay partitions its eight graph slots between sessions "
    "(`src/owner.c`), rather than invalidating every graph on each switch. Coordinator activation "
    "does drain static GPU work on an actual slot change (`fg_runtime_activate`); the ready-first "
    "scheduler avoids unnecessary same-slot/sleeping-job activation. These facts do not quantify "
    "the residual GPU/cache/transport scheduling cost.",
    "",
    "Changing prefill chunk shapes can change floating-point trajectories. The cold mixed-agent "
    "A/B visible hashes are reported diagnostically below; they are not conflated with the "
    "required single-active-session greedy identity or with identical-path warm-session isolation.",
]
lines.append(f"The exact first-round cold agent input was additionally rerun alone in B's retained "
             f"agent slot: A/B visible hash, prompt/generation counts and cold-prefix parity "
             f"**{capacity.get('single_active_agent_parity', False)}**. See "
             "`capacity/single-active-agent-parity.json`; this is a safety diagnostic, not a sixth "
             "paired throughput sample. It does not by itself prove the cause of mixed-path divergence.")
for i in range(5):
    a, b = [arms[arm][i]["mixed"]["concurrency"]["trials"][0]["load"]["streams"][1] for arm in ("A", "B")]
    lines.append(f"- Round {i + 1} cold mixed-agent visible hash equal: **{a['sha256'] == b['sha256']}**.")
lines += ["", "## Capacity and memory", ""]
if capacity["phase1"]:
    lines.append(f"Phase 1 cold agent: **{capacity['phase1']['prompt_tokens']} prompt tokens**, "
                 f"**{capacity['phase1']['completion_tokens']} generated**, "
                 f"**{capacity['phase1']['wall_seconds']:.1f} s** client wall, alongside live chat.")
if capacity["phase2"]:
    lines.append(f"Phase 2 chat: **{capacity['phase2']['PromptTokens']} prompt tokens**, with the long "
                 "agent state retained. Subsequent simultaneous continuations verify both resident "
                 "prefixes and sustained generation.")
lines.append(f"Completed: **{capacity['complete']}**; both contexts near 262K: "
             f"**{capacity.get('both_near_limit', False)}**. Do not interpret a 262K-agent/32K-chat "
             "run as two full 262K contexts.")
if capacity["dual_decode"]:
    for stream in capacity["dual_decode"]["streams"]:
        lines.append(f"- {stream['name']}: {stream['prompt_tokens']} input, "
                     f"{stream['prefix']['reused_tokens']} reused, {stream['completion_tokens']} generated, "
                     f"{stream['client_decode_tps']:.2f} decode-window TPS.")
    lines.append(f"- Near-limit concurrent aggregate wall: "
                 f"{capacity['dual_decode']['aggregate_wall_tps']:.2f} TPS.")
minimum = (f"{memory['minimum_available_gib']:.3f} GiB available"
           if memory["minimum_available_gib"] is not None else "unavailable")
lines += [
    "",
    f"Memory safety: **{memory['safe']}**; minimum across all ranks "
    f"**{minimum}**. Two-second manifest/executable-scoped "
    "samples require >=512 MiB available, zero process swap and no OOM/swap-counter growth; "
    "observer coverage through completion is checked. DRM allocation and RSS are separate views "
    "of shared iGPU memory, not additive totals. QSA state-file inventories cover the pack directory. "
    "Raw observations are retained as gzip with raw/compressed SHA256 in `memory-summary.json`.",
    "",
    "| Rank | Minimum available GiB | Peak RSS GiB | Peak DRM GiB | Samples | "
    "OOM / swap-in / swap-out growth | Safe |",
    "|---|---:|---:|---:|---:|---|---|",
]
for row in memory["ranks"]:
    lines.append(f"| {row['rank']} | {row['min_mem_available']/1024**3:.3f} | "
                 f"{row['max_rss']/1024**3:.3f} | {row['max_drm_bytes']/1024**3:.3f} | "
                 f"{row['samples']} | {row['oom_kill_growth']} / {row['swap_in_growth']} / "
                 f"{row['swap_out_growth']} | {row['safe']} |")
lines += [
    "",
    "## Persistent selection, restart and reproduction",
    "",
    f"Selected source **`{selection['source_commit']}`**, release **`{selection['release']}`**, "
    f"**sessions {selection['sessions']}**, unchanged mix-04. `selection.json` and `selected/` "
    "contain the pinned identity, launcher, all-eight stop/start/verification receipts, model/health "
    "responses and actual API reply. For sessions 2, a fresh real two-stream generation is also "
    "required after the complete restart. Verification checks live executable and manifest hashes, "
    "argv, context/cache settings and the exact allowed FG environment.",
    "",
    "Persistence means retained hash-pinned launchers plus detached serving processes that survive "
    "SSH/logout; **no boot-time/systemd auto-restart claim** is made.",
    "",
    "On every blade the retained launcher is:",
    "",
    "```bash",
    "bash /home/xander/flash-gordon-live/20261015-abq-operator/launch-current.sh start",
    "```",
    "",
    "For a ring restart or rollback, use the same fleet transport as this run: stop all eight "
    "owned numeric PIDs using the retained launcher, start .43–.49 first, then .42, wait for "
    "`http://192.168.42.42:8080/health`, and make a real `/v1/chat/completions` request. "
    "The pinned **mix-04 main** fallback is `launch-main-mix04.sh`; it is not the historical "
    "Q8 `launch-original.sh`. Exact tested commands are in `protocol/pin-multisess-selection.ps1`.",
    "",
    "To repeat the qualification from this Windows worktree (exclusive lease required):",
    "",
    "```powershell",
    "$root = 'D:\\looking-glass-labs\\fg-wt\\multi-session-int'",
    "Set-Location $root",
    "$p = 'results\\hillclimb\\multisess-fleet-20261016\\protocol'",
    "New-Item -ItemType Directory -Force .build | Out-Null",
    "Copy-Item \"$p\\multisess-*.ps1\",\"$p\\analyze-multisess-*.py\","
    "\"$p\\capacity-*.sh\",\"$p\\pin-multisess-selection.ps1\" .build",
    "python tools\\test_hillclimb_concurrency.py",
    "pwsh -NoProfile -File .build\\multisess-fleet.ps1",
    "python .build\\analyze-multisess-fleet.py",
    "pwsh -NoProfile -File .build\\multisess-capacity.ps1",
    "python .build\\analyze-multisess-memory.py",
    f"pwsh -NoProfile -File .build\\pin-multisess-selection.ps1 -Arm {'B' if promoted else 'A'}",
    "```",
    "",
    "**Do not overwrite these evidence directories on a new run**: choose a fresh output date "
    "and paired salts/time-budget deadline in the archived scripts, preserve the same workload, and install/hash-check "
    "the immutable releases first (`protocol/install-multisess.sh`, `install.json`). Deployment "
    "uses the archived `ab-fleet.ps1`, `ab-deploy.ps1` and `ab-instance.sh` through the existing "
    "`D:\\looking-glass-labs\\fg-wt\\ab-harness` operator setup; credentials are not archived. "
    "This experiment must not run concurrently with another lease holder.",
    "",
    "Current harness example (candidate, real simultaneous SSE, exact token accounting):",
    "",
    "```powershell",
    "pwsh -NoProfile -File tools\\hillclimb.ps1 -ConcurrentGeneration -ConcurrentRepeats 1 "
    "-ConcurrentTokens 256 -ChatContextTokens 64 -AgentContextTokens 32768 -MixedPrefill "
    "-CapacityProbe -PromptSalt fresh-pair-0 -ResultsDirectory results\\fresh-mixed",
    "```",
    "",
    "`-CapacityProbe` avoids a duplicate isolated cold agent in mixed mode; a 32K argument "
    "is still only a 32K test. `-AllowSerializedBaseline` permits FIFO/no-prefix-overlap baseline "
    "measurement but cannot manufacture usage: missing usage leaves the result incomplete until "
    "exact request-ID server-log reconciliation.",
    "",
    "Six local harness fixtures pass, including simultaneous SSE, mixed prefill, single-cold "
    "capacity protocol, rejected short generation with retained raw data, deferred missing usage "
    "and deterministic paired salts. No C/Vulkan arithmetic or pack files changed in this "
    "qualification. Experiment memory observers are stopped and scratch cleaned before lease release.",
]
Path("results/hillclimb/MULTI_SESSION_FLEET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote report; selected {'B' if promoted else 'A'}.")
