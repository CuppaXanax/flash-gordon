# Adaptive two-session fleet qualification

**Decision: PROMOTED — sessions 2.** Exclusive .42–.49 qualification, 2026-10-16. Selection was pinned, all eight owners stopped/restarted through the retained launcher, and real API generation verified.

[Complete dated evidence and reproducibility protocol](multisess-fleet-20261016/);
[current deployment/rollback runbook](https://github.com/CuppaXanax/flash-gordon/blob/cuppaxanax/ab-harness/docs/DEPLOY_CURRENT.md).

## Identities and decision rule

| | A: previous production | B: candidate |
|---|---|---|
| Engine source | `a63a166` | `2d0f8a7` (already contains `a63a166`) |
| Release | `20261015-abq-main` | `20261016-abq-multisess` |
| Sessions | 1 | 2 |
| Binary SHA256 | `1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789` | `de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58` |

Both arms use unchanged `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm`, SHA256 `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`; 262144 context/index tokens, 700 MiB QSA page cache, zero hot tokens, wide kernel disabled. No decode profiler and no scheduler overrides. Candidate archive SHA256: `55404066fa9a902fa7474637a380d8c1ec85caf7324bfe51741f9a1204e8cdbb`.

The requested practical gate is median solo performance within 3%, with no 95% interval wholly beyond the 3% regression boundary, and improved warm concurrent aggregate wall TPS (a win requires the lower CI above 1). Both marginal and paired medians are screened. Every solo engine/client metric and solo chat TTFT is protected separately. Passing this practical rule alone does **not** prove strict statistical noninferiority: the latter is evaluated separately in `analysis.json` and all nine protected pointwise checks pass here. Same-input A/B solo output identity, candidate loaded/isolated warm output identity, functional capacity and safe memory are additional safety checks.
The user's revised operator gate is narrower than the original design's full QoS/parity acceptance battery. Promotion under this rule must not be represented as passing all of those older targets; loaded-gap limitations and cold mixed-agent hash divergence remain explicitly disclosed below.

- Protected practical gate: **True**.
- Separately proven pointwise strict noninferiority: **9/9** protected metrics (not a simultaneous familywise guarantee).
- Aggregate improvement: **True**; lower CI above 1: **True**.
- Solo A/B visible greedy identity: **True**.
- Candidate loaded/isolated visible identity: **True**.

## Frozen protocol and accounting

Five fresh-salt paired rounds, orders **AB, BA, AB, BA, AB**. Each arm restarts the ring. Prompt salt `multisess-20261016-pair-N` makes corresponding A/B bodies identical. Solo cold prompts target 4K/32K, with one cold generation followed by a 128-token cached decode; exact actual counts are retained. Solo chat is a fresh 128-token SSE response. Warm concurrency uses two independently primed conversations and 256 actual tokens each; mixed concurrency starts a cold 32K agent, then chat after 200 ms. The lists and hello-padding are synthetic workload proxies, not real-repository coding or a model-quality evaluation.

The sustained-generation prime is a complete `Output exactly Ready.` answer with a 32-token ceiling, not a one-token truncation of the list task. The earlier one-token-list pilot produced only one agent token and is retained under `pilot-*`, excluded from inference. The first frozen A arm was operationally resumed: **all valid prior solo/chat/warm samples were retained**, and only its missing mixed mode was generated after restart.

Production A does not supply SSE usage/prefix metadata. Counts are reconciled from the **exact SSE request IDs** against its coordinator completion log, with provenance and log SHA256 in each battery. Event counts are never substituted for tokens. Raw CLI JSON remains explicitly incomplete pending that reconciliation. B uses usage on those exact SSE responses. CLI exit 2 signals a cold-versus-cache hash warning or accounting still pending. The driver retains a performance trial only after it is otherwise complete; exit 1/incomplete trials are failures, not zero samples. A's FIFO cannot retain both independent prefixes, so its loaded/cached-isolated hash differences are not passed off as correctness success.

Rates below use actual generated tokens. Primary aggregate **wall** TPS divides by first launch through latest DONE. The secondary decode-window aggregate starts at earliest visible token; especially in mixed mode this excludes A's entire queued prefill but includes B's overlapped prefill, so it must not be used as the mixed-workload speedup gate. TTFT and gaps are client-visible SSE events; hidden/control/partial-UTF8 tokens are not individually visible.

Inference: 20,000 fixed-seed paired round bootstraps of median log B/A ratios. The nominal 95% bootstrap interval is conservatively enveloped by a df=4 Student-t interval on mean log-ratios; this small-sample envelope is explicitly not a simultaneous familywise bound. A crossing interval is inconclusive, not proof of noninferiority.

## Five-pair measurements

TPS metrics: higher is better; milliseconds: lower is better. Entries are medians; ratios and intervals are B/A. Complete samples and bootstrap-only intervals are in `analysis.json`.

| Metric | A | B | B/A | Paired 95% envelope | Protected solo gate |
|---|---:|---:|---:|---|---|
| `solo-4096.prefill_engine_tps` | 331.429 | 331.496 | 1.000 | [0.979, 1.047] | pass |
| `solo-4096.prefill_wall_tps` | 328.691 | 328.876 | 1.001 | [0.979, 1.047] | pass |
| `solo-4096.decode_engine_tps` | 27.992 | 27.861 | 0.995 | [0.986, 1.008] | pass |
| `solo-4096.decode_client_tps` | 27.982 | 27.848 | 0.995 | [0.986, 1.008] | pass |
| `solo-32768.prefill_engine_tps` | 368.758 | 383.879 | 1.041 | [0.994, 1.070] | pass |
| `solo-32768.prefill_wall_tps` | 368.176 | 383.242 | 1.041 | [0.994, 1.070] | pass |
| `solo-32768.decode_engine_tps` | 25.642 | 25.715 | 1.003 | [0.990, 1.033] | pass |
| `solo-32768.decode_client_tps` | 25.613 | 25.677 | 1.002 | [0.990, 1.032] | pass |
| `chat.ttft_ms` | 1927.383 | 1876.164 | 0.973 | [0.955, 1.003] | pass |
| `concurrency.chat-style.tps` | 24.631 | 24.400 | 0.991 | [0.738, 1.100] | diagnostic |
| `concurrency.chat-style.end_to_end_tps` | 10.494 | 22.112 | 2.107 | [1.569, 2.322] | diagnostic |
| `concurrency.chat-style.ttft_ms` | 13753.616 | 1101.108 | 0.080 | [0.077, 0.081] | diagnostic |
| `concurrency.agent-style.tps` | 24.827 | 24.216 | 0.975 | [0.708, 1.104] | diagnostic |
| `concurrency.agent-style.end_to_end_tps` | 22.659 | 22.147 | 0.977 | [0.726, 1.098] | diagnostic |
| `concurrency.agent-style.ttft_ms` | 986.337 | 988.862 | 1.003 | [0.975, 1.010] | diagnostic |
| `concurrency.aggregate_tps` | 21.866 | 48.324 | 2.210 | [1.608, 2.451] | diagnostic |
| `concurrency.aggregate_wall_tps` | 20.980 | 44.203 | 2.107 | [1.568, 2.322] | diagnostic |
| `concurrency.chat.gap_p95_ms` | 78.502 | 83.660 | 1.066 | [0.783, 2.433] | diagnostic |
| `mixed.chat-style.tps` | 29.496 | 8.903 | 0.302 | [0.293, 0.307] | diagnostic |
| `mixed.chat-style.end_to_end_tps` | 2.335 | 8.211 | 3.517 | [3.365, 3.558] | diagnostic |
| `mixed.chat-style.ttft_ms` | 100951.750 | 2407.824 | 0.024 | [0.024, 0.026] | diagnostic |
| `mixed.agent-style.tps` | 25.430 | 25.651 | 1.009 | [0.993, 1.036] | diagnostic |
| `mixed.agent-style.end_to_end_tps` | 2.588 | 2.064 | 0.798 | [0.763, 0.804] | diagnostic |
| `mixed.agent-style.ttft_ms` | 88391.314 | 114080.639 | 1.291 | [1.274, 1.347] | diagnostic |
| `mixed.aggregate_tps` | 24.403 | 4.217 | 0.173 | [0.169, 0.178] | diagnostic |
| `mixed.aggregate_wall_tps` | 4.661 | 4.128 | 0.886 | [0.849, 0.893] | diagnostic |
| `mixed.chat.gap_p95_ms` | 55.481 | 206.240 | 3.717 | [3.242, 4.248] | diagnostic |

### Individual paired samples

| Round/order | 4K engine prefill A/B | 32K engine prefill A/B | 4K engine decode A/B | 32K engine decode A/B | Warm aggregate wall A/B |
|---|---|---|---|---|---|
| 1/AB | 319.65/334.71 | 367.58/393.39 | 27.66/27.86 | 24.78/25.60 | 20.94/44.20 |
| 2/BA | 335.32/331.50 | 381.78/383.88 | 28.17/27.81 | 26.07/25.80 | 20.95/32.86 |
| 3/AB | 335.94/333.25 | 382.66/381.98 | 28.32/28.05 | 25.71/26.04 | 21.28/43.82 |
| 4/BA | 331.43/330.45 | 368.23/380.84 | 27.99/27.86 | 25.64/25.71 | 20.98/44.53 |
| 5/AB | 320.93/330.30 | 368.76/384.47 | 27.91/28.08 | 25.29/25.57 | 21.23/44.21 |

## Loaded chat QoS and ring interpretation

Mixed prefill is a real tradeoff, not a hidden overall win: median mixed batch-wall throughput is **0.886x** A and cold-agent TTFT is **1.291x** A, while chat TTFT falls from **100.95 s** to **2.41 s**. The primary aggregate gate is the predeclared **warm decode** workload, not this mixed batch. Loaded per-stream warm-rate intervals remain inconclusive despite near-solo medians.
- A, chat during agent 32K prefill: median visible-gap p95 **55.5 ms**, isolated p95 **79.2 ms**, median within-round ratio **0.74x**, median per-trial maximum gap **67.1 ms**.
- B, chat during agent 32K prefill: median visible-gap p95 **206.2 ms**, isolated p95 **81.8 ms**, median within-round ratio **2.53x**, median per-trial maximum gap **2235.5 ms**.

**The ~1.5x loaded-gap goal is not guaranteed by this scheduler.** Report the observed ratio above rather than calling aggregate throughput a QoS pass. A is FIFO: its short gaps occur only after the long prefill finishes, while B actually progresses chat during prefill. The default 15 ms service-time slice is per-rank and soft, not an end-to-end 15 ms budget; already-issued frames and ring traversal add delay. A smaller `FG_MULTI_PREFILL_SLICE_US` is a future tuning experiment, not a measured or selected configuration here.

The measured historical ring ledger gives ~37.437 ms token latency, 3.45–4.32 ms rank service, and ~0.8 ms head work. The ideal independent-stream bound is `T_stream >= max(L, N*d_max)`: two streams remain latency-limited, about **26–27 TPS each, ~53 TPS aggregate**, not half the solo rate. These are old-profile ideal estimates, not a mix-04 performance guarantee. Actual warm per-stream rates are in the table. Real overlapping visible decode and aggregate improvement rule out generation-long FIFO serialization; remaining slowdown versus the ideal is not localized to a particular lock by this battery.
Code inspection also confirms `fg_owner_set_active_session` only changes the active-state selector, and owner static replay partitions its eight graph slots between sessions (`src/owner.c`), rather than invalidating every graph on each switch. Coordinator activation does drain static GPU work on an actual slot change (`fg_runtime_activate`); the ready-first scheduler avoids unnecessary same-slot/sleeping-job activation. These facts do not quantify the residual GPU/cache/transport scheduling cost.

Changing prefill chunk shapes can change floating-point trajectories. The cold mixed-agent A/B visible hashes are reported diagnostically below; they are not conflated with the required single-active-session greedy identity or with identical-path warm-session isolation.
The exact first-round cold agent input was additionally rerun alone in B's retained agent slot: A/B visible hash, prompt/generation counts and cold-prefix parity **True**. See `capacity/single-active-agent-parity.json`; this is a safety diagnostic, not a sixth paired throughput sample. It does not by itself prove the cause of mixed-path divergence.
- Round 1 cold mixed-agent visible hash equal: **False**.
- Round 2 cold mixed-agent visible hash equal: **False**.
- Round 3 cold mixed-agent visible hash equal: **False**.
- Round 4 cold mixed-agent visible hash equal: **False**.
- Round 5 cold mixed-agent visible hash equal: **False**.

## Capacity and memory

Phase 1 cold agent: **261871 prompt tokens**, **64 generated**, **731.0 s** client wall, alongside live chat.
Phase 2 chat: **32830 prompt tokens**, with the long agent state retained. Subsequent simultaneous continuations verify both resident prefixes and sustained generation.
Completed: **True**; both contexts near 262K: **False**. Do not interpret a 262K-agent/32K-chat run as two full 262K contexts.
- chat-style: 32929 input, 32894 reused, 64 generated, 23.08 decode-window TPS.
- agent-style: 261970 input, 261935 reused, 64 generated, 22.35 decode-window TPS.
- Near-limit concurrent aggregate wall: 31.92 TPS.

Memory safety: **True**; minimum across all ranks **2.248 GiB available**. Two-second manifest/executable-scoped samples require >=512 MiB available, zero process swap and no OOM/swap-counter growth; observer coverage through completion is checked. DRM allocation and RSS are separate views of shared iGPU memory, not additive totals. QSA state-file inventories cover the pack directory. Raw observations are retained as gzip with raw/compressed SHA256 in `memory-summary.json`.

| Rank | Minimum available GiB | Peak RSS GiB | Peak DRM GiB | Samples | OOM / swap-in / swap-out growth | Safe |
|---|---:|---:|---:|---:|---|---|
| 0 | 2.248 | 0.143 | 11.883 | 418 | 0 / 0 / 0 | True |
| 1 | 3.442 | 0.206 | 10.852 | 418 | 0 / 0 / 0 | True |
| 2 | 4.141 | 0.204 | 10.149 | 418 | 0 / 0 / 0 | True |
| 3 | 3.496 | 0.206 | 10.657 | 418 | 0 / 0 / 0 | True |
| 4 | 3.709 | 0.204 | 10.630 | 418 | 0 / 0 / 0 | True |
| 5 | 3.258 | 0.206 | 10.901 | 418 | 0 / 0 / 0 | True |
| 6 | 4.389 | 0.204 | 9.991 | 418 | 0 / 0 / 0 | True |
| 7 | 3.035 | 0.211 | 11.151 | 418 | 0 / 0 / 0 | True |

## Persistent selection, restart and reproduction

Selected source **`2d0f8a7`**, release **`/home/xander/flash-gordon-live/20261016-abq-multisess`**, **sessions 2**, unchanged mix-04. `selection.json` and `selected/` contain the pinned identity, launcher, all-eight stop/start/verification receipts, model/health responses and actual API reply. For sessions 2, a fresh real two-stream generation is also required after the complete restart. Verification checks live executable and manifest hashes, argv, context/cache settings and the exact allowed FG environment.

Final API verification completed **2026-10-16T01:54:05 UTC-07**, after all-eight
observer cleanup: health `ok`, exact model ID, real greedy reply `OK`.
[API receipt](multisess-fleet-20261016/final-api-health.json);
[all-eight cleanup/hash receipts](multisess-fleet-20261016/final-cleanup.json).

Persistence means retained hash-pinned launchers plus detached serving processes that survive SSH/logout; **no boot-time/systemd auto-restart claim** is made.

On every blade the retained launcher is:

```bash
bash /home/xander/flash-gordon-live/20261015-abq-operator/launch-current.sh start
```

For a ring restart or rollback, use the same fleet transport as this run: stop all eight owned numeric PIDs using the retained launcher, start .43–.49 first, then .42, wait for `http://192.168.42.42:8080/health`, and make a real `/v1/chat/completions` request. The pinned **mix-04 main** fallback is `launch-main-mix04.sh`; it is not the historical Q8 `launch-original.sh`. Exact tested commands are in `protocol/pin-multisess-selection.ps1`.

To repeat the qualification from this Windows worktree (exclusive lease required):

```powershell
$root = 'D:\looking-glass-labs\fg-wt\multi-session-int'
Set-Location $root
$p = 'results\hillclimb\multisess-fleet-20261016\protocol'
New-Item -ItemType Directory -Force .build | Out-Null
Copy-Item "$p\multisess-*.ps1","$p\analyze-multisess-*.py","$p\capacity-*.sh","$p\pin-multisess-selection.ps1" .build
python tools\test_hillclimb_concurrency.py
pwsh -NoProfile -File .build\multisess-fleet.ps1
python .build\analyze-multisess-fleet.py
pwsh -NoProfile -File .build\multisess-capacity.ps1
python .build\analyze-multisess-memory.py
pwsh -NoProfile -File .build\pin-multisess-selection.ps1 -Arm B
```

**Do not overwrite these evidence directories on a new run**: choose a fresh output date and paired salts/time-budget deadline in the archived scripts, preserve the same workload, and install/hash-check the immutable releases first (`protocol/install-multisess.sh`, `install.json`). Deployment uses the archived `ab-fleet.ps1`, `ab-deploy.ps1` and `ab-instance.sh` through the existing `D:\looking-glass-labs\fg-wt\ab-harness` operator setup; credentials are not archived. This experiment must not run concurrently with another lease holder.

Current harness example (candidate, real simultaneous SSE, exact token accounting):

```powershell
pwsh -NoProfile -File tools\hillclimb.ps1 -ConcurrentGeneration -ConcurrentRepeats 1 -ConcurrentTokens 256 -ChatContextTokens 64 -AgentContextTokens 32768 -MixedPrefill -CapacityProbe -PromptSalt fresh-pair-0 -ResultsDirectory results\fresh-mixed
```

`-CapacityProbe` avoids a duplicate isolated cold agent in mixed mode; a 32K argument is still only a 32K test. `-AllowSerializedBaseline` permits FIFO/no-prefix-overlap baseline measurement but cannot manufacture usage: missing usage leaves the result incomplete until exact request-ID server-log reconciliation.

Six local harness fixtures pass, including simultaneous SSE, mixed prefill, single-cold capacity protocol, rejected short generation with retained raw data, deferred missing usage and deterministic paired salts. No C/Vulkan arithmetic or pack files changed in this qualification. Experiment memory observers are stopped and scratch cleaned before lease release.
