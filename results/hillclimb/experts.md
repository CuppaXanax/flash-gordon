# Expert decode kernels

2026-10-15. Branch `cuppaxanax/expert-kernels`, based on main `1863c95`.

**Kernel improvement, not a qualified serving deployment.** Cooked Q8 down is approximately 43% faster; ordinary Q5_1 down improves approximately 3%. The measured format-weighted synthetic saving is **0.300 ms/token**, not the roadmap's 1.0-2.2 ms target. No production rank, model pack, quantization, service configuration, CU mask, or clock setting was changed. Do not claim a 1 ms client-wall improvement.

## Hardware and scope

Only `192.168.42.50`, hostname `bc250-59b36628`, was contacted. All experiment files were under `/home/xander/hc-experts/`, subsequently removed. `bc250-embed` and `bc250-rerank` stayed active; both `/health` endpoints returned `{"status":"ok"}` before final cleanup and afterwards. Their observed main PIDs were 1396/1397. No service was stopped or restarted.

**Hardware discrepancy:** the requested blade was described as 40 CU, but `fg_vk_active_compute_units()` reported **24 active CUs** throughout the final measurements. That value comes from Vulkan AMD shader-core properties, not an inferred throughput ratio. The ISA capture confirms subgroup **64**; accelerated integer dot is disabled. These are **not verified 40-CU results**. No attempt was made to change the blade to 40 CUs or to use another blade.

Approximately 11.8 GiB system memory was available around measurement; services occupied approximately 2.39 GB GTT. The largest benchmark has about 2.045 GB of synthetic cooked GPU weights and a corresponding CPU construction copy. It fit without stopping services. The real `/home/xander/fg-ring-pack-abliterated/manifest.fgm` was unavailable on this permitted blade, so these are synthetic, production-shape/format measurements, not timings of real layer payloads or a real-model correctness qualification.

## What already exists

`src/expert.c:fg_expert_decode_chain` already records GPU router top-10, GPU tile scheduling, **one gate/up/SwiGLU dispatch for all ten selections**, and **one down/gate-reduction dispatch**. Gate/up uses a two-dimensional grid: 80 row tiles x 10 routing slots. There are not thirty independent projection launches to fuse. Shared expert work is separate and unchanged.

Each selected expert has gate and up matrices **K=2560, M=640**, and down **K=640, M=2560**. Ten selections therefore read:

| Layer family | Gate/up/down | Gate+up bytes | Down bytes | Total bytes/layer-token |
|---|---|---:|---:|---:|
| 43 ordinary layers | Q4_K/Q4_K/Q5_1 | 18,432,000 | 12,288,000 | 30,720,000 |
| Layer 2 | Q5_K/Q5_K/Q8_0 | 22,528,000 | 17,408,000 | 39,936,000 |
| Layers 4, 30, 46, 47 | Q4_K/Q4_K/Q8_0 | 18,432,000 | 17,408,000 | 35,840,000 |

This reproduces the historical **1,504,256,000 B/token** expert ledger using the existing cooker size functions and the documented layer-format inventory. It is not a new live-pack identity check. All four Q4/Q8 exceptions have the same shape and kernel; they share one measured format case, rather than four invented independent layer measurements.

The chained owner path does **not** read routing back or fence per expert/layer. Legacy standalone expert submit/finish can wait and read the reduced output, but removing that readback is not a new optimization for the production chain. The normal static owner graph and dedicated fixed expert graph retain their two projection dispatches. No new host round-trip, intermediate output allocation, or reduction dispatch was added.

## Implemented changes

1. **Cooked Q8 down:** `fg_moe_decode_down_q8.comp` gives a 64-thread WG two output rows. Each row has four independent block walks, eight lanes per walk, and one four-value quant/input word per lane. Quant addresses cover consecutive 128-byte row runs across the four walks. Routing slots retain their original accumulation order; block/reduction reassociation remains FP32. Five subgroup shuffles finish the row, with no LDS or barrier.
2. **Q5_1 down:** retain the existing paired-row, 32-thread geometry and exact arithmetic order, but unroll two owned blocks instead of four. This is a small measured instruction-scheduling win, not a representation or precision change.

The new Q8 grid has **1,280 WGs instead of 320**, giving more independent work for latency hiding. The old requested 32-thread WG uses half of the observed wave64; the new Q8 WG uses all 64 lanes. Capability checks require compute-stage subgroup shuffle and subgroup size >=32; other devices retain the old Q8 path. Raw Q8 fallback is unchanged. Both dynamic chained dispatch and fixed expert-graph recording use the new selection/grid.

The final **gate/up SPIR-V is byte-identical to baseline**: SHA256 `22a4450d0ef8f4566997b5aa7f145eb7557bd94b25eb194b07c95b5671541398`. Its existing cooked K-quant SoA metadata, eight-row/64-lane geometry and vector loads remain. No runtime or sealed on-disk repack is needed: every existing weight byte, format tag and original half-scale bit pattern remains unchanged. Unlike dense compacting, expanding native expert Q4_K from 18 to 18.5 B/32 would increase traffic; the dense layout is not automatically a win here.

ISA evidence (`expert-isa*.log.gz`) shows gate/up's highest vector register is v83, Q5 down's v95 both before and after, and standalone Q8 down's v20. These are disassembly register high-water marks, not hardware occupancy counters or exact rounded allocation sizes. The original monolithic Q5/Q8 shader retains Q5's register footprint when used for Q8. No native integer-dot or scratch load/store instruction appears. The workgroup counts, full-wave usage, short block walks and lower standalone Q8 register footprint explain the occupancy/latency rationale; no measured occupancy percentage is claimed.

## Measurement protocol

Three final independent runs, **30 baseline and 30 candidate measurements per format per run**, with a full **512-expert slab** and ten unique, deterministic random selections. Pair slot mapping is permuted. Each baseline/candidate pair sees identical selections and inputs. ABBA order alternates the first kernel variant.

Each format gets a fresh Vulkan process/context and 1,024 continuously batched warm gate/down pairs, alternating variants. GPU timestamps bracket each dispatch; each sample asserts exactly two dispatches. CPU recording, submission, readback and the subsequent scalar oracle are excluded from reported kernel times. Both variants run through the same instrumented benchmark executable; only their selected pipeline and Q8 grid differ. Benchmark-only instrumentation lives in `tools/expert-measure.py`, never in production sources.

Clocks were not locked. Absolute times vary between runs and even between format cases; this is why comparisons are matched/alternating, not separate cold process timings. The main table averages all **90 samples per variant/format**, not the fastest pass. Effective decimal GB/s is the unchanged selected cooked weight payload divided by summed GPU kernel time, not a memory-controller counter. Shared expert/router/scheduling costs are excluded.

| Family | Gate/up before/after us | Down before/after us | Pair before/after us | Pair before/after GB/s |
|---|---:|---:|---:|---:|
| Ordinary Q4_K/Q4_K/Q5_1 | 77.118 / 77.375 | 55.214 / 53.469 | **132.332 / 130.844** | **232.1 / 234.8** |
| Layer 2 Q5_K/Q5_K/Q8_0 | 84.118 / 83.924 | 104.211 / 59.523 | **188.329 / 143.447** | **212.1 / 278.4** |
| Layers 4/30/46/47 Q4_K/Q4_K/Q8_0 | 77.933 / 77.737 | 108.891 / 61.424 | **186.824 / 139.160** | **191.8 / 257.5** |

Gate/up differences are noise: the shader is identical. Q8 down alone reaches **292.5 GB/s** with Q5 gate/up and **283.4 GB/s** with Q4 gate/up. Q5_1 down reaches **229.8 GB/s**, versus **222.6 GB/s** baseline.

Repeatability, per-format **pair** times in us, baseline -> candidate:

| Run | Ordinary | Layer 2 | Q4/Q8 exceptions |
|---|---:|---:|---:|
| 1 | 145.105 -> 143.224 | 188.187 -> 143.343 | 178.024 -> 133.112 |
| 2 | 125.892 -> 124.665 | 188.519 -> 143.664 | 181.579 -> 135.632 |
| 3 | 125.997 -> 124.640 | 188.283 -> 143.336 | 200.869 -> 148.737 |

Summing 43/1/4 family means gives a **synthetic** 48-layer expert time of **6.626 -> 6.326 ms**, approximately **227.0 -> 237.8 GB/s**, saving **0.300 ms**. This is neither a new eight-rank measurement nor a prediction that normal client time improves by precisely 0.300 ms. Do not combine it additively with the intrusive profile's 7.517 ms or normal wall ledger. The roadmap's <6.1 ms diagnostic target and >=1.0 ms normal client-wall gate are **not established**.

## Correctness

* Every timed sample has a full 2,560-row independent scalar down oracle, using that sample's GPU gate/up output, cooked source decoder and routing map. It accumulates a double-precision reference and `sum(abs(weight*activation*gate))`; require error <= `2e-5*max(1, sumabs)`. The final three runs check **1,382,400 row outputs**. Worst conditioned scalar error is **1.704e-7**.
* Current-kernel/candidate parity checks every row for every matching sample with a `4e-5*max(1, sumabs)` two-result bound; worst is **8.657e-8**. Ordinary Q5_1 results are additionally **exactly equal** to baseline for every row.
* A new one-hot test checks **all 640 x 2560 dequantized weights**, separately for Q5_1 and Q8_0: **3,276,800 exact equalities** against independent raw-source decoding. This distinguishes exact weight preservation from tolerated accumulated-result reassociation.
* Final uninstrumented sources build `flash-gordon`, the Vulkan harness and core tests. Core passes, including existing cooked Q4_K/Q5_K/Q8/Q5_1 preservation tests. Its intentional negative fixture prints `VERIFICATION FAILED: 5 pass, 1 fail` before the actual `core tests: PASS`.
* Targeted Vulkan gates pass: exact down; fused K and raw/cooked Q8 down; mixed/random routing with dead slots and negative gates; production-shape fused parity; fixed fused graph/replay; K-quant cooking; Q5 cooking; GPU scheduling; static replay; batch and pipeline-flush parity. See `expert-validation.json`.

An early `1e-3*max(1,abs(output))` baseline comparison failed near cancellation (for example -3397 vs -3406 with very large opposing products). That failure is retained in exploratory JSON. It was not hidden by relaxing a relative threshold: the final independent scalar/product-sum oracle provides a conditioning-aware bound, while exact one-hot dequantization remains mandatory. Existing small/production-wide legacy comparisons also pass their original tolerances.

No real-model token/logit, KLD, long-context, prefill, TTFT or client-wall result is claimed. The source quantization formats did not change, but FP32 Q8 reassociation still needs serving qualification before deployment.

## Rejected experiments and incidents

* Replacing gate/up LDS with an identical-tree subgroup reduction, and specializing homogeneous K types, gave no meaningful matched gain. Both were removed.
* Q8 eight-lane rows without block partitioning remained about 102 us, including an exactly equivalent 32-thread/four-row variant. Two block walks gave about 64 us; four walks won at about 59-64 us.
* Single-row Q5 down with more block parts regressed (approximately 57/69 us versus 52 us). Combining its code with Q8 also slowed Q8, consistent with a compiler/resource interaction, not a proven VGPR explanation.
* Q5 unroll-one regressed approximately 29%; unroll-two consistently beat original unroll-four by a small margin. Q5's register high-water mark did not fall.
* Two initial **unmodified-baseline** multi-format/shared-context benchmark attempts hit a guilty-context gfx-ring timeout/reset when reaching Q5 after Q4 cases. Root cause was not established. Fresh per-format processes eliminated the failure; no further timeout occurred in final measurements or validation. This is not claimed as a fixed engine bug. Both existing services remained active and later health checks passed; they were not restarted.

Exploratory 32-expert timing captures are retained but are not the final full-slab evidence. Early separate-process clock-ramp comparisons and combined instrumented summary lines are not used to claim a speedup; final aggregation uses only `EXPERT_SAMPLE` timestamps.

## Evidence and reproduction

Tracked: `expert-512-{1,2,3}.json`, `expert-summary.json`, `expert-validation.json`, `expert-{hardware,health,artifacts,isa,cleanup}.json`, compressed original/candidate ISA, and exploratory captures. Local raw captures are retained in `expert-evidence.tar.gz` (not committed). `python tools\expert-summary.py` reconstructs the summary and checks all sample counts.

From the worktree, with only the assigned blade:

```powershell
$fleet = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
$target = @('192.168.42.50')
$root = 'D:\looking-glass-labs\fg-wt\experts'
git archive --format=tar --output=source.tar 1863c95 include src tests shaders vendor Makefile glslc
tar -cf candidate.tar shaders src\vk.c tests\test_fg_vk.c tools\expert-measure.py tools\expert-paired.sh tools\expert-validate.sh
& $fleet -ScriptPath .\tools\expert-preflight.sh -Targets $target
& $fleet -ScriptPath .\tools\expert-preflight.sh -Targets $target -UploadPath "$root\source.tar" -UploadDestination /home/xander/hc-experts/source.tar
& $fleet -ScriptPath .\tools\expert-baseline.sh -Targets $target -UploadPath "$root\candidate.tar" -UploadDestination /home/xander/hc-experts/candidate.tar
& $fleet -ScriptPath .\tools\expert-paired.sh -Targets $target
& $fleet -ScriptPath .\tools\expert-validate.sh -Targets $target
& $fleet -ScriptPath .\tools\expert-isa.sh -Targets $target
& $fleet -ScriptPath .\tools\expert-collect.sh -Targets $target -DownloadSource /home/xander/hc-experts/evidence.tar.gz -DownloadPath "$root\results\hillclimb\expert-evidence.tar.gz"
& $fleet -ScriptPath .\tools\expert-clean.sh -Targets $target
```

The baseline script refuses an existing baseline directory. Remote scripts reject a host without the assigned `.50` address. Validation redirects legacy core fixtures into this workspace, restores edited test sources, and uses a workspace-local shader cache. Benchmark instrumentation is restored on exit. Use absolute local transfer paths: the fleet runner's .NET path resolution does not follow PowerShell `Set-Location`. Preserve evidence before cleanup; do not alter production services or use another blade to bypass the CU discrepancy.
