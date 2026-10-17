# MTP handoff: measured state, blockers and next-session plan

**PARKED for this session. No more leases or host access; default off, no promotion.**
Implementation/evidence tip before this handoff: `277e2bf`, branch `cuppaxanax/mtp-serving`.
Production remains source `31508ee`, binary SHA prefix `02930a164e38`, sessions2/wide1:
native prefill, then resident-state N1; `FG_SMALL_KERNELS` is absent/unsupported there.
All .42–.49 restored/verified/clean/released **2026-10-17T01:50:21Z**; .51 released01:50:29Z.
Target weights must remain **≥Q4**; draft-only lower precision is allowed, not acceptance-qualified.

## 1. Current measured state

Authentic a038dec A0/F3/A1: same binary/target, full Q8 draft head, fixed k3,
single session, mask330,440MiB cache; six requests/category, **not six independent arm pairs**.

| Category | Off decode TPS | MTP decode TPS | Median paired ratio | Tokens/verify | Accepted/proposed |
|---|---:|---:|---:|---:|---:|
| Code |31.200 |33.409 |**1.074×** |3.386667 |0.811659 |
| Chat |31.063 |22.378 |**0.736×** |2.219512 |0.418033 |

Warm performance-round means, including N2/N3 generation tails:

| Stage, ms | Code | Chat |
|---|---:|---:|
| Draft chain |9.853 |9.851 |
| Checkpoint preparation |9.832 |9.831 |
| Subtract draft/checkpoint overlap |−9.832 |−9.831 |
| Verify |89.968 |90.176 |
| Target retain / native repair |0.259 /0.884 |0.561 /0.527 |
| Commit / host |0.285 /0.031 |0.260 /0.025 |
| **Total** |**101.280** |**101.401** |

N4-only code owner compute65.160ms; input+compute+read67.667ms; verify residual22.629ms.
That residual includes transfer/head/control, **not isolated wire**. All-shape compute64.979ms.
Wire anchor14.216±0.440ms is an earlier measured-frame pair→eight-work-hop model.
Parallel target-head compute: maximum≈3.096ms, **not its12.203ms summed across ranks**.
Latest selected six-layer synthetic replay **N4/N1=2.167×**, not a measured new ring ratio.
Intrusive dense/shared2.426×, HC2.356×; real draft expert-union bytes remain **unmeasured**.
Cold retain/restore is repaired;775 authentic ledgers and2,167 pass-local argmax choices align.
Adaptive true-serial k0 fallback exists; chat≥1.0×/retained code win remain **unmeasured**.

## 2. Blockers ranked by potential ms/round

Quality, genuine capture rehearsal and serial-fallback noninferiority are prerequisites, not savings.
Budgets below are **hypotheses**, disjoint only as defined; none is a credited fleet improvement.

| Rank | Blocker / measured scope | Conditional saving | Next evidence |
|---|---|---:|---|
|1 |Owner compute64.979ms |≈25ms to40ms; dense/HC5–15ms is a **subset** |Real unions, counters, warmed per-owner replay |
|2 |Work transport model14.216ms |All-hop FP16≈7ms optimistic; mask224≈2–3ms; F32→11ms saves3.216ms |Mutually alternative wire budgets; N4 codec/quality gates |
|3 |Draft chain9.853ms |≈5.353ms to4.5ms |Full-vocabulary lower-bit draft kernels, unchanged-yield test |
|4 |Other verify/control and post-verify |≈3.8ms:3.6 within verify +0.2 post |Disjoint timestamps excluding owner/head/wire |
|5 |Pipelined future-window drafting |≈2ms **after** low-bit draft |Measure useful rank7 idle window, waste/queueing and branch matches |
|6 |Target head maximum3.096ms |≈2ms |N-column Q8 head/argmax; native draft calls remain sequential |
|7 |Further GDN partitioning |No robust additional saving |Selected state path already zero-spill; deprioritize |

GU/down byte growth may be physical; shared-weight dense/HC gaps need issue/occupancy evidence.
Planning proxy only: `T4_us=437.84*R_GU+296.16*R_down+1.1*(854.20+434.44)+671.24`;
`N1_us=2693.88`. Assumed union3× gives1.593×, or1.735× with current-other costs.
These are **not measured unions or hardware floors**; never rescale the101ms ring by this proxy.
Weight loads/dequant already shared; zero spills. Smaller HC/dense tiles regressed0.56%/4.95%.
HC scale hoisting gave B/A1.001426×, VALU1034→1098, VMEM46 unchanged: rejected.
Token-parallel HC epilogue confirmed only0.50% median benefit and remains unselected.

## 3. Capture plan: trained cold-prefill rehearsal FIRST

Previous CLI checks,960 all-layer runtime-double records and four HTTP-double cases were insufficient.
The latest first real request had only **≈2 seconds**: staging finished01:48:58.251Z,
work deadline01:49:00Z. Its timeout is **not evidence of a diagnosed prefill/kernel deadlock**.
The last log completed coordinator layers6..11; zero real verifier routes/HTTP cases completed.
The private `.build/mtp-union-lease/capture.ps1` now blocks `Run`; keep it blocked until ready.

1. Next session, prepare artifacts before any production lease. Rehearse the **exact** launcher,
   API/workers, trained weights, fresh session/cold prefill, native warmfill, first N4 and capture client
   on an isolated trained cold-prefill ring. Compute/protocol doubles do not satisfy this gate.
   If that ring cannot be provisioned, report the blocker; do not substitute mocks or consume a lease.
2. Parameterize both HTTP and outer-process timeouts. Reserve **≥180s initially per cold request**
   (planning allowance, not measured cold TTFT), covering prefill/native initialization/first N4/output.
   Then use measured cold p99 plus explicit margin. The historical first cold N4 alone was1.405s.
   Never truncate this to leftover seconds: refuse to start unless cold budget + restoration reserve fit.
3. Require completed chat **and** code cases, receipt IDs/session bindings and all48 layers/window.
   Run `make test-mtp-capture test-mtp-capture-ring`; these are prerequisites, not the trained rehearsal.
4. Use `--mtp-capture-dir` on API **and every worker**; validate exact CLI options before stopping anything.
   Save `verify-routes-SESSION-LAYER.jsonl`, manifest/source/flags/head/cache hashes and case bindings.
   `tools/analyze_mtp_union.py` excludes restore replays and byte-weights unique experts using exact GU/down
   strides. Teacher sliding windows, synthetic IDs and draft-router IDs are not real verifier unions.
5. Only after rehearsal, seek fresh explicit user-direct authorization next session; none exists now.
   Bound staging/capture/restoration separately; retain first errors, watchdogs and exact restore checks.

## 4. Obtain counters before another kernel sweep

The queried RGP/rocprof executables were not on .51's PATH; **no VALU/memory busy counters were measured**.
Install GPUOpen RGP and validate Mesa RADV SQTT/headless-compute capture support on this GFX1013 device.
Install a compatible `rocprofv2`/`rocprofv3` stack only if it exposes this GPU/workload's counters:
HIP/HSA tracing alone does **not** profile Vulkan dispatches. Preserve drivers/governors; no hardware tuning.
Request per-op VALU/VMEM activity, cache/DRAM bytes, stalls and achieved waves/occupancy; name counter units.
Fallback: add opt-in Vulkan timestamp queries around dense/GU/down/GDN/QSA/HC/head dispatches,
checking `timestampValidBits`/`timestampPeriod`, reading results after drain without per-op host waits.
If supported, enable pipeline-statistics queries for compute invocations and enumerate
`VK_KHR_performance_query` counters/profiling-lock requirements for actual activity metrics.
Invocation counts/timestamps and compiler `Subgroups per SIMD` are **not** busy percentages/achieved occupancy.
Separate instrumented attribution from uninstrumented warmed A/B/A replay; keep12 component oracles.

## 5. Expected TPS per step: conditional model, NOT new measurements

Use `TPS=1000*yield/round_ms`, fixed code/chat yields3.386667/2.219512 and the code round budget.
These differ from per-request measured medians. Every saving and unchanged yield below must be tested.

| Sequential hypothetical step | Round ms | Code TPS | Fixed-k3 chat TPS |
|---|---:|---:|---:|
| Measured-round/yield model |101.280 |33.44 |21.91 |
| Owner compute→40ms |76.301 |44.39 |29.09 |
| Target head−2ms |74.301 |45.58 |29.87 |
| Full-vocab lower-bit draft→4.5ms |68.948 |49.12 |32.19 |
| Useful pipelined speculation−2ms |66.948 |50.59 |33.15 |
| Other verify/post-control−3.8ms |63.148 |53.63 |35.15 |
| Alternative A: F32 work transport→11ms |59.933 |56.51 |37.03 |
| Alternative B: mask224 saves≈2.2ms from63.148 |60.948 |55.57 |36.42 |
| Alternative C: all-hop FP16 work→7.1ms from63.148 |56.033 |60.44 |39.61 |

A/B/C are alternatives, **not additive**. Mask224's2.2ms is4×the N1 paired552µs model, not an N4 result.
Code60TPS needs≤56.444ms at current yield;60ms alone is insufficient. Fallback first targets≈31chat/
≈33code TPS without regressions, not guaranteed gains. Restricted vocabulary lost4.184% code yield offline;
prefer the full head except for required sessions2 fit. Low-bit draft yield retention is also unmeasured.

## 6. Pipelined speculation, mask224 and promotion gates

Pipeline rank7 drafting while ranks0–6 verify, in a **separate private speculative native-state bank**.
Predict the future bonus/pending token too; full draft acceptance alone does not reveal the target bonus.
Reuse proposals only for matching accepted prefix/pending/branch/session/version; otherwise discard/rebase.
Target GDN/QSA rollback and actual retained HCs remain authoritative; all emissions require target verification.
Prioritize arriving target work, yield between draft steps and measure wasted work/target queueing.
Credit at most the useful remaining≈2ms after reducing draft to4.5ms, not another9ms overlap win.

Precision peer `cuppaxanax/bf16-hops` (`9531f7e`, reconciliation `8a148ed`): mask224 selects destinations5–7.
It passes **incremental vs F32** KL0.000679/top1 100%; all-hop FP16 fails KL0.009877/top1 98.75%.
Mask224 is **not cumulative-original-Q8/N4 qualified**. Existing codecs exclude MTP and fail closed with it;
integration needs batched/per-slot scale framing, native-owner checks and fresh N4 quality—not flag toggling.
Its3-hop scope cannot claim the optimistic all-hop≈7ms saving.
Original-Q8 N4 gates: KL≤0.005/top1≥98.5%, pass-local argmax, same-prefix top2 divergence margins<0.10.
Natural480 selected-native baseline passes0.003084/98.75%; Forced80/all560 failures remain; no KL subtraction.
Full-head sessions2 fits offline only at128MiB cache, not qualified at440MiB; eager-warm every new private bank.
Promotion needs actual cached-serving quality, state/memory/cancel/coexistence gates and **six balanced AB/BA
arm pairs**, chat≥1.0× and retained code gain. No promotion or lease request this session.

Evidence: [paired ledger](../results/hillclimb/MTP_RETAIN_SERVING.md),
[block/union work](../results/hillclimb/MTP_UNION_KERNELS.md),
[capture/hoist failure and rehearsal limits](../results/hillclimb/MTP_CAPTURE_REHEARSAL.md);
precision details: canonical `results/hillclimb/FP16_HOPS.md` and `QUALITY_RECONCILIATION.md`.
