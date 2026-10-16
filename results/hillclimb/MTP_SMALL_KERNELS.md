# Small-N verification kernels

Follow-up: [MTP_BANDWIDTH.md](MTP_BANDWIDTH.md) records the subsequent
expert-union/QSA/bandwidth research and its newer N2 <= 1.2 / N4 <= 1.6
stopping targets. The tables below retain their original qualification and
historical thresholds; no newer research flag is implicitly promoted by them.

## Result and status

**Useful improvement, target not reached.** The uniform research selection is
`FG_SMALL_KERNELS=330` (`256+64+8+2`), still **default off**. It combines an
eight-row/cohort large dense kernel, FP32 expert FMA/subgroup reductions,
register-cached recurrence, and resident-prefix causal QSA batching.

Ratios are six-layer block latency divided by the unchanged N1 reference.
Intervals are paired five-round 95% Student-t confidence intervals, not
confidence intervals over individual, correlated dispatch samples.

| Blade | N1 reference, us | N | Baseline ratio | Mask 330 ratio | Target |
|---|---:|---:|---:|---:|---:|
| .50, live 40 CU | 2673.15 ± 5.74 | 2 | 1.417 ± .032 | **1.322 ± .050** | 1.15 |
| .50 | same | 3 | 1.850 ± .049 | **1.706 ± .030** | 1.30 |
| .50 | same | 4 | 2.283 ± .037 | **1.968 ± .082** | 1.45 |
| .53, driver reports 24 CU | 2945.57 ± 3.65 | 2 | 1.468 ± .014 | **1.360 ± .005** | 1.15 |
| .53 | same | 3 | 2.012 ± .026 | **1.889 ± .030** | 1.30 |
| .53 | same | 4 | 2.534 ± .068 | **2.199 ± .122** | 1.45 |

The .50 CU count was independently exposed by its live manager. The .53 live
manager/override was unavailable: **reported 24 CU is not proof of an independently
verified 24-CU restriction**. No CU or clock restriction was applied for these runs.

| Blade | N | Before, us | After, us | Paired delta, us | Reduction |
|---|---:|---:|---:|---:|---:|
| .50 | 2 | 3787.12 | 3532.51 | -254.61 ± 193.67 | 6.68% ± 4.99% |
| .50 | 3 | 4944.16 | 4561.11 | -383.06 ± 190.71 | 7.70% ± 3.73% |
| .50 | 4 | 6101.97 | 5260.82 | -841.15 ± 187.74 | 13.79% ± 3.10% |
| .53 | 2 | 4325.44 | 4007.00 | -318.44 ± 28.11 | 7.36% ± .59% |
| .53 | 3 | 5925.99 | 5563.40 | -362.59 ± 151.58 | 6.10% ± 2.48% |
| .53 | 4 | 7463.42 | 6476.88 | -986.54 ± 443.05 | 13.17% ± 5.59% |

For comparison, mask `74` omits dense row reuse. Its N2/N3/N4 ratios are
1.295±.028 / 1.730±.072 / 2.105±.096 on .50 and
1.376±.039 / 1.844±.053 / 2.395±.075 on .53. There is no claim that mask 330
wins every individual shape; its clearest additional benefit is N4.

## Measurement

* Replay-faithful synthetic compact-Q4 six-layer fixture, layers 6..11, about
  9.739 GB resident. These are not trained-model end-to-end serving measurements.
* Five seeds, alternating `A/B/A/C/A` order, reversing B/C each round:
  `A=0`, `B=74`, `C=330`. Each candidate is compared with its two adjacent A runs.
* All N start after **264 tokens**, with three windows and `72/N` iterations:
  216 measured tokens each. This avoids the old N3 warmup truncating 256/3 to
  255 and exercising a different QSA boundary.
* Primary statistic: median block time within each run; mean of the five
  paired run statistics. N1 is bracketed before/after each round. CI:
  `2.776445105 * stdev(five paired statistics) / sqrt(5)`. All-sample run means
  and sample counts are retained too; first static-capture/host outliers are
  not silently removed from those means.
* `--no-clocks`, no profiling during wall qualification. Existing dynamic
  governor/configuration was retained; this is **not fixed-2000-MHz evidence**.
  Services were stopped only when originally active and restored afterward.
* An earlier .50 diagnostic used a pinned governor profile. That run is
  **quarantined and excluded from every qualification result above**; the
  original daemon/configuration was restored and independently checked.

The original 1/1.41/1.80/2.41 physics figures are historical, not the paired
baseline denominator here. `mtp-small-kernels.json` retains all five paired
rounds, oracle receipts, requested flags, and per-kernel diagnostic profiles.
`tools/analyze-mtp-kernels.py` reproduces the main tables from retained fleet
logs; `tools/mtp-kernels-qualify.sh` defines the exact A/B ordering.

## Implementations and rejected experiments

| Bit | Implementation | Disposition |
|---:|---|---|
| 1 | Two rows per 32-lane cohort; Q4/Q5/IQ4 dequantization reused across N; cooked Q8 N-column weight reuse | Component-qualified, optional; not in mask 330 |
| 128 / 256 | Four/eight rows per cohort, no LDS reduction, explicit vector accumulators; M,K >=1024 | R4 explored; R8 in mask 330 |
| 2 | WG512/head GDN; 32 state values/thread retained across all N causal updates; one final state store | In mask 330 |
| 4 | One-dispatch union-down with per-token gates and no global parts | Correctness passed; roughly 2x slower in exploration, not selected |
| 8 | Commit N rows in order, gather maximal resident prefix once, batched split/merge with per-query causal counts | In mask 330; short-prefix only |
| 16 / 32 / 64 | Subgroup GU reductions / next-block prefetch / scalar FP32 FMA GU and down | 64 selected; prefetch not promoted |
| 512 | Packed FP16 GU accumulation | **Rejected component precision**, not a qualified setting |
| 1024 | Exact software-packed integer GU products, two signed 16-bit lanes | CPU/component-qualified but slower, not selected |

All dense variants issue one family dispatch for all N and load each decoded
weight for reuse across N accumulators. R8 keeps the existing lossless
compact/SoA layouts; small HC/low-M shapes retain their legacy geometry unless
bit 1 is explicitly selected. Workgroup 32/64/128/256 exploration did not justify
replacing R8's WG128 default. These are 32-lane reduction cohorts, not a claim
of unsupported hardware integer-dot or mixed-FP16-to-FP32-dot acceleration.

Expert GU retains the union schedule and per-token/pair association. The
selected down path retains split partials and reduction; the one-dispatch union
experiment is available only behind bit 4. No selected-path union-down speedup
is claimed.

QSA batching is eligible only when the entire maximal visible prefix is
resident and at most 2051 records. Each query's original causal count clips
the shared gather: committing later rows does not make them visible early.
Cold, indexed/long-context, and eviction cases keep the existing causal path.
**Long-context QSA batching remains an optimization gap.**

Representative stock RADV shader statistics:

| Kernel | VGPRs | Spilled VGPRs | Scratch bytes | LDS bytes |
|---|---:|---:|---:|---:|
| Previous GDN small | 256 | 130 | 33792 | existing |
| Cached GDN | 128 | **0** | **0** | 6656 |
| R8 compact Q4 N4, WG128 | 100 | 0 | 0 | 0 |
| Four-pair GU FMA | 84 | 0 | 0 | 4096 |
| Four-pair packed-integer GU | 64 | 0 | 0 | 1024 |

The packed helper in `shaders/fg_small_packed_dot.glsl` is exact for its
integer partials: at most eight Q5-by-signed-Q8 products per 16-bit lane,
`8*31*128=31744`; low-lane sign restores the high-lane borrow. It needs no
hardware dot extension. Nevertheless its stock A/B/A intrusive GU profiles
were roughly 7–17% slower, so fewer VGPRs/LDS were not treated as a win.

Five additional paired A/B/A **intrusive N4** rounds give this diagnostic
attribution (all values us, delta CI95):

| Family | .50 before -> after | .50 delta | .53 before -> after | .53 delta |
|---|---:|---:|---:|---:|
| Dense | 3507.46 -> 3145.38 | -362.07 ± 334.55 | 4266.23 -> 3925.36 | -340.87 ± 391.53 |
| Expert GU | 1223.78 -> 1280.42 | +56.64 ± 192.04 | 1719.25 -> 1926.68 | +207.43 ± 248.56 |
| Expert down | 483.51 -> 485.77 | +2.26 ± 31.16 | 519.35 -> 562.11 | +42.76 ± 56.06 |
| Recurrence core | 194.82 -> 189.48 | -5.34 ± 20.51 | 190.31 -> 177.44 | -12.87 ± 21.31 |
| QSA non-GEMV | 636.70 -> 145.20 | **-491.50 ± 65.06** | 542.28 -> 159.58 | **-382.70 ± 28.23** |

QSA is the strongest isolated win. There is **no demonstrated expert GU/down
speedup**, and eliminating GDN scratch did not establish a large N4 latency win.
Mask 330 is the tested bundle with the best N4 wall result, not proof that
every bit in that bundle is individually beneficial. Further GU/down ablation
and large-shape arithmetic/bandwidth work remain necessary. Profiling changes
serialization and operating point; these sums are not a decomposition of the
unprofiled wall times in the first table.

FP16 inputs/quant integers are representable, but FP16 **accumulation is not
exact**. Its independent mixed-Q4/Q5 CPU oracle reached normalized error
`7.58543611e-4`, exceeding `1e-4`, including cancellation/extreme-input cases.
It remains explicitly rejected/default off; no tolerance was relaxed to admit it.
The enabled-half Vulkan feature chain is requested only with bit 512.

## Gates and limits

* Independent CPU GU oracle: production hidden 2560, mid 640, top-10,
  full 512-expert-addressable slabs, both Q4/Q5 directions, N2..4, shared and
  differing per-token unions/gates. Random, extreme signed-byte, cancellation,
  and nonzero second-Q8-header-word fixtures. Outputs are poisoned with NaNs
  before each dispatch so missing stores cannot reuse earlier passing values.
  Original/FMA/packed-integer
  maximum normalized error `6.14238307e-6`; FP16 fails as above.
* Dense Q4/Q5/IQ4/Q8 and full N-column output/state versus N sequential N1
  component gates use `1e-4 * max(1,abs(reference))`, with finite checks.
  Exact output/state is no longer incorrectly required for reassociation.
* Final mask-330 causality boundary runs on **both blades** at prefixes
  3,255,256,2047,2048,2049,2050,2051,2052 pass N2..4 output/state tolerance
  and exact unchanged-prefix checks. Worst selected-mask output absolute drift
  is `3.10540199e-5`. GDN absolute state drift reaches `8.77380371e-4` on .50
  at large state magnitudes; it passes the stated **relative** gate, not an
  invented `1e-4` absolute-state requirement.
* Session isolation, repeat, union schedule, cleanup, and transactional
  restore/replay checks pass. Discrete counts and causal/rollback invariants
  retain their exact gates.
* Flag 0 output and GDN-state hashes match the preserved pre-change executable
  **bytewise for N1..4**, using identical deterministic inputs.
* Rebased onto serving `9b93714`; local C17/Werror build and flag registry pass.
  Integrated GPU rebuilds compare qualified small-column/cached-GDN SPIR-V
  hashes and rerun component/session/rollback gates.
* Final output-poisoned CPU tests pass masks 0,64,330,1024 on both blades.

Neither the requested 1.15/1.30/1.45 targets nor an 85%-of-bandwidth claim is
established. Arithmetic/dequantization, expert work, long-context QSA, and
remaining tiny dispatches still leave substantial excess. Intrusive profiles
are diagnostic and alter serialization/operating point; unpaired profile
differences are **not** used to manufacture a wall-time or bandwidth attribution.

No default enablement, trained-model greedy equivalence, blade-pair serving
qualification, or whole-ring speedup is claimed. Before promotion, run the
real-pack KL/top-1 gate (mean KL <=~0.005, top-1 >=98.5%) and serving greedy/
transaction qualification with sol-mtp's orchestration.

## Final restoration

Both original governor services are active; configuration SHA256 on both is
`a5e83975e4fbea57d6110cbf4aa81fd58367d044c2bc602815ff0ca21ede6757`,
matching the original copies. The .50 live manager still reports **40/40**
active/routed CUs. Its originally active embed/rerank services are active and
both health endpoints return `{"status":"ok"}`. The .53 original service state
is unchanged. `/home/xander/hc-mtpk` was removed on each blade and absence was
checked twice. No additional fleet invocation was made after cleanup.
