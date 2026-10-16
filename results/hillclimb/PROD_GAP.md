# Production promotion and decode-gap trace

2026-10-15, `cuppaxanax/ab-harness`, serving source `main a63a166`.
Fleet lease: **.42–.49 only**. No .50–.53 access, hardware/CU/clock changes,
native-binary changes, or arithmetic switches.

## Decision and final state

**Main + sealed mix-04 is promoted and left serving**, sessions 1, wide 0,
at `192.168.42.42:8080`, model `Qwen3.8-Flash-Next-Uncensored`.
The user explicitly revised the deployment decision after the original study:
take mix-04's observed gains and passed KL with zero significant regressions.
The six inconclusive protected metrics **remain inconclusive**, not newly
proved noninferior. `AB_QUALIFY.md`, its evidence, and the statistical gate
are unchanged. In particular, the three-pair/df=2/76-comparison power issue
has not been redesigned in this task.

The reused sealed-pack gate is 480 full-vocabulary rows: mean KL
**0.003128577865342439**, p99 **0.031049241889376272**, top-1 agreement
**0.9895833333333334**; thresholds .01/.05/.97 with at least 400 rows.
No new quality pass is claimed from these performance probes.

Pinned launchers were installed and the complete ring actually restarted.
The one-command original rollback was also **actually executed**, verified,
and followed by re-promotion. A final complete plain-mix restart passed at
23:50 UTC-07: exact live binary/manifest/argv on all eight, original identities
unchanged, `/health=ok`, correct model, and greedy chat **`OK`**.
All diagnostic `FG_` switches and `LD_PRELOAD` were absent afterward.
All eight `/home/xander/hc-abq-gap` directories were removed only after
unmapping the instrumentation and downloading its evidence.

| Identity | Pinned value |
|---|---|
| Release | `/home/xander/flash-gordon-live/20261015-abq-main` |
| Binary SHA256 | `1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789` |
| Pack | `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm` |
| Manifest SHA256 | `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252` |
| Persistent operator | `/home/xander/flash-gordon-live/20261015-abq-operator` |

Exact commands and tested rollback: this branch's `docs\DEPLOY_CURRENT.md`,
also frozen in `prod-gap\current\DEPLOY_CURRENT.md` for mirrored-result readers.
Persistence matches the original `nohup` launch: survives SSH/client exit;
**not an automatic host-reboot service**. Original release/pack are retained.
Fleet work is finished; release is recorded in `prod-gap\lease-release.json`.

## Quick sanity versus recorded original

Three fresh requests per operating point; every decode emitted **128 actual
tokens**. Numbers are engine TPS, not a new paired study.

| Metric | Current three trials | Current median | Historical original median | Median ratio |
|---|---|---:|---:|---:|
| Short/nominal 4K decode | 28.276506, 28.496619, 28.591156 | 28.496619 | 26.550685 | 1.0733 |
| Nominal 32K decode | 25.898103, 25.773859, 25.779520 | 25.779520 | 24.303021 | 1.0608 |
| Cold nominal 4K prefill | 334.588768, 350.898801, 361.397678 | 350.898801 | 310.686616 | 1.1294 |

These are **descriptive, unpaired historical comparisons**, with different
prompt bodies/run times; they do not turn the earlier inconclusives into
passes. Client wall TPS, exact usage, bodies' hashes, output hashes, and all
headers are retained in `prod-gap\sanity.json` and `PROD_GAP.json`.

## Measurement controls and instrumentation cost

Fixed case bodies/salts across phases: three nominal 4K continuations and one
32K continuation. Each is cold one-token priming followed by a cached-prefix
128-token continuation. The latter still prefills **31 new tokens**.
All 17 decode cases emitted 128 tokens; all 13 non-reference response-hash
comparisons match normal, as do exact serialized request hashes and priming
outputs. Nothing substitutes a continuation's timing for cold prefill.

Native trace flags were `FG_DECODE_RING_TRACE=1`, `FG_FABRIC_PROFILE=1`,
`FG_GENERATE_TRACE=1`, and `FG_OUTPUT_SPLIT_TRACE=1`. **Neither
`FG_DECODE_PROFILE` nor `FG_FRAME_TRACE` was enabled**: their GPU instrumentation
changes flushing/static replay. Setting `FG_GENERATE_TRACE=0` is not off in
this binary; final verification requires it absent.

| Phase | Short engine TPS, median of three | 32K engine TPS | Interpretation |
|---|---:|---:|---|
| normal | 27.161641 | 24.288064 | Plain production command stream |
| trace | 26.912223 | 24.208326 | Native wall/fabric/generation logs |
| wall | 25.567665 | 23.566659 | Native logs plus CPU-only Vulkan-call interposer v1 |
| engine | 19.780342 | 20.704121 | Additional read-only DRM accounting inside existing calls |
| post | 27.361158, one short case | — | Plain configuration restored |

These are sequential descriptive controls, not confidence-qualified overhead
estimates. The short medians change approximately **−0.9%, −5.9%, −27.2%**
versus normal. Do not subtract these percentages from individual buckets.
The engine-counter probe is **not nonintrusive in wall time**, despite adding
no GPU commands. Its timing is not substituted for normal production.

The interposers forward the same five Vulkan calls, arguments and results;
they add no submit, fence, timestamp query, shader, or readback command.
Their fake-Vulkan fixture checks all five calls, including timeout and
device-error forwarding. Versioned source/binaries and the predeclared
engine-phase plan are retained under `prod-gap\instrumentation`.
All eight rank token sequences align exactly in each traced phase.
Coordinator counter deltas stay **10 submissions/91 newly recorded
dispatches per token at short context, 10/93 at 32K**, in all three traced
phases: no transition into the intrusive profiling/static-replay-off path.

For sustained descriptive tables below, discard the first eight decoded
tokens of each continuation, retaining 120 each: 360 short-context token
observations and 120 long. Raw logs retain everything. Token observations
are correlated and **not independent statistical trials**.

## Native production ledger

Short-context means from **trace**, milliseconds/token:

```
36.975778 cycle
 = 1.537428 inclusive preparation
 +30.144242 rank-service sum
 + 5.294108 residual.

Preparation = 1.334961 local n-gram lookup
            +0.015600 embedding-function call/recording
            +0.186867 post-lookup readback/encoding/initial-dispatch work.
```

`RING_DECODE embed_ms` is misleadingly named: it is the **entire preparation**
through initial dispatch, not embedding alone. The .0156-ms difference is the
embedding function's call interval, **not a GPU embedding timestamp**; its
queued completion/readback can be charged later in preparation.

| Rank / blade | Layers | Block/run wall | Readback wall | Egress wall | Inclusive service |
|---|---|---:|---:|---:|---:|
| 0 / .42 | 6–11, 4 GDN + 2 QSA | 4.1935 | .0331 | Not in `own_ms` | 4.2272 |
| 1 / .43 | 0–5, 5 GDN + 1 QSA, PLE | 3.5932 | .0330 | .1189 | 3.7480 |
| 2 / .44 | 12–17, 5 GDN + 1 QSA | 3.3264 | .0337 | .1172 | 3.4800 |
| 3 / .45 | 18–23, 4 GDN + 2 QSA | 3.8100 | .0342 | .1230 | 3.9698 |
| 4 / .46 | 24–29, 5 GDN + 1 QSA | 3.1692 | .0334 | .1181 | 3.3238 |
| 5 / .47 | 30–35, 4 GDN + 2 QSA | 3.6695 | .0335 | .1207 | 3.8265 |
| 6 / .48 | 36–41, 5 GDN + 1 QSA | 3.1640 | .0335 | .1203 | 3.3204 |
| 7 / .49 | 42–47, 4 GDN + 2 QSA | 3.6558 | .0334 | .5566 | 4.2485 |

Worker input write is ~.002 ms and publish ~.001 ms; rounded columns need
not sum exactly. Worker service includes egress; coordinator `own_ms` does
not include its forward send. The residual therefore is not purely network.

**Exceptional work:** rank 1's block is .373 ms above the mean of the other
5-GDN/1-QSA blocks. This is an **observational difference, not isolated PLE
GPU cost**: it also includes first-block n-gram upload, different weights,
blade/clock/routing differences. There is no separate nonintrusive PLE
timestamp. Rank 7's .557-ms egress includes final HC and four hidden-slice
sends, versus ~.12 ms on ordinary workers. In the CPU-only hook, its last
HC batch has .061 ms fence blocking, .043 ms submit and .015 ms recording;
those are call-wall intervals, not shader timings.

At 32K the native ledger is **41.113292 = 2.165567 preparation +
33.631783 service + 5.315942 residual**. Keep this operating point separate.

## CPU submit, recording, fence and host attribution

CPU-only **wall phase**, short-context means, milliseconds/token.
Columns are additive because recording spans include begin/end API time and
the analyzer unions nested intervals before assigning the remaining wall.

| Rank | Service | Fence-call blocking | Submit calls | Recording spans | Reset calls | Other service wall |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.048 | 3.009 | .464 | .212 | .011 | .353 |
| 1 | 3.934 | 3.185 | .294 | .108 | .005 | .341 |
| 2 | 4.032 | 2.871 | .293 | .110 | .005 | .752 |
| 3 | 4.041 | 2.940 | .447 | .200 | .010 | .443 |
| 4 | 3.691 | 2.922 | .311 | .110 | .006 | .342 |
| 5 | 3.828 | 2.733 | .444 | .200 | .009 | .443 |
| 6 | 3.527 | 2.775 | .304 | .101 | .006 | .341 |
| 7 | 4.200 | 2.831 | .498 | .222 | .012 | .638 |
| **Sum** | **31.300075** | **23.266155** | **3.054130** | **1.262991** | **.063534** | **3.653264** |

The four 5-GDN/1-QSA blocks each issue six submits/seven waits; the
4-GDN/2-QSA blocks issue nine/ten. Rank 7 adds one HC submit/wait.
That is **61 block/HC submissions**, versus a single six-layer replay
submission in the synthetic benchmark. This is not 61 removable empty
launches: QSA staging/readback dependencies and useful GPU work matter.
Other service wall includes control, copies, encoding/CRC, egress,
instrumentation/logging and scheduling. It is not CPU utilization.

Fence blocking includes GPU execution, driver and wake/scheduling delay;
**do not add a GPU counter to it**. GPU work can overlap submit/record/host
intervals. The wall phase's service sum is 1.156 ms above native trace;
individual ranks also move in both directions, so its partition is not a
precise counterfactual decomposition of the 30.144-ms native sum.

Four output shards run concurrently on ranks 4/0/1/2. Their CPU-hook Vulkan
windows average **.784/.781/.779/.780 ms**, and the mean per-token maximum is
**.790476 ms**, close to LATENCY's historical ~.785 ms. These windows start
at the first existing Vulkan call, not at synchronized cross-host arrival.
They exclude transport/other host work; fanout/head execution can overlap
rank 7's already-charged egress. Do not sum the four windows as latency.

### What is and is not measured on the GPU

No existing hardware-timestamp path preserves static replay in this binary.
Consequently **normal-production per-kernel GPU, PLE and GPU-versus-driver
fence splits are not exactly identifiable here**.

Read-only DRM engine/job accounting gives these additional observations:

| Rank | Normal whole-request engine ms / output token | Engine-hook block/HC engine ms |
|---|---:|---:|
| 0 | 4.819 | 4.458 |
| 1 | 4.641 | 4.461 |
| 2 | 4.231 | 3.998 |
| 3 | 3.880 | 4.521 |
| 4 | 4.173 | 3.974 |
| 5 | 3.803 | 4.440 |
| 6 | 3.503 | 3.984 |
| 7 | 3.923 | 4.252 |

Normal snapshots are outside the request and include its **31-token prefill
and head work**, not pure decode-block GPU time. One unique DRM client was
observed per rank. The engine hook reads the same client's counter before
existing submits and after existing waits, adding no GPU commands; it
records engine/job lifetimes rather than hardware per-shader timestamps.
Its repeated fdinfo reads materially perturb host scheduling/execution:
**43.443-ms service / 50.549-ms cycle** at short context. These engine values
must not be transplanted into the normal ledger or used to claim an exact
normal GPU time. Missing counters are unknown, not zero. No “GPU-only
30-versus-20 ms discrepancy” is established by these data.

## Receive/send and residual

Route: coordinator → 1 → 0 → 2 → 3 → 4 → 5 → 6 → 7:
**eight work transfers**, one 51,256-byte initial framed message and seven
41,016-byte ordinary frames.

| Receiving rank | Receive-call wall, ms | Payload read, ms | Validation, ms |
|---|---:|---:|---:|
| 0 | .2922 | Not populated | Not populated |
| 1 | .2282 | .2035 | .0157 |
| 2 | .1670 | .1456 | .0127 |
| 3 | .2651 | .2358 | .0155 |
| 4 | .0910 | .0704 | .0116 |
| 5 | .0977 | .0707 | .0124 |
| 6 | .1628 | .1382 | .0138 |
| 7 | .1163 | .0896 | .0124 |

Receive-call walls sum to **1.420298 ms**, not total wire latency.
Timers start after socket-readiness selection; arrival/serialization can
already have elapsed in poll or buffered data. Coordinator subfields are
unpopulated, and reported zero `wait_ms` does **not** prove no wait.
Conversely, polling intervals that include remote computation cannot be
summed as transport. Cross-host clocks are not synchronized.

Eight work-send call profiles sum to **.263844 ms**. Their timer starts
*after* frame encoding/CRC and measures enqueue/completion, not socket drain.
Most of that and the remaining worker encoding are already charged to
preparation/service. Do not add them to the residual a second time.

LATENCY's separate real-fabric RTT/2 experiment estimates .44–.47 ms per
ordinary hop, ~**3.38–3.60 ms net work transport** after already-charged sends.
That remains a **transferred estimate, not a measured production one-way
decomposition**. Physical payload serialization alone is ~2.70 ms at 1 Gb/s.
Together with the observed ~.79-ms parallel-head window, it leaves roughly
.9–1.1 ms of the 5.294-ms native residual for head fanout/delivery, results,
combine, polling and unassigned overlap/variation. Exact current sub-buckets
are not isolated. One wall-hook case has a 7.078-ms residual; it is retained,
not discarded as inconvenient noise.

## Real production n-gram interval

This is the coordinator's actual `fg_ngram_store_lookup_prefill` call,
not seven resident-lookup network round trips and not .53's storage proxy.
Units: milliseconds; 120 retained token observations per row.

| Fixed case | Mean | Median | p95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| Short 0 | 1.3775 | 1.4055 | 1.990 | .458 | 2.457 |
| Short 1 | 1.3114 | 1.3215 | 1.955 | .457 | 2.177 |
| Short 2 | 1.3159 | 1.3415 | 1.842 | .458 | 2.233 |
| 32K | 1.9559 | 1.3050 | 2.105 | .462 | **59.325** |

The three short cases average **1.334961 ms/token**. The long mean contains
the real 59.325-ms outlier; its median is not slower than short.
Different greedy histories/cache states mean this is **not evidence of
context-linear n-gram cost**. The interval includes local lookup/I/O,
assembly and dequantization; the traces do not split those components.
Main uses local O_DIRECT, bounded application caching and the existing
uncached-UMA host dequantizer. LATENCY's opt-in buffered/cacheable staging
change is **not merged or enabled**; its synthetic gains are not credited.

## Why the replay reference is not a 20-ms main prediction

Frozen references are in `prod-gap\references`, from roofline
`e073c5ddd84c205147454194f53388bef9ffc3ea` and latency
`8c4f9beb77d72d2c194419fac84aec8e9dfe028f`.

ROOFLINE's restricted-24-CU, fixed-2000-MHz six-layer wall is
**2.939196 ms baseline mask 0**, versus **2.503129 ms candidate mask 2047**.
The latter is an **unmerged optimized candidate with no real-pack KL gate**,
not main a63a166. Layers 6–11 contain four GDN/two QSA; blindly multiplying
by eight invents 16 QSA instead of the model's 12.

| Correct type-weighting, five-round mean | Baseline | Optimized candidate |
|---|---:|---:|
| GDN layer interval, us | 433.3974 | 366.3866 |
| QSA layer interval, us | 545.3455 | 461.6013 |
| 36 GDN + 12 QSA layer intervals, ms | 22.146451 | 18.729135 |
| Add eight measured wall-minus-layer remainders, ms | **23.065779** | **19.644174** |

The remainder is not pure CPU time; this is transparent reference arithmetic,
not a format/clock/context-matched production prediction.
Thus **3.421604 ms** of the apparent ~10.5-ms service gap compares against
optimizations not deployed here. The remaining native-versus-baseline
reference difference is **7.078463 ms**, still not wholly removable excess.

The CPU-only phase helps locate it: its **23.266-ms fence-call sum** is near
the baseline's 22.146-ms layer intervals, while **8.034 ms** is outside fence
calls, versus .919 ms of eight replay wall-minus-layer remainders. This
accounts for the instrumented wall ledger, not a proof of pure GPU time or
7 ms of safely deletable CPU work.

Other unmatched conditions matter: replay is synthetic selected Q4-ish
weights/short context; mix-04 restores **all QSA projections to Q8**, keeps
HC up Q8 and HC down/GDN output Q5, with different expert slabs/working sets.
Production clock policy is `auto`; idle endpoint samples show 1000–1400 MHz,
**not the active GPU execution frequency**. No clock was pinned/changed.
Vulkan's reported 24 CUs does not prove production live CU routing matches
the explicitly restricted replay. No matched-clock/format or per-kernel
causal saving is claimed.

## Reproduction, validation and retained evidence

* `tools\prod-gap-probe.py`: actual-token/request/output/header evidence.
* `tools\prod-drm-snapshot.py`: read-only, PID/executable/manifest-validated
  snapshots; deduplicated DRM clients and CPU ticks.
* `tools\prod-vk-wall.c`: engine-probe source; CPU-only v1 source is separately
  frozen under `prod-gap\instrumentation`, not silently replaced by v2.
* `tools\analyze-prod-gap.py`: regenerates `PROD_GAP.json`, validates request/
  greedy/length/rank alignment, separates absent measurements from zero,
  computes interval unions and correctly type-weights the frozen replay.
* `prod-gap\logs`: **40 compressed owner logs**, five phases × eight ranks.
* `prod-gap\ops`: exact local orchestration and transcripts, including tested
  rollback/re-promotion/final restart. The frozen v1 script is also retained.
  These are lab-specific operational records; do not rerun without a lease.
* `prod-gap\current`, `final-cleanup.json`, `final-api-health.json`: final
  pinned launchers, restart/identity/immutable-original checks, trace-off,
  unmapping/cleanup and visible generation.

Recompute locally from this branch, without fleet access:

```powershell
python .\tools\analyze-prod-gap.py
$env:PYTHONPATH = Join-Path $PWD 'tools'
python -m unittest test_hillclimb_ab test_ab_fleet test_prod_gap -v
```

The C probe builds with C17, `-O2 -Wall -Wextra -Werror -fPIC -shared`,
`-ldl -pthread`; GCC intermediate files were directed into project `.build`.
The forwarding fixture passes once its project-local fake library is on
`LD_LIBRARY_PATH`; the initial loader-path failure is retained, not hidden.
PowerShell/Bash syntax and the complete local regression result are archived.
All **17 targeted local tests passed**, including unchanged harness/fleet
fixtures and the new actual-token, interval, head/block and replay-weight tests.
`prod-gap\evidence-sha256.json` covers exact retained evidence bytes under
the directory's own `-text` attributes, without changing the historical
A/B evidence attributes or checksums.
