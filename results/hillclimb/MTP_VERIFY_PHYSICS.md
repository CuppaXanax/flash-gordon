# Small-N verification: byte floor and implementation gap

Base: multisession `2d0f8a7` plus MTP foundation `8aaa70f`. Same synthetic
six-layer replay (layers 6..11; four GDN, two QSA), Q4_K/IQ4 dense and
Q4_K gate/up + Q5_1 down experts. No precision/layout change is assumed.

**Capture limitation:** the saved real-text acceptance captures contain final
target HC, token IDs and coverage, **not per-layer target router inputs/IDs**.
Those IDs cannot be recovered exactly from final HC. Consequently no measured
real-target expert-union statistic is available from that capture. Do not
substitute the final synthetic layer's 10–12 experts, native draft routing,
or an IID uniform-router calculation for that missing observation.

## Exact packed-byte ledger

`bench-decode-fusion --q4 --ledger-only` reports the actual cooked allocation
for every matrix without allocating GPU weights. Dense/shared bytes are
**302,110,208** for all six layers. One union expert in every layer adds
**11,059,200 gate/up bytes + 7,372,800 down bytes**. Thus, with per-layer
union size U_l:

```
weight_bytes(N) = 302110208
                + sum_l U_l * (1843200 gate/up + 1228800 down)
```

Here expert traffic is 184.32 MB at N=1, **37.9%** of the 486.43 MB weight
stream. This particular synthetic Q4 replay does not have the approximately
22% expert fraction quoted for a different production weight mix.

Four GDN state matrices occupy **12,582,912 bytes**. An implementation retaining
state across N updates needs only **25,165,824 bytes** at the round edges,
independent of N. Arithmetic remains sequential and scales with N. The baseline
shader logically reads each state twice and writes it once **per token**
(37,748,736*N bytes), with unmeasured cache reuse. Four sequential
tokens are processed within one recurrence dispatch per layer, not four host
dispatches. QSA query work remains per token: at a 256-token prefix the
logical single-read record traffic is 635,304 / 1,273,080 / 1,913,328 /
2,556,048 bytes for N=1/2/3/4 across both QSA layers. This is logical work,
not a DRAM counter; cross-query/head caching can reduce physical traffic.

## Optimistic bandwidth bounds, not projected service time

Dense weights are shared across N. The following includes the minimal GDN
read/write traffic but excludes activation/intermediate transfers, arithmetic,
barriers, submissions, disk/network and scheduling. The **448 GB/s**
normalization is theoretical; **340 GB/s is a sensitivity assumption**, not
a bandwidth measurement from this experiment.

| N | U_l=10, MB | U_l=10, us @448 /340 | U_l=10*N, MB | Disjoint us @448 /340 |
|---:|---:|---:|---:|---:|
| 1 | 511.596 | 1142 /1505 | 511.596 | 1142 /1505 |
| 2 | 511.596 | 1142 /1505 | 695.916 | 1553 /2047 |
| 3 | 511.596 | 1142 /1505 | 880.236 | 1965 /2589 |
| 4 | 511.596 | 1142 /1505 | 1064.556 | 2376 /3131 |

These are overlap/disjoint **bounds**, not acceptance/routing estimates or an
MTP ceiling. Measured per-layer unions are needed to place real text between
them. QSA cache residency and recurrence arithmetic must also be measured.

## Equal-prefix baseline attribution

Shapes now warm **256 input tokens** (N3 rounds down to 255), unlike the previous intrusive
12-batch profiles whose QSA context length varied with N. One timed intrusive
pass follows warmup. These GPU timestamps perturb replay; they diagnose
families and must not replace nonintrusive wall-time A/B/A qualification.
Classification uses shader names rather than the stale inherited scope label.

| Family | N1 us | N2 us | N3 us | N4 us |
|---|---:|---:|---:|---:|
| Dense/shared projections | 1611.20 | 2201.96 | 2996.60 | 3781.80 |
| Expert gate/up | 505.76 | 689.84 | 873.96 | 1277.44 |
| Expert down/reduce | 346.40 | 480.24 | 466.24 | 724.96 |
| GDN convolution/recurrence | 155.92 | 248.92 | 353.36 | 517.64 |
| QSA prepare/cache/attention | 226.08 | 330.40 | 762.44 | 718.04 |
| Routing/HC/elementwise/other | 537.20 | 480.20 | 463.40 | 587.12 |

The clearest gap is **dense**: constant 302.11 MB weights yet +2170.60 us at
N4. Its optimistic weight-only 448-GB/s floor is 674.35 us. Expert gate/up's
weight floor is 246.86 us at full overlap, rising with measured U_l; expert
down's is 164.57 us. Do not claim the full expert increase is overhead until
its union bytes have been measured.

Code audit identifies a concrete dense defect: N2 assigns columns to separate
lanes, while N3/4 forcibly compile as two independent two-column cohorts.
Both duplicate weight reads/dequantization instead of loading once for all
N accumulators. Dispatch count alone hides this defect. The new candidate
uses the existing shared `fg_dense_soa_impl.glsl` arithmetic with true N-column
accumulators and one dispatch per projection.

## Measured synthetic unions, not real-text routing

A separate N1 causal trace covers positions 256..447 for all six layers.
Sliding windows never join streams or gaps. This measures target routing in
the **synthetic replay**, not chat/code routing in the earlier model capture.

| N | L6 | L7 | L8 | L9 | L10 | L11 | Expected MB including retained state |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.000 | 10.000 | 10.000 | 10.000 | 10.000 | 10.000 | 511.596 |
| 2 | 10.304 | 10.864 | 10.843 | 10.895 | 10.869 | 10.932 | 526.055 |
| 3 | 10.589 | 11.689 | 11.647 | 11.747 | 11.711 | 11.837 | 539.923 |
| 4 | 10.878 | 12.439 | 12.386 | 12.519 | 12.487 | 12.672 | 552.702 |

The corresponding 448-GB/s bounds are 1142/1174/1205/1234 us:
**1/1.028/1.055/1.080**. The implementation's N-dependent excess is therefore
not explained by synthetic expert-union bytes. This is not a prediction of
real prompt routing or a serving limit.

| Family weight-only @448 GB/s, us | N1 | N2 | N3 | N4 |
|---|---:|---:|---:|---:|
| Dense/shared | 674.35 | 674.35 | 674.35 | 674.35 |
| GU, measured synthetic union | 246.86 | 266.22 | 284.80 | 301.91 |
| Down, measured synthetic union | 164.57 | 177.48 | 189.86 | 201.27 |

For example, N4 baseline dense/GU/down service exceeds these byte
normalizations by 3107.45/975.53/523.69 us, respectively. Those differences
include required arithmetic/dequantization, not just removable overhead.
The retained GDN state transfer is 56.17 us at 448 GB/s, but its sequential
FMA/reduction work cannot be assigned a zero-time floor. Similarly the QSA
record byte normalization excludes score/reduction/softmax compute.

Future `--mtp-capture-dir` runs now preserve each owned target layer's ten
global IDs after drained N1 blocks, without adding GPU copies or dispatches.
Per-layer files are `routes-SESSION-LAYER.u32` with position coverage.
The lease harness retains these on **every rank**, not only ranks 0 and 7.
`mtp_physics.py --captured-routes DIRECTORY` validates all generated positions
before computing 256 eligible sliding windows. Native draft router IDs and
synthetic replay IDs carry separate provenance.

## Candidate investigation

* True N-column compact projections reuse each dequantized weight across N.
  Eight-row tiles preserve the original reduction arithmetic. All six-layer
  N2/N3/N4 output/state, causal, rollback and two-slot component gates pass
  byte-for-byte with the default geometry.
* Expert down now has one four-token accumulator cohort (WG32), rather than
  two two-token cohorts reloading weights. Union tiles already deduplicate
  expert IDs; their token pair indices retain each token's independent gate.
* Whole-activation GU caching was rejected: 12.8 KB LDS and 80 VGPR made N4
  GU rise to 1898.52 us. The replacement caches only one K256 block in
  decoded FP32, 4.6 KB LDS/84 VGPR at N4, and remains byte-exact.
* The isolated Q4 dense sweep favors WG64/row8 at N2, WG32/row8 at N3/4 for
  large projections. Changing WG changes reduction order: these are
  **tolerance-checked diagnostic choices**, not an exact greedy qualification.
  The largest 2560x10240 projection still costs 2.264x at N4 in the sweep;
  geometry alone does not close the gap. Small 10240x320 favors WG256/row8.
* Actual GFX1013 reports Vulkan integer-dot support but **no accelerated**
  signed-I8, packed-I8 or mixed-I8 dot. Do not infer a hardware integer-dot
  optimization from extension availability alone.

New default-geometry intrusive profiles, same prefix:

| Family | N2 old/new us | N4 old/new us |
|---|---:|---:|
| Dense/shared | 2201.96 /2141.96 | 3781.80 /3522.00 |
| Expert GU | 689.84 /773.84 | 1277.44 /1180.68 |
| Expert down/reduce | 480.24 /322.76 | 724.96 /460.80 |
| GDN convolution/recurrence | 248.92 /267.88 | 517.64 /490.28 |
| QSA prepare/cache/attention | 330.40 /367.96 | 718.04 /632.28 |
| Other | 480.20 /536.08 | 587.12 /535.96 |

These instrumented comparisons are attribution, **not A/B wall-time wins**.
Dense remains the largest gap, including N-column accumulator register
pressure (100 VGPR for compact row8 N4 at WG64/128). N2 GU caching is a
regression and must not be selected merely because N4 improved.

An additional exact candidate retains each GDN lane's 128-value state column
in shader-local storage across all N updates, loading/storing the recurrent buffer only
at round edges. N2/N3/N4 output/state, masking and all four retained-prefix
rollback checks remain byte-identical to N1 reference. Intrusive recurrence
alone on .51 falls to **155.44 us at N2 and 194.92 us at N4** (four layers);
the previous convolution-plus-recurrence totals were 267.88/490.28 us.
The selected candidate therefore disables N2's GU cache and stages GDN state.
It does not change the N1 or normal prefill shaders.

**ISA caveat:** GFX1013 allocates 256 VGPR and spills 130, with 33,792 bytes
of scratch per wave for this local-state cache. It is **not an on-chip
register-retention success**, and its scratch transfers do not realize the
optimistic one-read/write physical floor. The measured service gain includes
unrolled state arithmetic; eliminating those spills is unfinished work.

The first 24-CU .52 bracket, **before state retention and N2 GU-cache removal**,
gave candidate/baseline service ratios 0.912/0.876/1.016 for N2/N3/N4.
N4 was a regression, not a selected improvement. All baseline/candidate
output/state hashes matched over complete runs; the WG32 geometry diagnostic
did not preserve hashes and is not selected as exact. Results from .52 must
not be relabeled as the requested .51 40-CU qualification.

## Selected .51 bracketed results

Separate baseline and candidate executables use their matching SPIRV trees.
Three seeds (0/7/17), A/B/A at each N, equal logical warmup, and 192/96/63/48
samples per run give:

| N | Median candidate/N1 | Range across seeds | Median candidate/bracketed baseline |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000–1.000 | 1.000 |
| 2 | **1.407** | 1.403–1.431 | 0.916 |
| 3 | **1.798** | 1.791–1.879 | 0.842 |
| 4 | **2.414** | 2.344–2.432 | 0.966 |

All twelve shape/seed comparisons preserve complete output and final GDN
state hashes between A1/B/A2. N1 service is unchanged. N2/N3 improve; N4's
small improvement is insufficient. **The 1.2/1.4/1.6 targets are not met.**
These are synthetic component timings, not an MTP limit or serving TPS.
No clocks, governors, services or production owners were changed.

A subsequent interleaved-row cohort experiment reduced per-thread accumulator
count but was rejected: on .52 N4 wall was 8852.97 us, and its reductions
changed output/state (maximum component differences 7.15e-7/7.63e-6).
Scalar dense oracles passed their diagnostic tolerance; the strict component
gate did not. The experiment is not selected and its source changes were
removed. WG geometry is not a substitute for the same-kernel model greedy gate.

Real-target routing capture and native head fanout/causal ring/adaptive
serving orchestration remain unfinished. The previous real-text acceptance
numbers remain valid historical measurements, but neither they nor the
component gates qualify this still-unlinked serving path. No lease is
requested. Future restore/watchdog identity is updated offline to stack-1
`31508ee`, sessions 2/wide 1, with binary and shader-tree hashes plus all
explicit flags from the newly current runbook.

Validation: final `-Werror` release, flag registry, API/fabric/session fixtures,
two real QSA namespaces, native scalar/head self-tests, causal N2–4 and
rollback gates pass. All 25 local Python fixtures pass without skips;
the blade runs 25 with four optional numerical fixtures skipped. Full native
2.77-GB overlay and linked serving greedy qualification were not rerun.
Both .51/.52 scratch directories were removed and verified absent after
SHA-verified download; .50/.53 and production .42–.49 were untouched.

Structured results: `mtp-verify-physics.json`.
Compact timing/ISA/fixture receipts: `mtp-verify-physics-evidence.tar.gz`
(219,358 bytes; SHA256
`ec00c0f31e074caeaa90e9451960ecd68dc5f496a995dbb5b3cea26dc7858361`).
The archive includes failed experiments separately; its metadata records
the original downloaded archive digests.

Reproduce the ledger/profile analysis with `tools/mtp-physics-build.sh`,
`tools/mtp-physics-collect.sh`, and `tools/mtp_physics.py`. The Python analyzer
accepts explicit real route records and checks duplicate/gapped positions
without joining different streams, categories, or layers.
