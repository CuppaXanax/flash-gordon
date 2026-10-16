# Small-N bandwidth follow-up

2026-10-16. **Opt-in optimization research, not a serving release.**
The current stopping targets are **N2/N1 <= 1.2 and N4/N1 <= 1.6**.
Neither component tolerance nor synthetic replay performance is a substitute
for real-pack KL mean <= approximately .005, top-1 >= 98.5%, same-build greedy
verification, and serving/session/residency gates. Defaults remain unchanged.

**Outcome:** some measured wins, but both stopping ratios remain missed.
The final mixed long-N4 combination saves **15.53 +/- .36% on .50** and
**8.52 +/- .37% on .53**; short-Q4 N2 improves **5.25 +/- .19% on .50**.
There is no across-the-board winning mask or trained-model promotion.

## Experiment and byte accounting

The fixture is the replay-faithful synthetic six-layer block, layers 6..11:
four GDN and two QSA layers, 512-addressable expert slabs, top-10 routing,
compact Q4 common projections, Q4 expert gate/up and Q5_1 expert down.
It is not trained-model inference or a whole-ring measurement. The separate
`--mixed-experts` fixture replaces layer 8 gate/up with Q5_K and layers 8/10
down with cooked Q8_0. It exercises the formats that interrupted the real
qualification; it is still synthetic, not the authenticated real tensor.

The comparator is the already-published mask **330**, not a fresh mask-zero
implementation. N1 remains mask zero/native, bracketed before and after every
round. Short-prefix runs start after 264 tokens; long runs after 4096.
Every phase measures 216 tokens (`--windows 3 --iterations 72/N`).
Five seeds use adjacent A/B/A brackets, reverse candidate order and N2/N4
order every other round, and use the median per run. Reported intervals are
Student-t over five independent paired round statistics, not over correlated
individual dispatch samples.

Both blades retain their original dynamic governor, clocks and CU configuration;
there is **no fixed-2000-MHz or forced-24-CU experiment** in this follow-up.
.50 exposes live 40 CU; .53 reports 24 CU through Vulkan without an independently
verified live-manager restriction. The family defaults captured by the fixture
are roofline mask 0, dense 7, QSA 2, GDN 8, and other roof families 0.
They are **not** the full real-pack production recipe 2047/19/3/33024/96/3/1.
Mixed-format follow-up timings explicitly use `FG_EXPERT_Q8_WIDE=1`.

Consequently .53 is a **stock second-blade replication, not a verified forced
24-CU qualification**. Its live-manager utility is absent. No CU restriction
was imposed because this follow-up preserves the hardware configuration.
Driver CU reports alone must not be substituted for a routed-CU measurement.

Weights must not be counted N times when a union kernel reuses them. Conversely,
the last layer's expert-union size must not be extrapolated to all six layers.
The routed trace records every selected expert of every layer and query.
For the seed-1 diagnostic:

| N | All-six-layer expert union | Compulsory selected weight MB | Ratio |
|---:|---:|---:|---:|
| 1 | 60 | 486.430208 | 1 |
| 2 | 66 | 504.862208 | 1.037892 |
| 4 | 107 | 630.814208 | 1.296824 |

That seed-1 table is **not** the byte floor of every timed round. At the first
profiled step of the final five-seed .50 study, the standard-Q4 fixture averages
492.574 MB at N2 and 501.176 MB at N4: ratios **1.0126 / 1.0303** versus its
own N1. Mixed short-prefix ratios are **1.0133 / 1.0319** and mixed long-prefix
ratios **1.0159 / 1.0509**. These workloads have substantially more inter-column
expert overlap. All-layer traces, not a universal overlap assumption, determine
the floor. One profiled step is also not the average traffic of 216 timed tokens.

Common weights are 302,110,208 bytes; each selected Q4 gate/up pair contributes
1,843,200 bytes and Q5_1 down contributes 1,228,800. The mixed fixture uses
its own ledger, not those constants. These are compulsory weight payloads,
not hardware DRAM counters; activation, recurrence-state and scratch traffic
are excluded. Percentages of 450 GB/s use an optimistic historical ceiling,
not a newly measured stock-clock bandwidth roof. Instrumented profiles change
serialization and operating point and must not be substituted for wall A/B.
In particular, the native Q8 compatibility path can reread the same expert for
different columns; the unique-byte normalization still counts that expert once.
Its reported percentage is therefore roofline efficiency against compulsory
payload, **not** a measurement of actual bus utilization.

## Flagged implementations

All entries below are bits of `FG_SMALL_KERNELS`, default off.

| Bit | Hypothesis / implementation |
|---:|---|
| 2048 | int32 gate/up integer partials before existing scale/minimum FP32 stages |
| 4096 | Deduplicate causal long-context QSA requests; one K/V decode per KV-head group, shared across N |
| 8192 | Request actual wave32, not merely 32-lane cohorts, for R2/R4/R8 dense kernels |
| 16384 / 262144 | Eight/sixteen expert cohorts in one-dispatch Q5 union-down; WG256/512, gated token totals in registers |
| 32768 / 65536 | Two/four-way K splitting for low-wave-count large R4/R8 GEMVs; separate merge and preallocated 1-MiB scratch |
| 131072 | Six-head QSA union tile; four 16-lane record-score reductions and query cache in LDS |
| 524288 | Prefetch eight compact scale/minimum pairs with two uvec4 loads and row codes with one uvec4 |
| 1048576 | Adaptive 1024/2048-element bitonic top-512 tiles, preserving descending score/ascending-ID ordering |
| 2097152 | Lossless Q4 quant-word transpose, two uvec4 loads per eight output rows |
| 4194304 | Cooked Q8-down union: one WG64/two-row dispatch, weights reused across selected tokens and gates in registers |
| 8388608 | Hybrid top-512 bitonic: subgroup exchange below distance 64, LDS only for distances 64/128, private slots for larger distances |

The existing mask-330 expert GU and Q5 down already use the deduplicated union
schedule, with up to four token/pair members per expert. The 8/16-cohort
experiments principally remove the down partial/merge structure; they do not
create another multiplicative weight-reuse benefit. The new Q8 union does
remove the compatibility path's independent per-token walks.

Long-context QSA descriptors contain absolute block ID, physical cache slot
and per-query/per-record membership. Splits use **absolute block ID modulo
32**, not position in the union: appending future-only blocks cannot repartition
or reassociate earlier queries. Missing pages fall back to the old attention
path without committing the batch twice. With bit 8 set, prefixes through 2051
retain the cheaper existing resident-prefix kernel. QSA union workgroups use
LDS for decoded K/V, while N-column scores/outputs remain separate.

The Q4 vector shadow is constructed before command recording and owned by
the immutable weight tensor. Metadata and canonical weights remain unchanged;
N1 and unsupported layouts still read canonical data. Prepared shadows add
**187,924,480 bytes** (179.219 MiB) in the six-layer Q4 fixture. They are charged
against the model persistent allowance and Vulkan requested/allocated live
memory counters and freed with their tensor. This is explicitly **not**
a memory-neutral persisted pack format or proof that the real dual-session
native workload fits. Mask 0/330 creates no shadows or split-K scratch.
Padded final row tiles are transposed too; the 1024x1025 tail fixture passes
all N2..4 dense/fallback comparisons on both blades.

Small HC/low-M shapes retain their previous geometry unless the older R2 bit 1
is selected; the earlier 331 exploration did not justify adopting it globally.
The new vector transpose targets large compact Q4, not IQ4 HC up, BF16/F32
routers or every production family. Unchanged families remain visible in the
ledger rather than being silently removed from the denominator.

## Component and structural gates

Component comparisons use finite checks and `1e-4 * max(1, abs(reference))`.
Full-block reassociation at a few e-6 is not a rejection criterion.
Causal prefixes, isolated-session behavior, rollback and explicit native-exact
claims retain their exact tests.

* Expert GU has independent CPU oracles at hidden 2560, mid 640, N2..4,
  mixed Q4/Q5 directions, varying gates/overlap, extreme and cancellation inputs.
* Down no longer uses an all-zero fixture. Q5 negative minima and sparse quants,
  and mathematically equivalent signed Q8 weights, give independent nonzero
  CPU sums. Q8 covers all four Q4/Q5 gate/up combinations.
* QSA union compares with independent double-precision softmax over permuted
  cache slots, sparse/disjoint membership, partial tails, empty splits, and
  future-only blocks; each N poisons output and partials before dispatch.
* Top-512 tests counts 513/1024/1025/2048/2051/4096/5000, noncontiguous query
  strides, permuted IDs and many tied finite scores against a CPU sort.
* Vector-layout admission tests deny an insufficient cap before allocation,
  verify live-memory charges and idempotence, forbid allocation during
  recording, preserve canonical bytes and verify lifetime cleanup.
* Dense fixtures alternate prepared weights and an unprepared canonical view
  in the same context, exercising separate vector/legacy pipeline cache entries.
* `fg_vk_small_n_q8_down` remains the unchanged native-order compatibility
  interface. Bit 4194304 changes only the optional expert path; its component
  comparison is tolerance-based, not a replacement for the exact helper gate.

The authenticated real-Q8 slab and full-recipe mask-0/330 qualification belong
to the separate compatibility/format-audit commits. Those receipts do **not**
qualify these new research flags or their model/serving behavior.

The publication rebase adopts serving readiness `7d1fe65` without moving the
serving branch. All **329 research SPIRV binaries** remain byte-identical across
that rebase; this is not a claim that the research leaves every release shader
unchanged. Post-rebase gates additionally use the complete production family
recipe, masks 0/330 and the combined opt-in mask, with reassociation allowed only
where specified. Those are synthetic correctness gates, not a retiming of the
production recipe or trained-pack KL/serving qualification.

## Pilot diagnosis, not wall qualification

All original union8/16, split2/4, integer and QSA union/tile16 pilots passed
their component and causal/session/rollback gates. They did not establish a
wall-time win. Representative native-wave64 N4 resource usage:

| Shader | VGPRs | Spill / scratch | LDS |
|---|---:|---:|---:|
| R8 baseline / split2 / split4 | 100 | 0 / 0 | 0 |
| Compact metadata prefetch | 100 | 0 / 0 | 0 |
| Union-down 8 / 16 cohorts | 84 / 88 | 0 / 0 | 2 / 4 KiB |
| QSA union / record-tile16 | 84 / 64 | 0 / 0 | 8 / 32 KiB |
| Adaptive select 1024 / 2048 | 12 / 16 | 0 / 0 | 8 / 16 KiB |
| Vector quant layout, N4 | 100 | 0 / 0 | 0 |
| Q8 union-down | 24 | 0 / 0 | 0 |
| Hybrid select 1024 / 2048 | 16 / 24 | 0 / 0 | 8 / 16 KiB |

Metadata prefetch reduces VMEM instructions 58 -> 51 but increases VALU
797 -> 947 and code 5812 -> 6440 bytes. Fewer load instructions alone are
not a roofline win. The quant-word transpose separately makes row component
indices constant and tests vectorized payload loads without that same
dynamic metadata-extraction cost.

The single unpaired intrusive .50 N4 QSA diagnostic drops 1844 -> 1013 us
with union/adaptive selection, and .53 drops 1522 -> 1192 us. These are
attribution pilots, **not** paired full-block savings. In the same diagnostics
the metadata-prefetch dense family gets slower: .50 3029 -> 3835 us and
.53 4407 -> 4646 us. Clock/serialization changes in the other families make
single-run wall comparisons particularly unsafe.

The lossless vector-layout N4 specialization lowers VMEM 58 -> 45, VALU
797 -> 733 and code 5812 -> 5268 bytes, but still uses 100 VGPRs: no
occupancy improvement is established by these instruction counts.
Hybrid selection uses no register spills despite its private four/eight-slot
arrays. References studied read-only were llama.cpp `58367713a`,
`mul_mat_vec_q4_k.comp` (packed integer decode, vector activation reads,
multiple rows/columns) and `mul_mat_vecq.comp` (activation preload and
four/two-iteration unrolling). Those techniques motivate experiments; they
are not evidence that an existing GGUF/pack layout reaches the bus roof.
In particular, packed INT8 dot-product support must not be assumed from the
RDNA name: an [independent gfx1013 silicon investigation](https://github.com/D-Ogi/amdgpu-wddm/blob/main/docs/gfx1013-isa.md)
reports incorrect results from those opcodes. That is external evidence, not
a measurement on these blades. No Vulkan capability override, undocumented
instruction injection or hardware experiment was performed here.

## Completed original five-round A/B

Times are six-layer **batch** milliseconds. Positive reduction is faster;
the interval is the 95% half-width in percentage points. These are exploratory,
unadjusted per-candidate intervals, not family-wise winner certification.
All individual rounds and every tested mask are in `mtp-bandwidth.json`.

| Blade | Prefix | N | Candidate mask | A -> B ms | Reduction % +/- CI | B/N1 |
|---|---:|---:|---:|---:|---:|---:|
| .50 | 264 | 2 | 8522: wave32 | 3.516 -> 3.524 | -0.30 +/- 4.80 | 1.322 |
| .50 | 264 | 4 | 33098: split2 | 5.232 -> 5.374 | -2.71 +/- 0.83 | 2.015 |
| .53 | 264 | 2 | 8522: wave32 | 4.068 -> 4.104 | -0.87 +/- 2.08 | 1.393 |
| .53 | 264 | 4 | 8522: wave32 | 6.743 -> 6.636 | +1.57 +/- 1.34 | 2.252 |
| .50 | 4096 | 2 | 1179978: adaptive + tile16 | 4.703 -> 4.703 | +0.01 +/- 0.55 | 1.463 |
| .50 | 4096 | 4 | 1179978: adaptive + tile16 | 7.313 -> 6.712 | +8.21 +/- 0.72 | 2.088 |
| .53 | 4096 | 2 | 1179978: adaptive + tile16 | 5.170 -> 5.314 | -2.79 +/- 0.43 | 1.542 |
| .53 | 4096 | 4 | 1053002: adaptive | 8.424 -> 8.091 | +3.95 +/- 0.30 | 2.348 |

The short-prefix rows show the least-bad investigated candidate per N, not
a recommendation over mask 330. Integer GU, union8/16, extra K-splitting and
metadata prefetch do not deliver a replicated short-prefix wall win.
For N4 long QSA the adaptive selector does: the plain union/adaptive mask
1053002 saves **8.03 +/- .45% on .50 and 3.95 +/- .30% on .53**.
Its N2 change is **-.78 +/- .74% and -4.84 +/- .31%**, respectively.
Union-only and tile16-only N4 gains are smaller; QSA gains must not be
generalized to N2 or to reaching the stopping ratios.

## Final layout / Q8 / hybrid-selection A/B

These are fresh same-build A/B/A comparisons; do not subtract their absolute
times from the earlier build. Ratios include their own native N1 brackets.
Rows show representative per-N choices, **not one universally selected mask**.

| Blade | Fixture / prefix | N | Mask | A -> B ms | Reduction % +/- CI | B/N1 +/- CI |
|---|---|---:|---:|---:|---:|---:|
| .50 | Q4 / 264 | 2 | 2105674 | 3.499 -> 3.315 | +5.25 +/- .19 | 1.242 +/- .008 |
| .50 | Q4 / 264 | 4 | 2621770 | 5.384 -> 5.218 | +3.09 +/- .48 | 1.955 +/- .011 |
| .53 | Q4 / 264 | 2 | 2105674 | 4.041 -> 3.862 | +4.44 +/- 3.99 | 1.310 +/- .054 |
| .53 | Q4 / 264 | 4 | 2097482 | 6.500 -> 6.447 | +.79 +/- 1.43 | 2.188 +/- .091 |
| .50 | Mixed / 264 | 2 | 4194634 | 3.687 -> 3.611 | +2.06 +/- .78 | 1.324 +/- .011 |
| .50 | Mixed / 264 | 4 | 4194634 | 5.691 -> 5.475 | +3.79 +/- 1.47 | 2.007 +/- .013 |
| .53 | Mixed / 264 | 2 | 4194634 | 4.207 -> 4.155 | +1.24 +/- 1.33 | 1.389 +/- .006 |
| .53 | Mixed / 264 | 4 | 4194634 | 6.847 -> 6.813 | +.49 +/- 4.49 | 2.277 +/- .088 |
| .50 | Mixed / 4096 | 2 | 14684490 | 4.823 -> 4.773 | +1.04 +/- .33 | 1.470 +/- .011 |
| .50 | Mixed / 4096 | 4 | 14684490 | 7.668 -> 6.477 | +15.53 +/- .36 | 1.994 +/- .009 |
| .53 | Mixed / 4096 | 2 | 14684490 | 5.312 -> 5.541 | -4.30 +/- .48 | 1.590 +/- .010 |
| .53 | Mixed / 4096 | 4 | 14684490 | 8.807 -> 8.057 | +8.52 +/- .37 | 2.311 +/- .009 |

Mask 2105674 is vector layout + actual wave32 + 330; 2621770 is vector
layout + metadata prefetch + 330; 4194634 is Q8 union + 330.
14684490 combines vector layout, Q8 union, shared-KV QSA and hybrid selection.
The mixed Q8-only .53 wall result is **inconclusive**, despite its GPU diagnostic
improvement. Vector layout alone or combined with Q8 is not a replicated
mixed-fixture wall win. Adding split2 to the vector path loses.
The long hybrid combination improves N4 on both blades, but regresses .53 N2:
it is not an across-the-board replacement.

### Exact-shape repeated GEMV diagnostics

Five A/B/A rounds use repeated static command buffers, not full-block wall
measurements. These are **cacheable compact-Q4 / IQ4 fixtures**, not DRAM counters.
Times are us; BW columns are effective unique cooked-weight throughput as a
percentage of the historical 450-GB/s roof. Different shapes need not have the
same roofline intensity. Source bytes and all four N values/intervals are in JSON.
The profile-selection policy maximizes mean N2/N4 percentage reduction:
.50 chooses mask **2105674**, .53 **2621770**. This differs from selecting
the best N4-only mask, particularly in the long-QSA profile below.

| Blade | K x M / type | N2 us A -> B | N2 % roof A -> B | N4 us A -> B | N4 % roof A -> B |
|---|---|---:|---:|---:|---:|
| .50 | 2560 x 6144 / Q4 | 46.62 -> 33.33 | 43.3 -> 60.6 | 55.39 -> 48.03 | 36.5 -> 42.1 |
| .50 | 2560 x 10240 / Q4 | 78.17 -> 57.72 | 43.1 -> 58.4 | 95.78 -> 85.82 | 35.2 -> 39.3 |
| .50 | 2560 x 12288 / Q4 | 96.76 -> 69.14 | 41.8 -> 58.5 | 115.22 -> 106.36 | 35.1 -> 38.0 |
| .50 | 6144 x 2560 / Q4 | 41.96 -> 34.38 | 48.2 -> 58.8 | 55.84 -> 49.54 | 36.2 -> 40.8 |
| .50 | 10240 x 320 / Q4 HC down | 19.18 -> 19.20 | 21.9 -> 21.9 | 31.18 -> 31.22 | 13.5 -> 13.5 |
| .50 | 320 x 10240 / IQ4 HC up | 17.51 -> 17.50 | 23.4 -> 23.4 | 33.64 -> 33.66 | 12.2 -> 12.2 |
| .53 | 2560 x 6144 / Q4 | 52.85 -> 51.12 | 38.2 -> 39.5 | 70.76 -> 62.53 | 28.6 -> 32.3 |
| .53 | 2560 x 10240 / Q4 | 90.66 -> 90.40 | 37.2 -> 37.3 | 120.99 -> 102.64 | 27.8 -> 32.8 |
| .53 | 2560 x 12288 / Q4 | 107.35 -> 104.39 | 37.6 -> 38.7 | 147.83 -> 125.51 | 27.3 -> 32.2 |
| .53 | 6144 x 2560 / Q4 | 57.16 -> 53.95 | 35.4 -> 37.5 | 74.02 -> 66.12 | 27.3 -> 30.6 |
| .53 | 10240 x 320 / Q4 HC down | 20.69 -> 20.70 | 20.3 -> 20.3 | 41.47 -> 41.44 | 10.2 -> 10.2 |
| .53 | 320 x 10240 / IQ4 HC up | 27.15 -> 27.16 | 15.1 -> 15.1 | 50.63 -> 50.60 | 8.1 -> 8.1 |

The forced-compact 2560x640 shared shape is also measured, essentially unchanged;
that is not a measurement of its normally preferred SOA layout. HC shapes keep
their previous path. The vector experiment does **not** achieve 85% on the
large measured multi-column shapes. N1 retains canonical weights and is not
optimized by this new flag. Unique weight bytes stay constant while N-column
FMA work grows, so lower effective weight GB/s need not mean a bandwidth-bound
kernel; the instruction/occupancy hypothesis is consistent with, not proven by,
these diagnostics.

### Expert and QSA attribution

Five-repeat **intrusive** means below are total GPU us across six layers, not
wall deltas or independently qualified per-family wins. Scope labels are pooled;
classification uses kernel names. Unchanged GU timings visibly move with the
instrumented operating point. Do not sum these differences to explain wall A/B.

| Blade / phase / selected mask | Dense us A -> B | Expert GU us A -> B | Expert down us A -> B | QSA us A -> B |
|---|---:|---:|---:|---:|
| .50 Q4 / 2105674 | 3310.6 -> 3485.0 | 1360.6 -> 1506.5 | 485.8 -> 513.7 | 153.7 -> 165.5 |
| .53 Q4 / 2621770 | 3917.5 -> 3673.7 | 1869.6 -> 1857.6 | 544.2 -> 539.3 | 159.1 -> 157.6 |
| .50 mixed / 4194634 | 3198.9 -> 3347.7 | 1300.4 -> 1395.5 | 836.8 -> 625.6 | 148.3 -> 157.1 |
| .53 mixed / 4194634 | 3776.5 -> 3769.8 | 1786.5 -> 1790.0 | 977.6 -> 782.8 | 152.5 -> 151.9 |
| .50 mixed long / 14684490 | 2690.1 -> 2644.6 | 968.8 -> 1067.9 | 775.8 -> 543.2 | 1712.0 -> 1095.0 |
| .53 mixed long / 6291786 | 3253.3 -> 3043.8 | 1441.8 -> 1442.2 | 867.3 -> 680.8 | 1552.7 -> 1535.0 |

The last .53 diagnostic selects the better *mean N2/N4* mask, without the QSA
hybrid; it is **not** an attribution of the N4-only 14684490 wall improvement.
The JSON includes compulsory-byte efficiencies for every family and round.

For the exact **640x2560 cooked-Q8 expert-down** shape in the mixed short
fixture (two layers, N4), unique bytes average 37.94944 MB versus 139.264 MB
of per-query payload without cross-query reuse:

| Blade | Q8-only aggregate GPU us A -> union | Unique GB/s A -> union | % historical roof A -> union |
|---|---:|---:|---:|
| .50 | 521.224 -> 297.112 | 72.889 -> 129.024 | 16.2 -> 28.7 |
| .53 | 628.776 -> 434.992 | 60.404 -> 87.298 | 13.4 -> 19.4 |

This is a useful reduction of duplicated work, not an 85%-bus-saturation result.
The already-unioned GU shape is 2560x640; integer/packing/prefetch and 8/16-way
down regrouping did not establish a replicated new GU/union wall win.

## Stopping decision and eight-stage lower bound

**Targets missed; stop this bounded search.** The best short-Q4 N2 point
estimates are 1.242 (.50) and 1.310 (.53), not <=1.2. N4 remains approximately
1.95 / 2.17 even allowing per-N mask choices, not <=1.6. Mixed formats and long
QSA do not meet both ratios either. The tested set covers integer/packed GU,
prefetch, 8/16-cohort down, real wave32/64, split2/4, metadata/payload layouts,
and adaptive/hybrid shared-KV QSA. This is not an exhaustive cross-product
search, a proof of a hardware ceiling, or a reason to force unsupported ISA.

For an **eight-equal-stage verification-only lower bound**, multiply batch ms
by 8/N: the Q4 N2 rows above project **13.26 / 15.45 ms per verified column**
on .50/.53; the shown Q4 N4 rows project **10.44 / 12.89 ms**. Mixed long N4
projects **12.95 / 16.11 ms**. These exclude native proposal/head work,
acceptance/rejection, network, n-gram/embedding/sampling and serving overhead.
They are **not accepted-token latency or measured eight-rank TPS**.

All new bits in the implementation table remain default-off and on the
**real-pack KL/top-1 gate list**, including reassociated Q8 union and split-K.
Any eventual selection must additionally pass the full production recipe,
same-kernel greedy/causal/rollback and actual persistent **and total** residency
admission before readiness. The optional 179.219-MiB shadow has not passed a
trained-pack residency gate; the separate mask-0/330 fit receipt does not
qualify it. No serving ref, release selection or default mask is promoted.

Reproduction: `tools/mtp-kernels-round{3,4,5,6,7}.sh`,
`tools/mtp-kernels-paired-bandwidth.sh`, `tools/mtp-kernels-paired-final.sh`
and `tools/analyze-mtp-bandwidth.py`. Post-rebase full-recipe checks use
`tools/mtp-kernels-publication-gates.sh`. The four analyzer unit tests cover
adjacent brackets, the N1-before glob, all-layer unique bytes, pooled scope
names, cacheable shape payloads and route records without a `kind` field:
`python -m unittest discover -s tests -p test_mtp_bandwidth.py`.

Compressed raw receipts are committed in `mtp-bandwidth-receipts/`; their
SHA256 identities are in `mtp-bandwidth.json`. The two `source-round*.tar.gz`
files preserve all required measured-build sources/headers/shaders/Makefile,
without unrelated binary assets. Each contains per-file hashes and the
original complete-upload tar SHA in `source-snapshot.json`. Snapshot build
prerequisites were checked with `make -n`. Original short/long pairing used
the round6 source; final layout/mixed/long pairing used round7b:

| Measured build | bench SHA256 | test SHA256 |
|---|---|---|
| Original paired | `9547e4fa97a60c0917ac3e431dd9d4d8dfa73e59fbc36e44a2df2ba9dc5d2d4b` | `853e1ecc4938e91ca211be4e79bb36d2d2598e48c374b99e4dfce87cdb2f96f5` |
| Final paired | `f83330f4a179cab0ae0fc0c74b30a63215d47d620a3c9d2f797c2463943624a1` | `a1a10722a4e11cc727c1dac332b7c50949ca732fdbe69a9075df53b0cdaca2fe` |
| Post-readiness gates | `7e7aeb463536814ec1ca550367906fae8696e6dbae93a670d0508373ed0a28a7` | `a7d5399331f4126c3c386744d31205b8667746384240d32a1917181fc1f3b693` |

Each row is identical on both blades. The machine-readable result preserves
all five paired statistics, shape/family diagnostics, all-layer union byte
counts, component receipts and explicitly unpaired pilots. To reconstruct its
input, copy all committed receipt archives to `.build\bandwidth-evidence`,
extract `*-stock-50.tar.gz` into its `50` subdirectory and `*-stock-53.tar.gz`
into `53`, then run:

```
python tools\analyze-mtp-bandwidth.py --input .build\bandwidth-evidence --output results\hillclimb\mtp-bandwidth.json
```

`mtp-bandwidth-cleanup.json` records final independent root-absence checks and
service/governor health: both original governor hashes remain
`a5e83975e4fbea57d6110cbf4aa81fd58367d044c2bc602815ff0ca21ede6757`;
.50 embed/rerank are restored and healthy, .53 retains their original inactive
state. Both `hc-mtpk` roots were removed and checked absent twice. No production
blade was accessed. After .53 timing completed, the separate authorized export
of 48 changed rank7 common tensors plus HC down/up was SHA-verified locally;
its remote staging was removed and retained `hc-q4` was not modified.
