# MTP whole-block gap search

## Scope and result

Offline work on **192.168.42.51 only**, whose driver reports24 CUs,
starting from `06e71c2e45ad3b87448df3205d1e310c1fe239c0`.
No production lease/access, clock/governor change, target requantization,
restricted draft vocabulary or serving-TPS claim.
Target weights remain≥Q4; all new kernel selections are default-off.

The replay-faithful synthetic six-layer baseline measures
**2.292063 ms N1 /5.314298 ms N4 =2.319×**, using an N1/N4/N1 bracket.
The accepted Q8-down union experiment reduces N4 cost by a median
**3.96%** across three brackets, with N1 essentially unchanged; its N4/N1
is **2.227–2.233×**, not≤1.5×.
These synthetic mixed-expert weights are not the production mixed04 pack.
Authentic-rank measurements and their separate limitations are below.

Machine-readable data: [mtp-block-gaps.json](mtp-block-gaps.json).
Weight-free raw receipts, rejected-source patch and recipes:
[mtp-block-gaps-evidence.tar.gz](mtp-block-gaps-evidence.tar.gz).
Archive size5,189,007 bytes, SHA256
`9fdb758fc9ff551643e8e5c709cd554344f61802b75baacf1689132e023beb0a`.

## Correct operation attribution

The old small-N profile inherited `gdn_prefill_output` across subsequent
HC/experts, badly over-attributing GDN. `owner.c` now sets precise small-N
scopes. `tools/analyze_mtp_block.py` rejects stale scopes, maps native BF16
QSA index projection to the same category as its N4 counterpart, and checks
closure including timestamp gaps. Eight independent regression tests cover
these classifications and closure.

**Intrusive GPU timestamps are attribution, not a decomposition of the
uninstrumented replay wall time.** This separate profile has170/224 dispatches,
10/6 submits and2.694/6.764 ms GPU spans for N1/N4. Timestamping changes the
execution; do not mix its sums with the2.292/5.314 ms replay bracket.

| Operation |N1 µs |N4 µs |N4/N1 |N4−1.5×N1 µs |
|---|---:|---:|---:|---:|
|Routed expert gate/up |437.84 |1515.00 |3.460 |858.24 |
|Dense projections/shared |854.20 |2079.68 |2.435 |798.38 |
|HC/projections/small ops |434.44 |1211.24 |2.788 |559.58 |
|Expert down/reduction |296.16 |901.88 |3.045 |457.64 |
|GDN convolution/recurrence/normalization |101.96 |216.20 |2.120 |63.26 |
|Routing/union |164.28 |304.68 |1.855 |58.26 |
|Inter-dispatch gaps |127.36 |184.48 |1.448 |−6.56 |
|QSA attention/cache/index |277.64 |350.36 |1.262 |−66.10 |

The four largest groups contribute about98% of the **net** excess above1.5×.
GDN contributes2.3%; QSA is already below the1.5× attribution line.
At the profiled264..267 frontier, union cardinalities for layers6..11 are
11/18/18/21/19/20, versus10 each at N1: an unweighted1.783× cardinality
ratio, **not a measured byte ratio**. Growth is real, but is not a complete
explanation for gate/up's3.46× ratio. The existing GU kernel already places
weight decoding outside its token-pair loop; the new Q8-down optimization
fixes a separate path. Lower register count alone is not a speedup.

## Bounded kernel search

Every matrix entry uses264 warmup tokens and216 timed tokens:
N1 has216 samples; N4 has54. Commands use the same binary, target geometry,
seed and synthetic input sequence, no intrusive profiling and no clock control.
N1 output/state hashes remain unchanged. The following variants passed their
dense/block component fixtures but did not improve the whole block:

| Variant |N1 µs |N4 µs |N4/N1 |
|---|---:|---:|---:|
|Same-pass baseline330 |2295.330 |5345.383 |2.329 |
|FP32 dense dot |2289.636 |5343.671 |2.334 |
|Dense split columns |2290.075 |5830.930 |2.546 |
|Dot+split |2301.286 |5878.024 |2.554 |
|Singleton expert-GU bucket |2315.708 |5562.371 |2.402 |
|Singleton+dot |2292.769 |5418.427 |2.363 |
|Singleton+split |2299.398 |5878.539 |2.557 |
|Singleton+dot+split |2295.882 |5953.226 |2.593 |

These exploratory paths/bits are removed from shipping source; the raw
archive retains the patch and receipts. Compact R8 Q4 uses100 VGPR/no scratch.
Splitting columns lowers that to64 VGPR but repeats decode/work and loses
wall time. Four-pair GU uses84 VGPR/4096 LDS/no scratch; singleton GU lowers
registers to56 or64/1024 LDS but still loses. The selected RADV integer-dot
extension is not hardware-accelerated on GFX1013; this is not evidence for
an available accelerated integer-dot rewrite.

### GDN: the audit describes an already-replaced path

Selected330 already uses four groups,32 state values/thread,128 VGPR,
zero spilled registers/scratch,6656 LDS. The old unpartitioned kernel has
256 VGPR,130 spills and33792 scratch bytes.
The eight-way experiment uses16 values/thread,64 VGPR, zero scratch,
10752 LDS, and lowers the four-layer GDN core196.88→175.72 µs.
Three whole-block B/bracket-A ratios are1.005636,1.005391,0.949013;
the last is a drift/outlier, not a reproducible5% win.
The eight-way path is removed. There is no remaining selected-path spill
elimination credit to add to the roadmap's owner-compute budget.

### Final GU probe: exact FP16 operands, FP32 accumulation

Unlike the earlier rejected **FP16 accumulator**, the extra isolated probe
stores only the unscaled integer weight/activation operands in FP16, where
their small integer values are exact, and accumulates dot products in FP32.
All dense/block component fixtures pass. The prototype deliberately replaces
the half shader in an isolated build using its existing experimental bit512;
**shipping bit512 still means the old rejected half-accumulator experiment**.

Three N4 B/bracket-A ratios versus Q8-union baseline are
0.961969,0.992854,0.994181; the first again has a drifting A0.
The robust gain is only about0.6–0.7%, with whole-block N4/N1 still2.215–2.219×.
The shader's LDS halves4096→2048 bytes, VGPR becomes80/84, no spills/scratch.
Actual disassembly contains **zero native `v_dot2` instructions**: explicit
FP16 operands do not deliver the hoped-for native mixed-precision dot on
this compiler/device. This marginal synthetic candidate is retained as
research evidence, not added to shipping flags or credited as a fleet win.
It does not use a low-bit target or relax the component bound.

## Genuine Q8 expert-down union reuse

Existing N4 Q8 down rereads each selected expert separately per token.
New bit131072 (`330+131072=131402`) loads/decodes a weight fragment once
per union expert and applies it to up to four token-pair activations.
It preserves router IDs/per-token gates and uses the existing four-part
scratch/reduction; no additional GPU scratch allocation or CPU scheduling.
N1 and the default330 selection are unchanged.

| Bracket |A0 N4 µs |B N4 µs |A1 N4 µs |B/bracket-A |B N4/N1 |
|---|---:|---:|---:|---:|---:|
|1 |5332.449 |5109.285 |5648.502 |0.930572 |2.227 |
|2 |5350.487 |5126.960 |5326.482 |0.960377 |2.233 |
|3 |5328.797 |5114.124 |5313.859 |0.961062 |2.229 |

Bracket1 has a visible slow A1; the robust median saving is3.96%, **not7%**.
N1 B/bracket-A is0.999566,0.999721,1.000572.
A separate intrusive profile reduces the two Q8-down kernels534.96→381.16 µs,
with12.56 µs additional reducer cost: approximately141 µs net attributed
kernel saving, not an independent wall-time measurement.

The native nine-fixture Q8-down bitwise oracle remains unchanged.
The union path passes nine N2..4 random/extreme/impulse cases, poisoned output
coverage, independent CPU basis checks and malformed stride/parts/format/bank
rejection. FP32 reassociation is **not bit-exact**. With the established
`abs(candidate-reference)/max(1,abs(reference)) <=1e-4` component tolerance:

| Weights |Maximum native normalized error |Maximum CPU basis normalized error |
|---|---:|---:|
|Synthetic cooked Q8 |4.15806179e−5 |3.98752276e−9 |
|Authentic rank7 layer46 cooked Q8 |7.62939453e−5 |3.64341446e−10 |

Whole-block N2/3/4, causal/session and24 coordinator retention fixtures pass.
The authenticated source reconstruction verifies all152 target trained tensors
against the sealed manifest. The native GGUF SHA is
`536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b`,
full Q8 head SHA
`2801d79db74bcae38b47fd5051f03136d8ca7715abeba7d92b49ba46c696bc33`,
target manifest SHA
`0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
No trained weights are redistributed in the evidence.

## Authentic full-head, sessions2 block measurement

Reconstructed rank7 owns layers42–47, including two authentic Q8 expert
downs and12 Q5 HC-down matrices. All processes keep the **full248,320-row
head**, two private target/native sessions and128 MiB worker cache. Three
brackets run `330 →131402 →393546 →330`;393546 additionally tests a
compact-HC-only two-row reuse guard without enabling the regressing tiny-K
Q8 path. All inputs are synthetic sinusoidal HC/zero ngram, **not natural
text, captured authentic HCs, acceptance or live TPS**.

The diagnostic benchmark now aligns all N1..4 at264 warmup tokens plus
216 timed tokens. N1/N2/N3/N4 have216/108/72/54 samples, all covering exactly
positions264..479 with reset causal state. JSON carries the first position
and window lengths; the analysis asserts full coverage on both slots in
every process. The older24+32-iteration probe compared different causal
positions across N and is not used as this denominator.

|Round/slot |Bracket-A N4 ms |Q8-union N1 ms |Q8-union N4 ms |N4/N1 |B/bracket-A |
|---|---:|---:|---:|---:|---:|
|1/0 |6.351219 |2.821258 |5.321124 |1.886 |0.837812 |
|1/1 |6.784816 |2.629462 |5.319862 |2.023 |0.784083 |
|2/0 |5.536678 |2.963373 |5.328720 |1.798 |0.962440 |
|2/1 |5.533592 |2.626083 |5.316399 |2.024 |0.960750 |
|3/0 |5.535315 |2.967976 |5.334341 |1.797 |0.963692 |
|3/1 |5.532440 |2.632037 |5.329282 |2.025 |0.963279 |

The very first baseline drifts to7.17/8.05 ms; all raw samples remain.
Do not claim its16–22% apparent gain. Rounds2–3 give a median **3.71% N4
cost reduction**; all six bracket medians give3.84%. The warmed second-slot
N4/N1 is about **2.024×**, not≤1.5×. First-slot N1 has unrelated extra
cost/variation; dividing by that slower denominator does not establish a
better N4 implementation. The optimization leaves N1 code unchanged.

The HC guard makes N4 **slower in all six comparisons** against Q8-only:
1.113383,1.002997,1.003390,1.004100,1.001785,1.003316.
Median added cost is0.335%; one first-slot process drifts by11.3%.
Its90 dense/component cases pass, but it has no performance justification.
The guard and bit262144 are removed from shipping source; its patch is
retained only as rejected experimental evidence.

All12 processes finish both native k3 graph warmups and final admission at
**14,427,790,128 requested /14,427,790,144 allocated bytes**, leaving
**67,724,496 requested bytes** below the unchanged13.5 GiB policy cap.
This is real full-head **sessions2 offline short-context fit at128 MiB**,
not full-head fit at the production440 MiB cache or long-context
service/coexistence qualification. No restricted vocabulary is needed for
this bounded experiment; changing production cache policy is not authorized
or implied. The Q8-union ISA uses24 VGPR, zero spills/scratch/LDS.

## Qualification boundary

**The bounded offline search is exhausted for this iteration;≤1.5× is not
met.** This is not a physical-impossibility claim. The authentic warm-slot
result still needs about26% additional N4 cost reduction,5.32→3.94 ms at
unchanged N1. Future work must address GU arithmetic/instruction scheduling
and dense/HC fusion rather than rediscover already-eliminated GDN spills.
No existing roadmap research budget becomes measured fleet credit.

Component tests do not qualify full-vocabulary KL/top1, same-prefix
small-margin decisions, pass-local argmax or chat/code noninferiority.
No new acceptance measurement or powered six-pair ring gate is claimed.
The full draft head is preferred; restricted vocabulary remains a sessions2
fit contingency only. The precision sibling's all-hop quality failure and
matched-Q8 conditioning investigation are unchanged; no FP16-MTP saving
is credited. See [MTP_TO_PROD.md](../../docs/MTP_TO_PROD.md).

Final-source gates: **75 Python tests**, C17 `-Werror`, flag registry,
90 independent dense cases,12 scheduling cases, N2/3/4 block and causal
oracles, dual-session isolation,24 coordinator retention cases, native-Q8
bitwise checks, authentic-Q8 union tolerance and8 bounded TCP-ring cases.
All563 listed normalized source inputs match the compiled snapshot; four binaries
and321 SPIRV files were retained/authenticated locally. Toolchain is GCC
16.1.1, Mesa26.2.4, Vulkan loader1.4.341 and glslang16.2.0; exact receipts
and per-file hashes are in the raw archive.

**.51 released2026-10-17T00:06:12Z**: no root-visible owned processes or
test listeners; complete owned root absent twice. The first unprivileged
cleanup hit root-owned shader-cache directories; bounded sudo removal of
that same already-verified owned root completed the cleanup. CPU governor
node inventory is empty both at reservation and release; no clock/governor
settings were changed. Production and .50/.52/.53 were untouched.
