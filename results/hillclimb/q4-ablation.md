# Dense Q4 ablation and source control

**Final: NO-GO; immutable original build/pack restored on all eight owners and
`.42:8080` verified with health, models and a real chat.**

- Found 96 source-mismatched residual matrices, not a demonstrated cooking bug.
  Aligned Q8 control: 480 rows, mean KL 3.96e-14, 100% top-1.
- All nine isolated Q4 families tested on the same frozen 480 frontiers.
  Best tested numerical mix: mean .003129, p99 .031049, top-1 98.9583%.
- Mixed full harness: 18/30 performance gates, 4/5 exact greedy references.
  Fresh same-binary Q8 full control: 22/30 performance gates, 5/5 greedy.
  Neither performance run qualifies against the unchanged historical baseline.
- Real-N=128 dense Q4 kernels all beat paired Q8. Fresh canonical Q8 itself
  loses 14.50% engine 128K prefill versus history; no Q4-only dense-path cause
  for the historical regression is established. Cause remains unisolated.
- All candidate packs retained; stopped scratch/raw remote captures cleaned.
  Compact metrics/identity receipts: [`q4-ablation-evidence.json`](q4-ablation-evidence.json).

## Binding reference and initial control

The immutable reference is build `20260926-abl2`, pack
`/home/xander/fg-ring-pack-abliterated`, on ranks `.42`–`.49`.
Numerical comparisons use the same `de8e1067…` engine/SPVs for teacher and
candidate. `.50` is excluded. All lifecycle operations are manifest/PID scoped.

The originally extracted BF16 checkpoint does **not** reproduce the deployed
dense weights. BF16 → ggml Q8_0 → cooked pack → loader → full-vocabulary capture
on the original 80 frontiers gives:

| Mean KL | p99 KL | Top-1 | Samples |
|---:|---:|---:|---:|
| 0.0463669231 | 0.5905052937 | 87.5% | 80 |

Consequently the earlier ordinary/protected Q4 qualification was not a clean
quantization-only ablation. Its measured rejection remains valid, but attributing
all its KL to Q4 sensitivity is invalid.

### Complete deployed source check

`q4_extract_pack.c` reads the sealed production manifest and verifies the saved
SHA256 of **every** selected tensor before reversing the Q8 16-row layout
byte-for-byte. No production file is changed. Its 496-matrix GGUF is
3,801,939,104 bytes, SHA256
`918a6175bf6082863b2306c125802970164cbe9777847997e5c1c51aa3096684`.

All 496 complete Q8 payloads were compared against the BF16 pipeline control:
**400 identical, 96 different**. The differences are exclusively:

| Family | Matrices | Median sampled relative L2 | Median cosine |
|---|---:|---:|---:|
| Shared expert down | 48 | 0.0202385 | 0.9997952 |
| GDN out | 36 | 0.0192030 | 0.9998157 |
| QSA output | 12 | 0.0187329 | 0.9998245 |

The hash comparison covers every payload byte; L2/cosine characterize 128
uniformly distributed blocks per differing matrix. Codes change, and scale
ratios vary around one: this is **not** a missing constant scale factor.
The exact upstream provenance of the deployed residual edits is not established
by this comparison; it must not be invented.

Imatrix mapping was independently checked by exact tensor name, F32 type and
input width: **494/496** matrices have matching sums/counts. The supplied
architecture-matched calibration lacks `output_hc_down.weight` and
`output_hc_up.weight`; these two globals use unweighted quantization, not an
incorrectly substituted importance vector. Imatrix SHA256:
`fd1bcb4e4b57e19c89b2583182f97d7f1e7c74ce941bb02f56835d648caaefe5`.
It is not calibration from the deployed residual-edited checkpoint.

The corrected source is explicit: retain the extracted BF16 for the 400 verified
matrices, but use exact FP32 dequantization of the deployed Q8 for the 96 differing
residual matrices. This is **not** a claim to have recovered their original BF16
values. `--q8-overrides` requires the comparison map and validates each override's
shape and complete production payload hash. Routers, norms, embeddings, routed
experts, and the Q8 vocabulary head remain original pack records.

Requantizing this aligned source to Q8 reproduces **496/496 complete payloads
exactly**. Full control GGUF SHA256:
`f6b5646bcb214a4784ab662b229b62dbac77db44f36b749c9faee87f722eed8b`.
Normalizing its matrix selection/order/header gives the same entire-file
`918a6175…` hash as the deployed extraction; thus the staged normalized control
is byte-identical to the actual quantizer output, not a substitute assertion.
`q4_compare_source.py --require-identical` is the fail-closed provenance gate.

### Corrected live Q8 control: PASS

The aligned control traversed quantization, repacking, persisted cooking,
single-arena loading, and the same engine/full-vocabulary capture as the
private original-Q8 teacher:

| Mean KL | p99 KL | Top-1 | Samples |
|---:|---:|---:|---:|
| 3.955364631535805e-14 | 5.304962918027488e-13 | 100% | 480 |

Canonical comparison-corpus SHA256:
`153b30d2cc11f7488615b4add737a82b06a3c419e751e5649cc1b021a74ad2ca`.
The actual UTF-8 JSON file (including serialization/newlines) SHA256 is
`9f888ee1c252d1ec41bf45ff0b50876adb2c4ae0ad7fe49e2f59427e04037c07`.
Aligned manifest SHA256, identical on all eight owners:
`e23e8f66d44adc014ac56d3dc44772d038440b91b8b8bac190ff23c750378808`.
The capture completed all 480 frontiers; the immutable original service was
restored and its health checked afterward. This rules out a material numerical
error in the tested extraction/requantization/cooking/loader/capture chain.
It does not turn the 96 deployed-Q8 overrides into recovered BF16 source.

### Position dependence in the failed control

The old fixed-prefix corpus has 20 prompts × four artificial assistant prefixes:

| Prefix | Mean KL | Largest KL | Correct top-1 / 20 |
|---|---:|---:|---:|
| Empty | 3.7364e-8 | 3.5942e-7 | 20 |
| `First` | 0.0507372 | 0.2668611 | 19 |
| `First, consider` | 0.0367818 | 0.2028555 | 15 |
| `First, consider the` | 0.0979487 | 0.5905053 | 16 |

There is no common absolute-position failure in this evidence. The nearly forced
first token masks drift; later contexts expose it nonuniformly. A new corpus
freezes 24 distinct complete UTF-8 prefixes from each of 20 actual Q8 teacher
continuations: **480 next-token frontiers**, no candidate feedback. Both sides
reencode identical frozen text; native generation positions/token IDs are saved
separately rather than pretending text reencoding preserves original BPE splits.

## Family ablation: 480 frozen frontiers per case

Only the named family changes; all other eligible dense matrices remain deployed
Q8. These clean isolation cases also keep original F32 routers. The same 20
prompts contribute 24 teacher-text frontiers each: 480 correlated next-token
samples, not 480 independent prompts. Every comparison uses all 248,320 logits.

| Q4 family | Matrices | Mean KL | p99 KL | Top-1 | Gate | Owner weight bytes saved |
|---|---:|---:|---:|---:|---|---:|
| All nine together | 496 | .0163354141 | .2367556977 | 97.9167% | FAIL | 1,706,778,624 |
| GDN qkv | 36 | .0013931986 | .0248478437 | 99.1667% | PASS | 457,113,600 |
| GDN out | 36 | .0034019298 | .0547739203 | 99.1667% | FAIL p99 | 274,268,160 |
| GDN gates (`attn_gate`) | 36 | .0007236581 | .0113986171 | 99.1667% | PASS | 274,268,160 |
| QSA q/k/v/output | 48 | .0086683254 | .1042523429 | 97.9167% | FAIL p99 | 284,098,560 |
| Shared expert gate | 48 | .0005312644 | .01021051 | 99.1667% | PASS | 24,576,000 |
| Shared expert up | 48 | .0005345945 | .01002578 | 99.5833% | PASS | 24,576,000 |
| Shared expert down | 48 | .0008419033 | .01406947 | 99.3750% | PASS | 39,321,600 |
| HC, including global HC | 194 | .004473990 | .06449179 | 99.1667% | FAIL p99 | 312,684,544 |
| PLE key/value | 2 | .0005483204 | .01249128 | 99.5833% | PASS | 15,872,000 |

Source-corrected all-Q4 position bins:

| Teacher continuation steps | Samples | Mean KL | Top-1 |
|---|---:|---:|---:|
| 0–3 | 80 | .0012079114 | 100% |
| 4–7 | 80 | .0058483218 | 98.75% |
| 8–15 | 160 | .0188723415 | 98.75% |
| 16+ | 160 | .0266057841 | 95.625% |

Largest divergences are context-specific (`03-009`, .583889; `03-013`, .483607;
`05-014`, .352355), not one universal broken token position. The original 80-row
and new 480-row corpora differ; do not interpret their means as a controlled
quantitative improvement ratio.

Native owner Q8 weights sum to **82,524,160,000 bytes**. Summing the nine isolated
family savings exactly reproduces the all-Q4 saving on **each** owner; the direct
aligned-Q8 native ledger independently supplies the baseline.

### Joint mixed plans

`mix-01` protects HC with Q5_K down / Q8 up; the other eight families remain
Q4_K/IQ4_NL. `mix-02` additionally protects GDN out and all QSA projections with
Q5_K. Unsupported K=320/640 Q5/Q6 requests explicitly remain Q8, never mislabeled
IQ4_NL. Both import only the 48 BF16 router records from the historical GGUF;
the native repacker proves **every** value equals the original F32 router.
The 96 mismatched historical residual payloads are not imported.

| Plan | All-owner manifest SHA256 | Owner weights | Saved versus Q8 |
|---|---|---:|---:|
| mix-01 | `1eef0ee4e02efa481a818bd65897f4ea759876365a8c5f006914023123751396` | 80,890,208,256 | 1,633,951,744 |
| mix-02 | `f97a5b43a770510f683eba4e20ac2115c4b1218df2e43643995db2f00f1a2b2c` | 81,035,698,176 | 1,488,461,824 |
| mix-03 | `98583b41f1a87c05dba7128ce1b8d62e11f4e84e709104220537587435a2414c` | 81,052,745,728 | 1,471,414,272 |
| mix-04 | `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252` | 81,245,085,696 | 1,279,074,304 |

Joint quality is not the sum of isolated-family KL. All mixed captures use the
same frozen 480 full-vocabulary frontiers and exact-text metadata binding:

| Plan | Mean KL | p99 KL | Top-1 | Gate |
|---|---:|---:|---:|---|
| mix-01 | .0119914453 | .2068446229 | 98.1250% | FAIL mean/p99 |
| mix-02 | .0044906006 | .0554534419 | 99.5833% | FAIL p99 |
| mix-03 | .0047627401 | .0648117876 | 98.9583% | FAIL p99 |
| **mix-04** | **.0031285779** | **.0310492419** | **98.9583%** | **PASS** |

`mix-03` starts from `mix-02` but restores PLE and both global HC matrices to Q8.
Its manifest is
`98583b41f1a87c05dba7128ce1b8d62e11f4e84e709104220537587435a2414c`.
It did not improve the gate, illustrating context-dependent error cancellation;
more protected bytes are not automatically a quality improvement.
`mix-04` instead starts from `mix-02` and restores all QSA projections to Q8;
GDN out and HC down remain Q5, with Q4 GDN qkv/gates, shared experts and PLE.
This is the best **jointly passing tested** mixed policy, saving
**1,279,074,304 owner bytes (1.55%)** versus original Q8. It is not a global
optimality claim. Q6 source artifacts were prepared, but no Q6 mixed plan was
live-qualified; unsupported widths never silently drop below Q4.

For `mix-02`, seven frontiers exceed KL .05: `07-014` (.246216),
`01-017` (.117433), `04-019` (.079094), `03-022` (.072831),
`14-022` (.055453), `19-006` (.054212), `15-006` (.051348).
At 480 samples the implemented nearest-rank p99 is the fifth-largest KL.

These are owner-weight totals, not total process/DRM memory. Coordinator foreign
globals and helper head slices require the separate accounting below. No source
Q8 arena is retained alongside Q4 weights.

### Matched source-only prefill control

The diagnostic engine is the identical eight-owner binary
`20261015-q4-prefill`, SHA256
`5eee9006b950676777e7b7ef94070c6004caa9fceeeb44c38e1eea34a1063b9d`.
Its only additional serving change is optional read-only route geometry logging;
normal numerical/performance gates use `de8e1067…`, not this intrusive profiler.

The corrected ordinary all-Q4/BF16-router source has SHA256
`78283e1938fe5297a4f4cfdae9be57f40e52bba2e35ec1f9853c2820a8195a4e`;
its sealed eight-owner manifest is
`4f147223ba46273841711e738a2d785b25d98f5d52a521635a1a30837fb307ff`.
All 496 raw Q4 payloads were compared with historical ordinary Q4:
400 identical, precisely the known 96 residual overrides different.
`q4-compare-packs` independently reads both sealed native manifests:
**1,225 records, identical tensor order, type, shape, layout, byte size, owner and
topology; exactly those same 96 payload hashes differ**. Self-comparison changes
zero records; comparison against a different Q8 format policy is rejected.
Thus old-versus-corrected ordinary profiles hold cooked layout and BF16 router
policy fixed, unlike a comparison against the F32-router isolation pack.

Each profile uses the same 4K warm request and byte-identical frozen 128K request.
SHA256 equality of serialized request bodies is mandatory. Eight log byte offsets
exclude warm-up; owner RPC durations and route tile occupancy are reported
separately from GPU queries. Rank 0 queries cover one chunk, and worker GPU queries
cover owner-layer windows—not all distributed expert work. Neither overlapping
worker durations nor partial GPU sums are whole-model wall time.

### Controlled whole-model prefill findings

The nominal 128K frozen request encodes **131,099 prompt tokens**. Its serialized
body SHA256 is
`da5e48288d331e0895d2b002d1f9fecabf6d2b169635482e481bcf536b8fe98f`.
Each case has a 4K warm-up, identical request bytes, the same eight owners, and
the same binary within the paired comparison:

| Case | Profiling | Wall seconds | Engine prompt tokens/s |
|---|---|---:|---:|
| Deployed Q8, diagnostic binary | on | 357.136493 | 367.401945 |
| Historical ordinary Q4, diagnostic binary | on | 348.672346 | 376.323525 |
| Source-aligned ordinary Q4, diagnostic binary | on | 346.304687 | 378.897366 |
| Deployed Q8, diagnostic binary | off | 372.568866 | 352.170119 |
| Historical ordinary Q4, diagnostic binary | off | 341.881300 | 383.804126 |
| Immutable original binary and original pack | off | 417.452765 | 314.262985 |

The unprofiled paired Q4 run is **8.98% faster** than paired Q8; profiled
source-aligned Q4 is **3.13% faster** than paired Q8. The original-build check
first verified original binary/manifest identities on all eight owners and used
the same frozen body. It is diagnostic, **not a replacement baseline**.
These are distinct requests from the canonical harness, so their absolute rates
must not be substituted into its fixed performance gates.

Route coverage is exactly 131099 × 48 × 10 = **62,927,520 token/expert pairs**,
49,200 dispatches. The actual grouped expert tile width is **4**:

| Case | Tiles | Padded capacity | Distinct experts across calls | Occupancy |
|---|---:|---:|---:|---:|
| Q8 | 16,934,144 | 67,736,576 | 2,698,272 | .9290035564 |
| Historical Q4 | 17,114,548 | 68,458,192 | 3,056,980 | .9192109543 |
| Aligned Q4 | 17,121,599 | 68,486,396 | 3,082,691 | .9188324058 |

Q4's padded tile work is about 1.1% larger than Q8's; the source correction adds
only .0412% tiles relative to historical Q4. No `PREFILL_WORKER` RPC records
appear in these runs; no remote-fanout explanation is inferred from absent data.

Partial GPU query totals are Q8 **1259.782099 s**, historical Q4
**1100.375402 s**, aligned Q4 **1100.504729 s**. These overlapping/partial owner
windows are explicitly **not** whole-model time. Selected accumulated GPU ms:

| Kernel | Q8 | Historical Q4 | Aligned Q4 |
|---|---:|---:|---:|
| Dense cooked Q8 tile | 460654.115 | 0 | 0 |
| Dense compact tile | 0 | 285449.8 | 285343.2 |
| Dense compact IQ4 tile | 0 | 31535.4 | 31521.9 |
| Dense wide tile | 0 | 24715.7 | 24709.0 |
| Dense F32 | 115495.995 | 16614.2 | 16613.9 |
| Dense BF16 router | 0 | 53895.4 | 53894.4 |
| Grouped routed kquant | 288402.721 | 291868.2 | 292026.2 |
| GDN recurrence | 81037.427 | 80980.6 | 80965.3 |
| QSA top-k merge | 54396.109 | 54403.4 | 54406.6 |
| QSA attention split | 48191.604 | 48256.0 | 48251.5 |
| QSA index score | 46367.766 | 46389.1 | 46413.3 |

Thus the historical −16% 128K result is **not reproduced by these controlled
pairs**, and no regressing dense kernel/path has been found. The historical
cross-run cause remains unisolated; this is not evidence for speculative cache,
thermal, network or routing causes. The earlier Q8 same-binary quick control
already measured only 309.049 engine 4K tokens/s versus historical 355.12.
Its runtime ledger matches the baseline configuration: batch 128, window 2,
eight prefill frames, QSA cache 131072 pages / 648019968 bytes, context 262144,
and original weight total 82524160000 bytes. There is no measured configuration
mismatch to “fix,” and no duplicate Q8 prefill-only weight allocation was added.
The final canonical full harness remains authoritative and unchanged.

## Independent cooking and real-N prefill checks

Independent ggml dequantizers versus both cooked layouts passed
**22,527,744 exact FP32-value checks**, spanning 992 ordinary/protected tensor
instances and first/middle/last rows. This rules out the tested scale/min/IQ4-grid
cooking errors, not arbitrary accumulation or source-provenance errors.

Actual GGUF matrices, `.53` (24 CU), supplementary N=256, 20 warmed GPU-timestamp
repetitions, same blade/run; all sampled GPU parity checks pass.
The actual production microbatch is **128**, as printed in `FG_LEDGER batch=128`.
The 10,485,760-byte ping-pong buffer contains **two** input slots
(`runtime.c`: microbatch × 10240 × 4 × 2), not one N=256 input.
These are individual dense kernel times, **not** end-to-end throughput:

The corrected-source run at the verified **N=128**, again 20 warmed paired GPU
timestamp repetitions on `.53`, gives:

| K × M | Compact Q4/IQ4 µs | Wide µs | Q8 µs | Compact / Q8 speedup |
|---|---:|---:|---:|---:|
| 2560 × 10240 | 2721.618 | 2724.328 | 3652.396 | 1.342× |
| 2560 × 6144 | 1650.938 | 1646.862 | 2254.666 | 1.366× |
| 6144 × 2560 | 1910.130 | 2251.216 | 3070.284 | 1.607× |
| 2560 × 12288 | 3229.764 | 3227.730 | 4373.992 | 1.354× |
| 10240 × 320 | 793.036 | 895.236 | 983.956 | 1.241× |
| 320 × 10240 | 398.594 | 398.404 | 474.932 | 1.192× |
| 640 × 2560 | 201.636 | 201.944 | 225.742 | 1.120× |
| 2560 × 640 | 252.456 | 252.682 | 481.242 | 1.906× |
| 2560 × 512 | 245.994 | 246.860 | 415.006 | 1.687× |

Every sampled Q4 output passes scalar parity. Original F32 MoE router versus
losslessly narrowed BF16 router, actual `blk.0.ffn_gate_inp.weight`, K=2560,
M=512, **every output bit-exact**: at N=128, F32 2232.864 µs versus BF16
1217.154 µs (1.834×); at N=256, 4485.332 versus 2473.832 µs (1.813×).
Both are warmed paired 20-repetition GPU timestamps. Thus the prior router
narrowing also does **not** explain the whole-model prefill slowdown.
The same actual router also passes **every-output bit-exact N=1 decode parity**:
F32 16.454 µs, BF16 11.136 µs, 20 warmed GPU repetitions on `.53`.

Supplementary N=256:

| K × M | Compact Q4/IQ4 µs | Wide µs | Q8 µs | Compact / Q8 speedup |
|---|---:|---:|---:|---:|
| 2560 × 10240 | 5349.698 | 5354.414 | 7231.656 | 1.352× |
| 2560 × 6144 | 3217.176 | 3213.384 | 4365.642 | 1.357× |
| 6144 × 2560 | 3614.562 | 4159.270 | 5030.650 | 1.392× |
| 2560 × 12288 | 6360.206 | 6370.858 | 8685.918 | 1.366× |
| 10240 × 320 | 1053.836 | 1316.560 | 1646.328 | 1.562× |
| 320 × 10240 | 782.124 | 782.582 | 960.022 | 1.227× |
| 640 × 2560 | 382.232 | 382.578 | 443.926 | 1.161× |
| 2560 × 640 | 442.854 | 446.876 | 669.926 | 1.513× |
| 2560 × 512 | 341.344 | 343.898 | 545.746 | 1.599× |

The benchmark's `DENSE_PRIOR` column is the alternative wide layout; its storage
label names the compact candidate rather than the prior. Dispatch/profile
classification was checked in `quant_dense_shape()`. The deployed M=640/512
GEMV policy still retains wide storage; both variants beat Q8 at N=256.
All paired Q4 candidates beat
Q8, so the previous whole-model −16% 128K regression is not established as a
dense path-choice regression. Source-dependent routing/attention behavior
must be separated from kernel speed. No resident Q8 prefill fallback is used.

## Resident weight accounting

The mixed policy uses six owner layers/rank and the original Q8 vocabulary head.
Native sealed ledgers for `mix-04` (decimal bytes):

| Rank / owner | Original Q8 owner weights | Mixed owner weights | Mixed native residency budget |
|---|---:|---:|---:|
| 0 / .42 | 10,623,664,128 | 10,478,895,104 | 11,841,650,688 |
| 1 / .43 | 10,723,561,472 | 10,536,951,808 | 11,934,474,240 |
| 2 / .44 | 9,954,455,552 | 9,783,717,888 | 11,114,131,456 |
| 3 / .45 | 9,948,233,728 | 9,803,464,704 | 11,166,220,288 |
| 4 / .46 | 10,636,890,112 | 10,464,976,896 | 11,795,390,464 |
| 5 / .47 | 10,210,377,728 | 10,065,608,704 | 11,428,364,288 |
| 6 / .48 | 9,954,455,552 | 9,783,717,888 | 11,114,131,456 |
| 7 / .49 | 10,472,521,728 | 10,327,752,704 | 11,690,508,288 |
| Sum | 82,524,160,000 | 81,245,085,696 | — |

The native budget includes additional planned runtime allocations; it is **not**
a measured process-memory total. Coordinator rank 0 also maps **681,259,008**
foreign-global bytes, making its weight total **11,160,154,112** bytes. Ranks
1/2 additionally map **168,857,600** Q8 head-slice bytes each. These are shared
serving roles, not hidden raw/cooked duplicates. Persisted layout 6/7 tensors map
directly into the verified arena; no original dense source arena or Q8-only
prefill duplicate is resident.

Measured immediately after candidate startup, before the full harness:

| Owner | DRM resident GTT + VRAM bytes | Process RSS bytes |
|---|---:|---:|
| .42 | 11,965,501,440 | 84,238,336 |
| .43 | 11,198,885,888 | 188,178,432 |
| .44 | 10,443,599,872 | 188,342,272 |
| .45 | 10,629,296,128 | 188,985,344 |
| .46 | 10,960,633,856 | 188,403,712 |
| .47 | 10,891,444,224 | 188,977,152 |
| .48 | 10,274,078,720 | 188,370,944 |
| .49 | 11,159,564,288 | 189,358,080 |

DRM clients are deduplicated by client ID (one client per process here). These
include runtime allocations and are snapshots, **not peak 128K memory**. RSS is
separate host process accounting, not an alternate measure of GPU weights.

## Full canonical mixed qualification: NO-GO

The unmodified full harness ran `mix-04` on `.42`–`.49`, normal `de8e1067…`
binary, batch 128/window 2, 700-MiB CLI QSA cache (native capacity
648019968 bytes), without profiling. Run ID
`4d7b30259f34447e94e4f22206096369`, started
`2026-10-15T14:35:13.7830840-07:00`, elapsed **1040.1183724 s** versus
historical 930.04 s (**+11.84%**).

Quality passes, but only **18/30 performance gates** and **4/5 exact greedy
references** pass. Therefore the candidate was **not promoted**. No baseline,
threshold, expected text/hash, or metric was changed.

Prefill (historical baseline → candidate):

| Context | Client TPS | Engine TPS | Stream first content ms | Failed fields |
|---|---:|---:|---:|---|
| 4K | 339.910 → 341.197 | 355.120 → 300.660 | 12226.820 → 12192.364 | engine (−15.34%) |
| 32K | 357.020 → 383.146 | 420.860 → 379.974 | 91952.510 → 85682.743 | engine (−9.71%) |
| 128K | 415.580 → 377.555 | 415.470 → 342.981 | 315527.540 → 347303.062 | client −9.15%, engine −17.45%, latency +10.07% |

Client streaming and engine nonstream timings use independently salted requests;
they are not two measurements of the exact same generation. The matched fixed-body
profiles above do **not** erase this canonical failure.

All 12 decode metrics pass:

| Context | Client TPS | Engine TPS | Inter-event p50 ms | p90 ms |
|---|---:|---:|---:|---:|
| Short | 26.760 → 27.636 | 25.800 → 25.809 | 40.730 → 40.504 | 42.030 → 41.328 |
| 32K | 24.440 → 25.860 | 23.840 → 25.816 | 40.750 → 40.702 | 41.550 → 41.585 |
| 128K | 22.320 → 23.212 | 21.770 → 22.986 | 44.680 → 42.979 | 45.430 → 43.738 |

Remaining metric gates:

| Metric | Baseline | Candidate | Gate |
|---|---:|---:|---|
| Chat 30-token ceiling TTFT ms | 2096.320 | 2185.939 | FAIL +4.28% |
| 4K cold TTFT ms | 12226.820 | 12192.364 | PASS |
| Cached agent-prefix TTFT ms | 713.040 | 696.935 | PASS |
| Concurrent chat token TPS | .590 | .549 | FAIL −6.98% |
| Concurrent chat wall s | 15.280 | 16.398 | FAIL +7.32% |
| Concurrent agent token TPS | 2.100 | 1.160 | FAIL −44.77% |
| Concurrent agent wall s | 15.270 | 16.383 | FAIL +7.29% |
| Concurrent aggregate token TPS | 2.680 | 1.707 | FAIL −36.29% |
| Complete harness wall s | 930.040 | 1040.118 | FAIL +11.84% |

Arithmetic, capital, sequence and logic exact hashes match. Spelling emits the
correct word but stops at 57 actual tokens, changing the reference hash; this
is still a strict **failure**, not silently accepted as semantic equivalence.
Concurrency TPS counts actual emitted tokens and may reflect EOS behavior;
its failed metric is retained without normalization or exception.

The harness's `finally` block stopped only candidate-manifest PIDs, removed
stopped candidate QSA state, restored all eight immutable-original instances,
and recorded `promoted=false, healthy=true` at
`2026-10-15T14:53:43.7597944-07:00`. A subsequent same-binary **full Q8 control**
is diagnostic only and cannot promote or redefine the reference.

### Full same-binary Q8 performance control

The aligned Q8 pack `e23e8f66…`, same normal `de8e1067…` binary on all eight
owners, was then run through the **entire unchanged canonical harness**. Its
near-zero-KL 480-row quality record is the control, not the mixed record.
Promotion is explicitly forbidden for this diagnostic control.

The control matches **5/5 greedy references**, but only **22/30 performance
gates**. Elapsed **1069.9695589 s**, versus historical 930.04 (+15.05%).

| Prefill metric | Historical | Fresh Q8 control | Mixed Q4/Q5/Q8 |
|---|---:|---:|---:|
| 4K client TPS | 339.910 | 345.591 | 341.197 |
| 4K engine TPS | 355.120 | 316.778 | 300.660 |
| 32K client TPS | 357.020 | 368.907 | 383.146 |
| 32K engine TPS | 420.860 | 359.089 | 379.974 |
| 128K client TPS | 415.580 | 355.636 | 377.555 |
| 128K engine TPS | 415.470 | 355.218 | 342.981 |
| 128K stream first content ms | 315527.540 | 368719.863 | 347303.062 |

Fresh Q8 alone has **−14.42% client / −14.50% engine 128K prefill** and
**+16.86%** stream-first-content latency versus history. Its other failures are
4K/32K engine prefill, concurrent agent/aggregate token TPS, and full elapsed
time. All 12 decode metrics and all three TTFT metrics pass.

Mixed versus this fresh Q8 run is +6.16% client 128K TPS, −3.44% engine 128K
TPS, and −2.79% total wall time. These full harness runs use different fresh
salts, so they are **not byte-identical workload A/Bs**; the explicitly frozen
profiles above provide that separate control. They show that most of the
historical prefill deficit also exists without any Q4 weights. They do not
explain every residual difference, prove a universal Q4 model-level speedup,
or authorize ignoring the fixed baseline.

The exact historical cause remains unisolated. No demonstrated regressing
dense kernel or selection path was found to fix; introducing a lossy half path
or resident Q8 fallback would not be justified. The original service was again
restored by the control's `finally`, healthy at
`2026-10-15T15:14:09.0546664-07:00`.

## Final state, retained artifacts and disk

After the last control, all eight serving processes were independently matched
to the **exact original manifest argument and original executable path**.
Each owner verifies:

- Original executable SHA256
  `e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2`.
- Original manifest file SHA256
  `35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`.
- Exactly one original-manifest serving process per owner.

`.42:8080/health`, `/v1/models`, and a greedy real chat all succeed; the chat
returns `OK OK OK OK OK OK OK OK`. Final health is `ok`, not busy.
No candidate is serving. No original model/build artifact was edited.
`.50` was never contacted.

Only explicitly enumerated task-owned, stopped scratch was removed: staging
source/build trees, redundant source GGUFs, copied profiler logs/tars, and raw
quality captures in our private release operators. Cleanup refuses paths
referenced by live command lines and refuses symlink cleanup roots. Every new
candidate pack remains sealed in its separate directory. The best tested pack
is `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04` on every owner:

| Owner | Retained mix-04 allocated extents B | Apparent B | Filesystem available B |
|---|---:|---:|---:|
| .42 | 11,425,914,880 | 20,944,743,127 | 119,687,278,592 |
| .43 | 12,289,277,952 | 21,002,799,839 | 42,592,161,792 |
| .44 | 10,730,737,664 | 20,249,565,919 | 52,730,228,736 |
| .45 | 9,804,341,248 | 9,804,335,839 | 13,844,611,072 |
| .46 | 10,465,853,440 | 10,465,848,031 | 64,842,285,056 |
| .47 | 10,066,485,248 | 10,066,479,839 | 46,061,887,488 |
| .48 | 9,784,594,432 | 9,784,589,023 | 48,208,367,616 |
| .49 | 11,073,445,888 | 20,793,600,735 | 9,552,834,560 |

`du` includes shared COW extents in each pack and is **not exclusive incremental
disk cost**; apparent sizes include sparse foreign files. The evidence lists
all retained packs, not only `mix-04`. `.49` is still tight; do not stage
additional multi-GB overlays there without checking space.

On `.53`, pre-existing verified BF16/imatrix/historical overlays/ggml build were
preserved. New verified aligned Q4, Q5 and selected mixed source GGUFs remain in
`/home/xander/hc-q4/ablation-source-check` (**6,506,151,296 bytes** total), with
provenance maps/inventories. The deployed Q8 extraction remains separately at
`/home/xander/hc-q4/production-dense-q8.gguf` (**3,801,939,104 bytes**).
The three retained aligned-source SHA256s are:

- Q4 `bd9e6c23e2e17f81ac6c1372b769c521fdf0b2af82e057ca0544570cb1fee48c`.
- Q5 `cdb8fd6eff916b4ad1b239999f22b45d570b52dff1ac9551693c3c0e81c979dd`.
- `mix-04` `e742dbf593f03c5c608de229329755e7cb19138e7b1c4031632eaf49f1acf4d1`.

`.53` has **59,027,693,568 available bytes** after cleanup. Pre-existing `.52`
evidence was not removed. Nine redundant local staging GGUFs were also removed
(26,503,622,912 bytes); local capture evidence is retained uncommitted.
Staging build trees are intentionally gone: rebuild tools from this commit
before constructing another pack, not from a stale remote staging executable.

## Reproduction and validation

All remote work stays under `/home/xander/hc-q4*`; do not rerun an existing
exclusive-output capture or overwrite a sealed pack. The original pack and
release remain required read-only dependencies for external artifacts.

The corrected source uses the recorded full-payload comparison map, not a
shape-based guess:

```sh
cd /home/xander/hc-q4
# Run from a checkout containing the committed tools; choose new output names.
python3 tools/q4_quantize.py --source dense-bf16.gguf \
  --q8-overrides production-dense-q8.gguf \
  --override-map ablation-source-check/source-comparison.json \
  --deployed-dense-only --q8-control \
  --ggml-library llama-build/bin/libggml-base.so \
  --output aligned-q8-new.gguf
python3 tools/q4_compare_source.py --production production-dense-q8.gguf \
  --control aligned-q8-new.gguf --require-identical --output aligned-proof-new.json
# Same source/override arguments, without --q8-control, plus:
# --imatrix dense-imatrix.gguf produces ordinary Q4; --q5-dense produces Q5.
```

For a staged, fresh mixed pack, the actual qualification commands are:

```powershell
Set-Location D:\looking-glass-labs\fg-wt\q4-dense
$r='.\results\hillclimb\q4-ablation'
& .\tools\q4-ablation-capture.ps1 `
  -Manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm `
  -Tag mix480-fresh -OutputDir $r `
  -Corpus "$r\natural-corpus\natural480.json" -TeacherDir "$r\teacher480r2"
# Only after that fresh full-vocabulary quality capture passes:
& .\tools\q4-ablation-harness.ps1 `
  -Manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm `
  -ManifestSHA 0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252 `
  -Quality "$r\mix480-fresh-quality.json" -Tag mix04-fresh `
  -OutputDir "$r\final"
```

Use fresh tags/destinations (the example names above must not already exist). Omit
`-PromoteOnPass` for a diagnostic run; failed quality is refused before service
mutation. The helper uses the **unmodified full**
`D:\looking-glass-labs\flash-gordon\tools\hillclimb.ps1` and historical
`BASELINE.json`, SHA256
`089f798462e11d25608ae5dfa3b073aa959934958c2d0d3426735de8f7874f72`.
All 30 comparable performance metrics and all five greedy hashes must pass;
no relaxed baseline or fresh-control rebaselining is permitted.

Validation: 17 Python tests pass with the native ggml library on `.53`; locally,
17 pass with the one native-library case skipped. PowerShell stage and
capture/profile/harness lifecycle mocks pass, including rollback failure paths
and conditional promotion. All Q4 PowerShell files parse; `git diff --check`
passes. Native sealed-manifest comparison passes self/source-only checks and
rejects a different-format policy. The previous eight native repacker tests and
expert-prefill profiling-off/on Vulkan parity checks pass. No full Vulkan-suite
pass is claimed; unrelated baseline failures remain documented separately.
