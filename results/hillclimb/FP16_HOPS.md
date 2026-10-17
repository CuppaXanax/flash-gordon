# Scaled FP16 decode-ring work hops

Research-only, default off. All shader computation and accumulation remain f32.
Transport measurements are not serving throughput, and do not waive either
the incremental or cumulative model-quality gate.

**No promotion:** both all-hop FP16 modes fail incremental KL. Mask 224 passes
incremental KL/top-1, but the historical-original-Q8 cumulative gate is not met,
including by the unchanged f32 baseline. Matched-reference control is separate.

**Reconciled scope:** [QUALITY_RECONCILIATION.md](QUALITY_RECONCILIATION.md)
replays both the same 80 forced-prefix contexts and the original 480 natural
frontiers through cold N1 and native prefill. Historical Q8/mix-04/stack
captures reproduce byte-for-byte. Mix-04 passes the original native
Natural480 gate (0.003128578 / 98.9583%), but fails Forced80 even with the
original method (0.059809481 / 85%). Stack's 0.000246 is incremental versus
mix-04; direct cumulative stack/Q8 is 0.003084267 / 98.75% on Natural480.
The baseline discrepancy is chiefly cohort sensitivity, not a bad Q8 control.
The Forced80 failures below remain valid; mask 224 is still unqualified.

## Codec and safety

`FG_FP16_HOPS=1` uses one f32 scale per work message, independently per batch
slot. `FG_FP16_HOPS=128` uses one scale per 128 concatenated hyper/n-gram values.
`FG_FP16_HOP_MASK` optionally selects destination ranks; unselected sends remain
f32. `FG_FP16_SCALAR=1` selects the diagnostic portable implementation.
All flags are registered in `tools/check-flags.sh` and `docs/FLAGS.md`.

Each scale is a positive power of two:
`max(FLT_MIN, 2^(ilogb(maxabs)-14))`, or one for an all-zero group.
IEEE binary16 round-to-nearest-even retains ten fraction bits. Normalized
magnitudes are at most 32768, leaving exponent headroom; the finite-f32 endpoint
is explicitly restored as signed `FLT_MAX`, never infinity. Scale/value checks
reject malformed and nonfinite inputs. Signed zero and subnormals are tested.
Rescaling is exact binary arithmetic, not another trained-weight quantization.

The big-endian scale precedes big-endian halves. Single/batched protocol-6 work
uses self-describing wire tags 64/32, mutually exclusive with BF16 tag 128.
Upgraded receivers decode either f32 or compressed work regardless of their
local send setting; old receivers reject new tags. F32 bytes remain unchanged.
BF16 and FP16 cannot be enabled together.

Compression covers ordinary decode work, including its initial n-gram vector.
It excludes prefill, MTP verification, expert results, head/output and native
state traffic. Either low-precision codec plus experimental MTP remains
fail-closed before readiness: that mixed-path configuration is not qualified.
No production rollout, long-context or dual-session residency claim is made.

## Five paired rounds: actual codec, framing and transport

Same `.52`/`.53` `fg_fabric` io_uring exchange as `BF16_HOPS.md`: real
serialization, CRC, validation and decoded-vector checks are included.
The estimate is RTT/2, not synchronized one-way time.
Five alternating A/B/A and B/A/B rounds; 500 RTTs and 50 warmups per leg;
ordinary/initial shape order alternates. Student-t 95% intervals are over five
paired rounds, not thousands of correlated packets.

| Mode / work | f32, µs | Scaled FP16, µs | Paired saving, µs |
|---|---:|---:|---:|
| One scale, ordinary | 560.658 ± 4.492 | 376.581 ± 3.000 | 184.077 ± 2.068 |
| One scale, initial + n-gram | 631.437 ± 33.070 | 446.024 ± 13.409 | 185.412 ± 24.314 |
| One scale, eight-hop model | 4,556.042 ± 39.600 | 3,082.088 ± 23.127 | **1,473.954 ± 22.988** |
| Block-128, ordinary | 543.959 ± 35.275 | 371.104 ± 33.946 | 172.855 ± 15.161 |
| Block-128, initial + n-gram | 636.122 ± 27.991 | 456.093 ± 20.994 | 180.029 ± 10.754 |
| Block-128, eight-hop model | 4,443.832 ± 255.471 | 3,053.820 ± 254.379 | **1,390.012 ± 100.637** |
| Initial f32 + seven FP16, model | 4,556.042 ± 39.600 | 3,267.501 ± 38.245 | **1,288.541 ± 14.473** |
| Destination 0 f32, model | 4,556.042 ± 39.600 | 3,266.166 ± 23.732 | **1,289.876 ± 22.622** |
| Destinations 5–7 FP16 only, model | 4,556.042 ± 39.600 | 4,003.810 ± 38.363 | **552.232 ± 6.203** |

Eight-hop estimates pair **one initial + seven ordinary** transfers within each
round. They exclude output/head and do not measure live-ring TPS.
Masked estimates compose the measured global-codec legs, rather than claiming
a separate masked-policy transport benchmark.
The two codec experiments have separate f32 controls; do not compare their
absolute candidate times without those paired controls.

| Mode | Ordinary frame | Initial frame | Eight work frames |
|---|---:|---:|---:|
| f32 | 41,016 B | 51,256 B | 338,368 B |
| One scale FP16 | 20,540 B | 25,660 B | 169,440 B |
| Block-128 FP16 | 20,856 B | 26,056 B | 172,048 B |

Global scaling adds only 32 B/token over BF16; block scaling adds 2,640 B/token.
The raw application-payload serialization saving at negotiated 1 Gb/s is
1.351424 / 1.330560 ms respectively. This is a serialization term, not a bound
on total measured CPU/framing/TCP savings.
The f32 ordinary-frame effective rate is 73.157 MB/s (585.255 Mb/s), including
all benchmark overhead; it is not independently measured wire-only TCP throughput.

The initial portable implementation **lost** 0.779 ± 0.344 ms/token globally
and 1.176 ± 0.034 ms/token with block scales. These negative receipts are retained.
The positive results above require runtime-checked AVX/F16C/SSSE3 conversion,
SIMD byte swaps, max reduction and exact power-of-two multiplication.
Unsupported CPUs retain the portable path; its transport win is not presumed.

## Authentic model evaluation and reference controls

The sealed mix-04 manifest is
`0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
All 1,223 trained spans are authenticated against it. Prepared embedding inputs,
the complete n-gram source, tokenizer and 80 supplied contexts are the same
authentic assets described in `BF16_HOPS.md`.

Each independent causal owner session executes all 48 trained layers, real
routed experts and the full 248,320-vocabulary head on isolated `.50`.
Four supplied trailing tokens pass through the actual codec; common prefixes
remain f32. Primary: 80 final positions. Auxiliary: 320 trailing positions.
The predeclared primary bar is mean full-vocabulary KL ≤ 0.005 and top-1 ≥ 98.5%.
KL is `KL(P_reference || P_candidate)` and is never added or subtracted as an
error budget. Common-prefix component tolerance is 1e-4 relative, not a bitwise
model-quality requirement.

Global/block/BF16 policies compress all eight boundaries. Each `single-R` policy
compresses only destination rank R and propagates the perturbation through all
subsequent trained layers/state. Boundary order is destinations
**1, 0, 2, 3, 4, 5, 6, 7**, covering six layers each.
`mixed-first-f32` uses mask 253: initial destination 1 stays f32, seven later
boundaries use globally scaled FP16. A second diagnostic, `mixed-no-rank0`,
uses mask 254 to retain the second boundary (destination 0, layers 6–11) in f32.
This second mixed policy was selected after the destination-0 ablation failed
the incremental gate; it is an in-corpus diagnostic, not an independent holdout.
`mixed-screened` retains f32 at every destination whose isolated ablation fails
either primary threshold, then separately replays the combined selected mask.
Its selection receipt is retained; passing isolated hops does not imply that
their combination passes, and this screen is not a held-out fleet qualification.
Single-boundary KL values are not additive.

### Causal single-boundary results versus mix-04/f32

| Hop (1-based) | Destination | Layers entered | Mean KL | Top-1 | Isolated gate |
|---|---:|---|---:|---:|---|
| 1 | 1 | 0–5 | 0.009481269 | 100% | FAIL KL |
| 2 | 0 | 6–11 | **0.010740349** | 96.25% | FAIL both |
| 3 | 2 | 12–17 | 0.008337006 | 97.50% | FAIL both |
| 4 | 3 | 18–23 | 0.004869215 | 96.25% | FAIL top-1 |
| 5 | 4 | 24–29 | 0.003240267 | 97.50% | FAIL top-1 |
| 6 | 5 | 30–35 | 0.000647000 | 100% | PASS |
| 7 | 6 | 36–41 | 0.000036629 | 100% | PASS |
| 8 | 7 | 42–47 | 0.000004395 | 100% | PASS |

The second boundary dominates isolated KL, followed by the first and third.
These are causal perturbation experiments, not additive error shares.
All-hop global and block-128 modes produce the same primary result:
mean KL **0.009877370**, top-1 **79/80 = 98.75%** — **FAIL KL**.
The BF16 replay reproduces its prior **0.007226893 / 97.50%** result.
More mantissa bits reduce local wire error but do not guarantee monotonically
better full-model KL. Block scales do not rescue this failure.
All 320 recorded global/block full-vocabulary logit hashes agree; the additional
block scales provide no measured numerical benefit on these inputs.

### Combined masked policies and direct cumulative historical-Q8 comparison

| Policy | Incremental KL vs mix-04/f32 | Top-1 | Cumulative KL vs historical Q8 | Top-1 |
|---|---:|---:|---:|---:|
| f32 mix-04 | 0 | 100% | 0.055515349 | 83.75% |
| BF16 all hops | 0.007226893 | 97.50% | 0.054925715 | 85.00% |
| FP16 all hops | 0.009877370 | 98.75% | 0.059180929 | 82.50% |
| FP16 block-128 all hops | 0.009877370 | 98.75% | 0.059180929 | 82.50% |
| Initial boundary f32, mask 253 | 0.010835968 | 96.25% | 0.062519215 | 86.25% |
| Destination 0 f32, mask 254 | 0.016271618 | 98.75% | 0.057810786 | 85.00% |
| **Only destinations 5–7 FP16, mask 224** | **0.000678952** | **100%** | **0.057188228** | **83.75%** |

All columns use the same 80 final positions; cumulative KL is directly
recomputed from the original-Q8 and candidate vectors, not assembled from
incremental KLs. Every historical cumulative comparison fails the strict bar,
including the pre-existing f32 mix-04/profile baseline.

Retaining only the largest isolated-error boundary in f32 does **not** repair
the combined policy. The separately replayed screened mask **224 = 0xE0**
retains f32 for the first five boundaries and compresses only destinations 5–7.
It passes the incremental 80-position gate and auxiliary 320-position check
(mean KL **0.000478621**, top-1 **100%**).
Its modeled transport saving is **0.552232 ± 0.006203 ms/token**, not the
quality-failing all-hop model's 1.474 ms/token.
It remains a default-off, in-corpus research candidate, **not serving-qualified**.

The global/block auxiliary result is mean KL **0.006172791**, top-1
**317/320 = 99.0625%**; KL still fails. The primary cohort and thresholds were
not changed to select a passing auxiliary statistic.

Original-Q8 historical full-vocabulary captures come from
`q4-qualification/r3-teacher`, with byte-identical corpus SHA
`ad99ce71c9f87edc6e876fa956223b058036ebb718c9a198f9c908023011446c`.
Original manifest file SHA:
`35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`;
its distinct internal seal begins `4708a6600142d094`.
This 80-position cohort is **not** the older 480-natural-frontier Q4 cohort.

The fixed numerical profile is `FG_ROOFLINE=2047`, `FG_ROOF_DENSE=19`,
`FG_ROOF_QSA=3`, `FG_ROOF_EXPERT=33024`, `FG_ROOF_HC=96`,
`FG_ROOF_RECURRENT=3`, `FG_ROOF_GDN=1`, `FG_SMALL_KERNELS=330`,
`FG_EXPERT_Q8_WIDE=1`, `FG_NGRAM_LOCAL_CACHE=1`,
`FG_DECODE_SUBMIT_GROUP=2`. Both the mixed-pack codec comparison and matched
original-Q8 control use this same source/binary and profile. The historical
original-Q8 captures are not relabeled as a run of this profile.

The new f32 baseline reproduces every prior BF16 experiment teacher-vector SHA.
Nevertheless, historical-Q8 versus f32 mix-04 already yields mean KL
**0.055515349**, top-1 **67/80 = 83.75%**, before any codec is enabled.
The empty-assistant-prefix subset is near zero KL; supplied `First...` prefixes
carry the discrepancy. Historical serving/prefill versus frozen decode/kernel
geometry is a confound; this observation cannot be blamed on FP16 or solely on
mix-04 weight precision.

The separate same-geometry original-Q8 control restores original trained spans
from exact deployed dense Q8 extraction SHA
`918a6175bf6082863b2306c125802970164cbe9777847997e5c1c51aa3096684`,
unchanged mixed-pack spans and SHA-verified lossless BF16-router expansion.
It uses the original sealed manifest but the **same frozen decode/kernel stack**
and prepared token/embedding/ngram inputs as the codec candidates.
This control distinguishes historical-path differences from weight/codec
differences; it does not silently replace the historical original-stack gate.

### Cumulative result against authenticated, same-geometry original Q8

All **1,223 original-Q8 trained spans** pass restoration and independent
readback: 824 copied unchanged spans, 351 exact dense-Q8 cooked spans and
48 losslessly restored F32 routers. The original sealed manifest is unchanged.
The independently restored local/remote rank-0 files share SHA
`825879ad4068c1c610571064d1a92b0890ff2ada35cf8210c708e1b7a91f8c57`.
As elsewhere, whole-file identity is between reconstructions, not a claim about
unavailable deployed-file padding.

| Current frozen-decode policy | Direct KL(Q8 control \|\| candidate) | Top-1 | Gate |
|---|---:|---:|---|
| f32 mix-04 | **0.051191953** | **85.00%** | FAIL both |
| BF16 + mix-04 | 0.050399171 | 85.00% | FAIL both |
| FP16 + mix-04, all hops | **0.056666349** | **85.00%** | FAIL both |
| FP16 block-128 + mix-04, all hops | 0.056666349 | 85.00% | FAIL both |
| Initial f32, mask 253 + mix-04 | 0.057342843 | 86.25% | FAIL both |
| Destination 0 f32, mask 254 + mix-04 | 0.059804464 | 86.25% | FAIL both |
| **Destinations 5–7 FP16, mask 224 + mix-04** | **0.054000836** | **85.00%** | **FAIL both** |

The original-Q8 current-profile control itself differs from historical Q8:
mean KL **0.010931423**, maximum **0.122030488**, top-1 **92.50%**.
Thus the historical execution/profile/conditioning mismatch is real, but it
does **not** explain away the cumulative failure: the matched-geometry
mixed-pack baseline still fails by a large margin. This experiment does not
separate dense weight precision from dtype-dependent kernel arithmetic, nor
revise the earlier, different 480-natural-frontier Q4 qualification.
No subtraction of these KLs is a valid decomposition.

**Decision:** the masked codec is a useful incremental candidate, not a
qualified cumulative-Q8 stack. Both references fail the requested bar for
every combined codec policy; keep all new flags off pending a proper fleet
quality gate and resolution of the baseline/profile discrepancy.

## Reproduction and provenance

The immutable scalar numerical executable is
`11ce46e6cb662c4d70509f28301c50a1bb3ff1dd9be49b1516cd9cd453f57239`.
The measured optimized transport executable is
`77b93c5d30da462666f81eb6bb064746cf1790a12887c652322e112d2e18db1e`.
An independent hardware oracle compares complete serialized bytes and expanded
f32 structs on actual saved model inputs; it does not alter the numerical
executable or shaders while model evaluation runs.

`tools/analyze_fp16_hops.py` validates corpus/reference identities, prior f32
baseline continuity, candidate logit hashes, causal attribution and paired
transport arithmetic. `fp16_q8_restore.py` fails closed unless every restored
trained span matches the original sealed manifest, then independently reads
those spans back. Reconstructed whole-file hashes identify reconstructed padding,
not unavailable original whole-FGW identity.

Only isolated `.50`, `.52`, `.53` were used. Production `.42`–`.49` and `.51`
were not accessed. Governors and residency policy remain unchanged.
The transport pair was cleaned/released at 18:25:46Z; a later bounded read-only
Q8-source export on `.53` was independently cleaned/released at **18:48:37Z**.
The retained `/home/xander/hc-q4` source was not modified.

`.50` was finally cleaned/released at **20:51:54Z**: owned root absent twice,
no owned executable, both original embed/rerank services healthy, governor
configuration and performance level unchanged. No host was revisited after
its final cleanup. Numerical evaluation used the 40-CU `.50`; the CPU/network
transport pair used stock 24-CU `.52`/`.53`. No live-ring or 24-CU GPU timing
claim is inferred from these measurements.

### Validation and retained evidence

`make all`, focused C codec/decode-batch tests, seven Python tests, both
FP16-mode eight-process/two-session pipeline tests, flag registration and
20 LF-normalized Linux helper syntax checks pass. RNE ties, finite extremes,
subnormals, signed zero, malformed frames, default f32, masks, codec conflicts,
MTP rejection and SIMD/portable equivalence are covered.
The independent actual-input oracle passes **41 sites / 13,120 work vectors**:
complete wire bytes and expanded structs are identical between optimized and
portable codecs. All trained common-prefix component checks observed zero
relative error against a 1e-4 tolerance; this is not a requirement for
bit-exact model quality.

The unrelated pre-existing full fabric selfcheck reserved-byte failure remains
disclosed in `BF16_HOPS.md`; a passing full suite is not claimed.

| Weight-free archive | SHA256 |
|---|---|
| `fp16-evidence-50.tar.gz` | `4b9654cfbb32d0454f33fb0eb097bb109b3f316da45534c55ff64e3c52c6bf13` |
| `fp16-evidence-52.tar.gz` | `2cae9a7432b5b74629687f4e29477ebdd936e399086f77a24f7c45dcbd5bf33c` |
| `fp16-evidence-53.tar.gz` | `835634ef98631f33390cf217583c96a1e0b6dd4b69dd91608b091ad5516e954d` |
| `fp16-q8-evidence-50.tar.gz` | `6d2761f174a2a6855c5ebf418dc31b554943780396c8a9275f02411eaaba32ea` |
| `fp16-receipts.tar.gz` | `98bd3638f00e68eb507c16d919cc18c7a6384b65265bd0afb6bd935def95eb8e` |

The bounded archives retain source snapshots, compiler/binary identities,
stage/head observations, original-Q8 restoration/readback proofs, paired raw
legs including the initial scalar regression, validation and cleanup receipts.
`fp16-hops.json` records direct comparisons and all causal policies.
No `.fgw`, `.gguf` or full-vocabulary `.fgl` payload is committed.

Large private artifacts remain ignored locally:

| Artifact | SHA256 |
|---|---|
| Fifteen-policy full final logits, `fp16-logits.tar.gz` | `97cd06feb1d7f629a9819e42bb14bb0b0542ac181b21ce101df9935d6ad0fb97` |
| Same-geometry Q8 final logits/proofs, `fp16-q8-control.tar.gz` | `d4ca5b1351204674082c580e3623e90bb97fb518cd0c577efbf81b3b67fd48c3` |
| Authentic prepared80 seed/NG/token frames, `fp16-prepared80.tar.gz` | `84c00abdcd0a97160c75b27c309f5d5c0aed5cc6524652586256012c579e8d5c` |

Prepared80 contains 74,055,680 seed-embedding bytes and 18,513,920 n-gram
bytes, plus `cases.bin`, prompt frames, sealed manifests and internal hashes.
The initial `.teacher` and `.candidate` frames were verified identical; only
`.teacher` is stored, so copy it for the candidate alias when replaying.
These bounded inputs avoid re-fetching the full 28.8-GB n-gram source.
Shared local mixed-pack FGWs, overlay and metadata remain untouched.

After extracting the private logits/reference and bounded evidence to their
corresponding `.build` directories, reproduce the analysis with:

```powershell
python tools\analyze_fp16_hops.py --root .build\fp16-quality `
  --reference ..\q4-dense\results\hillclimb\q4-qualification\r3-teacher `
  --previous .build\quality-50\quality.jsonl `
  --q8-control .build\fp16-q8-control --pair .build\fp16-hop-pair.json `
  --output results\hillclimb\fp16-hops.json
```

The previous teacher rows come from the completed BF16 experiment. Their
baseline hashes are checked, not used as an additive KL estimate.
