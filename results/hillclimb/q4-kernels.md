# Exact cooked Q4 dense kernels

## Verdict and scope

**Ordinary Q4 kernel latency gate: PASS on 192.168.42.51. Deployment: NOT
QUALIFIED.** All nine non-head GEMV shapes beat their own cooked-Q8 baseline
in all three runs (27/27 comparisons). All 63 ordinary prefill comparisons
at N=8,16,32,64,128,256,512 also win. All scalar parity checks pass.
The Q6 vocabulary head loses: retain the original Q8 head using
`q4-repack --keep-q8-head`. Protected Q5 is supported exactly, but does not
universally win; do not silently substitute ordinary Q4 where quality needs Q5.

Primary blade: `.51`, AMD BC-250 / RADV GFX1013, subgroup 64, user-described
40-CU configuration. CU count was not independently read back. Earlier `.52`
experiments used its user-described 24-CU configuration; final tuning was not
rerun there. The historical physics report's CU count is not a current readback.
No hardware masks, clocks, governors, services, production blades `.42`–`.50`,
or `.53` verified artifacts were changed/contacted.

The complete measurement records, unabridged benchmark output, source hashes,
validation output, and earlier `.52` results are in
[`q4-kernels.json`](q4-kernels.json).

## Method

- Existing Vulkan harness and **GPU timestamps**, 20 timed invocations of each
  candidate and Q8 per shape. Both execute in one profiled command batch,
  alternating and reversing order each pair. Profile counts must equal 20/20.
- Three warm dispatches of each variant; the first 2560x10240 shape additionally
  runs 2048 single-token candidate warm dispatches. Clocks were not locked.
  Three full GEMV passes are retained, including the slower first pass.
- Candidate parity is against an independent scalar source-format decoder,
  not against Q8 output. Q8 uses independently initialized same-shaped weights.
  Small tests check every row/token, including padded 17/19-row matrices.
  Large benchmarks check 32 evenly distributed rows for **every** token.
- Output tolerance is `2e-5 * max(1, sum(abs(products)))`; Q5 CPU packed decode
  and GPU one-hot tests additionally check exact dequantized values.
- GB/s is decimal **stored-weight payload / GPU seconds**, not a memory
  controller measurement. Cooked Q4 bytes include expanded metadata. Source
  GB/s uses the original compressed payload. Prefill payload GB/s counts a
  matrix once, not all tile rereads; it is not actual DRAM traffic.

## GEMV results

Representative **third run**, not best-of-three. Times are microseconds;
speedup is Q8 time / candidate time.

| K x M | Candidate | Candidate us | Q8 us | Speedup | Cooked GB/s | Source GB/s | Q8 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 2560 x 10240 | Q4_K | 59.108 | 81.012 | 1.371x | 332.6 | 249.5 | 343.8 |
| 2560 x 6144 | Q4_K | 43.660 | 57.524 | 1.318x | 270.2 | 202.6 | 290.5 |
| 6144 x 2560 | Q4_K | 52.870 | 75.832 | 1.434x | 223.1 | 167.3 | 220.4 |
| 2560 x 12288 | Q4_K | 77.078 | 99.090 | 1.286x | 306.1 | 229.6 | 337.3 |
| 10240 x 320 | Q4_K | 12.892 | 28.140 | 2.183x | 190.6 | 143.0 | 123.7 |
| 320 x 10240 | IQ4_NL | 10.332 | 12.730 | 1.232x | 178.4 | 178.4 | 273.5 |
| 640 x 2560 | IQ4_NL | 6.328 | 7.436 | 1.175x | 145.6 | 145.6 | 234.1 |
| 2560 x 640 | Q4_K | 5.972 | 7.376 | 1.235x | 205.8 | 154.3 | 236.0 |
| 2560 x 512 | Q4_K | 5.608 | 6.780 | 1.209x | 175.3 | 131.5 | 205.4 |
| 2560 x 248320 | Q6_K, raw | 4955.856 | 2129.136 | **0.430x** | 105.2 | 105.2 | 317.2 |

All-three-run variation:

| K x M | Candidate us range | Paired speedup range |
|---|---:|---:|
| 2560 x 10240 | 58.880–64.006 | 1.330–1.379x |
| 2560 x 6144 | 43.660–48.386 | 1.227–1.318x |
| 6144 x 2560 | 52.870–58.286 | 1.352–1.434x |
| 2560 x 12288 | 77.078–83.008 | 1.255–1.286x |
| 10240 x 320 | 12.892–14.634 | 2.160–2.634x |
| 320 x 10240 | 10.332–15.794 | 1.017–1.232x |
| 640 x 2560 | 6.328–9.268 | 1.018–1.175x |
| 2560 x 640 | 5.972–7.816 | 1.194–1.235x |
| 2560 x 512 | 5.608–7.614 | 1.192–1.216x |

The narrow first-run IQ4 margins are not hidden: they are only 1.7–1.8%.
Two large shapes reach the >=300 GB/s goal **using cooked bytes**, but the
others do not, and **none reaches 300 GB/s using original compressed bytes**.
The largest Q4 stream reaches 333.9 GB/s cooked in run two, about 74% of the
historical 450.7 GB/s read reference; its source-equivalent rate is 250.4 GB/s.
This is a latency win with a larger-than-canonical Q4 layout, not a claim of
300+ GB/s canonical Q4 traffic or an achieved serving-byte ledger.

## Batched results

All entries are paired Q8/candidate speedups; exact candidate/Q8 microseconds
and both bandwidths for every entry are in JSON. These cover the same ordinary
Q4_K/IQ4_NL plan as the GEMV table, not the protected Q5 variant.

| K x M | N=8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2560 x 10240 | 1.804 | 1.491 | 1.169 | 1.274 | 1.182 | 1.414 | 1.449 |
| 2560 x 6144 | 1.752 | 1.474 | 1.088 | 1.216 | 1.159 | 1.330 | 1.398 |
| 6144 x 2560 | 1.535 | 1.385 | 1.172 | 1.176 | 1.205 | 1.452 | 1.406 |
| 2560 x 12288 | 1.810 | 1.486 | 1.149 | 1.188 | 1.202 | 1.374 | 1.452 |
| 10240 x 320 | 1.436 | 1.222 | 1.015 | 1.028 | 1.104 | 1.215 | 1.050 |
| 320 x 10240 | 1.684 | 1.481 | 1.024 | 1.053 | 1.212 | 1.263 | 1.305 |
| 640 x 2560 | 1.669 | 1.425 | 1.036 | 1.047 | 1.300 | 1.355 | 1.385 |
| 2560 x 640 | 1.671 | 1.431 | 1.169 | 1.179 | 1.152 | 1.946 | 2.249 |
| 2560 x 512 | 1.669 | 1.432 | 1.170 | 1.166 | 1.167 | 1.927 | 1.930 |

The smallest margin is 1.5%; these are not confidence intervals or fixed-clock
results. The LDS implementation reuses **FP32** dequantized weights per tile.
Packed-FP16 math/storage was not implemented: narrowing arbitrary source
dequantized weights to half would not preserve the required exact values.
The retained-Q8 head uses the unchanged baseline path; head prefill was not
separately benchmarked.

## Why this is faster

The existing cooked-Q8 implementation is the useful model: metadata separated
from a regular quant stream, cohorts processing adjacent activation vectors,
multiple row accumulators, wide metadata loads, and subgroup reductions.
The slow prior K-quant shader repeatedly reconstructs packed GGML scale/min
fields, mixes metadata with quant loads, and uses a less favorable row mapping.

The implemented layout has eight-row tiles. Let `G=K/32`; rows are zero-padded
to a multiple of eight.

| Format | Per tile metadata | Row-local quant stream | Bytes/32 weights |
|---|---|---|---:|
| Q4_K | G groups of eight FP32 scales then eight FP32 minima | 16 bytes/group; low/high nibble halves | 24 |
| Q5_K | same exact FP32 metadata | 20 bytes/group; four contiguous five-byte/eight-quant lane packets | 28 |
| IQ4_NL | G groups of eight original half-scale bit patterns | 16 bytes/group; low/high nibbles | 18 |

Metadata precedes the tile's `[row][group]` quant stream. Q4 therefore stores
6 effective bits/value, versus the source's 4.5; Q5 stores 7 versus 5.5.
IQ4 has no expansion. This deliberately spends metadata bytes to remove
unpack dependencies and permit vector scale/min loads. Quant loads are regular
32-bit words across four lanes rather than a promise that every quant fetch
is a 128-bit transaction.

GEMV uses four-lane cohorts, vector activation loads and FP32 dot/FMA arithmetic,
four/eight row accumulators, subgroup sums and a short cross-subgroup reduction:

- Q4 large matrices: eight rows/WG64.
- 2560x6144: eight rows/WG128.
- Outputs <=1024: four rows/WG64; long-K Q4 uses WG256.
- IQ4: four rows/WG64 and a shared exact 16-entry nonlinear lookup table.
- Q5 uses the corresponding four/eight-row paths and exact five-bit extraction.

Prefill uses WG256, 64 output rows, K tiles of 32, and token tiles of 8, 16 or
32. Each tile dequantizes weights once into LDS and shares staged activation
vectors. The key compiler fix was hoisting the live-row guard around all eight
decoded weights **before** the LDS stores, and vectorizing input staging under
one live-token guard. The earlier form lowered to separate predicated
descriptor/dequant blocks interleaved with stores and was much slower.

Rejected approaches included scalar SoA loads, merely widening row-local quant
loads, repeated arithmetic IQ4 lookup, and separate scattered Q5 high bits.
They did not recover Q8-like instruction efficiency. Suppressing the K-loop
unroll also regressed small shapes. The final RADV ISA capture is retained in
[`q4-kernels-isa.txt.gz`](q4-kernels-isa.txt.gz) (23,472 lines); no native integer
dot or scratch load/store instruction occurs. The ISA capture is a correctness
run, not timed evidence.

## Exactness and integration limits

For finite source scales, a half has at most 11 significant bits. Multiplying
by the six-bit group scale needs at most 17; multiplying additionally by a
five-bit Q5 quant needs at most 22, within FP32's 24-bit significand. Minimum
expansion is similarly exact. Pre-expanding these products therefore does not
round a source dequantization scale/minimum. IQ4 retains the original half bits
and exact integer grid. No FP16 scale narrowing or weight-value narrowing is
introduced. Reduction/FMA reassociation can change the accumulated output,
which is why output parity uses an explicit tolerance rather than bit identity.

`FG_DENSE_SOA=1` opts common Q4_K/Q5_K/IQ4_NL projections into the new path.
Embedding and output remain excluded; default Q8 behavior is unchanged.
Crucially this is **load-time additional allocation**, not an on-disk cooked
pack: original verified arenas remain resident. The extra buffers are counted
by `fg_model_weight_bytes`, and the persistent cap is checked before allocation.
Peak/resident memory and whole-model speed need separate qualification; do not
apply the canonical-compressed DESIGN.md byte ledger to these buffers.

Earlier protected-plan `.52` measurements still lost at Q5 2560x12288
(121.352 vs 110.880 us Q8), and Q5 2560x512 (10.114 vs 9.428 us).
Those are earlier sequential timing results, not the final alternating protocol,
and are retained as evidence against claiming a universal Q5 win.
There is no cooked-Q6 implementation here: the tested raw Q6 head failed,
so the recommended configuration keeps the original Q8 head, not a requantized
Q8 approximation of the Q6 values.

## Validation

Final `.51` validation, exit 0:

- Built `flash-gordon`, `tools/q4-repack`, and targeted native test binaries.
- Core PASS. Its intentional negative verification case prints
  `VERIFICATION FAILED: 5 pass, 1 fail` before the suite's actual `core tests: PASS`.
- Model-loader PASS, including Q4_K/Q5_K/IQ4_NL SoA replacement bytes/formats,
  original embedding/head, extra-byte accounting, and cap refusal.
- Vulkan PASS: `dense_quant`, `dense_quant_cooked`, `dense_quant_soa`,
  `q8_dense`, `q8_dense_cooked`, `q8_cooked_prefill_parity`,
  `q8_decode_shape_parity`, `q8_split_silu_parity`, `router_bf16_exact`.
- Repacker accepts `--keep-q8-head` and refuses a missing source manifest
  before producing an output; a complete real-pack repack was **not** tested.
- Local Python: eight tests, seven passed, native-library case skipped because
  the library was not configured. `git diff --check` passed.

Legacy native fixture paths are relocated into the experiment working directory
by the existing `q4-safe-tests.sh` **before building and running** validation.
No full-suite pass is claimed: previously documented unrelated build failures
and baseline `qsa_record_commit` crash were not fixed by this kernel work.

## Reproduce

From this Windows worktree, archive the sources and safe validation helper:

```powershell
tar -cf .\kernels.tar .\include .\src .\tests .\shaders .\vendor `
  .\Makefile .\glslc .\tools\q4_repack.c .\tools\q4-safe-tests.sh `
  .\tools\q4-kernel-validate.sh
& .\tools\Invoke-Q4Fleet.ps1 -ScriptPath .\tools\q4-kernel-run.sh `
  -Targets @('192.168.42.51') -UploadPath .\kernels.tar `
  -UploadDestination /home/xander/hc-q4/kernels.tar -PassThru
& .\tools\Invoke-Q4Fleet.ps1 -ScriptPath .\tools\q4-kernel-validate.sh `
  -Targets @('192.168.42.51') -PassThru
```

The wrapper invokes the authorized fleet runner with staging inside `hc-q4`;
the shared runner is unchanged. Scripts build/run only in that workspace.
The benchmark script selects the ordinary Q4 plan, runs all seven batches,
repeats GEMV twice more, and measures the losing head separately.
`q4-kernel-isa.sh` optionally captures RADV shader dumps there.
After preserving evidence, this session's `kernels` subdirectory and uploaded
archive were removed from `.51`/`.52`; earlier code/overlay artifacts were
left untouched. Reproduction uploads a fresh archive.

**Deployment blockers remain:** real abliterated-pack identity/source proof,
real-pack repack and memory budgets, complete ring startup, full-vocabulary
KLD/top-1 and exact-router gates, existing reference/long-context gates, and
end-to-end serving measurements. No production pack or serving API was changed.
