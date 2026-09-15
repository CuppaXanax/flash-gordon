# Dense Kernel Occupancy and Trip-Count Experiments (2026-09-15)

Worktree `D:\workspace\fg-work-lat11`, branch `perf/dense-kernel-latency`
rebased onto main `5fbeb29` (post-privacy-scrub, includes the qualified 4-way
output split).  Measured on the 8-blade ring 2026-09-15 with fleet binary
`b47c12ba4d4b85ac49ad80081b6c67d6716afa2bcec8a425e61f8a2e8edb2380`; the fleet
was left on the default config afterwards.  This is the follow-up to
`PERFORMANCE_DENSE_KERNEL_LATENCY_2026-09-14.md`; it works the two levers that
doc parked.

| commit | change |
|---|---|
| `cf684ce` | gated wave-split r8 variant (`FG_DENSE_R8_WAVE_SPLIT`) |
| `035fa47` | gated 32-block pair r8 variant (`FG_DENSE_R8_PAIR`) |
| docs update | measured A/B, VGPR ground truth, decision (this file) |

Both variants are new SPIR-V files selected by an env flag in the existing
`dense_cooked` promotion path (`src/vk.c`); default is the qualified r8 kernel,
so the branch's default behaviour and the shipped pack are unchanged.

## 0. TL;DR

The r8 dense family is one instruction stream at three different speeds: 298 GB/s
on GDN qkv, 170 on the GDN output, 148 on GR up, 92-277 on the QSA shapes.  The
round-11 fleet geometry sweep already showed that row tile and K-split changes
do not fix it.  The two remaining structural hypotheses are occupancy (the
8-row kernel holds two vec4 accumulator sets; a 4-row wave holds one) and the
trip count / contiguity of the weight stream (the output shape walks 192
blocks in 12 trips of 512 B; the qkv shape walks 80 in 5).  Two gated kernels
test them:

* **`_ws`**: 128-lane workgroup unchanged, each 64-lane wave owns a 4-row half,
  one vec4 accumulator, 7 loads per trip, no cross-wave barrier.  LLVM gfx1013
  proxy 86 -> 53 VGPRs, i.e. 2 -> 4 waves/SIMD.  Cost: input L2 reads double.
* **`_pair`**: 8-row layout unchanged, each lane owns two adjacent words
  (uvec2) of one row-block, so a trip covers 32 blocks and a wave load reads
  1 KiB contiguous.  Output shape: trips 12 -> 6, per-lane loads 120 -> 66,
  loop-body ops 1476 -> 1098 (-26%).  Cost: proxy 116 VGPRs, gated to
  `blocks > 16` so the 320-wide GR up shape stays on r8.

Both variants were measured on the fleet and **both are rejected**: the wave
split costs 1.4 short / 0.8 4K / 0.8 32K TPS and the pair costs 3.9 / 3.4 / 2.5
TPS against the same-session control.  The VGPR ground truth explains it: ACO
already allocates the 8-row r8 kernel only **36 VGPRs**, the lowest of the
three (`_ws` 40, `_pair` 48), so the occupancy lever the round-11 doc
hypothesised does not exist and both variants pay register cost for nothing.
The measured ordering (24.5 > 23.2 > 20.7 TPS short) matches the
register-limited waves per SIMD (7 > 6 > 5).  The fleet is left on the default
r8 config.  The last untested lever remains the **row-interleaved cooked
layout** (section 4); it needs a pack-format version and a pack rebuild, which
is a fleet round, and its payoff is not proven by the available fleet data.

## 1. Where the two levers came from

### 1.1 Fleet per-shape rates (r8 kernel, round-11 geometry sweep)

| shape | in -> out | row stride | blocks | wgs | trips | GB/s (weights) |
|---|---|---:|---:|---:|---:|---:|
| GDN qkv | 2560 -> 10240 | 2560 B | 80 | 1280 | 5 | 286 |
| QSA q/g | 2560 -> 12288 | 2560 B | 80 | 1536 | 5 | 277 |
| PLE value | 2560 -> 2560 | 2560 B | 80 | 320 | 5 | 222 |
| GDN output | 6144 -> 2560 | 6144 B | 192 | 320 | 12 | 170 |
| shared down | 640 -> 2560 | 640 B | 20 | 320 | 2 | 168 |
| GR up | 320 -> 10240 | 320 B | 10 | 1280 | 1 | 148 |
| shared gate | 2560 -> 640 | 2560 B | 80 | 80 | 5 | 112 |
| QSA kv | 2560 -> 512 | 2560 B | 80 | 64 | 5 | 92 |

Rows/splits variants were measured in the same sweep: rows=4 loses 9-15% on
the large shapes and every split > 1 loses 2-3x on the large shapes.  The table
rows above are therefore the best known geometry per shape, and the deficit is
in the work assignment inside one (rows=8, splits=1) shape.

### 1.2 Register/occupancy proxy (LLVM AMDGPU `gfx1013`, not ACO)

| geometry | VGPRs | waves/SIMD | wave64/CU |
|---|---:|---:|---:|
| r8, 8 rows/lane | 86 | 2 | 8 |
| 4-row wave (`_ws`) | 53 | 4 | 16 |
| 8 rows/lane, uvec2 (`_pair`) | 116 | 2 | 8 |

ACO's allocator differs (it rematerialises and splits live ranges more
aggressively), so these are bounds, not the blade numbers.  The fleet step in
section 5 reads the real numbers.  The important ratio: `_ws` carries half the
per-lane state, `_pair` carries roughly 1.35x the per-lane state of r8.

## 2. Variant `_ws`: wave-split (occupancy)

`shaders/fg_dense_q8_0_cooked_r8_ws.comp`, selected by
`FG_DENSE_R8_WAVE_SPLIT=1`.

* 128-lane workgroup, 8 rows per group as before; `row_half = local_id >> 6`
  owns four rows, `lane = local_id & 63`.
* Per 16-block trip each lane does 4 uvec2 weight loads (4 rows x 8 weights),
  2 input vec4, 1 scale uvec4: 7 loads / 32 weights, one vec4 accumulator.
* No cross-wave LDS sum and no barrier: each wave reduces its own four rows
  with four `subgroupAdd` ops and the workgroup elect writes both halves.
  A `gl_SubgroupSize > 64` branch falls back to a per-lane LDS tree so the
  64-lane half assumption is guarded (the only failing configuration is a
  subgroup wider than 64, which no Vulkan target implements).
* Input reads: each lane needs the eight input floats behind its two-word
  chunk for its four rows, so the input is read by both waves; input L2 traffic
  doubles (output call 7.7 -> 15.4 MB, still far below the 16.7 MB weight
  stream).  Weight loads are unchanged per byte and stay fully coalesced
  (16 blocks x 32 B contiguous per wave instruction).

SPIR-V loop body 117 result ops (r8 123) for the same 16 blocks; the win, if
any, is the register budget and the removed barrier, not the op count.

## 3. Variant `_pair`: 32 blocks per trip, 1 KiB contiguity

`shaders/fg_dense_q8_0_cooked_r8_pair.comp`, selected by
`FG_DENSE_R8_PAIR=1` and only for `blocks > 16`.

* 8 rows per lane as r8, but each lane owns two adjacent words of one
  row-block (`word_pair = local_id >> 5`, `block_in_group = local_id & 31`),
  loaded as one `uvec2` (8 B) per row per trip.
* A trip covers 32 blocks; every weight wave instruction reads 32 blocks x
  32 B = 1 KiB contiguous (r8 reads 512 B).
* Per lane per trip: 8 uvec2 weight loads (64 weights), 2 input vec4 (the two
  vec4 behind the lane's word pair, shared by all eight rows), 1 scale uvec4.
* Epilogue is the r8 epilogue (8 subgroupAdds plus the LDS combine), so no new
  wave assumption is introduced; the shared array is sized 256 vec4 so the
  `gl_NumSubgroups` combine is in bounds for every subgroup size.

Accounting, output shape 6144 -> 2560 (192 blocks):

| quantity | r8 | pair | delta |
|---|---:|---:|---:|
| trips per workgroup | 12 | 6 | -50% |
| per-lane weight-load instructions | 96 | 48 | -50% |
| per-lane load instructions (weight + input + scale) | 120 | 66 | -45% |
| loop-body result ops x trips | 1476 | 1098 | -26% |
| wave weight-load contiguity | 512 B | 1024 B | 2x |
| proxy VGPRs | 86 | 116 | +35% |

For qkv (80 blocks): trips 5 -> 3 and per-lane loads 50 -> 33.  For GR up
(10 blocks) the kernel is not selected: at 32-block granularity only 31% of
the lanes carry useful work, 183 ops against r8's 123 for the same bytes, so
`blocks > 16` keeps that shape on r8 (shared down, 20 blocks, does select the
pair kernel: 2 trips -> 1, 20/32 lane utilisation, strictly fewer ops).

## 4. The remaining lever: row-interleaved cooked layout

The two variants change the work assignment, not the memory layout.  The
layout is the last difference between the 2560-stride shapes (286 GB/s) and the
6144-stride output shape (170 GB/s): within a 16-row tile the quant region is
row-major, so the eight rows one workgroup walks sit `blocks*32` bytes apart
(6 KiB for the output, 2.5 KiB for qkv) and a trip touches ~50 pages.

Design (tile size unchanged, scale plane already block-major):

```
today:  [scales 16*blocks*2 B][row0 blocks*32 B][row1 ...][... row15]
v2:     [scales 16*blocks*2 B][block0 16*32 B][block1 ...][... block_{n-1}]
```

Addressing change: quant(row r, block b) = tile_base + quant_offset +
`b*512 + r*32` instead of `b*32 + r*blocks*32`.  At block granularity the
eight rows of a workgroup trip become 256 B contiguous (two 128 B lines)
instead of eight 32 B chunks 6 KiB apart, so the pages touched per trip drop
from ~50 to ~2-3, with identical byte count, identical tile size and identical
row/tile slicing in `output.c`.

Touchpoints and why this is a fleet round, not a local one:

* `src/quant.c` (`fg_cook_q8_0_rows` placement), `src/pack.c` (calls it) and
  the seven cooked readers (`fg_dense_q8_0_cooked`, `_r8`, `_r8_ws`, `_r8_pair`,
  `_split`, `_tile`, `fg_moe_decode_down_reduce`).
* The pack bytes change, so the qualified pack must be regenerated and the
  local tests that compare cooked against raw must run on the new pack; the
  manifest needs a version/fingerprint decision so an old pack cannot be
  silently consumed (the fleet gates would catch it, but a version is safer).
* Payoff is not proven: QSA kv (2560-stride, 64 workgroups, 92 GB/s) and shared
  gate (2560-stride, 80 workgroups, 112 GB/s) are grid-starved, not
  stride-starved, and shared down (640 B stride) is ramp-starved at 168 GB/s.
  The one shape the layout is aimed at, the GDN output, is also the only large
  shape below 200 GB/s.

Promotion bar for a future pack round: all local parity oracles on the v2
pack, gates green on the fleet, and the GDN output call <= 80 us (>= 220 GB/s)
without regressing qkv/z/GR.  If the layout is tried and the output does not
move, the remaining explanation is the grid ramp of a 320-workgroup dispatch,
which no layout change can fix.

## 5. Fleet occupancy ground truth (measured)

The round-11 doc asked for the blade VGPR counts.  Method: start rank 0 with
`RADV_DEBUG=shaderstats`; the driver prints one stats block per shader it
compiles.  With a warm shader cache a variant run compiles only the one new
kernel (ws 40, pair 48), and the r8 count was pinned by two `nocache` runs with
identical dispatch order whose VGPR lists differ in exactly one position: 36 in
the control run, 40 in the wave-split run.  ACO's numbers are 2.4x lower
than the LLVM AMDGPU proxy of section 1.2, which is why the proxy's 86
misled the round-11 write-up.

| kernel | VGPRs (ACO) | register-limited waves/SIMD |
|---|---:|---:|
| `fg_dense_q8_0_cooked_r8` | **36** | 7 |
| `fg_dense_q8_0_cooked_r8_ws` | 40 | 6 |
| `fg_dense_q8_0_cooked_r8_pair` | 48 | 5 |

Decision rule application:

* r8 = 36 <= 64 -> the occupancy lever does not exist; do not promote `_ws`.
  The measured ordering confirms it: short decode 24.6 (r8) > 23.2 (ws) >
  20.7 (pair) TPS, exactly the register-limited wave order 7 > 6 > 5.
* `_pair` is rejected on its A/B (section 7): it has fewer instructions and the
  same input traffic per weight as r8, but the wider in-flight window costs 30%
  of the register-limited occupancy and that dominates.

The occupancy hypothesis from round 11 is closed for the dense family: the
existing kernel is already the most occupancy-efficient of the three shapes
tested, and its 170-286 GB/s spread across shapes is not a register problem.

## 6. Local validation (llvmpipe, correctness only)

```
make all -j8                                       warning-free (-Werror)
tests/test_qsa_prefill, tests/test_expert_prefill, tests/test_owner_reduce  PASS
focused tests/test_fg_vk, flags off and each flag on:
  q8_decode_shape_parity, q8_dense_cooked, q8_cooked_prefill_parity,
  gdn_project_cooked, gdn_decode, gdn_algebraic, expert_graph_fused_q8_0_cooked,
  expert_graph_replay, gr_mix, gr_batch, hc_finalize, ple_decode,
  ple_prefill_scan, q8_cooked_view_slice, q8_dense_subgroup, group_norm,
  hc_inject_partial, gr_partial_boundaries                       PASS
FG_BENCH_Q8_COOKED 11-shape parity, flags off and on               PASS
```

Numerics: both variants re-associate the per-lane partial sums, so they are not
bit-identical to r8.  The `_ws` kernel sampled against the raw reference at a
1e-9 test tolerance shows element differences at the 1e-8 relative level (GR up
row 1: 18.5010509 vs 18.5010548); `_pair` passes the same oracle at its 2e-4
threshold.  No tolerance was changed.

llvmpipe A/B over two alternations (median `cooked_gpu_us`, indicative only;
llvmpipe is instruction-bound and not representative of GFX1013, and the
`output_vocab` shape dominates the total):

| shape | r8 | ws | pair | ws/r8 | pair/r8 |
|---|---:|---:|---:|---:|---:|
| GR up | 1792 | 1638 | 1689 | 0.91 | 0.94 |
| GDN qkv | 2764 | 2692 | 2577 | 0.97 | 0.93 |
| qsa kv | 1062 | 1041 | 1019 | 0.98 | 0.96 |
| PLE value | 1373 | 1381 | 1371 | 1.01 | 1.00 |
| GDN output | 1752 | 1751 | 1720 | 1.00 | 0.98 |
| QSA q/g | 2946 | 3061 | 3012 | 1.04 | 1.02 |
| output vocab | 34536 | 37550 | 40474 | 1.09 | 1.17 |

The mixed result (wins on the mid-size shapes, regressions on the 31040-workgroup
vocab shape) is the expected behaviour of an instruction-bound CPU rasteriser
and is not used to promote or reject anything; it only shows both kernels are
live and self-consistent.

## 7. Fleet A/B results (measured 2026-09-15, binary `b47c12ba`)

Same ring pack, same build; variants differ only by the env flag on all eight
ranks.  Every measurement is the standard attach battery (one 128-token sanity,
two 4k prefills with one decode each, one 32-token decode) plus a 32k context
sweep.  Run-to-run spread on this fleet is +/-10-15% thermal, so each variant
was run twice and compared against controls taken at both ends of the session:

| config | short decode | 4k warm decode | 32k decode | 4k prefill (r1 / r2) |
|---|---:|---:|---:|---:|
| control (r8), start of session | 24.61 | 23.54 | - | 274.5 / 282.9 |
| control (r8), end of session | 24.21 | 23.44 | 19.36 | 277.3 / 266.4 |
| wave split `FG_DENSE_R8_WAVE_SPLIT=1`, run 1 | 23.18 | 22.81 | - | 276.6 / 280.1 |
| wave split, run 2 | 23.21 | 22.68 | 18.60 | 268.7 / 281.2 |
| pair `FG_DENSE_R8_PAIR=1`, run 1 | 20.60 | 20.13 | - | 275.2 / 277.2 |
| pair, run 2 | 20.76 | 20.09 | 16.89 | 274.9 / 282.7 |

Deltas against the two controls that bracket the variants: wave split
-1.4 short / -0.75 4k / -0.76 32k TPS; pair -3.9 / -3.4 / -2.5 TPS.  Prefill
stays in the 266-283 band for every config, so the launch path is unaffected
and the regressions are in the decode kernels.  The ordering is exactly the
register-limited occupancy order of section 5, and both variants are
reproducible across their two runs, so thermal drift (which moves the control
by only ~0.3 TPS between the bracketing runs) does not explain them.

### 7.1 Kernel scopes (rank 0, `FG_DECODE_PROFILE=1`, one 6-layer block)

r8 control, captured the same session as the batteries:

| scope | calls | ms/block |
|---|---:|---:|
| gr_attn_read: rms 0.087 / split 0.115 / reduce 0.021 / inject 0.042 / silu 0.007 / **up r8 0.152** / mix 0.047 | 7 | 0.471 |
| gr_ffn_read: rms 0.086 / split 0.114 / reduce 0.021 / inject 0.040 / silu 0.007 / **up r8 0.149** / mix 0.045 | 7 | 0.462 |
| gdn_projection: **qkv r8 0.400** / z generic 0.233 / controls | 4+4+8 | 0.634 |
| gdn_output: **r8 0.382** | 4 | 0.382 |
| shared_expert: **r8 0.164** / swiglu 0.008 / scalar | 18+12 | 0.173 |
| qsa_projection: **r8 0.262** / prepare+bf16+quantize+index | 6+14 | 0.472 |
| qsa_output: **r8 0.193** | 2 | 0.193 |
| gdn_recurrent: conv 0.012 / algebraic 0.148 | 4+4 | 0.160 |

Wave split, captured during its 32k sweep (the same-session control for that
run was not profiled, and the run's non-dense kernels - expert pair 0.851 vs
1.385 ms - show a different machine state, so these are raw data, not a clean
per-kernel delta): up ws 0.123 and 0.121, qkv ws 0.437, output ws 0.396,
shared_expert ws 0.143, qsa_projection ws 0.252, qsa_output ws 0.192,
algebraic 0.125.

Final state: both gates `[12]`/`[Paris]`, pi-stability PASS (6 stages, 4-turn
conversation, 8 ranks, 0 failures, 4k prefill 275.34, short decode 22.37), and
all eight blades left running one process of the default config (no dense
flags, no `RADV_DEBUG`, no profile env) on `b47c12ba`.

## 8. Ranked next steps

1. **Row-interleaved cooked layout** (section 4) is now the top structural
   item: occupancy (36 VGPRs, section 5), row tile and K-split (round-11 fleet
   sweep) and the trip/contiguity restructure (measured here) are all refuted,
   so the 170 GB/s GDN output call can only be the weight layout or the
   320-workgroup ramp.  It is a pack round: block-major quant placement, new
   format version, all parity oracles on the rebuilt pack, and the promotion
   bar in section 4.
2. **If the layout round is declined**, the output-shape deficit should be
   documented as a dispatch-ramp property: 320 workgroups is 13.3 per CU and
   the same kernel reaches 286 GB/s at 1280 workgroups with the same bytes per
   row, so the remaining gap needs either dispatch fusion or a persistent
   kernel, not a shader edit.
3. **Small-grid shapes** (QSA kv 64 workgroups at 92 GB/s, shared gate 80 at
   112 GB/s): unchanged from round 11; dispatch fusion is the only fix and it
   is a vk.c/caller change.
4. **Keep the default**: the two gated kernels stay in the branch as measured
   and rejected experiments (`FG_DENSE_R8_WAVE_SPLIT` / `FG_DENSE_R8_PAIR`
   must not be enabled on the fleet); the default path is byte-identical to
   the qualified build.
