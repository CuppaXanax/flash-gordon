# Dense Kernel Occupancy and Trip-Count Experiments (2026-09-15)

Worktree `D:\workspace\fg-work-lat11`, branch `perf/dense-kernel-latency`
rebased onto main `7a3151b` (round-11 qualified binary `cbcee93a`, short decode
24.28/24.68, 4K warm 23.41-23.74).  This is the follow-up to
`PERFORMANCE_DENSE_KERNEL_LATENCY_2026-09-14.md`; it works the two levers that
doc parked.

| commit | change |
|---|---|
| `89fb528` | gated wave-split r8 variant (`FG_DENSE_R8_WAVE_SPLIT`) |
| `e862917` | gated 32-block pair r8 variant (`FG_DENSE_R8_PAIR`) |
| this doc | accounting, occupancy gate, layout design, A/B plan |

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

Neither is promoted by this round: the occupancy ground truth (RADV shader
stats on the blade) is still the gate for `_ws`, and `_pair` trades register
budget for trips.  Both are one-env-restart A/Bs with a documented revert.
The last untested lever is the **row-interleaved cooked layout** (section 4);
it needs a pack-format version and a pack rebuild, which is a fleet round, and
its payoff is not proven by the available fleet data.

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

## 5. Fleet occupancy ground truth (required step before promoting `_ws`)

The round-11 doc asked for the blade VGPR counts; they are still not measured.
On a decode run with the standard profile environment, a RADV shader-stats
trace (`RADV_DEBUG=shaderstats`) reports the per-shader VGPR/SGPR allocation
for the compiled pipeline.  Record it for `fg_dense_q8_0_cooked_r8.spv` (and
the two variants when their flags are on) at the first token.

Decision rule:

* r8 <= 64 VGPRs: occupancy is not the limiter (4+ waves/SIMD), do not promote
  `_ws`; keep the round's value in the geometry/trip accounting and move to the
  layout item.
* r8 > 64 VGPRs: check `_ws` on the same trace; promote only if its count
  drops below r8's threshold and the A/B is positive.
* `_pair` has no gate; it is a direct A/B (its proxy registers are higher, so
  the fleet either rewards the trip/contiguity cut or it does not).

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

## 7. Fleet A/B plan

Both variants are one-env restarts on the same pack and binary; the default
path is byte-identical to the qualified binary, so the control is the round-11
result.  Run the standard attach battery, gates and soak for each restart:

| restart | env on all ranks |
|---|---|
| control | none (round-11 `cbcee93a` numbers) |
| wave split | `FG_DENSE_R8_WAVE_SPLIT=1` |
| pair | `FG_DENSE_R8_PAIR=1` |
| occupancy trace | none, plus `RADV_DEBUG=shaderstats` captured at the first token |

Promotion: gates `[12]`/`[Paris]` green, pi-stability PASS, 4K prefill in band,
short/4K decode at or above the round-11 band, and the `gdn_output` /
`gdn_projection` / `gdn_recurrent` scope sums no worse than 0.361 / 0.597 /
0.158 ms per block.  Revert is unsetting the env; dropping the branch removes
both kernels without touching the default path.

## 8. Ranked next steps

1. **Read the blade VGPR counts** (section 5) and promote/reject `_ws` on the
   rule.  This is the one measurement the previous round asked for and it has
   not been taken.
2. **A/B `_pair`**; if it wins on the output and GR shapes, the trip/contiguity
   hypothesis is confirmed and the layout item drops in priority.
3. **Row-interleaved cooked layout** (section 4) as its own pack round if
   `_pair` is neutral or negative and the VGPR trace rules out occupancy: it is
   the only remaining structural change that matches the winning shapes'
   memory pattern, and it is cheap to evaluate once a pack rebuild is in the
   loop.
4. **Small-grid shapes** (QSA kv 64 workgroups at 92 GB/s, shared gate 80 at
   112 GB/s): neither tile nor split nor the two variants above address a
   64-workgroup dispatch.  The only fix is dispatch fusion (a batched
   projection covering several small shapes in one grid), which is a vk.c and
   caller change, not a shader one.
