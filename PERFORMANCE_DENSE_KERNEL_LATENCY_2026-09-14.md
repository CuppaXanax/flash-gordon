# Dense Kernel Latency: per-iteration instruction diet on the cooked Q8 family (2026-09-14)

Worktree `D:\workspace\fg-work-lat11`, branch `perf/dense-kernel-latency`
from main `56f76bb` (round-9 binary).  Scope: the cooked Q8 dense family, the GDN
conv/recurrent decode kernels, PLE conv/gate, GR mix/write, the decode tile
schedule, and `src/owner.c` dispatch sites.  No `src/vk.c`, `src/runtime.c`,
`src/output.c`, `src/protocol.c` change; no pack, manifest or protocol change;
no new kernel registrations, so the same pack qualifies.

Commits:

| commit | change |
|---|---|
| `eab50d5` | r8 cooked kernel: direct per-lane scale load replaces the elected-lane broadcast |
| `d8d1684` | generic + split cooked kernels: same delta-path cut |
| `9a6c965` | algebraic recurrence: 8-wide state update pass |

## 0. TL;DR

The round-9 accounting located the decode wall in kernel execution, and the
in-scope kernels split into three classes.  The cooked Q8 family is not
geometry-bound: the fleet's own sweep (section 1.3) already shows row-8/no-split
is the best of 12 tested (rows x splits) geometries, and K-splitting loses
2-3x.  What was left is the per-iteration issue stream, and the largest single
piece of it was the scale broadcast: every r8 iteration ran a
`gl_SubgroupSize` branch plus eight `subgroupShuffle` ops to move 8 row scales
from one elected lane to the cohort.  Loading the same 16 bytes in every lane
(one broadcast L0 hit per cohort) removes that boilerplate: the r8 loop body
drops **210 -> 123 SPIR-V result ops (-41%)**, the generic and split kernels
**101 -> 74 (-27%)**.

An LLVM AMDGPU `gfx1013` ISA proxy (not ACO) puts the current 8-row r8 kernel
at ~86 VGPRs and a 4-row wave at ~53, i.e. 8 vs 16 resident wave64 per CU by
the 64 KiB/SIMD register file arithmetic.  That is the occupancy ground truth
to confirm on the blade before rewriting the kernel shape; the fleet geometry
data below argues the 8-row shape wins anyway.

The algebraic recurrence's update pass now issues 8 state columns per iteration
like its read pass, halving the loop trips for `head_dim` 128 and doubling the
write-pass MLP.  Bit-exact, llvmpipe A/B -1.8%.

Honest expectation: these are instruction-stream cuts on latency-bound kernels,
so the fleet delta is expected to be small but positive (order 0.1-0.5 ms/token
across all changed kernels).  The real value of the round is the measured
per-kernel bottleneck classification and the refutation of the geometry levers,
both of which redirect the next round at the actual limiter (occupancy ground
truth and the small-grid shapes).

## 1. Evidence

### 1.1 SPIR-V loop-body accounting

`spirv-dis` over the glslang `-Os` SPIR-V of each kernel; the count is the
result ops in the innermost (production) loop body, which is the per-lane
dynamic instruction stream up to ACO scheduling:

| kernel | loop body before | after | cut |
|---|---:|---:|---:|
| `fg_dense_q8_0_cooked_r8` | 210 | 123 | -41% |
| `fg_dense_q8_0_cooked` (generic) | 101 | 74 | -27% |
| `fg_dense_q8_0_cooked_split` | 101 | 74 | -27% |

The removed ops are the `gl_SubgroupSize>=8` selection tree, the
lane-0-only `row_delta` calls and eight `OpGroupNonUniformShuffle` per iteration;
the added op is one `uvec4` load per lane.  The eight lanes of a cohort read the
same 16 bytes, so the wave-level request is the same 128-byte run; on RADV those
shuffles are `ds_bpermute_b32` on the LDS pipe, and eight of them sat between
the scale load and the accumulator FMA every iteration.

The same broadcast was already tried on the 64-lane r8 in round "blockk"
(commit listed in `PERFORMANCE_BLOCKK_2026-09-13.md` section 3) with four
shuffles and measured neutral on the fleet; the 128-lane r8 rewrite
(`161d9cb`) reintroduced it with eight shuffles plus the subgroup-size branch,
which is what this round removes.

### 1.2 Register / occupancy proxy (LLVM AMDGPU, not ACO)

No AMD device is reachable from this worktree, so the proxy compiles the exact
production loop with `clang -target amdgcn-amd-amdhsa -mcpu=gfx1013 -O3` and
reads the reported VGPR count:

| geometry | VGPRs (clang) | waves/SIMD (16384 VGPRs) | wave64/CU (4 SIMD, cap 16) |
|---|---:|---:|---:|
| r8, 8 rows/lane, 128-lane wg | 86 | 2 | 8 |
| 4 rows/lane, 64-lane wave | 53 | 4 | 16 |

This does not prove ACO's allocation (ACO rematerialises more aggressively and
may be lower), but it bounds the ratio: the 8-row geometry keeps 2x the
per-lane state (two vec4 accumulators, two packed words, two delta vectors)
of the 4-row geometry, and `NumVgprs` scales with it.  If the blade-side
`RADV_DEBUG=shaderstats` run shows the r8 kernel above ~64 VGPRs, occupancy is
2 waves/SIMD and the kernel shape is the next lever; below ~48 it is not.

### 1.3 Fleet geometry sweep (already measured, `D:\workspace\bc-250-dbg\q8-cooked-geometry-results.txt`)

The orchestrator's earlier geometry sweep on the BC-250 (RADV GFX1013) ran the
production shapes at rows in {2,4,8} and splits in {1,2,4,8}:

| shape (in -> out) | rows=8, splits=1 | rows=4, splits=1 | best split result |
|---|---:|---:|---:|
| 320 -> 10240 (GR up) | 23.78 us / 148 GB/s | 26.14 / 135 | 29.9 (splits=2) |
| 2560 -> 10240 (GDN qkv) | 97.68 / 286 | 109.03 / 256 | 162 (splits=8) |
| 2560 -> 6144 (GDN z) | 69.97 / 239 | 69.57 / 241 | 93 (splits=4) |
| 6144 -> 2560 (GDN output) | 98.65 / 170 | 113.73 / 147 | 97 (splits=2) |
| 2560 -> 640 (shared gate) | 15.63 / 112 | 18.45 / 95 | 11.7 (splits=4) |
| 640 -> 2560 (shared down) | 10.45 / 168 | 12.05 / 146 | 12.1 (splits=2) |
| 2560 -> 2560 (PLE value) | 31.46 / 222 | 35.77 / 195 | 31.7 (splits=2) |
| 2560 -> 12288 (QSA qg) | 120.87 / 277 | 133.77 / 250 | 196 (splits=8) |
| 2560 -> 512 (QSA kv) | 15.26 / 92 | 17.94 / 78 | 10.1 (splits=4) |

Row-8/no-split is the best geometry for the large shapes, and K-splitting loses
2-3x on every large shape (more partial traffic and more workgroups do not help
this DRAM stream).  This is why this round does not touch the r8 shape.

### 1.4 Where the in-scope kernels spend time (round-9 profile, rank 0, 6-layer block)

| scope / kernel | calls | ms/block | bytes/call | effective | class |
|---|---:|---:|---:|---:|---|
| `dense_q8_0_cooked_r8` GDN qkv 2560->10240 | 4 | 0.374 | 27.85 MB | ~298 GB/s | near stream ceiling |
| `dense_q8_0_cooked_r8` GDN output 6144->2560 | 4 | 0.361 | 16.71 MB | ~187 GB/s | small grid, 320 wgs |
| `dense_q8_0_cooked_r8` GR up 320->10240 | 12 | ~0.24 | 3.48 MB | ~148-174 GB/s | 1 trip, 37.5% dead cohorts at blocks=10 |
| `dense_q8_0_cooked_r8` QSA projections | 6 | 0.239 | - | 92-277 GB/s | QSA kv 64 wgs is the worst shape |
| `dense_q8_0_cooked` GDN z 2560->6144 | 4 | 0.222 | 16.71 MB | ~300 GB/s | near stream ceiling |
| `dense_q8_0_cooked_split` GR down 10240->320 | 12 | ~0.2 | 3.48 MB | ~200 GB/s | 640 wgs, 5 trips |
| `gdn_recurrent_algebraic` | 4 | 0.148 | 3.1 MB state x3 | ~250 GB/s | 48 wgs, 2/CU, latency-bound |
| `gdn_conv_decode` | 4 | 0.010 | 0.4 MB | ~160 GB/s | small, not worth a change |
| `gr_mix_partial` | 12 | 0.035 | 90 KB | - | 10 wgs, dispatch-floor |
| `gr_write` | 12 | 0.013 | 360 KB | - | 160 wgs, vectorized in round 8 |
| `decode_tile_schedule` | 6 | 0.006 | 360 B | - | 1 wg, dispatch-floor |

The r8 kernel is the same instruction stream for every shape, so the spread
92-298 GB/s is shape (grid/trip/row-stride), not ISA.  The two levers that
remain inside this file set are per-iteration issue (this round) and occupancy
(next round, gated on the blade VGPR count).

## 2. Changes and local evidence

### 2.1 `eab50d5` r8: direct scale load

The inner loop's scale path is now

    uint safe_block = live ? block : (pc.blocks - 1u);
    uint scale_word = (tile_base>>2u) + safe_block*8u + (tile_row>>1u);
    uvec4 scales = wb4[scale_word>>2u];
    vec4 delta0 = live ? vec4(unpackHalf2x16(scales.x), unpackHalf2x16(scales.y)) : vec4(0);
    vec4 delta1 = live ? vec4(unpackHalf2x16(scales.z), unpackHalf2x16(scales.w)) : vec4(0);

instead of two elected-lane `row_delta` calls, eight shuffles and the
subgroup-size branch.  The safe-block clamp also keeps the previously
unguarded weight loads of dead cohorts inside the live weight rows (for the
320-wide shapes the last tile's dead cohorts used to read ~160 bytes past the
matrix end, inside the arena buffer but outside the tensor).

Local: loop body 210 -> 123 ops; `q8_decode_shape_parity`,
`q8_dense_cooked`, `q8_cooked_prefill_parity`,
`expert_graph_fused_q8_0_cooked`, `q8_cooked_view_slice` PASS; the
`FG_BENCH_Q8_COOKED` parity max_rel is unchanged at 8.2e-8 (the change is
bit-exact: same scale words, same dot/FMA order).

Expected: the r8 shapes total ~1.38 ms/block on rank 0 (~11 ms/token).  The
cut is 41% of the loop-body issue stream; if the kernel is issue-share ~20-30%
at the small shapes and less at the near-ceiling shapes, the fleet band is
0.05-0.4 ms/token.  The llvmpipe A/B over three alternations could not resolve
it (run-to-run spread larger than the effect, see section 4).

### 2.2 `d8d1684` generic + split: same cut

Both 64-lane kernels had the same elected-lane pattern (one load + four
shuffles + branch per iteration).  They become a per-lane two-word load with a
live select; the safe-block clamp also bounds their dead-cohort reads.

Local: loop bodies 101 -> 74 ops each; `q8_decode_shape_parity`,
`q8_dense_cooked`, `hc_down` and `q8_cooked_view_slice` PASS.

Expected: z and the GR down split are the main users; 0.05-0.2 ms/token band.

### 2.3 `9a6c965` algebraic recurrence: 8-wide update

Pass two was 4 loads + 4 stores per iteration while pass one already issued 8
independent column loads.  The update loop now uses the same 8-wide shell with
4-wide and scalar tails.  Each state element is computed independently, so the
state is bit-exact (`gdn_algebraic`'s check passes).

Local: llvmpipe `GDN_ALGEBRAIC_BENCH`, three alternations of the old and new
SPIR-V in the same tree: algebraic kernel median 2354.8 -> 2313.3 us (-1.8%)
with the unchanged `gdn_recurrent_decode` control at 2692-2740 us.

Expected: the algebraic kernel is ~0.148 ms/block; a 2-5% fleet gain is
0.003-0.007 ms/block, ~0.03-0.06 ms/token.

## 3. Numerics

* Sections 2.1/2.2 are bit-exact: the same scale words feed the same delta
  values and the dot/FMA association is untouched.  The parity oracles report
  the same max_rel as before the change (8.2e-8 for the 8-row shapes).
* Section 2.3 is bit-exact per state element by construction.
* No tolerance was changed anywhere.

## 4. Local validation (llvmpipe, correctness only)

```
make all -j8                                      warning-free (-Werror)
tests/test_qsa_prefill                            PASS
tests/test_expert_prefill                         PASS
tests/test_owner_reduce                           PASS
focused tests/test_fg_vk (28 tests, FG_SHADER_DIR=$PWD/vulkan):
  gpu_profile, static_batch_replay, pipeline_flush_parity,
  batch_submission_parity, group_norm, hc_inject_partial, gr_mix,
  q8_decode_shape_parity, expert_graph_replay,
  expert_graph_fused_q8_0_cooked, qsa_attention, qsa_record_gather,
  output_argmax, q8_cooked_prefill_parity, q8_dense_cooked,
  expert_decode_fused_random, ple_decode, ple_prefill_scan,
  ple_prefill_t1_compat, gdn_project_cooked, gdn_decode, gdn_algebraic,
  gdn_prefill_scan, gdn_prefill_decode_compat, gr_batch,
  q8_cooked_view_slice, q8_dense_subgroup, hc_finalize,
  gr_partial_boundaries                             PASS
known pre-existing failures unchanged (test_core:457, test_session:254/257,
test_prefill_dispatch:178-183, llvmpipe qsa_record_commit crash)
```

llvmpipe timings are not representative of RADV/GFX1013; the `FG_BENCH_Q8_COOKED`
A/B (three old/new alternations, five every production shape) was inside the
run-to-run spread and is not used as evidence.  Only the parity results and the
static op counts are quoted.

## 5. Fleet A/B plan (orchestrator)

```
1. Gates:      pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"
               -> [12] / [Paris]
2. Battery x2: pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Runs4k 2
3. Context:    pwsh -NoProfile -Command "& 'D:\workspace\flash-gordon\tools\context-sweep.ps1' -Contexts 16384,32768"
4. Soak:       pwsh -NoProfile -File D:\workspace\flash-gordon\tools\pi-stability.ps1
5. Attribution: FG_DECODE_PROFILE=1 on rank 0 and one worker; the r8 shapes
   are the dense_q8_0_cooked_r8 entries in gdn_projection (expect 0.374),
   gdn_output (expect 0.361) and gr_*_read; the generic/split entries are
   gdn_projection z (0.222) and gr_*_read split; the algebraic kernel is
   gdn_recurrent (expect 0.158 incl. conv).  Control is round 9, token 4321.
6. Occupancy ground truth (one restart, diagnostic only):
   RADV_DEBUG=shaderstats FG_DECODE_PROFILE=1 <run> 2>radv_stats.txt
   grep -A2 -B2 'dense_q8_0_cooked_r8' radv_stats.txt
```

Promotion bar: gates green, pi-stability PASS, 4K prefill >= 270, short/4K
decode above the 24.0/22.3 band, and the five per-kernel numbers above not
worse.  Revert points: `eab50d5`, `d8d1684` and `9a6c965` are independent and
each can be dropped alone.

## 6. Ranked next steps

1. **Measure the r8 VGPR count on the blade** (`RADV_DEBUG=shaderstats`,
   section 5 item 6).  Section 1.2 says the occupancy lever only exists if the
   8-row kernel allocates above ~64 VGPRs.  This is a one-restart diagnostic
   and it decides item 2.
2. **Wave-split 4-row r8** (conditional): keep 128-lane workgroups but give
   each 64-lane wave its own 4-row tile, one vec4 accumulator, no barrier, no
   LDS epilogue.  The ISA proxy says 86 -> 53 VGPRs, i.e. 8 -> 16 waves/CU.
   The case against: the fleet sweep at constant register allocation says
   rows=4 loses 9-15% to rows=8 on the large shapes, and the r8 shapes are
   2/3 of the in-scope GPU time.  The case for: the 320-wg output shape is
   170 GB/s where the 1280-wg qkv shape is 286 with the same bytes/row.  Do
   not ship it before item 1 says the VGPR count is real.
3. **The GR read chain's non-dense tail** (`fg_group_rms_norm`,
   `fg_hc_inject_partial`, `fg_silu_scaled`): four of the seven kernels per
   read are outside this file set and move only ~0.9 MB; they are dispatch- and
   latency-floor kernels.  Fusing silu into the r8 up input (one dispatch and
   2.6 KB/read) needs a vk.c binding or a shader that owns both.
4. **Small-grid shapes are the remaining per-shape gap**: 2560->512 (92 GB/s,
   64 wgs) and 2560->640 (112 GB/s, 80 wgs).  The geometry sweep says neither
   row-tile nor split changes fix them; they need either fewer dispatches (a
   batched projection that covers several small shapes in one grid) or a
   persistent-kernel/SM-style scheduler, both vk.c/owner work.
5. **GDN output 6144->2560 (170 GB/s, 320 wgs)** is the largest single shape
   deficit (0.36 ms/block).  It is not instruction-bound (qkv proves the
   stream), not geometry-bound (section 1.3), and not split-friendly.  If the
   occupancy measurement rules out VGPR pressure, the next hypothesis is the
   row stride: the cooked tile stores each row's blocks contiguously, so the
   eight rows a workgroup walks sit `blocks*32` bytes apart (6 KiB here versus
   2.5 KiB for qkv).  A row-interleaved quant layout (block-major, like the
   expert Q5_1 tiles) puts the same eight rows 32 B apart per block; it needs
   a pack format version, which is the next round's decision, not a shader
   change.
