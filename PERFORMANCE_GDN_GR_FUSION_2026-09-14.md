# GDN / GR fusion round (2026-09-14, `fg-work-gdn8`, base `b721ceb`)

Scope: batch-1 decode only. Kernels in this workstream: GDN conv/recurrent/PLE
decode shaders, the dense Q8 r8/split shaders, the GR mix/write shaders,
`fg_decode_tile_schedule` and `src/owner.c`. `src/vk.c`, `src/runtime.c`,
`src/expert.c` and the `fg_moe_*` shaders are owned by other workstreams this
round and were not touched.

Mission targets: GDN ~10.9 -> <=7 ms/token, GR chains ~6.6 -> <=4 ms/token at
short decode 22.52/23.01 TPS and 4K 21.64/21.56.

## 0. TL;DR

Five commits: three shader cuts (r8 dense geometry, GDN recurrent load
pipeline, GDN conv vector access), one GR write vectorisation and one new
decode-shape parity oracle. All local oracles pass, including the bit-exact
GDN state oracles and the new r8/split decode-shape parity test. The changes are
expected to take the r8-dominated share of GDN and the GR `up` projection closer
to the measured 278-324 GB/s band, but they do not reach the mission targets on
their own: the remaining gap is in per-dispatch serialisation (`vk.c`,
other workstream) and in the rms/inject/silu kernels that are outside this
workstream's file set. Section 3 ranks the next steps with byte evidence.

## 1. Byte and instruction accounting (baseline, from prior profiles)

Per 6-layer block, rank 2 (`PERFORMANCE_DECODE_OVERLAP_2026-09-14.md` §1.2):
expert pair 1.42 ms, QSA 0.80, GDN projection+recurrent+output 1.21,
GR reads 0.83, QSA projection 0.49, shared expert + router 0.37, QSA output
0.19, remains 0.12 -> 5.43 ms/block. Eight blocks per token give the mission's
GDN ~10.9 and GR ~6.6 ms/token.

### 1.1 GDN (4-5 GDN layers per 6-layer block)

| piece | bytes/layer | measured | useful rate | note |
|---|---:|---:|---:|---|
| qkv 10240x2560 q8_0 cooked | 27.85 MB | ~100 us | 278 GB/s | r8, 1280 wgs |
| z 6144x2560 cooked | 16.71 MB | ~52 us | 324 GB/s | row-4 cooked, 1536 wgs |
| output 2560x6144 cooked | 16.71 MB | ~97 us | **172 GB/s** | r8, only 320 wgs |
| alpha/beta 2x48x2560 f32 | 0.98 MB | ~4 us | - | dense_f32 |
| conv | 0.57 MB | ~5-10 us | - | 40 wgs of 256, 56 B/lane |
| recurrent state | 9.0 MB | ~45 us | ~200 GB/s | 3 passes over 48x128x128 f32 |

The two r8 projections are the same kernel at the same bytes/row; the output
shape runs 1.6x slower per byte than qkv because it exposes only 320
workgroups (640 warps) where qkv exposes 1280 (2560 warps) - occupancy, not
DRAM. That is the target of commit 1.

The recurrent kernel streams the 3 MB state three times (read for
`memory`/`prior`, read+write for the update) and each lane's loads at
`state_base + k*128 + lane` are stride-512 B: with only 48 workgroups (2 per
CU) the 4x unroll leaves ~4 loads in flight per lane. Commit 2 doubles that.

### 1.2 GR read chain (12 reads per block, ~0.83 ms/block measured)

| kernel | bytes/read | instruction shape |
|---|---:|---|
| group_rms_norm | 80 KB + weights | 4 wgs, 7-barrier tree (not this set) |
| down split (10240->320) | 3.48 MB w + 40 KB x + 10 KB partials | 640 wgs of 64, split/reduce |
| split reduce | 10 KB | 2 wgs (not this set) |
| hc_inject_partial | 203 KB | 24 wgs, 8-barrier tree (not this set) |
| silu_scaled | 2.6 KB | 2 wgs (not this set) |
| up r8 (320->10240) | 3.48 MB w + 2.6 MB input rereads | 1280 wgs of 64 |
| gr_mix_partial | 90 KB | 10 wgs |

Total ~7.45 MB per read -> 89 MB/block -> `89 MB / 0.83 ms = 107 GB/s`
effective against a 353 GB/s ceiling. The chain is dispatch- and
latency-bound, not byte-bound: the weights alone (6.96 MB/read) would take
19.7 us at the ceiling versus ~69 us measured per read. The r8 `up` call is
the only one of these kernels in this workstream's set, so this round can only
move ~1/3 of the chain.

## 2. Changes, expected effect, local evidence

### Commit `9164653` - 16-cohort 128-lane r8 dense kernel

`fg_dense_q8_0_cooked_r8`: 64 lanes -> 128 lanes, 8 -> 16 cohorts, group
stride 8 -> 16 blocks. Per lane the serial loop halves (output 24 -> 12
iterations, qkv 10 -> 5, up 2 -> 1) and each workgroup carries 4 warps instead
of 2; the scale vector is loaded by one elected lane per 8-lane cohort and
shuffled (the row-4 cooked and split kernels already use this pattern, safe at
llvmpipe's subgroup size 8; a `gl_SubgroupSize>=8` guard keeps a direct-load
fallback). Arithmetic per lane and the final subgroup reduction are unchanged.

Expected: recover most of the 172 -> 278-324 GB/s gap on the latency-bound
shapes (GDN output 5 calls, GR up 12 calls, shared expert 18 calls per block).
Estimate -0.2..-0.45 ms/block (-1.6..-3.6 ms/token) across GDN + shared +
GR up, most of it in GDN output.

Evidence: `q8_decode_shape_parity` (new) passes for 320x10240, 6144x2560,
2560x10240 and the 10240x320 split at 2e-4 relative; `FG_BENCH_Q8_COOKED=1`
parity passes on all 11 production shapes (max_rel 8.2e-8), including under
llvmpipe's subgroup size 8.

### Commit `13c7d20` - 8-deep load pipeline in the algebraic recurrence

`fg_gdn_recurrent_algebraic`: the first pass issued 4 independent column loads
per lane; now 8, with the `memory` and `prior_result` accumulations kept
strictly sequential (bit-exact state). A 4-accumulator variant was tried first
and rejected by `gdn_algebraic`'s bit-exact state check - delta depends on the
memory rounding order, so the recurrence arithmetic must not be reordered.

Expected -10..-20% on the 0.22 ms/block recurrent total (-0.15..-0.35
ms/token). Evidence: gdn_algebraic, gdn_decode, gdn_prefill_scan,
gdn_prefill_decode_compat and all four gdn_chunked_prefill parity oracles pass.

### Commit `ff8b8d8` - vector state/weight access in the conv

`fg_gdn_conv_decode`: 14 global instructions per channel -> 6 (two vec4 loads,
two vec4 stores, projection read, output store) via vec4 views of the same
buffers. Same expression association as `fg_gdn_conv_prefill`, so
`gdn_prefill_scan`'s bit-exact decode-vs-prefill comparison still passes.
Small: ~0.02 ms/block.

### Commit `db38c06` - vectorized gated residual write

`fg_gr_write`: 4 features per thread through vec4 views. `block_output` is
read once per call instead of four times (40 KB -> 10 KB per call; 12 calls
per block, ~3.9 MB/token less traffic, plus 4x fewer threads). Small:
~0.02 ms/block. `gr_mix`, `gr_batch`, `gr_partial_boundaries` pass.

### Commit `1ac8c4d` - decode-shape parity oracle

`tests/test_fg_vk.c`: `q8_decode_shape_parity` runs the four production decode
shapes (GR up r8, GDN output r8, GDN qkv r8, GR down split) against the
raw-layout generic kernel at 2e-4 relative. The r8 rewrite and the split path
had no ungated local oracle (`q8_cooked_benchmark` is gated and
`tests/test_hc_down_split` does not link - its Makefile target omits
`src/topology.o`, pre-existing).

## 3. Ranked next steps

1. **Per-dispatch barriers (`src/vk.c`, other workstream).** `dispatch_impl`
   with `batch_barrier` inserts a full compute->compute pipeline barrier before
   every dispatch; ~190-220 dispatches/block are serialised at their stage
   boundaries. A decode block that could overlap independent dispatches (qkv
   with z, down with inject, rms with the previous layer's tail) would recover
   the 3x gap between the 107 GB/s GR chain and its 353 GB/s traffic floor.
   Estimated -1..-3 ms/token across GDN+GR; not reachable from this file set.
2. **Fuse the GR read chain tail (`rms`, `inject`, `silu` into split/up).**
   These four kernels move only ~0.9 MB/read but cost 4 dispatches; the r8 up
   already reads `low` - passing the raw tensor plus a silu push flag would
   remove one dispatch and 2.6 KB/read. Needs vk.c binding/flag work or
   ownership of `fg_hc_inject_partial`/`fg_group_rms_norm`.
3. **16-row r8 tiles (r16).** Halves the per-workgroup input re-reads (7.7 MB
   -> 3.8 MB per GDN output call, 2.6 -> 1.3 MB per GR up call) and halves the
   workgroup count further. Risk: ACO register pressure - the 4x4 prefill tile
   restructure regressed and was reverted. Try as its own commit behind the
   fleet A/B.
4. **Split-kernel balance.** A 16-cohort split kernel maps 320 blocks to 20
   groups over 8 splits (2-3 iterations, 20% tail); changing
   `FG_HC_DOWN_SPLITS` to 10 makes it exact (2 iterations each) if the reduce
   still validates. `tests/test_hc_down_split` must be made linkable first
   (add `src/topology.o` to its Makefile target).
5. **GDN conv state layout.** The 9-float-per-channel shift cannot be
   vectorised at 4-byte alignment; plane-major state (9 arrays of 10240)
   would let the conv use one vec4 shift load/store, but the prefill conv
   kernel shares the layout.

## 4. Local validation (llvmpipe, correctness only)

```
make all -j8                                     warning-free (-Werror)
tests/test_qsa_prefill                           PASS
tests/test_expert_prefill                        PASS
tests/test_owner_reduce                          PASS
tests/test_fg_vk (focused, 36 tests run one by one)
  q8_dense, q8_dense_subgroup, q8_dense_cooked, q8_cooked_prefill_parity,
  q8_decode_shape_parity (new), q8_embedding, hc_finalize, gr_mix, gr_batch,
  hc_inject_partial, gr_partial_boundaries, group_norm, ple_decode,
  ple_prefill_scan, ple_prefill_t1_compat, gdn_project_cooked, gdn_decode,
  gdn_algebraic, gdn_prefill_scan, gdn_prefill_decode_compat,
  gdn_chunked_prefill_parity_{zero,random,extreme},
  gdn_chunked_prefill_{composition,decode_compat}, batch_submission_parity,
  pipeline_flush_parity, static_batch_replay, gpu_profile, qsa_* ...          PASS
tests/test_fg_vk (full run)                      known llvmpipe crash at
                                                 qsa_record_commit (expected)
tests/test_core                                  1 pre-existing fail (457)
tests/test_session                               2 pre-existing fails (254/257)
tests/test_prefill_dispatch                      pre-existing fails (178-183)
```

`tests/test_hc_down_split` parity was checked by linking it manually with
`src/topology.o` appended (its Makefile target omits it; parity PASS,
max_rel 3.8e-6 vs the 2e-4 tolerance). The llvmpipe timings in the benchmark
are not representative of RADV/GFX1013 and were not used for the estimates.

## 5. Fleet A/B plan (for the orchestrator)

No new flags or environment variables; all changes are unconditional. A/B the
branch against `main` at the same ring pack and thermal state:

1. Gates: `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"` ->
   `[12]` / `[Paris]`.
2. Short + 4K battery x2:
   `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Runs4k 2`.
3. Context sweep 16384/32768/65536: `tools/context-sweep.ps1`.
4. Soak: `tools/pi-stability.ps1` (6 stages, 4-turn conversation, 8 ranks).
5. Per-kernel attribution (optional): `FG_DECODE_PROFILE=1` on rank 0 and one
   worker; compare `fg_dense_q8_0_cooked_r8` (expect the largest move on the
   6144->2560 output shape), `fg_gdn_recurrent_algebraic`, `fg_gdn_conv_decode`
   and `fg_gr_write` against their unchanged neighbours in the same run.

Revert points: each commit is self-contained; if the fleet disagrees, drop
`9164653` (r8 geometry) first - it moves the most and carries the most
scheduling risk.

## 6. Blockers / honest gaps

- GDN <=7 ms/token and GR <=4 ms/token are not reachable from the
  workstream's file set alone. The GDN floor is a roofline fact: 36 GDN layers
  per token x 61.2 MB of projection weights = 2.2 GB, plus ~0.32 GB of
  recurrent state, at the measured 353 GB/s ceiling is ~7.1 ms/token before
  any launch or barrier cost. The target therefore requires every GDN kernel
  (qkv 278, z 324, output 172, recurrent ~200 GB/s) to run at ~100 % of the
  stream ceiling - the 172 GB/s output shape is the only one meaningfully
  below it, and commit 1 is aimed exactly there. For GR the byte floor is
  84 MB/block (0.24 ms/block at 353 GB/s, 1.9 ms/token), so <=4 ms/token is
  reachable in principle, but ~3x of the current time is per-dispatch
  serialisation plus the rms/inject/silu kernels owned by other workstreams;
  items 1-2 of section 3 are required.
- `tests/test_hc_down_split` cannot link on this tree (Makefile target missing
  `src/topology.o`) - pre-existing, out of set; parity was verified manually.
