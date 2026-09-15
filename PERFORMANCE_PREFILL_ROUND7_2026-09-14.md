# Prefill round 7: long-context selection scan (2026-09-14)

Worktree `fg-work-pref7`, branch `perf/prefill-round7`, base `main` @ 2c9be2a.
Control binary `59b2f559` (decode-overlap round 2, all 8 blades). Round-7 binary
after the first commit: `eb84975b`.

Scope: prefill only. No pack, manifest, protocol, decode-path or geometry
change. The measurement below is the round's primary artifact (the mission
asked for the stage costs first); the shader work attacks the measured top
item.

## 1. Method

The fleet was restarted from the control binary with `FG_PREFILL_PROFILE=1`
on the seven workers (`pref7-start-workers-prof.sh`), rank 0 unchanged
(`start-rank0-ring.sh`). Profiling adds timestamp queries per layer inside
`handle_prefill_layer_work`; the resulting wall/GPU costs were collected with
the existing `PREFILL_LAYER_PROFILE` / `PREFILL_LAYER_KERNEL` lines and
aggregated per layer sample (one sample = one 128-token microbatch) in
`pref7-prof-full.sh`.

`tools/context-sweep.ps1` drove the three points. Sample indices per layer in
the worker log are `1..4` correctness gate, `5..36` 4K prefill, `37..68` 4K
decode leg, `69..580` 64K prefill, `581..1092` 64K decode leg, `1093..3140`
256K prefill. TPS on the profiled fleet was 263/274 (4K), 348/310 (64K),
236.5 (256K) - i.e. profiling is free at long context and the 64K number is
inflated by warm prefix state; per-kernel GPU times are the comparable
quantity.

## 2. What the profile says

Per-microbatch GPU time and the top kernels, rank 1 (layers 0-5; layer 3 is a
QSA layer, layer 0 a GDN layer), means over the window:

| context | QSA layer gpu | index_score | topk_merge | attention | q8 proj tile | gate/up | GDN layer gpu |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K | 37.1 | 0.57 | 0.96 | 5.30 | 8.77 | 7.67 | 30.8 |
| 64K | 53.4 | 11.95 | 4.51 | 6.92 | 8.64 | 7.53 | 30.0 |
| 256K (whole request) | 96.3 | 47.96 | 11.27 | 7.01 | 8.65 | 7.54 | 30.0 |
| 256K (last 33K depth) | 147.8 | 91.84 | 18.94 | 7.07 | 8.67 | 7.54 | - |

Same-block shares at 256K tail: selection (`index_score` + `topk_merge`) is
110.8 of 147.8 ms, 75% of the QSA layer. The context-independent half of the
stack is flat: expert gate/up 7.54, down (q5_1) 2.41, router 2.26, q8 tile
projections 8.65, GDN total 30.0, `record_gather_batch` 2.10, attention 7.01.
The 4K -> 256K TPS delta (281 -> 238) is therefore not the experts, the GDN
stack or the attention kernel: it is the candidate-block scoring scan and the
top-512 merge that grow linearly with context.

Wall vs GPU per layer (PREFILL_BLOCK_LAYER vs PREFILL_LAYER_PROFILE) shows a
constant ~5.5 ms of host/fence time per GDN layer and ~14 ms (4K) to ~30 ms
(256K) per QSA layer; `QSA_PREFILL_TRACE` splits the QSA wall into
`select_ms` (GPU selection + one readback) and `attend_ms` (32 four-query
tiles, each with a staging fence). At the 256K tail, `select_ms` was 94-99 ms
of a 160 ms layer wall. That part is QSA staging/residency work, not a shader
item; evidence and a hand-off are in section 5.

## 3. `89f0ee1` - block-major index score

`fg_qsa_index_score.comp` was dispatched one workgroup per (candidate block,
query) pair, 128 threads. Every pair re-ran the block-scoped pooled key
transform (4x q8 decode, RMS tree with seven barrier stages, RoPE with
`pow`/`cos`/`sin`) and then four 128-lane shared-memory dot trees, again with
seven barrier stages each. With 16 queries per selection batch the transform
was computed 16x redundantly per block, and the inner loop paid 28 barriers
for 512 MACs.

The rewrite keeps the same bindings, push constants and output layout but
makes the grid one-dimensional (one workgroup per block):

* the pooled key decode and RMS reduction are done once per block (RMS via
  `subgroupAdd` plus one shared combine);
* RoPE is unchanged and still applied once (it was per pair before);
* the rotated key is published as `vec4[32]`; each query is scored by four
  32-lane head groups with a four-element FMA chain and a five-step
  `subgroupShuffleXor` butterfly;
* head sums are staged per query slot, so the query loop has no barriers
  between queries; visibility and the `-inf`/`0xffffffff` sentinel are
  resolved in the per-query epilogue.

Numerics: the RMS reduction order and the 128-lane dot trees are replaced by
subgroup trees, so score values can differ from the old kernel in the last
ulp (same class as the round-6 expert/QSA subgroup reductions). Selection
ids and tie-breaking (`id` ascending) are unchanged. Local tests
(`test_fg_vk` qsa_indexer, qsa_segmented_index_score, qsa_prefill_chunk_liveness,
qsa_prefill_prepare, qsa_attention, qsa_resident_hierarchical_topk;
`test_qsa_prefill`, `test_expert_prefill`, `test_owner_reduce`) all pass under
llvmpipe, and the whole tree builds `-Werror` clean.

## 4. A/B on the fleet

Binary `eb84975b` (main + `89f0ee1`), same ring pack, standard ring scripts,
caches dropped before each session, gates first. The control is the validated
`59b2f559` curve measured today on all eight blades (4K 280.8-284.4, 64K ~267,
256K 237.9; the control's own battery spread for 4K is 296-320 and its
context-sweep 64K has been seen at 348 in a warm session, so 4K/64K single
runs are dominated by run-to-run state).

| metric | control `59b2f559` | round 7 `eb84975b` |
|---|---:|---:|
| gates | [12] / [Paris] | [12] / [Paris] (three sessions) |
| 4K first request after start | 280.8-284.4 | 278.8 (battery r1) / 284.0 (repeat session) |
| 4K warm repeats | 296-320 (battery band) | 325.5 / 325.8 / 326.6 / 328.6 |
| 4K decode-leg re-prefill | - | 300.3 |
| 64K prefill | ~267 (validated) | 318.9 / 312.0 / 309.3 |
| 64K decode-leg re-prefill | - | 311.4 / 312.0 |
| 256K prefill | 237.9 | **274.8 / 275.0 (two runs, +15.6%)** |
| short decode (32 tok) | 21.76-21.93 | 22.32 |
| 4K warm decode | 21.02-21.19 | 21.13-21.17 |
| 64K decode | ~8.1 (cache follow-up) | 11.55 / 11.65 |

Reading:

* **256K: +15.6%** on the one position where the rewrite's target dominates
  (selection was 61% of the QSA layer mean and 75% at depth). Two independent
  runs landed at 274.83 and 275.02 TPS; the control's own 256K result was
  reproduced twice today (237.9 sweep, 236.5 profiled), so the reference is
  solid.
* **4K: neutral within spread.** The first request after a fresh start is
  278.8-284.0 on the new build against the 280.8-284.4 control band; warm
  repeats are slightly above the control's battery band but the control's
  warm band itself spans 30 TPS. The selection scan is only 1.5 ms of the
  37 ms 4K QSA layer, so no 4K gain was expected from this change.
* **64K**: 309-319 TPS across three runs. The validated control number is 267
  but the control's own profiled session ran 348/310 at this point, so the
  64K control band is wide; treat the delta as positive but not definitive.
* **Decode untouched**: short/4K/64K decode at or above the control.

### 4.1 Second candidate: `fg_dense_q8_0_cooked_tile` 4x4 restructure (measured, reverted)

The tile kernel is the single biggest 4K bucket (8.65 ms per QSA layer,
70-79 ms per six-layer block). A locally parity-clean variant - one 128-thread
workgroup, four rows x four tokens per thread, both operands as `vec4` shared
loads, LDS cycles per FMA 0.75 -> 0.5 at a 16-accumulator budget - was
deployed and measured: 4K sweep points 271.0 / 318.4 / 318.4 / 291.7 against
325.5 / 325.8 / 326.6 on the index build. No gain; the kernel is not
LDS-bandwidth bound on this part. Reverted, not committed.

### 4.2 Third candidate: GDN recurrence shuffle reduction (reverted, blocked locally)

`fg_gdn_prefill_recurrence` spends 14 barriers per token in two 32-lane shared
reductions (2-3 ms per GDN layer, 15 ms per six-layer block). A wave-32
butterfly with a lane-0 broadcast reproduces lane 0's sum bit-for-bit without
shared memory. It fails `gdn_chunked_prefill_parity_random` under llvmpipe,
whose subgroup size is 8 (the 32-lane shuffle sequence is undefined there);
it needs a `gl_SubgroupSize >= 32` fast path with the old shared fallback.
Not deployed this round.

### 4.3 Fleet state left behind

The round-7 build is not promoted, so the fleet was restored to the validated
control sources: `src/vk.c` and `shaders/fg_qsa_index_score.comp` reverted to
`2c9be2a` and rebuilt on .42. The previous `59b2f559` binary had been
overwritten by the round's deploys and no copy existed on the blades, so the
restored control is a clean rebuild of the exact base sources
(`6be74071...`, all eight blades, standard `start-rank0-ring.sh` /
`start-workers-ring.sh`, ring pack). It was re-gated: `correctness64.ps1`
[12]/[Paris], `tools/pi-stability.ps1` PASS (6 stages, 4-turn conversation,
8 ranks, 0 failures; 4K 279.42 TPS, short decode 19.87). The round-7 package
(binary `eb84975b` + shader `e86690e3`) is reproducible from commit `89f0ee1`.

## 5. QSA selection/residency overhead handed to the cache workstream

Measured, not fixed in this round:

* **Per-tile staging fences.** `prefill_tile_attention` fences at entry for
  each 4-query tile (32 tiles per QSA layer per microbatch) and rewrites the
  shared `batch_slots`/`batch_counts`/tails. Cost: 6-8 ms per QSA layer per
  microbatch at 4K (attend wall 14.4 ms vs 7.4 ms of GPU kernels), growing to
  ~25 ms at 256K as the same fences cover slower selection work.
* **Selection readback.** One 128x512x4 B = 256 KiB host read per QSA layer
  per microbatch, preceded by `fg_qsa_submit_host_reads`. At 4K the select
  wall is 2.24 ms against ~1.5 ms of `index_score` + `topk_merge` GPU time;
  in the 64K deep phase it is 28.4 ms against ~16.5 ms; the 512-entry host
  cache lookups and the per-selection-batch turnarounds account for the rest.
* **Record gather is flat** (2.1 ms at 64K and 256K): with `missing=0` in the
  sampled traces the workers' 4096-page/layer cache does not thrash at these
  depths; the growth is entirely selection scan, not page fetch.
* The natural next steps are a queries-sized staging arena (batch_slots is
  8 KiB today; batch_records at 81 MiB for all 128 queries is too large for
  rank 0's mirror, so double-buffered 4-query arenas with pipelined host
  staging would need ~2.5 MiB more headroom), and keeping block->slot
  resolution on the GPU so the per-layer readback disappears.

## 6. Ranked next steps

1. `fg_qsa_resident_topk_merge` (11.3 ms mean / 18.9 ms tail at 256K): the
   4096-wide bitonic sort costs 78 barrier passes; a top-512 partial selection
   (bitonic top-k truncation or a radix/bucket select on sortable float keys)
   is the remaining selection cost after this round.
2. `fg_qsa_index_score` second pass: the kernel loads 32 KiB of query vectors
   per block (16 queries x 2 KiB) on every dispatch; with 65536 blocks at
   256K that is ~2 GiB per dispatch and the likely reason the speedup is
   bounded. Processing B blocks per workgroup with the queries in registers
   (64 VGPRs) cuts query traffic by B, or stage 8 queries per pass in shared.
3. `fg_gdn_prefill_recurrence` (2-3 ms per GDN layer, 15 ms per six-layer
   block at 4K): the wave-32 shuffle reduction above with a subgroup-size
   guard; bit-exact on wave64, needs the shared fallback for lavapipe.
4. Host-side QSA staging pipeline (section 5), owned by the QSA cache
   workstream: per-tile staging fences and the per-layer selection readback.
5. `fg_dense_q8_0_cooked_tile`: the 4x4 variant was neutral (section 4.1);
   the next shape to try is one keeping 256 threads (ROWS=64, TOKEN_TILE=64)
   before giving up on the LDS hypothesis.
