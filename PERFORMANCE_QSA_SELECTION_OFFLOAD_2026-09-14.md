# QSA decode selection: device-side slot resolution and the chunked top-k (2026-09-14)

Worktree `D:\workspace\fg-work-qsel11`, branch `perf/qsa-selection-offload`
from main `56f76bb`. Scope: `src/qsa.c`, `src/qsa_cache.c`, `include/fg_qsa_cache.h`,
`shaders/fg_qsa_select_resolve.comp` (new), minimal `src/vk.c`/`include/fg_vk.h`
kernel plumbing, `tests/test_qsa_prefill.c`. No pack, manifest, protocol or
prefill-path change.

| commit | change |
|---|---|
| `b360de1` | page-cache eviction hook for mirrored slot bookkeeping |
| `90b0ad4` | decode selection resolves block ids to cache slots on the device |
| `4898aea` | chunked top-k (`FG_QSA_TOPK_V2`) becomes default; v1 is the `=0` control |
| `eb8a18f` | resolved slots refresh the LRU, preserving eviction order |

## 0. TL;DR

The round-9 accounting put the QSA decode selection at `fg_topk_reduce`
0.271 ms per 6-layer block plus ~0.09 ms of exposed host work per selection
(512 hash lookups into the page cache, a 2 KiB id read, and a 2 KiB slot
write), all under a fence that drains the block. This round moves the
block-id to cache-slot mapping onto the device:

* a per-layer GPU page table mirrors the record cache exactly (inserts,
  evictions, reset);
* a new resolve kernel maps the sorted top-512 block ids to cache slots in
  one dispatch and counts misses;
* the decode path reads only two miss-counter words (8 B) in the common case,
  then hands the resolved slot buffer straight to the record gather; the id
  list never crosses to the host;
* a miss (cold page) falls back to the byte-identical old host path, which
  also repairs the table;
* the chunked top-k network (`FG_QSA_TOPK_V2`) is promoted to default — exact
  parity, 1.24-1.85x fewer comparators at the decode counts.

Estimated saving: ~0.05-0.07 ms per selection of host work plus ~25-45% of
the top-k kernel, i.e. **~1.1-1.6 ms per token summed over the eight ranks at
4K** (≈ +0.5 TPS at the 20 TPS baseline), growing with context because the
top-k term is the part that scales. The fence at selection is unchanged: the
selection GPU work is on the critical path either way and the miss counter
must be valid before the gather is recorded.

## 1. Where the time goes (end to end)

Decode QSA layer, cache-backed session (`fg_qsa_session_decode` ->
`commit_and_attend_cache` -> `attend_cache`):

```
projections (flushed behind the previous layer)
commit kernel (new page acquire + pin)
index_score            GPU  0.019 ms / 2 layers at 4K (round-9)
topk reduce loop       GPU  0.271 ms / 2 layers at 4K (round-9), v1
fg_vk_end              wait fence + drain every pending submission
tensor_read ids        2 KiB D2H, MOVNTDQA stream
locality records       host (env-gated)
cache_lookup x512      host hash + LRU (~0.09 ms/selection round-1)
tensor_write slots     2 KiB H2D
fg_vk_begin            new batch
gather + attention     GPU (unchanged)
```

Three separable costs (round-9 `RING_DECODE` block, rank 0, 4K):

| quantity | value | notes |
|---|---:|---|
| `fg_topk_reduce` GPU | 0.271 ms/block | 2 QSA layers, v1 network |
| exposed host work | ~0.18 ms/block | 2 selections x ~0.09 ms, GPU idle |
| fence wait | selection GPU time | `fg_vk_end` also drains pending flushes |
| index_score GPU | 0.019 ms/block | 4K; grows linearly with blocks |

The host work is what this round removes. The fence is not removable without a
miss-free guarantee: the gather is recorded only after the host knows the
selection did not hit a cold page, and a miss requires the host state-file /
mirror fetch.

New runtime split (env `FG_FRAME_TRACE=1`, printed per selection):

```
QSA_SELECT_TRACE layer=%u tokens=%u count=%u misses=%u wait_ms=%.3f read_ms=%.3f
```

`wait_ms` is the post-record fence (`fg_vk_end`), `read_ms` is the flags +
slots reads and the LRU touch loop. This is the line the A/B should quote.

## 2. Design

### 2.1 GPU page table

`fg_qsa_session` gains three tensors (created with the record cache in
`ensure_page_cache`, destroyed with the session):

* `slot_table`: `layer_count x max_blocks` u32, `0xffffffff` = not cached.
  Rank 0 at 256K is 12 x 65536 x 4 B = 3 MiB; a worker 3 x 65536 x 4 B = 768 KiB.
* `select_resolved`: 512 u32, one resolved cache slot per selected id.
* `select_flags`: 2 u32, miss counts per resolve workgroup (256 ids each).

Exactness is maintained at every mutation, not rebuilt:

* `qsa_cache_acquire` / `qsa_cache_acquire_soft` wrappers write the entry on
  every insert (all six qsa.c call sites, prefill and decode);
* `fg_qsa_page_cache_set_evict_hook` (new, `qsa_cache.c`) clears the entry
  when `cache_acquire` recycles a slot, including the all-pinned recycle path;
* `qsa_cache_reset` clears the whole table with the cache.

A page inserted but not mirrored would read as a miss (fallback, safe); an
evicted page not cleared would read as a stale slot (wrong records), which is
why evictions are hooked inside the cache rather than at the call sites.

### 2.2 Resolve kernel

`shaders/fg_qsa_select_resolve.comp`, 256 threads per workgroup, one workgroup
per 256 selected ids (at most two; the decode scan always trims to <= 512):

```
resolved[i] = ids[i] < block_limit ? slot_table[layer_base + ids[i]] : MISS
flags[group] = number of MISS in the group   (shared reduction, no atomics)
```

The dispatch is preceded by `fg_vk_host_write_visible` (new, 8 lines in
`vk.c`): the commit path inserts the current token's page inside the same
batch, so the host-written table needs a HOST -> COMPUTE barrier before the
resolver reads it.

### 2.3 Decode path

`attend_cache` now chooses between three cases:

1. `complete_blocks <= 512`: unchanged identity selection (host lookup loop).
2. resolve enabled (`FG_QSA_SELECT_GPU`, default on) and no locality trace:
   `select_blocks_resolve` runs the same scan (`qsa_select_scan`, shared with
   the host path), records the resolver, ends the batch, and reads only the
   flags. If the miss count is zero it reads the 2 KiB resolved slots and
   calls `fg_qsa_page_cache_touch` per slot, then the gather binds
   `select_resolved`.
3. otherwise (`FG_QSA_SELECT_GPU=0`, locality tracing, or no table): the old
   `select_blocks` host path, unchanged.

The LRU touch matters at depth: the old path refreshed every selected page
through `cache_lookup`, and the resident set is exactly what keeps selections
from missing at 64K-256K. Touching resolved slots reproduces the same LRU
order without the 512 hash lookups. If a miss is detected, the fallback path
reads the ids and runs the authoritative lookup/fetch/re-lookup sequence,
which also stores every selected hit into the table, so the next selection is
device-resolved again.

Counts and order are untouched: `count = tokens/4`, trim loop to <= 512,
sorted by the same `before` comparator. The resolver never reorders.

### 2.4 Flag state (one winner per axis)

| flag | default | effect |
|---|---|---|
| `FG_QSA_SELECT_GPU` | **on** | device slot resolution + host fallback; `=0` = old readback path |
| `FG_QSA_TOPK_V2` | **on** | chunked 64-slot-run network; `=0` = wide bitonic v1 |

`FG_QSA_TOPK_V2` was opt-in from the decode-cuts round with a full parity
oracle and no A/B. It is algorithmically dominant at every decode count
(same 256-thread geometry and shared footprint, fewer compare-exchanges:
1.85x at 1090, 1.24x at 2048, 1.43x at 4096) and exact parity is covered for
counts 1..8200 including 511/512/513, ties, NaN, +-Inf and the multi-group
shapes. There is no third path: the wide network stays only as the `=0`
control, so a regression is a one-env restart or a one-line revert.

## 3. Expected ms/token

Per selection (4K, count ~1024-1090):

| item | before | after | delta |
|---|---:|---:|---:|
| id read 2 KiB D2H | ~0.003 ms | 0 | -0.003 |
| 512 hash + LRU lookups | ~0.07-0.09 ms | 0 | -0.07-0.09 |
| 2 KiB slot write H2D | ~0.002 ms | 0 | -0.002 |
| flags 8 B + slots 2 KiB read + 512 touches | 0 | ~0.01-0.02 ms | +0.015 |
| resolve dispatch | 0 | ~0.003-0.006 ms GPU | +0.005 |
| **host path net** | | | **-0.06-0.08 ms/selection** |
| top-k v1 -> v2 | 0.135 ms/layer | ~0.07-0.10 ms/layer | -0.04-0.06 ms/layer |

The twelve QSA layers per token are spread 1-2 per rank block, so summed over
the eight ranks: **~0.7-1.0 ms/token host + ~0.5-0.7 ms/token top-k = 1.1-1.7
ms/token**, i.e. +0.45-0.65 TPS at the round-9 4K band (22.3-22.8 TPS after
round 8). The estimates are derived from the recorded host-loop share; the A/B
battery is the measurement.

Long context: the resolved-path cost is flat (table read + 512 touches); the
fallback is the old code plus 8 B. The top-k term scales with the candidate
count and the v2 network cuts it 1.4-1.85x per pass, largest at the long
contexts where the three-pass geometry dominates. No new per-context growth is
introduced.

## 4. Long-context correctness argument

1. **Same selection.** Score kernel, reduce loop, comparator, count and
   tie-break are untouched. The resolve pass is a bijection on the gather
   input, not a selection.
2. **Table exactness.** Every cache insert, eviction and reset updates the
   mirror; local tests drive a two-page cache through insert/insert/evict and
   check the mirrored entries, and an end-to-end test checks device-resolved
   slots byte-for-byte against host lookups.
3. **Miss handling.** Any miss takes the previous host path verbatim. Cold
   pages at 64K-256K therefore behave as before, and the fallback repairs the
   table for the next selection. A mis-detected miss is only slower.
4. **Eviction order.** The LRU is refreshed from the resolved slots in
   selection order, matching the old lookup order even though the hash lookups
   are gone; the record cache hit rate at depth is unchanged by construction.
5. **Bounded memory.** 3 MiB rank 0 / 768 KiB per worker at the 256K geometry,
   allocated with the existing record cache and reported by the existing
   memory stats.
6. **Bit-identity gates.** The `[12]`/`[Paris]` fleet gates are the authority;
   this branch adds no arithmetic outside the resolver, which only moves u32
   slot indices.

Remaining long-context cost, measured by round 7/9 and not addressed here:
`index_score` grows linearly with blocks (0.0095 ms/layer per 1024 decode
blocks, i.e. ~0.4-0.6 ms/layer extrapolated at 256K) and the multi-pass
reduce still runs one 4096-wide network per 4096 candidates. At 256K that is
three reduce dispatches per layer. A radix/threshold top-512 would collapse
them to O(candidates) and is the ranked next step.

## 5. Local validation (llvmpipe, correctness only)

```
make all -j8                                        warning-free (-Werror)
tests/test_qsa_prefill                              PASS (both FG_QSA_SELECT_GPU=0
                                                    and =1, both FG_QSA_TOPK_V2=0/=1)
tests/test_owner_reduce                             PASS
tests/test_expert_prefill                           PASS
test_fg_vk focused (FG_QSA_SELECT_GPU=1 and =0):
  topk_reduce_chunked_parity, output_topk, qsa_indexer, qsa_record_gather,
  qsa_attention, qsa_attention_single, qsa_segmented_index_score,
  qsa_prefill_prepare, qsa_prefill_chunk_liveness, generation_topk_selector,
  output_argmax, qsa_quant_and_bf16                 PASS
known pre-existing failures unchanged: test_core:457,
  test_session:254/257, test_prefill_dispatch:178-183,
  llvmpipe qsa_record_commit crash (not run)
```

New coverage in `test_qsa_prefill.c`:

* `select_resolve_mirror`: insert/insert/evict mirror exactness, the kernel's
  slot mapping including an out-of-range id, and the per-group miss counts.
* `select_resolve_parity(...,8191)`: host vs device ids for a 2047-candidate
  window; all-resident case resolves with `misses=0` and slots match host
  lookups; the resolved buffer feeds `fg_vk_qsa_record_gather` and the arena
  matches a host-side gather; one evicted selected page reports `misses=1`
  and hands back the exact host id list.

## 6. A/B plan (orchestrator)

Same ring pack, standard scripts, caches dropped, gates first. Build from
`perf/qsa-selection-offload` @ `eb8a18f`.

1. Control: export `FG_QSA_SELECT_GPU=0 FG_QSA_TOPK_V2=0` on all eight ranks
   (`start-rank0-ring.sh` / `start-workers-ring.sh` with the env exported
   before the script). `correctness64.ps1` -> `[12]`/`[Paris]`; then
   `Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2`.
2. Final: defaults (no env). Same gates and battery. Compare short decode,
   4K warm decode, 4K prefill band.
3. Isolate (optional, one restart each): `FG_QSA_SELECT_GPU=0` (v2 only) and
   `FG_QSA_TOPK_V2=0` (resolve only) to attribute the delta.
4. Context sweep: `tools/context-sweep.ps1` control vs final at 16K/32K/64K
   (and 256K if the session allows); quote decode-leg TPS. This is the
   long-context regression gate: the resolve path must be neutral-to-positive
   at depth, and `QSA_SELECT_TRACE misses=` should stay at the control
   `missing=` level.
5. Traces: `FG_FRAME_TRACE=1 FG_DECODE_PROFILE=1` on the workers; the
   `QSA_SELECT_TRACE` `wait_ms/read_ms/misses` fields and the
   `qsa_state_attention` scope show where the block time went. `FG_QSA_TOPK_V2=0`
   must switch the scope line back to `fg_topk_reduce.spv`; default shows
   `fg_topk_reduce_v2.spv`.
6. Soak: `tools/pi-stability.ps1` (6 stages, 8 ranks, 0 failures).

Expected: short/4K decode +0.4-0.7 TPS, 4K prefill in the existing band,
64K/256K decode leg no worse than control and the resolve-path `misses` not
above the control's `missing`. Rollback = unset both envs (no rebuild) or
revert `4898aea` for the top-k default alone.

## 7. Ranked next steps

1. **Radix/threshold top-512** for `fg_topk_reduce`: O(candidates) instead of
   the multi-pass bitonic, targeting the three reduce dispatches per layer at
   256K. The sortable-key order must reproduce `before` exactly (non-finite
   class first, then score desc, id asc); parity oracle like
   `topk_reduce_chunked_parity`.
2. **`index_score` query reuse** (prefill round 7 ranked item 2): the kernel
   re-loads the query vectors per block dispatch; a B-blocks-per-workgroup
   variant cuts query traffic at depth for both prefill and decode.
3. **Fully fence-free resolve**: only possible if misses are excluded by
   construction (e.g. prefetching selected-adjacent pages), since the gather
   must not be recorded before the miss count is known.
