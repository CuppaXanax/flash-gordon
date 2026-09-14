# QSA record-cache capacity: long-context prefill/decode root cause and fix

Date: 2026-09-14
Branch: `fix/qsa-cache-capacity` (base `f224fe6`)
Commits: `5e9943d` (cache), `8b79a4d` (lifecycle)

## Symptom

Context sweep on the ring pack: 4K prefill 280.8 TPS, 16K 312.4, 32K 309.2,
then the 65536 request returned HTTP 400
`{"error":{"message":"QSA cache has no evictable slot","type":"invalid_request_error"}}`
after 182.8 s (~54K of ~65.5K prompt tokens). Rank 0 logged
`io_uring operation: Connection reset by peer` mid-chunk; workers ended with
`stale or misrouted prefill layer work` / `peer closed fabric socket`.

## Root cause

The ring pack gives rank 0 a contiguous six-layer block (contig2/contig3 owner
map: rank 0 owns layers 6-11, i.e. QSA layers 7 and 11). Rank 0 executes its own
block inline during ring prefill and ring decode, through the coordinator QSA
session opened as a **stateless mirror**.

Pin lifecycle on that session:

1. `commit_prefill_records` acquires and pins every page it commits
   (`src/qsa.c:975`, pin call) so the records survive until publication.
2. The only release path is `persist_prefill_state`, called at the end of
   `fg_qsa_session_prefill` (`src/qsa.c:1543`) and recursively
   (`src/qsa.c:1460/1484/1497` -> `fg_qsa_session_page_published` ->
   `fg_qsa_page_cache_unpin`).
3. `persist_prefill_state` starts with
   `if(!s->cache||!s->state)return FG_OK;` (`src/qsa.c:1422`). The coordinator
   mirror has **no state file**, so it returns without writing or unpinning.
4. The ring-mode publication path that could have released pins is disabled:
   `coordinator_publish_qsa_pages` skips self-owned pages
   (`src/runtime.c:2481`) and skips every remote owner while ring prefill is
   active (`src/runtime.c:2484`).

Net effect: every page rank 0 computes for layers 7 and 11 stays pinned for the
life of the session. When the page cache is full and every slot is pinned,
`fg_qsa_page_cache_acquire` returned
`FG_ERR_LIMIT "QSA cache has no evictable slot"` (`src/qsa_cache.c`, old line
139), which propagated to the API as a 400.

## Why it dies at ~54K tokens

Record page size is 4944 B (`FG_QSA_PAGE_RECORD_BYTES` = 4 x 1236 B token
record). A 128 MiB mirror holds 134217728 / 4944 = **27147 pages**.

Rank 0 owns two QSA layers, and each 4-token block pins one page per layer, so
pins grow at 2 per 4 tokens (0.5 pins/token). The cache is exhausted at
27147 / 0.5 = **54294 tokens**, inside the 65536 request - matching the observed
~54K. The earlier rows never reached the wall because each request cold-resets
the session, clearing pins:

| request | pinned at end | of 27147 |
| --- | ---: | ---: |
| 4K | 2048 | 7.5% |
| 16K | 8192 | 30% |
| 32K | 16384 | 60% |
| 64K | 54294 (death) | 100% |

Warm mirror inserts are unpinned and evict normally; they merely fill whatever
slots the pins have not consumed. The prefill never failed because of record
volume - 32 pages per layer per 128-token chunk is trivial - but because the
pins were never released.

## Fix

### `5e9943d` qsa-cache: soft capacity acquisitions and pinned accounting

- `fg_qsa_page_cache_acquire` no longer returns `FG_ERR_LIMIT`. When every slot
  is pinned it recycles the oldest pinned page (clearing its pin) and reports
  the acquisition as a miss. A cache-capacity condition can no longer fail a
  request.
- `fg_qsa_page_cache_acquire_soft` reports the same condition as a soft miss
  (`*slot = UINT32_MAX`, `*hit = false`) without evicting pinned data; the
  optional insertion sites (`fg_qsa_session_warm_pages`, tile/gather cold-page
  inserts in `src/qsa.c:1233/1344/1599`) use it and stream the page from their
  staging buffer instead of caching it.
- `fg_qsa_page_cache_pinned_count` exposes the pin window to tests and traces.

### `8b79a4d` qsa: state-backed coordinator session keeps rank-0 pages evictable

- New `fg_qsa_session_open_state_mirror_with_scratch`
  (`src/qsa.c`, `include/fg_qsa.h`) and `fg_owner_qsa_open_state_mirror`
  (`src/owner.c`, `include/fg_owner.h`): a session that keeps a state file for
  the layers this rank executes while still mirroring remote-owned pages through
  the fetch callback.
- `coordinator_open_qsa` (`src/runtime.c:4016`) creates
  `<model-dir>/qsa-owner-rank-00.state` and opens the coordinator session over
  it **only when rank 0 owns at least one QSA layer**; deployments that give
  rank 0 no QSA layer keep the stateless mirror.
- `coordinator_fetch_qsa_pages` owner==0 (`src/runtime.c:2545`) prefers the
  record cache and falls back to `fg_qsa_session_state_records` (the existing
  state-file slow path) when a self-owned page was evicted.
- `persist_prefill_state` now flushes a lagging decode range in bounded
  512-block batches (`src/qsa.c:1433`) instead of returning `FG_ERR_LIMIT`
  when the lag exceeds staging capacity.

With this, prefill pins are released by the per-chunk persist exactly as they
already are on worker ranks; the mirror cache becomes a true cache whose misses
are served from the rank's own state file. Numerics are unchanged: commit,
attention, selection and persist write/read the same records; only when pins are
released changed.

## Local evidence

- `make all -j8`: clean, `-Werror`.
- `tests/test_qsa_prefill`: PASS, including two new cases:
  - all-pinned commit recycles the oldest page and returns `FG_OK`;
  - `prefill_persist_lifecycle`: without a state file a 64-token chunk leaves
    16 pages pinned (the old leak); with a state file 24 consecutive chunks
    (1536 blocks through a 32-page cache) persist to the state file and return
    the pin count to 0 every chunk.
- `tests/test_core`: new soft/recycle/pinned-count assertions pass; only the
  known pre-existing replica-depth failure at line 457 remains.
- Focused `tests/test_fg_vk`: `qsa_resident_causal_batch_attention`,
  `qsa_resident_t1_compat`, `qsa_resident_hierarchical_topk` PASS.
- `test_prefix`, `test_chat`, `test_chat_runtime`, `test_api`,
  `test_ngram_deployment` PASS. `test_session` and `test_prefill_dispatch` fail
  identically on pristine `main` (pre-existing environment failures).
- No fleet run yet: the orchestrator holds the fleet for its A/B.

## Expected behavior after the fix

- 65536: prefill completes (was dying at 54,294 tokens); decode reads rank-0
  pages from the state file only when the mirror evicted them. Prefill TPS is
  expected within noise: the added work is 2 layers x 32 state pages (256 KiB)
  per 128-token chunk, the same per-chunk persist pattern the six worker ranks
  already pay.
- 131072: rank-0 working set for its two layers is 32768 blocks (324 MiB),
  larger than a 128 MiB mirror, so eviction and state fetch are exercised
  during decode. Prefill is unaffected.
- 262144: rank-0 writes ~1 GiB of records to `qsa-owner-rank-00.state` during
  the prefill (full file is 12 layers x 65536 blocks x 8 KiB = 6.0 GiB,
  preallocated at open). Decode correctness is expected; decode TPS is bounded
  by how many of the top-512 selected pages per layer miss the mirror and are
  read from disk. This is the same trade-off workers already have with their
  4096 pages/layer cache.

## Fleet validation commands

Deploy the branch to a separately named install first, then attach:

```powershell
# targeted repro of the failure
pwsh -NoProfile -File tools\context-sweep.ps1 -Contexts 65536

# decode-regression points and the full native-context curve
pwsh -NoProfile -File tools\context-sweep.ps1 -Contexts 16384,32768
pwsh -NoProfile -File tools\context-sweep.ps1

# 4K prefill band and correctness battery (two 4K runs)
D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2

# gates and stability (12/Paris correctness probes inside)
pwsh -NoProfile -File tools\pi-stability.ps1
```

Acceptance: sweep completes all six targets; 16K/32K decode back toward
18-19 / 17 TPS; 4K/16K/32K prefill stays >= 270 TPS; battery and pi-stability
PASS; gates [12]/[Paris].

Operational notes for the deploy:

- Rank 0 now needs write access to the model directory and ~6 GiB free for
  `qsa-owner-rank-00.state` (preallocated on open; wipe it with the other QSA
  state files between runs).
- If rank 0 is given no QSA layer by the owner map, it still opens the
  stateless mirror and behaviour is identical to before.

## Follow-up 2026-09-14: decode regression from the state-backed session

Fleet sweep on the fixed build (binary 6dee7282): prefill 4K 259.0 / 16K 286.6 /
32K 282.2 / 64K 266.6 / 128K 243.2 TPS (the 54K wall is gone), but decode
(32 tokens) fell to 4K 20.4 / 16K 10.9 / 32K 9.7 / 64K 8.1 / 128K 7.7. The
previous leaky build decoded at ~18.8 (16K) and ~17.1 (32K) only because leaked
pins kept rank-0 pages resident.

Root cause of the per-token cost (two compounding paths):

1. **Mirror warm traffic evicted rank 0's own pages.** Ring prefill still issued
   `coordinator_warm_qsa_issue` for every worker-owned QSA layer, inserting
   10 layers x 32 pages per chunk (~1.5 MB/chunk over the fabric) into the
   128 MiB mirror. Before this branch those inserts competed with *pinned*
   rank-0 pages and could not evict them; after the fix rank-0 pages are
   unpinned, so the warm stream pushed them out. Prefill attention then
   re-read blocks it had just written (mid-prefill state reads), and decode
   found 16K/32K selected pages missing even though the rank-0 working set
   (40/80 MiB) fits the mirror.
2. **The owner==0 fetch fallback was one synchronous pread per page.**
   `coordinator_fetch_qsa_pages` served rank-0-owned misses with a per-block
   `fg_owner_qsa_state_records` loop; at 16K a token can miss several hundred
   of its top-512 selected pages per layer, i.e. hundreds of 8 KiB io_uring
   round trips (~30-50 us each) per token.

Fixes:

- `coordinator_warm_qsa_issue` returns immediately when `ring_decode` is
  active (runtime.c). Ring decode executes every QSA layer on its owner, so the
  rank-0 mirror's remote copies are never read; the mirror now holds only
  rank 0's own pages. Warm stays enabled for ring prefill + legacy decode.
- New `fg_qsa_session_state_records_batch` / `fg_owner_qsa_state_records_batch`
  (qsa.c, owner.c) issue one `fg_qsa_state_read_blocks` submission for the
  whole fetch list; `coordinator_fetch_qsa_pages` owner==0 uses it directly.
  The caller has already checked the record cache, so nothing is re-fetched.

Expected deltas (single-run fleet variance applies):

- 16K/32K decode: rank-0 working set (8192/16384 pages) is fully resident again
  with warm off, so the per-token state reads disappear; expect a return to the
  ~18-19 / ~17 TPS band.
- 64K/128K decode: rank-0 pages exceed the mirror (32768/65536 needed), so
  selected misses still read the state file, but batched: one uring submission
  per layer per fetch instead of one per page. Expect a large cut in the
  per-token I/O stall; the floor is the NVMe read of up to ~5 MB/token at 128K.
- Prefill: warm removal deletes 10 fabric fetches and 320 cache inserts per
  128-token chunk from the rank-0 critical path and stops the mid-prefill
  eviction/re-read of rank-0 pages; expect recovery toward the 280-310 band.
  The persist cost added by the state-backed session is ~4 io_uring submissions
  per chunk (<0.2% at 4K), so the 4K 259 vs 280 delta is read as single-run
  variance rather than a structural cost.
- 262K correctness: unchanged; pins are still released by persist, the state
  file remains authoritative for evicted pages, and cache pressure is still a
  soft miss. Decode beyond the mirror's capacity is disk-bound by design.

## Known limits / follow-ups

- Decode pins are still held until the next prefill persist (or session reset).
  For generations longer than the mirror can hold, `recycle` may evict an
  unpersisted page and the next continuation flush can then fail. Bounding the
  decode pin window (write-through on block completion) is the next lifecycle
  step; the 32-token decode legs of the sweep do not hit it.
- The recycle path in `fg_qsa_page_cache_acquire` is a safety net, not a
  correctness mechanism: under the configured caches (4096 pages/layer on
  workers, 27147 on the mirror) a single chunk cannot pin more pages than the
  cache holds.
- `coordinator_publish_qsa_pages` still skips self-owned pages in ring mode;
  that is now consistent because the state file is the durable copy.
