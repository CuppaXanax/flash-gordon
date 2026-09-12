# To myself, on the other side: 50 → 200+ TPS prefill

You are the post-compaction me. Read this top to bottom before touching anything.
Everything here is measured, not hoped. The fleet is healthy right now; keep it that way.

## 0. TL;DR — the one thing

**The 200 TPS lever is the layer ring nothing else.** Rank 0 currently executes the
common path for all 48 layers; every other improvement is noise until that moves to
the layer owners. The reshard (single-owner experts) bought +5 TPS and the 4th frame;
the contiguous pack is staged plumbing for the ring, not a perf win by itself. If you
do only one thing: implement the ring. Expected result: **~200 TPS at 4K with 4 frames,
250-300 with 6**, versus 50.9 today. Math is in section 3.

## 1. Where everything is

Fleet: 8x BC-250 (192.0.2.42-49), UMA, 15.56 GB RAM each, budget ~14.5 GB safe.
Release dir: `/home/user/flash-gordon-live/20260910-230032-21d65c` on every blade.
Helper scripts (Windows): `%LOCALAPPDATA%\Temp\opencode\`
(`start-rank0-clean.sh`, `start-workers-clean.sh`, `quiesce.sh`, `fg-wait-listen.sh`,
`correctness64.ps1`, `fg-swap-contig.sh`, `fg-pack-fix` lives on .42).
Battery: `D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 1 -LogRunDir <dir>`.
Fleet tool: `D:\workspace\bc-250-dbg\Invoke-BC250Fleet.ps1` (creds in its `.env`).

**Live pack (running now):** `/home/user/flash-gordon-q38-cooked` — single-owner
`l%8` layout, 4 frames. Measured 4K **48.9-50.9 TPS**, 128 prefill 21-25, short decode
9.9, 4K decode 7.1, gates 12/Paris. Recovery copies: `/home/user/flash-gordon-q38-single`
on .42 and each worker's own file in its `q38-single` dir. Manifests backed up in
`q38-cooked` as `.bak-singleowner` and `.bak-contig2-failed`.

**Staged pack (ring layout, NOT distributed):** `/home/user/flash-gordon-q38-contig2`
on .42 only. Rebalanced contiguous ownership map
(`/home/user/contiguous2.expert-map`):
- blocks: 0-5→rank1, 6-11→rank0, 12-17→rank2, 18-23→rank3, 24-29→rank4, 30-35→rank5,
  36-41→rank6, 42-47→rank7; all 512 experts per layer on the block owner; `token_embd`
  on rank7 (pack special case).
- Rank 0's replicated arena in this pack = 14,943,436,800 B — the exact size that has
  booted before. Rank 7 persistent 10.382 GiB (embedding), rank 4 9.906, all under cap.
- To deploy: upload `rank-01..07.fgw` from .42 to their blades (Windows relay, ~3 min
  each, **verify sizes; the relay has silently corrupted a shard once** — the runtime's
  per-tensor SHA check catches it at worker startup), upload `manifest.fgm` to all 8,
  run `fg-swap-contig.sh` (change its `contig=` path to `q38-contig2`), then restart
  rank 0 first, wait for `ss -ltn` to show 19100/19101, then workers.

Disk on .42: 179 GB free. Do **not** delete `/home/user/flash-gordon-pipeline-candidate`
(131 GB, the ceiling harness's protected LKG pack). `q38-single` (104 GB) stays until
contig2 has served production for a while.

## 2. Why perf is flat today

Rank 0's coordinator loads **all** common tensors (5,506,252,800 B) plus its own six
layers' expert shards and executes all 48 layers' common path on its 24 CUs. Ownership
maps only change where weights live, not who computes. Measured per-128-token-stage
budget (rank 0, Sept 3 profile): experts 252 ms, GDN projections 153 ms, f32 gr 67 ms,
router 36 ms, QSA 28 ms, stage wall ~650 ms, GPU duty 88-94%. Four frames hide the
collect; the GPU sum is the wall. At 50 TPS the completion interval is 2.53 s per
128-token chunk, consistent with rank 0 serialising ~0.6 s of common work per chunk.

## 3. The ring: design and math

For each chunk, the token batch flows through **contiguous layer blocks**:
rank1 (0-5) → rank0 (6-11) → rank2 (12-17) → rank3 → rank4 → rank5 → rank6 →
rank7 (42-47) → rank0 (output projection). Each rank executes its six layers
end-to-end with local experts; the only cross-rank traffic is the hyper state at block
boundaries: 128 tokens x 10240 x 4 B = 5.24 MB per hop, 8 hops per chunk ≈ 42 MB.
Measured bulk rate ~114-150 MB/s → ~0.3 s per chunk transfer ceiling ~420-450 TPS.

Compute per rank per chunk after distribution: experts ~252 ms + common for six layers
~42 ms ≈ 300 ms. Chain latency for one chunk ≈ 8 x 330 ms ≈ 2.6 s; with 4 frames in
flight completion interval ≈ 0.65 s → **~200 TPS**; 6 frames → **~300 TPS**. Frames
must be issued without blocking on results (section 4C).

Chunk-coverage constant (why blocks are contiguous): with `l%8` ownership a token hops
every layer — 48 x 5.24 MB per chunk ≈ 250 MB, ~60 TPS ceiling. Contiguous blocks are
non-negotiable. Per-layer expert bytes measured (`EXPERT_LAYER_SUMS` in the probe):
44 layers 1,572,864,000 B; layers 2/30/46/47 1,835,008,000 B; layer 3 2,044,723,200 B.
Cheapest six-layer blocks are 6-11, 12-17, 18-23, 24-29, 36-41 at 9,437,184,000 B.
That is why rank 0 owns 6-11 in contig2 (its arena then equals the proven-boot size).

## 4. Implementation plan (in order)

**A. Protocol — DONE, do not rewrite.** `FG_MSG_PREFILL_LAYER_WORK/RESULT` (enum 15/16),
`fg_prefill_layer_work_encode/decode` and `fg_prefill_layer_result_encode/decode` exist
and are tested (`tests/test_core.c` round-trip, `tests/test_fabric.c:136-137` even has a
chain-hop test: rank r receives from r-1, runs layer r, forwards to r+1, rank 7 returns
to rank 0). Message carries layer, source/destination, flags, first_token, token_count,
positions (3*N u32), hyper (N*10240 f32), optional ngram embeddings
(`FG_LAYER_WORK_HAS_NGRAM`). 5.24 MB at N=128.

**B. Worker: owner executor + layer handler.**
- `fg_rank_main` (`src/runtime.c:1160`) currently creates `fg_model` (non-replicated),
  an expert executor, and for rank 4 an output executor. Add an owner executor:
  `fg_owner_executor_create(&owner, model, err)` — it already owns exactly this rank's
  six layers (common + experts live in its `rank-XX.fgw`). Keep the existing expert
  executor; it is the local expert path.
- `rank_worker_loop` (`src/runtime.c:1149`) handles bulk messages at ~1156. Add:
  on `FG_MSG_PREFILL_LAYER_WORK`, decode with a reusable buffer sized
  `FG_PREFILL_LAYER_WORK_MAX_BYTES`, write hyper+positions into the owner's prefill
  input path, run `fg_owner_prefill_layer_begin/finish` for each layer from
  `work.layer` through the end of this rank's block, with a **local** expert
  fire/collect pair built from `fg_expert_prefill_enqueue` / `fg_expert_prefill_finish`
  (see `src/runtime.c:506-513` for the self-route pattern rank 0 already uses), then
  encode a result and forward it as the next block's work if another block follows,
  or as `FG_MSG_PREFILL_LAYER_RESULT` to rank 0 if this was the last block.
  Reuse the receive buffer for the next hop encode to avoid per-hop malloc.
- Layer 1 ngram: layer 1 lives in rank1's block. Rank 0 does the ngram lookup for the
  chunk (it owns the ngram store) and attaches embeddings to the first hop with
  `FG_LAYER_WORK_HAS_NGRAM`. Do not build ngram stores on workers.

**C. Coordinator: async chain dispatch.**
- Replace the per-layer `fg_owner_prefill_layer_begin/finish` calls in
  `coordinator_prefill_pipeline` (`src/runtime.c`, function near line 2018) for
  non-rank0 blocks with: begin = embed (rank 0) + ngram lookup + send work for the
  first layer of block 0 (rank1); finish = receive the result from the last block.
- **Issue frames without waiting**: keep the existing four frame buffers and loop, but
  sends must not block on the previous frame's completion. Use the fabric's direct
  send (or a small send queue) and collect results asynchronously by frame sequence
  (`fg_frame_sequence`). A simple v1: per outer iteration, send all `FG_PREFILL_FRAMES`
  chunks' first hops, then drain `FG_PREFILL_FRAMES` results in order before publishing
  QSA pages. That alone gives the chain overlap.
- Keep the local path for rank 0's own block (6-11) exactly as today.
- Preserve `coordinator_publish_qsa_pages` ordering per chunk.

**D. QSA state for all ranks (the hard part; blocks C).**
Every block contains one or two QSA layers (3,7,11,...47), so each rank's owner
executor needs a QSA session for its layers. Today only rank 0 runs QSA layers and the
page transport feeds two replica owners (ranks 3/7): `qsa_page_transport`,
`qsa_owner_index` (`src/runtime.c:1217`), `qsa_replica_*`. Extend to all ranks:
- On each rank, open a QSA session mirror like the coordinator does
  (`fg_owner_qsa_open_mirror`, runtime.c:2290 area) for its own QSA layers.
- Generalize the page fan-out: each rank needs pages only for its own QSA layers
  (~1-2 layers x 32 blocks x 3.6 KB ≈ 230 KB per chunk ≈ 3 MB total per chunk) —
  cheap. Watch the two backpressure rings: replica depth 64 (fine), owner writer
  depth must stay **8** (raising it thrashed I/O and regressed to 3.4 TPS once).
- Ordering: a rank may not run a QSA layer beyond its committed frontier. Commit
  pages per chunk after the layers that generated them.
- Gate before enabling: correctness 12/Paris and no decode regression. If QSA state
  proves too large per rank (index ~411 MiB each; residency margins are 1.8 GB on
  the tightest ranks), implement section 3's chain for GDN layers first (star is not
  worth it: 2 transfers per layer ≈ 40 TPS ceiling) and keep QSA layers local to
  rank 0 until D lands.

**E. Frames.** After C+D, re-test 5-8 frames. Keep `FG_PREFILL_FRAMES=4`,
`FG_OWNER_SLOT_COUNT=4`, replica depth 64 as the known-good baseline.

## 5. Memory budget (UMA)

Rank 0's replicated arena = shared (all common) + own experts. The split-arena commit
(`fc4fc4a`) allocates them as two buffers; the probe (`a0b64c2`) prints both sizes
(`REPLICATED_PROBE`) and per-layer expert sums (`EXPERT_LAYER_SUMS`). Known values:
shared 5,506,252,800 B; rank 0 experts must be <= ~9,437,184,000 B for a ~14.94 GB
total (boots by ~30 MB). The ring removes the shared-all-layers term entirely: rank 0
will load only its own block's common tensors + experts (~2-3 GB), which is the real
reason the ring is also the memory fix. Until then, any pack for rank 0 must use a
cheapest six-layer block.

## 6. Verification protocol

Every change: (1) build clean `make flash-gordon` in WSL; (2) deploy via the standard
chain (quiesce, upload patch, build on .42, distribute `runtime-fix.tar.gz`, extract);
(3) restart **rank 0 first**, wait for `ss -ltn | grep 191`, then workers; (4) run
`correctness64.ps1` (12 / Paris, >= 64 max tokens); (5) run the attach battery and
report 128/4K/short-decode like-for-like. Never claim perf without (5). Never enable an
unverified pack swap without a hash or a boot-before-swap check. Decode must never
regress: ring work is prefill-only until decode gets its own plan.

## 7. Operational pitfalls (learned the hard way)

- Start order: rank 0 first, wait for its fabric listener, then workers. A worker that
  connects twice kills rank 0 with `duplicate rank N channel 0`.
- Transfers: the Windows relay corrupted one 10 GB shard at identical size. The runtime
  verifies per-tensor SHA at worker startup; if it fails, re-transfer that file.
- `flash-gordon pack` refuses a non-empty output dir. Pack to a fresh dir.
- `common_owner` must use `m->layer_owner[layer]` (fixed in `9383009`); `token_embd`
  is placed on rank 7 for cap balance.
- Profiling mode (`FG_PREFILL_PROFILE`/frame trace) hung the fleet once and is parked.
  Do not restart the fleet into profiling mode without a soak test.
- Disk: delete a pack only after confirming nothing references it; `pipeline-candidate`
  is the harness's; keep one recovery pack (`q38-single`) while trying a new layout.
- The 16 GB "blade" is 15.56 GB system RAM minus ~0.6 GB OS; the driver accepts ~15.5 GB
  total but a single allocation above ~15.5 GB fails with Vulkan result -2.

## 8. First actions, in order

1. Deploy contig2 (section 1) and verify the fleet boots at ~49-51 TPS on contiguous
   ownership with gates green. This is the ring's foundation.
2. Implement B (worker owner executor + layer handler) with a one-chunk round trip to
   the next block and back, under a flag, and gate on 12/Paris.
3. Implement C (async chain for all blocks, frames issued without waiting).
   Gate: 4K >= 100 TPS.
4. Implement D (QSA fan-out). Gate: 4K >= 150 TPS, decode unchanged.
5. Tune frames (5-6) and re-measure. Target: **200-300 TPS at 4K**, decode >= 9.9 short
   until MTP is unfrozen.

The user's goal, in their words: Qwen running at interactive coding speeds, Frontier-
class intelligence on their own hardware. The ring is the hill. Climb it.
