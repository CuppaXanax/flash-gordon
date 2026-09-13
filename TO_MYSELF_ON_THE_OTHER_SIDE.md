# To myself, on the other side: 50 → 200+ TPS prefill

You are the post-compaction me. Read this top to bottom before touching anything.
Everything here is measured, not hoped. The fleet is healthy right now; keep it that way.

## 0. RING STATUS 2026-09-13 NIGHT (commit series 2125ed0..652d59d — READ FIRST)

**The ring answers correctly now.** `correctness64.ps1` on the ring pack returns
`answer=[12]` and `answer=[Paris]`, short decode 9.5-10.0 TPS, 4K battery
prefill **243.7-254.7 TPS** (4322 tokens), 4K decode 2.06 TPS.
Binary hash `81105d0d...`, fleet on `/home/user/fg-ring-pack`.

**Root cause of the empty answers was decode state, not prefill numerics.**
With `FG_NUMERICS_TRACE=1` every prefill layer's hyper digest in the ring was
byte-identical to the single-owner reference (through layer 47, first sampled
token matched). The first decode step then diverged: ring prefill advances the
GDN conv/recurrent and PLE conv state on the block owners, but rank 0's decode
replay used its own unadvanced state for every layer it did not execute. Fixed
in `652d59d` with `FG_MSG_GDN_STATE_FETCH/RESULT` (protocol enum 42/43): owners
serve one layer's conv+recurrent (~3 MiB) plus PLE state for layer 1, rank 0
pipelines the fetches and writes them into its executor before decode; ring
requests force a cold reset (prefix reuse disabled until state push-back
exists). Cost: ~115 MB + ~36 messages per cold request (~1-2 s prefill; the
4K battery went 254.7 -> 243.7 TPS).

**Kernel rounds (commits 2125ed0/562b7ad/2c69599, 98ff525/2a2b693, 652d59d,
cc299df).** Per-kernel at 4K after round 2 (FG_PREFILL_LAYER_PROFILE, ranks
3/5): topk 31.7 -> 1.9 ms, QSA attention split_batch 35.1 -> 20.3 ms, moe
gate_up 29-31 -> 21.2-22.3, moe down 9-12 -> 6.4-10.1, QSA layer 110 -> 77-80,
GDN layer 58 -> 46-49, six-layer stage GPU 455 -> 343 ms. Batch-1 decode got a
fused gate/up/SwiGLU + down/reduce pair (cc299df, 2 dispatches/layer instead of
5; short decode 9.84 -> 10.00). Decode experts are still instruction-bound;
QSA decode at 4K is dominated by mirror record-cache thrash (32 MiB cache vs a
12-layer x 1024-block working set).

**Next decode lever:** allocate the mirror's QSA index segments lazily (segment
1 is 204 MB and unused below 131072 tokens) and use the freed rank-0 headroom
for a larger `--qsa-page-cache-mib` (128 MiB holds a 4K context's 12288 pages),
which should take 4K decode from 2.06 toward short-decode speed. After that,
ring decode (per-token chain with 40 KB hops, owners using their own QSA/GDN
state) is the remaining architecture piece.

## 0z. UPDATE 2026-09-13 LATE-LATE (binary 0dd3deda, ring pack)

**Blank short answers fixed.** The API never set `think_mode`, so `/no_think`
was plain text; the model spent ~29 tokens thinking, the stripper moved it to
reasoning, and content was empty while finish=stop. `src/api.c` now detects a
leading `/no_think` in user messages, strips the directive, sets THINK_OFF,
and parses generation as non-thinking. Measured: max_tokens 1/2/4 return
`1`/`12`/`12`; gates [12]/[Paris]; battery 4K prefill 237.6 TPS, short decode
10.72, 4K single decode 8.18.

Also integrated since 0a: the ring-decode architecture (`FG_DECODE_RING=1`,
default off; correct per gates but no speedup yet - the chain serializes the
same compute the legacy path overlapped with the expert wait; 8 ranks would
need ~2x faster expert kernels to clear 15 TPS; 4K ring request killed rank0
once and the transport is not reusable after a second ring request - both open)
and the vector-tiled batch-1 expert pair (short decode 10.0 -> 10.7, covers
raw-Q8_0 downs so layers 2/4/30/46/47 use the 2-dispatch path without a repack).
Repack plan for q8 experts is documented in the edec agent report: source shards
live on the pack producer host, not this machine.

## 0a. FINAL NUMBERS 2026-09-13 LATE (binary 7e47ddae, ring pack)

Validated on my own deploy, gates then battery, back to back:

| metric | mission start | final |
|---|---|---|
| gates 12/Paris | empty answers | **[12] / [Paris]** |
| 4K battery prefill | 48.81 / 47.4 | **236.2 TPS** (band 234-257) |
| short decode (32 tok) | 9.20 | **10.06 TPS** |
| 4K single decode row | 1.96 | **7.78 fresh proc / 9.5-9.8 warm** |
| 4K sustained decode (32 tok) | ~2 | **9.08-9.14 TPS** |

Same-night workstreams: kernel rounds 1-2 (prefill 150->255), numerics root
cause + GDN/PLE state handoff (`652d59d`), fused batch-1 expert pair
(`cc299df`), lazy index segments + 128 MiB mirror cache (`170f8f7`), overlapped
mirror warming during ring prefill (`9233146`), QSA decode split attention +
bounded top-k + descriptor reuse (`b9329ea`, `87e34ce`).

Remaining, in priority order:
1. **Short decode is ~10 TPS** (~100 ms/token): rank-0 GPU 39 ms, remote-expert
   wait 35 ms (worker expert kernels at 0.6 ms/layer, ~50 GB/s for cooked
   q5_1 - a pack re-cook or a better 4-bit kernel is needed), CPU/submission
   ~20 ms. Rank-0 common path hot spots: gdn projection/recurrent 12 ms, gr
   reads 9.4 ms, shared 4.2 ms, router 2.1 ms. Successive decode rounds took
   short 9.2 -> 10.1 and 4K 2.06 -> 9.1; reaching 20+ short needs the worker
   expert format/kernels or a ring decode architecture (token walks the 8
   blocks with 40 KB hops; owners use their own QSA/GDN state - no mirror).
2. **Layers 2/4/30/46/47 take the slow non-fused expert path** (0.97 vs 0.60
   ms) - needs a repack to make them fusable.
3. First 4K decode after a fresh process pays a ~90-100 ms lazy pipeline-compile
   for the new split/merge kernels; prewarm pipelines at session open.
4. A custom 4K prompt reportedly crashed rank 0 during ring prefill in the
   decode agent's testing (not reproduced here: a 2.2K and a 4K custom prompt
   both completed, rank 0 alive). If it reappears, capture the exact prompt and
   the rank-0 tail.
5. Prefix reuse stays disabled under ring until GDN/PLE state push-back exists
   (correctness > reuse).
6. rank 0 lives on ~25-90 MB available; drop_caches before heavy phases.

## 0b. QSA RING STATUS 2026-09-13 (commit 1dc1bc0 — historical)

**The QSA-in-block cost was the page cache never being created on state-backed
worker sessions.** `fg_qsa_session_open_state` only made the page cache when
`state_path==NULL` (the coordinator mirror), so every prefill token on a worker
ran `fg_qsa_state_write_block` + `select_blocks` + `fg_qsa_state_read_blocks`
with GPU syncs in between. Fixed in `1dc1bc0`:

- bounded worker record cache (`FG_QSA_WORKER_CACHE_PAGES=4096/layer`) for
  state sessions, gather misses served from the authoritative state file
  (`state_fetch_pages`);
- completed pages persist once per layer with a batched uring write
  (`persist_prefill_state`), not once per token.

Measured with the new `PREFILL_BLOCK_LAYER` / `QSA_PREFILL_TRACE` lines:

| point | before | after |
|---|---|---|
| QSA layer @1.3K | 260-430 ms | 84-100 ms |
| GDN layer @1.3K | 62-100 ms | unchanged |
| rank1/3/5 block @1.3K | 676/797/873 ms | ~420-480 ms |
| 4K battery prefill, depth 4 | 48.81 TPS | **95.14 TPS** |
| 4K battery prefill, depth 8 | n/a | **150.25-152.17 TPS** |
| 4K decode / short decode | 1.96 / 9.20 | 2.03-2.07 / 9.42-9.71 |

Gates: both correctness requests complete, decode >= 9 (answers still empty —
the distributed-prefill numerics defect in section 8 is unchanged and is still
the merge blocker; this work is speed-only).

**Depth 8 details.** `FG_PREFILL_FRAMES=8`. Worker owners allocate only owner
slot 0 (`create_decode_slots`), the coordinator processes its own block inline
on the shared base slot, and the final-result tensor is ping-ponged; rank 0's
Vulkan budget is therefore unchanged from depth 4. 128 prefill 21.4-22.0 TPS.
Deploy hash at this state: `a7f4134250c3dac5becdc0662baa42c23191ebe19a1f0ac5386f0f6a174501b4`.

**Why it is not 200 yet (measured, not guessed).** At 4K the chain still runs
in ~2.4 s / ~1.7 s waves; effective in-flight depth is ~6 of 8, and rank-0's
own inline block measures **518 ms mean (max 620 ms)**, which alone caps the
ring at ~171 TPS even with a perfect pipeline. QSA select is now the growth
term: `select_ms` 29.8 ms/layer at first=1024 (all blocks selected, fast path)
growing to 88-105 ms/layer at 4K where the top-512-of-1024 selection runs.

**Next levers, in order (each needs a battery):**
1. Pipeline rank-0's own block: use `fg_fabric_wait_ready` polling to
   interleave owner `begin/finish` with the message loop (the non-ring
   `coordinator_prefill_pipeline` already models begin/finish in flight).
   Expected 170-200 TPS because the serial own-block cap disappears.
2. Cut `select_prefill_tile`: 1.26 s of chain latency across the 14 QSA
   layers at 4K. Keeping selection ids on the GPU (block==cache-slot when the
   cache covers the context) removes the host read+fence per tile.
3. More depth only after socket buffers grow. `FG_PREFILL_FRAMES=12`
   regressed to **26.87 TPS** with 16 MiB SO_*BUF; the extra hops exceed the
   buffers and the chain locks step. Raise `socket_configure` (fabric.c) and
   the sysctl caps together, then re-test 10-12.

**Per-kernel GPU budget 2026-09-13 (new `PREFILL_LAYER_PROFILE` /
`PREFILL_LAYER_KERNEL`, FG_PREFILL_PROFILE=1 on workers, 4K battery, ranks
1/3/5). Kernel rewrite is authorized; this is the hit list.**

GDN layer ~58ms GPU: expert grouped GEMMs 40ms
(`fg_moe_kquant_cooked_grouped` 29-31 + `fg_moe_q5_1_down_cooked_grouped`
9-12), `dense_q8_0_cooked_tile` 2.8, `gdn_prefill_recurrence` 3.0, all other
kernels <1ms each.
QSA layer ~110ms GPU: `fg_qsa_attention_split` 35.1, `fg_topk_reduce` 31.7
(slow path only: 32 queries tiles/fences per layer), experts 37,
`qsa_index_score` 1.3, `qsa_record_gather` 1.6, `qsa_attention_merge` 0.7.

Per six-layer stage: GPU 455ms of ~650ms wall. Experts are 51% of GPU; QSA
attention+topk 29%. Utilization: expert GEMMs ~28% of the 1.47 TFLOPS prefill
primitive; QSA prefill attention ~6% (it is a batch-1 decode kernel called
per query). llama.cpp on a same-lineage 35B MoE (2 blades) is the control that
this silicon can do ~450 prefill TPS with real kernels.

Rewrite priority: (1) expert grouped GEMM (kquant gate_up + down) — biggest
term on every layer; (2) one batched QSA prefill attention dispatch for all
128 queries (replace split+merge per query) plus fused batched top-k with ids
left on GPU; (3) `dense_q8_0_cooked_tile` tiling. Target after (1)+(2):
six-layer stage ~350ms → pipeline bound ~350+ TPS.

**Dead ends already paid for (do not repeat):**
- `FG_PREFILL_FRAMES=12` at 16 MiB buffers: 26.87 TPS, wave stalls.
- Depth 8 with 8 owner slots + 8 ring-output tensors: rank-0 OOM within the
  first 4K request (33 s first chunk, then fleet down). Slot-0 sharing +
  ping-pong outputs fixed it.
- Query tile 4 -> 8 (halve selection fences): select got *worse*
  (88 -> 105 ms/layer), cost +28 MiB on rank 0 and tripped the frozen pack's
  scratch ledger (64 MiB alignment) in `fg_q38_runtime_scratch_bytes`; the
  manifest cannot be repacked cheaply. Reverted.

**KERNEL ROUNDS 1-2 (2026-09-13, commits 2125ed0/562b7ad/2c69599 + 98ff525/2a2b693):
4K battery prefill 150.2 -> 169.3 -> 254.7 TPS, short decode 9.84, 4K decode
2.06.** Per-kernel at 4K, rank 3/5, after round 2 (PREFILL_LAYER_PROFILE):

| kernel | before | after |
|---|---|---|
| qsa topk_reduce (slow path) | 31.7 ms | 1.9 ms (`resident_topk_merge`) |
| qsa attention (split_batch) | 35.1 ms | 20.3 ms |
| moe gate_up | 29-31 ms | 21.2-22.3 ms |
| moe down | 9-12 ms | 6.4-10.1 ms |
| QSA layer GPU total | ~110 ms | 77-80 ms |
| GDN layer GPU total | ~58 ms | 46-49 ms |
| six-layer stage GPU | ~455 ms | ~343 ms |

Round-2 changes: expert grouped kernels moved to 4-pair tiles with 32-row
workgroups, activation hoisting, folded K-quant min, no shared staging in the
float path (dp4a is not available on GFX1013 - RADV excludes it and RDNA1 has
no dot4; documented at the selection site); QSA prefill attention rewritten to
workgroup=(query, KV head, split) with head blocking so the 12 heads sharing a
KV head decode each record once (wave64 fast path).

Deploy note: the first kernel deploy OOM-killed rank 0 at the first slow-path
chunk (the lazy batch scratch needed ~54 MiB on a mirror with ~167 MB free).
Fixes: batch 8->4 queries, selection sides 8->4 MiB, mirror
`--qsa-page-cache-mib 32`. Even so rank 0 lives with ~25-60 MB available;
`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches` before heavy test phases
prevents the global OOM killer from picking flash-gordon. A real fix is on the
numerics/memory workstream (mirror index keys are 408 MB, the largest reducible
item).

**Fleet state left behind:** ring pack `/home/user/fg-ring-pack`, workers
`start-workers-ringprof.sh` (FG_PREFILL_PROFILE=1) on .43-.49, rank 0
`start-rank0-ring.sh` (FG_RING_TRACE=1) on .42. Note the ringprof worker
script also starts a rank process if run on .42 — it killed the coordinator
once; target .43-.49 only.

## 0c. ORIGINAL TL;DR — the pre-ring plan (historical)

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

**Live binary (ring audit, 2026-09-12):** sha256
`032137d39b2cd8371f29748987b72933bb312007554e57fe9404ebe93c9c5043`, built from commits
through `5468c0b`. Fleet gates green, 4K 50.89, 128 prefill 24.25, short decode 9.79.
Worker instrument: `FG_BLOCK_BENCH=1` (bench mode, exits after printing BLOCK_BENCH
lines; rank 4 is the ideal subject: 6 GDN layers, no PLE/QSA). `FG_WORKER_OWNER=1`
enables the worker owner executor in the live loop (off by default; QSA layers need
section 4D before enabling in production).

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

  **DO NOT DEPLOY contig2 ON THE CURRENT BINARY.** Contiguous blocks put QSA layers
  (3, 7, 11, ... 47) on all eight ranks (1-2 each). The runtime hardcodes QSA
  ownership to ranks {3,7} with exactly 6 layers each:
  - `qsa_owner_runtime_create` (`src/runtime.c:842`) errors unless a rank owns
    exactly `FG_QSA_OWNER_LAYER_COUNT` (6) QSA layers — every worker would exit
    at startup on contig2.
  - `qsa_owner_index` (`src/runtime.c:1217`) maps only 3→0, 7→1; is used by
    `coordinator_publish_qsa_pages` (1614), `coordinator_fetch_qsa_pages` (1676)
    and `coordinator_qsa_barrier` (1737) — rank 0 would hard-error at the first
    chunk ("QSA page owner is not rank 3 or 7").
  Section 4D must land before contig2 can boot. Also commit the working-tree
  `src/pack.c` + `src/q38_schema.c` change first: contig2 was packed with
  `token_embd` on rank 7 and HEAD's validator rejects that manifest.

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
  `fg_owner_executor_create(&owner, model, err)`. **Caveat:** that constructor
  hardcodes `executor->replicated=true` (`src/owner.c:216`), so `owns_layer` claims
  all 48 layers and `gdn_state` allocates for all 36 GDN layers (~120 MB wasted).
  Add a `replicated` parameter (coordinator passes true, workers false) and scale
  the gdn_state/ple_state allocation to owned layers. Keep the existing expert
  executor; the owner's fire/collect uses the local expert path (no tree, all 512
  experts of the block's layers are local).
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

**D. QSA state for all ranks (the hard part; blocks B and C).**
Status 2026-09-12: the owner generalization is implemented and live-parity verified
(dynamic owners from `layer_owner` in publish/fetch/barrier, per-rank sequences,
replica commit accepts any rank, `qsa_owner_runtime` accepts up to 6 owned layers;
fleet on the new binary measures 4K 50.89, gates green). Remaining for contig2:
worker QSA session/mirror for its 1-2 layers, owner-local page publish during the
block (local `qsa_owner_writer_enqueue`, bypass the `peer==0` append path), and rank
0's decode mirror page sidecar (or verified cold-fetch). Worker layer-work handler,
block bench, and non-replicated owner executor are in (`5468c0b`).
In the ring each block owner *computes* its own 1-2 QSA layers, so it must hold the
authoritative session + state file for those layers, and decode's cold-fetch must
route per layer instead of to ranks {3,7}. Changes, all in `src/runtime.c` plus a
small session open on the worker:
- `qsa_owner_runtime_create` (runtime.c:834): drop the
  `layer_count==FG_QSA_OWNER_LAYER_COUNT` requirement; size the layer array to the
  rank's actual ownership (0-2 for contiguous blocks); ranks with zero QSA layers
  keep `enabled=false`.
- `qsa_owner_index` (1217): replace with `manifest->layer_owner[layer]` everywhere
  (publish 1614, fetch 1676, barrier 1737). Barrier loops over owners {3,7} —
  generalize to the distinct set of QSA owners (or barrier only owners that
  received appends this session).
- Publish becomes owner-local: the block owner commits its own QSA pages during
  its block via its existing `qsa_owner_writer_enqueue` (runtime.c:790) — bypass
  the `peer==0` guard (handle_qsa_page_append 937, fetch 982, barrier 962) or call
  the writer directly. Rank 0 keeps publishing only its own block's layers.
- Rank 0 decode still runs all 12 QSA layers via its mirror. Its hot cache must be
  fed: either have each owner attach its page records for the block to the layer
  result message (a few KB), or accept cold-fetching the context from the eight
  owners on the first decode token (`coordinator_fetch_qsa_pages` generalized per
  above). Prefer the sidecar at first; cold fallback must be tested.
- Worker session: on the worker, open the QSA session for the block's layers in
  the owner executor (authoritative, `fg_owner_qsa_open`/`open_decode` in owner.c
  with a local state path) or mirror with a local fetch callback. Residency cost:
  index ~34 MiB/layer + selection scratch; the tightest ranks have ~1.4-1.8 GB
  headroom, so validate against the pack ledgers before enabling.
- Keep writer depth **8**; raise replica depth only after correctness gates pass.
- Gate: correctness 12/Paris, decode unchanged (9.9 short), then battery.

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
- **Never build on a serving blade.** `make -j8` on .42 next to live rank 0 caused an
  OOM livelock: SSH command execution stalled for ~20 min, rank 0 was OOM-killed, and
  the box only recovered after that. Quiesce all ranks first (`.42` free: 14.5 GiB).
- **The fleet tool's 15 s socket read timeout does not kill the remote command** — it
  keeps running detached from the client. Never run builds/packs in the foreground
  through it: `nohup <cmd> > /tmp/x.log 2>&1 &`, then poll with a short script.
- The 16 GB "blade" is 15.56 GB system RAM minus ~0.6 GB OS; the driver accepts ~15.5 GB
  total but a single allocation above ~15.5 GB fails with Vulkan result -2.

## 8. First actions, in order

**RING STATUS 2026-09-13 (read this first).** The layer ring now boots, handshakes,
runs a full chain prefill and decodes: first ring request = 29 tokens at 13.6 TPS
prefill, 64-token generation at 9.0 TPS, unoptimized and not yet numerically
correct. All eight ranks run the contig3 pack from `/home/user/fg-ring-pack`
(rank 0 carries `token_embd` again so rank 7 fits next to its n-gram shard; that
pack was built with `FG_PACK_EMBED_RANK=0`). Ring mode is gated: rank 0 needs
`FG_PREFILL_RING=1`, workers need `FG_WORKER_OWNER=1` (scripts
`start-rank0-ring.sh` / `start-workers-ring.sh`). Without the env gates the binary
is live-parity and the fleet runs the single-owner pack at 4K 50.89 with gates
green (current state after restore).

Faults found and fixed getting this far: non-replicated worker owner executor;
`HAS_NGRAM` allowed on a block-start work message; workers cannot open fileless
mirrors, so QSA sessions are state-backed (`fg_qsa_session_open_state`); owner
guard and mirror allow one-or-two owned QSA layers; per-token position map instead
of slot-zero writes; committed frontier advanced after ring prefill; cold fetches
read from the owner's authoritative session.

**DECODE EXPERIMENT 2026-09-13 (failed, reverted).** Ported the prefill grouped
pair-tile kernels into `fg_expert_decode` for batch-1 (`3c17717`). Measured:
worker expert GPU **0.77 ms -> 1.49 ms** per layer (sel=10) - a 16-pair tile with
one pair wastes lanes; the fixed per-slot graph is better at batch 1. Reverted in
`1198fb0`; binary hash restored to `0b91ca9c...`; fleet gates green at 9.1 TPS
decode. Baseline decode budget per token (48 layers, rank-0 log): total ~101 ms =
sync1 ~40 ms + collect ~48 ms + fire/shared/reduce ~13 ms. Worker expert GPU
0.77 ms for ~50 MB = 65 GB/s vs 350 GB/s roofline; rank-0 common path ~144 GB/s
vs roofline. Both are kernel-efficiency problems, not topology.
Next decode lever: a purpose-built batch-1 expert kernel (all 10 experts in one
X-flattened dispatch, vectorized loads), or MTP/spec after.
Bench caveat: `FG_BLOCK_BENCH` skips QSA layers, so an in-ring QSA stage time has
never been measured; extend the bench (open a state-backed session for the rank's
QSA layer) before tuning ring stage depth.

Remaining defects, in order:
1. **FIXED 2026-09-13**: `invalid QSA complete-page lookup` — worker fetches now read
   the session state directly (`fg_qsa_session_state_records`); no fetch errors, two
   consecutive requests complete without poisoning the transport.
2. **OPEN — numerics**: ring output is wrong (sampler picks control tokens 16/17,
   empty text) although requests complete. Distributed prefill is the suspect, not
   decode's expert path (unchanged). Isolate by comparing the ring's final prefill
   hyper against the single-owner reference, or by dumping per-layer hyper at each
   chain hop vs a local run.
3. **RING BATTERY 2026-09-13 (new binary ringC, wrong output but valid timing):
   128 prefill 21.40 TPS, 4K prefill 48.81 TPS (88.55 s), 4K decode 1.96 TPS,
   short decode 9.20 TPS.** Single-owner reference: 4K 50.89 / 4K decode ~7.
   So the ring currently gives **no prefill win and a 4x decode regression** — do
   not tune frames until stage time is understood.
4. **Why the benchmark block lied**: `FG_BLOCK_BENCH` skipped QSA layers. Real
   blocks carry 1-2 QSA layers whose state-backed session does state I/O +
   selection over a growing context every chunk; stage time is ~1.5-2.6 s, not
   0.4 s. 4K decode fetches cold pages from workers per token (cache misses).
5. **Path to 200, revised**: (a) fix numerics; (b) profile and kill QSA state I/O
   on the hot path (record cache residency, avoid per-chunk state writes/reads,
   confirm the selection path stays on GPU); (c) only then widen the chain to 6-8
   frames (Little's law: depth x 128 / stage time). Target stage ~0.5 s at depth 8
   ≈ 280 TPS; today's stage time is the blocker, not the topology.


Ops notes: each binary cycle is build on .42 (detached) -> package ->
download -> extract on 8 -> restart (~10 min). Keep the fleet quiesced while
developing; the user approved eviction on demand.

0. **Measurement gate — PASSED (2026-09-12).** `FG_BLOCK_BENCH=1` on worker rank 4
   (new binary, single-owner shard, local experts): tokens=128, 6 GDN layers,
   **mean block 400.42 ms (min 388.46, max 442.94), ~66-68 ms/layer.** Ring model
   `128 / (T_block + hop)` ≈ **260-300 TPS** at 4 frames; ~5.1-5.6x the current
   50.89. Proceed with the ring. Re-run the bench on a QSA-containing block before
   trusting QSA-layer service time.
1. Do **not** deploy contig2 until section 4D lands (section 1 warning). Keep the
   single-owner fleet healthy; commit the pack.c/q38_schema.c fix.
2. Implement B (worker owner executor with non-replicated create + layer handler +
   worker buffers) with a one-chunk round trip to the next block and back, under a
   flag, and gate on 12/Paris.
3. Implement C (chain driver in `coordinator_prefill_pipeline`, frames issued
   without waiting) using the current single-owner pack as a *correctness-only*
   fallback where each hop stays local. Gate: 4K >= 100 TPS once D+contig2 land.
4. Implement D (QSA authority per block owner, per-layer cold fetch, page sidecar).
   Then deploy contig2 and gate: 4K >= 150 TPS, decode unchanged.
5. Tune frames (5-6) and re-measure. Target: **200-300 TPS at 4K**, decode >= 9.9 short
   until MTP is unfrozen.

The user's goal, in their words: Qwen running at interactive coding speeds, Frontier-
class intelligence on their own hardware. The ring is the hill. Climb it.
