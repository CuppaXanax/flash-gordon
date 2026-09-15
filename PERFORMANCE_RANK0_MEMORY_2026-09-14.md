# Rank-0 Memory Headroom and Prefill Allocation Resilience (2026-09-14 session)

Worktree `fg-work-mem12`, branch `fix/rank0-memory-headroom`, base `56f76bb`.
Fleet: 8x BC-250 (RADV GFX1013, 15.94 GB UMA physical each), pack
`/home/user/fg-ring-pack`, ring prefill + ring decode, 262144 context,
128 MiB QSA record cache.

## 1. Rank-0 consumer inventory (measured)

Method: `COORDINATOR_VK_COMPONENTS`, `COORDINATOR_VK_LEDGER`,
`COORDINATOR_HOST_LEDGER`, `COORDINATOR_DEFERRED_LEDGER`, plus the QSA canary
and session lines, all emitted at startup by the coordinator and captured from
`/home/user/.fleet-ab/attach-20260911b/ep-rank-0.log` (before) and
`/home/user/fg-mem12-rank0.log` (after).

| Rank-0 consumer | Before (11ee2352 lineage) | After (this branch) |
|---|---|---|
| replicated model arena | 14,943,436,800 | 14,943,436,800 |
| QSA mirror index, eager segments | 213,909,504 (12 layers) | 35,651,584 (2 owned layers) |
| QSA record cache (CLI 128 MiB) | 134,214,768 | 134,214,768 |
| owner GDN/PLE state | 119,513,088 (36 layers + PLE) | 13,238,272 (4 owned GDN layers) |
| owner attention family scratch | 38,010,880 | 38,010,880 |
| ping-pong, activations, positions, prefill token ids | ~14.8 MB | ~14.8 MB |
| **Vulkan live (startup)** | **15,537,791,136** | **15,213,929,120** |
| host prefill staging (8 frames x pair-output arena) | 105,021,440 | 105,021,440 |
| host qsa/ngram/fabric | ~5.6 MB | ~4.1 MB |
| deferred host: QSA page transport | 246,058,456 | **0 (lazy, never needed in ring mode)** |
| deferred host: ngram cache / wires / work wire | 19,358,888 | 19,358,888 |
| deferred ngram Vulkan payload tensors | 11,960,320 | 11,960,320 |
| **projected peak + driver reserve** | **16,461,058,329** | **15,891,137,857** |
| **conservative_peak_margin** | **-524,087,577** | **+45,832,895** |
| readiness | insufficient | unknown-os-overhead |
| rank-0 QSA state file on disk | 6,442,459,136 (12 layers) | 1,073,750,016 (2 layers) |

Steady-state Vulkan drops 323,862,016 B. The first prefill chunk no longer
allocates the 246 MB QSA page transport (232 MB of it was the fixed 64-slot
replica payload), so the committed peak during prefill drops by ~570 MB
against the same physical budget. Kernel `MemAvailable` at idle went from
12 MB to ~160 MB free with ~90 MB reclaimable.

`unclassified` (Vulkan bytes not attributed by the ledger) is 23 MB at
startup and ~35 MB after the first prefill; it is the per-layer expert graph
tensors and output-slice scratch that the first dispatch materializes. No new
instrumentation was added because the ledgers already explain every larger
consumer.

## 2. The "3.912 GiB n-gram rows" decision

The 3.912 GiB figure is the resident n-gram shard on **workers**, not rank 0.
`fg_rank_main` rejects `rank==0` for resident shards (`src/ngram.c:66`), and
the rank-0 environment has always shown `ngram=0` in
`COORDINATOR_VK_COMPONENTS`. Repeated `rank N READY` lines show 3.912 GiB
(ranks 1, 2, 3, 5, 6, 7) and 3.353 GiB (rank 4).

Rank 0 has two n-gram paths:

- prefill: `fg_ngram_store_lookup_prefill` on a lazily opened, disk-backed
  store (`O_DIRECT`, 8 MiB io ring, 4 MiB host cache, ~11.96 MB Vulkan
  payload/embedding tensors allocated on first lookup);
- decode: `resident_ngram_lookup` fetches exactly 16 rows per token from the
  row owner over the control fabric (`FG_MSG_NGRAM_WORK/RESULT`), then
  dequantizes into the same store tensors.

Decision: keep the root shards on workers and keep rank 0 lazy. Moving rows
to rank 0 would consume the headroom this session created; the worker shards
are mlock'd but each worker still reported 150-870 MB available after the
soak. The only plausible shrink (pageable mmap instead of mlock on workers)
is a worker-side change with no rank-0 benefit and is left to a later session.

## 3. What changed

- `82dcc7a` perf(mem): allocate the QSA page transport lazily and at
  microbatch size.
  - `coordinator_publish_qsa_pages` returns before `qsa_page_transport_ensure`
    when ring prefill is active: block owners publish their own pages, so
    rank 0 never needs the transport in the deployed ring configuration.
  - The deferred ledger now reports `transport=0` for
    `FG_PREFILL_RING=1 FG_DECODE_RING=1`, and a smaller capacity otherwise.
  - `fg_qsa_replica_create` takes a per-slot payload size; the coordinator
    sizes it from the sealed microbatch
    (`12 + 6 * ceil(microbatch/4) * entry_bytes`), and slot payloads allocate
    on first reservation instead of 64 x 3.8 MB up front. Legacy (non-ring)
    transport capacity drops from 246 MB to ~64 MB worst case, and only the
    slots actually used are touched.
- `9314901` perf(mem): keep ring-mode rank-0 QSA and GDN state to owned
  layers.
  - `fg_runtime_ring_enabled()` (both ring envs) gates the coordinator
    executor: GDN conv/recurrent state is allocated for the layers the
    manifest assigns to rank 0 (4 GDN layers, was 36), PLE state is skipped
    when layer 1 is remote, and only decode slot 0 is created (was 4 slots,
    ~39 MB).
  - The coordinator's state mirror opens the QSA state file with only the
    QSA layers rank 0 owns (2 of 12); the file itself shrinks 6.0 GiB -> 1.0
    GiB. Non-ring decode keeps all 12 layers and the fetch/warm paths.

Diff is separable: transport allocation in the first commit, executor/mirror
state in the second. No shader source changed, so `vulkan/*.spv` is
byte-identical to the base build.

## 4. Measured prefill and stall absence

Fleet battery (`Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2`), new
build, three clean runs:

| Run | 128 prefill | 4K run 1 | 4K run 2 | 16K | short decode |
|---|---|---|---|---|---|
| ab-005827 | 38.91 TPS | 286.37 TPS / 15.09 s | 280.03 TPS / 15.43 s | 303.11 TPS / 56.48 s | 23.68 TPS |
| ab-010406 | 40.59 TPS | 285.54 TPS / 15.13 s | 283.00 TPS / 15.27 s | - | 24.36 TPS |
| ab-010723 | 36.39 TPS | 276.47 TPS / 15.63 s | 281.28 TPS / 15.36 s | - | 23.89 TPS |

Every 4K prefill completed in ~15 s; the failing log's final 4K chunks had
`RING_OWN_BLOCK` of 16,452 / 47,467 / 103,576 / 142,975 / 298,941 / 561,328 ms
(9.4 min class) followed by `double free or corruption (!prev)`.

Trace run on the new build (`FG_RING_TRACE=1`, 4K, 34 chunks):
`RING_OWN_BLOCK n=34 min=215.6 ms med=261.4 ms p95=276.7 ms max=363.4 ms`.
No chunk over 1 second; no `double free`.

`tools/pi-stability.ps1`: **PASS** - soak 128/1024/4096/8192/12288/16384 all
PASS (16384 at 303.63 TPS), 4-turn conversation to 15,381 tokens PASS, gates
`12` and `Paris` PASS, 4K prefill band 273.59 vs 240 required, short decode
21.26 vs 15 required, all 8 ranks alive, 0 failures.

## 5. Double-free status

The corruption has not reproduced across three batteries, the full
pi-stability soak, and a trace run (~60 minutes of continuous service,
including the former 16K repro). Every observed occurrence in the failing
log immediately followed a multi-minute thrash phase, and the rank-0 process
was last measured at `MemAvailable=12 MB` with zram engaged. The one code
path that reliably allocated and touched ~246 MB at the start of a request
under that pressure was `qsa_page_transport_ensure` (64 x 3.8 MB replica
slots); that allocation is now gone from the ring path and the replica slots
allocate on demand at microbatch size.

This is a strong correlation, not a proven fix: no backtrace was captured
because the blades have no gdb/ASAN toolchain. Ranked next step 1 below
covers pinning it if it ever returns. All touched allocation failures now
fail cleanly (no state mutation before success: replica reserve, transport
ensure, ring prefill wire allocation).

## 6. Ranked next steps

1. ASAN/gdb provisioned rank-0 build for the corruption reproducer; add
   `FG_VK_ALLOC_TRACE` to dump allocation sizes at first-prefill end so the
   next unknown growth is attributed automatically.
2. Host prefill staging is now the largest rank-0 host block (105 MB: 8
   frames x 13.1 MB pair-output arena). Allocate outputs per frame lazily or
   drop ring in-flight depth 8 -> 4 after an A/B; expected -52 MB, watch
   prefill TPS.
3. `--qsa-page-cache-mib 128` is the largest single tunable CLI allocation.
   Measure hit rate at 64 MiB; freeing 64 MB would make the margin ~110 MB
   without touching the code.
4. Rank-0 model arena is 13.9 GiB (all common tensors + its 6 expert layers).
   A coordinator-only load mask that skips common tensors for layers outside
   its block would free ~4 GiB but requires model.c/manifest changes and is
   outside this branch's scope.
5. Worker n-gram residency pins 3.912 GiB per worker via mlock. A pageable
   mmap with readahead would return that memory to the worker kernel pool;
   ranks 5 and 7 showed the lowest post-soak headroom (576/346 MB available).

## 7. Fleet state left behind

- Serving from `/home/user/flash-gordon-live/20260915-mem12` on all eight
  blades, binary sha256 `8bb9911938632d0e...` (all ranks identical).
- Rank 0: `flash-gordon api` on 0.0.0.0:8080 with
  `FG_PREFILL_RING=1 FG_DECODE_RING=1`, context 262144, page cache 128 MiB.
- Workers 1-7: `flash-gordon rank` with `FG_WORKER_OWNER=1`.
- Pack `/home/user/fg-ring-pack` untouched; rank-0 QSA state file
  recreated at the new 2-layer size.
- Rollback binary/shaders still present at
  `/home/user/flash-gordon-live/20260910-230032-21d65c`.
- Rank-0 `COORDINATOR_MEMORY_LEDGER` reports
  `conservative_peak_margin=45832895 readiness=unknown-os-overhead`, gates
  `[12]`/`[Paris]` green, `tools/pi-stability.ps1` PASS, no wedge.

## 8. Files changed

- `include/fg_qsa_replica.h`, `src/qsa_replica.c` - sized lazy slot payloads.
- `include/fg_qsa.h`, `src/qsa.c` - `owned_only` layer selection for the
  coordinator state mirror.
- `include/fg_owner.h`, `src/owner.c` - ring-mode owned-layer executor
  allocation and `owned_only` mirror open.
- `include/fg_runtime.h` - `fg_runtime_ring_enabled()`.
- `src/runtime.c` - lazy/microbatch-sized QSA transport, ledger accounting
  for the reduced transport/owner/QSA state.
- `tests/test_core.c`, `tests/test_prefill_dispatch.c` - replica create
  signature.
