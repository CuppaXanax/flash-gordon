# Layer-Ring Prefill Design (2026-09-12)

Status: design only. Prerequisite pack (`contiguous.expert-map`, rank r owns layers
6r..6r+5) is generated and pack is running into `/home/user/flash-gordon-q38-contig`.

## Goal

Move the common path off rank 0 so all eight ranks compute during prefill.
Target: 150-250 TPS at 4K with 4-6 frames, versus 50.7 today (four frames,
centralised common path, rank 0 GPU-bound at ~0.59 s per 128-token chunk).

## Why the ring and not a star

- Star (every remote layer's input and output through rank 0): 2 transfers per
  layer, ~96 x 5.24 MB per 128-token chunk = ~500 MB. At the measured ~150 MB/s
  effective bulk payload rate that is 3.3 s per chunk, a ~39 TPS ceiling. Dead.
- Chain (each rank computes its six layers, hands the 5.24 MB hyper state to the
  next rank): eight handoffs per chunk = ~42 MB, ~0.28 s at 150 MB/s -> ~450 TPS
  bandwidth ceiling. Per-rank compute is ~0.29 s per chunk (252 ms experts +
  ~42 ms common). Four frames hid 0.6 s/chunk -> ~210 TPS; six frames more.
- Chain requires contiguous ownership, which is why `owners=` and arbitrary
  `layer_owner` were added (`2862f31`).

## The QSA problem (why this is not a one-day change)

QSA layers are every fourth layer (3, 7, 11, ...). In a contiguous chain each
rank owns one or two QSA layers, so each rank needs the QSA state for those
layers. Today only rank 0 runs QSA layers and its session holds the index
(411 MiB) plus the record cache; the page transport ships pages to the two
replica owners (ranks 3 and 7) for durability. A ring therefore needs:

1. QSA session mirrors on all eight ranks (index ~411 MiB device each; workers
   currently use ~9.5 GiB weights + 3.9 GiB ngram, so it likely fits, but it
   must be measured before betting on it).
2. Page distribution per chunk to each rank for its own QSA layers only:
   ~1-2 layers x 32 blocks x 3.6 KB = ~230 KB per rank per chunk (~3 MB total)
   - cheap. This is a generalisation of `qsa_page_transport` from two owners
   to N owners.
3. The attention/record kernels already run on the owner rank for prefill
   (`fg_owner_qsa_prefill`); they need a valid local session.

## Phases

**A. Protocol + plumbing, flag off by default.**
- New messages `FG_MSG_PREFILL_LAYER` / `FG_MSG_PREFILL_LAYER_RESULT`.
  Wire: `{u32 layer, u32 first_position, u32 token_count, u32 sequence,
  u32 positions[3*tokens], f32 hyper[tokens*10240]}`. ~5.24 MB for 128 tokens.
  Add encode/decode plus a `test_core` round-trip and a bounds test.
- Worker rank loop: on layer work, copy the hyper into the owner's prefill
  input tensor, run `fg_owner_prefill_layer_begin` / `_finish` with the LOCAL
  expert callbacks (`fg_expert_prefill_enqueue` / `fg_expert_prefill_finish`,
  the same pair rank 0 already uses for self-routed experts), copy the output
  back, send the result.
- Coordinator: per layer choose local (owner 0 or QSA) vs remote behind an env
  flag; fire = send, collect = recv. Reuse the existing frame schedule; the
  remote path is symmetric to expert fire/collect.

**B. GDN layers only (no QSA).** Force QSA layers local, distribute the 36 GDN
layers. Star topology is acceptable here (each remote layer's result returns to
rank 0) because the transfer volume is 72 x 5.24 MB = ~377 MB/chunk -> ~2.5 s,
which caps at ~50 TPS. This phase validates correctness of remote layer
execution; it is not expected to be fast.

**C. Chain topology.** Rank r receives from r-1, computes its six layers, sends
to r+1 (rank 7 sends back to rank 0 for output projection). Four frames hide
the chain latency. Gate: correctness 12/Paris and 4K >= 100 TPS before
enabling QSA.

**D. QSA session mirrors + page fan-out to all ranks.** Distribute QSA layers;
full ring. Gate: 4K >= 150 TPS, no adjacent-context regression.

**E. Deeper frames.** The 6-frame attempt failed from QSA owner-writer I/O
thrash (raised depth made it worse; reverted). With QSA pages going to owners
as they are computed, the backlog shape changes; re-test frames after D.

## Decode

Unchanged in phases A-D. Decode currently runs 48 layers synchronously through
rank 0 with local expert dispatch (~2.0 ms/layer: sync1 0.8, collect 0.7). The
ring does not help decode without token-level pipelining; treat decode
separately (MTP is frozen by request). Do not let the ring regress decode:
every phase runs the short-decode and 4K-decode legs of the attach battery.

## Open questions / risks

- Fabric effective bulk rate for 5.24 MB messages: measured p50 45.9 ms for
  5.24 MB one way (~114 MB/s) and 2.0 Gbps on 3.89 MB sends. The chain's
  450 TPS ceiling assumes ~150 MB/s sustained; verify with a direct A/B.
- Worker Vk memory headroom for QSA mirrors (measure before phase D).
- QSA page transport to N owners: ordering guarantees must match the current
  two-owner path (pages are committed after each chunk; a rank must not run a
  QSA layer beyond its committed frontier).
- The 4-frame schedule currently finishes a frame's previous layer immediately
  before beginning its next; the chain may prefer issuing the send earlier
  (fire after GR write, collect one layer later) - measure, do not assume.
