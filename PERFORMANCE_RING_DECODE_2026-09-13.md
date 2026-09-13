# Ring Decode: Shipping Fast Path Session (2026-09-13)

Binary `2ca92ba5` (ring pack, `FG_PREFILL_RING=1`; ring decode is now the
default whenever ring prefill is active, `FG_DECODE_RING=0` opts out).

## TL;DR

Ring decode now beats legacy on sustained decode: **13.33 TPS short** (legacy
10.82) and **12.08 TPS at 4K/32 tokens** (legacy 10.14), with prefill at
276-303 TPS and gates [12]/[Paris] green. The 4K ring crash was a heap
use-after-free in the QSA prefill selection path, found with ASAN and fixed
(commit `206aa30`); SIGPIPE is now ignored so a closed fabric peer returns
EPIPE instead of silently killing a rank. Per-token decode budget is down from
~95 ms to ~83 ms. Ten ms/token needs another ~8x; the budget below shows where.

## Fleet battery (attach, same ring pack, 2026-09-13)

| Config | 128 prefill | 128 decode | 4K prefill | 4K decode(1 tok) | short decode(32 tok) | 4K decode(32 tok) |
|---|---|---|---|---|---|---|
| legacy decode (ring prefill) | 27.72 | 6.09 | 264.95 | 8.37 | 10.82 | 10.14 |
| **ring decode (final)** | **35.68** | 5.99 | **279.22** | 8.30 | **13.33** | **12.08** |

Best observed ring prefill: 303.4 TPS (4K) / 38.2 (128) on the same binary
lineage before the QSA fix. Correctness: `/no_think` arithmetic -> `12`;
capital -> `Paris`.

The two decode-TPS columns at 1 completion token measure first-token latency
(ring pays the QSA first-token warm ~1%); the 32-token columns are sustained.
Prefill never regressed (>=230 TPS gate).

## Where 83 ms/token goes (final, 4K sustained)

| Phase | Before (stock ring) | After (final) |
|---|---|---|
| embed (rank 0) | 1.5 | 1.2 |
| rank 1 block (layers 0-5) | 11.3 | 10.6 |
| rank 0 own block (6-11) | 10.2 | 9.5 |
| ranks 2-7 (6 blocks) | 67.1 | 57.0 |
| output head round trip (rank 4) | 3.9 | 3.9 |
| **total** | **~95-100 ms** | **~82.8 ms** |

Per layer (worker, production path): `batch1` 0.81-1.15 ms (common path GPU +
fence), top-K 0.07, expert submit 0.08, shared batch + expert wait 0.37-0.75,
collect 0.10, CPU reduce 0.06 -> 1.5-1.9 ms/layer. Kernel GPU time is
7.1-7.3 ms per 6-layer block (~1.2 ms/layer); the remaining ~0.3-0.6 ms/layer
is host and fence overhead.

### Kernel cost per 6-layer block (Vulkan timestamps)

| kernel | ms/block |
|---|---|
| fg_dense_q8_0_cooked_r8 (GR read chain) | 2.0 |
| fg_moe_decode_down_reduce | 1.4 |
| fg_moe_decode_gate_up | 1.0 |
| fg_gdn_recurrent_algebraic | 0.8 |
| fg_dense_f32 / gr_mix_partial / cooked / split / rms | 0.2-0.4 each |

## Changes this session

1. `206aa30` **fix(qsa)**: `ensure_select_scratch` freed `s->select_ids` while
   `select_prefill_batch` held the pointer captured before the call. The 4K
   selection readback wrote 2 KiB through a freed 256 KiB buffer -> heap
   corruption (rank SIGABRT in glibc/radv, silent rank exits). ASAN trace on a
   worker pinned it; `select_ids` is now allocated once. Also ignore SIGPIPE
   process-wide (a closed fabric peer was a silent death).
2. `da8bbc9` ring transport hardening (stale QSA mirror results dropped, QSA
   barrier ack skips stale frames, warm drain before "not reusable"),
   `FG_DECODE_MS`/`FG_DECODE_PROFILE` instrumentation, async expert graph
   submission so the shared-expert batch queues behind it, and MOVNTDQA
   streaming reads for write-combining Vulkan memory (host readback of the
   40 KiB hyper handoff: 0.34 ms -> 0.01 ms).
3. `41d6898` stream router/activation/shared reads off WC memory.
4. Multi-request ring sessions stable (128 -> 4K -> decode battery and repeated
   4K32 requests run back to back); prefill 128/4K and gates re-verified.

## Remaining gap to 10 ms/token (100 TPS)

1. **48 serial layers x ~1.2 ms GPU = ~57 ms** (top-3 kernels above). Needs
   fusion into ~0.2 ms/layer: fold the GR read chain (norm, down, SiLU, up,
   inject, mix: 12 dispatches/layer), keep the fused expert pair, and retile
   the GDN projections/recurrent for batch 1.
2. **Host/fence overhead ~15-29 ms**: two submit+fence waits per layer (router
   readback and shared batch) plus CPU top-K and reduce. Next step is a GPU
   shared-scaled reduce + gr_write chained into the following layer's batch
   (one fence/layer), then GPU routing to remove the last fence.
3. **Fixed rank-0 overhead ~6.5 ms**: output-head round trip 3.9 ms, embed
   1.2 ms, hop ingress/egress ~1.4 ms. Overlapping it needs speculative/MTP
   (frozen) or folding the output head into the chain.

Reaching 20 TPS (50 ms) additionally needs block execution under ~5.5 ms
(~0.9 ms/layer): items 1 and 2 combined. 100 TPS needs token-level pipelining
so blocks of adjacent tokens overlap across ranks.
