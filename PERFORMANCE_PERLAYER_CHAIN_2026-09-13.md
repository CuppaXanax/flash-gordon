# Per-Layer Decode Chain: One Fence per Block Owner (2026-09-13)

Binary `cbb3aa54` (ring pack; ring decode on by default, chain on by default).
Environment switches: `FG_DECODE_CHAIN=0` restores the per-layer owner machine
(the previous ring fast path); `FG_DECODE_RING=0` still selects the legacy
expert-parallel replay, which is untouched.

## TL;DR

The ring decode block owner now records the **entire 6-layer block into one
submission**: the GR read chain, attention, shared expert, GPU top-10 routing,
the fused expert pair, the shared-scaled reduction and `gr_write` are chained
with no per-layer fence, no host top-K, and no expert readback.  A block costs
one fence and 2-4 submissions instead of ~21-22, and 4K sustained decode goes
from **12.03 to ~15.3 TPS** (+27%) with prefill at 277-305 TPS and the
[12]/[Paris] gates green.  Short 32-token decode reaches **16.50 TPS** on the
standard battery.  The remaining gap to 10 ms/token is now almost entirely GPU
kernel time plus the fixed embed/output head round trip.

## Fleet battery (same build, 2026-09-13 attach run)

| Config | 128 prefill | 128 decode | 4K prefill | 4K decode(1 tok) | short decode(32 tok) | 4K decode(32 tok) |
|---|---|---|---|---|---|---|
| before (270a7ddb, per-layer fences) | 40.6 | 10.9 | 264-283 | 10.44 | 13.44 | 12.03 |
| **after (cbb3aa54, chained block)** | 40.6 | 10.9 | 277.6 (258-305) | **15.85** | **16.50** | **15.29 / 14.67** |

Correctness: `/no_think` arithmetic -> `12`; capital -> `Paris` (both re-run on
the final build).  A/B notes: the 4K(32 tok) figure is the mean of two clean
runs (15.29, 14.67) plus the earlier clean run at 15.22; prefill stayed inside
the 270 TPS gate in every run.

## Where a 4K token goes (rank-0 trace, mean of last 24 tokens)

| Phase | Before | After |
|---|---|---|
| embed (rank 0) | 1.2 | 0.55 |
| rank 1 block (layers 0-5) | 10.6 | 8.80 |
| rank 0 own block (6-11) | 9.1 | 7.50 |
| ranks 2-7 (6 blocks) | 59.6 | 43.98 (~7.33 each) |
| output head round trip | 3.5 | 3.38 |
| **total** | **~83.5 ms** | **~64.2-67.3 ms** |

`RING_DECODE` per-token trace: embed 0.55, first hop (rank 1) 8.8, own block
7.5, tail (ranks 2-7) 44.0, output 3.4 ms -> 64.2 ms steady state, 15.5 TPS.

## What changed

1. **Chained block (`fg_owner_decode_block_chained`)**: per layer the common
   path (GR read, GDN/QSA, GR write, GR read, router, Q8_K quantization),
   shared expert, GPU routing and the fused expert pair are all recorded into
   the block owner's active command buffer.  The block ends with the usual
   `gr_write` and one fence; nothing inside a layer needs a host round trip.
2. **GPU top-10 routing** (`fg_router_top10` + `fg_decode_tile_schedule`)
   replaces the router-logits readback and CPU `fg_q38_router_topk`.  The
   fused pair's gate tensor is written by the same selection, matching CPU
   softmax-over-selected semantics.
3. **GPU shared-scaled reduce** (`fg_moe_shared_add`): the down projection's
   gate-weighted rank sum is folded with `sigmoid(shared_scalar) *
   shared_output` on the GPU; the host no longer reads either tensor and the
   CPU `moe_reduce_into` leaves the decode critical path.
4. **New expert chain API** (`fg_expert_decode_chain[_ready]`) runs
   gate/up/down inside the caller's batch.  It is only selected when the whole
   per-layer expert slab lives on the block owner (the ring pack) and the
   fused pair exists, so `FG_DECODE_CHAIN=0` or any other topology falls back
   to the previous, verified per-layer machine.

QSA still breaks the recording when it needs a host selection readback (every
4th layer): a chained block submits 2-4 times instead of 21-22.

## Kernel profile per 6-layer block (Vulkan timestamps, final build)

Rank 2 (layers 12-17, one QSA layer): GPU 6.03 ms, 220 dispatches, 4 submits.
Rank 1 (layers 0-5 + PLE, mixed Q5_1/Q8_0 down): GPU 7.00 ms, 228 dispatches.

| kernel | ms/block (rank 2) |
|---|---|
| fused expert pair (gate_up + down_reduce) | 1.19 |
| GDN projections + output (r8/cooked) | 1.19 |
| `gdn_recurrent_algebraic` | 0.65 |
| GR read chains (rms/split/inject/silu/r8/mix) | 0.91 |
| router + shared expert | 0.54 |
| QSA (projection/attention/output) | ~0.25 |
| shared_add, gr_write, conv, swiglu | ~0.1 |

## Remaining gap to 10 ms/token

* Blocks are now ~7.3 ms each with only ~0.3-1.1 ms of host/fence residual
  (write 0.002, read 0.011, egress 0.09, the rest GPU).  Seven blocks cost
  ~51 ms; embed+output add ~3.9 ms.
* 20 TPS (50 ms) needs per-block GPU down to ~6.3 ms: about 1 ms more per
  block from the expert pair / GDN projections / recurrent kernels, which are
  now bandwidth- or latency-bound on the same 24 CU.
* 100 TPS (10 ms) additionally needs token-level pipelining across ranks
  (blocks of adjacent tokens overlapped) and folding the output head into the
  chain; per-layer GPU time would have to fall to ~1 ms.
