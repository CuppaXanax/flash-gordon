# Batched Decode Block (B=2): design and round-1 prototype

Status: **round 1** (2026-09-24) - design complete, dense batch-2 MMV
prototype measured on the fleet (bit-identical, byte-weighted 1.583x
amortization).  The block wiring and the expert union dispatch are the next
round; the prototype kernel is inert on branch `feat/batch-block-r1`
(commit `d1a2630`), the serving build is unchanged.

Related: `docs/DEPTH_B_DECODE_RING.md` (B=2 ring state, transactions, wire),
`PERFORMANCE_BYTE_BUDGET_2026-09-15.md` (bytes/token),
`bc-250-dbg/results/depthb2-20260923-2310/EVIDENCE.md` (per-phase profiles),
`bc-250-dbg/results/batchblock-20260924-101336/EVIDENCE.md` (this round).

## 1. Goal and why the serial block is the wall

One weight pass per layer must serve both slots' tokens so 2 tokens cost ~1
weight read, not 2.  Depth-B today runs the block twice per step (once per
slot), so the ring is ~2x the B=1 ring: 91-95 ms vs 51.3 ms, decode-only
speedup 1.46x against the 1.6x gate.

Measured per 6-layer block per token (rank 0, layers 6-11, `FG_DECODE_PROFILE`
from `depthb2-20260923-2310/logs2/selftest-prof.log`; GPU ms, block total
3.683):

| phase (profile scope) | kernels | GPU ms/token |
|---|---|---:|
| dense MMV (r8 decode) | `fg_dense_q8_0_cooked_r8` x50 (gdn/qsa/shared/gr-inject) | **1.502** |
| experts | `fg_moe_decode_gate_up` + `_down_reduce` + schedule | **0.878** |
| dense MMV (other) | cooked z-projection, bf16 indexer, f32, quant/prep | 0.400 |
| GDN state | conv + recurrent algebraic | 0.160 |
| router | `fg_dense_f32` + top10 + q8_k quant | 0.210 |
| QSA attention/records | attention, gather/commit/prepare | 0.078 |
| gr/hc/norm/mix/write | rms, hc inject, gr mix/write, swiglu, add | 0.400 |
| **total** | | **3.628** |

A serial pair costs 7.26 ms GPU (10.8 ms wall) per rank; the ring is
8 x 10.8 + ~8 ms of hops/waits = 91-95 ms.  The byte model says the ceiling
for B=2 is `dense/2 + experts + state` = 2.42 + 1.84 = 4.28 GB/token
(1.56x), because two random top-10-of-512 selections share almost no expert
weights; the design's 1.87x assumes expert sharing that statistically does not
exist.  The block can still approach the 1.56x byte ceiling by halving the
dense term, and the 1.6x gate needs the per-step overheads (recording drain,
sampling relay, n-gram wake) folded in as well.

## 2. Per-layer dispatch plan

Layout: the block is fed two contiguous token rows (`2 x FG_HIDDEN_SIZE` f32)
instead of one, and produces two.  Per-slot state (GDN conv/recurrent, PLE,
QSA namespace) stays in the depth-B `state_slot` tables; only the
weight-consuming passes change shape.

| phase | plan | bit-exact route |
|---|---|---|
| embedding/n-gram | already token-indexed (`fg_vk_embedding_q8_0_batch`, per-slot lookups) | unchanged |
| PLE gate/conv | batch over rows (tokens dim) | existing token params |
| gr_read / hc inject / rms | elementwise row-wise, existing `tokens` args | unchanged per row |
| GDN projections (qkv/z/alpha/beta, ssm_out) | **b2 MMV** (`fg_vk_dense_q8_0_b2`) | r8 arithmetic per token (measured) |
| GDN conv + recurrent | stay per token (state recurrence, tiny) | unchanged |
| QSA projections (q/gate/k/v, output, bf16 indexer) | **b2 MMV**; bf16/indexer b2 variants next | per-row dot order |
| QSA attention/select/commit | stay per slot (per-session records/cursors) | unchanged |
| router | `fg_dense_f32` with `tokens=2` (weights shared, rows independent) | per-row dot |
| shared expert gate/up/down | **b2 MMV** | r8 arithmetic per token |
| experts | **union dispatch**: one work with `token_count=2`, `2*top_k` pairs | per-pair accumulation unchanged |
| moe_shared_add / gr_write | batch over rows | existing token params |

The b2 MMV kernel (round-1 prototype) is a drop-in replacement for the r8
decode dispatch at the same call sites: same cooked layout, same output
row-major `[token][row]`.

The alternative rejected for now: two *row-strided contributors* (each token's
rows processed by different workgroups over the same weight tile, coordinated
through LDS).  It halves register pressure but adds a cross-workgroup
reduction that cannot keep the r8 subgroupAdd order, so bit-exactness would
have to come from a new numeric contract.

## 3. Expert dispatch/collect for two tokens

The wire and executor already carry per-token records: `fg_prefill_pair` has
`token_slot`, `fg_expert_prefill_enqueue` validates `token_count` up to
`manifest->prefill_microbatch` and builds per-expert tile schedules, and the
batch result returns `token_count x FG_HIDDEN_SIZE` outputs.  The decode path
today submits `top_k` pairs for one token (`fg_expert_decode_submit`).

Plan:

1. Build one decode work per layer with `token_count=2` and the union of the
   two tokens' `(expert, routing_slot, gate)` records tagged `token_slot
   0/1`; order all pairs by expert id so tiles coalesce and one schedule
   serves both tokens.
2. `fg_vk_moe_decode_gate_up` / `_down_reduce` already index tiles by pair
   and accumulate per pair; a tile may carry rows from both tokens, and the
   reduce adds them to the right token row.  No weight sharing is claimed;
   the win is one submission, one schedule build, and 20 expert tiles in
   flight instead of 2 x 10 serial kernels (the expert pair runs at
   133-217 GB/s of a 353 GB/s peak, so there is latency/occupancy headroom).
3. Results come back per token exactly as today's per-slot records; the
   collect step keeps the existing per-slot ordering.

Wire: `FG_MSG_DECODE_BATCH_WORK` already carries per-slot records; the expert
work message is per rank and internal, so this is a runtime/executor change,
not a protocol change.  Fail-closed rule: reject a union that would exceed
`executor->max_pairs` or mix layers.

## 4. State and positions

Depth-B already gives every sequence its own `state_slot`: GDN conv/recurrent
and PLE state, a QSA session with its own record cursor and page stream, and
per-slot M-RoPE positions in the batch work/result.  The batched block reads
`position` only through the per-token attention paths, which stay per slot, so
positions need no new handling.  The transactions (`fg_owner_session_device_snapshot`)
cover the whole batch; the batched block must not introduce a new state write
path outside them.

## 5. QSA access

Keep the depth-B rule: a batch session owns its QSA namespace and page stream,
and depth-B stays ring-only (no live rank-0 page transport).  Round 1 batches
only the QSA *projections* (b2 MMV) and leaves selection/attention/commit per
slot; the QSA scan budget (`FG_DECODE_BATCH_QSA_SCAN_BUDGET_BYTES`) is
unchanged.  At 4K the QSA share is 0.08 ms/block, so there is no amortization
pressure there yet; at 262K the per-sequence scan dominates and is already
scheduled batch-depth-aware.

## 6. Bit-exactness

The invariant is per-token: a batched pass must produce, for each token, the
exact float sequence of the B=1 pass.  Concretely:

- **b2 MMV** keeps the r8 kernel's order for each token: same weight words,
  same `dot(unpack8(word), input)` operand order, same per-thread fma chain
  over block groups, same subgroupAdd tree, same cross-subgroup sum.  Verified
  bit-identical on every production dense shape (round-1 prototype,
  `bits_exact=1`).
- **router/bf16/quant** row-wise kernels are independent per output row; the
  `tokens` dimension only changes the grid.
- **expert union** changes which *tile* carries a pair, never the pair's
  arithmetic: each pair still accumulates its own gate-weighted down
  projection with the same order, and the reduce adds per token.
- **elementwise** phases (rms/hc/gr/swiglu/add) are per row.

Rejected: any batching that sums partials across tokens (e.g. one accumulator
for both tokens) or reorders a token's accumulation.

## 7. Round-1 prototype (measured)

`shaders/fg_dense_q8_0_cooked_r8_b2.comp` + `fg_vk_dense_q8_0_b2` +
`tests/test_batch_block.c`.  For each r8-eligible production dense shape the
test builds a cooked q8_0 weight, runs the serial pair (two r8 dispatches,
exactly what the ring does per slot) and the batch-2 pass, compares both
tokens bit-for-bit and times GPU ms over 24 iterations.

Fleet (rank 0 idle, ring serving), bin `b7c27df6b2334485`, SPV
`162e36343a6bf523` (`bc-250-dbg/results/batchblock-20260924-101336/`):

| shape (in x out) | role | serial pair us | b2 us | factor | bits |
|---|---|---:|---:|---:|---|
| 2560 x 10240 | gdn qkv/gate | 178.3 | 112.8 | 1.582 | exact |
| 2560 x 12288 | qsa q+gate | 215.7 | 136.4 | 1.581 | exact |
| 2560 x 512 | qsa k/v | 17.9 | 10.4 | 1.727 | exact |
| 6144 x 2560 | qsa output | 163.4 | 101.7 | 1.607 | exact |
| 2560 x 2560 | ssm_out | 47.2 | 30.6 | 1.544 | exact |
| 2560 x 640 | shared gate/up | 18.5 | 10.8 | 1.724 | exact |
| 640 x 2560 | shared down | 19.1 | 13.8 | 1.376 | exact |

Byte-weighted factor **1.583**; all seven shapes bit-identical.  The first
run (d1a2630) measured 1.587 with the same shape table; sharing the `unpack8`
expansions across the token pair was neutral (the kernel is weight-load bound,
not ALU bound), so the factor is stable.

The saved time is the weight-read half of the r8 phase: 1.502 ms/token block
-> ~0.95 ms/token-pair, i.e. the dense pair costs ~1.9 ms instead of 3.0 ms.

## 8. Amortization budget (per rank, 6-layer block, GPU ms)

| phase | serial pair | batched (measured/est) | basis |
|---|---:|---:|---|
| dense MMV (r8) | 3.004 | **1.893** | measured 1.583x |
| dense other | 0.800 | 0.400-0.600 | b2/row-row variants |
| experts | 1.756 | 1.756 -> 1.3-1.5 if the union dispatch lifts occupancy | design; needs measurement |
| router + top10 + quant | 0.420 | 0.315 | router rows batch, top10 per token |
| GDN state | 0.320 | 0.320 | per token |
| QSA attention/records | 0.156 | 0.156 | per slot |
| gr/hc/norm/mix | 0.800 | 0.440 | row-wise batching |
| **block GPU** | **7.256** | **5.3-5.5** | |
| block wall (5.4 -> 10.8 serial) | 10.8 | **7.0-7.6** | GPU + non-GPU per submission |
| ring (8 ranks + ~8 ms hops) | 91-95 | **64-69** | |
| step (ring + sampling 6 + assembly ~1 + prepare 1-4.5) | 100-132 | **72-81** | |

Projected aggregate at a ~75 ms step: 26.7 tok/s vs 16.5 = **1.6x**; at 81 ms
it is 1.5x.  Without the expert union lift and the recording interleave the
ring lands ~66-69 ms and the step ~75-85 ms (1.4-1.55x).  The 1.87x byte-model
figure is not reachable with disjoint top-10 selections; the honest target is
1.5-1.6x, with the dense b2 pass contributing about 1.6 ms/step/rank of the
2.0 ms theoretical saving.

## 9. Cheap depth-B follow-ups (fold into the same window)

- **n-gram prefetch / larger block**: the pageable n-gram read pays a ~24 ms
  wake after idle and two sequences consume blocks twice as fast; a prefetch
  or bigger read recovers ~10-24 ms/step on miss steps.
- **Batched sampling relay**: the two slots' relays are two rank-4 round trips
  (~6 ms total); one batched relay message saves ~3 ms/step.
- **Interleaved slot recording**: record both slots' layer work into one
  sequence so the block drains once per step instead of twice (~1.5 ms/rank =
  ~12 ms/step).

## 10. Next round

1. Wire the b2 MMV into `owner_record_layer`/`handle_decode_batch_work` for
   the 2-token case (layout + dispatch only; no new flags), behind the
   existing depth-B construction.
2. Switch the decode expert path to the union work (one work, `token_count=2`,
   sorted pairs) and measure `DECODE_BATCH_WORK_MS`/`OWNER_TXN_MS` deltas per
   phase with `FG_DECODE_MS`/`FG_DECODE_PROFILE`.
3. Interleave the slot recording and batch the sampling relay; prefetch the
   n-gram block.
4. Gates before promotion: `correctness64.ps1` `[12]`/`[Paris]`,
   `tools/pi-stability.ps1` soak PASS (ledger + 32K band), battery no
   regression, **B=1 byte-identical**, `depth-b-selftest --depth 2` parity
   (tokens/logits/state/QSA), `--abort-step` rollback PASS, and the depth-B
   selftest perf fields (`b2_decode_speedup` >= the promoted factor).  Keep
   the rollback path: the batch path stays test-only until the gates pass, and
   `FG_DECODE_RING` remains the opt-out.

## 11. Files

- Round-1 prototype (branch `feat/batch-block-r1`):
  `shaders/fg_dense_q8_0_cooked_r8_b2.comp`, `fg_vk_dense_q8_0_b2`
  (`src/vk.c`, `include/fg_vk.h`), `tests/test_batch_block.c`,
  `make tests/test_batch_block`.
- Next round touches `src/owner.c` (`owner_record_layer`),
  `src/runtime.c` (`handle_decode_batch_work`), `src/expert.c`
  (multi-token decode work), `src/ngram.c` (prefetch), `src/runtime.c`
  (sampling relay).
