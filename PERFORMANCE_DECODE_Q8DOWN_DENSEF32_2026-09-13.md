# Ring Decode: Cooked-Q8 Expert Down and Dense F32 (2026-09-13)

Worktree `fg-work-dec4`, base `929bb62`. Scope: the ring decoder's per-block
GPU time, attacking the two named items - the cooked-Q8 expert pair and the
batch-1 Q8 matvec chain - without touching ring routing, placement, or
`runtime.c` dispatch.  Binary `0fdb5209fde82f152f93ce107f56844dc5e15bac`
(deployed to all eight blades, `start-rank0-ring.sh` / `start-workers-ring.sh`),
commit `3b6196b`.

## TL;DR

Two kernels that were well off the memory roofline are fixed:

- `fg_moe_decode_down_reduce` on the cooked-Q8_0 expert layers (2/4/30/46/47)
  ran the q5_1 geometry against a row-major quant run: each warp touched eight
  640B-strided half-sectors for 128 bytes used, ~42 GB/s.  The branch now maps
  eight lanes per row onto aligned four-value words so each row-block is one
  fully used 32B sector.  **Rank 1 per-block down time 1.249 -> 0.578 ms,
  rank 5 0.759 -> 0.455, rank 7 1.075 -> 0.474** (Vulkan timestamps, 4K steady
  state).
- `fg_dense_f32` (router, shared scalar, GDN controls; 22 calls/block) replaced
  its six-barrier shared tree and serial FMA chain with `subgroupAdd` plus one
  cross-subgroup slot and four accumulators.  **0.35-0.43 -> 0.127-0.132 ms per
  block on every rank**, which is now close to the 31 MB/block traffic floor.
- `fg_qsa_attention_split` drops a redundant third per-token barrier (small,
  within noise on the fleet).

Fleet gates stay green: `/no_think` arithmetic -> `12`, capital -> `Paris`,
prefill 273-304 TPS, `tools/pi-stability.ps1` full PASS (six soak stages,
four-turn conversation, both correctness probes, all eight ranks alive, perf
band PASS).  Short 32-token decode moves 19.83 -> **20.5-20.9 TPS** and 4K
first-token decode 16.74 -> **18.5-18.8 TPS** at comparable thermal state; a
40-token 4K listing measures 15.5-17.4 TPS (the blades drift with soak heat;
the per-block kernel deltas below are the reproducible part).

## Per-kernel before/after (Vulkan timestamps)

`FG_DECODE_PROFILE=1`, 24 consecutive blocks after a 4K request; the changed
kernels are compared against their unchanged neighbours in the same run to
control for clock/thermal drift.

| kernel (per 6-layer block) | rank | before | after | delta |
|---|---:|---:|---:|---:|
| `fg_moe_decode_down_reduce` | 1 | 1.249 ms | **0.578** | -54% |
| `fg_moe_decode_down_reduce` | 5 | 0.759 ms | **0.455** | -40% |
| `fg_moe_decode_down_reduce` | 7 | 1.075 ms | **0.474** | -56% |
| `fg_dense_f32` | 1 | 0.380 ms | **0.132** | -65% |
| `fg_dense_f32` | 2 | 0.354 ms | **0.127** | -64% |
| `fg_dense_f32` | 7 | 0.303 ms | **0.125** | -59% |
| `fg_qsa_attention_split` | 2 | 0.505 ms | 0.520 | noise |
| block GPU total | 1 | 6.432 ms | **5.646** | -0.79 |
| block GPU total | 2 | 5.125 ms | **5.057** | -0.07 |
| block GPU total | 5 | 5.778 ms | **5.406** | -0.37 |
| block GPU total | 7 | 5.902 ms | **5.354** | -0.55 |

All 44 `fg_dense_q8_0_cooked_r8` calls per block are unchanged; the large
shapes are at the roofline already and were not tuned: GDN qkv 277-278 GB/s,
z (row-4 cooked) 324 GB/s, GDN output 172 GB/s, GR up 174 GB/s, shared expert
172-174 GB/s, expert gate/up 135-200 GB/s, q5_1 down 154 GB/s.  The measured
per-blade stream ceiling is ~350 GB/s (clpeak 353).

## Where a 4K token goes now (no VK profile, `FG_DECODE_RING_TRACE`)

Rank-0 trace, mean of the last 24 tokens of a 4K/40-token request:

| phase | ms |
|---|---:|
| embed + ngram | 0.25 |
| rank 1 block (layers 0-5) + hop | 7.29 |
| rank 0 own block (6-11) | 6.69 |
| ranks 2-7 (6 blocks serial) | 41.37 |
| output head / tail (direct handoff) | included above |
| **token total** | **57.9** |

Worker block walls without profiling (`RING_DECODE_BLOCK`): rank 1 6.23,
rank 2 5.92, rank 3 6.16, rank 4 5.88, rank 5 6.19, rank 6 5.66, rank 7 5.49
(write 0.002, read 0.011, egress 0.09 each).  The gap between a block wall and
its GPU time is ~0.6 ms of host recording, submit and fence time.

## Remaining gap to 10 ms/token

The 4K steady token is 57.9 ms (17.3 TPS); 10 ms is **5.8x** away, and 35.7 ms
(28 TPS) is 1.62x away.  The budget in measured order:

1. **Weight streams are the floor.** Per rank the block streams ~400-600 MB of
   q8/q5 weights; at 350 GB/s that is 1.2-1.7 ms, but the instruction-bound
   kernels (expert gate/up/down, QSA attention, the small-K r8 shapes) run at
   only 40-200 GB/s on an ISA without `dp4a`.  The expert pair alone is
   ~9.6 ms/token across ranks at ~168 GB/s; a perfect kernel would save ~5 ms.
2. **QSA decode attention** is ~5 ms/token across the ten QSA layers and is
   DRAM-latency-bound on scattered 34-byte record reads with a serial
   online-softmax chain per token; the 4-token batching experiment regressed
   previously, and a head-sharing rewrite was not attempted here.
3. **Fixed rank-0/4 tail**: output head ~3.4 ms/token (1.93 ms of it is the
   675 MB vocabulary GEMM at roofline), embed ~0.25, hops ~0.7.
4. **Host recording** ~0.6 ms/block (4.8 ms/token): descriptor sets are already
   cached per dispatch index, so this is mostly the 220 `vkCmd*` calls plus
   submit/fence latency per block, not descriptor rewrites.

## Fleet evidence

| metric | before (c1ca8354) | after (0fdb5209) |
|---|---:|---:|
| 4K prefill (battery) | 272-304 TPS | 278-304 TPS |
| short decode 32 tok (battery) | 19.83 | **20.50 / 20.91** |
| 4K first decode token | 16.74 | **18.53 / 18.79** |
| 4K sustained 40-token listing | 16.74 (state) | 15.55 / 17.43 (thermal band) |
| pi-stability | PASS | PASS (short_decode 17.90, 4k_prefill 281.87) |
| gates | `[12]` / `[Paris]` | `[12]` / `[Paris]` |

## What changed

- `shaders/fg_moe_decode_down_reduce.comp`: cooked-Q8_0 branch rewritten to
  the 8-lane-per-row mapping (one aligned word per lane per block, 32B sector
  per row-block, no cross-block funnel); q5_1 and raw paths untouched.
  `src/vk.c` dispatches `(out+7)/8` groups for that branch in both the chained
  path and the fixed expert graph.  `expert_decode_fused*` oracles pass.
- `shaders/fg_dense_f32.comp`: subgroup reduction, four-way unrolled
  accumulators with a 64-stride tail; `dense_f32_and_silu` passes.
- `shaders/fg_qsa_attention_split.comp`: one barrier per token removed.

Build is warning-free (`make all -j8`, gcc `-Wall -Wextra -Wpedantic -Werror`);
the release tree on `.42` was rebuilt from the full worktree source because the
previous partial patches had left stale headers (`fg_vk.h` lacking
`FG_VK_TENSOR_FORMAT_Q8_0_EXPERT_COOKED`) and stale `expert.c`/`model.c`/
`pack.c` relative to the deployed binary.
