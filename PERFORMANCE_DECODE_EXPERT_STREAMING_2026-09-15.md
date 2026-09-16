> **Fleet verdict (2026-09-16): the two kernel changes below were REJECTED and
> reverted.**  Measured on the fleet they cost +12/+16 VGPRs and -2/-4
> waves/SIMD, and the pair median rose to 1.377 ms/block against a same-session
> control of ~1.283.  The instruction cuts were real; the occupancy loss
> dominated.  Reverts: `728d702` (gate/up prefetch), `ab03657` (two-row
> down/reduce).  The rate tool, the `FG_BENCH_EXPERT_DECODE` probe and this
> accounting stay.  Salvage round and the promoted configuration:
> `PERFORMANCE_DECODE_EXPERT_OCCUPANCY_2026-09-16.md`.

# Decode Expert Pair: Streaming Round (2026-09-15)

Worktree `D:\looking-glass-labs\fg-work-exp13`, branch `perf/expert-streaming`
from main `405c3f9`.  Scope: the batch-1 fused expert pair (`gate_up` +
`down_reduce`) on the ring decode path.  Shader-only plus a local width probe;
no vk.c / runtime.c / owner.c change, no pack or manifest change, so the
qualified pack and the stored expert metadata are untouched.

| commit | change |
|---|---|
| `61b1fe2` | two-row down/reduce geometry, paired uvec2 scale and high-bit reads |
| `1da0af1` | next-block K-quant prefetch in the gate/up pair loop, width probe bench |
| `58501e1` | tools: decode expert achieved-GB/s table from `FG_DECODE_PROFILE` logs |

## 0. TL;DR

* The pair runs at **158-184 GB/s** on the fleet (rank-0 six-layer block,
  1.001-1.161 ms over 80 decoded tokens); the audit's 133-217 GB/s band is that
  number plus session/thermal spread.  `gate_up` is the larger half
  (0.606-0.863 ms, 110.6 MB) and `down_reduce` the smaller
  (0.395-0.563 ms, 73.7 MB).
* **The epilogue is not the problem**: the fused decode graph dispatches only
  the two GEMMs (`fg_moe_reduce` / `fg_swiglu` run only on the legacy
  five-dispatch path, and the shared-expert SwiGLU is 0.005 ms per block).
  The whole deficit is inside the two GEMMs, and inside them the weight stream
  is already 100% sector-efficient.
  The overhead is the **instruction mix around the stream**: 8 of 15.5
  load instructions per gate/up lane-block are broadcast metadata scalars, and
  the Q5_1 down path spent 4 input loads per 16 values for 8 B of weights.
* Two changes, both bit-identical to the round-8 kernels (verified by
  `FG_EXPERT_PARITY_DUMP` diff over all 17 oracle combinations):
  * down/reduce: 32 lanes, **two adjacent rows per lane** share one scale
    word, one fifth-bit word and one activation read.  Load instructions per
    layer drop **224,000 -> 128,000 (-43%)** on Q5_1 and **-33%** on cooked
    Q8_0, and requested L1 bytes drop ~40%.
  * gate/up: the next block's two K-quant words are **issued one block early**
    (rolling double buffer) so a wave keeps the DRAM stream in flight while it
    decodes the current block.  Load count unchanged; +~12 loop-carried values.
* Expected: **+8-20%** on the pair (conservative) to **+30%** if the load
  window was the binding constraint.  That is **-0.5 to -2.0 ms/token**
  fleet-wide, i.e. **+0.3 to +1.3 TPS**, with the AB as the only arbiter.
* Blocker: no AMD device and no working AMDGPU/ACO compiler is reachable from
  this worktree, so the register/occupancy proxy could not be measured.  Both
  changes are register-positive (~+8-12); if the fleet shows an occupancy
  regression the pair experiment's lesson applies and the gate/up prefetch is
  the first revert candidate.

## 1. Achieved-GB/s table (baseline, measured)

Source: `ab-20260913-204953` rank-0 log, `scope=expert_decode`, `calls=6`
(one six-layer block per token), 80 decode tokens, priced with
`tools/fg_expert_rates.py` (weights only, manifest-exact per-rank layer mix:
rank 0 owns layers 6-11, all Q4_K/Q4_K/Q5_1).

| kernel | MB/block | ms min | ms median | ms mean | ms max | GB/s min(ms) | GB/s median | GB/s mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fg_moe_decode_gate_up` | 110.59 | 0.606 | 0.668 | 0.703 | 0.863 | 182.4 | 165.6 | 157.3 |
| `fg_moe_decode_down_reduce` | 73.73 | 0.395 | 0.437 | 0.458 | 0.563 | 186.7 | 168.7 | 161.0 |
| **pair** | **184.32** | **1.001** | **1.105** | **1.161** | **1.423** | **184.1** | **166.8** | **158.8** |

Fleet roll-up: 1,504.3 MB of expert weights per token (audit section 2), so
the measured pair rates imply **8.8 ms/token** of expert time at the median
(the audit's 6.8-10.7 ms band) and **5.9-6.7 ms/token** at 250-280 GB/s.

Control scopes in the same profile (per six-layer block).  The fused graph
dispatches only the two GEMMs, so there is no routed-expert epilogue line to
price; the schedule and router dispatches are shown to bound what is left
outside the pair:

| scope | kernel | ms/block | note |
|---|---|---:|---|
| `expert_decode` | `fg_moe_decode_gate_up.spv` | 0.668 | the pair, larger half |
| `expert_decode` | `fg_moe_decode_down_reduce.spv` | 0.437 | the pair, smaller half |
| `expert_decode` | `fg_decode_tile_schedule.spv` | 0.007 | schedule build, 10 words |
| `expert_decode` | `fg_router_top10.spv` | 0.072 | routing, 512 experts |
| `shared_expert` | `fg_swiglu.spv` | 0.005 | the *shared* expert, not the routed pair |

The legacy five-dispatch fallback (`FG_DECODE_EXPERT_LEGACY`) is what prices
`expert_activation` (0.46 MB/block) and `expert_reduce` (0.68 MB/block) in
`tools/fg_expert_rates.py`; neither is dispatched on the qualified fused path.
So the ranked work is entirely the two GEMMs, and the routed epilogue is
already 0.5% of the pair at most.

## 2. Where the loads are inefficient

### 2.1 gate/up (production Q4_K gate + Q4_K up, 640x2560, ten slots)

Per lane per 256-value block (16 lanes of weight per row-tile, eight lanes per
row, 4 rows per wave):

| load | width | instructions | unique bytes per row-block | sector use |
|---|---|---:|---:|---|
| gate quant word | `uvec4` | 1 | 128 | 4/4 (wave = 512 B contiguous) |
| up quant word | `uvec4` | 1 | 128 | 4/4 |
| `d`/`dmin` word | `u32` | 2 | 8 (broadcast x8 lanes) | 1/1 |
| scale/min words | `u32` x3 | 6 | 24 (broadcast x8 lanes) | 1/1 |
| q8_k quant words | `uvec2` | 4 | 256 (4 rows share) | 8/8 |
| q8_k block delta | `u32` | 1 | 4 | 1/1 |
| q8_k group sums | `uvec2` | 0.5 | 32 / 8 lanes | 1/1 |

The **weight stream is fully coalesced** (512 B per wave instruction, 100%
sector use) and cannot be widened: the cooked metadata runs are 12 B per row
and the q8_k pairs are 8 B aligned.  The inefficiency is the **8 broadcast
scalar metadata loads per lane per block** (52% of all load instructions,
32 unique bytes per row-block).  They are L1-resident, so this is an issue-slot
and LSU cost, not DRAM.

### 2.2 down/reduce Q5_1 (2560x640, ten slots, 20 blocks)

Per lane per block, *before*:

| load | width | instructions | unique bytes per row-block |
|---|---|---:|---:|
| scale+min | `u32` | 1 | 4 (broadcast x4 lanes) |
| fifth bits | `u32` | 1 | 4 (broadcast x4 lanes) |
| quant half-block | `uvec2` | 1 | 16 |
| activation | `vec4` | 4 | 128 (16 rows re-read the same 64 B four ways) |

7 instructions per **16 values** = 0.44 loads/value; 64 B of activation per
8 B of weights.  A wave's quant request is two 128 B contiguous chunks, so
sector use is fine; the cost is that the activation is re-read once per output
row (16 rows per workgroup, 4 lanes each) and the scale/high words are
per-row scalars.

### 2.3 down/reduce cooked Q8_0 (2560x640, ten slots)

Before: 3 instructions per 4 values (0.75 loads/value): one `u32` scale
(broadcast x8), one `u32` quant word, one `vec4` activation.

## 3. What changed

### 3.1 `61b1fe2` down/reduce: two rows per lane

`layout(local_size_x = 32)`, four lanes per output row, **two adjacent rows per
lane** for the same (parity, half) split as before.  A row pair's scale/min
words and fifth-bit words are adjacent, so one `uvec2` load feeds both rows
(`wv[sm_uvec]`, `wv[hw_uvec]`), the two quant halves are two aligned `uvec2`
loads, and the four activation `vec4` loads are shared by both rows.  The row
order, the two-step dot accumulation, the per-slot fma chain and the four-way
row reduction are unchanged, so every output bit matches the previous kernel.
The cooked Q8_0 branch gives each lane two four-value words of a row block
(one aligned `uvec2`) and two accumulators, so its loads per value drop from
0.75 to 0.5.

Accounting per layer (wave-level load instructions, Q5_1):
**224,000 -> 128,000 (-42.9%)**; per 32 values the lane now issues 8 loads
(2 quant `uvec2`, 1 scale `uvec2`, 1 fifth-bit `uvec2`, 4 activation `vec4`)
instead of 14.  Requested L1 bytes per layer drop 81.9 -> 49.2 MB (-40%;
this is request traffic including the broadcast repeats, not DRAM bytes).
Cooked Q8_0: **384,000 -> 256,000 (-33.3%)**.

### 3.2 `1da0af1` gate/up: one-block-ahead K-quant prefetch

The block loop now carries the projection words in a rolling double buffer:
block b+1's two `uvec4` (plus the Q5_K high-bit words on the compile-time Q5
paths) are issued at the top of block b's body, so the DRAM request for the
next block overlaps the current block's decode and SwiGLU math instead of
sitting behind the loop back-edge.  The tail load is guarded by
`block + 1u < pc.blocks`.  Load counts, addresses and arithmetic are
unchanged; the walked cursors replace the per-block `block * step` IMADs.

## 4. Expected effect

| pair rate | ms/block | fleet expert ms/token | delta vs median | TPS delta at ~24 TPS |
|---:|---:|---:|---:|---:|
| 167 GB/s (baseline median) | 1.105 | 8.84 | - | - |
| 185 GB/s (baseline best) | 1.001 | 8.01 | -0.83 | +0.5 |
| 200 GB/s (+20%) | 0.922 | 7.37 | -1.47 | +0.9 |
| 250 GB/s (+50%) | 0.737 | 5.90 | -2.94 | +1.8 |
| 300 GB/s (+80%) | 0.614 | 4.91 | -3.93 | +2.4 |

The conservative band (load-issue and L1 pressure were part of the limit)
is 200-230 GB/s, i.e. **-1.5 to -2.5 ms/token**; the optimistic band (the pair
was latency-bound and the prefetch plus the halved load count land) is
250-280 GB/s, i.e. **-3 to -4 ms/token**.  The mission's 250-300 GB/s is the
upper half of that range and is not claimed before the A/B.

## 5. Local evidence

* Bit-identity: `FG_EXPERT_PARITY_DUMP` diffs of the fused rows for all 17
  `expert_decode_fused_random` combinations (seeds 0-3 x {12/13,13/12,12/13
  cooked,12/12 raw}, seed 7 x 8/8/8 cooked) are byte-identical between the
  round-8 kernels and the new pair.  Baseline hashes match the round-8
  deployment record: gate_up `6c6826a7...`, down `45f800f9...`; new
  gate_up `f84cb6b9...`, down `6c296c94...`.
* Oracles: `expert_decode_fused(12/13)`, `expert_decode_fused_q8_0(12/13)`,
  `expert_decode_fused_q8_0_cooked(12/13)`, `expert_decode_fused_q8_gates`,
  `expert_decode_fused_random`, `expert_graph_fused(7/8)`,
  `expert_graph_fused_q8_0_cooked`, `expert_graph_replay`,
  `expert_graph_rejects_overlap`, `kquant_cooked`, `q5_1_down_cooked`,
  `q8_cooked_prefill_parity`, `q8_cooked_prefill_sweep`,
  `grouped_kquant_prefill(12/13)`, `grouped_down_prefill`,
  `q8_cooked_token_tiles`, `decode_tile_schedule`, `router_and_expert_packing`,
  `moe_prefill_scatter_reduce`, `test_expert_prefill`, `test_owner_reduce` all
  PASS on llvmpipe.  The known llvmpipe `qsa_record_commit` crash and the
  pre-existing `test_core:457`, `test_session:254/257`,
  `test_prefill_dispatch:178-183` failures are unchanged.
* New probe `FG_BENCH_EXPERT_DECODE=<iterations>`
  (`expert_decode_pair_benchmark`) runs the production shapes through the fused
  entry points and prints per-iteration GPU us and GB/s for both down layouts.
  llvmpipe is instruction-bound and ~25x slower than the blade, so its medians
  (gate/up ~2.67 vs ~2.74 ms, down ~1.95 vs ~1.98 ms over six alternating
  runs) only show "no regression"; no bandwidth conclusion is drawn from them.
* Static counts (spirv-dis, whole module): gate/up 2728 -> 2796 ops,
  129 -> 141 `OpLoad` (four loop instances x the two prefetch loads plus the
  Q5 high-bit variants); down 464 -> 655 ops, 61 -> 76 `OpLoad` (the same
  module now covers a row *pair* per lane).

## 6. Fleet A/B plan

Deploy is shader-only: replace `vulkan/fg_moe_decode_down_reduce.spv` and/or
`vulkan/fg_moe_decode_gate_up.spv` in the distributed shader directory; the
binary, pack and manifest are unchanged.  Both commits can be A/B'd
independently (disjoint files) or together.

1. Control: current qualified shaders (gate_up `6c6826a7`, down `45f800f9`).
2. Down only: `61b1fe2`'s `fg_moe_decode_down_reduce.spv` (`6c296c94`).
3. Gate/up only: `1da0af1`'s `fg_moe_decode_gate_up.spv` (`f84cb6b9`).
4. Both: HEAD.

Per configuration: correctness gates, then the attach battery, then one
`FG_DECODE_PROFILE=1` run on rank 0 (or all ranks) to read the expert scopes:

```
DECODE_PROFILE_KERNEL rank=0 scope=expert_decode kernel=fg_moe_decode_gate_up.spv calls=6 gpu_ms=...
DECODE_PROFILE_KERNEL rank=0 scope=expert_decode kernel=fg_moe_decode_down_reduce.spv calls=6 gpu_ms=...
python3 tools/fg_expert_rates.py <rank-0 log>        # MB, ms, achieved GB/s
FG_BENCH_EXPERT_DECODE=5 (local probe, any rank, llvmpipe only)
```

Promotion bar: gates green, 4K prefill unchanged, the rank-0 pair median below
1.00 ms/block (>=185 GB/s) with neither kernel above its control, and a
32k-context spot check unchanged.  If the pair improves but one kernel does
not, keep the winner and revert the other; if either kernel shows an occupancy
regression (its own gpu_ms up while the other improves), revert that commit and
record the ACO VGPR count from `RADV_DEBUG=shaderstats` before the next
attempt.

## 7. Blockers, risk and ranked next steps

* **Register risk (main unknown).** No AMD device and no AMDGPU compiler is
  reachable here (WSL has no sudo for `clang-18`/`llvm-spirv-18`), so the
  VGPR/occupancy proxy could not be measured.  The down geometry adds a second
  accumulator plus the pair's word registers (~+8-10), the gate/up prefetch
  adds ~12 loop-carried values (4 walked cursors + one word pair per
  projection; the production Q4/Q4 instance carries no fifth-bit words).  The
  pair/wave-split experiments showed instruction wins can lose to occupancy, so
  the A/B must read both kernels' scope times separately.
* If the pair does **not** move, the two remaining structural levers are
  (a) **slice-paired gate/up** (one lane carries both halves of a group pair,
  sharing all eight metadata words: -35% load instructions per MAC, ~+20
  VGPRs) and (b) **four rows per down lane** (input reads halve again, grid
  drops to 80 workgroups).  Both are register-positive and need the blade's
  VGPR counts first.
* The pack-format lever (row-interleaved or block-major cooked layouts) would
  remove the 12 B metadata runs and the 16 B per-row quant strides entirely,
  but it is a pack rebuild round and is not attempted here.
