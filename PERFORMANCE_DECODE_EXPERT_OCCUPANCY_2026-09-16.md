# Decode Expert Pair: Occupancy Round (2026-09-16)

Worktree `D:\looking-glass-labs\fg-work-exp13`, branch `perf/expert-streaming`.
This round salvages the rejected streaming round under the fleet's measured law:
**never cross a wave/occupancy boundary, and spend the register budget only
where the grid is already too small to use it.**

## 0. TL;DR

| config | gate_up | down_reduce | pair | pair GB/s | short / 4K decode |
|---|---:|---:|---:|---:|---|
| control (round-13 default) | 0.578 ms / 191 GB/s | 0.454 ms / 162 GB/s | 1.033 ms | 178.4 | 24.50 / 23.68 TPS |
| **promoted** | **0.506 / 218.6** | **0.356 / 206.8** | **0.862** | **213.8** | **24.84 / 24.39** |

* Pair median **-16.6%** same-session (1.033 -> 0.862 ms/block, 178 -> 214 GB/s),
  both kernels faster, gates `[12]`/`[Paris]`, soak PASS (failures=0, 8/8 ranks).
* Two commits:
  * `0cda056` splits the **test-only Q8/mixed fallback** of gate_up by projection:
    72 -> **64 VGPRs, 14 -> 16 waves**, instruction count unchanged (2627).
  * `054b6b1` batches **four down_reduce K-blocks per lane iteration**:
    48 -> 64 VGPRs but the down grid is only 13.3 waves/CU, so the extra
    registers are occupancy-free; 162 -> 207 GB/s.
* Rejected in this round: block-unroll-8 (1449 instr, 189 GB/s, pair 0.954),
  the two-pass gate/up loop (+12% instructions, still 64 VGPRs), the
  elected-load+shuffle scalar broadcast (48 VGPRs, +13 instructions, no gain),
  and every attempt to move down below 48 VGPRs.
* Fleet left on the promoted config: live dir `20260915-mem12`, binary
  `4c63f43998cda1c0`, gate_up `22a4450d0ef8f456`, down `42da68fe7ce3c714`,
  no debug env, gates and soak green.

## 1. The rejection this round recovers from

Round 13 measured the streaming round against a same-session control
(`results/round13-combined-20260915-203419/SUMMARY.md`):

* gate_up 72 -> 84 VGPRs (14 -> 12 waves), down_reduce 48 -> 64 (20 -> 16).
* pair median 1.377 vs control ~1.283 ms/block; the -43% load instructions and
  -40% L1 bytes were real but the wave loss dominated.

The reverts (`728d702`, `ab03657`) restore the round-8 kernels; the rate tool
and the `FG_BENCH_EXPERT_DECODE` probe stay.

## 2. Control and the probe loop

Control was re-measured on the fleet the same night (rank 0, 38 profiled
tokens, `FG_DECODE_PROFILE=1` plus `RADV_DEBUG=shaderstats,nocache` in the
rank-0 start script):

* gate_up 72 VGPRs / 14 waves / 2633 ACO instructions, 0.578 ms median.
* down_reduce 48 / 20 / 502, 0.454 ms median. gate_up and down are the 2nd and
  3rd `Compute Shader:` blocks in the rank-0 log; `Subgroups per SIMD` is the
  wave figure.

The probe loop that made this round cheap: variants are generated from the
control source, built here with the worktree glslang, checked for bit identity
with the `FG_EXPERT_PARITY_DUMP` diff over all 17 oracle combinations, then
rebuilt with the **blade's** glslang 16.2 (the serving toolchain) and measured
by running `tests/test_fg_vk` with `RADV_DEBUG=shaderstats,nocache` while the
ring is quiesced. A VGPR/instruction reading costs ~1 minute; only a promotion
candidate pays for a full restart + gates + battery.

## 3. What won and why

### 3.1 gate_up: the fallback path owned the register count

Removing the Q8/mixed fallback proved it: 72 -> 64 VGPRs with only the K-pair
path left. The production gate/up types (Q4_K, Q5_K) always take the
specialised K-pair path, so the fallback is exercised only by the Q8 oracles
and by `expert_decode_fused_random`; it was nevertheless setting the whole
kernel's occupancy. Splitting it into a gate branch and an up branch (at most
one projection is K-active when the fallback runs) drops the kernel to 64
VGPRs with **2627 vs 2633 instructions** - the split costs nothing and the
kernel crosses 14 -> 16 waves.

Neither further register cut moved it: the K-pair path alone is also 64; the
late scale/min decode is 64; the two-pass (gate loop then up loop) is 64 at
+12% instructions. 64 is the floor for this source shape and ACO.

### 3.2 down_reduce: batch the K blocks, spend registers where the grid is small

The down kernel runs 160 workgroups x 2 waves = 320 waves over 24 CUs, i.e.
**13.3 waves per CU**, while 48 VGPRs already allow 20 waves per SIMD. The
kernel is grid-limited, not register-limited: every register up to ~76 is free.
Four K-blocks per lane iteration therefore costs 48 -> 64 VGPRs at zero
occupancy, doubles the in-flight weight bytes per wave, and lands at 207 GB/s
(+28%). Eight blocks over-unrolls (1449 instructions, worse scheduling,
189 GB/s); four is the knee.

### 3.3 The gate_up gain in the down-4 config is a cross-kernel effect

gate_up's shader is byte-identical across the variant ladder, yet its median
moved 0.564 -> 0.505 ms when down switched from unroll-2 to unroll-4, and back
to 0.565 with unroll-8. Reproduced exactly across two runs of the same process
(73 profiled tokens, medians identical to three decimals). The most plausible
mechanism is DRAM: the batched down stream issues larger contiguous runs, so
all eight ranks contend less and the following gate_up dispatch inherits a
cleaner memory system. Whatever the mechanism, the effect is stable and the
promoted numbers are measured on a fresh restart, not on a warm session.

## 4. Variant ladder (rank 0, 6-layer block, 38-73 profiled tokens)

| variant | gate_up VGPR | down VGPR | gate_up med | down med | pair med | GB/s |
|---|---:|---:|---:|---:|---:|---:|
| control | 72 / 14w | 48 / 20w | 0.578 | 0.454 | 1.033 | 178.4 |
| gate_up lean fallback | 64 / 16w | 48 | 0.564 | 0.454 | 1.018 | 181.1 |
| + down unroll-2 | 64 | 64 | 0.564 | 0.442 | 1.006 | 183.2 |
| **+ down unroll-4** | 64 | 64 | **0.505** | **0.355** | **0.860** | **214.3** |
| + down unroll-8 | 64 | 64 | 0.565 | 0.389 | 0.954 | 193.1 |
| (reprofile of unroll-4, fresh restart) | 64 | 64 | 0.506 | 0.356 | 0.862 | 213.8 |

Rejected experiments (VGPR probes only): two-pass K-pair 64 VGPRs at 2961
instructions; address-trimmed/late-scale K-pair 64 at 2621; elected-load +
`subgroupShuffle` scalar words 48 VGPRs at 515 instructions; raw-Q8 path
removed 48; cooked-Q8 path removed 48. The last two show the down Q5_1 loop
alone sets its 48-VGPR floor, and no bit-identical reordering moved it.

## 5. Local evidence

* Bit identity: `FG_EXPERT_PARITY_DUMP` diff over all 17
  `expert_decode_fused_random` combinations is byte-identical to the round-8
  control for both promoted shaders (and for every measured variant).
* Oracles: `expert_decode_fused*`, `expert_graph_fused*`,
  `expert_decode_fused_random`, `q5_1_down_cooked`, `kquant_cooked`,
  `q8_cooked_prefill_*`, `grouped_*`, `test_expert_prefill`,
  `test_owner_reduce`, `test_qsa_prefill` PASS on llvmpipe; `make all -j8`
  warning-free. Known pre-existing failures unchanged.
* Fleet gates `[12]`/`[Paris]` on every deployed variant; final soak
  (`tools/pi-stability.ps1`) PASS: stages=6 conversation=4 correctness=2
  ranks=8 failures=0, 4K prefill 258.2, short decode 23.0.
* Deploy identifiers (blade glslang 16.2 builds):
  gate_up `22a4450d0ef8f456...`, down `42da68fe7ce3c714...` on all 8 blades,
  binary `4c63f43998cda1c0...` unchanged.

## 6. What did not work, and why it matters

* Occupancy alone is not the lever: gate_up +2 waves bought +2.3%; the real
  gains came from the down kernel's memory-level parallelism and from the
  cross-kernel DRAM effect.
* Register-positive prefetching and two-row sharing remain refuted; the
  fallback split is the only register cut that survived ACO.
* The down kernel's 48-VGPR Q5_1 loop is an ACO allocation floor for the
  control geometry; the way to buy its speed was to spend registers on
  batching, not to save them.

## 7. Ranked next steps

1. **Batch the gate_up K-pair path the same way** - if the down unroll's DRAM
   locality is the real mechanism, a two-block gate/up batch may pay even at
   its register cost (gate_up is at the 16-wave boundary, so it needs either
   -8 VGPRs first or a measured 64 -> 72 trade). Not attempted here.
2. **Unroll-4 with a two-block tail** (4+4+2 is 4+4+2 today; try 6+4 or 4+6)
   to see whether the knee is the count or the tail shape.
3. **Down raw-Q8/cooked-Q8 batching** - both branches are unchanged and still
   use single-block loads; the cooked Q8 path covers five production layers.
