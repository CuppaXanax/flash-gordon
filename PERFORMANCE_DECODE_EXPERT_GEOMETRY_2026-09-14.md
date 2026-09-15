# Decode Expert Geometry: Gate/Up + Down/Reduce Instruction-Diet Round (2026-09-14)

Worktree `D:\workspace\fg-work-exp8`, branch
`perf/decode-expert-geometry` from main `b721ceb`.  Scope: the batch-1 fused
expert pair (`fg_moe_decode_gate_up` + `fg_moe_decode_down_reduce`) on the
ring decode path.  No vk.c / runtime.c / owner.c change, no pack or manifest
change, no interface change: the same SPIR-V files are replaced, so the
same pack qualifies.

Commits:

| commit | change |
|---|---|
| `b92e192` | unrolled, type-specialized gate/up K-quant pair + random parity oracle |
| `7bf4b87` | unrolled down/reduce cooked paths with walked addresses |
| `0aeb731` | drop the unreachable down/reduce cooked-Q8 duplicate |

Fleet target: expert pair 1.42 ms/block (0.86 gate_up + 0.56 down) -> <=0.85
ms/block, i.e. ~150 -> ~260 GB/s effective.  This round removes the
instruction overhead that sat between the memory stream and the MACs; it does
not change the pack layout or the arithmetic meaning.

## 1. Load-pattern accounting that motivated the round

Production shapes: gate/up `out=640, in=2560` (10 K-blocks), down
`out=2560, in=640`; 10 routed experts per layer, all local on the ring block
owner; 48 layers, 6 per rank block.

### 1.1 Cooked layouts and access width

* Q4_K tile (gate): 8 rows per tile; per 256-value block a row has 4 B
  `d/dmin` at `tile_row*4`, 12 B packed scale/min at `32 + tile_row*12`, and
  128 B quant at `128 + tile_row*128`.  A lane's uint4 covers one 32-value
  slice of a row-block (16 B, the two nibble halves of a group-pair).
* Q5_K tile (up): same plus a 32 B fifth-bit plane at
  `128 + tile_row*32`; one extra uint4 per lane per block.
* Q5_1 down tile: 16 rows; per 32-value block a row has 4 B `d/m` at
  `tile_row*4`, 4 B fifth bits at `64 + tile_row*4`, 16 B quant nibbles at
  `128 + tile_row*16`.  A lane part covers 16 values as a uvec2.
* Q8_0 cooked down tile: 16 rows; 2 B f16 scale per `[block][row]`, 32 B
  quant per row-block, read as one aligned word per lane (4 values).

Every production load is naturally aligned at its access width (16 B uint4,
8 B uvec2, 4 B scalar).  The width cannot be widened further without changing
the pack: the metadata runs are 12 B (Q4/Q5) and 4 B (Q5_1) per row, the q8_k
activation header forces 8 B granularity for the `uvec2` reads, and the Q5_1
halves are 8 B.  The raw row-major Q8_0 path (unaligned word funnels) is not
reachable in production: the manifest cooks type-8 down experts, so only the
tests use it.

### 1.2 Loads per FMA and re-reads

Per gate_up lane and 256-value block (Q4 gate + Q5 up): 2 x 16 B weight
uint4, 4 x 8 B activation uvec2, 1 x 4 B activation delta, 4 + 4 scalar
metadata words, and (sub==0 lanes only) 1 x 8 B activation sums.  Against 64
FMA (two projections x 32 values) that is 0.23 load instructions per FMA.

The weights are never re-read across lanes: the eight lanes of a row cover
disjoint 32-value slices.  The activation and the per-row-block metadata are
broadcast eight ways (all lanes of a row load the same words); both stay
L1-resident.  The workgroup streams one contiguous 11.5 KB gate tile region
and one 14 KB up tile region per (row tile, slot); adjacent workgroups read
adjacent regions, so the DRAM access is sequential within an expert.

### 1.3 The actual overhead: dynamic indexing and rolled loops

SPIR-V disassembly (`spirv-dis` over the glslang `-Os` output; result-op
counts include labels/phis, which ACO mostly coalesces, but they bound the
issue stream):

* gate_up Q4 gate + Q5 up, per lane per block: the outer block loop carried
  599 result ops plus a 113-op word loop executed four times = **1051**.  The
  four weight words, two high words and eight activation words lived in
  function-local arrays indexed by the word loop, so each iteration had
  4 x (AccessChain + Load) of scratch traffic, plus two select chains for the
  word-to-activation-word mapping (16 phi/select ops per iteration).
* down Q5_1, per lane per block: a 28-op loop shell plus two 50-op word
  iterations = **128**.  The `uvec2` quant lived in a function-local vector
  read through a dynamic access chain.
* down Q8_0 cooked, per lane per block: **40**, of which 9 IAdd + 6 IMul were
  per-block address rebuilding, and the f16 scale multiplied all four quant
  words before the dot.

## 2. What changed

### 2.1 `b92e192` gate_up: unrolled, type-specialized pair loop

The Q4_K/Q5_K pair is now a compile-time-specialized straight-line loop, one
instance per (gate type, up type) combination behind the uniform
`pc.gate_type`/`pc.up_type` branches.  Named registers only (no function
arrays, no dynamic indexing), the eight activation words are converted once
and shared by gate and up, the word-to-activation select chains are gone
because the four words are unrolled, Q4_K never emits the fifth-bit merge,
and all per-lane offsets advance by a fixed per-block stride.

Accounting per lane per block (result ops): **1051 -> 443** for the
production Q4/Q5 pair (404 Q4/Q4, 443 Q4/Q5 and Q5/Q4, 484 Q5/Q5).  Loads are
unchanged (2 vector weight, 4 uint2 activation, 8 scalar metadata, 1 delta);
the 16 function-memory accesses per block are gone.  The Q8_0/cooked and
mixed-type cases keep the old generic loop.

Bit-exactness: every partial dot is an exact sum of small integers
(max |value| 32 x 31 x 128 < 2^24), so the unrolled accumulation order cannot
change a bit.  Verified: old fused kernel (spv `8c877b32`) and new fused
kernel (spv `6c6826a7`) produce byte-identical fused rows for all 17
random-routing combinations in the new oracle.

### 2.2 `7bf4b87` down: unrolled cooked loops, walked addresses

Q5_1: the two weight words, their fifth-bit slices and the two input pairs
are unrolled from one `uvec2`; the scale/min broadcasts are constructed once
per block; the scale word, high word, quant uvec2 and input vec4 offsets
advance by fixed strides (192 words, 192 words, 96 uvec2, 16 vec4 per two
blocks) from a parity-adjusted base.  Q8_0 cooked: all four quant words share
one scale application (`slot_acc = fma(delta, dot(w,x), slot_acc)`), and the
addresses walk by 8 words / 8 vec4 per block.

Accounting per lane per block (result ops): Q5_1 **128 -> 87**, Q8_0 cooked
**40 -> 24** (IAdd 9 -> 4, IMul 6 -> 1).  Raw Q8_0 unchanged.

Numerics: Q5_1 is bit-identical to the old kernel (dump oracle, 0 of 4352
fused values differ across 17 combinations).  Q8_0 cooked differs by at most
1.2e-05 relative because the f16 block scale now multiplies the summed dot
instead of each quant word; the legacy-path comparison stays inside the
existing 3e-4 fused-pair tolerance.

### 2.3 `0aeb731` down: dead cooked-Q8 duplicate removed

The main-body `else if (pc.down_type == 8u && pc.tile_bytes != 0u)` branch is
unreachable (the same condition returns at entry).  Removing it cut the
disassembly 858 -> 742 lines, 7 -> 6 loops, and 77 dead result ops; the dump
output is byte-identical.

## 3. Expected effect (instruction-derived; fleet A/B is the gate)

The audit record (`PERFORMANCE_BLOCKK_2026-09-13` section "Where the block
time is now") classifies gate_up/down as instruction-bound, not DRAM-bound:
"each int8 MAC costs an unpack8, a convert, and an FMA" on GFX1013.  Under
that model the time tracks the dynamic issue stream:

| kernel | ops/block/lane | expected time | old measured |
|---|---:|---:|---:|
| gate_up (Q4/Q5) | 1051 -> 443 | 0.86 -> ~0.36 ms/block | 0.86 |
| down (43 Q5_1 + 5 Q8 layers) | 128 -> 87 / 40 -> 24 | 0.56 -> ~0.38 ms/block | 0.56 |
| **pair** | ~0.55x | **1.42 -> ~0.74 ms/block** | 1.42 |

That is 11.4 -> ~5.9 ms/token summed over the eight blocks, i.e. the <=7
ms/token mission target *if* the kernels are issue-dominated as classified.
The conservative band, if only half the time is issue and half is
latency/DRAM, is 1.42 -> 0.95-1.15 ms/block.

Register budget: the specialized gate_up loop holds 4 integer accumulators
plus 4 float accumulators, 8 metadata words, 2-4 uvec4 weight/high words,
8 activation words and the walked offsets: no vec4 accumulator arrays, no
function-memory spills (0 `_ptr_Function` in the specialized loops vs 4
dynamic accesses per word iteration before), so no repeat of the reverted
four-lane VGPR regression.

## 4. Local validation (llvmpipe, correctness only)

New oracle `expert_decode_fused_random` in `tests/test_fg_vk.c`: random
weights, gates, activation and slot maps (10 slots, dead slots included)
across `(gate,up,down)` = `(12,13,7)`, `(13,12,7)`, `(12,13,8 cooked)`,
`(12,12,8 raw)`, `(8,8,8 cooked)`; the fused pair must match the legacy
five-dispatch path at <=1e-3 relative (measured max 9.5e-05).  It also
supports `FG_EXPERT_PARITY_DUMP=<file>` to diff two kernel revisions
bit-for-bit; the same-workspace scratch is zeroed so the K-quant row builders
are deterministic.

PASS: `expert_decode_fused(12/13)`, `expert_decode_fused_q8_0(12/13)`,
`expert_decode_fused_q8_0_cooked(12/13)`, `expert_decode_fused_q8_gates`,
`expert_decode_fused_random`, `expert_graph_fused(7/8)`,
`expert_graph_fused_q8_0_cooked`, `expert_graph_replay`,
`q8_cooked_prefill_parity`, `q8_cooked_prefill_sweep`,
`grouped_kquant_prefill`, `grouped_down_prefill`, `test_expert_prefill`,
`test_owner_reduce`, `test_qsa_prefill`; `make all -j8` warning-free.
Known pre-existing failures unchanged: `test_core:457`, `test_session:254/257`,
llvmpipe `qsa_record_commit` crash after the expert tests.

Old-vs-new dump hashes: gate_up `8c877b32` -> `6c6826a7` (bit-identical),
down `2fee13c9` -> `de2da5c6` (Q5_1 bit-identical, Q8 <=1.2e-05 relative),
dead-code `de2da5c6` -> same output.

## 5. Fleet A/B plan (orchestrator)

Shader-only deploy; the binary is unchanged, so the two `.spv` files and the
usual runtime distribution are enough.  Same pack/manifest as the current
control (`872ba4b5` lineage).  Gate `FG_DECODE_PIPELINE` stays at its default.

1. Correctness: `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"`
   -> `[12]` / `[Paris]`.
2. Attach battery x2:
   `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2`
   -> short and 4K decode; 4K prefill must stay >=270.
3. Context sweep:
   `pwsh -NoProfile -Command "& 'D:\workspace\flash-gordon\tools\context-sweep.ps1' -Contexts 16384,32768"`
   and the 256K prefill spot check if convenient.
4. Kernel deltas: on rank 0 with `FG_DECODE_PROFILE=1 FG_DECODE_RING_TRACE=1`
   read the `expert_gate_up` and `expert_down_reduce` scope sums per 6-layer
   block; control is 0.86 + 0.56 ms.
5. Soak: `pwsh -NoProfile -File D:\workspace\fg-work-exp8\tools\pi-stability.ps1`.

Promotion bar: gates green, pi PASS, expert pair clearly below 1.42 ms/block,
short and 4K decode above the 22.52/23.01 and 21.64/21.56 baseline band, 4K
prefill unchanged.  If the pair lands above 0.85 ms/block, capture the
`expert_gate_up`/`expert_down_reduce` split and an occupancy counter run
before adding more kernel work.

## 6. Ranked next steps

1. **Fleet-qualify this round.**  All expected deltas come from issue
   accounting; only the blade can say how much of the 150 GB/s is instruction
   versus latency.
2. **Slot-paired gate_up workgroups** (2 routed slots per workgroup, grid y=5)
   would share the activation conversion and the loop shell across two
   experts: another ~10-15% of the gate_up stream.  It needs the slot count or
   pairing rule in a push constant or a vk.c contract, so it belongs to a
   round that can touch vk.c.
3. **Down Q5_1 integer-dot accumulation** (`d*sum(q*x) + m*sum(x)`, applying
   the scale/min once per block with the per-block activation sum taken from
   a cooperate/LDS pass): removes ~16 dequant FMAs per lane-block at the cost
   of the sum tree; estimated 8-15% of the Q5_1 path with a small numeric
   drift.
4. **LDS activation staging for gate_up** (2.96 KB per workgroup): same LDS
   load count as the current L1 reads plus a barrier; only worth doing if a
   hardware profile shows L1 sector pressure.
5. **Q8_0 gate/up and raw-Q8 down paths**: test-only; do not spend another
   round on them.
6. **Occupancy instrumentation** (RADV shader stats / counters) so the next
   geometry step has a register-pressure ground truth instead of inference.
