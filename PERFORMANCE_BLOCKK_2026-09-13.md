# Decode Block Time: QSA Selection, GR Mix, r8 Scales (2026-09-13)

Worktree `fg-work-blockk`, base `06cbe6c` (ring decode round 2). Scope: cut the
per-block GPU time of the chained ring decoder without touching the ring
routing or the owner/expert interfaces. Four kernel edits land; one experiment
(QSA attention token batching) was measured, regressed, and reverted.

Final binary `8f94286b0a5995f3b7bbe3bb395a9b7d5e3e4fb24bdf6e33c25ae996819a09da`
(deployed to all eight blades; `start-rank0-ring.sh` / `start-workers-ring.sh`).
`FG_DECODE_CHAIN=0` fallback is untouched and still the previous per-layer
machine; no runtime.c, protocol.c, or output.c changes.

## Changes

1. `fg_topk_reduce.comp`: the bitonic selection always sorted all 4096 padded
   slots even when the caller passed a smaller `count` (4K decode selects
   top-512 of ~1091 blocks -> `pc.width` = 2048, so half the passes and half
   the slots were sentinel padding). The three loops now bound by `pc.width`.
   Output is bit-identical: the sentinel slots already sorted to the bottom.
2. `fg_gr_mix_partial.comp` + `vk.c` dispatch: the GR mix was one 256-thread
   workgroup per token with a 1-workgroup dispatch (`gx=tokens`), 10.4 us per
   call at 120 KB of traffic. It is now `tokens * ceil(hidden/256)`
   workgroups; the per-token injection term stays in chunk 0. Same arithmetic
   per output element.
3. `fg_dense_q8_0_cooked_r8.comp`: the per-block scale words were loaded by
   one lane per 8-lane cohort and broadcast with four `subgroupShuffle`s every
   iteration. The scale words are always 16-byte aligned for this dispatch
   geometry, so every lane loads the `uvec4` directly. Removes 4 shuffles per
   group; measured neutral on the current shapes, kept for the instruction
   count.

### Measured and reverted

QSA decode attention token batching (`fg_qsa_attention_split`, 4 tokens per
barrier round instead of 1). Rank 2 0.489 -> 0.574 ms/block, rank 3 0.981 ->
1.151, rank 7 0.840 -> 0.962 - a consistent 15-17 % regression across every
rank. The per-token barrier chain is not the limiter; the kernel is bound by
the serial lane-0 online-softmax updates and the scalar record loads. The
original kernel is restored byte-for-byte.

## Late addition: GDN recurrent unrolled by 4 (commit 2, `8f94286b` shaders)

`fg_gdn_recurrent_algebraic` was load-latency bound: one column per lane, 128
serial dependent state loads per lane, only 48 workgroups (2 per CU). The two
state loops are now unrolled by 4 with independent temporaries so the loads
issue ahead of the FMA chain; the per-index arithmetic order is unchanged and
`test_gdn_algebraic`'s bit-exact state check still passes.

| rank | before ms/block | after ms/block |
|---|---|---|
| 1 | 0.722 | **0.240** |
| 2 | 0.650 | **0.224** |
| 3 | 0.515 | **0.178** |
| 7 | 0.456 | **0.164** |

Block GPU: rank 1 7.26 -> 6.78, rank 2 5.71 -> 5.28, rank 3 6.42 -> 6.08.
Fleet after this change: short decode **19.30 / 19.55 TPS** (two back-to-back
batteries), 4K first-token 17.36/17.64, 4K real 36-token 16.57/14.96/14.99.
Prefill is untouched by construction (the decode recurrent kernel is not used
in prefill); measured 261-300 TPS across the session band.

## Fleet evidence (4K, 4360-token context, attach battery)

| metric | before (`06cbe6c`) | commit 1 (`8f94286b`) | final (commit 2) |
|---|---|---|---|
| 4K prefill | 276.4 TPS (gate 270) | 279.2 | 261-300 (noise, decode-only change) |
| short decode, 32 tok (battery) | 17.35 TPS | 17.87 | **19.30 / 19.55** |
| 4K first decode token | 9.84 TPS (cold) | 17.02 (warm) | **17.36 / 17.64** |
| 4K real decode, 36-token listing prompt | 13.99 / 14.39 / 14.25 | 15.65 / 14.55 | **16.57 / 14.96 / 14.99** |
| gates | `[12]` / `[Paris]` | `[12]` / `[Paris]` | `[12]` / `[Paris]` |

The real 36-token listing prompt (`fg-4k-list32.ps1`,
"List the integers from one to forty") is the sustained comparison; the
shipped `fg-sustained-4k.ps1` stops after a few tokens on "Reply with one
word" and measures first-token speed, not sustained decode.

## Per-kernel before/after (Vulkan timestamps, `FG_DECODE_PROFILE=1`)

Profiled 4K token 4360+, ranks 1/2/3/7, mean of the last 12 blocks. Rank 2 is
the cleanest subject (1 QSA layer, no raw-Q8_0 down). The two profiling
sessions are ~1 h apart and the GPU drifts thermally: every untouched kernel
is 5-15 % slower in the after run, so compare the changed kernels against
their unchanged neighbours, not across the columns.

| kernel (rank 2) | before ms/block | after ms/block | note |
|---|---|---|---|
| `fg_topk_reduce` | 0.277 | **0.178** | -36 %, changed |
| `fg_gr_mix_partial` | 0.125 | **0.035** | -72 %, changed |
| `fg_dense_q8_0_cooked_r8` | 0.087 | 0.090 | changed, within noise |
| `fg_qsa_attention_split` | 0.489 | 0.531 | unchanged (drift) |
| `fg_moe_decode_gate_up` | 0.614 | 0.695 | unchanged (drift) |
| `fg_moe_decode_down_reduce` | 0.437 | 0.492 | unchanged (drift) |
| `fg_gdn_recurrent_algebraic` | 0.596 | 0.224 (commit 2) | -62 %, unrolled |
| `fg_dense_q8_0_cooked` (z) | 0.251 | 0.260 | unchanged (drift) |
| `fg_group_rms_norm` | 0.062 | 0.068 | unchanged (drift) |
| `fg_dense_q8_0_cooked_split` | 0.098 | 0.102 | unchanged (drift) |

QSA-heavy ranks gain more: rank 3 `fg_topk_reduce` 0.583 -> 0.348 (2 QSA
layers), rank 7 0.492 -> 0.292; `fg_gr_mix_partial` 0.126 -> 0.034 and
0.110 -> 0.030. Twelve GR mixes and ~10 QSA selections run per token, so the
two changes remove roughly 2-2.5 ms/token of GPU time.

Raw-Q8_0 down layers (2/4/30/46/47) still cost what they did before this
round: rank 1 `fg_moe_decode_down_reduce` 1.301 ms/block, rank 7 1.098,
versus ~0.49 for the cooked-Q5_1 ranks. That is the largest single per-rank
outlier left.

## Where the block time is now, and why 3.5-4 ms was not reached

Per-block GPU (4K steady state, profiled): rank 1 **6.78**, rank 2 **5.28**,
rank 3 **6.08**, rank 4 5.57, rank 5 6.5, rank 6 5.58, rank 7 **6.22**;
plus ~0.4-1.8 ms of wall above GPU per hop (QSA submission breaks, 40 KB hops).
The sustained 4K listing measures ~64.5 ms/token (short 32-token decode is
51 ms/token). The token-rate floor implied by the weight streams is
~2-2.6 ms/block per rank, so the block GPU still carries roughly 2x of
instruction/latency overhead above the stream floor.

The block is **not** uniformly DRAM-bound. Tallying the traffic against the
clocks:

- Near the ~350 GB/s roofline already (stop tuning): GDN qkv r8 ~291 GB/s,
  z dense ~314 GB/s, gdn output r8 ~186 GB/s, GR up r8 ~164 GB/s.
- Instruction-bound, not DRAM-bound: gate_up ~194-220 GB/s and down ~168 GB/s
  at 5-6 layers x 10 experts; both are nibble-unpack kernels. GFX1013 has no
  integer dot product (`dp4a` excluded by RADV), so each int8 MAC costs an
  unpack8, a convert, and an FMA. The raw Q8_0 down is worse still: 8 scalar
  word loads and funnel per 4 values, ~37 GB/s effective on ranks 1/5/7.
- Latency-bound: `fg_qsa_attention_split` (scalar 1-byte record loads + a
  serial lane-0 online softmax) and the GR chain's tiny 4-1000 workgroup
  dispatches.

## Remaining gap to 10 ms/token

The measured token is ~64.5 ms sustained (4K, real 36-token listing); 10 ms is
another ~54 ms (6.5x), or ~41 ms against the 51 ms short-decode measure. The
next levers, in measured order:

1. **Raw-Q8_0 expert down format change** (~1.5-2.0 ms/token on ranks 1/5/7).
   The kernel is instruction-bound at ~37 GB/s against 168 GB/s for the cooked
   Q5_1 twin. A coalesced rewrite (one warp per row) keeps the same instruction
   count; the real fix is the cooked 16-row tile layout, and the source GGUF is
   not on this machine. The runtime-repack option needs +891 MB/layer, which no
   blade has. Documented q8 repack plan stands; do not tune the existing funnel.
2. **QSA decode attention** (~2-3 ms/token across 10 QSA layers). The 4-token
   batching failed (above). Next candidate is a two-pass score/softmax split
   with the record reads vectorized through `uint16_t` (the 34-byte q8 record
   blocks are 2-byte aligned), plus dropping the redundant third barrier per
   token.
3. **GDN/qkv r8 small-shape latency** (GR up ~164 GB/s at 3.3 MB/call): merge
   the two GR up projections per layer is not possible (different inputs);
   raise the row count per workgroup only if a profile shows the r8 dispatch
   tail dominating.
4. **Expert pair instruction reduction.** No dp4a on GFX1013. Packed f16 math
   doubles weight bytes unless conversion stays in registers; the gate/up
   kernel is at 63 % of roofline already. Low expected value without a format
   change.
5. **Per-hop wall gaps** (~0.4-1.8 ms/block, 4-7 submissions per block). QSA
   selection still forces a host readback every 4th layer; the docs' "ids stay
   on GPU" item removes submissions, not kernels. Out of scope for this
   worktree (runtime.c).
