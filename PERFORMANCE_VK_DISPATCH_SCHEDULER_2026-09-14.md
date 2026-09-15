# vk Dispatch Scheduler: per-dispatch accounting and barrier/submission cuts (2026-09-14)

Worktree `D:\workspace\fg-work-sched9`, branch
`perf/vk-dispatch-scheduler` from main `f700967` (round-8 binary `9ec61a98`).
Scope: `src/vk.c`, `src/owner.c`, minimal `src/runtime.c` (report/trace only),
`include/fg_vk.h`. No shader, pack, manifest or protocol change.

Commits:

| commit | change |
|---|---|
| `2cf2fcc` | per-dispatch GPU gap + host op/barrier/submit accounting in `FG_DECODE_PROFILE` |
| `7ab5a93` | one-shot barrier elision for independent decode siblings |
| `d024e24` | hold a static replay and coalesce it into the next dynamic submit |
| `9be3f5a` | `FG_VK_HOLD_STATIC` gate + per-block submission/dispatch counters in `RING_DECODE` |

## 0. TL;DR

The mission hypothesis was that the per-dispatch `vkCmdPipelineBarrier` in
`dispatch_impl` and the ~224 dispatches per 6-layer decode block dominate the
remaining decode cost. Measured with the extended `FG_DECODE_PROFILE`, **the
barriers are not the dominant cost on GFX1013/RADV**:

* 224-230 dispatches per block, 10-12 submissions at 4K (profile path).
* GPU dispatch-timestamp spans sum to 4.13-4.88 ms per block; the sum of
  inter-dispatch gaps is only **0.125-0.209 ms per block** (0.75-0.9 us per
  gap; max single gap 1 us). A barrier costs about **0.3 us of recorded
  command time** and the GPU drain it forces costs under 1 us.
* Host-side per-dispatch cost is 1.3-1.6 us (`op_ms` 0.30-0.36 per block);
  `vkQueueSubmit` is 40-60 us per submission.
* The per-block wall is dominated by kernel execution (the six largest scopes
  are expert pair 0.92 ms, QSA 1.12 ms, GDN 1.12 ms, GR read chains 0.67 ms,
  shared expert 0.18 ms, router 0.10 ms) and by the host exposure at the two
  QSA selection readbacks per block, not by dispatch barriers.

Consequently the shipped changes are small and safe: two barrier-elision
sites in the decode graph, a gated static-replay submission coalescer, and the
accounting itself. Same-session fleet A/B: short decode 23.90/24.01 ->
24.01/24.16, 4K warm decode 22.27/22.54 -> 22.57/22.76 TPS, i.e. +0.13/+0.28
TPS, most of which the mechanism accounting attributes to noise. Gates
`[12]`/`[Paris]`, pi-stability PASS, 4K prefill 270-281 band.

## 1. Method

`profile_resolve` already read every per-dispatch timestamp pair. The
instrumentation adds, all only when `FG_DECODE_PROFILE` is on:

* `gap_ms`: sum over each submission of `begin(first) - submission start`,
  `begin(i+1) - end(i)`, and `end(last) - submission end`; `gap_max_ms` and
  the kernel name following the largest gap.
* `op_ms`: wall time spent inside `dispatch_impl` (descriptor update, bind,
  push, barrier, `vkCmdDispatch`); `barrier_ms`: the barrier share.
* `submit_ms`: wall time inside `vkQueueSubmit` for the batch/flush/standalone
  paths.

`DECODE_PROFILE` prints them; `runtime.c` gained two lines of format. Run:
4K prefill 4321 tokens, 32-token decode, `FG_DECODE_PROFILE=1` on all ranks,
rank 0 owns layers 6..11. Profiling disables static replay, so the profile
path is the dynamic 10-12 submission geometry; the numbers below are steady
state (token 4321) and reproduced across tokens 28/29 and the earlier
`ab-20260913-204953` session.

## 2. Accounting (rank 0, 6-layer block, 4K steady state)

| quantity | value |
|---|---:|
| dispatches | 224-230 |
| submissions (profile, dynamic) | 10-12 |
| GPU span `gpu_ms` | 4.25-5.09 |
| kernel spans `kernel_ms` | 4.13-4.88 |
| **inter-dispatch gaps `gap_ms`** | **0.125-0.209** |
| largest single gap `gap_max_ms` | 0.001 |
| host record `record_ms` | 0.56-0.59 |
| host dispatch ops `op_ms` | 0.30-0.36 |
| barrier record `barrier_ms` | 0.063-0.074 |
| `vkQueueSubmit` `submit_ms` | 0.47-0.61 |
| host fence wait `wait_ms` | 4.6-5.5 |

Derived per-dispatch averages: barrier 0.29 us, complete host op 1.34 us,
submission 45-55 us, GPU gap 0.75-0.9 us.

### 2.1 Where the GPU time goes (rank 0, token 4321, ms)

| scope | kernels | calls | ms |
|---|---|---:|---:|
| gr_attn_read | rms/split/reduce/inject/silu/up/mix | 7 | 0.336 |
| gdn_projection | qkv r8 0.374 + z cooked 0.222 + controls | 4+4+8 | 0.597 |
| gdn_recurrent | conv 0.010 + algebraic 0.148 | 8 | 0.158 |
| gdn_output | r8 6144x2560 | 4 | 0.361 |
| gr_attn_write | gr_write | 6 | 0.007 |
| gr_ffn_read | same seven kernels | 7 | 0.333 |
| router | dense_f32 | 6 | 0.097 |
| router_quantization | quantize_q8_k | 6 | 0.010 |
| shared_expert | r8 18 calls 0.140 + swiglu + scalar | 30 | 0.184 |
| expert_decode | top10 0.061 + schedule 0.006 + gate_up 0.476 + down_reduce 0.373 | 24 | 0.922 |
| moe_reduce | shared_add | 6 | 0.006 |
| gr_ffn_write | gr_write | 6 | 0.006 |
| qsa_projection | r8 0.239 + prepare 0.008 + bf16 0.072 + quantize 0.069 + idx 0.004 | 20 | 0.392 |
| qsa_state_attention | commit 0.004 + index_score 0.019 + topk_reduce 0.271 + gather 0.038 + attention 0.201 + merge 0.012 | 12 | 0.545 |
| qsa_output | r8 | 2 | 0.180 |
| **total** | | | **4.13** |

Kernel spans sum to slightly less than the submission GPU span in every
sampled block, i.e. dispatches inside a submission do **not** overlap even
where the barrier is already absent (GDN z/alpha/beta, HC inject). Removing a
barrier cannot recover the 107 GB/s GR chain toward the 353 GB/s byte floor;
the chain is dependent and latency-bound, and RADV serialises consecutive
compute dispatches in one command buffer regardless.

### 2.2 Where the host time goes

Per block: record 0.56 + submit 0.5 + ops 0.31 = ~1.1 ms of host work, with
0.30 of `op_ms` inside `dispatch_impl` and 0.26 in command-buffer
begin/end/reset. The rest of `own_run` is fence waits (4.6-5.5 ms) that
overlap GPU execution. The exposed part is the two QSA selection breaks per
block: `select_blocks` drains (fence wait), then the host maps 512 selected
blocks to cache slots, writes 2 KiB of ids to mapped memory, and records the
gather/attention/merge/output dispatches while the GPU queue is empty. The
round-1 `QSA_ATTEND_TRACE` measured ~0.09 ms of host work per selection above
the GPU's `topk_reduce`, i.e. ~0.18 ms per block, and `topk_reduce` itself is
0.271 ms per block. Neither is addressable from `vk.c`.

## 3. Changes, expected vs measured

### 3.1 `2cf2fcc` accounting (expected 0, measured 0)

Report-only. `DECODE_PROFILE` gains `op_ms barrier_ms submit_ms gap_ms
gap_max_ms gap_max=`. `RING_DECODE` gains cumulative
`submissions_total dispatches_total` (trace only).

### 3.2 `7ab5a93` barrier elision (expected -0.5..-1.0 ms/token, measured ~0)

`fg_vk_next_dispatch_independent()` sets a one-shot flag consumed by
`dispatch_impl`; `owner.c` marks three sibling pairs per layer that read the
same input and write disjoint outputs: router `quantize_q8_k` after the
router logits dense, shared-expert `up` and `scalar` (reordered before
`swiglu`). 18-20 barriers per block are no longer recorded. Expected saving
from the 0.9 us/gap accounting is ~17 us/block (0.14 ms/token), below A/B
resolution.

Measured effect in the profile: the affected dispatches' GPU spans collapse
to 0.000 ms (shared-expert scalar dense_f32 0.024 -> 0.000, router quantize
0.010 -> 0.000), i.e. the elision does let them overlap, but the recovered
time is bounded by those kernels' own size (~0.03 ms/block). `barrier_ms`
stays 0.063-0.074 because 200+ barriers are still recorded. Numerics
unchanged by construction (no data dependency removed).

### 3.3 `d024e24`/`9be3f5a` static replay submission coalescing (expected -0.1..-0.2 ms/block, measured ~0)

`fg_vk_static_submit` holds a recorded static run when the GPU already has a
tracked flush in flight; the next `fg_vk_end`/`fg_vk_flush` submits
`[held static][dynamic]` as one `vkQueueSubmit` with the existing semaphore
chain; `fg_vk_static_wait/drain/abort` flush held runs first. Gated by
`FG_VK_HOLD_STATIC` (default on). The round-2 descriptor-epoch discipline is
untouched: held static command buffers only use the static descriptor window,
which is never rebound, and the merge does not change when dynamic sets may
be rewritten (drain-before-reset is unchanged).

Production measurement with `FG_DECODE_RING_TRACE=1`: cumulative counters
after one 4321-token prefill plus 24 decode tokens differ by **2 submissions**
between `FG_VK_HOLD_STATIC=1` (10318) and `=0` (10320); dispatches identical
(50000). The hold condition (`pending_fence_count>0` at the static submit) is
almost never true in the production geometry, so the coalescer is inert in
practice. Kept because it is correct, tested and flag-gated; ranked below as
"not worth more work without a submission-cost profile that shows the cost is
exposed".

### 3.4 Not attempted: expert gate_up 2 slots per workgroup

Byte accounting after `9164653` (16-cohort 128-lane r8): gate_up is
0.476 ms/block for 6 x 10 experts x (gate+up) ~20 MB/layer, i.e. ~250 GB/s
against the 353 GB/s ceiling. Sharing the activation conversion and loop shell
across two routed slots can only save the conversion share of the issue
stream; the weight stream is unchanged. Expected <=0.05-0.07 ms/block
(0.4-0.56 ms/token) at best, against a documented VGPR-regression risk (the
round-5 four-lane rewrite regressed to 258 TPS and was reverted, and the r8
geometry itself was the fix). Not worth the risk in a round whose measured
dispatch overhead is 0.2-0.3 ms/block total; ranked with an evaluation plan
below.

## 4. Fleet A/B (same session, this worktree's fleet)

Baseline = instrumented binary with no behavioural change (equals main
`9ec61a98`), batteries `ab-20260914-211125/211214`. Final = build
`1f082f3d`, batteries `ab-20260914-214135/214220`. Attach battery, 4K runs are
prefill 4321 + 1 token; the first 4K decode in a session is cache-cold and not
comparable.

| metric | baseline | final | delta |
|---|---:|---:|---:|
| short decode (32 tok) | 24.01 / 23.90 | 24.01 / 24.16 | +0.13 |
| 4K warm decode | 22.54 / 22.27 | 22.76 / 22.57 | +0.28 |
| 4K prefill | 269.03 / 278.31 | 264.50 / 279.82 | band |
| gates [12]/[Paris] | PASS | PASS | PASS |
| pi-stability | (round-8 PASS) | PASS 6 stages, 8 ranks, 0 failures | PASS |

Context sweep, final build `1f082f3d`: 16K decode 21.32, 32K decode 17.95
(prefill 306/311). Build `e2db7401` (same code) measured 16K 21.31/21.48 and
32K 18.91/18.46 in two sweeps; the round-8 doc records 16K 20.96 / 32K 16.60
from a different session, so 16K/32K cannot be cleanly attributed. The
pi-stability run on the final serving config measured 4K prefill 271.89 and
short decode 21.28 (soak conditions).

Honest attribution: the same-session deltas are +0.13 TPS short and +0.28 TPS
4K, which is at the edge of the battery spread. The mechanism accounting
explains at most ~0.1-0.2 ms/block (barrier overlap of the elided siblings),
so the changes are best described as neutral-to-slightly-positive, and the
round's value is the measurement that refutes the barrier hypothesis.

## 5. Hard constraints

* Bit-identical numerics: barrier elision removes no data dependency; the
  affected dispatches read only tensors visible since the previous barrier and
  write disjoint outputs. Gate check `[12]`/`[Paris]` on every deploy.
* Round-2 descriptor-epoch discipline preserved: no change to when dynamic
  descriptor sets may be rewritten; static sets live in the upper window and
  are never rebound; held submissions only change *when* the static command
  buffers are submitted, not what they bind.
* Prefill >= 270: final 4K prefill band 264-282, pi gate 271.89 (the one
  264.50 reading is the first-run noise seen on every build including main).
* Soak: `tools/pi-stability.ps1` PASS (6 stages, 4-turn conversation, 8
  ranks, 0 failures).

## 6. Local validation (llvmpipe, correctness only)

```
make all -j8                                       warning-free (-Werror)
test_fg_vk focused: gpu_profile, static_batch_replay, pipeline_flush_parity,
  batch_submission_parity, group_norm, hc_inject_partial, gr_mix,
  q8_decode_shape_parity, expert_graph_replay, expert_graph_fused_q8_0_cooked,
  qsa_attention, qsa_record_gather, output_argmax, q8_cooked_prefill_parity,
  q8_dense_cooked, expert_decode_fused_random              PASS
tests/test_owner_reduce, test_qsa_prefill, test_expert_prefill  PASS
known pre-existing failures unchanged (test_core:457, test_session:254/257,
test_prefill_dispatch:178-183, llvmpipe qsa_record_commit crash)
```

Only `include/fg_vk.h`, `src/vk.c`, `src/owner.c`, `src/runtime.c` changed.
The `src/runtime.c` edits are two format changes: the six new
`DECODE_PROFILE` fields and the `submissions_total`/`dispatches_total` suffix
on the trace-only `RING_DECODE` line.

## 7. Ranked next steps

1. **QSA selection readback is the exposed host cost** (0.09 ms/block GPU idle
   per selection plus 0.27 ms/block of `topk_reduce`). A GPU-side page table
   (`block -> cache slot`) removes the readback entirely (round-3 item 2); a
   partial-selection `topk_reduce` (round-3 item 1) attacks the kernel. Both
   are `qsa.c`/shader work, not `vk.c`.
2. **Expert pair is 0.92 ms/block (6.8 ms/token) and still the largest single
   GPU block.** If the 2-slot gate_up is revisited, size it against the
   measured 250 GB/s and gate it behind an env flag with the
   `expert_decode_fused_random` dump oracle and a VGPR check, as the round-5
   four-lane regression demands.
3. **GDN output r8 (0.361 ms/block, ~185 GB/s)** is the slowest GDN shape per
   byte (320 workgroups where qkv exposes 1280). A split variant or wider
   cohort grid is the next geometry try; it is not a scheduler problem.
4. **Submission coalescing** (already implemented, inert): only re-open if a
   production `record_ms/submit_ms` profile shows host time exposed. The
   `FG_VK_HOLD_STATIC=0` gate and `submissions_total` counter make that
   measurement a one-restart experiment.
5. **Barrier merging is closed** for the decode graph: 0.125-0.209 ms/block
   total gap, 18-20 barriers already elided with no measurable fleet effect.
   Do not spend another round here; the round-8 estimate of -1..-3 ms/token
   from overlap is not supported by the timestamps.
