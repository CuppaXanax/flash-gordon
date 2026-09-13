# Rank-0 Fixed-Cost Cuts: Embed and Output Projection (2026-09-13)

Worktree `fg-work-ranko`, based on `b05a032` (ring decode default). Scope: the
~6.5 ms/token of fixed rank-0 cost in the C17 + GLSL + Vulkan ring stack. Two
changes, both inside existing files and kernels; no new shader registrations.

## 1. Single-row embedding is now a host gather

`fg_vk_embedding_q8_0` (`src/vk.c`, the existing single-row embedding wrapper)
previously issued one isolated compute dispatch per decode token: reset fence,
begin command buffer, host barrier, bind, dispatch 40 workgroups, shader
barrier, end, `vkQueueSubmit`, `vkWaitForFences`, then the caller streamed the
40 KiB result back out of the mapping. The trace attributes ~1.2 ms/token to
this scope. The kernel itself is trivial; the cost is the non-batched
submit/fence round trip plus queue drain (the decode path calls it outside any
`fg_vk_begin` batch).

The wrapper now decodes the token row on the host from the mapped arena into
the mapped output tensor when both mappings are available, and keeps the
original GPU dispatch as a fallback when they are not:

- per element: `value = fg_f16_to_f32(scale) * (float)(int8_t)q`, the same
  operands and the same single fp32 multiply as `fg_embedding_q8_0.comp`;
- 2560 q8 values read + 10240 floats written once, no fence, no submit;
- `fg_vk_tensor_write` into the same tensor and any later batch keep working:
  both `fg_vk_begin` and the non-batched `dispatch_impl` already insert a
  HOST->COMPUTE barrier, and `fg_owner_prefill_input` is explicitly the
  ping-pong buffer, not `up_logits` (see the comment at `src/owner.c:316`);
- exactness is asserted by the existing `q8_embedding` oracle in
  `tests/test_fg_vk.c`, which compares the wrapper output bit-for-bit against
  the CPU `fg_dequantize_q8_0` oracle.

Measured locally (WSL, llvmpipe context, host-cached arena, 2000 calls):

```
cpu_embed_us_per_row=5.28
```

The fleet arena is write-combining, so the real cost is 2.7 KiB of weight row
plus a 40 KiB sequential write; even a pessimistic 2 GB/s write path puts the
call at ~25 us. Assumption: `token_embd.weight` stays host-visible (all tensors
are created HOST_VISIBLE|HOST_COHERENT and every decode rank already reads the
40 KiB hyper from the same mapping). Expected saving: **~1.1-1.15 ms/token** on
ring decode and on the legacy local decode path.

## 2. Output logits use the qualified row-8 cooked kernel

`output.weight` is cooked Q8_0, so `fg_output_logits` (`src/output.c:83`)
dispatched `fg_dense_q8_0_cooked` (4 rows/workgroup, 62080 workgroups, each
workgroup re-reads the full 10 KiB activation vector). `dense_cooked_rows8_shape`
(`src/vk.c:303`) already routes the measured winning shapes to
`fg_dense_q8_0_cooked_r8` (8 rows/workgroup, 31040 workgroups, half the
activation traffic). The output shape `2560 -> 248320, tokens=1` was never in
the selection list; it is now added there, one condition on the existing line.

Fleet-measured shootout data for the same 2560-wide input
(`PERFORMANCE_PAST_WINS_SHOOTOUT.md`): `2560 -> 10240` 255.9 -> 285.7 GB/s
(+11.6 %), `2560 -> 12288` 250.3 -> 277.0 GB/s (+10.7 %). The output projection
streams 675 MB per token (636 MB quant + 40 MB scales), so a similar uplift
moves it from ~2.6 ms to ~2.35 ms. Expected saving: **~0.2-0.3 ms/token**,
fleet-gated.

Numerics: for a given row, both PSOs visit blocks in the same cohort order,
accumulate with the same FMA order, and reduce with the same subgroup tree, so
results are bit-identical; the only difference is how many rows one workgroup
owns. `248320 % 16 == 0`, so no bounds tail exists.

## Expected result

| Phase | Before | After | Basis |
|---|---|---|---|
| embed | 1.2 ms | <= 0.05 ms | local 5.3 us, WC-write bound |
| output logits GEMM | ~2.6 ms | ~2.35 ms | fleet +11 % row-8 on same input width |
| rank-0 fixed total | ~6.5 ms | ~5.05 ms | -1.35 to -1.45 ms/token (~1.7 % of 83.5 ms) |

The <= 2 ms output-head target is **not reachable inside this scope**. The
projection is DRAM-bandwidth-bound on 675 MB of q8 weights (~257 GB/s measured
vs ~350 GB/s clpeak on the BC-250), and the 3.9 ms round trip also contains the
7->0 and 0->4 40 KiB hops plus rank-4 host staging. Reaching 2 ms needs the
direct handoff below and/or splitting the vocabulary rows across the replicated
arenas (every rank holds `output.weight` in the replicated layout), which is a
protocol-level change, not a kernel change.

## Local evidence (llvmpipe, correctness only)

```
make all -j8                                            warning-free (-Werror)
./tests/test_qsa_prefill                                PASS
./tests/test_expert_prefill                             PASS
./tests/test_owner_reduce                               PASS
./tests/test_chat                                       PASS
./tests/test_chat_runtime                               PASS
DS4_REMOTE_TEST_FILTER=q8_embedding ./tests/test_fg_vk  ok (bit-exact)
DS4_REMOTE_TEST_FILTER=q8_dense ./tests/test_fg_vk      ok
DS4_REMOTE_TEST_FILTER=hc_finalize ./tests/test_fg_vk   ok
DS4_REMOTE_TEST_FILTER=output_argmax ./tests/test_fg_vk ok
DS4_REMOTE_TEST_FILTER=output_topk ./tests/test_fg_vk   ok
```

The known pre-existing `test_core` failure (`FG_QSA_REPLICA_DEPTH`) is
untouched. No shader source changed, so `vulkan/*.spv` is byte-identical to the
parent commit.

## Fleet validation instructions

1. Quiesce the fleet; never build on a serving blade. In a clean copy of this
   worktree on one blade: `make all -j8` (must be warning-free), then stage the
   binary + `vulkan/` shaders to all eight ranks as usual. Both changes are in
   `src/vk.c`; the SPIR-V set is unchanged from `b05a032`.
2. Correctness gates first, same ring pack and env as the `2ca92ba5` session
   (`FG_PREFILL_RING=1` on rank 0, `FG_WORKER_OWNER=1` on workers):
   `/no_think` arithmetic -> `12`, capital -> `Paris`, exact token continuity
   over the standard 32-token decode.
3. Instrumented A/B (one blade per configuration is enough for the fixed
   scopes; the orchestrator's eight-blade A/B owns TPS):
   - run `FG_PREFILL_RING=1 FG_DECODE_RING_TRACE=1` for a 4K prompt and 32
     decode tokens, on the new binary and on `b05a032`;
   - `grep RING_DECODE` both logs. Compare the mean `embed_ms` and `output_ms`
     over tokens 2..31 (token 0/1 include cold start);
   - accept embed if `embed_ms <= 0.3` on the new binary (expect ~0.02-0.05);
   - accept the output change only if `output_ms` improves by >= 0.15 ms; if it
     regresses, drop `||output_width==FG_Q38_VOCAB_SIZE` from
     `dense_cooked_rows8_shape` (`src/vk.c:303`) and keep the embed change.
4. Whole-frame gate: eight-blade 4K/32-token ring battery against the `2ca92ba5`
   log; require exact response/token parity, no gate regression, and a
   final-20 mean improvement of at least the measured embed saving. Per the
   shootout experience, kernel micro wins can shrink on the frame, so treat the
   1.3 ms as an upper bound and the embed ~1.1 ms as the dependable part.
5. Do not deploy from this worktree; hand the commits to the orchestrator.

## Stretch not implemented: direct rank 7 -> rank 4 output handoff

The current tail is `rank 7 --40 KiB--> rank 0` (DECODE_LAYER_RESULT), then
`rank 0 --40 KiB--> rank 4` (OUTPUT_WORK), rank 4 computes, 16-byte
OUTPUT_RESULT returns to rank 0. Forwarding the final hidden straight to the
output owner removes one 40 KiB hop and rank-0's read/re-encode/stage, an
estimated 0.3-0.6 ms of the 3.9 ms round trip. It is not isolated: rank 4 must
run the output for a worker result, and rank 0 must supply the sampler config
and `uniform` without ever seeing the final hidden, which touches protocol and
worker state. The exact patch map for the orchestrator:

- `include/fg_protocol.h:102-103,133`: add `FG_MSG_OUTPUT_HIDDEN` next to
  `FG_MSG_OUTPUT_WORK = 12` / `FG_MSG_OUTPUT_RESULT = 13`; reuse
  `FG_DECODE_LAYER_RESULT_BYTES` as the payload (layer, source, destination,
  token_index, `hyper[FG_HYPER_WIDTH]`).
- `src/runtime.c:809-820`: in `handle_decode_layer_work`, when
  `last + 1 == FG_LAYER_COUNT`, set `result->destination_rank` to
  `fg_output_owner_rank(manifest)` and `fg_fabric_send` to that rank with the
  new type. Rank 0 is no longer a receiver.
- `src/runtime.c:3640-3656`: rank 0's `coordinator_decode_token_ring` drops the
  DECODE_LAYER_RESULT branch and waits for OUTPUT_RESULT from the output owner.
- `src/runtime.c:1867`: rank 4's worker loop adds the hidden branch. It stores
  the 40 KiB into the existing `hyper` tensor (already allocated at
  `src/runtime.c:1864` for the output rank) and pairs it with the config.
- `src/runtime.c:1804-1818`: split `handle_output_work` so that, when ring
  decode is active, rank 0 sends only sampler+token_index (same fixed message,
  hyper bytes unused/zero but still finite-checked) at token start, before the
  chain runs; rank 4 computes as soon as both the config and the hidden for the
  same `token_index` have arrived. This is the stateful part: either arrival
  order must be handled, stale tokens must be dropped, and the
  `fg_output_history`/penalty bookkeeping on rank 4 is unchanged.
- `src/runtime.c:3314-3361`: `coordinator_output` becomes send-config +
  recv-16-bytes. Its `sequence = token_index*FG_LAYER_COUNT+FG_LAYER_COUNT`
  check moves to the config send.

Validation must prove: both arrival orders, stale-token rejection under
pipelined requests, exact greedy token/logit parity, and that rank 4's own
layer work still flows while an output config is pending. If those cannot be
shown locally, leave it out.

A larger, protocol-level option that actually reaches the 2 ms target: every
rank already holds all shared weights, so rank 0 can broadcast the final 40 KiB
to N ranks, each computes a 16-row-tile-aligned slice of `output.weight` with
the existing `fg_vk_dense_q8_0_f32` + one `fg_vk_argmax_reduce` pass, and rank 0
combines N 16-byte partials. That divides the 675 MB stream by N (~1 ms at
N=4) and needs no new kernels, only a gather message. It is strictly larger
than the direct-handoff change and should be sequenced after it.
