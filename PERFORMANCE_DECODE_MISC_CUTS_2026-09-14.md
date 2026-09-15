# Decode misc cuts: output split fix, QSA decode top-k, sampler residual (2026-09-14)

Worktree `fg-work-cut8`, branch `perf/decode-cuts` from main `b721ceb`.
Scope: the `FG_OUTPUT_SPLIT` fleet failure, the decode QSA selection cost, and
the per-token sampler/host accounting. Shader + protocol + test changes only;
no pack, manifest or prefill-path change.

| commit | change |
|---|---|
| `cce22e0` | route the split vocabulary slice to rank 0 in its own payload |
| `340cb50` | chunked top-512 selection for the decode QSA scan (`FG_QSA_TOPK_V2`) |

## 1. `FG_OUTPUT_SPLIT` root cause and fix (priority)

### Root cause: protocol framing, not state

The final block owner sends two frames from one `fg_layer_result`: the
`FG_MSG_OUTPUT_HIDDEN` frame to the output owner (rank 4) and the
`FG_MSG_OUTPUT_SLICE` frame to rank 0, both encoded once with
`destination_rank = output_owner`:

```c
result->destination_rank=(uint8_t)(direct?output_owner:0u);   /* 4 */
status=fg_decode_layer_result_encode(context->result_wire,result,err);
... send FG_MSG_OUTPUT_HIDDEN to output_owner ...
if(direct&&fg_output_split_requested())
    ... send FG_MSG_OUTPUT_SLICE to rank 0 from the same wire ...
```

Rank 0's `coordinator_output_slice` validates the decoded route as
`slice.destination_rank != 0u -> "misrouted ring decode output slice"`. The
first decode token therefore failed on rank 0 with `FG_ERR_MISMATCH` before
rank 4 could produce any partial and before any slice executor check ran (the
"no `slice executor` error" observation). The request error then went through
`api.c`'s failure reset while rank 0 still had the config `transport_pending`,
so `runtime_reset_state` surfaced `distributed transport is not reusable;
reopen the runtime` and rank 0 exited; the workers ended on
`peer closed fabric socket`. The fleet symptom is fully explained by this
one wrong destination byte (there is no separate state or handoff-ordering
defect: `fg_output_handoff_*` already handles both arrival orders and the
transport accounting balances config(+1)/result(-1)).

### Fix

`fg_output_slice_encode` (new, `src/protocol.c`) re-encodes the layer result
with `destination_rank = 0`; rank 7 uses it for the slice frame while the
hidden frame keeps the output owner. The flag gate and the default path are
unchanged (`fg_output_split_requested()` still gates both the slice executor
creation and the extra send).

### Local proof

* `tests/test_fabric.c` `protocol_output_handoff_selfcheck` now encodes the
  same result both ways and asserts the coordinator's exact route predicate:
  the hidden encoding (destination 4) fails `destination_rank==0`, the new
  slice encoding passes and preserves source/layer/token/hyper bit-for-bit.
* `output_handoff_roundtrip` (eight-process dual-channel mesh) now runs the
  full split exchange: rank 7 sends the hidden to rank 4 and the slice to
  rank 0, rank 0 validates the route and returns a partial, rank 4 combines
  with `fg_output_better/combine` semantics and rank 0 accepts only the
  combined winner. Pre-fix the slice frame carried destination 4 and the new
  rank-0 validation fails the test.
* `q8_cooked_view_slice`, `output_split_combine`, `test_fabric` PASS.

### Expected delta and fleet A/B

Task estimate -0.5..-1.1 ms/token; the round-3 doc estimate for the 2-way
head was -1.0..-1.4 ms/token (rank 4's head GEMM 2.62 ms, dense 0.058, HC
chain 0.101 at the last recorded profile).

A/B (the orchestrator runs it; scripts already exist from the failed run):

1. Default control, no env: `start-rank0-ring.sh` on .42 and
   `start-workers-ring.sh` on .43-.49; then
   `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"` ->
   `[12]`/`[Paris]`, and
   `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2`.
2. Split on: `start-rank0-ring-split.sh` on .42 and
   `start-workers-ring-split.sh` on .43-.49 (both export `FG_OUTPUT_SPLIT=1`);
   same gates then battery. Compare short and 4K decode.
3. If rank 0 exits, capture rank 0's stderr before the reset overwrites the
   error: the primary error line now is `misrouted ring decode output slice`
   or `ring decode received an output slice without a slice executor`; the
   former must be absent with this fix.

The 4-way extension is not attempted this round: the remaining two slices
need a foreign-`rank-04.fgw` tensor loader (~169 MB per slice) and rank
startup memory changes. 2-way keeps the change inside the existing
`fg_output_slice`/protocol path.

## 2. QSA decode top-k

### What is actually on the decode path

`fg_qsa_resident_select.comp` has no production caller in this tree
(`fg_vk_qsa_resident_select` appears in tests only); the resident merge kernel
is the prefill batched selector's merge step (`fg_vk_qsa_select_merge`), not a
decode path. Decode selection is `select_blocks` (`src/qsa.c`) ->
`fg_vk_topk_reduce` -> **`shaders/fg_topk_reduce.comp`**, which matches the
recorded ~0.2 ms/call (`fg_topk_reduce` 0.41 ms per 2-layer block in
`PERFORMANCE_DECODE_OVERLAP_2026-09-14.md` section 1.2). The resident kernels
are prefill/merge experiments and were left untouched. The round-7 idea
already landed for decode scoring (`fg_qsa_index_score` is block-major and
measured 0.022 ms/call); the remaining 0.2 ms is the bitonic selection
itself, so the partial-selection step recorded in that document's ranked
next steps is what was implemented.

### Change (`FG_QSA_TOPK_V2`, default off)

`shaders/fg_topk_reduce_v2.comp`: fixed 64-slot runs sorted in place, pairwise
merges keep both halves until a run reaches the 512-slot output width, and
only wider merges keep the top half. Unused shared memory stays
sentinel-filled so an odd run merges safely against an all-padding partner.
Both kernels stay resident; `FG_QSA_TOPK_V2=0` or unset runs the old kernel.

Dynamic accounting derived from the two networks (stages = barrier-separated
passes, comparators = compare-exchanges, iters = per-lane loop iterations):

| count | wide v1 (stages/comps/iters) | chunked v2 (stages/comps/iters) | comps |
|---:|---|---|---:|
| 513 | 55 / 28160 / 220 | 55 / 18784 / 160 | 1.50x |
| 1000 | 55 / 28160 / 220 | 55 / 25856 / 194 | 1.09x |
| **1090 (4K window)** | **66 / 67584 / 528** | **65 / 36608 / 287** | **1.85x** |
| 2048 | 66 / 67584 / 528 | 65 / 54528 / 408 | 1.24x |
| 4096 | 78 / 159744 / 1248 | 75 / 111872 / 836 | 1.43x |

### Proof

`topk_reduce_chunked_parity` (test_fg_vk) runs the same inputs through both
kernels and compares scores and ids bit-for-bit over counts
1..8200 including 511/512/513, the exact 1090/1091 shapes, ties, NaN, +Inf
and -Inf, and the multi-group >4096 shapes. `qsa_indexer` and `output_topk`
also pass with `FG_QSA_TOPK_V2=1` forced. A Python model of the network was
checked against a reference sort for the same shapes before the GPU run.

### Expected delta and fleet A/B

If the measured 0.2 ms/call is proportional to the comparator/iteration work,
1090 lands at ~0.11 ms/call (-1.3 ms/token over 12 layers); the 0.8 ms/token
target needs a bigger algorithmic change (radix/threshold select) and is not
claimed. If the kernel is barrier/latency-bound the win is smaller; the A/B
decides.

`FG_QSA_TOPK_V2=1` on **all ranks** (the env is read per call in
`fg_vk_topk_reduce`, so every rank that selects must have it) versus unset;
same pack, gates then battery. The decode QSA trace
(`FG_DECODE_PROFILE=1`) reports the selection kernel per block: the control
shows `fg_topk_reduce.spv`, the flag-on build shows `fg_topk_reduce_v2.spv`
in the same scope.

## 3. Sampler / host residual

Timestamps from the recorded 4K session `ab-20260913-204953` (rank 4,
`FG_PROFILE_TOKEN`, direct output handoff, greedy):

```
TOKEN_PROFILE rank=4 token=30 kind=output wall_ms=2.922 gpu_ms=2.751
  kernel_ms=2.742 vk_overhead_ms=0.009 wall_residual_ms=0.171
  submissions=1 dispatches=8
TOKEN_PROFILE_KERNEL ... scope=output fg_group_rms_norm.spv   calls=1 gpu_ms=0.015
TOKEN_PROFILE_KERNEL ... scope=output fg_dense_q8_0_f32.spv   calls=1 gpu_ms=0.058
TOKEN_PROFILE_KERNEL ... scope=output fg_silu_scaled.spv      calls=1 gpu_ms=0.001
TOKEN_PROFILE_KERNEL ... scope=output fg_dense_q8_0_cooked_r8 calls=1 gpu_ms=0.024
TOKEN_PROFILE_KERNEL ... scope=output fg_hc_finalize.spv      calls=1 gpu_ms=0.003
TOKEN_PROFILE_KERNEL ... scope=output fg_dense_q8_0_cooked.spv calls=1 gpu_ms=2.620
TOKEN_PROFILE_KERNEL ... scope=output_argmax fg_argmax_reduce.spv calls=2 gpu_ms=0.020
```

* The C sampler is not on the default per-token path at all: temperature 0 and
  no penalties take `fg_output_greedy`; `fg_sampler_select` (insertion sort,
  <=64 candidates), `fg_sampler_uniform` (one xorshift64) and
  `fg_sampler_apply_penalties` (test-only) cost <1 us combined when they run.
* `fg_apply_penalties.comp` (970 workgroups over 248320 logits) and
  `fg_topk_select.comp` (3 radix passes) only run for penalties / nonzero
  temperature and were not in this profile.
* The sampler-adjacent GPU work per token is 0.101 ms HC chain + 0.020 ms
  argmax; the head GEMM is 2.620 ms and is what the split head halves.
* The 0.171 ms wall residual is host staging (40 KiB hyper write, map/validate,
  16-byte result send) plus pre/post-batch overhead; it is ~0.35% of the 4K
  token and the only reducible fraction is the 40 KiB memcpy (a direct-decode
  variant of `fg_decode_layer_result_decode` into the mapped tensor would save
  ~10-20 us). Not worth a protocol-surface change, so **no code change**: the
  sampler path is left as is. The one place with real headroom is the
  `output_hc_down` GEMM running generic q8_0 (0.058) while the same-FLOP up
  projection runs cooked_r8 (0.024); that needs a pack/layout change outside
  this task.

## Local validation

```
make all -j8                                        warning-free (-Werror)
tests/test_fabric                                    PASS (split slice route + roundtrip)
tests/test_qsa_prefill                               PASS
tests/test_expert_prefill                            PASS
tests/test_owner_reduce                              PASS
tests/test_fg_vk (focused, v2 off and v2 on where applicable) PASS:
  batch_submission_parity, pipeline_flush_parity, gpu_profile, output_argmax,
  output_split_combine, q8_cooked_view_slice, qsa_attention, qsa_attention_single,
  qsa_record_gather, qsa_quant_and_bf16, qsa_prefill_prepare,
  qsa_segmented_index_score, qsa_prefill_chunk_liveness, qsa_indexer,
  memory_telemetry_and_canary, tensor_view_rebind, expert_graph_replay,
  expert_graph_fused_q8_0_cooked, expert_graph_rejects_overlap,
  expert_decode_fused_q8_gates, topk_reduce_chunked_parity
tests/test_core                                      1 pre-existing fail (line 457)
tests/test_session                                   2 pre-existing fails (254/257)
```

The full `test_fg_vk` suite was not run (known llvmpipe `qsa_record_commit`
crash per the round brief).
