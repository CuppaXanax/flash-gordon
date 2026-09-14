# Ring Decode Overlap: Per-Layer Pipelined Submission and QSA Queue Retention (2026-09-14)

Worktree `fg-work-overlap`, branch `perf/decode-overlap` (rebased by the
orchestrator onto main `7f9f3c0`; no merge to main). Scope: ring decode wall
time per token. No shader, pack, manifest, protocol or prefill-path change.
Round-1 commits (rebased hashes): instrumentation `19890d2`, layer pipeline
`beadeee`, QSA queue retention `63dbca1`, standalone-slot wait `b45e670`.
Round-2 fix commits: `a79792f`, `443e8ce`.

Gate: `FG_DECODE_PIPELINE=0` disables both overlap paths (per-layer flush and
QSA flush) and restores the previous submission geometry without a rebuild.

## Round 2 result (fix verified on the live fleet, pipeline ON)

The first fleet A/B of the async pipeline corrupted numerics (non-finite block
output on the first decode token) and was reverted to `FG_DECODE_PIPELINE=0`.
The root cause is now found and fixed; the pipeline-on build
`59b2f559369873ead009cb3673c00efa616013ed3ed239e949e8cbf564a9a5c7`
passes every gate and beats the pipeline-off control:

| metric | pipeline off (control) | pipeline on (fix) | delta |
|---|---:|---:|---:|
| correctness gates | [12] / [Paris] | [12] / [Paris] | PASS |
| short decode (32 tok) | 21.03 | **21.76 / 21.88 / 21.93** | +3.5-4.3% |
| 4K warm decode (32 tok) | 19.98 | **21.02 / 21.05 / 21.10 / 21.19** | +5.2-6.1% |
| 16K decode | 18.90 | **19.91** | +5.3% |
| 32K decode | 16.95 | **17.67** | +4.2% |
| 4K prefill | 279-285 | 296-320 (spread band) | unchanged code path |
| `tools/pi-stability.ps1` | PASS | **PASS** (6 stages, 4-turn, 8 ranks, 0 failures) | PASS |

Evidence: `bc-250-dbg/results/ab-20260914-144338`, `ab-20260914-144629`
(attach batteries), `flash-gordon/results/context-sweep-20260914-145101`,
pi-stability log in `.fleet-ab` on rank 0.

### Root cause (validation-layer proof)

The failure was not a semaphore or fence bug. With
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` on ranks 0/1 the first decode
token produced, from rank 1 alone:

```
VUID-vkUpdateDescriptorSets-None-03047
vkUpdateDescriptorSets(): ... VkDescriptorSet 0x90000000009 is in use by
VkCommandBuffer 0x2d590da0. This is only possible with flags found in
VK_EXT_descriptor_indexing.
```

`fg_vk_begin` reset `batch_set_count = 0` on every new top-level batch. With
the async pipeline, the layer-after-a-flush begins a new batch while the
previous layer's flush **submission is still queued**. The recording then
rebound descriptor sets 0..N, and RADV reads a descriptor set at execution
time, so the queued command buffer executed with the new bindings (wrong
buffers/offsets) and the block output went non-finite. The submission chain
ordered *execution* correctly; the corruption was a host-side rebind of state
a queued submission had not consumed yet. This also explains all the isolation
results: `FG_DECODE_FLUSH_WAIT=1` (every flush fence-waited) passes, while
per-layer async (b2) and QSA-only async (b1) both fail, and why llvmpipe could
not reproduce it (software execution consumed descriptors at record time).

### Fix

- `443e8ce`: `fg_vk_begin` no longer restarts the descriptor-set epoch. The
  epoch resets only in `fg_vk_end` after every pending submission has been
  fence-drained, and in `fg_vk_begin` only at the half-capacity mark after the
  same drain (amortized, rare).
- `a79792f`: `fg_vk_end` drains all pending submissions before resetting the
  epoch, and the standalone (non-batched) dispatch path drains before it
  reuses descriptor set 0.

Both are pure host-side ordering fixes; no kernel, protocol or numerics change.

## TL;DR (round 1)

The ring token is a data-dependency chain across eight ranks; the remaining
per-token cost is not only GPU kernel time but the fact that **host recording
of each submission is serialized against GPU execution of the previous one**.
The decode path used one command buffer and fence-waited at every submit
boundary, so every block paid `record + exec + signal` per chunk with no
overlap. Measured on the recorded `ab-20260913-204953` 4K session, rank 0's
6-layer block ran `own_run_ms` 6.3-7.1 ms against `gpu_ms` 5.13-5.61 and
`kernel_ms` 4.98-5.43: **1.2-1.5 ms per block (21%) of wall time that is not
GPU kernel execution**. Scaled over eight blocks that is ~10 ms/token, and the
QSA drain barriers explain most of it: rank 0's own block issued **7
submissions for one 6-layer block** (230 dispatches) with two full drains per
QSA layer, and the QSA layer wall (`QSA_TRACE proj_ms`) was 1.2-3.6 ms.

The pipeline makes the host record ahead of the GPU (`fg_vk_flush`), keeps the
QSA projection/commit work in the pipeline instead of draining at every layer,
and adds the timestamps needed to see the remaining gap (`wait_ms`,
`record_ms`, `flush_ms`). Measured combined effect is **+0.7-1.2 TPS** on
short/4K/16K/32K decode (table above).

## 1. The accounting we found (evidence)

### 1.1 Per-block: the cross-submission time is real and large

Source: `bc-250-dbg/results/ab-20260913-204953` (4K context, tokens 4358-4370,
`FG_DECODE_RING_TRACE=1 FG_DECODE_PROFILE=1`, rank 0).

```
RING_DECODE token=4369 embed_ms=0.252 first_hop_ms=7.773 own_layers=6..11
  own_ms=7.104 own_run_ms=7.094 own_read_ms=0.010 tail_ms=43.925 output_ms=0.000
  total_ms=59.054 handoff=1
DECODE_PROFILE rank=0 token=4369 layers=6..11 gpu_ms=5.606 kernel_ms=5.434
  submissions=7 dispatches=230
```

`own_run - gpu_ms = 1.49 ms` for that block; over tokens 4358-4370 it is
1.19-1.49 ms. `gpu_ms` sums each submission's timestamp span, so it excludes
everything that happens between submissions: host recording plus GPU idle.
With a single command buffer the GPU *cannot* execute while the host records
(the buffer must be fence-waited before it can be re-recorded), so this gap is
exactly the host-side serialization the mission asked us to reclaim.

Submission count per 6-layer block, rank 0 (layers 6-11, QSA at 7 and 11):
two QSA layers x three flush points = 6, plus the block-end fence = 7. The
three flush points per QSA layer in `fg_qsa_session_decode` were:

1. entry drain (`fg_qsa_submit_host_reads`) - not needed for data, only to
   separate batches;
2. projection drain after the 10 projection dispatches - not needed for data;
3. the selection fence inside `select_blocks` - genuinely needed at 4K
   because the host maps 512 selected block ids to cache slots.

### 1.2 Per-kernel: where the GPU time goes (same block)

`DECODE_PROFILE_KERNEL` scope sums for token 4369 (6 layers):

| scope | ms/block | detail |
|---|---:|---|
| expert pair (`fg_moe_decode_gate_up` + `down_reduce`) | 1.42 | 0.86 + 0.56, 6 layers |
| QSA state/attention | 0.80 | `fg_topk_reduce` 0.41 of it |
| GDN (projection + recurrent + output) | 1.21 | 4 GDN layers |
| GR reads (attn + ffn) | 0.83 | 12 small kernels/layer |
| QSA projection | 0.49 | 6 `dense_q8_0_cooked_r8` calls + 4 misc |
| shared expert + router | 0.37 | |
| QSA output | 0.19 | |
| everything else | 0.12 | writes, reduce, schedule |
| **kernel_ms total** | **5.43** | matches the profile |

The `fg_topk_reduce` figure is the single largest QSA kernel: 0.20 ms/call at
4K (1024 candidate blocks, 55 barrier-separated bitonic passes, one
workgroup). Twelve QSA layers per token = **~2.4 ms/token of GPU sort** that is
untouched by this round.

### 1.3 QSA layer wall and the selection cost

```
QSA_TRACE layer=7  token=4369 proj_ms=1.328 commit_attend_ms=0.394 out_ms=0.002 total_ms=1.724
QSA_TRACE layer=11 token=4369 proj_ms=3.581 commit_attend_ms=0.393 out_ms=0.002 total_ms=3.976
QSA_ATTEND_TRACE layer=7 tokens=4369 selected=512 missing=0 fetched=0
  select_ms=0.312 lookup_ms=0.027 fetch_ms=0.000 gather_ms=0.021 attn_ms=0.003 total_ms=0.363
```

`proj_ms` is wall time from QSA entry through the second drain. Most of it is
the GPU executing the previous layers while the host blocks on the drain (the
waits are "useful" in that the GPU is busy), but the drain turnarounds are
not: after each fence the host must wake, record the next chunk and submit
while the GPU has nothing queued. At 4K `select_ms` is 0.312 ms of which the
GPU executes ~0.225 ms (`fg_topk_reduce` 0.203 + `fg_qsa_index_score` 0.022);
the remaining ~0.09 ms is host readback + 512 cache lookups + write + re-begin,
i.e. GPU idle. Twelve QSA layers contribute roughly 1-2 ms/token of this kind
of idle before this round.

### 1.4 What is *not* here

* No rank sits idle because of scheduling: the ring is strictly
  token-serial (block N+1 consumes block N's 40 KiB hyper), so "rank N+1
  starts block K while rank N runs block K" is impossible for one sequence.
  The hop exposure is ~0.09 ms/block (`egress_ms`), already small.
* The direct rank7->rank4 handoff (`a711e29`) is correctly off the rank-0
  relay; its output head cannot start before the hidden arrives, so it cannot
  overlap block K+1. Nothing to change there.
* GDN (~11 ms/token) and GR chains (~7.4 ms/token) run at 172-324 GB/s and
  ~174 GB/s respectively on the same 24 CU / DRAM. They are bandwidth/latency
  bound on the owners; overlapping them with expert streaming (itself
  ~150 GB/s, instruction/latency bound) does not create bandwidth, and this
  round deliberately does not touch GPU concurrency. The change below is
  host-vs-GPU overlap only: no new DRAM traffic, no new contention.

## 2. What changed and why

### 2.1 `fg_vk_flush` - submit without draining, rotate the buffer

`src/vk.c` now owns two command buffers and two fences. `fg_vk_flush` ends the
recording, submits on the current fence **without waiting**, and rotates to the
other buffer; `fg_vk_begin` waits that buffer's fence only if it is still
pending (bounded backpressure), and `fg_vk_end` keeps the old
submit-and-wait behaviour for real readbacks. Descriptor sets stay monotonic
across a flush so a pending submission never shares a set with the next
recording.

Submissions from a flush are chained with a binary semaphore
(`vk_submit_attach`): the next submission waits the previous flush's signal.
That is the spec-correct execution + memory dependency between the command
buffers; it costs nothing when the queue is already in order and prevents a
later buffer from reading a tensor the previous one is still writing.

Why: it converts `wall = sum(record_i + exec_i + signal_i)` into
`wall ~= max(sum(record), sum(exec)) + sync points`. The host records layer
N+1 (0.15-0.25 ms) while the GPU executes layer N (0.8-1.2 ms), so the GPU
stops starving at submission boundaries.

### 2.2 Chained decode block flushes per layer (`src/owner.c`)

`fg_owner_decode_block_chained` calls `fg_vk_flush` after each layer except the
last (the last layer is still submitted synchronously by `finish_batch`, which
is immediately followed by the 40 KiB hidden readback - there is nothing left
to overlap there). Gated by `FG_DECODE_PIPELINE`.

Why: the block previously submitted 1-2 chunks per QSA break and held
everything else until the end; the first submission of a block could happen
after 0.6-0.8 ms of recording. Now the GPU starts after layer 0 (~0.2 ms) and
never waits for the host for the rest of the block except at the QSA selection
fences.

### 2.3 QSA decode keeps its projections queued (`src/qsa.c`)

For a cache-backed session, the two `fg_qsa_submit_host_reads` drains in
`fg_qsa_session_decode` become `fg_vk_flush`. The only blocking point left is
the selection readback (4K+; none below 2048 tokens) or the block-end fence.
Non-cache sessions (legacy/local paths) keep the exact old behaviour.

Why: those two drains were not needed for any host read; they existed to
separate batches. Removing the waits lets the projection recording run while
the previous layer is still executing on the GPU.

## 3. Expected effect vs measured (round 1 estimate, round 2 result)

Round-1 estimate (kept for the record; the measured column is the fleet A/B on
the fix build, pipeline on vs pipeline off):

| change | estimated | measured |
|---|---:|---:|
| instrumentation | 0 | 0 (report-only) |
| per-layer pipeline | -3 to -7 ms/token | combined +0.7-1.2 TPS short/4K/16K/32K |
| QSA queue retention | -1 to -3 ms/token | (same combined figure) |
| **combined** | short 22-25 TPS, 4K 21.5-24 | short 21.76-21.93, 4K 21.02-21.19, 16K 19.91, 32K 17.67 |

The per-layer pipeline estimate was high for the same reason the first A/B
crashed: the recording share of `own_run - gpu_ms` was inferred, and the real
recoverable part is the host/GPU serialization plus the QSA drain turnarounds,
not the full gap. The honest round-2 number is **+~1 TPS at 4K, +0.7-0.9 at
short/32K**, decided by the two attach batteries and the context sweep above.
The `record_ms`/`wait_ms`/`flush_ms` instrumentation remains the tool for
finding the rest.

## 4. Local validation (llvmpipe, correctness only)

```
make all -j8                                        warning-free (-Werror)
tests/test_qsa_prefill                              PASS
tests/test_expert_prefill                           PASS
tests/test_owner_reduce                             PASS
tests/test_fg_vk (focused)                          PASS:
  batch_submission_parity, pipeline_flush_parity (new), gpu_profile,
  qsa_attention, qsa_attention_single, qsa_record_gather,
  expert_graph_replay, expert_graph_fused_q8_0_cooked,
  expert_graph_rejects_overlap, expert_decode_fused_q8_gates,
  qsa_quant_and_bf16, qsa_prefill_prepare, qsa_segmented_index_score,
  qsa_prefill_chunk_liveness, memory_telemetry_and_canary, tensor_view_rebind
tests/test_core                                     1 pre-existing fail
  (tests/test_core.c:457 stale_reserve==FG_ERR_LIMIT)
tests/test_session                                  2 pre-existing fails
  (tests/test_session.c:254/257, verified identical on unmodified main)
```

Round-2 note: the descriptor-set corruption could not be reproduced locally
(llvmpipe consumes descriptor bindings at record time); it was found with the
KHRONOS validation layer on two blades, which reported
`VUID-vkUpdateDescriptorSets-None-03047` for a set in use by a queued command
buffer. `vulkan-validation-layers` was installed on 42 and 43 for that
diagnosis and is inert unless `VK_INSTANCE_LAYERS` is set.

`pipeline_flush_parity` runs a dependent 13-dispatch chain split by flushes,
compares it bit-for-bit with both the single-batch and the synchronous
fallback paths, and checks the flush actually produced extra submissions. On
llvmpipe the async flush chunks are forced through a host-ordered sync before
the next dependent chunk, because lavapipe does not order same-queue
submissions on its own (it will happily execute a later command buffer while
an earlier one is still running). The semaphore chain added in `beadeee` is
what orders them on a real driver; RADV executes a single queue in order, as
the existing async expert-graph path already relies on. Round-2 correction:
the chain orders execution but does not protect host-side rebinding of a
queued submission's descriptor sets; the epoch discipline in `a79792f` /
`443e8ce` is what keeps a queued command buffer's bindings stable until it has
executed.

No fleet access was used for the round-1 accounting. All performance numbers
above come from previously recorded fleet logs in
`bc-250-dbg/results/ab-20260913-204953`; the round-2 measurements are from the
live fleet on 2026-09-14 (see the table at the top).

## 5. Fleet A/B (completed 2026-09-14, pipeline on, fix build)

Deployed and verified: binary
`59b2f559369873ead009cb3673c00efa616013ed3ed239e949e8cbf564a9a5c7` on all
eight blades, standard ring scripts (`FG_PREFILL_RING=1` on rank 0,
`FG_WORKER_OWNER=1` on workers, `FG_DECODE_DIRECT_OUTPUT` unset), caches
dropped before the run.

1. Correctness gate: `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"` -> `[12]` / `[Paris]`.
2. Attach battery x2: `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2` -> 4K decode 21.02-21.19, short 21.76-21.88.
3. Context sweep: `pwsh -NoProfile -Command "& 'D:\workspace\flash-gordon\tools\context-sweep.ps1' -Contexts 16384,32768"` -> 16K 19.91, 32K 17.67.
4. Soak gate: `pwsh -NoProfile -File D:\workspace\fg-work-overlap\tools\pi-stability.ps1` -> PASS (6 stages, 4-turn conversation, 8 ranks, 0 failures).

Promotion bar met: gates [12]/[Paris] and pi PASS, 4K prefill >= 270 TPS
(296-320 measured; no prefill path touched), short and 4K warm decode clearly
above the pipeline-off control. The orchestrator integrates; do not merge.

Instrumented cross-check (optional):
`FG_DECODE_PROFILE=1 FG_FRAME_TRACE=1 FG_DECODE_RING_TRACE=1` on rank 0 and
one worker, 4K prompt + 32 tokens. Per block, compare against the control:
* `submissions=` should rise (per-layer flushes) while `own_run_ms` falls;
* `record_ms + wait_ms` should approach `gpu_ms` instead of `own_run`;
* `QSA_TRACE flush_ms` should drop to tens of microseconds.
If the block does not improve, run the same battery with
`FG_DECODE_PIPELINE=0` on all ranks to isolate the overlap path without a
rebuild; if only one commit is suspect, `beadeee` alone can be tested before
`63dbca1` since the QSA flush calls it.

## 6. Next levers (ranked, not this round)

1. `fg_topk_reduce` 0.2 ms/call at 4K, ~2.4 ms/token across 12 QSA layers. The
   gather only needs the *set* of top-512 blocks (the host overwrites ids with
   cache slots), so a partial selection network that does not fully sort the
   bottom half is worth the shader work.
2. GPU-side page table for `block -> cache slot` so cached contexts need no
   selection readback at all (dec5 item 1b): removes the last per-QSA-layer
   fence (~0.09 ms host turnaround + drain at 4K) once the cache is provably
   authoritative for the whole working set.
3. Expert pair geometry (~1.42 ms/block, instruction-bound at ~150 GB/s) and
   GR-chain fusion (12 small read kernels/layer) remain the largest GPU
   blocks; both are shader work and neither changes this overlap structure.
