# Ring Decode Overlap: Per-Layer Pipelined Submission and QSA Queue Retention (2026-09-14)

Worktree `fg-work-overlap`, branch `perf/decode-overlap`, base `f224fe6`
(main: context sweep harness on top of the validated decode-20.8 TPS state).
Scope: ring decode wall time per token. No shader, pack, manifest, protocol or
prefill-path change. Four commits:

| commit | file(s) | what |
|---|---|---|
| `57b2248` | `include/fg_vk.h`, `src/vk.c`, `src/runtime.c`, `src/qsa.c` | instrumentation: `wait_ms` / `record_ms` in `DECODE_PROFILE`, `flush_ms` in `QSA_TRACE` |
| `e4771ec` | `src/vk.c`, `include/fg_vk.h`, `src/owner.c`, `tests/test_fg_vk.c` | `fg_vk_flush`: async submit + command-buffer rotation + semaphore chain; chained decode block flushes after every layer |
| `aecb1a2` | `src/qsa.c` | cache-backed QSA decode keeps the projection/commit recordings queued until the selection fence |
| `b0d6301` | `src/vk.c` | wait the current slot fence before a standalone dispatch (robustness; not on the decode critical path) |

Gate: `FG_DECODE_PIPELINE=0` disables both overlap paths (per-layer flush and
QSA flush) and restores the previous submission geometry without a rebuild.

## TL;DR

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

This round makes the host record ahead of the GPU (`fg_vk_flush`), keeps the
QSA projection/commit work in the pipeline instead of draining at every layer,
and adds the timestamps needed to see the remaining gap (`wait_ms`,
`record_ms`, `flush_ms`). Expected combined effect **5-9 ms/token**: short
decode 20.8 -> **22-25 TPS**, 4K warm 19.9 -> **21.5-24 TPS**. The fleet A/B
decides; the local box is llvmpipe-only and cannot measure any of it.

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

## 3. Expected effect, honestly

| change | expected | basis |
|---|---:|---|
| instrumentation | 0 | report-only, default path identical |
| `e4771ec` per-layer pipeline | **-3 to -7 ms/token** | per-block cross-submission was 1.19-1.49 ms on rank 0 and 0.5-1.0 ms on the 1-QSA blocks; recording is 3-5x faster than layer execution, so the idle collapses to block-start + select turnarounds. Up to 6 extra submits/block of ~10-25 us host each are hidden behind GPU execution, but the semaphore/fence overhead per boundary is not free; the lower bound reflects that. |
| `aecb1a2` QSA queue retention | **-1 to -3 ms/token** | 12 QSA layers x 2 removed drain turnarounds (~0.05-0.15 ms each) with the selection fence still present. The old `proj_ms` totals (1.3 + 3.6 ms on rank 0) are mostly GPU-busy waits and are *not* all recoverable. |
| **combined** | **-4 to -9 ms/token** | short decode 20.8 -> 22-25 TPS, 4K warm 19.9 -> 21.5-24 TPS |

Uncertainty that the A/B must resolve:

* The size of the recording share of `own_run - gpu_ms` is inferred, not
  directly measured: the new `record_ms`/`wait_ms` split is exactly what the
  re-run should report. If `record_ms` is small compared with `wait_ms`, the
  layer flush will buy less than estimated.
* Extra submissions can raise the *host* cost per token. If a rank's CPU is
  the bottleneck (fabric thread contention), more submits could eat part of
  the win; `submissions=` in `DECODE_PROFILE` shows it directly.
* Semaphore writes on a single queue cost a few microseconds each; 6
  flushes/block add ~24 signals/waits per token across the ring.
* 262K contexts cross index segments and stress the record cache; the QSA
  change is neutral there (it only removes waits), but the per-layer flush
  interacts with segment allocation, which is why the segment materialisation
  is still done at QSA entry.

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

`pipeline_flush_parity` runs a dependent 13-dispatch chain split by flushes,
compares it bit-for-bit with both the single-batch and the synchronous
fallback paths, and checks the flush actually produced extra submissions. On
llvmpipe the async flush chunks are forced through a host-ordered sync before
the next dependent chunk, because lavapipe does not order same-queue
submissions on its own (it will happily execute a later command buffer while
an earlier one is still running). The semaphore chain added in `e4771ec` is
what orders them on a real driver; RADV executes a single queue in order, as
the existing async expert-graph path already relies on.

No fleet access was used for this work (the 262K sweep is running). All
performance numbers above come from previously recorded fleet logs in
`bc-250-dbg/results/ab-20260913-204953`.

## 5. Fleet A/B (orchestrator)

Deploy from `D:\workspace\fg-work-overlap` (do not serve from this
worktree). Build warning-free, stage binary + `vulkan/` (SPIR-V unchanged; no
shader source touched), same ring pack and env as the previous session
(`FG_PREFILL_RING=1` on rank 0, `FG_WORKER_OWNER=1` on workers,
`FG_DECODE_DIRECT_OUTPUT` unset on every rank).

1. Correctness gate (64 max tokens; expect `[12]` and `Paris`):
   `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"`
2. Like-for-like attach battery against the current control binary:
   `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2`
3. Ring/soak gate:
   `pwsh -NoProfile -File D:\workspace\fg-work-overlap\tools\pi-stability.ps1`

Promotion bar: gates [12]/[Paris] and pi PASS, 4K prefill >= 270 TPS
(unchanged; no prefill path touched), short decode and 4K warm decode clearly
above control.

Instrumented cross-check (one blade per rank, optional but decisive):
`FG_DECODE_PROFILE=1 FG_FRAME_TRACE=1 FG_DECODE_RING_TRACE=1` on rank 0 and
one worker, 4K prompt + 32 tokens. Per block, compare against the control:
* `submissions=` should rise (per-layer flushes) while `own_run_ms` falls;
* `record_ms + wait_ms` should approach `gpu_ms` instead of `own_run`;
* `QSA_TRACE flush_ms` should drop to tens of microseconds.
If the block does not improve, run the same battery with
`FG_DECODE_PIPELINE=0` on all ranks to isolate the overlap path without a
rebuild; if only one commit is suspect, `e4771ec` alone can be tested before
`aecb1a2` since the QSA flush calls it.

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
