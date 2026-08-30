# Flash Gordon 20 TPS Burn-Down

## Contract

Flash Gordon is an eight-BC250 Qwen3.8 appliance. Raw greedy single-session decode is the frame loop, rank 0 is the game thread, and expert workers are useful only when they shorten that frame. Pipeline parallelism, MTP, and concurrent sessions are out of scope until this contract is met.

- Correctness is absolute: canonical Qwen3.8 state progression, routing, logits, and generated text must remain valid.
- Release floor: 10 tok/s raw single-stream decode.
- Engineering target: 20 tok/s, or 50 ms/token.
- Hardware-side stretch: approximately 35 tok/s.
- Active bytes: 6.0-6.4 GB/token. At 20 tok/s the fleet must sustain 120-129 GB/s of useful aggregate model traffic.
- Every promotion requires the real eight-rank France evaluation, the 48-layer expert-parallel trace validator, Vulkan timestamps, and homogeneous fleet fingerprints.
- Isolated benchmarks may reject an idea cheaply; they cannot promote it.

## Historical Baseline

Source-equivalent LKG: `3269b08` / `9caeecb`.

| Measurement | Baseline |
|---|---:|
| Correct response | Paris |
| Steady decode | 3.92 tok/s |
| Token 30 wall | 255.369 ms |
| Rank-0 GPU / kernels | 87.169 / 85.460 ms |
| Vulkan submissions / dispatches | 171 / 1,534 |
| Pre-route `sync1` | 92.825 ms |
| Expert fire | 24.461 ms |
| Shared expert | 9.966 ms |
| Expert collect/join | 102.793 ms |
| Finish | 8.114 ms |
| Non-layer tail | 17.216 ms |

Target frame allocation:

| Stage | Baseline | 20 TPS budget |
|---|---:|---:|
| Pre-route rank-0 graph | 92.8 ms | 32-34 ms |
| Expert launch, shared work, and join | 137.2 ms | 9-11 ms |
| Final layer writes | 8.1 ms | <=1 ms |
| Output and other tail work | 17.2 ms | 5-6 ms |
| Total | 255.4 ms | 47-52 ms |

## Current LKG

Behavior commit/tag: `96fd0a2` / `lkg-4.91tps-crc32c`. Later critical-path and benchmark commits are behavior-neutral unless explicitly enabled.

| Measurement | Three-run median |
|---|---:|
| Correct response | Paris |
| Steady tail decode | 4.91 tok/s |
| Token 30 wall | 211.832 ms |
| Layer graph | 175.020 ms |
| Mean layer | 3.646 ms |
| Rank-0 GPU / kernels | 82.298 / 80.653 ms |
| Vulkan submissions / dispatches | 170 / 1,518 |
| Pre-route `sync1` | 71.089 ms |
| Expert fire | 10.908 ms |
| Shared expert | 9.777 ms |
| Expert collect/join | 74.314 ms |
| Finish | 9.050 ms |
| Non-layer work | 36.812 ms |
| Useful aggregate traffic | 28.32-30.21 GB/s |

Neighboring unprofiled tokens have a 174.1 ms median layer graph, so token-30 worker profiling does not change the priority order.

## Critical-Path Audit

Four endpoint timestamps per remote request were aligned with an NTP-style offset estimate. Local durations and coordinator readiness are exact; one-way request/response estimates assume path symmetry. Timestamp query pools are primed before capture, removing the false first-use spikes previously charged to embedding and layer 2.

Post-CRC critical-worker medians:

| Component | Median |
|---|---:|
| A. Actual slowest-worker GPU | 786 us/layer |
| B. Estimated request path | 157 us/layer |
| Worker receive + bookkeeping | 22 us/layer |
| C. Worker reduction + encoding | 69 us/layer |
| D. Worker synchronous result send call | 24 us/layer |
| E. Estimated response path | 157 us/layer |
| F. Coordinator header + payload mechanics | 12 us/result |
| G. Coordinator CRC + result decode | 16 us/result |

For remote-only layers, median collect is 1,478 us: 1,290 us waiting for required results, about 152 us receiving all results, and about 66 us validating/decoding them. Coordinator serialization after the final required result becomes available is only about 59 us. The remaining collect path is principally worker GPU plus request/response latency, not `poll()` or payload-copy mechanics.

The source audit did confirm serialized fire sends. Before hardware CRC, synchronous send calls consumed 14.54 of 24.19 ms/token. After hardware CRC they consume 2.76 of 10.91 ms/token, averaging 17.6 us/request. Generic batching can recover at most that 2.76 ms and is not the next transport experiment.

### CRC32C result

The former bit-at-a-time Castagnoli implementation cost 243.6 us per 10 KiB result validation and the same algorithm also inflated frame encoding on both send directions. Runtime-dispatched Zen 2 SSE4.2 CRC32C reduced validation to 2.8 us/result while retaining the portable fallback and identical wire checksums.

- Layer graph: 216.734 -> 175.020 ms median.
- Fire: 23.836 -> 10.908 ms.
- Collect: 102.310 -> 74.314 ms.
- Token wall: 258.191 -> 211.832 ms versus the resident-QSA median, a 46.359 ms / 17.96% reduction.

## Sync1 Dependency Audit

The router/activation CPU-visible fence is a real dependency and remains. The defect before that fence is that generic `dispatch()` inserts a global compute-write to compute-read/write barrier before every dispatch, including independent siblings.

| Region | Current barriers | Resource barriers | Removable/token |
|---|---:|---:|---:|
| First HC read | 6/layer | 4/layer | 96 |
| Second HC read | 6/layer | 5/layer | 48 |
| GDN qkv/z/alpha/beta graph | 7/GDN layer | 4/GDN layer | 108 |
| QSA projection/prepare/quant graph | 13/QSA layer | 6/QSA layer | 84 |
| Router + activation quantization | 2/layer | 1/layer | 48 |
| Layer-1 PLE | 8 | 5 | 3 |
| **Total** | | | **387** |

A resource-aware graph removes 387 global barriers but no dispatches by itself. Production BC250 A/Bs show about 3-18 us per unnecessary barrier depending on shape, bounding barrier-only savings to a few milliseconds/token. Dispatch fusion remains a separate optimization.

Measured benchmark-only production candidates:

- HC injection: 24 workgroups compute four partial dots and HC mix reduces them. Numerical parity passes. `dense_f32(10240->4)+gr_mix` falls from 106.9 to 38.2 us GPU per HC read, predicting 6.595 ms/token across 96 calls.
- GDN recurrence: preserve the state update and compute `S' q = decay * (S q) + delta * (k^T q)` without rereading updated state. State remains exact and output passes the production tolerance. RADV falls from 217.4 to 161.9 us/GDN layer, predicting 1.999 ms/token.

Neither benchmark candidate is wired into inference yet.

## N-Gram Locality Audit

The canonical trace contains 64 decode positions after prompt warming:

- 1,047 requested 4 KiB blocks, 203 hits and 844 misses: 19.39% hit rate.
- 961 unique decode blocks and zero overlap between consecutive tokens.
- An 8 MiB 4-way cache reaches 19.29%; every size from 16 through 512 MiB reaches the same 19.39%. Larger cache capacity has no value on this trace.
- Two-gram heads hit 20.6-24.2%; three-gram heads hit 15.1-20.0%.
- Each token needs only 1,440 useful bytes, but filesystem alignment turns misses into 32-72 KiB of direct reads. Cold batches remain 24.5-24.75 ms.

Exact addresses depend on the newly emitted token, so history-only prefetch cannot begin during the preceding layer graph. Starting lookup at token availability and joining before layer 1 can overlap only layer 0: 3.571 ms median, leaving about 21.1 ms exposed.

Full resident sharding costs 28.8 GB total, 3.6 GB for two heads per blade. Sealed worker residency is already 10.73-11.78 GiB, rank 0's replicated runtime reports 13.978 GiB, and distributing 16 heads over seven workers requires at least one three-head/5.4 GB assignment. It is not safe without first shrinking worker scratch/model residency. Storage-only sharding does not help because one NVMe already issues all head reads concurrently and the network adds about 157 us each way.

## Ranked Experiments

Ranked by predicted whole-frame savings, not API novelty:

1. Dependency-aware expert frame graph: target the remaining slowest-worker GPU/command path, not generic preposted receives. Predicted 19-26 ms/token from collect plus 6-8 ms from route/fire preparation.
2. Rank-0 resource graph plus measured HC/GDN candidates: 8.6 ms measured kernel savings plus roughly 2-4 ms barrier savings; predicted 10-13 ms/token.
3. Carry 47 final residual writes into the next layer graph: current finish is 9.050 ms while its kernels total 0.083 ms; predicted 8.7-9.0 ms/token.
4. Output hierarchical argmax and corrected Q8 projection: predicted about 4.8 ms/token to the existing <5 ms gate.
5. Start n-gram reads before layer 0: bounded to 3.571 ms/token with the current storage layout.

Even these near-term changes leave the frame around 145-160 ms. Reaching 10 TPS also requires the typed/subgroup Q8 work and lower expert GPU time. Reaching 20 TPS requires those changes plus removal of the remaining n-gram storage floor and deeper rank-0/expert graph compilation.

## 1. Resident Causal QSA

**Status:** CAUSAL PATH KEPT; WHOLE-FRAME GATE OPEN

The retired QSA path recorded projection/quantization into an outer Vulkan batch and then mapped those outputs before submission. It also performed synchronous NVMe page/header writes and selected-page reads inside every QSA layer, and sorted a 4,096-entry tile even when fewer than 512 blocks existed.

- Keep hot QSA records in UMA: 2.1 MiB at the current 147-token run, 463.5 MiB at 32K, 927 MiB at 64K.
- Make record commit, index-key publication, selection, gather, and attention a causal GPU graph.
- Remove NVMe reads/writes from token frame time; checkpoint completed records outside the frame.
- Bypass top-k while complete blocks fit within the 512-block selection budget.
- Add multi-layer, multi-token tests that detect stale and cross-layer records.

**Gate:** QSA state must match a sequential oracle; no hot-window storage I/O in the frame; all 12 QSA layers remain valid; target >=15 ms/token frame reduction.

## 2. Reshape Rank-0 Projections

**Status:** BENCHMARKED; INTEGRATION PENDING

- Fuse HC Q8 low-rank projection and four-row F32 injection into one mixed projection launch.
- Fuse GDN QKV, gate, alpha, and beta projections by shared input dependency.
- Fuse router and shared-scalar projection.
- Avoid materializing normalized 10,240-wide intermediates where group scales and gamma can be consumed directly.
- Treat low-output-width utilization as the defect: HC injection currently moves only 15.7 MB/token but costs about 8 ms.

**Gate:** production-dimension parity, rank-0 GPU <=65 ms after items 1-2, and a material whole-frame improvement.

## 3. Replace Generic Q8 Decode

**Status:** PENDING

- Use typed 16-bit access to the native 34-byte Q8_0 blocks.
- Remove repeated scale decode and avoid reconstructed unaligned dword loads.
- Use native subgroup reduction instead of LDS reduction ladders.
- Select multi-row geometry from production dimensions, not synthetic shapes.

**Gate:** rank-0 Q8 scopes fall from about 39.6 ms to <=25 ms, CPU/Vulkan production parity passes, and aggregate useful bandwidth moves toward 120-129 GB/s without semantic drift.

## 4. Compile the Expert Frame Graph

**Status:** PENDING

- Pre-record worker command buffers per owned layer using fixed activation, schedule, scratch, and weight addresses.
- Dispatch a fixed maximum tile count with invalid unused entries.
- Reuse descriptors and commands together; descriptor caching alone was insufficient.
- Apply the same packed/subgroup execution to local and remote experts.

**Gate:** worker request latency <=0.35-0.45 ms, complete ten-slot route coverage, and lower coordinator join time. Worker throughput alone does not count.

## 5. Compile Rank 0's Frame Graph

**Status:** PENDING

- Pre-record one pre-route graph and one post-route graph per layer.
- Put shared and local expert work in the post-route graph.
- Merge layer N's final residual write into layer N+1's pre-route graph.
- Preserve the router readback and expert completion dependencies.
- Reduce approximately 171 submissions toward 99.

**Gate:** pre-route plus finish <=35 ms, exact state/logit parity, and no additional frame bubbles.

## 6. Fix Output as the Frame Tail

**Status:** PENDING

- Replace the 512-finalist bitonic sort with hierarchical argmax for greedy decode.
- Reuse the corrected Q8 primitive for output projection.
- Keep output execution on rank 4 unless the integration trace proves movement shortens the frame.

**Gate:** rank-4 output latency <5 ms with identical token and logit.

## Rejected Approaches

- `910b02b`: batched io_uring sends plus preposted `MSG_WAITALL` receives regressed to 3.44 tok/s / 290.980 ms. Reverted.
- `3c29be5`: direct blocking sockets produced 3.91 tok/s / 259.221 ms, an immaterial regression. Reverted.
- `4978c48`: guessed 4-row/64-lane expert geometry regressed to 3.37-3.38 tok/s. Reverted.
- Persistent descriptor caching alone had no material gain. Reverted.
- Clock tuning and transport API selection are not current work items.

## Burn-Down Log

Update this section after every fleet-qualified experiment with hypothesis, predicted ceiling, commit, correctness result, before/after frame time, useful aggregate GB/s, and keep/revert decision.

### Resident causal QSA candidate

- Hypothesis: QSA's premature host reads, synchronous page/header writes, selected-page reads, and unnecessary short-context top-k account for at least 15 ms of the token frame and violate causal state ownership.
- Change: bounded resident record/index history, GPU causal commit, GPU top-k/gather above 512 blocks, direct contiguous attention below that threshold, and post-generation checkpointing.
- Predicted result: zero `fg_topk_reduce` calls in QSA at token 30, QSA layer total reduced by at least 15 ms/token, correct Paris response, and a complete 48-layer EP trace.
- Qualification: all eight RADV blades passed core, Vulkan, model-load, tokenizer, and fabric suites with homogeneous binary, SPIR-V, and manifest fingerprints. Paris and the 48-layer EP validator passed on every retained sample.
- Three-run qualified median (`6f7f4d2` / `73f2dde`): 4.08 tok/s steady, 258.191 ms token wall, 218.566 ms layer wall, 82.253 ms rank-0 GPU, 0.760 ms QSA state/attention, 170 submissions, and 1,518 dispatches. Useful aggregate traffic is 24.48-26.11 GB/s.
- Versus baseline: layer wall improved by 19.587 ms and pre-route `sync1` improved by 22.135 ms, but median token wall regressed by 2.822 ms. The QSA mechanism passed; the required whole-frame reduction did not.
- Frame-trace correction: the apparent 27-28 ms embedding stage is profiler query-pool first-use cost. Neighboring unprofiled embedding calls are 0.09-0.11 ms. Cold n-gram lookups are the recurring tail: 8-16 useful direct reads (32-72 KiB) spend 24.5-24.75 ms inside io_uring; hashing, packing, and dequantization stay below 0.15 ms.
- Decision: keep the causal resident-QSA path because the prior path consumed stale GPU data and performed state I/O in-frame. Item 1 remains open until the newly isolated n-gram tail is removed or hidden and the whole-frame gate passes.
- Rejected follow-ups: disabling NVMe APST did not move n-gram latency; polled I/O is unsupported through the NVMe/LVM/XFS stack; padding useful reads to QD64 did not reduce ordinary misses and was reverted in `c106573`.

### Hardware CRC32C

- Hypothesis: synchronous transport looked expensive because every frame used a bit-at-a-time CRC32C implementation, not because io_uring completion itself dominated.
- Change: runtime-dispatched SSE4.2 CRC32C on x86-64 with the portable implementation retained as fallback; added standard golden vectors and preserved corruption tests.
- Qualification: three clean eight-rank runs passed Paris, all 48 expert layers, core/Vulkan/model/tokenizer/fabric tests, and homogeneous fingerprints.
- Result: 4.91 tok/s steady median, 211.832 ms token wall, 175.020 ms layer wall, and 28.32-30.21 GB/s useful aggregate traffic.
- Decision: keep and tag `96fd0a2` as `lkg-4.91tps-crc32c`.
