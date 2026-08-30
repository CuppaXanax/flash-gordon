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

## Verified Burn-Down

This ledger contains only fleet-qualified behavior that landed. It is experiment accounting, not a forecast of the endpoint.

### Tagged LKG

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

### Current qualified frame

Fastest verified behavior: `7f70b88` (two clean deterministic fleet-qualified runs; no LKG tag below 7.5 tok/s).

| Measurement | GPU worker reduction |
|---|---:|
| Correct response | Exact baseline text, including Paris |
| Steady tail decode | 5.78-5.81 tok/s |
| Token 30 wall | 181.533 ms |
| Layer graph | 143.190 ms |
| Rank-0 GPU / kernels | 70.887 / 69.433 ms |
| Vulkan submissions / dispatches | 122 / 1,536 |
| Pre-route `sync1` | 60.585 ms |
| Expert fire | 11.053 ms |
| Shared expert | 9.971 ms |
| Expert collect/join | 58.328 ms |
| Finish | 3.245 ms |
| Non-layer work | 38.343 ms |

Token 30 deliberately uses dynamic recording so Vulkan timestamps remain available. Ordinary fixed-graph tokens 26-29 and 31 measured 135.3-144.4 ms across the 48 layers and 51.0-55.4 ms in collect. The final `decode complete` summary is not a valid rate when EOS stops generation early because it divides the requested cap by elapsed time. The per-token cumulative rates above use the actual 64 completed decode frames.

Verified landed burn-down from the 255.369 ms historical baseline:

| Landed architecture | Qualified frame delta | Resulting token-30 wall |
|---|---:|---:|
| Resident causal QSA | correctness fix; no retained wall-time claim | 258.191 ms candidate median |
| Hardware CRC32C | -46.359 ms versus resident-QSA median | 211.832 ms |
| HC/GDN specialization | -10.485 ms versus CRC LKG | 201.347 ms |
| Residual successor folding | -4.001 ms | 197.346 ms |
| GPU expert reduction | -14.833 ms | 182.513 ms |
| Fixed worker graph replay | 30-150 us per matched worker request; no material whole-frame claim | 182.375 ms profiled fallback |
| Remove legacy worker common graph | capacity prerequisite; no material whole-frame claim | 181.533 ms |

These deltas describe what happened. They are not summed forward as a performance ceiling.

## Victory Gap

```text
CURRENT FRAME:       181.5 ms / 5.80 TPS
NEXT GATE:           133.3 ms / 7.5 TPS
NEXT-GATE GAP:        48.2 ms

TARGET FRAME:         50.0 ms / 20 TPS
TOTAL FRAME GAP:     131.5 ms
```

The subsystem budget debt is approximately 135.5 ms because the 47.0 ms subtotal must also create the unspent 3.0 ms frame reserve. This is deadline ownership, not a forecast derived from currently measured candidates.

| Frame subsystem | Current | 20 TPS budget | State | Architecture that owns the debt |
|---|---:|---:|---|---|
| 48-layer graph | 143.190 ms | 37.5 ms | **RED** | Compiled rank-0 and worker graphs; BC250/Qwen3.8 kernels; GPU job transport |
| Average layer | 2,983 us | 781 us | **RED** | Fixed deterministic layer program |
| `sync1` | 1,262 us/layer | 350 us/layer | **RED** | Pre-recorded rank-0 resource graph and native packed/subgroup Q8 |
| Expert collect proxy | 1,212 us/layer | 300 us/layer | **RED** | Fixed worker jobs, <=180 us expert unit, doorbell fabric, hidden shared expert |
| Join/handoff | 69 us/layer | 50 us/layer | **RED** | GPU contribution consumption and successor handoff |
| N-gram exposed | about 24.8 ms cold | 3.0 ms | **RED** | Resident distributed heads with useful-row responses |
| Output | about 9.8 ms | 4.0 ms | **RED** | Production Q8 projection and hierarchical greedy argmax |
| Embedding/setup | about 0.15 ms | 0.5 ms | **GREEN** | Preserve |

### Gate 7.5: remove 49.2 ms

The next gate is not limited to the optimizations already measured. It assigns required frame reduction to architectural work that can cross 133.3 ms:

| Workstream being implemented | Required contribution to this gate |
|---|---:|
| Lower-pressure BC250 expert unit after compiled worker replay | 18-22 ms |
| Resident distributed n-gram service after worker scratch slimming | 21-22 ms |
| Production output projection plus hierarchical argmax | 5-6 ms |
| First compiled rank-0 common-graph tranche | 2-5 ms |

The ranges are gate assignments, not predicted savings. If one workstream misses its assignment, its architecture changes or another new workstream is added; the 133.3 ms deadline does not move. Crossing 7.5 TPS creates the next LKG, after which the ledger resets to `CURRENT`, `NEXT GATE: 100.0 ms / 10 TPS`, and the newly measured gap.

### Full 20 TPS implementation

Closing the complete frame requires all of the following architectural surfaces, including work beyond the 7.5 TPS gate:

1. A fixed worker GPU job system: pre-recorded commands/descriptors, fixed scratch, fixed ten-tile schedules, native packed/subgroup quant kernels, GPU weighting, and one minimal contribution.
2. A compiled rank-0 layer program: resource barriers, fixed pre-route/post-route commands, GPU route publication, shared/local expert overlap, and direct successor handoff.
3. A <=3 ms resident n-gram service with no filesystem block amplification in the token frame.
4. A <=4 ms output pass using the production Q8 primitive and hierarchical greedy argmax.
5. A <=100 us shared expert hidden under the remote job, plus <=50 us request and <=45 us response paths driven as queue operations rather than RPC construction.

These workstreams own the 50.0 ms deadline. No finite endpoint above 50 ms is asserted unless a physical hardware bound is demonstrated.

## Frozen Critical-Path Evidence

The audit is accepted and frozen. The measurements below remain implementation evidence; they are not the active work product.

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

Fleet-qualified production integrations:

- HC injection (`43daa2b`): 24 workgroups compute four partial dots and HC mix reduces them. Numerical parity and the eight-rank France gate pass. Versus the CRC LKG, token-30 wall fell from 211.832 to 205.637 ms and rank-0 GPU fell from 82.298 to 74.821 ms.
- GDN recurrence (`7ca7519`): preserve the state update and compute `S' q = decay * (S q) + delta * (k^T q)` without rereading updated state. State remains exact and the eight-rank France gate passes. Versus HC alone, token-30 wall fell from 205.637 to 201.347 ms and rank-0 GPU fell from 74.821 to 71.060 ms.

Both candidates are wired into inference and retained.

## N-Gram Locality Audit

The canonical trace contains 64 decode positions after prompt warming:

- 1,047 requested 4 KiB blocks, 203 hits and 844 misses: 19.39% hit rate.
- 961 unique decode blocks and zero overlap between consecutive tokens.
- An 8 MiB 4-way cache reaches 19.29%; every size from 16 through 512 MiB reaches the same 19.39%. Larger cache capacity has no value on this trace.
- Two-gram heads hit 20.6-24.2%; three-gram heads hit 15.1-20.0%.
- Each token needs only 1,440 useful bytes, but filesystem alignment turns misses into 32-72 KiB of direct reads. Cold batches remain 24.5-24.75 ms.

Exact addresses depend on the newly emitted token, so history-only prefetch cannot begin during the preceding layer graph. Starting lookup at token availability and joining before layer 1 can overlap only layer 0: 3.571 ms median, leaving about 21.1 ms exposed.

Full resident sharding costs 28.8 GB total, 3.6 GB for two heads per blade. `323f37d` removed dead worker owner/QSA and distributed-layer arenas; live workers now use 9.64-10.29 GiB GTT with 8.47-9.31 GiB system memory available. Six two-head shards plus one three-head shard now fit across ranks 1-7; rank 6 is the lowest-GTT placement for the three-head shard. Storage-only sharding still does not help because one NVMe already issues all head reads concurrently and the network adds about 157 us each way.

## Victory Workstreams

Ordered by the next RED stage being implemented, not by a sum-of-candidates endpoint:

1. Complete the dependency-aware expert frame graph after retained GPU weighted reduction. Remaining work is fixed per-layer descriptors and command buffers; the monolithic gate/up/SwiGLU fusion shape regressed and generic preposted receives remain out of scope.
2. Complete the rank-0 resource graph after the retained HC/GDN integrations. The two production kernels reduced token-30 wall by 10.485 ms and rank-0 GPU by 11.238 ms versus the CRC LKG.
3. Complete residual graph compilation after retained successor folding. Folding removed 47 submissions and reduced token-30 wall by 4.001 ms, but 4.110 ms of measured finish remains.
4. Replace the output pass with production Q8 plus hierarchical argmax and enforce the 4.0 ms budget.
5. Replace the current n-gram storage path with resident distributed execution and enforce the 3.0 ms exposed budget.

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

**Status:** HC AND GDN INTEGRATED; GRAPH COMPILATION PENDING

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

**Status:** GPU WEIGHTED REDUCTION AND COMMAND REPLAY INTEGRATED; PAIRED FUSION REJECTED; EXPERT KERNEL SPECIALIZATION PENDING

- Pre-record worker command buffers per owned layer using fixed activation, schedule, scratch, and weight addresses.
- Dispatch a fixed maximum tile count with invalid unused entries.
- Reuse descriptors and commands together; descriptor caching alone was insufficient.
- Apply the same packed/subgroup execution to local and remote experts.

**Gate:** worker request latency <=0.35-0.45 ms, complete ten-slot route coverage, and lower coordinator join time. Worker throughput alone does not count.

## 5. Compile Rank 0's Frame Graph

**Status:** RESIDUAL SUCCESSOR FOLDING INTEGRATED; FULL GRAPH PENDING

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

- `8410b16`: one kernel streamed gate then up weights through shared memory and applied SwiGLU after both reductions. It reduced rank-0 dispatches from 1,541 to 1,492 but regressed token-30 wall from 182.513 to 188.719 ms, layer wall from 145.545 to 151.893 ms, and steady decode from 5.73-5.77 to 5.00-5.12 tok/s. Worker totals rose by 1.67-3.39 ms and the greedy continuation diverged. Reverted by `c282a1c`.
- `910b02b`: batched io_uring sends plus preposted `MSG_WAITALL` receives regressed to 3.44 tok/s / 290.980 ms. Reverted.
- `3c29be5`: direct blocking sockets produced 3.91 tok/s / 259.221 ms, an immaterial regression. Reverted.
- `4978c48`: guessed 4-row/64-lane expert geometry regressed to 3.37-3.38 tok/s. Reverted.
- Persistent descriptor caching alone had no material gain. Reverted.
- Clock tuning and transport API selection are not current work items.

## Burn-Down Log

Update this section after every fleet-qualified experiment with hypothesis, milestone budget assignment, commit, correctness result, before/after frame time, useful aggregate GB/s, and keep/revert decision.

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

### HC and algebraic GDN integration

- Hypothesis: production-shaped HC injection and algebraic GDN recurrence remove low-width launch waste and one recurrent-state read without changing model state.
- Change: integrate 24-piece HC partial injection in `43daa2b`, then the sibling-projection and algebraic recurrence graph in `7ca7519`.
- Qualification: both revisions passed Paris, all 48 expert layers, core/Vulkan/model/tokenizer/fabric tests, and homogeneous eight-blade fingerprints.
- Result: HC reduced token-30 wall from 211.832 to 205.637 ms and rank-0 GPU from 82.298 to 74.821 ms. GDN then reduced token-30 wall to 201.347 ms, layer wall to 164.581 ms, and rank-0 GPU to 71.060 ms. The integrated run reported 5.20 tok/s steady tail.
- Decision: keep both integrations.

### Residual-write successor folding

- Hypothesis: carrying layers 0-46 final residual writes into their successors removes one host submission per layer while preserving the exact dependency graph.
- Change: defer each nonterminal `fg_gr_write` and record it at the front of the successor's existing command batch; retain layer 47 as the terminal write. Layer 1 consumes layer 0 inside the PLE batch.
- Qualification: `ef03e6d` passed Paris, all 48 expert layers, all local and fleet suites, and homogeneous eight-blade fingerprints. Dispatches remained 1,518 while submissions fell from 170 to 123 exactly.
- Result: versus `7ca7519`, token-30 wall fell from 201.347 to 197.346 ms, layer wall from 164.581 to 160.562 ms, and finish from 8.881 to 4.110 ms. GPU time remained 71.060 ms, confirming a host/submission-path gain.
- Decision: keep. The 4.110 ms finish still misses the <=0.5 ms generation-2 budget, but the 4.001 ms whole-frame improvement is material; terminal/frame-tail work remains open.

### GPU expert reduction

- Hypothesis: worker-side CPU weighting still reads every selected 2,560-float down vector through write-combined UMA. Reducing by routing weight in the existing GPU batch should leave one vector to read and encode per participating rank.
- Change: `773319c` adds a compact-order GPU FMA reduction after expert down projection, returns the existing pre-reduced sentinel for both local and remote routes, and removes the remote CPU reduction loop. The worker submission count is unchanged.
- Qualification: all eight blades passed core/Vulkan/model/tokenizer/fabric tests with homogeneous fingerprints. Paris, all 48 expert layers, and exact generated-response comparison against `ef03e6d` passed.
- Result: steady decode rose from 5.24-5.26 to 5.73-5.77 tok/s. Token-30 wall fell from 197.346 to 182.513 ms, layer wall from 160.562 to 145.545 ms, and collect from 74.323 to 57.955 ms. Worker CPU reduction fell to zero; per-rank worker totals fell by 2.97-5.35 ms. Rank-0 GPU rose only 0.169 ms for 23 local reduction dispatches.
- Decision: keep as the fastest verified mainline. Do not tag before the 7.5 tok/s threshold.

### Fixed worker graph replay

- Hypothesis: fixed per-layer commands and descriptors can remove worker tensor lookup, descriptor allocation/update/free, and command recording while preserving live activation, schedule, and gate state.
- Milestone assignment: contribute to the 49.2 ms gap to the 7.5 tok/s gate; the assignment remains open until the complete worker unit supplies its required 18-22 ms.
- Change: `b02fb2f` pre-records one five-dispatch, ten-tile command buffer per locally owned layer with persistent descriptors and scratch addresses. Invalid tiles exit uniformly; profiled tokens retain the dynamic path.
- Qualification: sparse one/two-tile replay with poisoned intermediates is bit-exact against the dynamic graph. All eight blades passed the full suite with homogeneous fingerprints; Paris, the retained tail token sequence, and all 48 expert layers passed.
- Result: matched worker requests save about 30-150 us. Ordinary tokens measured 135.3-144.4 ms layer graphs and 51.0-55.4 ms collect; steady decode was 5.76-5.78 tok/s. Token 30's dynamic-profile fallback measured 182.375 ms.
- Decision: keep as job-system infrastructure. This slice did not materially reduce the 49.0 ms next-gate gap; expert kernel specialization and resident n-gram work remain required.

### Worker residency reclamation and deterministic join

- Hypothesis: the accepted expert-parallel runtime does not need worker owner/QSA state or distributed common-layer prefill/decode arenas; removing them should create enough residency for the required three-head n-gram shard without changing the frame graph.
- Milestone assignment: make resident distributed n-gram execution physically deployable for the 7.5 tok/s gate.
- Change: `323f37d` restricts workers to session, expert decode/prefill, and rank-4 output messages. `7f70b88` orders pre-reduced contributions by source rank so changed worker timing cannot change accumulation order.
- Qualification: all local and eight-blade suites passed. Two clean France runs produced identical tail token sequences, Paris, and complete 48-layer traces at 5.78-5.81 tok/s.
- Result: worker GTT is 9.64-10.29 GiB with 8.47-9.31 GiB system memory available. Token-30 wall is 181.533 ms. This is a capacity prerequisite, not a claimed frame-time win.
- Decision: keep. The resident 16-head service is now the active implementation, with two heads on six ranks and three heads on rank 6.

### Fused expert gate/up/SwiGLU

- Hypothesis: loading routed activation tiles once and replacing gate, up, and SwiGLU with one mixed-Q4/Q5 kernel would save two dispatches and two intermediate rows per worker request.
- Change: `8410b16` streamed gate and up blocks sequentially through one shared buffer, preserved each quant type and stride independently, and wrote `mid` directly.
- Qualification: local mixed-Q4/Q5 CPU parity and all eight fleet suites passed; the 48-layer France trace passed, but the greedy continuation diverged from the retained result.
- Result: versus `773319c`, rank-0 dispatches fell from 1,541 to 1,492, but token-30 wall rose from 182.513 to 188.719 ms, layer wall rose from 145.545 to 151.893 ms, collect rose from 57.955 to 65.864 ms, and steady decode fell from 5.73-5.77 to 5.00-5.12 tok/s. The larger fused execution cost outweighed launch savings on GFX1013.
- Decision: revert in `c282a1c`. Do not retry this sequential two-weight-stream shader; pursue fixed command/descriptors or a lower-pressure fusion geometry instead.
