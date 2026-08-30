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

## Validated Baseline

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

## 1. Resident Causal QSA

**Status:** CAUSAL PATH KEPT; WHOLE-FRAME GATE OPEN

The current decode path records QSA projection/quantization into an outer Vulkan batch and then maps those outputs before submission. It also performs synchronous NVMe page/header writes and selected-page reads inside every QSA layer, and sorts a 4,096-entry tile even when fewer than 512 blocks exist.

- Keep hot QSA records in UMA: 2.1 MiB at the current 147-token run, 463.5 MiB at 32K, 927 MiB at 64K.
- Make record commit, index-key publication, selection, gather, and attention a causal GPU graph.
- Remove NVMe reads/writes from token frame time; checkpoint completed records outside the frame.
- Bypass top-k while complete blocks fit within the 512-block selection budget.
- Add multi-layer, multi-token tests that detect stale and cross-layer records.

**Gate:** QSA state must match a sequential oracle; no hot-window storage I/O in the frame; all 12 QSA layers remain valid; target >=15 ms/token frame reduction.

## 2. Reshape Rank-0 Projections

**Status:** PENDING

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
