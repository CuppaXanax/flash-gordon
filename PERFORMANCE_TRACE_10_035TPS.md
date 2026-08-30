# Flash Gordon 10.035 TPS Trace

## Management Readout

| Question | Exact evidence | Decision |
|---|---|---|
| Did the run qualify? | Final-20 mean `99.64675 ms`, or `10.03545 TPS`; exact response and token parity; all 48 layers and routes validated | Preserve commit `340e7eb` and tag `lkg-10.035tps-cooked-experts` as the LKG |
| Is there operating margin? | Mean headroom is only `0.353 ms`; 17/20 frames are at or below 100 ms, P95 is `100.400 ms`, and max is `100.623 ms` | Treat 10 TPS as qualified, but not yet comfortably provisioned |
| Where is steady wall time? | Layers are `95.329 ms` (`95.67%`); output is `3.872 ms` (`3.89%`) | Work on repeated layer cost first; output is the only material non-layer slice |
| Is one layer pathological? | Profiled layer mean is `2.078 ms`, standard deviation `0.088 ms`; reducing the three largest layers to the mean saves only `0.815 ms` | Do not start an isolated-layer cleanup campaign |
| Is placement the next lever? | Collect/remote-count correlation is `0.161`; all seven worker ranks appear on the slow-worker path | Do not derive a map from this prompt; retain round-robin ownership |
| What is the next general avenue? | Worker gate, up, and down kernels are balanced; coordinator cooked dense kernels own `18.863 ms` of token-30 GPU time | First test a general expert-kernel improvement, then the common cooked-dense path |

The France prompt is a validation oracle only, never placement training data. Twenty TPS is not one narrow optimization away: its 50 ms frame budget requires `49.647 ms`, or `49.82%`, off this LKG.

## Qualification

This is the complete qualified trace for `lkg-10.035tps-cooked-experts` on eight 24-CU BC250 blades. It is raw greedy single-stream decode with one token per frame, expert parallelism only, resident distributed n-gram, hierarchical argmax, round-robin expert ownership, and no expert-placement map.

- Mean final-20 frame: 99.64675 ms, or 10.03545 tok/s.
- Median final-20 frame: 99.603 ms, or 10.040 tok/s equivalent.
- Exact France response and generated token IDs match the retained baseline.
- All 48 layers, 48 route records, 156 remote worker requests, and 480 selected expert slots pass the EP trace validator.
- All ranks 0-7 report one manifest, binary, and cooked-shader fingerprint.
- Rank files are hash-verified before selected tensors are transformed in memory. No model file is rewritten or transferred.
- Candidate identity: base `cb2ad0b5d9e0883a53dfa964fa89813d2ce28555`, patch `ee5a0bbf375453002dbd192525dcbb769ec9612683947b3ed8f34d39b7136acf`.
- Runtime identity: binary `6cf6054c7a7b72c07187c55bea3d66c80b1fb4ee7f38f1efa94bc706ed261fc5`, manifest `9aecb1456a372a41c444b0d966b8801d2d00b9f2f9db254e6a2ebf651a433bbf`, cooked shaders `5704cdb1d514a23bea37a3766d48f9663e46419f7fb232c8e30bcaadafe51d11`.

## Steady Frame

Final 20 unprofiled frames, tokens 42-61:

| Stage | Mean | Median | P95 | Min | Max | Std. dev. | Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Embedding | 0.093 | 0.092 | 0.095 | 0.088 | 0.112 | 0.005 | 0.09% |
| N-gram | 0.353 | 0.349 | 0.369 | 0.338 | 0.416 | 0.017 | 0.35% |
| 48 layers | 95.329 | 95.261 | 96.086 | 94.658 | 96.310 | 0.409 | 95.67% |
| Output | 3.872 | 3.869 | 3.882 | 3.845 | 3.943 | 0.019 | 3.89% |
| **Total** | **99.647** | **99.603** | **100.400** | **99.040** | **100.623** | **0.403** | **100.00%** |

The standard deviation is the population standard deviation of these 20 frames. Layer variation explains essentially all frame variation; the other stages are stable at this scale. The qualification TPS is the reciprocal of the unrounded final-20 mean, not the cumulative `decode complete` display and not the instrumented token-30 result.

## Qualified Changes

1. Convert routed Q4_K/Q5_K gate/up and Q5_1 down tensors to size-neutral GPU-native layouts in memory after rank-file verification.
2. Execute those layouts with cooked routed-expert kernels while preserving routing slots, gates, reductions, and exact generated tokens.
3. Convert the common `10240 x 320` HC down projection to the existing size-neutral Q8 layout and execute it as an eight-way split-K projection plus reduction.
4. Elide the first redundant dispatch barrier in a fresh Vulkan batch.
5. Preserve immediate request order while replacing synchronous io_uring submit/completion handling with direct blocking `send()` and `recv(MSG_WAITALL)` on the already-connected TCP sockets.

Batched request fan-out is not enabled: its unprofiled final-20 result regressed to approximately 104.7 ms because delaying the first peer outweighed the reduced submission count.

## Profiled Trace

Token 30 enables Vulkan timestamps, detailed routes, and worker records. It is validation evidence, not a steady-frame estimate.

### Measurement Domains

The trace contains deliberately different measurement domains. They must not be added together as if they came from one uninstrumented frame.

| Domain | Population | Result | Proper use |
|---|---|---:|---|
| Qualification wall | Unprofiled tokens 42-61 | `99.64675 ms` total, `95.32905 ms` layers | Throughput and latency authority |
| Profiled outer frame | Token 30 | `105.993 ms` total, `101.535 ms` layers | Scope and trace container only |
| Profiled per-layer timers | Token 30, 48 layers | `99.760 ms` summed | Relative phase ownership |
| Coordinator Vulkan profile | Token 30, rank 0 | `43.926 ms` GPU, `42.457 ms` kernels | Coordinator shader ownership |
| Production expert wall | Unprofiled tokens 26-27, 322 worker requests | Fixed-graph timings below | Worker elapsed-time authority |
| Profiled expert kernels | Token 30, 156 worker requests | `35.676 ms` aggregate remote GPU | Per-kernel ownership and selected-count scaling |

The profiled outer layer loop is `6.206 ms` above the unprofiled layer mean. Of that, `1.775 ms` is outside the individual layer timers, including per-layer trace output. More importantly, profiling intentionally bypasses the production pre-recorded expert command graph so that individual kernels can receive timestamps. Token-30 worker wall and local-expert wall therefore describe the instrumented five-dispatch path; tokens 26-27 describe the production fixed graph.

Cross-host request and response path estimates below use four timestamps and assume symmetric one-way delay. They are useful for scale and ordering, not additive accounting.

### Layer Phases

| Layer phase | 7.924 TPS LKG ms | Current ms | Delta ms | Current mean/layer | Profiled share |
|---|---:|---:|---:|---:|---:|
| `sync1` | 45.298 | 43.602 | -1.696 | 0.908 | 43.71% |
| Fire | 10.604 | 15.314 | +4.710 | 0.319 | 15.35% |
| Shared expert | 9.190 | 9.366 | +0.176 | 0.195 | 9.39% |
| Collect | 59.726 | 28.345 | -31.381 | 0.590 | 28.41% |
| Finish | 3.108 | 3.139 | +0.031 | 0.065 | 3.15% |
| **Layer total** | **127.932** | **99.760** | **-28.172** | **2.078** | **100.00%** |

The independently rounded phase columns sum to `99.766 ms`; the independently rounded per-layer Total column sums to `99.760 ms`. The `0.006 ms` difference is decimal formatting, not unassigned execution.

Detailed send and route clocks inflate the profiled Fire phase. The behavior gate uses the unprofiled 95.329 ms layer mean.

- `sync1`: deferred prior-layer residual write, optional layer-1 PLE, GR attention read, GDN or QSA, attention write, GR FFN read, router, activation quantization, and the router-visible Vulkan synchronization.
- Fire: CPU top-k handling, round-robin partitioning, request encoding and sends, plus token-30 route logging.
- Shared expert: the coordinator shared-expert GPU batch while remote requests are in flight.
- Collect: coordinator-local routed experts first, then remote arrivals in readiness order.
- Finish: routed-expert reduction and deferred residual write, except for the final layer's immediate write.

### Layer Shape

| Layer family | Layers | Profiled total | Mean/layer | Share |
|---|---:|---:|---:|---:|
| GDN | 35 | 72.520 ms | 2.072 ms | 72.69% |
| QSA | 12 | 24.768 ms | 2.064 ms | 24.83% |
| GDN + PLE | 1 | 2.469 ms | 2.469 ms | 2.47% |

GDN and QSA layer means are effectively equal. GDN owns more frame time because there are 35 GDN-only layers, not because an individual GDN layer is slower.

| Rank | Layer | Type | Total | Primary excess |
|---:|---:|---|---:|---|
| 1 | 1 | GDN + PLE | 2.469 ms | `sync1=1.215 ms`; the only PLE layer |
| 2 | 47 | QSA | 2.316 ms | `collect=0.776 ms`, `finish=0.163 ms`; final write is not deferred |
| 3 | 2 | GDN | 2.265 ms | `collect=0.734 ms`, `fire=0.374 ms` |
| 4 | 46 | GDN | 2.181 ms | `collect=0.675 ms` |
| 5 | 0 | GDN | 2.166 ms | `collect=0.671 ms` |

Across the 48 rows, total time correlates most strongly with Collect (`r=0.786`) and `sync1` (`r=0.672`). Fire is weaker (`r=0.375`). Collect has little correlation with the recorded slow-worker elapsed time (`r=0.112`) or remote fan-out (`r=0.161`), because local work, network arrival, and serial coordinator receive work also participate.

| Remote ranks in layer | Layers | Mean total | Mean Fire | Mean Collect |
|---:|---:|---:|---:|---:|
| 2 | 6 | 2.075 ms | 0.280 ms | 0.609 ms |
| 3 | 24 | 2.042 ms | 0.309 ms | 0.570 ms |
| 4 | 18 | 2.128 ms | 0.345 ms | 0.612 ms |

Reducing the three largest layers to the 48-layer mean saves only `0.815 ms`. This is a route-coverage observation for the validation token, not a placement training signal.

### Coordinator GPU Ownership

Rank 0 records `43.926 ms` of Vulkan GPU time: `42.457 ms` in named kernels and `1.469 ms` of Vulkan timestamped overhead. Its `62.086 ms` wall residual includes CPU work, distributed waits, rank-4 output, and profiling overhead.

| Semantic family | GPU time | Kernel share |
|---|---:|---:|
| GDN projection + recurrent + output | 16.564 ms | 39.01% |
| GR attention and FFN reads | 9.548 ms | 22.49% |
| QSA projection + state attention + output | 5.439 ms | 12.81% |
| Coordinator-local routed expert | 4.880 ms | 11.49% |
| Shared expert | 3.467 ms | 8.17% |
| Router + activation quantization | 2.191 ms | 5.16% |
| PLE | 0.207 ms | 0.49% |
| GR writes | 0.154 ms | 0.36% |

| Shader | Calls | GPU time | Kernel share |
|---|---:|---:|---:|
| `fg_dense_q8_0_cooked.spv` | 398 | 18.863 ms | 44.43% |
| `fg_gdn_recurrent_algebraic.spv` | 36 | 5.848 ms | 13.77% |
| `fg_dense_f32.spv` | 168 | 3.312 ms | 7.80% |
| `fg_moe_kquant_cooked.spv` | 44 | 3.239 ms | 7.63% |
| `fg_gr_mix_partial.spv` | 96 | 2.532 ms | 5.96% |
| `fg_dense_q8_0_cooked_split.spv` | 96 | 1.947 ms | 4.59% |
| `fg_group_rms_norm.spv` | 99 | 1.431 ms | 3.37% |
| `fg_moe_q5_1_down_cooked.spv` | 21 | 1.297 ms | 3.06% |

The cooked dense shader is the largest single coordinator target and spans GDN, GR, shared-expert, QSA, and PLE scopes. A general improvement there is preferable to another special-case layer rewrite.

### Expert Execution

Token 30 contains 156 remote requests and all 480 routed slots: 419 remote selections and 61 coordinator-local selections. Twenty-two layers have a local route. All seven worker ranks become the recorded slow rank on at least four layers; no blade is a universal straggler.

Production fixed-graph worker timings come from the unprofiled token-26 and token-27 records in the same exact capture. `Expert elapsed` brackets the pre-recorded graph execution on the host. `Worker service` runs from completed request payload receipt through response send completion.

| Selected experts | Requests | Expert elapsed mean | Expert elapsed P95 | Worker service mean |
|---:|---:|---:|---:|---:|
| 1 | 74 | 0.292 ms | 0.307 ms | 0.326 ms |
| 2 | 88 | 0.356 ms | 0.392 ms | 0.390 ms |
| 3 | 76 | 0.412 ms | 0.470 ms | 0.446 ms |
| 4 | 55 | 0.457 ms | 0.520 ms | 0.491 ms |
| 5 | 23 | 0.519 ms | 0.666 ms | 0.554 ms |
| 6 | 5 | 0.558 ms | 0.567 ms | 0.594 ms |
| 7 | 1 | 0.622 ms | 0.622 ms | 0.658 ms |

Across all 322 fixed-graph requests, expert elapsed averages `0.387 ms` and worker service averages `0.421 ms`. Taking the slowest worker service in each of 96 token/layer groups gives P50 `0.490 ms`, P90 `0.554 ms`, P95 `0.597 ms`, and max `0.715 ms`.

Token 30 supplies the shader split that the fixed graph cannot expose. Across its 48 slowest profiled workers, GPU time is `14.873 ms`; named kernels account for `14.575 ms`:

| Critical-worker kernel scope | GPU time | Named-kernel share |
|---|---:|---:|
| Gate projection | 4.831 ms | 33.15% |
| Up projection | 4.822 ms | 33.08% |
| Down projection | 4.671 ms | 32.05% |
| Reduce | 0.196 ms | 1.34% |
| Activation | 0.055 ms | 0.38% |

Gate, up, and down are balanced; there is no single expert leg to remove. Relative to the prior paired-Q8 LKG under the same profiling method, cooked experts reduce critical-worker wall total from `54.174` to `26.778 ms`, critical-worker P95 from `1.419` to `0.654 ms`, and all-worker GPU total from `92.143` to `35.676 ms`. That is why Collect fell by `31.381 ms` in the profiled comparison.

### Collect Critical Path

Collect begins by executing the coordinator-local routed expert, if any, and only then polls remote responses. The local-before-poll total is concentrated entirely in the 22 local-route layers: `427.27 us/layer` there versus `0.48 us/layer` in the other 26 layers. This is instrumented-path wall, so it is an ordering fact rather than a production fixed-graph estimate.

| Collect component | Mean/layer | Sum across 48 | Collect share |
|---|---:|---:|---:|
| Local routed expert before first poll | 196.09 us | 9.412 ms | 33.21% |
| Aggregate poll wait | 231.22 us | 11.098 ms | 39.15% |
| Frame headers | 15.35 us | 0.737 ms | 2.60% |
| Payload reads | 92.37 us | 4.434 ms | 15.64% |
| Frame validation | 9.36 us | 0.449 ms | 1.58% |
| Result decode | 45.56 us | 2.187 ms | 7.71% |
| Tail after final decode | 0.27 us | 0.013 ms | 0.05% |

Rounding in the per-layer phase records accounts for the approximately `0.015 ms` difference from the `28.345 ms` Collect total.

For the response that makes each layer complete, token-30 medians are `181.44 us` inferred request path, `7.14 us` worker receive, `2.66 us` bookkeeping, `295 us` GPU timestamps inside a `503.26 us` expert-call wall, `9.68 us` reduction/encode, `22.85 us` send call, and `181.44 us` inferred response path. These terms overlap with the coordinator shared-expert window and must not be summed into a synthetic layer time.

Once the final required peer is socket-ready, only `52.03 us/layer` remains on average before Collect ends, about `2.50 ms` across the token. This bounds the direct opportunity in receive ordering and decode cleanup. Fire's 156 send calls total only `2.354 ms` of its instrumented `15.314 ms`; the remaining token-30 Fire wall includes routing, encoding, and synchronous trace logging.

### Next-Avenue Ranking

| Priority | Avenue | Evidence and falsifiable gate |
|---:|---|---|
| 1 | General cooked expert kernels | Gate/up/down own `98.28%` of critical-worker named kernel time. Test an all-route optimization such as a combined cooked gate/up pass, not an exact-route graph. Require exact parity, at least `0.10 ms` improvement in fixed-graph slowest-worker P50, and at least `4 ms` final-20 layer-wall improvement before adoption. |
| 2 | Common cooked-dense path | `fg_dense_q8_0_cooked.spv` owns `18.863 ms` (`44.43%`) of coordinator kernel time; GDN plus GR reads own `61.50%`. Require a reproducible kernel win and at least `1.5 ms` final-20 frame improvement. |
| 3 | Rank-4 output | Output is a stable `3.872 ms` and currently has only wall attribution in the retained capture. First retain rank-4 output kernel records, then require at least `0.75 ms` wall improvement. |
| 4 | New receive architecture | Only about `2.50 ms` remains after final socket readiness. Revisit transport only with a mechanism materially different from the rejected preposting, direct-receive, busy-poll, batched-send, and scatter/gather variants. |

The fixed expert graph is already pre-recorded as one five-dispatch command buffer and waits on one fence. Exact-N graph variants and fence spinning have already failed fleet qualification. A future graph experiment must remove a real dependency or fuse general computation; re-recording the same work is not a new avenue.

| Target | Frame budget | Required saving from LKG |
|---:|---:|---:|
| 10.0 TPS | 100.000 ms | Already clears by 0.353 ms on mean |
| 10.5 TPS | 95.238 ms | 4.409 ms |
| 11.0 TPS | 90.909 ms | 8.738 ms |
| 12.0 TPS | 83.333 ms | 16.314 ms |
| 15.0 TPS | 66.667 ms | 32.980 ms |
| 20.0 TPS | 50.000 ms | 49.647 ms |

## Route And Blade Coverage

The validator reports 22 local routes and 156 remote routes. Rank 0 executes 61 selected experts locally; workers execute the remaining 419 selections.

| Rank | Requests | Selections | Expert-call wall ms | Send ms | Total ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 53 | 9.210 | 0.520 | 9.960 |
| 2 | 22 | 58 | 9.610 | 0.530 | 10.350 |
| 3 | 23 | 53 | 9.810 | 0.590 | 10.610 |
| 4 | 22 | 59 | 9.450 | 0.500 | 10.150 |
| 5 | 24 | 77 | 11.360 | 0.570 | 12.190 |
| 6 | 23 | 63 | 10.130 | 0.530 | 10.900 |
| 7 | 20 | 56 | 9.220 | 0.480 | 9.880 |

These rank totals are token-30 worker host intervals. They are not Vulkan GPU timestamps and are not additive across blades.

## Layer Table

`Delta` is current total minus the qualified 7.924 TPS token-30 layer total. `Slow` identifies the longest observed instrumented remote worker request for that layer.

| L | Type | Total | Delta | Sync1 | Fire | Shared | Collect | Finish | Remote | Local sel | Slow rank | Slow sel | Slow ms | Sub | Disp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | GDN | 2.166 | -0.555 | 0.928 | 0.310 | 0.193 | 0.671 | 0.063 | 3 | 2 | 1 | 3 | 0.540 | 3 | 34 |
| 1 | GDN+PLE | 2.469 | -0.721 | 1.215 | 0.351 | 0.197 | 0.642 | 0.064 | 4 | 0 | 2 | 3 | 0.530 | 3 | 38 |
| 2 | GDN | 2.265 | -0.229 | 0.899 | 0.374 | 0.196 | 0.734 | 0.062 | 4 | 0 | 3 | 4 | 0.690 | 2 | 30 |
| 3 | QSA | 2.126 | -0.712 | 0.909 | 0.309 | 0.192 | 0.653 | 0.063 | 3 | 5 | 4 | 3 | 0.470 | 3 | 41 |
| 4 | GDN | 2.150 | -0.472 | 0.899 | 0.330 | 0.192 | 0.666 | 0.063 | 4 | 0 | 5 | 5 | 0.700 | 2 | 30 |
| 5 | GDN | 2.005 | -0.476 | 0.899 | 0.302 | 0.192 | 0.549 | 0.063 | 3 | 3 | 5 | 4 | 0.550 | 3 | 35 |
| 6 | GDN | 2.016 | -0.816 | 0.881 | 0.308 | 0.201 | 0.563 | 0.063 | 3 | 0 | 6 | 5 | 0.590 | 2 | 30 |
| 7 | QSA | 2.006 | -0.568 | 0.901 | 0.301 | 0.209 | 0.532 | 0.063 | 3 | 2 | 4 | 4 | 0.540 | 3 | 41 |
| 8 | GDN | 1.996 | -0.766 | 0.897 | 0.285 | 0.193 | 0.559 | 0.062 | 2 | 4 | 5 | 4 | 0.540 | 3 | 35 |
| 9 | GDN | 2.066 | -0.477 | 0.885 | 0.345 | 0.195 | 0.579 | 0.063 | 4 | 0 | 4 | 4 | 0.540 | 2 | 30 |
| 10 | GDN | 2.129 | -0.454 | 0.909 | 0.359 | 0.198 | 0.601 | 0.063 | 4 | 0 | 5 | 5 | 0.600 | 2 | 30 |
| 11 | QSA | 2.024 | -0.500 | 0.928 | 0.303 | 0.194 | 0.536 | 0.062 | 3 | 2 | 4 | 4 | 0.530 | 3 | 41 |
| 12 | GDN | 2.071 | -0.691 | 0.896 | 0.326 | 0.190 | 0.596 | 0.062 | 4 | 0 | 5 | 5 | 0.600 | 2 | 30 |
| 13 | GDN | 2.040 | -0.575 | 0.904 | 0.307 | 0.193 | 0.574 | 0.063 | 3 | 1 | 5 | 4 | 0.550 | 3 | 35 |
| 14 | GDN | 2.076 | -0.770 | 0.890 | 0.331 | 0.190 | 0.601 | 0.063 | 4 | 0 | 7 | 4 | 0.560 | 2 | 30 |
| 15 | QSA | 2.046 | -0.753 | 0.910 | 0.303 | 0.191 | 0.580 | 0.063 | 3 | 0 | 2 | 5 | 0.600 | 2 | 36 |
| 16 | GDN | 2.069 | -0.419 | 0.894 | 0.338 | 0.192 | 0.584 | 0.062 | 3 | 3 | 1 | 3 | 0.470 | 3 | 35 |
| 17 | GDN | 2.134 | -0.407 | 0.886 | 0.331 | 0.191 | 0.664 | 0.062 | 4 | 0 | 2 | 3 | 0.490 | 2 | 30 |
| 18 | GDN | 2.069 | -0.561 | 0.894 | 0.353 | 0.196 | 0.564 | 0.063 | 4 | 0 | 7 | 3 | 0.480 | 2 | 30 |
| 19 | QSA | 2.028 | -0.491 | 0.916 | 0.325 | 0.191 | 0.533 | 0.063 | 3 | 2 | 4 | 4 | 0.530 | 3 | 41 |
| 20 | GDN | 2.160 | -0.348 | 0.903 | 0.359 | 0.194 | 0.640 | 0.064 | 4 | 0 | 4 | 3 | 0.470 | 2 | 30 |
| 21 | GDN | 2.047 | -0.432 | 0.893 | 0.301 | 0.193 | 0.598 | 0.063 | 3 | 4 | 2 | 3 | 0.480 | 3 | 35 |
| 22 | GDN | 2.035 | -0.908 | 0.898 | 0.303 | 0.191 | 0.580 | 0.063 | 3 | 0 | 7 | 5 | 0.620 | 2 | 30 |
| 23 | QSA | 2.037 | -0.866 | 0.906 | 0.282 | 0.190 | 0.597 | 0.062 | 2 | 3 | 7 | 6 | 0.640 | 3 | 41 |
| 24 | GDN | 1.975 | -0.534 | 0.887 | 0.279 | 0.190 | 0.556 | 0.062 | 2 | 4 | 5 | 5 | 0.590 | 3 | 35 |
| 25 | GDN | 2.033 | -0.520 | 0.891 | 0.300 | 0.190 | 0.590 | 0.062 | 3 | 0 | 1 | 5 | 0.590 | 2 | 30 |
| 26 | GDN | 2.065 | -0.792 | 0.900 | 0.330 | 0.192 | 0.579 | 0.063 | 4 | 0 | 5 | 6 | 0.660 | 2 | 30 |
| 27 | QSA | 1.998 | -0.520 | 0.900 | 0.309 | 0.203 | 0.523 | 0.063 | 3 | 2 | 6 | 4 | 0.530 | 3 | 41 |
| 28 | GDN | 2.088 | -0.389 | 0.891 | 0.331 | 0.189 | 0.615 | 0.063 | 4 | 0 | 5 | 4 | 0.540 | 2 | 30 |
| 29 | GDN | 2.006 | -0.806 | 0.892 | 0.313 | 0.253 | 0.485 | 0.063 | 3 | 2 | 2 | 5 | 0.600 | 3 | 35 |
| 30 | GDN | 2.102 | -0.335 | 0.888 | 0.365 | 0.195 | 0.590 | 0.064 | 4 | 0 | 1 | 3 | 0.550 | 2 | 30 |
| 31 | QSA | 2.104 | -0.751 | 0.924 | 0.292 | 0.193 | 0.609 | 0.086 | 2 | 0 | 2 | 6 | 0.640 | 2 | 36 |
| 32 | GDN | 2.040 | -0.483 | 0.897 | 0.322 | 0.195 | 0.564 | 0.062 | 3 | 3 | 3 | 4 | 0.530 | 3 | 35 |
| 33 | GDN | 2.018 | -0.816 | 0.897 | 0.310 | 0.191 | 0.559 | 0.062 | 3 | 0 | 2 | 5 | 0.610 | 2 | 30 |
| 34 | GDN | 2.046 | -0.889 | 0.900 | 0.364 | 0.192 | 0.529 | 0.062 | 4 | 0 | 5 | 4 | 0.540 | 2 | 30 |
| 35 | QSA | 2.014 | -0.530 | 0.899 | 0.304 | 0.192 | 0.555 | 0.064 | 3 | 2 | 6 | 4 | 0.530 | 3 | 41 |
| 36 | GDN | 2.082 | -0.506 | 0.901 | 0.337 | 0.192 | 0.589 | 0.063 | 4 | 0 | 7 | 3 | 0.490 | 2 | 30 |
| 37 | GDN | 2.015 | -0.609 | 0.899 | 0.303 | 0.192 | 0.558 | 0.064 | 3 | 2 | 6 | 4 | 0.550 | 3 | 35 |
| 38 | GDN | 2.075 | -0.783 | 0.904 | 0.338 | 0.194 | 0.577 | 0.063 | 4 | 0 | 7 | 5 | 0.600 | 2 | 30 |
| 39 | QSA | 2.021 | -0.781 | 0.918 | 0.274 | 0.209 | 0.559 | 0.061 | 2 | 4 | 4 | 4 | 0.550 | 3 | 41 |
| 40 | GDN | 1.994 | -0.516 | 0.905 | 0.307 | 0.194 | 0.525 | 0.064 | 3 | 3 | 5 | 3 | 0.490 | 3 | 35 |
| 41 | GDN | 2.059 | -0.720 | 0.899 | 0.301 | 0.192 | 0.604 | 0.063 | 3 | 0 | 6 | 5 | 0.590 | 2 | 30 |
| 42 | GDN | 2.083 | -0.509 | 0.899 | 0.348 | 0.198 | 0.576 | 0.062 | 4 | 0 | 3 | 4 | 0.540 | 2 | 30 |
| 43 | QSA | 2.053 | -0.487 | 0.935 | 0.323 | 0.195 | 0.536 | 0.064 | 3 | 1 | 3 | 5 | 0.590 | 3 | 41 |
| 44 | GDN | 2.077 | -0.777 | 0.896 | 0.319 | 0.192 | 0.609 | 0.061 | 3 | 0 | 5 | 4 | 0.550 | 2 | 30 |
| 45 | GDN | 2.085 | -0.515 | 0.912 | 0.304 | 0.194 | 0.611 | 0.064 | 3 | 1 | 5 | 4 | 0.530 | 3 | 35 |
| 46 | GDN | 2.181 | -0.317 | 0.912 | 0.336 | 0.194 | 0.675 | 0.064 | 4 | 0 | 7 | 3 | 0.560 | 2 | 30 |
| 47 | QSA | 2.316 | -0.320 | 0.916 | 0.269 | 0.191 | 0.776 | 0.163 | 2 | 6 | 2 | 3 | 0.540 | 4 | 42 |

## Evidence Artifacts

The exact captures are retained in the fleet diagnostics workspace:

- `expert-cooked-ee5a0bb-final20.txt`: unprofiled final-20 qualification frames.
- `ep-trace-ee5a0bb-coordinator.txt`: token-30 layer, route, coordinator GPU, send, and receive records.
- `ep-trace-ee5a0bb-workers.txt`: unprofiled token-26/27 fixed-graph records and token-30 profiled worker records.
- `expert-cooked-ee5a0bb-layer-analysis.txt`: validated 48-row layer reduction.
- `expert-cooked-ee5a0bb-fleet-fingerprints.txt`: all-rank model, binary, and shader identity.
- `expert-cooked-ee5a0bb-critical-path.txt`: recomputed coordinator/worker transport decomposition.
- `expert-cooked-ee5a0bb-worker-distribution.txt`: recomputed prior-LKG/current worker comparison.
- `expert-cooked-ee5a0bb-management-analysis.txt`: recomputed percentiles, correlations, outliers, and kernel ownership.

The exact qualification identities are recorded above. Artifact names use the accepted patch prefix `ee5a0bb`; that prefix is not the runtime binary digest.

All times in the full layer table are milliseconds.