# Flash Gordon 10.035 TPS Trace

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

| Stage | Mean ms | Median ms | Min ms | Max ms | Frame share |
|---|---:|---:|---:|---:|---:|
| Embedding | 0.093 | 0.092 | 0.088 | 0.112 | 0.09% |
| N-gram | 0.353 | 0.349 | 0.338 | 0.416 | 0.35% |
| 48 layers | 95.329 | 95.261 | 94.658 | 96.310 | 95.67% |
| Output | 3.872 | 3.869 | 3.845 | 3.943 | 3.89% |
| **Total** | **99.647** | **99.603** | **99.040** | **100.623** | **100.00%** |

The qualification TPS is the reciprocal of the unrounded final-20 mean, not the cumulative `decode complete` display and not the instrumented token-30 result.

## Qualified Changes

1. Convert routed Q4_K/Q5_K gate/up and Q5_1 down tensors to size-neutral GPU-native layouts in memory after rank-file verification.
2. Execute those layouts with cooked routed-expert kernels while preserving routing slots, gates, reductions, and exact generated tokens.
3. Convert the common `10240 x 320` HC down projection to the existing size-neutral Q8 layout and execute it as an eight-way split-K projection plus reduction.
4. Elide the first redundant dispatch barrier in a fresh Vulkan batch.
5. Preserve immediate request order while replacing synchronous io_uring submit/completion handling with direct blocking `send()` and `recv(MSG_WAITALL)` on the already-connected TCP sockets.

Batched request fan-out is not enabled: its unprofiled final-20 result regressed to approximately 104.7 ms because delaying the first peer outweighed the reduced submission count.

## Profiled Trace

Token 30 enables Vulkan timestamps, detailed routes, and worker records. It is validation evidence, not a steady-frame estimate.

| Layer phase | 7.924 TPS LKG ms | Current ms | Delta ms | Current mean/layer |
|---|---:|---:|---:|---:|
| `sync1` | 45.298 | 43.602 | -1.696 | 0.908 |
| Fire | 10.604 | 15.314 | +4.710 | 0.319 |
| Shared expert | 9.190 | 9.366 | +0.176 | 0.195 |
| Collect | 59.726 | 28.345 | -31.381 | 0.590 |
| Finish | 3.108 | 3.139 | +0.031 | 0.065 |
| **Layer total** | **127.932** | **99.760** | **-28.172** | **2.078** |

Detailed send and route clocks inflate the profiled fire phase. The behavior gate uses the unprofiled 95.329 ms layer mean.

| Family | Layers | Total mean ms | `sync1` mean ms | Collect mean ms |
|---|---:|---:|---:|---:|
| GDN | 35 | 2.072 | 0.898 | 0.592 |
| QSA | 12 | 2.064 | 0.914 | 0.582 |
| GDN + PLE | 1 | 2.469 | 1.215 | 0.642 |

## Route And Blade Coverage

The validator reports 22 local routes and 156 remote routes. Rank 0 executes 61 selected experts locally; workers execute the remaining 419 selections.

| Rank | Requests | Selections | GPU ms | Send ms | Total ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 53 | 9.210 | 0.520 | 9.960 |
| 2 | 22 | 58 | 9.610 | 0.530 | 10.350 |
| 3 | 23 | 53 | 9.810 | 0.590 | 10.610 |
| 4 | 22 | 59 | 9.450 | 0.500 | 10.150 |
| 5 | 24 | 77 | 11.360 | 0.570 | 12.190 |
| 6 | 23 | 63 | 10.130 | 0.530 | 10.900 |
| 7 | 20 | 56 | 9.220 | 0.480 | 9.880 |

## Layer Table

`Delta` is current total minus the qualified 7.924 TPS token-30 layer total. `Slow` identifies the longest observed remote worker request for that layer.

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

All times are milliseconds. Raw unprofiled, coordinator, worker, layer-analysis, and fleet-fingerprint captures are retained in the fleet diagnostics workspace with the `ee5a0bb` prefix.