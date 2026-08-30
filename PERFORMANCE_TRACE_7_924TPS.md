# Flash Gordon 7.924 TPS Trace

## Qualification

This is the complete qualified trace for `lkg-7.924tps-paired-q8` on eight 24-CU BC250 blades. It is raw greedy single-stream decode with one token per frame, expert parallelism only, resident distributed n-gram, hierarchical argmax, and selective size-neutral paired-scale Q8.

- Mean tail: 126.200 ms / 7.924 tok/s over 20 consecutive frames.
- Median tail: 126.000 ms / 7.937 tok/s equivalent.
- Exact France response and all 20 retained overlapping token IDs match the generic-Q8 baseline.
- All 48 layers, 48 routes, 157 remote worker requests, and 480 selected expert slots pass the EP trace validator.
- Candidate identity: base `6732507`, qualification patch `7fa6e2b505e72183d36ce0aa1616f6bfb9d87b5a782772e555b2f7b5727c40bf`.

## Steady Frame

Final 20 unprofiled frames, tokens 63-82:

| Stage | Mean ms | Median ms | Min ms | Max ms | Frame share |
|---|---:|---:|---:|---:|---:|
| Embedding | 0.100 | 0.097 | 0.094 | 0.119 | 0.08% |
| N-gram | 0.386 | 0.385 | 0.360 | 0.426 | 0.31% |
| 48 layers | 121.753 | 121.721 | 118.299 | 124.983 | 96.53% |
| Output | 3.891 | 3.899 | 3.843 | 3.922 | 3.08% |
| **Total** | **126.129** | **126.089** | **122.650** | **129.413** | **100.00%** |

The independently derived frame-delta result is 126.200 ms / 7.924 tok/s. The 0.071 ms difference from the stage-trace mean is logging/timestamp rounding.

## Ten TPS Gap

The next deadline is 100.0 ms, leaving 26.2 ms to remove. This is measured ownership, not a sum of candidate savings:

| Surface | Current | 100 ms implication | State |
|---|---:|---:|---|
| Non-layer stages | 4.377 ms mean | Preserve; already below their combined 7.5 ms 20-TPS allocation | GREEN |
| Layer graph | 121.753 ms mean | Must fall to about 95.623 ms if non-layer time stays fixed | RED |
| Mean layer | 2.536 ms | Must fall by about 0.544 ms/layer to about 1.992 ms/layer | RED |
| Output | 3.891 ms mean | Preserve below 4.0 ms | GREEN |
| N-gram | 0.386 ms mean | Preserve below 3.0 ms | GREEN |
| Embedding | 0.100 ms mean | Preserve below 0.5 ms | GREEN |

Token 30 is instrumented, so its 127.932 ms layer graph is slower than the steady 121.753 ms mean. Its phase split identifies ownership:

| Layer phase | Generic Q8 ms | Paired Q8 ms | Delta ms | Paired mean/layer | Profiled layer share |
|---|---:|---:|---:|---:|---:|
| `sync1` | 60.269 | 45.298 | -14.971 | 0.944 | 35.41% |
| Fire | 10.607 | 10.604 | -0.003 | 0.221 | 8.29% |
| Shared expert | 9.523 | 9.190 | -0.333 | 0.191 | 7.18% |
| Collect | 58.908 | 59.726 | +0.818 | 1.244 | 46.69% |
| Finish | 3.154 | 3.108 | -0.046 | 0.065 | 2.43% |
| **Layer total** | **142.461** | **127.932** | **-14.529** | **2.665** | **100.00%** |

`sync1 + collect` is 105.024 ms, or 82.10% of the profiled layer graph. Paired Q8 already took 14.971 ms out of `sync1`; collect did not improve. The 10 TPS work therefore belongs principally to:

1. Expert collect/critical-worker completion and its request/response path.
2. Remaining `sync1` common graph, router readback, and submission boundaries.
3. The layer-1 PLE exception and high-selection routed-expert outliers.

No finite saving is assigned to these surfaces until an implementation is measured.

## Layer Families

| Family | Layers | Total mean ms | `sync1` mean ms | Collect mean ms |
|---|---:|---:|---:|---:|
| GDN | 35 | 2.648 | 0.935 | 1.233 |
| QSA | 12 | 2.671 | 0.942 | 1.263 |
| GDN + PLE | 1 | 3.190 | 1.270 | 1.429 |

QSA is not a broad outlier after resident state and cooked Q8. Layer 1 is the unique common-graph outlier. Routed selection imbalance drives the largest collect tails: layers 22, 23, 31, and 34 each wait on a six-selection remote worker.

## Layer Table

`Delta` is paired-scale total minus the qualified generic-Q8 token-30 layer total. `Slow worker` is the longest observed remote worker request for the layer.

| L | Type | Total | Delta | Sync1 | Fire | Shared | Collect | Finish | Remote | Local sel | Slow worker | Sel | Worker ms | Sub | Disp |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | GDN | 2.721 | -0.342 | 0.936 | 0.210 | 0.190 | 1.322 | 0.063 | 3 | 2 | 1 | 3 | 1.250 | 3 | 32 |
| 1 | GDN+PLE | 3.190 | -0.246 | 1.270 | 0.240 | 0.189 | 1.429 | 0.063 | 4 | 0 | 2 | 3 | 1.330 | 3 | 36 |
| 2 | GDN | 2.494 | -0.324 | 0.936 | 0.245 | 0.189 | 1.048 | 0.077 | 4 | 0 | 3 | 4 | 1.050 | 2 | 28 |
| 3 | QSA | 2.838 | -0.273 | 0.938 | 0.211 | 0.190 | 1.436 | 0.062 | 3 | 5 | 4 | 3 | 0.980 | 3 | 39 |
| 4 | GDN | 2.622 | -0.310 | 0.947 | 0.246 | 0.192 | 1.174 | 0.063 | 4 | 0 | 5 | 5 | 1.120 | 2 | 28 |
| 5 | GDN | 2.481 | -0.310 | 0.930 | 0.208 | 0.189 | 1.091 | 0.063 | 3 | 3 | 5 | 4 | 1.040 | 3 | 33 |
| 6 | GDN | 2.832 | -0.300 | 0.934 | 0.210 | 0.189 | 1.439 | 0.061 | 3 | 0 | 6 | 5 | 1.350 | 2 | 28 |
| 7 | QSA | 2.574 | -0.293 | 0.941 | 0.224 | 0.188 | 1.158 | 0.062 | 3 | 2 | 4 | 4 | 1.040 | 3 | 39 |
| 8 | GDN | 2.762 | -0.310 | 0.936 | 0.178 | 0.188 | 1.399 | 0.062 | 2 | 5 | 5 | 3 | 0.980 | 3 | 33 |
| 9 | GDN | 2.543 | -0.322 | 0.923 | 0.239 | 0.188 | 1.131 | 0.063 | 4 | 0 | 4 | 4 | 1.040 | 2 | 28 |
| 10 | GDN | 2.583 | -0.290 | 0.937 | 0.255 | 0.190 | 1.138 | 0.063 | 4 | 0 | 5 | 4 | 1.040 | 2 | 28 |
| 11 | QSA | 2.524 | -0.238 | 0.940 | 0.203 | 0.188 | 1.129 | 0.063 | 3 | 2 | 4 | 4 | 1.030 | 3 | 39 |
| 12 | GDN | 2.762 | -0.338 | 0.931 | 0.237 | 0.187 | 1.344 | 0.063 | 4 | 0 | 5 | 5 | 1.340 | 2 | 28 |
| 13 | GDN | 2.615 | -0.276 | 0.933 | 0.224 | 0.190 | 1.206 | 0.062 | 3 | 1 | 2 | 4 | 1.030 | 3 | 33 |
| 14 | GDN | 2.846 | -0.272 | 0.935 | 0.236 | 0.189 | 1.423 | 0.063 | 4 | 0 | 7 | 5 | 1.350 | 2 | 28 |
| 15 | QSA | 2.799 | -0.248 | 0.937 | 0.209 | 0.186 | 1.404 | 0.062 | 3 | 0 | 2 | 5 | 1.350 | 2 | 34 |
| 16 | GDN | 2.488 | -0.341 | 0.935 | 0.212 | 0.188 | 1.090 | 0.062 | 3 | 4 | 1 | 2 | 0.670 | 3 | 33 |
| 17 | GDN | 2.541 | -0.326 | 0.946 | 0.257 | 0.191 | 1.084 | 0.062 | 4 | 0 | 6 | 3 | 0.990 | 2 | 28 |
| 18 | GDN | 2.630 | -0.312 | 0.921 | 0.237 | 0.189 | 1.220 | 0.062 | 4 | 0 | 3 | 3 | 1.000 | 2 | 28 |
| 19 | QSA | 2.519 | -0.249 | 0.932 | 0.212 | 0.220 | 1.092 | 0.062 | 3 | 2 | 4 | 4 | 1.030 | 3 | 39 |
| 20 | GDN | 2.508 | -0.297 | 0.928 | 0.236 | 0.188 | 1.093 | 0.063 | 4 | 0 | 7 | 3 | 1.000 | 2 | 28 |
| 21 | GDN | 2.479 | -0.325 | 0.938 | 0.208 | 0.186 | 1.085 | 0.062 | 3 | 4 | 2 | 3 | 0.990 | 3 | 33 |
| 22 | GDN | 2.943 | -0.268 | 0.932 | 0.207 | 0.188 | 1.554 | 0.062 | 3 | 0 | 7 | 6 | 1.430 | 2 | 28 |
| 23 | QSA | 2.903 | -0.200 | 0.941 | 0.182 | 0.188 | 1.532 | 0.062 | 2 | 3 | 7 | 6 | 1.430 | 3 | 39 |
| 24 | GDN | 2.509 | -0.315 | 0.922 | 0.205 | 0.202 | 1.117 | 0.063 | 3 | 4 | 5 | 4 | 1.040 | 3 | 33 |
| 25 | GDN | 2.553 | -0.335 | 0.930 | 0.205 | 0.188 | 1.167 | 0.062 | 3 | 0 | 1 | 4 | 1.040 | 2 | 28 |
| 26 | GDN | 2.857 | -0.292 | 0.938 | 0.260 | 0.191 | 1.405 | 0.063 | 4 | 0 | 5 | 6 | 1.420 | 2 | 28 |
| 27 | QSA | 2.518 | -0.287 | 0.939 | 0.206 | 0.187 | 1.123 | 0.063 | 3 | 2 | 6 | 4 | 1.030 | 3 | 39 |
| 28 | GDN | 2.477 | -0.315 | 0.929 | 0.236 | 0.190 | 1.060 | 0.062 | 4 | 0 | 5 | 4 | 1.030 | 2 | 28 |
| 29 | GDN | 2.812 | -0.364 | 0.926 | 0.208 | 0.193 | 1.422 | 0.062 | 3 | 2 | 2 | 5 | 1.340 | 3 | 33 |
| 30 | GDN | 2.437 | -0.325 | 0.930 | 0.242 | 0.207 | 0.996 | 0.062 | 4 | 0 | 7 | 4 | 0.900 | 2 | 28 |
| 31 | QSA | 2.855 | -0.320 | 0.935 | 0.179 | 0.188 | 1.491 | 0.062 | 2 | 1 | 2 | 6 | 1.400 | 3 | 39 |
| 32 | GDN | 2.523 | -0.263 | 0.938 | 0.227 | 0.190 | 1.104 | 0.064 | 3 | 3 | 3 | 4 | 1.040 | 3 | 33 |
| 33 | GDN | 2.834 | -0.274 | 0.931 | 0.212 | 0.189 | 1.440 | 0.062 | 3 | 0 | 2 | 5 | 1.360 | 2 | 28 |
| 34 | GDN | 2.935 | -0.312 | 0.937 | 0.236 | 0.188 | 1.512 | 0.063 | 4 | 0 | 5 | 6 | 1.410 | 2 | 28 |
| 35 | QSA | 2.544 | -0.230 | 0.943 | 0.206 | 0.191 | 1.141 | 0.063 | 3 | 3 | 6 | 4 | 1.050 | 3 | 39 |
| 36 | GDN | 2.588 | -0.333 | 0.932 | 0.288 | 0.192 | 1.113 | 0.063 | 4 | 0 | 1 | 3 | 0.990 | 2 | 28 |
| 37 | GDN | 2.624 | -0.335 | 0.938 | 0.206 | 0.193 | 1.224 | 0.062 | 3 | 2 | 6 | 4 | 1.030 | 3 | 33 |
| 38 | GDN | 2.858 | -0.306 | 0.942 | 0.235 | 0.193 | 1.426 | 0.063 | 4 | 0 | 7 | 5 | 1.370 | 2 | 28 |
| 39 | QSA | 2.802 | -0.263 | 0.947 | 0.190 | 0.194 | 1.408 | 0.062 | 2 | 4 | 4 | 5 | 1.340 | 3 | 39 |
| 40 | GDN | 2.510 | -0.339 | 0.949 | 0.225 | 0.194 | 1.078 | 0.063 | 3 | 3 | 3 | 3 | 0.990 | 3 | 33 |
| 41 | GDN | 2.779 | -0.353 | 0.942 | 0.208 | 0.191 | 1.376 | 0.062 | 3 | 0 | 6 | 5 | 1.350 | 2 | 28 |
| 42 | GDN | 2.592 | -0.303 | 0.938 | 0.235 | 0.194 | 1.163 | 0.062 | 4 | 0 | 3 | 4 | 1.060 | 2 | 28 |
| 43 | QSA | 2.540 | -0.287 | 0.958 | 0.224 | 0.193 | 1.101 | 0.063 | 3 | 1 | 3 | 4 | 1.050 | 3 | 39 |
| 44 | GDN | 2.854 | -0.352 | 0.944 | 0.211 | 0.193 | 1.444 | 0.062 | 3 | 0 | 1 | 5 | 1.360 | 2 | 28 |
| 45 | GDN | 2.600 | -0.375 | 0.939 | 0.213 | 0.192 | 1.193 | 0.062 | 3 | 1 | 5 | 4 | 1.040 | 3 | 33 |
| 46 | GDN | 2.498 | -0.355 | 0.938 | 0.244 | 0.194 | 1.059 | 0.063 | 4 | 0 | 3 | 3 | 0.840 | 2 | 28 |
| 47 | QSA | 2.636 | -0.286 | 0.956 | 0.177 | 0.203 | 1.142 | 0.158 | 2 | 5 | 2 | 4 | 0.890 | 4 | 40 |

All times are milliseconds. The raw captures are retained outside the source repository in the fleet diagnostics workspace; this document is the source-controlled qualified summary.