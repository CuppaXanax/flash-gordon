# Flash Gordon Past-Wins Shootout

## Scope

The accepted LKG was commit `340e7eb`, tag `lkg-10.035tps-cooked-experts`, with
`99.64675 ms / 10.03545 TPS` final-20 wall. This shootout was based on `1864877`
in the isolated `perf/past-wins-shootout` worktree.

All microbenchmarks ran on one 24-CU BC250 GFX1013 blade with Vulkan GPU
timestamps at production dimensions. Fleet results used eight blades, the exact
France response/token oracle, and the final 20 unprofiled frames. The France
trace was never used for placement.

## Decision Summary

| Candidate | Microbenchmark | Parity | Unprofiled fleet | Decision |
|---|---|---|---|---|
| F32 subgroup reduction | Weighted result changed sign between runs; row-2/4 reuse regressed | Numerical PASS | Not run | Reject cheaply: no reproducible production-shape win |
| Algebraic GDN subgroup reductions | Not timed after parity gate | Recurrent state FAIL | Not run | Reject cheaply |
| Cooked-Q8 row-8, unsplit | Predicted `1.941 ms/token` kernel saving | Numerical and exact token PASS | `99.3628 ms / 10.06413 TPS` | Survives; narrow `0.284 ms` frame win |
| Cooked-Q8 per-shape splits | Predicted `2.526 ms/token` kernel saving | Numerical PASS; exact token FAIL | `100.39115 ms / 9.96104 TPS` | Reject |
| Q5_1 Expert Supertile v2 | `118.030 -> 107.336 us`, `9.06%` | Bit-exact micro and token PASS | `100.8844 ms / 9.91234 TPS` | Reject: `+1.522 ms` versus row-8 |

No experiment touched placement, send batching, receive ordering, fence
spinning, exact-N graph recording, or monolithic expert fusion.

## 1. Dense F32

The 168 profiled calls consist of `2560x1` shared scalars (48), `2560x48` GDN
controls (72), and `2560x512` routers (48). The LKG attributes `3.312 ms/token`
to `fg_dense_f32.spv`.

| Shape | Calls | Current GPU | Subgroup row-1 GPU | Current GB/s | Candidate GB/s | Result |
|---|---:|---:|---:|---:|---:|---|
| `2560 -> 1` | 48 | 20.885 us | 20.543 us | 0.981 | 0.997 | Tiny win |
| `2560 -> 48` | 72 | 22.448 us | 22.060 us | 22.361 | 22.754 | Tiny win |
| `2560 -> 512` | 48 | 45.760 us | 46.816 us | 114.842 | 112.251 | Regression |

Row-2 and row-4 reuse regress severely: on `2560 -> 48`, `37.755` and
`70.426 us`; on `2560 -> 512`, `53.520` and `79.424 us`. Numerical parity
passes at the existing `2e-4` relative tolerance.

The call-weighted proxy is `4.815 ms` current versus `4.822 ms` candidate in the
first run. A later full-suite run flips the small difference to a `0.042 ms`
candidate win because router timing moves by about a microsecond. The result is
not reproducible and cannot plausibly own meaningful frame wall, so no fleet
run was made.

## 2. Algebraic GDN

The current algebraic kernel is `5.848 ms/token` across 36 calls
(`162.44 us/call` in the LKG trace). Five fresh current measurements give a
median of `160.938 us/call`.

The natural subgroup-total prototype changes the exact recurrent state at index
49152. A shuffle tree constructed to reproduce the existing
`64 -> 32 -> 16 -> ... -> 1` order moves, but does not eliminate, the first
mismatch (index 65536). This glslang toolchain cannot compile
`GL_EXT_subgroup_size_control`, so an explicit wave32 PSO is unavailable.

The existing full-state oracle remains unchanged and fails. Per policy,
candidate GPU time was not measured after parity failed, and no fleet run was
made.

## 3. Cooked-Q8 Geometry Matrix

The 398 `fg_dense_q8_0_cooked.spv` calls reconcile exactly into nine shapes:

| Shape | Calls | Current row-4 GPU / GB/s | Matrix winner GPU / GB/s | Matrix geometry |
|---|---:|---:|---:|---|
| `320 -> 10240` | 96 | 26.14 us / 134.81 | 23.78 us / 148.19 | row-8, unsplit |
| `2560 -> 10240` | 37 | 109.03 us / 255.93 | 97.68 us / 285.66 | row-8, unsplit |
| `2560 -> 6144` | 36 | 69.57 us / 240.73 | 69.57 us / 240.73 | row-4, unsplit |
| `6144 -> 2560` | 48 | 113.73 us / 147.25 | 96.96 us / 172.72 | row-8, split-2 |
| `2560 -> 640` | 96 | 18.45 us / 95.06 | 11.67 us / 150.31 | row-8, split-4 |
| `640 -> 2560` | 48 | 12.05 us / 145.57 | 10.45 us / 167.86 | row-8, unsplit |
| `2560 -> 2560` | 1 | 35.77 us / 195.22 | 31.46 us / 222.01 | row-8, unsplit |
| `2560 -> 12288` | 12 | 133.77 us / 250.29 | 120.87 us / 277.01 | row-8, unsplit |
| `2560 -> 512` | 24 | 17.94 us / 78.33 | 10.11 us / 138.96 | row-8, split-4 |

All 77 dimensionally sensible row `{2,4,8}` and split `{1,2,4,8}`
configurations pass numerical parity. The current weighted proxy is
`18.928 ms/token`, close to the retained `18.863 ms` profile. Matrix winners
predict `16.402 ms`, a `2.526 ms` kernel saving.

The split table adds 168 reduction dispatches per token. Its real run diverges
from the exact token oracle at decode token 22 and regresses to `100.39115 ms`
(`+0.7444 ms` versus the LKG), so all new split selections are rejected.

The no-new-dispatch table retains row-4 only for `2560 -> 6144` and selects
row-8 unsplit for the other eight shapes. It predicts a `1.941 ms` kernel saving
and passes the full eight-blade oracle:

- Exact response, token IDs, and logits match the LKG.
- Final-20 layers: `94.9939 ms`, `-0.33515 ms` versus LKG.
- Final-20 frame: `99.3628 ms`, `10.06413 TPS`, `-0.28395 ms` versus LKG.
- 17/20 frames are at or below 100 ms, the same count as the LKG.

This survives the shootout, but the exposed wall gain is only 14.6% of the
repeated-hot-matrix kernel estimate. It is a useful sealed PSO selection, not a
path to 20 TPS. The surviving row-8 selection was later committed to `main`.

## 4. Expert Supertile v2

The prototype removes the Q5_1 cooked-down shader's per-block workgroup
barriers. Eight-lane cohorts load scale/high/quant metadata directly and
distribute it with subgroup shuffles; the final cohort shuffle reduction
preserves the current addition order.

For six selected experts at production `640 -> 2560` dimensions:

| Metric | Current | Supertile v2 | Delta |
|---|---:|---:|---:|
| GPU time | 118.030 us | 107.336 us | -10.694 us (-9.06%) |
| Weight-stream bandwidth | 62.47 GB/s | 68.69 GB/s | +9.96% |
| Batched wall proxy | 125.78 us | 115.34 us | about -10.4 us |
| Maximum output difference | 0 | 0 | Bit-exact |

The profiled critical-worker Q5_1 scope owns `3.904 ms/token`; the micro result
therefore suggests only `0.354 ms` of critical-worker GPU saving, or at most
about `0.460 ms` across the 43 critical Q5_1 calls.

Fleet truth rejects it. Combined with the surviving row-8 table, exact token
parity still passes, but final-20 wall is `100.8844 ms / 9.91234 TPS`:
`+1.5216 ms` versus row-8 and `+1.23765 ms` versus the accepted LKG. No Q5 v2
production selection is retained.

## Fixed Expert-Call Intercept

Rank 1 was instrumented conditionally on a production-equivalent decode. Each
pre-recorded five-dispatch graph received GPU begin/end timestamps; host clocks
bracketed fence reset, `vkQueueSubmit`, fence wait, and query read. Existing
token-26/27 worker records provide the enclosing `fg_expert_decode` interval.
Normal graphs are unchanged when instrumentation is disabled.

The 48 paired real requests have this median decomposition:

| Component | Median |
|---|---:|
| Five-dispatch GPU execution | 223.140 us |
| Fence reset | 5.176 us |
| `vkQueueSubmit` call | 50.932 us |
| Fence wait beyond GPU span | 35.261 us |
| Diagnostic query read | 0.629 us |
| Graph host-to-host | 315.038 us |
| Graph non-GPU total | 92.141 us |
| Outer staging/schedule/result-copy | 118.887 us |
| Entire worker expert call | 433.336 us |

| Selected | Requests | GPU | Graph host | Graph non-GPU | Outer | Worker call |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 122.84 us | 214.87 us | 95.95 us | 121.02 us | about 336 us |
| 2 | 17 | 171.32 us | 263.54 us | 92.08 us | 118.77 us | about 382 us |
| 3 | 15 | 223.52 us | 315.53 us | 92.28 us | 118.82 us | about 434 us |
| 4 | 9 | 276.68 us | 368.86 us | 92.06 us | 119.53 us | about 488 us |
| 5 | 4 | 327.80 us | 422.78 us | 94.68 us | 119.88 us | about 543 us |

The non-GPU graph intercept is effectively constant with selected count. A
persistent doorbell that removes only reset, submit, and fence-wakeup residual
has a theoretical exposed ceiling of:

`(5.176 + 50.932 + 35.261) us x 48 layers = 4.39 ms/token`.

That is multi-millisecond, but not a 10 ms architecture. A full GPU work ring
becomes a 10+ ms candidate only if it also absorbs request staging, schedule
publication, and reduced-result copy:

`(91.369 + 118.887) us x 48 layers = 10.09 ms/token`.

This is an upper bound: shared-expert overlap, network arrival, and noncritical
workers can hide part of it. The design gate for the larger redesign is
therefore not merely a persistent submit thread. It is a resident request/result
ring with GPU-visible activation, route schedule, completion sequence, and
reduced output slots, eliminating the per-request fence and the outer host
staging/copy path together.

Even the full upper bound leaves the strategic 50 ms target roughly 39 ms away.
Additional architecture beyond the worker ring remains necessary for 20 TPS.

## Archived Evidence

The original raw geometry and worker timing captures were retained outside the
source repository. This file preserves the decisions needed to avoid repeating
rejected experiments.
