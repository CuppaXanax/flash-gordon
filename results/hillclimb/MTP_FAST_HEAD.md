# Native MTP draft fast path

Branch `cuppaxanax/mtp`; only .51 used. **Not a serving rollout, distributed-head wall time,
real-text acceptance measurement or full-target greedy-equivalence result.**

## Measured component latency

Three fresh runs per shard; four disjoint 62080-row quarters, all measured separately
on the same 40-CU .51 GPU. Each run has 16 warmups and 32 timed cells, positions 16..47.
Actual weights/token-198 embedding, changing synthetic final HC. No reduced vocabulary.

| Native pack | Core ms | One quarter + GPU argmax ms | Local sum ms | Window range ms |
|---|---:|---:|---:|---:|
| q8 | 0.939 | 0.627 | 1.566 | 1.544–1.581 |
| expert-q5 | 0.868 | 0.620 | 1.488 | 1.475–1.504 |

The local sum is measured core plus **one** quarter in the same process. Four physical
head ranks, fanout/global winner, target-owner embedding lookup/delivery and target
co-residency are still unmeasured/unlinked. Do not call this distributed draft latency.
The Q5 result is a precision **candidate**, not a qualified deployed/default pack.
**The <=1.5-ms target is only reached by the short-prefix candidate mean**, not every
window, not at longer prefixes and not a measured distributed/embedding-inclusive draft.

Historical monolithic execution was 3.062 + 5.724 = 8.786 ms/draft. The contemporary
legacy core is 2.356 ms. Stock clocks
were not changed/locked; raw before/during/after sysfs observations are archived.
Steady replay makes one core submission, zero newly recorded dispatches; diagnostic
profiling is separate and may count extra submissions.

## Context-cost check

One additional 32-cell window per prefix, Q5 candidate, one full-vocabulary quarter.
The synthetic warmup fills genuine native KV; it is not a real prompt/acceptance sample.

| Warmup cells | Core ms | Quarter ms | Local sum ms |
|---:|---:|---:|---:|
| 512 | 0.892 | 0.623 | 1.515 |
| 2048 | 1.043 | 0.631 | 1.674 |
| 8192 | 1.570 | 0.632 | 2.202 |

The old per-head serial attention grew to 1.027 ms core in the 576-cell clock sample.
Adaptive up-to-eight-way causal split attention shortens the serial recurrence;
64 fixed splits were rejected for excessive empty-group/merge overhead. Context costs
above, not just the short-prefix target result, must be included in integration planning.
Long-prefix GQA KV reuse/attention parallelism remains a performance gate.

## Correctness and precision

* GPU router/top-10, experts, shared combine, indirect positions and final HC mixer replay
  together. HC/embedding copies and graph share a submission/fence; deferred owner
  producers are drained before staging (covered by an explicit queued dependency test).
* Exact-order four-stream FC and fixed-shape 10240 RMS norm are component-byte-identical.
* Final contraction reuses production split projection/scaled-SiLU; vocabulary R8 GEMV
  emits GPU partial argmax with global ties, tail/nonfinite and multi-stage scalar oracles.
* q8: 48-cell slow/replay max-scaled 3.28e-06; same-pack independent NumPy/GPU vocab KL 2.95e-12, same first-cell argmax. Last-cell truncate/rebuild is byte-exact.
* expert-q5: 48-cell slow/replay max-scaled 2.18e-06; same-pack independent NumPy/GPU vocab KL 4.25e-12, same first-cell argmax. Last-cell truncate/rebuild is byte-exact.
* Routed Q5_K gate/up, Q5_1 down with common Q8: cross-pack KL **0.00495952**,
  same top-1, barely passing the unchanged 0.005 gate on **one synthetic fixture**.
  **Real chat/code KL and acceptance remain unqualified.** Cooked weights 1,881,294,848 B.
* Common/expert Q4/IQ4 candidate: KL 0.283014/top-1 changed, rejected. Experts-only
  Q4_K/Q5_1: KL 0.00539410, rejected. No precision below Q4 was used.
* After the shared copy-helper change, the six-layer synthetic Q4 owner regression
  passes N=2/3/4 bit-exact outputs/GDN versus N=1, causal-prefix masking, expert
  unions and rejection rollback for kept prefixes 1..4 with changed mRoPE suffix.
  Its one-window smoke timing is not a performance measurement.
* Werror build, MTP/sampler/depth-B host suites, flag registry, native tests and targeted
  Q4/Q5/Q8/static replay Vulkan selectors pass. Full-target same-kernel MTP-on/off
  greedy equality is still the mandatory end-to-end gate, not satisfied by head parity.

## Projection and remaining integration

Strata E=2.4–3.2 committed tokens/pass remains an external estimate, not measured FG
acceptance. Last six-layer verifier ratios remain 1 / 1.519 / 2.199 / 2.505; this work
does not remeasure/fix that remaining small-N regression.
With a **future qualified** 40–45 TPS baseline, target N=4 ratio 1.8, k=3 and the
measured short-prefix candidate local sum, the conditional range is **48.5–72.0 TPS before added fanout/commit**.
Neither that ratio nor zero added overhead is qualified. This is not current ring TPS
or a ceiling on MTP. Use prompt-specific context cost and measured acceptance next.

Still needed: final-owner embedding/head-shard hookup and residency, real four-rank
head wall time, authoritative prompt+continuation HC capture, candidate precision/
acceptance qualification, causal all-rank GDN/PLE/QSA transactions and full greedy gate.
Only .51 was used; .52/.53 and all leased blades were untouched.

## Evidence

All 438 remote source hashes match the uploaded archive; local match is exact for 354 and EOL-only for 84.
JSON contains every timing window, oracles, candidate rejection records and hashes.
Raw profiles, source snapshots, compiler/test logs, first-cell dumps, library/input
hashes and read-only clocks are in `mtp-fast-head-evidence.tar.gz`:
`cf47a572b78b1b8ce6024bb99a91b5a65927b88209bd9a0026e9c36b9e5e552b`.
See `docs/MTP.md` section 7 and `tools/mtp-fast-*.sh` for reproduction and contracts.

## Cleanup

The .51 `/home/xander/hc-mtp` directory was removed after evidence download;
absence was checked twice inside the cleanup invocation (exit 0). No experiment
process, hardware-setting change or other-blade work remains. The JSON preserves
the actual cleanup stdout. Do not run the fleet wrapper again merely to check absence: it recreates the directory.
