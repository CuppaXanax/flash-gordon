# MTP serving foundation on the promoted multisession base

## Linked orchestration update — 2026-10-16

The branch is now rebased onto promoted main `7381f93`. Default-off native
head fanout, causal small-N ring verification, authoritative native KV repair,
all-owner state transactions and independent per-stream adaptive policies are
linked. Explicit overlay and exact production-Q8 embedding sidecar paths are
required. This supersedes the historical “unlinked” status below, **not** its
qualification limits: no blade-pair end-to-end serving or serving speedup is
claimed, and no production lease has been requested.

Latest .51 gates, stock clocks/governor, `tools/mtp-serving-validate.sh`:

* C17 `-Werror` release/shader build; flag registry; existing session, API and
  two-session eight-rank coroutine-pipeline regressions: PASS.
* Greedy transactional cycle, rejection-sampling primitive, typed wire,
  malformed native arguments, causal prefix warmfill, authoritative rejection
  repair and independent native-session fixtures: PASS. Native manager
  state-machine fixtures use mocked compute; do not mistake these for serving.
* Exact raw/cooked sidecar export, independently reconstructed sealed source
  SHA256, corrupt-payload rejection even after replacing its raw digest: PASS.
  All 16 production-width (2560) Q8 rows and all four HC copies match the actual
  target batched GPU embedding kernel bit-for-bit.
* Current baseline N=2/3/4 six-layer output and GDN state: byte-identical to
  sequential N=1; causal prefix, retained 1/2/3/4-input rollback, session shape
  changes and the other session's retained replay: PASS.
* Real retained Q8 native overlay SHA256
  `536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b`:
  two interleaved sessions, 18 rows, eight rollback cycles, original-parent
  close first and KV-only warmfill equivalence: byte-exact, both full KV and
  window 4. These are short-prefix component tests, not 8192-window capacity
  or acceptance qualification.
* Python: 25 fixtures run, four optional numerical tests skipped.

Distributed draft latency, current-kernel greedy serving identity, EOS/output
limits/cancellation/callback failures under actual burst publication, true
blade-pair serving and long-context sessions-2 residency remain execution gates.
Historical acceptance and N ratios below are not new linked-serving evidence.

Branch: `cuppaxanax/mtp-serving`, based directly on `2d0f8a7`. The published
`cuppaxanax/mtp` capture history remains unchanged.

**Not ring-ready. No serving speedup is claimed.** The native serving/head
fanout and causal ring orchestration remain unlinked; `--experimental-mtp`
still fails closed. All work here used the idle .51 blade, not production
ranks or the production API.

## Session isolation and native preparation

* Immutable native Q8 weights are shared once. A second short-context session
  adds **5,200,432 GPU bytes**, not another 2,772,584,448 bytes of weights.
* Both sessions have private KV/frontiers, executors, scratch and static
  recordings. Closing the original session first leaves its sibling usable;
  closing both returns requested GPU storage to the initial count.
* Slots 0..3/4..7 are target-owner partitions. Native core slots are 8/10,
  prefix warm slots 9/11. Shape changes, N=1 transitions and failures no
  longer invalidate the other session's target recordings.
* Extending the static pool uncovered a layout-array overread during Vulkan
  initialization. Its allocation/initialization now covers the larger of the
  dynamic and static descriptor counts; all twelve slots pass GPU tests.
* Native prefix/repair now replays only input mix and K/V. The short-prefix
  Q8 means were **268.31 us** with full KV and **295.69 us** with bounded KV
  (15 timed cells each). This is **not full-core or full-draft latency**.
* Optional bounded draft attention retains W+4 physical cells for a W-cell
  window, preserving absolute RoPE. The extra cells protect rollback of k<=3
  proposals without copying a whole KV cache. Unsupported rewinds fail.
  At W=8192 the KV allocation is **32.016 MiB/session**, rather than 1 GiB
  for full 262K dense native KV. This remains a draft-only approximation
  requiring long-context acceptance qualification.
* A per-stream conditional-acceptance/cost EMA chooses k=1..3 by expected
  committed tokens/ms and periodically explores other lengths. Unit tests
  cover chat/code priors, cost observations, rejection censoring, invalid
  observations, output limits and independence. It is not yet attached to
  runtime publication.
* QSA service guards track their actual session namespace; restore drains
  that namespace's writes before truncating and repairing its frontier.

## Expert-union experiment

The retained N=3/4 variant loads/dequantizes one weight tile and accumulates
all four possible token pairs in a single 64-lane workgroup. The old variant
used two pair columns in a 128-lane group. N=1 and N=2 implementations are
unchanged. No roofline N=1 geometry or repack layout was forked.

Replay-faithful six-layer synthetic Q4/Q5 block, .51, seed 1, approximately
256 warmup input tokens and 192 measured input tokens per shape. A/B/A
swaps only the expert shader, restoring the selected shader on exit.
Times include normal replay but no profiling timestamp instrumentation:

| Shape | Old A ms | New B ms | Old A2 ms | B / bracketed old | B / B-N1 |
|---|---:|---:|---:|---:|---:|
| N=1 | 2.6651 | 2.6567 | 2.6785 | 0.9943 | 1.000 |
| N=2 | 4.0113 | 4.0658 | 4.0510 | 1.0086 | 1.530 |
| N=3 | 5.7891 | 5.6142 | 5.8713 | 0.9630 | 2.113 |
| N=4 | 6.9682 | 6.3412 | 7.0746 | 0.9031 | 2.387 |

Thus the bracketed local improvements are **3.7% at N=3** and **9.7% at N=4**.
These are one seed/one blade and three contiguous windows per phase, not a
fleet confidence interval. The requested **1.2/1.4/1.6** service ratios remain
unmet. Dense small-N work, recurrence and attention also remain material;
expert union is not the only remaining cost. Intrusive profiles are retained
only for attribution, not mixed into these replay ratios.

## Gates and receipts

Passed: `-Werror` release build; flag registry; adaptive MTP host tests;
existing multisession QSA, eight-rank loopback fabric, API and session tests.
The process-local pinned-memory limit was raised so the two QSA namespaces
actually execute instead of taking their resource-limit skip.

GPU gates passed:

* N=2/3/4 output and GDN state byte identity against sequential N=1.
* Causal intra-batch masking and every retained-prefix rollback length 1..4.
* Two target sessions with different N, replay reuse and N=1 transitions.
* Two real-weight native sessions, 18 interleaved rows and eight KV-only
  repair cycles per configuration, byte-identical to their serial runs.
* Bounded-attention scalar oracle: 36 cells, nonzero absolute start 100,
  three wraps and a rejected three-cell suffix; maximum absolute error
  **5.04e-8**.
* All output and state hashes match across all A/B/A performance phases.

Raw receipts: `mtp-serving-evidence.tar.gz` (26,060 bytes), SHA-256
`695286de7facf5b51f210c9a627f58ed29895ed84f6fbdb30016c2938b460943`.
Machine-readable results: `mtp-serving.json`.
The native Q8 overlay SHA matches the qualified capture source,
`536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b`.

Remaining serving gates: native embedding/head fanout and authoritative
warmfill/repair hooks, causal small-N ring transactions/publication,
live per-session policy integration, long-context acceptance and linked-path
greedy equivalence in a separately granted lease after stack-1.
The earlier measured acceptance and causal-N1 full-ring qualification remain
valid evidence for that reference path, not for this still-unlinked path.
