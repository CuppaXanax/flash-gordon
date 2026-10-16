# MTP serving foundation on the promoted multisession base

## Frozen single-session lease — 2026-10-16: FAILED_GREEDY

Frozen311be40 / releasec9d0a7ad, sessions1/mask330, stock recipe. Exclusive
grant15:22:47.332Z; first strict attempt stopped/restored at adaptive6/8.
Diagnostic retry retained the original lease/deadlines and immutable binary.
Final all-eight pinned restore, API sanity, watchdog cancellation and owned
scratch removal passed; explicit release15:56:43Z. No promotion.

| Width | Greedy | Off TPS chat/code | MTP TPS chat/code | Tokens/round chat/code |
|---|---:|---:|---:|---:|
| Adaptive cap3 | 6/8 | 31.30/31.60 | 16.11/26.88 | 2.20/3.36 |
| k1 | 6/8 | 29.32/30.17 | 22.35/30.10 | 1.63/1.91 |
| k2 | 5/8 | 29.12/29.78 | 18.96/27.65 | 2.04/2.70 |
| k3 | 6/8 | 29.21/29.73 | 17.19/28.12 | 2.28/3.39 |

All four code cases passed each arm; failures are chat. All same-session off
arms passed8/8. Cause is unproven. There is no demonstrated speedup and no60TPS
projection. Each off number matches request/repeat before/after; TPS includes
repair. Tokens/round counts speculative verification rounds only; including
repair passes reduces yields to1.18–1.31 chat and1.75–2.54 code.

Adaptive has six matched requests/category; fixed arms two each. Later suites
were explicitly reduced from three repeats to one without changing the128-token
maximum. F1 had already loaded the original three-repeat policy; its extra four
requests/category have no A2 repeat match and are excluded from this table, but
retained in raw/full-arm metrics. Results are diagnostic, not qualification.
Adaptive/fixed3 SSE/stop/cancel completed. Complete measurements and restoration
receipts: `mtp-single-session-perf.json` and `mtp-single-session-perf-evidence.tar.gz`.

## Historical preparation: isolated single-session performance

Prepared release `c9d0a7ad3170a088306b0e097c0a8a5088f53c14a861fd1b78b7be939a39a1c4`
uses sessions1/mask330. Nine bracketed arms compare adaptive cap3 and true
fixed k1/2/3 against serial MTP-off chat/code baselines. Fixed widths bypass
adaptive selection and clamp only at generation boundaries. Eight actual
greedy token-ID/public-completion cases gate each arm; SSE/stop/cancel run on
adaptive3/fixed3. No concurrent-session result is claimed.

.51 build, four numerical-double eight-rank rejection/repair runs, synthetic
Q8 wide0/1 and independent expert component gates pass. All320 shader binaries
are byte-identical to the actual-pack-qualified corrected release. Source and
artifact identities explicitly bind inherited real-format proof to this
performance-only build; no real-ring performance had been measured at preparation.
The eager dual-session admission/residency work is deliberately excluded.
The controller still requires a fresh exclusive lease, restores sessions2/wide1,
and never promotes. See `results/hillclimb/mtp-single-session-ready.json`.

## Corrected-Q8 trained-pack lease — 2026-10-16: BLOCKED ON RESIDENCY

Exclusive .42–.49 grant:14:02:38.992Z. Candidate `706d476`, sealed release
`54be56c2f1188a2847923f4d1ab9c81f39ea8bb95c4bbbe8653fc2eab107d6f5`.
Pinned production was restored and verified on all eight owners, scratch
removed and the ring released at **14:12:25.123Z**, under10min /45min cap.
No promotion, full MTP greedy pass, bracketed speedup or serving-TPS projection.

Completed MTP-off arms:

| Arm | Chat median decode TPS | Code median decode TPS | Sessions2 aggregate wall TPS |
|---|---:|---:|---:|
| A0: off, mask0 | 30.37 | 29.82 | 35.47 |
| B0: off, mask330 | 30.51 | 30.69 | 37.32 |

Both arms passed all eight actual token-ID SHA256/count and public-completion
comparisons. Each decode category has six requests; sessions2 has three
concurrent chat/code pairs. These are **unbracketed baselines**, not a mask
speedup or MTP-on performance result.

The first K1/mask0 request completed55 chat tokens, SHA256
`0995aa1aa72470655d2badba09736683f8447af26f9e7a8c02b1feaf5a2b8839`,
exactly matching A0's actual token IDs. Its33 native rounds accepted23/33
drafts and emitted55 tokens; the final EOS round emitted one token rather
than two. These are **one cold chat's observations only**, not generalized
acceptance or an eight-case greedy gate. The first native round took1439ms;
do not compare this cold request with warm performance medians.

The **second greedy HTTP request** disconnected. The primary rank7/.49 error
is `native MTP session KV/scratch exceed residency cap`, from the private
native-session clone check in `src/mtp_head.c:468-473`. Rank7's last startup
Vulkan ledger reports14,687,617,056 live-requested bytes; the exact manifest
cap is14,495,514,624 bytes (13.5GiB), already192,102,432 bytes lower.
The precise live value at the failing check was not logged. This is a
**software residency-policy rejection**, not a reported physical Vulkan OOM.
The coordinator then reported non-reusable distributed transport; ranks1–6
reported closed sockets. The actual cooked-Q8 down path no longer rejects
the trained tensor; format compatibility is not the blocker.

No K1 performance suite or S1/K2/S2/K3/S3 arm ran. MTP sessions2,
SSE/stop/cancel, bracketed ratios and full MTP greedy remain unqualified.
The earlier .51 native-session fixture ran in isolation from the full target
owner; it did not prove this full owner/native/session/deferred residency mix.

Restoration:31508ee /20261016-abq-stack-adaptive, sealed stack-1/mix-04,
sessions2/wide1. All eight binary/manifest/shader/argv/full-FG-environment
checks, API health/models and before/after greedy sanity passed. Watchdogs
were canceled/exited, owned candidates stopped, scratch absence checked on
all eight owners. All eight downloaded archives independently match their
remote SHA256 receipts. Finalization errors: **zero**. No clocks, governors,
current launchers, weights or production configuration were changed.

**Next gate:** fit owner/native weights plus both private sessions and all
deferred/lazy allocations within the actual residency policy, reserve/eagerly
provision them before declaring readiness, and fail admission without
poisoning transport. Requalify this full memory mix on a spare blade before
requesting another ring lease. Do not bypass the cap or rerun this candidate
unchanged. Machine-readable raw metrics, first-request rounds, all K1 owner
logs, exact error, restoration/cleanup and archive hashes:
`mtp-serving-q8-lease.json`. No further production access after release.

## Corrected real-format candidate — 2026-10-16: COMPONENTS PASS (pre-lease)

The cooked-Q8 compatibility fix `3c84f62` is qualified on idle **.51**, not
on the production ring. Public pinned-source GGUF ranges were cooked by the
existing C routine; the **entire891,289,600-byte /512-expert real layer2 slab**
matches the retained mix-04 manifest tensor SHA256 `065b86394f68...`.

* Both wide0/1 × masks0/330 × N2/3/4 × three input patterns: **36 byte-exact
  native-N1 Q8-down comparisons**; independent CPU basis error `<3.2e-10`.
* Exhaustive **1,225 target records /51 family-format-layout groups**, plus
  **31 native tensors /29 runtime aliases**, all accounted for. Three native
  learned-indexer tensors are explicitly unused by full/bounded draft attention;
  target verification still runs its learned indexer. See every family and
  dispatch in [MTP_FORMATS.md](../../docs/MTP_FORMATS.md).
* Full production flag recipe, masks0/330:150 dense comparisons across all18
  actual matrix shapes;72 independent expert CPU comparisons;30 PLE/GDN/QSA/
  HC/ngram operator oracles; mixed six-layer causal/session/rollback tests;
  four real native-overlay session/window/repair cases. All pass.
* Explicit `1e-4 * max(1,abs(ref))` component tolerance, not blanket byte
  identity: dense maximum absolute2.03e-6; mixed output9.54e-7, state9.54e-6.
  Causal prefixes, replay isolation and retained-prefix rollback remain exact.
  The initial strict dense-byte gate correctly stopped on9.54e-7; its archive
  and diagnostic are retained alongside the successful explicit-tolerance rerun.
* All318 previous SPIRV files are byte-identical; two new exact-Q8 down variants.
  No unpublished research, precision reduction or production promotion.
* Cooker corruption/overwrite protection and exact Q8 embedding sidecar pass.
  C17-Werror/host MTP/session build and flag registry pass. Local tests: six
  restoration-finalizer cases, eight sealed-audit preflight cases and seven
  Python parser/client cases. .51 reported24CUs/subgroup64; no hardware change.
* Governor checksum unchanged, no owned processes, scratch absence verified
  twice. No production access, service/current-launcher or governor changes.

Fresh sealed artifacts are `.build\mtp-serving-q8-lease`:

| Artifact | SHA256 |
|---|---|
| release.tar.gz | `54be56c2f1188a2847923f4d1ab9c81f39ea8bb95c4bbbe8653fc2eab107d6f5` |
| flash-gordon | `03d9c1a0dcc3637a47824db09a5d83586e88159ea1401e4293480207295c49ce` |
| shader-tree | `6901693b938d2a81cbba183026b9bfae040842467a3aef085701d3dbe6dca281` |
| raw .51 evidence archive | `8d97a641cf5990e1d7a5ee65f0df61c336270aa28ba16e1e5d3d115d3de07f02` |

`prepared.json` binds this release to `format-audit.json`; missing/tampered
seals, unsupported formats, unpassed real-weight gates and wrong release,
manifest or native-overlay identities are rejected **before production access**.
Complete inventory, provenance, source/compiled hashes, raw GPU receipts,
negative tests and cleanup receipt: `mtp-format-audit.json`.
The previous `419e303` release is retained as history and must not be reused.

**Historical prepared plan, subsequently executed above.** After an explicit exclusive
.42–.49 grant, use `Invoke-MTPServingQualification.ps1` with
`-Artifacts .build\mtp-serving-q8-lease` and a fresh lease ID. Planned arms:
A0/B0/K1/S1/A1/K2/S2/A2/K3/S3/A3. Both masks must pass all eight same-build
greedy token-ID/public-completion checks; measure bracketed chat/code and
two-session throughput, then SSE/stop/cancel. Stop work at35min, watchdog
workers/coordinator at38/39min,45min hard cap. Restore and verify the pinned
31508ee / stack-1 / mix-04 sessions2/wide1 deployment, clean and release.
**No end-to-end acceptance/TPS, trained-model greedy, long-context or promotion
claim is made by this component audit.**

## Granted trained-pack ring lease — 2026-10-16: BLOCKED

Candidate `419e303` contains `822be55`. The exclusive .42–.49 lease started
at13:13:24.678Z; all owners were restored, verified and cleaned, and the ring
released at13:24:39.938Z (**11m15s**, below the45-minute hard cap).
**No promotion, acceptance claim, MTP speedup or projected serving TPS.**

Completed MTP-off phases A0/mask0 and B0/mask330:

| Arm | Chat median decode TPS | Code median decode TPS | Sessions2 median aggregate wall TPS |
|---|---:|---:|---:|
| off, mask0 | 30.31 | 30.60 | 36.81 |
| off, mask330 | 30.19 | 30.25 | 36.55 |

Each decode category comprises six requests (two prompts, three repeats);
coexistence comprises three concurrent chat/code pairs. These two phases
are **unbracketed baselines**, not evidence of a mask improvement. All eight
greedy completions, actual-token-ID SHA256/counts and finish reasons matched
across masks. Mask330's small-N arithmetic is not exercised by this off gate.

The first K1 (MTP enabled, max-draft1, mask0) request returned HTTP500.
Rank1's primary diagnostic was:
`small-N expert path requires owned cooked K/Q5_1 tensors`.
Other owners then reported closed sockets; the coordinator marked the
transport non-reusable. No MTP completion, round or timing was produced.
Consequently k1/k2/k3, both MTP-on greedy gates, MTP sessions2 and trained-pack
SSE/stop/cancellation checks remain **unqualified**, not failed numerical
equivalence or zero acceptance.

The retained sealed manifest explains the format gap: rank1 owns all512
experts in layers0..5; layer2 has cooked Q5_K gate/up (type13, layout2),
but a cooked Q8_0 down projection (type8, layout5,640x2560x512,
891,289,600bytes). Layer4 also has a cooked Q8_0 down tensor. The kernel
requires down type7/cooked-Q5_1. Layer2 is the first unsupported layer
in owner order, **inferred from the manifest**, not a runtime layer trace.
Ownership is valid; the combined diagnostic does not isolate the failing
predicate. Supporting or safely falling back for real Q8 down projections
is required before another serving lease.

Restoration used the retained current launcher, runtime31508ee /
20261016-abq-stack-adaptive, mix-04, sessions2/wide1. All eight binary,
manifest, shader, argv and complete FG environment checks passed; health,
models and a real greedy response matched pre-lease. Candidate PIDs were
gone, watchdogs canceled/exited, and `/home/xander/hc-mtp` absent on every
owner. No hardware/governor/current-launcher changes.

A historical `capture/cases.tsv` assumption initially prevented evidence
archiving and automatic scratch removal **after successful restoration**.
The serving archive now treats that file as optional; all eight archives
were recovered, downloaded and SHA256-verified before explicit cleanup.
The controller now continues across evidence/download failures and always
attempts cleanup after verified restoration, preserving separate errors.
Six local finalization failure-injection cases and three Python client
tests pass. Owned candidate core dumps are disabled. No model binary changed.

Machine-readable raw phase metrics, token receipts, primary diagnostics,
restoration/release receipts and all eight evidence hashes:
`mtp-serving-lease.json`. Full owner archives remain in the local sealed
lease evidence directory. The earlier numerical-double pair gates below
remain valid but do not supersede this real-pack blocker.

## Two-host publication gate — 2026-10-16

Integrated kernel base `822be55` plus serving/API tests passed on **.51/.52**.
Eight logical ranks execute the real runtime/fabric/codecs/native manager,
head fanout and owner transactions; **numerical compute is doubled**.
This is not trained-model two-blade serving, measured acceptance or a TPS claim.
The compile-time final-N1 seam uses the causal verifier; normal serial output
transport remains a full-ring lease gate.

* k1..3, rejection prefixes0/1/2 and all-accept: PASS on both hosts.
* Actual generation loop: limits1..8, EOS at distances0..4, cancellation drain,
  retained-prefix repair and subsequent generation: PASS.
* Actual API SSE sink: ordered multi-token commit callbacks, split-stop
  suppression, correct finish event and `[DONE]`: PASS.
* Two simultaneously active coroutine jobs, **distinct token histories**,
  private native/policy state and separate SSE sinks: PASS.
* API stop parsing/malformed input, overlapping and UTF-8 stops, incomplete
  prefix flushing, JSON/SSE parity and matched-stop prefix invalidation: PASS.
* Both blades' actual GPU default0 component and N2/3/4 full-block oracles:
  output/GDN state byte-exact against sequentialN1; causal prefixes, every
  retained-input rollback length1..4 and other-session replay isolation: PASS.
* Integrated C17 `-Werror` release/shader build, MTP/session/API/eight-rank
  host regressions and flag registry: PASS. Python28 tests,4 optional skips.
* New serving controller: syntax/no-lease denial and full seven-phase local
  HTTP/fleet-double run pass; an injected K2 startup failure still restores,
  checks API/identity and cleans up before propagating failure. These controller
  doubles did not contact production.

The tested release archive SHA256 is
`ee8a095839fad26614b0ccd77cf6a7bcc2be461fe4a94d647cd6bc9a740aa0b3`;
binary `442ce6e629d66e3984085920dd969f4998a0c26d0db5c0fd5cd170ea24e5d345`.
Source and both evidence hashes are in `mtp-serving-pair.json`. Evidence and
the sealed, independently file-verified candidate are retained under `.build`.
All owned fixture PIDs, .51/.52 scratch and peer-restricted auto-expiring
runtime firewall rules were independently checked absent afterward.
No services, governors, clocks, current production launchers or production
owners were changed.

**Ready to request, not assume, an exclusive ≤45-minute ring lease.**
`Invoke-MTPServingQualification.ps1 -Action Plan` describes the35-minute work
budget, same-build token-ID hash/greedy gate, bracketed chat k<=1/2 and code
k<=3 performance, sessions2 and streaming/stop/cancel checks. Independent
watchdogs at38/39 restore current31508ee stack-1/mix-04 sessions2/wide1;
finish with all-rank identity, API and real greedy checks by minute45.
No small-kernel research mask or production promotion is selected.

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
