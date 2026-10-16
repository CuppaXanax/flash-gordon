# MTP: native head, causal verification and remaining execution gates

Offline follow-up adds a true serial goodput competitor with native-state-safe
fallback, size-guarded Q8 column reuse, authenticated real Strata ID sets and
local-head/deferred-graph residency probes. **No new serving TPS or promotion
claim:** chat noninferiority, whole-rank N4/N1≤1.5 and numerical qualification
remain gates. See the latest section of [MTP_TO_PROD.md](MTP_TO_PROD.md).

Latest: [production roadmap and authorized diagnostic](MTP_TO_PROD.md).
The subsequent user-granted **a038dec A0/F3/A1** diagnostic completed:
paired off/on decode TPS **chat31.063/22.378, code31.200/33.409**,
mean round stages101.401/101.280 ms and observed tokens/verify2.220/3.387.
The cold coordinator-retain fix is confirmed on the authentic ring;
775 completed stage ledgers align across all eight ranks and2,167 actual
MTP choices equal their target argmax. SSE/stop/cancel passed.
All eight pinned sessions2/wide1 services were restored/verified/cleaned,
with independent final release verification **21:15:32Z**. No promotion;
chat regresses and complete same-prefix/N4-Q8 qualification remains pending.
See [paired results and measured ledger](../results/hillclimb/MTP_RETAIN_SERVING.md).

The earlier 2026-10-16 lease measured actual N4 application frames, but its local-head
retry failed before a completed MTP round; no new on/off TPS claim is made.
All eight original sessions2/wide1 services were restored/verified and scratch
removed; final release verification20:23:07Z. User policy now permits
**below-Q4 draft layer/head weights only**; the target remains≥Q4.
The roadmap includes low-bit draft residency and rank7 future-window overlap;
neither is implemented or qualified.

## Revised numerical gate and local-chain candidate

**Policy update: N1/N4 arithmetic alignment is not required.** This supersedes
the bitwise greedy-identity requirement in the historical entries below; their
measurements and immutable receipts are unchanged. MTP remains default-off and
unqualified, not a numerical failure solely because a low-margin winner changed.
The current gates are:

1. Every emitted token must be the argmax of the target logits computed in
   that pass. The serving cycle checks this before publishing its result.
2. Compare N1/off and N4/on at the **same token prefix**, not later positions
   of diverged free-running trajectories. A differing winner must exchange
   the same top two tokens, with **both raw-logit top2 margins strictly below
   0.10**. All declared emissions and token-ID SHA/count receipts are required;
   selecting only the nine historical first differences is not qualification.
3. N4 full-model logits versus authentic original-Q8, on the frozen matched
   corpus and same prefix geometry: mean full-vocabulary
   `KL(P_Q8 || P_N4) <= 0.005`, top1 agreement `>= 0.985`
   (at least79/80 on the80-context corpus). The480-natural-context qualification
   and its cohort/p99 gate are separate and must not be conflated with these80.

`tools/mtp_correctness.py` enforces the combined gates, failing closed on
missing coverage, receipt mismatches, nonfinite metrics or unmatched geometry.
Quality artifacts pin the original-Q8 and mix04 manifest SHA256 values and
the `ad99ce71...` corpus SHA, exactly80 samples and `candidate_columns=4`.
They require `reference_kind="original-Q8"`, `same_prefix_geometry=true` and
`frozen_corpus_matched=true`; a480-context artifact is not interchangeable.
`mtp_serving_qualify.py verify --correctness-events FILE --quality FILE
--report-output FILE` additionally binds the event receipts to actual HTTP
request IDs and runtime token hashes. Without these artifacts its legacy
identity check is only the historical stronger check, not an alternate way
to waive the new model-quality gate. Full N4/Q8 quality and complete pass-local
capture remain pending; no production lease is implied by unit-test success.
The existing component/rollback exactness checks remain in force.

`--mtp-local-head FILE` (requires `--experimental-mtp`) loads a full raw Q8
head or an authenticated `FGDVOC1` draft-only subset on the native MTP owner.
It validates all original cooked matrix bytes
against the sealed manifest's full `output.weight`, rather than accepting
an arbitrary draft head. One typed RPC returns a k1–3 local draft chain;
contracted hidden states stay on GPU and the coordinator no longer rendezvous
with every distributed vocabulary slice after each draft token.
The coordinator overlaps the chain with target **checkpoint preparation**,
not with target transformer execution. Verification still uses the existing
causal N-column target path. Rejection uses the state-only saved-GDN/PLE commit
already implemented; private native KV reconstruction remains explicit.
Retain fanout starts all remote owners before coordinator-local retention,
then collects their acknowledgments, avoiding a serialized local/remote bubble.

Two native sessions require replay slots8/9 and10/11. The shared vocabulary
therefore owns slot12, not slot9: using9 collides with the first session's
warm graph. Total replay slots increase12→13, including command/fence and
descriptor-pool reservation, even when this option is off. The full-head
allocation is additional residency; native-only short-window benchmark fit
does not establish full-target8192/32768-context or dual-session admission.
Existing eager requested-live cap checks must still pass before readiness.
The new option adds no environment flag. `MTP_STAGE` records `overlap_ms`;
its disjoint sum subtracts this from draft+prepare before adding the other
components. Timing-mode `MTP_CHOICE` records the request namespace, pass-local
token and argmax.

### Restricted draft vocabulary and coordinator-retain regression

The optional subset changes **only draft proposals**. Target verification,
token IDs, target full-vocabulary argmax and target weight precision are
unchanged. A subset argmax maps its result back to the original global ID.
There is no new environment flag; the existing local-head option accepts
the typed container instead of a raw head.

The latest [offline goodput/fit receipts](../results/hillclimb/MTP_GOODPUT_LOCAL.md)
include actual Strata curated IDs, complete graph-warmed rank7 sessions2
admission at explicitly smaller cache budgets, and retained40-case teacher
acceptance. Subsets lose4.184% relative code yield in that counterfactual;
head-speed savings alone are not an end-to-end win. No new policy serving
noninferiority or whole-rank≤1.5× result is claimed.

Create a container with:

```powershell
python tools\mtp_vocab_pack.py head.q8 head.fgdv --ids ids.json --parent-sha256 COOKED_OUTPUT_WEIGHT_SHA256
python tools\mtp_vocab_pack.py head.q8 head.fgdv --ids-bin draft_vocab.bin --parent-sha256 COOKED_OUTPUT_WEIGHT_SHA256
python tools\mtp_vocab_inspect.py head.fgdv --raw-sha256 RAW_OUTPUT_WEIGHT_SHA256 --parent-sha256 COOKED_OUTPUT_WEIGHT_SHA256
```

`--ids` and `--ids-bin` are alternatives; binary IDs are little-endian u32.
The inspector authenticates the **complete raw source**, parent declaration,
IDs and exact file size before staging, returning the container SHA.
It does not replace the runtime's cooked-parent authentication or grant
production admission.

The parent is the sealed **cooked `output.weight` tensor SHA256**, not the
manifest SHA or row-major raw-file SHA. IDs must be unique valid global IDs;
the packer sorts them and refuses overwriting an existing file.
Format:8-byte `FGDVOC1\0`, LEu32 count,32-byte cooked-parent SHA,
count strictly ascending LEu32 IDs, then the entire original
675,430,400-byte raw Q8_0 head. The full source stays on disk intentionally:
runtime streams and hashes every cooked source tile against the sealed
parent and compares every selected GPU tile with that same stream.
An untrusted subset plus a claimed parent SHA would not authenticate its
rows. Corruption in even an unselected source row fails closed.
Only selected/padded rows and their ID map become GPU resident; this is
**not an artifact/disk-size optimization**.

`make test-mtp-vocab` exercises real-GPU mapped argmax/static replay and
rejects parent mismatch, duplicate IDs, unselected-source corruption and
truncation. `tools/bench-mtp-vocab MANIFEST FULL_RAW_HEAD_Q8` measures
full/40,525/106,299-row heads using synthetic hidden vectors and compares
subset results with the independent existing raw-row loader. Its contiguous
low-ID sets are reproducibility/performance examples, **not curated language
sets, measured draft acceptance or full-chain speedups**.
To measure actual curated IDs instead, append one or more authenticated
containers: `tools/bench-mtp-vocab MANIFEST FULL_RAW_HEAD_Q8 ENGLISH_FGDV CJK_FGDV`.
The full-head control remains mandatory and each subset is independently
compared with the original raw-row loader.

`tools/mtp-residency MANIFEST OVERLAY EMBEDDING 2 --local-head FILE --verify-bench`
adds the actual head before eager admission, then warms N1..4 target graphs
and three-step native graphs for **both private sessions**. Exit3 is a policy
denial, not a fit. Final `all_graphs_ready` is required; merely fitting
`target_verify_ready` is insufficient. The optional target timings use authentic
weights but synthetic HCs and zero ngram, not captured-state quality or acceptance.
Use an explicit valid `FG_QSA_WORKER_CACHE_MIB` (16..640); changing its budget
must be reported as a separate configuration.

The cold-retain fix validates QSA frontiers only inside the verified
journal's layer interval. A replicated coordinator intentionally reports
ownership of every layer; requiring foreign, untouched QSA frontiers to
advance caused `MTP retained QSA frontier mismatch`. Worker ownership is
rank-local, explaining why previous worker-only state tests missed it.
`bench-decode-fusion --q4 --mixed-experts --coordinator-state-test --no-clocks`
adds replicated-coordinator coverage: N1-only prefix warmup, first cold N4
at frontier22, later frontier127, both private slots and all proper retained
prefixes for N2–4. It checks full retention, saved-GDN state reconstruction,
continuation tolerance, exact rollback, unchanged HCs, stale rejection and
retained replay graphs. Run with the recorded roofline flags, unlimited
memlock and the full-shape fixture's≈10 GB GPU-memory allowance.

### Local-chain measurements and prelease model

The authenticated40-context local-chain benchmark checks **33,960/33,960**
draft IDs against an independent current direct GPU reference, including all
256 teacher-forced prefixes and a keep1..4 repair schedule. Seven differ from
the historical frozen draft IDs; those differences are retained, not aligned
away. Steady k3 drafting averages **8.989ms code /9.017ms chat**.

Simulating fixed-k3 frontiers over captured N1-teacher prefixes gives code
5120/1580=**3.240506 tokens/verification** and chat5120/2140=**2.392523**.
The all-prefix arithmetic average overweights skipped frontiers and is not
the online yield. These are offline estimates, not measured N4 acceptance.

The conservative code-round model is **99.816ms**; increasing target service
by10% gives **104.565ms**. It projects **32.46TPS** using simulated acceptance,
or33.93TPS using the different historical code-ring yield. It is not60TPS,
live-ring timing or a measured158→99.8ms repair improvement. It includes the
existing47.493ms trained owner proxy, an unmeasured proportional N4 transport
model, measured local draft/head and state-fixture costs, calibrated residual
and explicit control/host budgets. Only checkpoint preparation overlaps
drafting; no target-transformer or reduced-precision transport credit is taken.

The full head adds676,920,320 requested-live bytes. Adding it and the new
single-session journal to the authentic rank7 warm ledger projects
**14,660,611,624B**, above the13.5GiB cap with700MiB QSA cache. The diagnostic
single-session plan therefore uses **440MiB for both off/on arms**, projecting
14,387,981,864B with107,532,760B headroom. This excludes extra driver-pool
reservation and is **not measured full-target admission**. Original service
settings are restored; no dual-session or long-context fit is asserted.

The local-only plan is:
`Invoke-MTPServingQualification.ps1 -Action Plan -Profile SingleSessionPerf
-LocalHead FILE -DiagnosticGreedy`. A run still requires an explicit serialized
lease, sealed release artifacts and eager admission. Diagnostic runs never
qualify: complete actual serving receipts and same-geometry N4/original-Q8
model quality remain pending. The historical80 Q8 captures show a
geometry/cohort-confounded baseline discrepancy and cannot establish N4
attribution. The precision sibling has now retained all80 same-geometry
original-Q8 control vectors; the N4 candidate comparison remains pending.
Its final matched N1 comparison reports current-Q8 versus current-mix04
KL0.051191953/top1 85%, already failing the original-Q8 bar before N4.
All-hop FP16 and mask224 cumulative comparisons also fail. Historical-profile
differences are real but do not remove this same-geometry N1 target gap;
KLs cannot be subtracted. See the roadmap's matched-Q8 update. Attribution and
full N4 qualification remain blocked; no promotion or threshold waiver.
See [local-chain evidence and model](../results/hillclimb/MTP_LOCAL_CHAIN.md).

**Historical 2026-10-16 serving gate: FAILED_GREEDY under the superseded
identity rule; not promoted.**
The frozen sessions1/mask330 sweep completed adaptive and fixed k1/2/3
measurements without the dual-session residency blocker. Greedy identity passed
only6/8,6/8,5/8,6/8 respectively; every failed case was chat. All four code
cases passed each arm, but that is not full-model qualification. MTP-off
same-session comparisons passed8/8 throughout. No speedup is demonstrated:
best repeat-matched code TPS is30.10 versus30.17 off; chat loses at every k.
Offline target replay reproduces all nine low-margin top-two swaps; component
tolerance does not certify greedy identity. All eight owners were restored, verified and cleaned; the exclusive
lease was explicitly released at15:56:43Z, within the original45-minute grant.
See [current lease report](../results/hillclimb/MTP_SERVING.md) and
[raw receipts](../results/hillclimb/mtp-single-session-perf.json).
The two-session successor must fit both private target/native sessions,
including deferred allocations, before readiness. Do not bypass the residency
cap. A separate single-session performance candidate answers the speed question
without treating dual-session residency as a prerequisite.

### Single-session performance gate (measured, diagnostic only)

The frozen performance-only release is
`c9d0a7ad3170a088306b0e097c0a8a5088f53c14a861fd1b78b7be939a39a1c4`.
It adds `--mtp-fixed-draft 1..3`: true fixed width bypasses the adaptive chooser,
but still clamps to the remaining generation allowance. Without that option,
`--mtp-max-draft 3` retains adaptive exploration and width selection.
The fixed width must not exceed the maximum draft width. MTP remains default off.

`Invoke-MTPServingQualification.ps1 -Profile SingleSessionPerf -Action Plan`
performs no production access. Its candidate uses `--sessions 1`, mask330 and
nine arms: `A0/D3/A1/F1/A2/F2/A3/F3/A4`. D3 is adaptive with cap3; F1/F2/F3
are fixed1/2/3. Each MTP arm gets same-mask, same-session off baselines on both
sides, eight greedy token-ID/completion comparisons and serial chat/code TPS.
D3/F3 additionally exercise SSE/stop/cancel. This does **not** qualify concurrent
two-session serving. Restoration remains the pinned sessions2/wide1 service.
The unchanged 35-minute work limit and 38/39-minute watchdogs leave restoration
inside the 45-minute exclusive lease. No promotion is authorized.

| Arm | Same-session off TPS, chat/code | MTP TPS, chat/code | Tokens/verification round, chat/code | Greedy identity |
|---|---:|---:|---:|---:|
| Adaptive cap3 | 31.30 / 31.60 | 16.11 / 26.88 | 2.20 / 3.36 | 6/8 |
| Fixed k1 | 29.32 / 30.17 | 22.35 / 30.10 | 1.63 / 1.91 | 6/8 |
| Fixed k2 | 29.12 / 29.78 | 18.96 / 27.65 | 2.04 / 2.70 | 5/8 |
| Fixed k3 | 29.21 / 29.73 | 17.19 / 28.12 | 2.28 / 3.39 | 6/8 |

TPS is median request decode throughput, including repair. Off TPS is the
median of per-request before/after averages matched by case and repeat.
Tokens/round uses only those requests' `MTP_ROUND` records, joined to the
following actual request token-hash record; greedy/health/SSE are excluded.
It is not tokens per physical target pass: counting verification **and repair**
gives chat/code yields1.26/2.51 (adaptive),1.18/1.75 (k1),1.31/2.21 (k2),
1.30/2.54 (k3). Measured accepted/drafted fractions are41.6%/80.9%,
63.4%/91.0%,53.1%/86.6%,43.3%/81.2% respectively, not a distribution-exact
sampling or quality claim.

The first strict attempt stopped at adaptive6/8 and restored production.
An explicitly diagnostic retry used the **same original lease/deadlines and
unchanged frozen binary**, recording failures before continuing timings.
To preserve restoration time, later arms used one repeat instead of three,
with the128-token maximum unchanged. F1 had already started its three-repeat
suite, so only its repeat0 has a matching reduced A2 bracket. The table uses
six requests/category for adaptive and two/category for each fixed arm.
All full-arm observations, unmatched F1 samples, wall TPS, repair counts,
request IDs and the explicit diagnostic budget are retained in
[`mtp-single-session-perf.json`](../results/hillclimb/mtp-single-session-perf.json)
and its authenticated evidence archive. Small diagnostic samples are not
performance qualification. Adaptive/fixed3 SSE, custom stop and cancellation
checks completed, but do not override the failed greedy gate.

On .51, C17-Werror, four numerical-double eight-rank runtime runs (adaptive and
fixed1/2/3), session/publication tests, synthetic native-Q8 wide0/1 and independent
expert component checks pass. All320 SPIRV binaries and arithmetic-engine source
files match the prior authentic-pack qualification; its real-tensor proof is
inherited explicitly, not rerun or presented as new real-model greedy evidence.
The frozen source excludes subsequent eager-admission and residency-probe work.

### Offline diagnosis and saved-activation prefix commit

The apparent acceptance drop was an accounting mismatch, not evidence of bad
native conditioning. Comparable fixed-k3 **tokens per verification round** are
2.275 chat and3.38667 code, versus offline2.3899/3.2422 on different prompts.
The previously quoted1.26/2.51 counts extra target repair passes and adaptive
widths. These quantities must not be compared as acceptance measurements.
The repeat-matched fixed-k3 code round is120.516ms, not90ms; the latter divides
by physical target passes. At the measured3.38667 tokens/round and29.73051 off
TPS, break-even is113.912ms/round. No new performance sweep or production lease
has been used for this diagnosis.
Its254 emitted tokens came from75 verification rounds plus25 full target
repairs. Thus90.387ms per physical pass is120.516ms/round multiplied by75/100;
neither quantity is an isolated N4-kernel measurement. The synthetic block's
N4/N1 ratio2.41 excludes draft/head rendezvous, transport and target/native
repair, so it cannot be compared directly with90/33.

Existing code traces also separate rounds by observed outcome:

| Fixed-k3 code outcome | Rounds | Mean complete round, ms |
|---|---:|---:|
| All3 drafts accepted, keep4 | 50 | 101.589 |
| Reject first draft, keep1 | 6 | 140.457 |
| Accept1, keep2 | 7 | 161.600 |
| Accept2, keep3 | 10 | 171.013 |
| Final allowance clips to k2/keep2 | 2 | 137.582 |

All25 repair rounds average158.370ms versus101.589ms for the50 repair-free
rounds. These are measured outcome-group totals, not isolated target-repair
savings: routes, retained lengths, native warming and graph recording differ.
The serial full-ring repair is therefore a real cost, but the logs cannot
recover separate historical draft/transport/repair/host milliseconds.

The already-required authentic replay records wall time inside each owner
decode call. Summing the eight owners' medians gives the following diagnostic
service estimates on .51:

| N | Sum of owner-call medians, ms | Ratio to N1 |
|---|---:|---:|
| 1 | 20.3802 | 1.0000 |
| 2 | 31.0165 | 1.5219 |
| 3 | 40.5614 | 1.9902 |
| 4 | 47.4931 | 2.3304 |

These use mixed failure-prefix cohorts, not matched routing/work or a new
performance sweep. They exclude ring transport, distributed head/native
drafting and transaction RPCs; call wall time is not pure GPU event time.
They confirm that target small-N service remains expensive without inventing
a historical121ms component ledger.

`tools/mtp_diagnose.py` verifies original emitted-token SHA/count receipts,
joins rounds to their actual requests and reconstructs the frozen schedules.
Token hashes contain **little-endian** u32 IDs. Two public responses had stripped
trailing whitespace; reconstructing it matches the original receipts without
changing or guessing an unverified token sequence. The nine first differences
are zero-based generated-token indices:

| Arm/case | Index | Off token | On token | Off top2 margin | On top2 margin |
|---|---:|---:|---:|---:|---:|
| D3/chat-2 | 80 | 11394 | 90650 | 0.05844307 | 0.06398010 |
| D3/chat-3 | 83 | 28285 | 34044 | 0.03657532 | 0.03935623 |
| F1/chat-1 | 57 | 37099 | 42900 | 0.05163956 | 0.02357483 |
| F1/chat-3 | 72 | 15495 | 76186 | 0.07532501 | 0.02257919 |
| F2/chat-1 | 12 | 5790 | 2426 | 0.03769684 | 0.00788879 |
| F2/chat-2 | 80 | 11394 | 90650 | 0.05844307 | 0.08345413 |
| F2/chat-3 | 86 | 14273 | 5902 | 0.00860786 | 0.09817505 |
| F3/chat-2 | 80 | 11394 | 90650 | 0.05844307 | 0.06398010 |
| F3/chat-3 | 86 | 14273 | 5902 | 0.00860786 | 0.04960251 |

An independent native-wrapper/direct-reference replay on all40 authentic
captured contexts checks43,080 contracted hidden vectors bit-exact, including
private sessions and keep1/2/3/4 repairs. This checks native conditioning and
KV repair, **not** whole-target logits or historical greedy identity.
`tools/mtp_offline.c` replays the target rank-major with authenticated complete
rank packs and frozen N/keep schedules, collecting logits and recurrent-state
differences. Discarded speculative future tokens were not logged; the offline
tool substitutes teacher-forced future tokens and requires all earlier greedy
predictions and both historical first choices to reproduce before claiming
historical margins. All nine pairs reproduce, with1272 earlier greedy
predictions matching. In every pair the winners exchange first and second
place. Full-vocabulary logit differences reach0.1393–0.4825: this is accumulated
batch-dependent numerical drift, not a claim that whole-model errors are only
1e-4. Selected-prefix KL(off||on) is0.001014–0.004841; these failure-conditioned
samples are **not** an unbiased model-quality gate. All QSA logical frontiers
agree. Trained rollback/reconstruction and future-column perturbation checks
provide separate state/causality evidence; matching logits alone does not prove
every possible state transition correct.
Across all eight owners and nine failed prefixes,72 trained checks reconstruct
the complete journal state bit-exact, repeat checkpoint rollback with identical
HC/recurrent state, and leave the first output row bit-exact when later input
columns are changed. All16 complete regenerated off/on streams have the same
SHA256 as the original replay. No rollback/frontier/position failure was found
in these cases; the reproduced failure is batch-dependent numerical trajectory
drift crossing the low-margin top2 decision. These checks use256 logical
tokens, not a262K full-context or whole-ring serving qualification.

The rejected-prefix successor no longer runs the transformer, experts or
vocabulary head a second time. Each private session eagerly journals the
verified GDN qkv/z/alpha/beta projections and PLE gated/normalized inputs.
`fg_owner_mtp_retain` restores the pre-verification recurrent shadows, applies
only the saved convolution/recurrence inputs through `n_keep`, and truncates QSA
to that frontier. Original verified hidden vectors are retained. A typed
`TARGET_KEEP` control broadcasts the retained prefix to every target owner;
serving deliberately has no full-target replay callback. Legacy/reference
transaction users can still supply restore/replay. Native `FINISH` repair
remains necessary and is not replaced by target state commit.
`n_keep` counts target inputs, including the already-published pending input,
not just accepted drafts. The guarded target frontier is `first+n_keep`;
native cache frontier is `first+n_keep-1`. The newly emitted bonus remains
pending until the next pass.

Retained GDN updates must use the **same small-N recurrence kernel family**
as verification, even for keep1. Selecting legacy prefill recurrence instead
introduced a5.7e-6 reconstruction difference during development; the corrected
same-family full-journal reconstruction is bit-exact in all24 GPU cases.
The synthetic two-session PLE+five-GDN+QSA block tests first positions0/127,
N2..4 and every rejected prefix, with duplicate/stale rejection, exact original
rollback, unchanged hidden vectors and session isolation. Prefix/continuation
component errors versus independent N1 references are bounded by0.00011635
maximum state error and6.20e-6 maximum continuation error. These are component
gates, **not** an exemption from same-binary whole-model greedy identity.
Run `tools/bench-decode-fusion --q4 --state-commit-test` for this correctness-only
fixture; it exits before timing sweeps or optional clock controls.

Historical logs recorded round totals, not component milliseconds. Source
inspection identifies serial native draft/coordinator/head rendezvous, serial
per-token vocabulary dispatch on each shard, rejected-prefix target replay,
native KV warming, and alternating verification/repair shapes invalidating
static command replay. Do not fabricate a numerical historical breakdown.
New successful-round `MTP_STAGE` records separate draft, preparation,
verification, target repair, native repair, commit and residual host time;
`target_replay` must be zero in serving. Optional `FG_DECODE_MS` records expose
owner input/compute/read/forward and native/vocabulary stages. Offline
`block_ledger` records measure input/compute/read on one blade; they are not
historical ring timings or a new TPS claim. A local native draft chain and
draft/early-verification overlap are not implemented.

Final [diagnosis/provenance](../results/hillclimb/mtp-diagnosis.json) and
[weight-free evidence](../results/hillclimb/mtp-state-commit-evidence.tar.gz)
retain raw selected logits,72 trained oracles, binary/shader identities,
source and experiment scripts. The archive is19,407,395 bytes, SHA256
`652eea89a06a66a3b35f293749917574d2694f53cfb6e5738e14440a2f3f3af4`.
Its source bundle SHA256 is
`007046094abc005e1061e7bb31ec6efd111e9fea7327a915b8adbc55ef3cd1e5`;
all523 C/header/include/Python/shader/Makefile inputs match the final worktree.
All320 SPIRV files match the original native-oracle build. Eight final TCP
runs validate891 successful stage ledgers and7128 owner ledgers, with
`target_replay=0` and disjoint sums. These are numerical-double correctness
timers, not hardware performance measurements. Use `stdbuf -eL` when combining
forked test stderr; unbuffered shared-log fragments can interleave and are
retained separately, not treated as complete timer records.

Build the offline helpers with
`make tools/mtp-tokens tools/mtp-offline tools/mtp-native-replay`.
`mtp-tokens` supplies authenticated token/vocabulary input, `mtp-offline`
provides `prepare`, `stage`, `oracle`, `head` and plan validation, and
`mtp-native-replay` checks native wrapper/reference conditioning.
`tools/mtp_ngram_source.py` authenticates the full ngram source used during
preparation. Their argument usage and the archived experiment scripts define
the replay; reconstructed rank-pack identities are listed in the JSON.
No trained weights are redistributed in the archive. The shared local source
packs remain intact. The .51 owned root was removed and absence checked twice
at18:33:58Z, with no owned processes/test listeners or governor changes;
production was untouched.

### Rank7 two-session memory budget

**State-only commit successor:** the historical figures below predate the new
projected-activation journal. Add263,680 bytes/GDN layer/session and327,680
bytes/session on the PLE owner. Thus rank7 dual adds2,109,440 bytes (complete
projection14,756,598,352, still rejected); relocated rank6 dual adds2,636,800
bytes (projection13,536,171,088; cap margin959,343,536). These are accounting
projections, not a new authentic full-context dual-session admission.

**Real-rank load on .52: one session admitted; two sessions rejected before
readiness under the unchanged cap.** All152 trained tensor spans are reconstructed
losslessly and checked against the sealed manifest:104 from pinned public GGUF
ranges and48 from the authenticated retained mix04 source. Global HC spans,
the target embedding sidecar and the real native Q8 overlay are included.
This is a load-only rank7 probe, not a ring join or a real-model generation gate.
See [memory/provenance receipts](../results/hillclimb/mtp-residency.json).

The target's152 payloads total10,327,643,648 bytes; its4096-aligned Vulkan arena
is10,327,752,704 bytes. Shared native cooked weights measure2,772,584,448 bytes.
The independently measured private allocation is38,705,712 bytes per native
head at window8192/capacity8196, plus51,200 wrapper input/embedding bytes.
The distributed target vocabulary head is reused, not duplicated on rank7.

| Rank7 component | Bytes | Evidence |
|---|---:|---|
| Target trained-weight arena, including padding | 10,327,752,704 | Sealed152-record offsets |
| Target shared owner/expert/HC/activation scratch and both actual GDN state slots | 107,625,984 | .52 owner/expert/HC stage minus arena |
| Two target QSA GPU state/index/cache sets | 1,440,897,008 | 2×720,448,504 measured session delta |
| Both target sessions' GDN rollback shadows | 26,476,544 | 2×4 layers×3,309,568 measured bytes |
| Target small-N expert-down scratch | 1,638,400 | Eager N=4 capacity, shared between sessions |
| Shared native MTP core/head weights | 2,772,584,448 | Real-overlay .52 allocation |
| Two native KV pairs | 67,141,632 | 2×8196×512×4×2 |
| Two native private non-KV scratch/owner states and input buffers | 10,372,192 | Measured private delta minus KV, plus inputs |
| **Projected complete two-session GPU requested residency** | **14,754,488,912** | Measured components; second native context not admitted with the target |

The unchanged policy cap is14,495,514,624 bytes (13.5GiB). The complete two-session
projection exceeds it by258,974,288 bytes (246.98MiB). Actual admission rejects
at14,715,732,000 bytes, after both target sessions and their rollback shadows,
but **before** the second native context. The earlier cold projection omitted
28,114,944 bytes of deferred target rollback/small-N storage; it was a lower bound.

Actual single-session readiness is13,982,045,224 requested /13,982,045,232 allocated
bytes, leaving513,469,400 requested bytes below policy. Repeating target preparation
allocates nothing further. MemAvailable at the ready sample is1,156,321,280 bytes;
VmLck is134,217,728 bytes. Neither this single-rank load nor the synthetic
causal/session/rollback gates replace real-ring greedy/TPS qualification.

MemTotal is15,936,958,464 bytes (~14.84GiB). For the complete two-session projection,
subtracting512MiB driver reserve and128MiB pinned n-gram hot region leaves only
511,380,912 bytes for other host/kernel working sets, including12,140,752 bytes of
target QSA host metadata. Do not add all process RSS to GPU residency on UMA:
those mappings overlap. The4,200,020,730-byte n-gram file and embedding sidecar
are pageable mappings, not additional permanently resident whole-file copies.
The legacy manifest has no n-gram shard digest; the n-gram span instead has
independent pinned-original-GGUF range provenance. No physical Vulkan OOM was
reported and no fully warm two-session physical fit is certified.

The independent fit implementation adds typed `PREPARE` admission on **all**
target owners: eagerly create expert small-N scratch and each requested session's
GDN/PLE rollback shadows, check target live residency, and on the selected native owner
materialize all requested private native contexts and check their total. API
startup does this before opening its listener or announcing readiness. A bounded
memory denial is acknowledged without poisoning transport; the coordinator
drains all outstanding admission ACKs before returning a denial. Four
numerical-double eight-rank tests exercise denial/retry/idempotence, and
native/wire/API tests cover eager provisioning and rejection before listener
creation. No extra MTP target scratch is allocated when MTP is off. These changes
are not in the frozen single-session performance release above.

Reproduce on an owned, authenticated rank7 staging pack, without joining a ring:
`make tools/mtp-residency`, then
`tools/mtp-residency MANIFEST OVERLAY EMBEDDING 1` (ready, exit0) or
`tools/mtp-residency MANIFEST OVERLAY EMBEDDING 2` (policy denial, exit3).
Use the sealed production flags, mask330/wide1, and process-local256MiB memlock
limit; the probe uses production allocation factories. `--native-only MANIFEST
OVERLAY` measures isolated native allocation and explicitly is **not** a rank fit.
`mtp_rank_source.py` and `mtp-real-place` provide bounded public-range download
and per-tensor authenticated lossless cooking into owned staging files.

**Default-off relocation:** `--experimental-mtp --mtp-native-rank 6` selects
rank6 on the current contiguous block map. Omitting the placement flag retains
the original final-owner behavior and unchanged CONFIG bytes. Rank6 (.48) has a
measured target-only two-session footprint of10,648,701,936 bytes from the
released B0/K1 logs. Its five GDN layers require33,095,680 bytes of two-session
rollback shadows at the measured per-layer geometry; add1,638,400 bytes of small-N
scratch and2,850,098,272 bytes for both native contexts. This projects13,533,534,288
bytes, leaving961,980,336 bytes below policy. Rank7 would retain11,904,390,640 bytes
of target-only residency including eager verification storage. This is a
placement estimate, **not full-rank load-qualified**. The implementation carries
the selected owner in CONFIG and forwards authoritative final target hyperstates
from rank7 to rank6, preserving every F32 bit, with typed session/slot/sequence/
frontier/count acknowledgements. Draft, warmfill, repair and eager private-session
admission run on the selected owner. Both sessions are provisioned before API
readiness under the existing total-residency cap.
Keep target verification/QSA/GDN ownership and the existing four vocabulary
head shards unchanged. The ordinary native loader still rejects a wrong final
owner; a separate explicit placed-loader API validates the chosen predecessor.
Coordinator and vocabulary-shard relocation destinations are rejected.
Both selected blocks must be contiguous and unique. All ranks must use the
updated binary, and the overlay plus embedding sidecar must exist at CONFIG's
paths on the selected owner; the frozen `311be40` lease controller is not a
relocation qualification release.

One target frame may be in flight from the relocated predecessor to the final
block: after forwarding prefill/decode/verify (including repair), the predecessor
receives and applies the matching final HC before taking another target frame.
This prevents opposing blocking multi-MiB sends under pipelined prefill or two
sessions. The credit receiver reads only the final peer, services that peer's
QSA/GDN side requests in their own session namespace, then restores the original
slot. Unexpected/stale observations fail closed. Existing host wire/HC staging
is reused; there are no new lazy GPU allocations or arithmetic/shader changes.
The extra HC hop/ACK is real overhead and has no measured ring TPS claim.

`tools/mtp-residency --native-only MANIFEST OVERLAY --native-rank 6` checks the
authentic sealed runtime option and allocates two native contexts, then compares
eight trained-native-core replay steps (two private slots) bit-exactly with the
default owner. Synthetic HC/embedding inputs are **not** a real-text acceptance
gate or a full rank6 fit. Authentic rank6 target+native eager admission, full
model KL/top1, greedy identity, two-session serving and ring performance remain
required before any placement promotion.

On .51 the two trained native contexts request 2,849,995,872 bytes before **and
after** eight replay steps: no deferred native GPU allocation appears. Shared
weights are 2,772,584,448 bytes and each private context is 38,705,712 bytes.
The serving wrapper's two input/embedding pairs add 102,400 bytes separately.
Both private slots' HC and contracted outputs are bit-exact to the default
owner on the same GPU in an independent Vulkan replay namespace.
Eight real-network numerical-double ring runs cover both placements and all
four rejection positions, adaptive/fixed widths, continuation, repair,
coexistence, publication/SSE and cancellation. Four relocated runs additionally
stress 512-row (~20MiB) duplex HC credits across both sessions.
All 378 source files and all 320 frozen shader binaries were independently
matched; flags and six strict/diagnostic client tests pass. See
[`mtp-native-placement.json`](../results/hillclimb/mtp-native-placement.json).
The .51 owned root was removed and absence verified twice, with no owned
executables and the stock governor/config unchanged.

For a future **authentic rank6 staging pack**, without joining the ring, use
`tools/mtp-residency MANIFEST OVERLAY EMBEDDING 2 --native-rank 6`.
This owner-aware full probe provisions target QSA, rollback and small-N
storage plus both native contexts; the original unplaced probe API is retained.
No complete rank6 trained target file was available for that gate in this run.

**Real-format component gate: PASS.** The corrected `3c84f62` engine passed .51 with the exact
891,289,600-byte real-pack Q8 down slab:36 byte-exact native-N1 comparisons,
wide0/1 and small-kernel masks0/330. All1,225 target records /51 family-format
groups and31 native tensors are accounted for in
[the exhaustive format audit](MTP_FORMATS.md).
The full production recipe also passes150 actual-shape dense comparisons,
72 independent expert component checks,30 operator oracles, mixed-format
causal/session/rollback and real-overlay native-session tests.
No production access occurred for these gates; .51 scratch is removed.

The first
MTP k1/mask0 request failed closed because the then-current small-N kernel rejected
the real pack's cooked Q8_0 down projections. MTP-off masks0/330 matched all
eight greedy token-ID hashes; this does not qualify either MTP-on path.
Production was restored and verified on all eight owners, and the ring was
released after11m15s. Actual baseline numbers and evidence are in
[MTP_SERVING.md](../results/hillclimb/MTP_SERVING.md).

Cooked-Q8 down compatibility now has a conservative small-N path: one dispatch
contains N independent native-N1 walks, preserving routing-slot order, block
partition and reductions for both the default LDS path and the opt-in
`FG_EXPERT_Q8_WIDE=1` subgroup path. It deliberately does not share Q8 weights across
tokens or promote research kernels. Q4_K/Q5_K gate/up and both cooked Q5_1/Q8_0
down are accepted for complete owner-local 512-expert slabs. Q5_1 is unchanged.
The trained-pack serving gate remains unqualified after the residency failure
above. The corrected release and exhaustive format audit are hash-sealed
together; the controller rejects missing, tampered or mismatched qualification
before production access. All318 previous SPIRV binaries remain unchanged;
only the two exact-Q8 down variants are added. No research experiment is bundled.

Targeted component verification on an idle blade:

```sh
FG_SMALL_KERNELS=0 DS4_REMOTE_TEST_FILTER=expert_small_component ./tests/test_fg_vk
DS4_REMOTE_TEST_FILTER=small_q8_down_native ./tests/test_fg_vk
FG_TEST_Q8_DOWN_TENSOR=/path/to/authenticated-cooked-slab.bin \
  DS4_REMOTE_TEST_FILTER=small_q8_down_native ./tests/test_fg_vk
```

The last test requires the exact complete 891289600-byte cooked slab; verify
its SHA256 against the sealed manifest before running. It checks N2..4 bytewise
against native N1 with synthetic dense, cancellation and sparse-basis inputs,
plus an independent CPU basis oracle. This is a real-weight component gate,
not a trained-model greedy or serving qualification. `bench-decode-fusion
--q4 --mixed-experts` adds synthetic Q5_K/Q5_K/Q8_0 and Q4_K/Q4_K/Q8_0 layers
to the six-layer fixture for causal/session/rollback coverage.
The stock .50 synthetic compatibility receipts pass both native down modes,
strict mask0 N2..4 output/state identity, mask330 reassociation tolerance,
causal prefixes, two sessions and rollback; see
[component receipt](../results/hillclimb/mtp-q8-down-compat.json).
The actual .51 full-recipe receipts and candidate identities are in
[mtp-format-audit.json](../results/hillclimb/mtp-format-audit.json).
Production dense19 versus small-N compact arithmetic is not generally
byte-identical, even for mask0: actual-shape dense maximum absolute error is
`2.03e-6`; mixed-block output/state maxima are `9.54e-7` / `9.54e-6`.
Explicit component reassociation tolerance passes, while causal-prefix,
session-replay and rollback comparisons remain exact. Same-build full-model
greedy identity for **both** masks is still mandatory before reporting serving
performance; no model-level KL/top-1 or promotion gate is implied.

The offline-prepared acceptance lease and current-production restore procedure
are documented in [MTP_CAPTURE.md](MTP_CAPTURE.md). This diagnostic gate uses
replayed native proposals and the causal N=1 ring oracle, not MTP serving.

**2026-10-16 real-text qualification:** 20 chat + 20 code cases, 259 tokens each,
measured k=3 policy E **2.3899 / 3.2422**, and all **24 paired 256-token ring
greedy checks passed**. This is the causal N=1 reference gate, not optimized
small-N serving. See [measured acceptance and restoration evidence](../results/hillclimb/MTP_ACCEPTANCE.md).

2026-10-16. Branch `cuppaxanax/mtp-serving`, rebased onto promoted main `7381f93`.

**Experimental serving is linked, but not fleet-qualified or a 60 TPS result.**
The trained head exists. This branch implements a Q8-only overlay exporter, a transactional
greedy cycle, residual rejection sampling, dedicated owner-local small-N
verification kernels, durable QSA truncation and single-blade rollback tests. The actual
hyperconnection head now executes on Vulkan using the actual abliterated Q8
weights, with a separate causal draft KV cache and an independent CPU oracle.
The replayed GPU-routed core and fused vocabulary-shard argmax are described
in section 7; experimental native requantization remains a separate quality gate.
Batched causal ring orchestration and native contracted-hidden head fanout are
now linked behind the default-off `--experimental-mtp` gate. Startup requires
explicit, validated native-overlay and production-embedding sidecar paths.
Component tests are not trained-model end-to-end qualification. The two-host
runtime fixture described below tests orchestration with numerical doubles;
the trained eight-rank pack still requires a separately granted ring lease.

### Experimental serving contract

Export the *sealed target pack's* Q8 embedding without requantization:

```sh
make tools/mtp-embedding
./tools/mtp-embedding /path/to/pack/manifest.fgm /path/to/embedding.fgme
```

The sidecar binds the source tensor SHA256, layout, row count and width, and
independently authenticates its raw-Q8 payload. Raw and cooked Q8 inputs produce
bit-identical dequantized rows. Put this sidecar and the native overlay on the
last-layer owner at the same paths supplied to rank 0:

```text
--experimental-mtp --mtp-overlay /path/to/native-q8.gguf
--mtp-embedding /path/to/embedding.fgme --mtp-window 8192 --mtp-max-draft 3
```

The window defaults to min(context,8192), maximum draft length to 3. No MTP
options are admitted without the explicit gate. The target manifest need not
contain MTP tensors because the overlay is independently loaded. Startup also
requires protocol 6, text positions, complete owner-local expert slabs,
ring prefill/decode and the four-way output head.

The final owner runs the native draft head and fans out its **already
contracted** hidden vector directly to the vocabulary shards, without applying
the target HC head a second time. Prefill next-token IDs travel with their
ordinary work frames; native warmfill completes before each final-owner prefill
result. This avoids an extra embedding hop and a bidirectional warm-ACK
deadlock. Native weights are shared between sessions, but draft KV, scratch,
target-HC history, sequence namespaces and adaptive policies remain private.

Verification consumes pending input plus 1–3 drafts causally through every
owner block. Rejection restores device state **and host ingress frontiers**,
then causally replays only retained inputs. QSA service guards publish only
the committed frontier. A partially failed distributed transaction poisons
the transport instead of pretending released peer checkpoints can be restored.
Tokens are published only after target commit; a final pending token is flushed
normally at the output limit. Rolling per-stream acceptance and measured round
cost select k. Sampling/penalty and vision requests use the ordinary decoder;
the residual rejection-sampling primitive is not wired into serving yet.

An 8192-cell draft window is a draft-cache approximation, never a target-state
approximation. Existing real-text acceptance figures above do not qualify this
new warmfill/window/orchestration path. No distributed draft latency, new serving
TPS or sessions-2 memory capacity is claimed. Production-width raw/cooked
sidecar rows match the actual target GPU embedding kernel bit-for-bit on .51.
The generation-loop fixture covers bounded publication, cancellation, EOS,
split custom stops and SSE, including two active coroutine sessions. The
remaining trained-model gates are current-kernel MTP-on/off token identity,
target/native co-residency, real draft acceptance/latency and ring throughput.

### Two-host orchestration fixture and bounded serving lease

`tests/test_mtp_ring.c` includes the production runtime and links the real
fabric, wire codecs, native-manager state machine, head fanout and distributed
transactions. Deterministic CPU numerical doubles replace model/GPU arithmetic:
this is **not a two-blade execution of the trained eight-rank model**, nor a
TPS/acceptance measurement. The final one-input flush uses a compile-time test
seam into the causal verifier, not the production serial-output protocol; the
normal build is unchanged by that seam.

Eight logical ranks run as four processes on each of .51/.52. Two sessions
have distinct token histories, separate native managers/policies and actual
overlapping `fg_pipeline` jobs. Cases cover k=1..3, every rejection prefix,
prefill next-token trailers/warmfill, authoritative repair, output limits1..8,
EOS at distances0..4, draining cancellation, and continued generation. The
real API SSE sink receives callbacks from committed bursts; its decoded event
contents, stop suppression, final reason and `[DONE]` are checked independently.
`tests/test_api.c` additionally checks malformed/overlapping/UTF-8 stops,
partial prefixes at completion, JSON/SSE parity and discarded-prefix reset.

Reproduce local orchestration with `make test-mtp-ring`; for the pair, use
`tools/mtp-runtime-pair-test.sh` through `Invoke-MTPFleet.ps1`. Its optional
`open`/`close` actions own only a source-IP-restricted, runtime firewall rule
for the peer's fixture ports, automatically expiring after600seconds. No
permanent firewall, service, governor or clock configuration is changed.
Always close the owned rule and retain evidence before deleting scratch.

The separate **serving** controller is
`tools/Invoke-MTPServingQualification.ps1`; the older
`Invoke-MTPQualification.ps1` remains a historical HC-capture/oracle workflow.
`-Action Plan` never contacts production; `Run` requires an explicit exclusive
lease, fresh ID, validated release archive and the pinned native overlay.

Planned wall budget: **45minutes**, with work stopped at minute35, independent
worker/coordinator restore watchdogs at38/39, and the remaining margin for
API/identity checks and cleanup:

1. Preflight all pinned production identities/hashes and API; stage an
   immutable candidate, symlink the sealed mix-04 pack, and export its exact
   Q8 embedding sidecar. Do not modify current launchers, weights or hardware.
2. Restart the same candidate build in bracketed phases
   `A0/B0/K1/S1/A1/K2/S2/A2/K3/S3/A3`, always sessions2 and the current explicit
   wide1/roofline recipe. A phases are MTP-off/mask0, B0 is off/mask330,
   K1..3 cap drafts at1..3 with mask0, and S1..3 use the same caps with330.
3. Before timing each phase, compare eight fixed chat/code greedy completions
   against the first off run. Existing `FG_RING_TRACE` emits one per-request
   SHA256/count receipt over actual token IDs, so text equality cannot conceal
   a different tokenization. Any mismatch aborts and restores production.
4. Three repeats each of two chat and two code prompts,128 output tokens,
   retain engine decode TPS and client wall TPS separately. Report per-prompt
   candidate/bracketed-off ratios, actual adaptive-k/acceptance traces, and
   three concurrent chat/code pairs per phase. Exercise SSE parity, custom
   stops and disconnect/recovery with MTP enabled.
5. Always restore **31508ee /20261016-abq-stack-adaptive**, sessions2/wide1,
   using the retained hash-pinned current launcher. Check all eight argv/env/
   binary/manifest/shader identities, health/models and real greedy output;
   only then cancel watchdogs, archive evidence and remove owned scratch.
   Missing historical capture cases are optional when archiving serving logs;
   an evidence/download failure must not skip other owners or scratch cleanup
   after successful production verification. Finalization failures are recorded
   separately from the original request failure.

This lease is short-context serving qualification, not promotion, a long-
context capacity test or a qualified use of research kernel mask330. The
latest independently qualified kernel costs and remaining gaps are in
[MTP_SMALL_KERNELS.md](../results/hillclimb/MTP_SMALL_KERNELS.md).

## 1. Source facts, not the old feasibility assumptions

Range-only audit of these immutable Hugging Face revisions:

| Source | Revision | MTP tensors | BF16 bytes |
|---|---|---:|---:|
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` | `298f94632b784e26a7fe576114f82066689d5baa` | 31 | 5,214,301,696 |
| `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 31 | 5,214,301,696 |

Both complete configs are identical, as are the MTP tensor names, dtypes and
shapes. Tensor *data* equality was not tested. The original head is structurally
compatible as a draft head, but there is no need to borrow it: the abliterated
source retains its own. A less accurate draft still gives exact target output
only when verification, rollback and stochastic acceptance are correct.

`tools/mtp_source.py` fetches config/index JSON, then 8-byte safetensors lengths
and bounded JSON header ranges, not full shards. It pins the repository commit,
requires HTTP 206 with the exact Content-Range, and records offsets and header
SHA256. Audit records are in `results/hillclimb/mtp-{original-,}source.json`.

All following shapes are **HF row-major**, with the `mtp.` prefix omitted.
All source tensors are BF16:

| Name | Shape | Bytes |
|---|---|---:|
| `fc_embedding.weight` | 2560 x 2560 | 13,107,200 |
| `fc_hidden.weight` | 2560 x 2560 | 13,107,200 |
| `pre_fc_norm_embedding.weight` | 2560 | 5,120 |
| `pre_fc_norm_hidden.weight` | 10240 | 20,480 |
| `hyper_connection_mixer.hc_norm.weight` | 10240 | 20,480 |
| `hyper_connection_mixer.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 |
| `hyper_connection_mixer.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 |
| `layers.0.attn_hyper_connection.block_inject_weight.weight` | 4 x 10240 | 81,920 |
| `layers.0.attn_hyper_connection.hc_norm.weight` | 10240 | 20,480 |
| `layers.0.attn_hyper_connection.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 |
| `layers.0.attn_hyper_connection.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 |
| `layers.0.mlp_hyper_connection.block_inject_weight.weight` | 4 x 10240 | 81,920 |
| `layers.0.mlp_hyper_connection.hc_norm.weight` | 10240 | 20,480 |
| `layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 |
| `layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 |
| `layers.0.mlp.experts.down_proj` | 512 x 2560 x 640 | 1,677,721,600 |
| `layers.0.mlp.experts.gate_up_proj` | 512 x 1280 x 2560 | 3,355,443,200 |
| `layers.0.mlp.gate.weight` | 512 x 2560 | 2,621,440 |
| `layers.0.mlp.shared_expert.down_proj.weight` | 2560 x 640 | 3,276,800 |
| `layers.0.mlp.shared_expert.gate_proj.weight` | 640 x 2560 | 3,276,800 |
| `layers.0.mlp.shared_expert.up_proj.weight` | 640 x 2560 | 3,276,800 |
| `layers.0.mlp.shared_expert_gate.weight` | 1 x 2560 | 5,120 |
| `layers.0.self_attn.indexer.index_qk_proj.weight` | 640 x 2560 | 3,276,800 |
| `layers.0.self_attn.indexer.k_layernorm.weight` | 128 | 256 |
| `layers.0.self_attn.indexer.q_layernorm.weight` | 128 | 256 |
| `layers.0.self_attn.k_norm.weight` | 256 | 512 |
| `layers.0.self_attn.q_norm.weight` | 256 | 512 |
| `layers.0.self_attn.k_proj.weight` | 512 x 2560 | 2,621,440 |
| `layers.0.self_attn.v_proj.weight` | 512 x 2560 | 2,621,440 |
| `layers.0.self_attn.q_proj.weight` | 12288 x 2560 | 62,914,560 |
| `layers.0.self_attn.o_proj.weight` | 2560 x 6144 | 31,457,280 |

The two expert tensors alone contain 5,033,164,800 BF16 bytes. The old claim
that this head is merely hundreds of MB is wrong. With Q8_0 matrices and F32
vectors the overlay payload is **2,770,227,872 bytes (2.580 GiB)** before GGUF
metadata/alignment, excluding shared embedding/head and MTP KV allocations.
Q4-class formats could reduce residency later, but require a quality gate;
no format below Q4 is permitted. This branch exports only Q8_0/F32.

The exporter preserves HF names, fused expert/indexer tensors and **raw norm
gammas**. It is a source overlay, not an executable owner-layer alias. The
native importer splits expert gate/up along the 1280 axis, cooks supported
matrices, and applies each Gemma `+1` exactly once. Its dense draft attention
intentionally skips the sparse indexer, as Strata does. Do not silently feed
raw zero-centred gammas into ordinary RMSNorm. The complete native overlay and
shared vocabulary head were range-exported directly on .51; native architecture
validation is described in section 6. Real-context quantization quality and
acceptance are not yet qualified.

## 2. Correct architecture

The actual config says `Qwen4ExpForConditionalGeneration`, hidden 2560,
HC count 4, HC low rank 320, 512 experts/top-10, one hybrid full-attention
MTP layer and `mtp_use_dedicated_embeddings=false`.

The relevant reference is SGLang's
[qwen4_exp_mtp.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen4_exp_mtp.py),
particularly `_fuse_residual_linear_shared`:

1. Embed the known next token using the target embedding.
2. Gemma-RMS-normalize its 2560-wide embedding and project with `fc_embedding`.
3. Gemma-RMS-normalize the **entire 10240-wide target HC hidden**. Reshape it to
   four 2560-wide streams and apply the same `fc_hidden` to each stream.
4. Add the projected embedding to every projected HC stream.
5. Execute one HC/full-attention/MoE/HC layer, with no PLE and with separate
   draft KV state. Do not use the target's sparse indexer to restrict draft
   attention just because indexer tensors also occur in the source overlay.
6. Contract through the **MTP's own** `hyper_connection_mixer`; apply the shared
   target vocabulary projection. Preserve the pre-contraction HC output for
   recursive drafting and accepted-target hidden-state refresh.

This is not the plain normalized embedding/hidden concat into a 5120->2560
projection used by
[vLLM Qwen3Next MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py)
and
[SGLang Qwen3Next MTP](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next_mtp.py).
Local read-only `llama.cpp\src\models\qwen3next.cpp` implements that NextN
graph, but the local `qwen4exp.cpp` has no MTP graph and its
`conversion\qwen4exp.py` explicitly disables MTP export. Nothing in llama.cpp
was modified.

## 3. Placement and causal ring design

Prefer the rank owning the final target block and embedding, currently rank 7,
not a hardcoded rank number. It already has the full final HC hidden before
direct output handoff. Keeping the draft layer there avoids returning each
10240-float hidden to rank 0. Reuse the existing four-way vocabulary-head fanout,
but feed it the **MTP-contracted** 2560 vector rather than the target HC mixer.
Account for a head/fanout on each draft step; those costs are not free.
Prove the extra 2.58 GiB plus separate draft KV, target state checkpoints and
verification scratch fit that rank before loading an overlay.

Use configurable `k=1..3`; initially qualify k=1. Once target token `t` has
already been emitted and is pending consumption, draft `d1..dk`, then send
**one causal sequence** `[t,d1,...,dk]` through every stage, requesting every
row's hidden/logits. Positions increase within this sequence.

The existing `decode_batch` protocol/scheduler is **not this operation**:
its slots are independent sessions with distinct recurrent namespaces.
`coordinator_decode_batch_step` and `handle_decode_batch_work` cannot simply
be handed repeated state slots. The existing prefill layer supports causal
rows, but has a very different execution/cost/numerical path.

Ring transactions must:

* checkpoint GDN convolution/recurrent state and PLE convolution on **each
  owner**, plus target and draft QSA frontiers;
* isolate speculative QSA pages, index metadata, positions and replication
  cursors; do not publish them to storage ranks until acceptance;
* return all k+1 target logits/HC rows, not just a prefill chunk's last row;
* commit the verified prefix on all owners, with generation/session identity
  checks and poisoned transport on a partial distributed failure;
* restore the initial recurrent checkpoint and replay the accepted input
  prefix on a rejection, until per-position recurrent checkpoints exist.
  Keeping only a final GDN state and truncating QSA is **incorrect**.

The low-cost endpoint should retain per-position GDN/PLE state during the
short sequential recurrence, so rejection selects a prefix checkpoint instead
of executing accepted tokens again. A full-state copy on every draft position
must be included in the bandwidth ledger.

The MTP QSA cache also needs prompt warmfill/teacher-forced updates using
shifted embeddings and target HC hidden states. After verification, repair
its accepted prefix with target HC rows, not rejected recursive draft hidden
states. There are no captured abliterated target HC streams here; offline
acceptance was consequently **not measured**, and hidden-feature argmax
agreement in the synthetic benchmark is not an MTP acceptance measurement.

### Strata working reference

Read-only local reference: `D:\looking-glass-labs\Strata`, revision `9259cad`
([upstream](https://github.com/Niko1221/Strata/tree/9259cad)).
`docs/DETAILS.md:743` reports **2.4–3.2 committed tokens/pass**, with the native
MTP head drafting **up to three** tokens. This is a published cross-harness
estimate, not an acceptance measurement for our abliterated pack or Q8 overlay.
The nearby published speed table uses lower weight quantizations than our Q4
floor; its TPS numbers are not comparable and are not used as FG measurements.

The relevant implementation contracts are:

* `src/core/mtp.cpp`: draft on the last layer-split stage's device. Normalize
  the full HC vector, apply shared `fc_hidden` to its four streams, combine
  with the separately normalized/projected next-token embedding, run the
  HC/full-attention/MoE block and the draft-specific final HC mixer, then use
  the shared vocabulary head. Draft KV is independent of target sparse KV.
* `include/strata/core/coupled_draft.hpp` and its host test: cell c pairs
  target HC at c with token c+1 and predicts token c+2. After a window starts
  at p and accepts a drafts, the next window starts at p+a+1. Correct
  absolute positions and prefix penalty history matter as much as tensor
  arithmetic.
* The draft loop stops when the last proposal's probability is below
  `--spec-min-p`, up to the configured cap. `spec/draft_policy.hpp` compares
  measured committed-tokens/round-ms, including verify, commit **and draft**;
  its capped-window EMA and small decision margin are preferable to always
  assuming k=3 is best.
* `src/core/verify.cpp` and **`src/kernels/cuda/verify_kernels.cu`** (the
  requested `kernels/cuda` root path does not exist): graph-captured
  multi-column verification preserves single-token operation order.
  Verification leaves authoritative GDN state untouched; commit replays
  only retained recurrence/conv updates from cached per-layer projections.
  It also restores and rebuilds the accepted sparse-index tail. A final GDN
  state plus KV truncation is not a prefix commit.
* Strata's optional coupled sampling uses the target sampler and absolute
  Philox counter, accepting equal target draws. FG's requested standard
  speculative rejection sampler instead needs complete normalized p and q.
  These are distinct valid stochastic constructions; an argmax-only test is
  not a substitute for either.

Strata's `DETAILS.md:75` explicitly documents a grouped-CPU rounding issue
and an opt-in deterministic arithmetic fix. Our GPU verifier is therefore
tested against the **same binary's N=1 operation order**, rather than using
another harness's accuracy claim as evidence of FG bitwise equivalence.

## 4. Acceptance and state frontier

`fg_mtp_cycle_greedy` implements the cycle independently of a particular
transport/backend. Target row j predicts draft j+1. Emit the matching draft
prefix and the first target correction, or a target bonus if all drafts match.
If a cycle emits m new tokens, consume exactly m inputs: the already emitted
pending token plus the emitted prefix except its last token. The last emitted
token stays pending. EOS and output budgets shorten that frontier too.
No result is returned until checkpoint/verify/restore/replay/commit succeed.

Temperature sampling cannot use target/draft argmax equality. For proposal
distribution q, target p and proposed token d, accept with
`u < min(1,p[d]/q[d])`; on rejection draw from normalized `max(p-q,0)`.
`fg_mtp_rejection_sample` implements and tests this primitive, including
disjoint supports and an exactly integrated two-token distribution.
Distributions must include the same temperature, top-k/top-p and history
penalties as the baseline sampler, at each hypothetical prefix. The existing
sampler's filter/temperature ordering must be preserved. They are not raw
logits or only a proposal's scalar probability.

This stochastic primitive is **not connected to runtime logits or the cycle**.
Distributional equality does not imply identical seeded sample trajectories.
Greedy runtime equality also needs validated numerical semantics across the
complete target ring. The legacy prefill verifier does not have bitwise hidden
parity; the new small-N block has separate, stricter parity tests.

## 5. Diagnosing and replacing the broken verifier on .51

### Historical prefill fallback

Derived from `cuppaxanax/decode-fusion`'s `tools/bench_decode_fusion.c`,
retaining the production N=1 `fg_owner_decode_block_chained` static-replay
baseline. The imported fixture is adapted to build against main, with
nonintrusive wall timing and **no** fusion-branch clock API dependency.

N=2..4 calls `fg_owner_verify_block`, a causal block wrapper around the
existing prefill layers and local expert prefill. Same owner-local layers
6..11, 9,739,403,264 synthetic Q4/Q5 weight bytes, nonzero changing inputs,
full 512 expert slabs, real top-10 routing and QSA. These are not model
weights, full model logits, fleet transport or a fused small-N verifier.

Two fresh seeds, forward/reverse N order, 256-token warmup rounded down to a
whole batch and three windows of approximately 64 input tokens each.
N=3 starts at 255 and covers 189 measured inputs; others cover 192. Comparison
aligns overlapping absolute token positions. Stock governor is untouched:
endpoint samples range from 1000 to 2000 MHz, so these are coarse cost
measurements, not a tightly pinned qualification or confidence interval.

| N=k+1 | Mean block us | Mean per-seed cost ratio to N=1 | Individual ratios |
|---:|---:|---:|---|
| 1 | 2,675.68 | 1.000 | 1.000 / 1.000 |
| 2 | 22,862.45 | 8.545 | 8.506 / 8.583 |
| 3 | 23,946.70 | 8.950 | 9.028 / 8.871 |
| 4 | 25,066.67 | 9.369 | 9.151 / 9.587 |

**The legacy prefill verifier is the wrong decode implementation.**
It includes different kernels, CPU routing/readback, expert staging/reduction,
QSA persistence and no GDN static replay. This does not prove an optimized
small-N kernel cannot work; it proves existing prefill is not that kernel.
These fallback costs are not evidence about MTP's achievable speedup.
All compared hidden-feature argmaxes match, but **zero** N>1 output vectors
are bit-identical to N=1; max absolute hidden differences reach 0.18512.
No inference about actual vocabulary argmaxes/quality follows.

Raw window samples, endpoint clocks and configs are retained in the local
`mtp-measure.tar`; tracked summary: `results/hillclimb/mtp-batch.json`.
`results/hillclimb/mtp-evidence.tar` additionally preserves raw measurements,
all rollback runs, the validation build log and tested binary SHA256 hashes.

### Dedicated small-N decode path

`fg_owner_decode_small_n` is an explicit experimental API, not an automatic
change to production prefill or ordinary decode. `--small-n` selects it only
in `bench-decode-fusion`. It removes the historical fallback machinery:

* Q4_K/Q5_K/IQ4_NL compact **and noncompact SoA** matrices use N-column
  decode kernels; cooked Q8, F32, BF16 router and BF16 index projections
  also have dedicated N=2..4 variants. A weight tile feeds all columns
  instead of padding the operation to prefill's 8/32-column MMQ.
* Shape-specific workgroup widths match N=1's reduction geometry, including
  the 256-thread HC contraction and 128-thread GDN input projection.
  Merely sharing weights while changing reduction geometry was insufficient
  for bitwise recurrent-state equivalence.
  The selected Q4/Q5/IQ4 kernels interleave columns/pairs so identical weight
  addresses coalesce within a wave. N=3/4 keep two private columns per lane
  instead of retaining four columns' inputs and accumulators. Q8 uses the
  same two-column scheme; no K-loop LDS staging/barriers are needed.
* GPU top-10 selection and a bounded union over at most 40 selected pairs
  replace host route readback and CPU expert reduction. Four-pair tiles
  share expert gate/up and down reads, with each token's own routing gate.
  Repeated experts spanning more than one tile are handled explicitly.
  Q5_1 down projection saves four lane partials per selected pair, then
  replays each token's ordered ten-gate FMA chain before the original
  four-way reduction. Reducing each expert's full vector before applying
  gates would change N=1 rounding. This scratch is allocated lazily only
  for the experimental path. Gate/up uses two private pairs per lane;
  top-10 uniqueness bounds a two-token union to two pairs per expert.
* GDN processes the N recurrences sequentially inside one dispatch using
  the same unrolled algebraic step as ordinary decode, not the chunked
  prefill recurrence. GDN runs between QSA layers use shape-keyed static
  replay. QSA remains dynamically recorded.
* QSA batches projections, then performs ordered per-position cache commit
  and decode attention. Each position has distinct gather-ID storage; no
  mapped buffer is rewritten underneath queued GPU work. Future rows are
  not visible to an earlier query. The path avoids prefill persistence and
  asynchronous page-prefetch work, draining before necessary cold-page
  host writes.
* Failure paths abort recording, drain queued work and invalidate replay
  slots. The caller must still restore its checkpoint: submission cleanup
  is not a state rollback.

Intrusive profiling identified three concrete regressions: noncompact SoA
projections still falling into MMQ, chunked rather than algebraic GDN
recurrence, and an expert scheduler scanning all 512 experts serially.
All three now have explicit decode-specific dispatches. RADV shaderstats
for the tested four-row N=4 kernels reported **zero spills and zero scratch**;
the slowdown must not be attributed to register spilling without evidence.
Reducing the output tile to one/two rows regressed memory reuse and was
discarded. Performance must be judged by warmed, nonintrusive replay timing,
not per-dispatch instrumentation or a four-token smoke run.

Binary16 staging of *integer* decoded quants was also rejected. The stored
integers were lossless, but RADV folded their conversion into
`v_fma_mix_f32`, unlike N=1's `v_mac_f32`/`v_mad_f32`, producing an
8.94e-8 byte-parity failure. Retaining integer decoding avoids that change;
the production N=1 arithmetic was not changed to make the test pass.
Dense microbenchmarks now execute before the 9.7-GB owner fixture is
allocated: measurements interleaved with that large fixture showed severe
allocation-pressure variability and are not the isolated-kernel result.

The new benchmark tests dense outputs byte-for-byte for all supported
formats, both packing layouts and short/HC/GDN geometries. It also checks
whole-block outputs and recurrent state, causal prefix identity, union
scheduling and recovery after an injected expert failure.

### Warmed small-N results

Two seeds, forward/reverse N order, the same 256-input warmup and three
approximately 64-input measurement windows as the historical fallback:

| N=k+1 | Mean block us | Mean per-seed ratio | Individual ratios |
|---:|---:|---:|---|
| 1 | 2,684.09 | 1.000 | 1.000 / 1.000 |
| 2 | 4,077.88 | 1.519 | 1.507 / 1.531 |
| 3 | 5,902.78 | 2.199 | 2.268 / 2.130 |
| 4 | 6,724.68 | 2.505 | 2.448 / 2.563 |

All **1,144** aligned nonbaseline HC rows are byte-identical to serial N=1,
max absolute difference zero. This is the same six-layer synthetic owner,
not a full-ring run or an acceptance measurement. Final-layer expert unions
contain 10–12 experts, not 40; that diagnostic is not a measurement of every
layer or production routing. Stock clock endpoint samples include both
1000 and 2000 MHz; no governor/clock setting was changed.

The 2560→6144 isolated replay microbenchmark measured:

| Packing | N=1 us | N=2 ratio | N=3 ratio | N=4 ratio |
|---|---:|---:|---:|---:|
| compact Q4_K | 48.06 | 1.512 | 1.820 | 2.042 |
| compact Q5_K | 45.46 | 1.309 | 1.871 | 2.026 |
| IQ4_NL | 30.47 | 1.552 | 2.436 | 3.010 |
| cooked Q8_0 | 59.90 | 1.193 | 1.541 | 1.797 |

The Q8 path is near the requested small-N costs, but the **whole-block
N=2≤1.3 and N=4≤1.8 targets are still not met**. Q4/IQ4 projection and
remaining verifier overhead require more optimization. The measured N=4
shader-statistics run covers 30 pipelines with zero spills/scratch; the new
gate/up, union-down and ordered-reduce kernels use 64/64/16 VGPRs respectively.
The original 8.5x prefill defect has been substantially reduced, not replaced
with a qualified near-single-token implementation.

Validation also passed 90 strict dense cases, 12 union-scheduler cases,
N=2..4 whole-block output/state parity, causal prefix and injected-error
cleanup checks, and **16/16** retained-prefix rollback cases spanning all
four QSA page offsets. Prefix tests now switch N=1/N>1 replay modes without
manually clearing static graphs. The full Werror target, MTP/sampler/depth-B
host suites and flag registry pass. Summaries and raw configs/windows,
shader statistics, source/binary/SPIR-V hashes and logs are in
`results/hillclimb/mtp-small-n.json` and `mtp-small-n-evidence.tar.gz`.

### Projection, not a measured ring speedup

Let F be baseline single-token latency, r the actual **whole-ring** verification
cost/F, m the draft-step cost/F, and C rollback/commit overhead/F. At constant
per-position acceptance a, the independence approximation is:

```
E = 1 + a + ... + a^k
TPS = baseline_TPS * E / (r + k*m + C)
```

Use measured prefix survival probabilities instead of powers of a once they
exist. From 27 TPS, 60 TPS needs a speedup of 2.222. For k=3:

| Assumed a (not measured) | E | Maximum total cycle/F to reach 60 TPS |
|---:|---:|---:|
| 0.7 | 2.533 | 1.140 |
| 0.8 | 2.952 | 1.328 |
| 0.9 | 3.439 | 1.548 |
| 1.0 | 4.000 | 1.800 |

The old 8.5x–9.4x values above are a **verify-path implementation regression**,
not a property or ceiling of MTP. The earlier 11.5-TPS extrapolation from that
broken path is withdrawn. MTP's value for this architecture is established;
the engineering task is to remove our prefill fallbacks and qualify the
native head and prefix transactions.

For a Strata-referenced projection use E=2.4–3.2 **directly** rather than
inventing an FG per-position acceptance probability. At 27 TPS the estimate
is `(64.8–86.4)/(r+k*m+C)` TPS; at a qualified 40–45 TPS serial baseline it
is `(96–144)/(r+k*m+C)`. Here r must come from the corrected verifier,
eventually measured for the entire ring. Draft and commit costs are not
zero; neither these estimates nor a one-blade synthetic ratio are a fleet
TPS result. Strata's adaptive “up to three” policy is also not a measurement
of a fixed N=4 verifier.

For transparency, substituting the still-regressing synthetic N=4 block
ratio 2.505 for r would give 25.9–34.5 TPS from 27 TPS **before** draft and
commit. This is a diagnostic substitution, not a ring prediction or an MTP
ceiling: r is not yet a whole-ring measurement and our implementation
still misses its cost targets. At a future qualified 40–45 TPS serial
baseline and r=1.8, the same published E gives 53.3–80 TPS before draft
and commit. Neither scenario is a measured production speedup.

## 6. Native head execution on .51

`fg_mtp_head_open/step/truncate/close` now runs the full trained HC-aware head.
Placement is derived from `manifest.layer_owner[FG_LAYER_COUNT-1]`; opening on
another rank is rejected. Internal `blk.47` aliases are a private borrowed
model, not an assignment of the physical owner to rank 7. `fg_model_bind_view`
retains existing tensor spans without cooking them twice or closing the
caller's Vulkan context.

The importer consumes the pinned abliterated overlay: F32 Gemma norms,
Q8 cooked projections and all 512 Q8 native experts, with F32 router/injection/
shared-gate views. Resident cooked weights are **2,772,584,448 bytes**.
It preflights the remaining target-rank residency budget, then allocates
independent F32 draft KV (**4096 bytes per cell**). Production co-residency
still needs a real target-pack qualification.

A cell at position `c` consumes authoritative target HC[c] and the plain
embedding of token[c+1], and proposes token[c+2]. For recursive proposals the
previous draft HC replaces target HC, but rejected draft KV must subsequently
be truncated and repaired using authoritative target rows. The new dense
attention uses 24 query heads, 2 KV heads, dimension 256, causal prefix masking,
Q/K norm and partial rotary, and sigmoid gating. It never reads future rows.
This head modifies no target GDN, PLE or QSA state.

Three warmed runs use actual weights, an actual BF16-source token-198
embedding, changing **synthetic** final HC, and draft cells 16..47. Each runs
16 warmups and 32 timed cells. Native core timing includes its existing
CPU top-10/dispatch/reduction synchronization. A second timer includes the
entire 248320-token Q8 vocabulary projection and CPU argmax on **one GPU**:

| Run | Native core ms/draft | Single-GPU full-vocabulary ms |
|---:|---:|---:|
| 1 | 3.063 | 5.748 |
| 2 | 2.983 | 5.698 |
| 3 | 3.140 | 5.726 |

These are draft execution costs, **not measured acceptance or a ring TPS
result**. The monolithic vocabulary timer is not the existing production
distributed head/fanout cost. It must not be charged as zero, nor confused with
the timing of an already integrated distributed proposer. Mean core cost is
3.062 ms, mean local full-head cost 5.724 ms (8.786 ms combined). No operating
point was changed: sysfs listed 1000/1500/2000 MHz with idle 1000 MHz selected
before/after; these observations are not a loaded fixed-2000-MHz measurement.

Independent NumPy computation reads the original fused HF-order overlay,
not FG's cooked aliases. It validates pre-FC branches, both HC reads/injections,
attention, routing and Q8K activations, expert intermediate activations and
reduction, final HC, contracted hidden and full-vocabulary logits. On the
first cell the maximum scaled errors are **2.87e-6 HC**, **1.84e-5 contracted**;
vocabulary KL(CPU-reference || GPU) is **3.10e-12**, with identical argmax.
This is **synthetic architecture/import parity**, not BF16-vs-Q8 model
quality and not the MTP-on/off target-greedy equivalence gate.

The oracle exposed two tightly coupled defects:

* Small-microbatch owner scratch sizing omitted fixed QSA selection storage
  before the live GR/MoE tail. The constructor now sizes the actual tail as
  well as the attention family, allowing native microbatch 4.
* Both old Q8 expert gate/up paths used the wrong Q8K activation address:
  a one-word instead of two-word header, and a global sub-block index on top
  of a per-block base. The fused path now uses `block_base+2+lane*8`; the
  generic path uses `2+lane*8` in its staged current block. Existing
  GPU-vs-GPU tests shared the header error. The regression now spans two
  256-value activation blocks and checks both paths against scalar CPU
  dequantization/dot/SwiGLU, in addition to fused/legacy parity. Before the
  fix, native expert-mid scaled error was 15.7 and final-HC error 1.65.

At this execution milestone, no competing N=1 GEMV performance rewrite was made. The roofline report at
`fg-wt/roofline/results/hillclimb/ROOFLINE.md` was read; the only existing
arithmetic-kernel edits here are the Q8 expert correctness fix above.
The previous Q4 six-layer small-N ratios remain **1 / 1.519 / 2.199 / 2.505**;
this native-head work is not a new small-N performance measurement.

For a clearly conditional engineering projection, take a **future qualified**
40–45 TPS serial baseline, the **target** N=4 ratio 1.8, k=3 and published
Strata E=2.4–3.2. Charging the measured 3.062-ms core and an **ideal**, unmeasured
four-way division of the 5.724-ms local vocabulary cost gives **41–60 TPS before
fanout/commit overhead**. Charging the measured monolithic vocabulary cost
instead gives 34–48 TPS. Neither is the current ring's throughput: today's
N=4 block ratio is 2.505, the ring ratio/fanout are unmeasured, and there is no FG
acceptance sample. These scenarios identify integration/optimization work;
they are not limits on MTP. Never combine N=2's ratio with k=3's expected yield.

The dense-attention CPU oracle covers 16 causal rows (max absolute error
5.53e-8), including hostile future KV. Native last-cell truncate/rebuild is
byte-exact in all three runs. Borrowed tensor/context lifetime and wrong-owner
tests pass. Optional per-head trace callbacks add diagnostic readbacks only
when explicitly enabled by the benchmark's `--dump-first`; no new environment
flag or default serving path is introduced.

Reproduce after exporting the pinned overlay and a shared head/embedding:

```
make -j8 flash-gordon tools/bench-mtp-head tests/test_fg_vk
./tools/bench-mtp-head --self-test
DS4_REMOTE_TEST_FILTER=expert_decode_fused_q8_gates ./tests/test_fg_vk
./tools/bench-mtp-head --overlay native-q8.gguf --head-q8 shared-head.q8 \
  --embedding-f32 embedding.f32 --warmup 16 --iterations 32 \
  --dump-first .build/native-first.bin
OPENBLAS_NUM_THREADS=4 python tools/mtp_head_reference.py \
  --overlay native-q8.gguf --head-q8 shared-head.q8 \
  --dump .build/native-first.bin --output .build/native-reference.json
```

`tools/mtp-head-build.sh` and `mtp-head-measure.sh` provide the .51 recipe,
including source/input/binary hashes, raw timings, CPU trace comparisons and
clock observations. `mtp-head-export.sh` and `mtp-head-shared.sh` fetch only
required pinned source ranges; the large downloads stay on the blade.
See `results/hillclimb/MTP_NATIVE_HEAD.md` and the JSON/evidence archive.

### What is needed to measure real acceptance

No production final-HC capture is available; .42–.49 remain reserved for the
A/B agent. A partial-layer output is **not** a trained-head final-HC substitute.
Three available blades cannot turn partial-layer states into legitimate
acceptance data. Supply either a production-ring lease or full-target captures
covering **prompt warmfill plus greedy continuations** for chat and code:

* pre-final-mixer 4x2560 F32 HC for every prompt/continuation position;
* token IDs, actual next-token embedding vectors (or the matching embedding
  pack), target greedy next-token labels and prompt/decode boundaries;
* positions/session identity and exact pack, binary and kernel hashes.

Cold draft KV on a decode-only suffix must be labeled cold, not normal
acceptance. Once captures exist, warm the native KV with true prompt pairs,
measure k=1..3 conditional prefix survival, stop on first mismatch, and
truncate/repair from true target rows. Published Strata E=2.4–3.2 remains only
the labeled external estimate. End-to-end greedy identity still requires the
causal ring integration; the synthetic head oracle does not satisfy it.

## 7. Removing the draft's host-sync and monolithic-head regressions

The old 8.786-ms result above is historical, not the proposer that should be
integrated. `fg_mtp_head_step_replay` records GPU router/top-10, shared and routed
experts, their reduction, HC writes and the final mixer into one static graph.
Its private core slot is separate from the owner replay slots; vocabulary
replay is separately claimed. The multisession layout below supersedes the
original slot numbering. A persistent position tensor drives causal
attention and RoPE without re-recording. The first call records; warmed calls
record **zero** new dispatches and submit **once**, including the HC/embedding
staging copies and the graph in the same queue submission/fence. The profiled
dynamic path is diagnostic, not the static-replay latency measurement.
Staging first drains deferred static producers as well as dynamic work;
otherwise a held target-owner graph could leave the copied HC stale. The
private-slot test explicitly queues that dependency.

The original core spent about 0.85–0.90 ms on the GPU but made ten submissions,
CPU routing/readback and CPU shared/expert combination. Additional fixes:

* Four HC `fc_hidden` streams reuse the production Q8 R8 arithmetic with
  interleaved workgroups for weight-cache locality. A true N-column variant
  crossed activation quantization thresholds and failed the unchanged 1e-4
  head-parity gate; it was rejected. The retained four-stream component is
  byte-identical to four N=1 projections.
* The 10240-wide pre-FC norm's generic loop was a roughly 50-us latency bubble.
  Fixed-shape unrolling retains its FMA/tree order, is byte-identical in the
  component test and takes about 7 us in the diagnostic profile.
* The final 10240→320 contraction reuses production's eight-way split Q8
  projection plus fused scaled-SiLU reduction, rather than the slow generic
  narrow-output projection. Whole-head reference tolerance is unchanged.
* Native dense attention now partitions the causal prefix over up to eight
  splits, selected on-device from the current position, and reuses production's
  softmax-state merge/output gate. It writes current KV once and never reads
  future cells. A serial per-head loop was an increasing context-latency
  bubble; the split component is compared through 8192 cells. Fixed 64-way
  splitting was rejected because its mostly-empty workgroups and 64-way
  dynamic merge cost more than the useful work at short prefixes.
* Private-slot descriptor windows no longer collide with the unrelated
  dynamic descriptor-set limit. The multisession integration reserves four
  owner slots per stream, separately from native MTP recordings.
  The shared copy command pool now explicitly permits command-buffer reset.

`fg_mtp_vocab_open/argmax` cooks an **explicit global token map** from the
shared Q8 head. Its R8 GEMV emits per-workgroup argmax, followed by GPU reductions
and an eight-byte final readback, not a full-logits CPU scan. It retains the
producer's hidden-buffer view, adds a device-write visibility barrier and
re-records only if that storage changes. Negative-only tails, global-ID ties,
nonfinite propagation, two reduction stages and 2560-wide scalar dot oracles
are tested. The ordinary R8 kernel, native FC and fused head share one GLSL
body; this does not fork roofline's Q4/Q5 N=1 GEMV work.
When integrating roofline, carry both Q8 activation-address corrections into
its shared `fg_moe_decode_gate_up.glsl`, not only the `.comp` wrapper; retain
the generic staged correction in `fg_moe_kquant_cooked.comp` too.

Four disjoint 62080-row shards cover **all 248320 tokens** and can map to
production logical ranks 4/0/1/2. Each shard was measured separately on .51;
this is **not** a four-physical-rank head/fanout wall-time measurement.
Native placement still requires the final target owner. Its embedding lookup/
delivery and the contracted-hidden handoff to production head shards are not
implemented. A colocated plain embedding copy would add about 675 MB of Q8
weights; do not count that residency or network transport as free.

Strata `src/core/mtp.cpp:389–405` optionally gathers `draft_vocab.bin` rows
from the shared head. The read-only local subset has 106299 unique IDs,
SHA256 `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.
Its `tools/draft_vocab.py` warns that an English/code-only subset destroyed
Chinese acceptance. **No subset was copied or used here.** Tied input/output
weights do not eliminate the output GEMV. The fused argmax API is a greedy
proposal primitive; it does not supply the complete q needed for stochastic
rejection sampling. A successor draft needs its predecessor's token/embedding,
so no unsupported whole-core/head overlap speedup is claimed.

### Native precision candidates

`q4_quantize.py --mtp` is experimental; `--mtp-experts-only` preserves every
common Q8 projection/router/injection and changes only routed experts.
`--q5-dense` selects Q5_K gate/up; the 640-wide down projection uses Q5_1
(**24 bytes per 32 weights**, not 20). GGML row-size ABI validation precedes
output allocation. The importer accounts for each actual resident format.

| Candidate | Synthetic full-vocab KL(Q8 || candidate) | First top-1 | Disposition |
|---|---:|---|---|
| All eligible common/expert Q4/IQ4 + Q5_1 down | 0.283014 | changed | rejected |
| Routed Q4_K gate/up + Q5_1 down, common Q8 | 0.00539410 | equal | rejected at unchanged 0.005 gate |
| Routed Q5_K gate/up + Q5_1 down, common Q8 | 0.00495952 | equal | synthetic gate only |

The last result is **one synthetic HC fixture, barely below the gate**, not
real-text KL or acceptance qualification. Its cooked residency is 1,881,294,848
bytes versus 2,772,584,448 for Q8. It is not a new deployed/default pack.
Independent same-pack NumPy/GPU vocabulary KL is about 3e-12 for Q8 and 4e-12
for the Q5 candidate. Forty-eight-cell slow/replayed comparisons retain the
unchanged 1e-4 scaled-error gate; last-cell KV truncate/rebuild remains byte-exact.
Neither comparison replaces the complete-target MTP-on/off greedy gate.

For measured costs, three-window evidence, source hashes and remaining gates,
see `results/hillclimb/MTP_FAST_HEAD.md`, `mtp-fast-head.json` and its evidence
archive. The Q5 candidate averages **0.868-ms core + 0.620-ms quarter = 1.488 ms**
over short positions 16..47 (window range 1.475–1.504 ms); Q8 totals 1.566 ms.
Candidate local sums after 512/2048/8192 warmup cells are **1.515/1.674/2.202 ms**.
Thus the <=1.5-ms goal is not qualified at longer contexts or with distributed
fanout/embedding delivery. Long-prefix attention/GQA reuse is still a performance
gate; do not report the short-prefix mean as a production draft cost.
Bench CLI example:

```
./tools/bench-mtp-head --overlay native-q8.gguf --embedding-f32 embedding.f32 \
  --head-q8 shared-head.q8 --replay --head-gpu --head-ways 4 --head-shard 0 \
  --warmup 16 --iterations 32
./tools/bench-mtp-head --overlay native-q8.gguf --embedding-f32 embedding.f32 \
  --replay --compare-reference --warmup 8 --iterations 40
```

`mtp-fast-build.sh`, `mtp-fast-final-probe.sh` and `mtp-fast-measure.sh` reproduce
the tests, rejected/retained candidate gates and final measurements under the
allowlisted fleet wrapper. No new environment flags or hardware settings
are required. Serving stays default-off/fail-closed pending real-text precision/
acceptance, embedding/head fanout, causal ring transactions and same-kernel
greedy equivalence. The verifier ratios remain 1 / 1.519 / 2.199 / 2.505.

## 7. What is tested and what remains

* Host synthetic autoregressive model: k=1..3, every rejection position,
  output caps, EOS and injected verify/replay/commit errors. Speculative output
  tokens and retained mock GDN/PLE/KV state equal ordinary greedy decode.
* Actual six-layer Vulkan block: host and GPU snapshots, discard a four-row
  causal verification, restore and replay each retained prefix 1..4.
  Compare complete GDN state and last output bytes against serial decode.
  Change speculative MRoPE triples to prove rejected positions are reusable.
  Cover all four QSA page-boundary offsets. This tests exact rollback, not
  bitwise equivalence of prefill arithmetic and decode arithmetic.
* QSA truncation now repairs durable partial-page count/CRC and file frontier,
  reconstructs the retained tail, clears rejected MRoPE-written flags and
  unpins suffix pages. Prefix records remain append-only; suffix GPU bytes
  outside the frontier are inaccessible and overwritten before becoming visible.
* Source tool tests: ignored HTTP ranges rejected without reading the body,
  Q8 round-trip/zero blocks, raw F32 norm preservation, GGUF sizes and
  no overwrite of an existing overlay.

The .51 validation passed the full `-Werror` build, flag registry, MTP/sampler/
depth-B host suites and all **16/16** GPU rollback cases; see
`results/hillclimb/mtp-validation.json`. The three exporter tests passed locally.
The remote `/home/xander/hc-mtp/` experiment directory was removed after
downloading the evidence; no service or governor configuration was changed.

Reproduce from a Linux checkout with Vulkan build dependencies:

```
mkdir -p .build
make -j8 flash-gordon tools/bench-decode-fusion tests/test_mtp tests/test_sampler tests/test_decode_batch
./tests/test_mtp
./tests/test_sampler
./tests/test_decode_batch
bash tools/check-flags.sh
./tools/bench-decode-fusion --q4 --batch 1 --warmup 5 --iterations 2 --windows 1 --rollback-test
./tools/bench-decode-fusion --q4 --batch 4 --warmup 64 --iterations 16 --windows 3
./tools/bench-decode-fusion --q4 --small-n --batch 1 --small-test --rollback-test --warmup 4 --iterations 2 --windows 1
./tools/bench-decode-fusion --dense-only --dense-type 8
python -m unittest discover -s tools -p test_mtp_source.py -v
python tools/mtp_source.py --audit results/hillclimb/mtp-source.json
# Large ranges: execute on the blade, not through Windows:
python tools/mtp_source.py --audit mtp-export.json --pack mtp-q8.gguf
```

`--dense-only` tests all supported formats without constructing the owner
fixture; `--dense-type` selects GGML type 0, 8, 12, 13, 20, 30 (BF16 index)
or 31 (BF16 router). Unknown selectors fail rather than silently testing
nothing. `tools/mtp-small-qualify.sh` is the blade-specific build/oracle,
shader-statistics and two-seed replay measurement driver for `.51`.

Export requires NumPy; audit does not. Fleet scripts use only
`/home/xander/hc-mtp/`, and the PowerShell wrapper forwards the allowlisted
target set explicitly even when its default is used. An early wrapper call
omitted that forwarded default and attempted transfers to other blades;
those transfers failed before any file/script execution there. The wrapper
was fixed immediately; no production deployment or service was changed.

Remaining work before even an eight-rank correctness qualification:

1. Qualify the linked native head integration with the final target owner,
   actual next-token embedding, distributed shared head and prompt KV warmfill/
   repair. Qualify real-context Q8 quality, not only synthetic import parity.
2. Finish tuning and qualifying the dedicated small-N verifier, including
   every rank's service and output-head costs. Replace rejected-prefix
   whole-block replay with cached-projection recurrence/conv commit or
   per-position state checkpoints before treating commit overhead as cheap.
3. Qualify causal wire work/results, deferred page publication, all-rank
   prefix transactions, history/ngram cursors, cancellation/EOS and logits.
4. Connect stochastic distributions and per-prefix penalty histories before
   speculating on temperature requests; until then they use the exact serial path.
5. Real chat/code native acceptance and causal-N1 reference greedy identity
   have passed (see the capture report at the top). Repeat greedy identity on
   the linked small-N serving path, then fleet A/B and long-context gates in
   a separately granted exclusive lease. The offline integration below does
   not extend the completed capture lease.

## 8. Multisession serving foundation (original base `2d0f8a7`)

`cuppaxanax/mtp-serving` retains the published capture branch unchanged and
integrates its foundations onto the promoted two-session runtime. The foundation
was originally default-off/fail-closed; the linked orchestration now described
at the top is rebased onto `7381f93`. No ring lease is requested on the
strength of component tests.

* Owner static slots 0..3 and 4..7 belong to separate sessions. Small-N cache
  keys, shape changes, failures and N=1 transitions invalidate only that
  session's partition. Native core slots are 8/10; 9/11 are prefix-KV warm
  recordings (or the standalone vocabulary benchmark, never both owners).
  Static descriptor-layout allocation now covers all twelve slots instead
  of incorrectly reusing a smaller dynamic-pool layout array.
* `fg_mtp_head_fork` shares immutable model/weights once, with independently
  allocated executors, scratch, KV/frontier and replay recordings. Either
  session can close first. A second short-context session costs approximately
  **5.2 MB**, not another **2.77 GB** of Q8 weights.
* `fg_mtp_head_warm_replay` computes only the normalized input mix and K/V.
  It avoids Q/output projections, attention, experts and vocabulary work.
  The .51 short-prefix mean is approximately **0.27 ms/cell**. This is prefix
  preparation/repair, **not the complete draft cost**. Repair with authoritative
  HC remains necessary after verification even for accepted native proposals.
* Optional `fg_mtp_head_open_windowed` separates absolute RoPE positions from
  physical KV indices. A window of W retains W+4 physical cells; speculative
  k<=3 work cannot overwrite keys needed after rejection. Truncate refuses
  rewinds beyond retained slack; an explicit reset permits a new prefix at
  an absolute position. W=8192 uses 32.016 MiB of KV per session instead of
  1 GiB at 262K. This is a **draft approximation**, not a target KV change,
  and needs long-context acceptance qualification before serving selection.
* `fg_mtp_policy` is a per-stream rolling conditional-acceptance/cost policy.
  It compares k=0 serial with k=1..3 expected committed tokens per millisecond,
  requiring a speculative estimate to beat measured serial by0.5%. The first
  adaptive token measures ordinary target-only service, excluding native
  advancement. Conditional acceptance uses1/16 EWMA and never observes
  positions after the first rejection; prices use complete round time.
  Optimistic cold acceptance priors (.92/.87/.86) are hypotheses, not a
  confidence guarantee or prompt-text classifier. One extreme first-shape
  compile stall (>3× its estimated price) is excluded from price learning;
  actual elapsed time is still logged/charged, and recurrent slow work is not
  excluded. Ordinary baseline learning also resists a first-use cold stall.
  Forced periodic speculative probes are removed. A losing stream latches
  ordinary N1 and skips native ADVANCE RPC/GPU work until a successful new
  prefix plan synchronizes native state. Exact-prefix continuation stays
  serial, preventing drafts from stale native KV. A cold/new conversation
  resets the acceptance prior, retaining observed prices and serial service;
  a reused conversation keeps its learned acceptance. A one-token
  tail uses N1; fixed-width research overrides still bypass policy selection.
  Eight-rank CPU doubles cover both private slots, no-native-RPC fallback,
  exact-prefix latching, prefix reset/resumption, cancellation and EOS.
  **Chat≥1.0× and retained code gain remain measured serving gates, not claims
  established by this policy formula or these correctness doubles.**
  Runtime selection now uses a private policy per stream; publishing a burst
  follows target commit rather than the draft execution.
* The QSA page-service guard is now tied to its actual session namespace,
  rather than assuming slot zero. Restore drains that namespace's pending
  writes before truncation and repairs only its guard frontier.

Offline .51 gates: `-Werror` release build, MTP adaptive-policy tests, existing
multisession/fabric/API/session tests, N=2/3/4 six-layer output/state byte
identity, causal masking, retained-prefix rollback, and interleaved native
sessions with KV-only repair all pass. The bounded attention scalar oracle
covers three ring wraps, nonzero absolute start and a rejected three-cell
suffix (maximum absolute error about 5.04e-8).

Reproduce with `tools/mtp-serving-validate.sh` through the allowlisted fleet
wrapper. The script raises only its own process's pinned-memory limit so
both QSA io_uring namespaces execute rather than skip. It does not change
governors, services, launchers or production ranks. Timing and exact receipts:
`results/hillclimb/MTP_SERVING.md`.

### Verification physics and the next candidate

`results/hillclimb/MTP_VERIFY_PHYSICS.md` records exact six-layer packed
bytes, equal-prefix kernel attribution, synthetic per-layer expert unions,
and reproducible geometry/A/B experiments. The saved real-text HC capture
does not contain per-layer target routing; those unions cannot be recovered
from final HC. Opt-in `--mtp-capture-dir` now records global top-10 target
IDs per layer and position, with coverage, on every owning rank. Aggregate
all rank archives before `tools/mtp_physics.py --captured-routes DIRECTORY`.
No new production environment flag is introduced.

The candidate uses true N-column compact projections, four-pair weight reuse
in expert down, and shader-local GDN state across causal N updates.
The global recurrent buffer is loaded/stored at round edges, but the compiler
spills the local cache: physical state traffic still needs improvement.
Recurrence arithmetic is sequential. Component and rollback
identity do **not** qualify the newly linked native serving/ring path.
`--experimental-mtp` remains default-off and requires explicit sidecars.
The numerical-double two-host orchestration gate above now passes; trained-
model serving still awaits the separately granted eight-rank lease.

The restore/watchdog guard is pinned to the subsequently promoted
stack-1/adaptive runtime `31508ee`, release `20261016-abq-stack-adaptive`,
sessions 2/wide 1, its binary and shader-tree hashes, and the complete explicit
roofline/local-ngram/submit-group recipe in `DEPLOY_CURRENT.md`.
`2d0f8a7` is the old development base and historical qualification restore target,
**not current production**. No current launcher or production process was
changed while preparing these offline guards; a new lease remains required.
