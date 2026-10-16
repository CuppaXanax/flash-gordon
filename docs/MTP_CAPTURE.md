# MTP acceptance lease

This is a **diagnostic capture and correctness release**, not MTP serving
integration or a small-N performance qualification. Do not use the production
owners until the user grants the next exclusive lease. `Plan` is local-only.

The 2026-10-16 lease completed: [measured acceptance, all 24 exact greedy gates,
and sessions-2 production restoration](../results/hillclimb/MTP_ACCEPTANCE.md).

## Frozen workload and measurements

`tools/mtp_capture.py corpus` authors 24 realistic chat and 24 code requests.
They are original curated prompts, **not captured production user traffic**.
The production think-off chat template is used. Twenty eligible cases of each
category must actually generate **259 non-EOS tokens**; four reserves per
category cover early EOS. EOS is never ignored, padded or silently counted.
There are **256 acceptance anchors per case**, with three genuine lookahead
labels beyond each final anchor.

The final-layer owner captures every prompt and generated final-HC row at its
absolute position. A separate coverage bitmap rejects holes, short files or
incorrect session identities. The coordinator records exact token IDs and
exports the **actual deployed** Q8 embedding/shared head, uncooking without
requantization where necessary. Production weights and manifests are immutable.
Capture is gated by the rank-only `--mtp-capture-dir` CLI option; default off.

The standalone .51 native Q8 MTP executor:

* Warmfills the complete prompt using authoritative target HC and next-token
  embeddings, not a synthetic prefix.
* At cell `c`, consumes `HC[c] + embedding(token[c+1])` and predicts `token[c+2]`.
* Recursively drafts three tokens, feeding native next-HC and each predicted
  token's production embedding into the following cell.
* Truncates speculative draft KV to `c+1` and repairs the next anchor from
  authoritative target HC. Rejected native states never become warmfill.
* Uses the full production vocabulary: no English/code subset.

`acceptance.jsonl` preserves all **10,240 anchors** and draft/target IDs.
The strict summary reports chat/code/combined results for k=1/2/3: conditional
acceptance by draft position, accepted-prefix probabilities, every generation
position 0..255, anchored E, and fixed-k policy-walk committed tokens/pass.
Conditional denominators exclude anchors whose earlier drafts were rejected.
These are greedy proposals; no temperature-sampling acceptance is claimed.

## Full-ring paired greedy gate

Four chat and four code cases each run k=1,2,3, giving **24 paired 256-token
checks** against the same release's MTP-off greedy baseline. This gate is
separate from the 40-case acceptance measurement. The verifier is deliberately
the **causal N=1 ring reference**, not the unlinked small-N serving path.
Precomputed native proposals are replayed against real target ring state.

Every cycle snapshots all eight owners, verifies k+1 inputs sequentially under
the ordinary decode kernels, restores on rejection, replays only the retained
prefix and commits before publication. GDN/PLE tensors, QSA state/index frontier
and the page-service guard frontier are restored. Each paired run also injects
one wrong draft and executes the entire rejected suffix, forcing real rollback.
Every output token must equal its baseline ID; tolerance cannot replace this
gate. This does **not** qualify future native-head networking, small-N arithmetic
or serving integration; those must repeat the gate after being linked.

## Prepared artifacts

Use .51 only for prebuilding; no production-owner access is needed:

1. Archive the staged source to `.build/mtp-capture-source.tar`.
2. Upload as `/home/xander/hc-mtp/source.tar` with `Invoke-MTPFleet.ps1`.
3. Run `mtp-capture-build.sh`, then `mtp-capture-package.sh`.
4. Download `release.tar.gz`, `prepared.json`, and `prebuild-evidence.tar.gz`
   into `.build/mtp-lease`. Keep the pinned Q8 overlay on .51 until the lease.
5. Generate/freeze `corpus.tar`; add its SHA256 as `corpus_sha256` in the local
   preparation receipt. `Plan` verifies both release and corpus hashes.

```powershell
pwsh -NoProfile -File .\tools\Invoke-MTPQualification.ps1 -Action Plan
# ONLY after the user grants the exclusive lease:
pwsh -NoProfile -File .\tools\Invoke-MTPQualification.ps1 `
  -Action Run -LeaseConfirmed -LeaseId mtp-acceptance-YYYYMMDD
```

The prebuild includes a Werror C17 release, all shaders, native GPU self-tests,
host transaction/sampler/depth-B tests, capture I/O/coverage tests, analysis
fixtures and the flag-registry check. No new environment flags were added.

The frozen preparation receipt and raw build/GPU proof are
`results/hillclimb/mtp-capture-prepared.json` and
`results/hillclimb/mtp-capture-prebuild-evidence.tar.gz`. Seven analysis tests
and four local-only orchestration tests passed, including successful restore,
restore after an injected native-stage failure, local-only planning and refusal
without a confirmed lease. The release SHA256 is
`3ba71170aef43886793a5f83186e0042c9b8ece4351ae9b85a2a562abb92ca5f`.
These are preparation receipts, **not measured acceptance, full-ring greedy
results or a measured 45-minute completion time**. Only .51 was accessed.

## Budget, restoration and failures

The target is **at most 45 minutes of ring occupation**, with:

| Work | Budget basis |
|---|---|
| 40 × 259-token baselines | ~6.4 minutes at 27 TPS; short prompt prefill extra |
| Native real-HC draft replay on .51 | ~2–4 minutes, not timed on the ring yet |
| 24 × 256-token paired gates | ~4–16 minutes depending on rejection/replay |
| Deployment/loading/transfers | Reserve several minutes |
| Restoration | Dedicated final eight-minute reserve |

The controller stops qualification work at minute 37 (configurable downward
with `-WorkMinutes`). Independent remote
watchdogs restore workers at minute 38 and coordinator at minute 39 if the
client disappears. Coverage not completed before the deadline is **incomplete**,
never reported as a pass. Completion time/acceptance cannot be measured before
the lease; slower-than-budget service fails closed.

Retries must pass the original lease's Unix start time with
`-LeaseStartedUnixSeconds`; rebuilding does not reset the occupation budget.
The initial production attempt exposed a diagnostic units bug: `dispatch_ts()`
is milliseconds, not seconds. The corrected deadline helper and its C regression
test convert seconds explicitly. That attempt generated no qualifying tokens
and restored the selected sessions-2 production before retrying.

Restore is **current production**: adaptive scheduler `2d0f8a7`, sealed mix-04, sessions 2,
wide 0, using the unchanged retained `launch-current.sh`. **Do not invoke
`ab-rollback-original.ps1`**, which selects the older original pack instead.
The controller verifies executable/manifest SHA256, exact argv, pinned FG
environment, no LD_PRELOAD, health/model identity and real greedy chat, including
comparison against the pre-lease sanity answer.

The candidate uses an isolated manifest copy plus read-only weight/tokenizer
symlinks under `/home/xander/hc-mtp/pack`; all QSA and diagnostic files remain
there. Candidate stopping requires the exact numeric PID, executable and
manifest; no process-name kill or forced kill is used. Evidence is downloaded
before owned scratch is deleted. Native .51 scratch is removed last.

Emergency resume of restoration for the **same** lease:

```powershell
pwsh -NoProfile -File .\tools\Invoke-MTPQualification.ps1 `
  -Action Restore -LeaseConfirmed -LeaseId mtp-acceptance-YYYYMMDD
```

The user's production runbook remains authoritative:
`fg-wt\ab-harness\docs\DEPLOY_CURRENT.md`. A hash/environment mismatch aborts
before production is stopped. Do not replace pinned launchers or promote this
diagnostic release.
