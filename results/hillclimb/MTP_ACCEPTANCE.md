# Real-text native MTP acceptance and ring greedy qualification

2026-10-16, `cuppaxanax/mtp`. Production was restored to adaptive scheduler
`2d0f8a7`, sealed mix-04, **sessions 2**, not historical a63a166/sessions 1.
No candidate was promoted. There is **no measured serving-MTP speedup** here.

## Measured workload

* 20 original realistic chat requests and 20 code requests; think-off template.
* Every eligible case generated 259 genuine non-EOS target tokens.
* 256 anchors/case, **10,240 anchors total**, three recursive native drafts each.
* Native abliterated **Q8** overlay, actual deployed Q8 embedding/output weights,
  authoritative final-layer HC, full prompt warmfill and full vocabulary.
* Native speculative KV is truncated and repaired from authoritative HC at each
  anchor. These are greedy, authoritatively warmed proposals, not sampling results.
* No production user traffic was collected; all prompts are curated in
  `tools/mtp_capture.py`. The frozen corpus hash is in the preparation receipt.

### Acceptance

`p1`, `p12`, `p123` mean the first one/two/three draft tokens all match the
target greedy continuation. Conditional columns divide by surviving prefixes,
not by all anchors. k=1 uses the first column; k=2 the first two; k=3 all three.

| Cohort | p1 | p12 | p123 | a1 | a2 given 1 | a3 given 12 |
|---|---:|---:|---:|---:|---:|---:|
| Chat | 71.2500% | 45.8984% | 28.0078% | 71.2500% | 64.4189% | 61.0213% |
| Code | 89.1016% | 78.4570% | 67.3047% | 89.1016% | 88.0535% | 85.7854% |
| Combined | 80.1758% | 62.1777% | 47.6562% | 80.1758% | 77.5518% | 76.6452% |

| Fixed draft policy | Chat tokens/pass | Code tokens/pass | Combined tokens/pass |
|---|---:|---:|---:|
| k=1 | 1.7068 | 1.8812 | 1.7898 |
| k=2 | 2.1339 | 2.6357 | 2.3584 |
| k=3 | 2.3899 | 3.2422 | 2.7516 |

These policy walks include the target correction/bonus token and clip the final
pass to the 256-token window. They are not merely uniform-anchor averages.
The JSON preserves every generation position 0..255, denominator and count.
The earlier Strata-referenced estimate is now supplemented by **measured FG
acceptance**. The remaining verifier-cost gap is an implementation problem,
not evidence against MTP.

## Exact greedy ring gate

**PASS: 24 paired runs, 256 tokens each, 6,144 token-ID comparisons.**
Four chat and four code cases each ran k=1,2,3 against their same-binary
non-MTP baseline. There were **1,010 rollback cycles**, with at least one forced
wrong-first-draft/full-rejected-suffix execution in every paired run.

Verification used the real eight-rank ring's ordinary **causal N=1 kernels**,
all-owner state snapshots, rejected-state restore, retained-prefix replay and
commit. GDN/PLE, QSA frontier and worker page-service guard state are included.
Precomputed native proposals were replayed; native head networking and optimized
small-N ring serving remain **unlinked** and must repeat this gate after integration.
No floating tolerance substitutes for these token-ID comparisons.

## TPS projections, not measured serving results

Formula: `TPS = measured_policy_E / (verify_ratio / serial_TPS + k * draft_seconds)`.

* Final-owner HC file birthtime-to-mtime gives a **capture-inclusive, derived**
  baseline of 24.2801 TPS (chat 24.4740; code 24.0892). All prompts fit one
  128-token prefill chunk. This is not a calibrated kernel timer.
* Prior replay-faithful **six-layer .51** N=2/3/4 ratios remain
  **1.5193 / 2.1991 / 2.5054**, not newly measured full-ring batch ratios.
* Draft assumption: prior Q8 core + one vocabulary quarter = **1.565658 ms**.
  Distributed networking is excluded; this sum is not measured distributed draft
  latency. It assumes the intended four-way head, not the offline full-head replay.

| k | Combined TPS, capture baseline | Combined TPS, historical 27 TPS baseline |
|---|---:|---:|
| 1 | 27.90 | 30.95 |
| 2 | 25.17 | 27.88 |
| 3 | 25.50 | 28.22 |

Conditionally, with future serial **40–45 TPS**, N=4 verify ratio **1.8**, and the
same draft assumption, measured combined k=3 E projects **55.37–61.56 TPS**;
code projects **65.24–72.54 TPS**. Those operating points are **not achieved by
this release**. The small-N verifier and distributed-head integration remain the
speedup work; neither these extrapolations nor the serial oracle is a serving result.

## Restoration, budget and evidence

The first attempt found a diagnostic deadline units bug before qualifying tokens:
`dispatch_ts()` returns milliseconds. The seconds-to-ms helper is now explicitly
tested. Automatic restoration to the new sessions-2 production passed before
retry. Rebuild/export time did **not** reset the original lease start.

Both attempts, rebuild/export, qualification, restoration and scratch cleanup
finished **36.55 minutes after the first controller preflight**, within 45 minutes.
Soft work cutoff was minute 37; independent worker/coordinator restore watchdogs
retained their original minute 38/39 deadlines.

All eight executable/manifest hashes, argv and allowed FG environment passed.
Production binary SHA256:
`de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58`.
Manifest SHA256:
`0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
Health/models/real greedy `OK` passed, including a final check **after cleanup**.
All eight ring `hc-mtp` directories and .51 scratch were removed; .51 absence
was checked twice. No clocks/governors, pinned launchers or production pack were changed.

* `mtp-acceptance.json`: measured per-position summary, projections and provenance.
* `mtp-acceptance-evidence.tar.gz`: raw acceptance rows, 24 greedy receipts,
  both attempts' controller/owner evidence, fixed build/GPU proof and final API check.
* Evidence SHA256:
  `baa608baff8ae1c21d7c3d5b605129c1af6df4a8d1ea80f919c7d96ab30a5e20`.
* Large HC/shared-weight baseline archives and the hash-verified native Q8 overlay
  remain project-local under `.build/mtp-lease`, not in Git, for offline follow-up.
