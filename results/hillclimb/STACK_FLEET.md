# Stack-1: eight-rank production qualification

## Decision

**Promote the full stack, mix-04, sessions 2.** All six primary metrics passed
the predeclared five-pair policy, quality passed, and five primary paired
median benefit ratios exceeded 1.03. No roofline/rest bisection was needed.
**Tradeoff: secondary cached-prefix TTFT is 4.35% slower by paired median
latency ratio.** Secondary metrics cannot veto under the user's new rule.

This is the operational promotion policy, not family-wise proof of
noninferiority. Every primary's *marginal* 95% CI also happened to clear the
0.97 lower bound. The 32K prefill median win is not CI-certified improvement:
its interval includes 1. Historical three-pair/t-guard reports are unchanged.

## Integration and immutable identities

Native build source `31508ee` merges current adaptive production `2d0f8a7`
with roofline `e073c5d`, latency `8c4f9be`, host-overhead `198583f`, and
the three adapted MTP Q8 activation-address corrections from `a6344e6`.
Stack branch tip `7381f93` only adds quality-helper build-product ignores.
No full MTP or fusion serving integration was added.

Merge resolution retained per-session static bindings, the adaptive
dynamic/static descriptor-pool limit distinction, the roofline dispatch
ledger and the active-batch grouping guard. Grouping can use the independent
session bindings rather than silently restricting its benefit to session 0.
New experimental flags remain default off in source.

| Identity | A: previous production | B: selected stack |
|---|---|---|
| Release suffix under `/home/xander/flash-gordon-live/` | `20261016-abq-multisess` | `20261016-abq-stack-adaptive` |
| Source | `2d0f8a7` | `31508ee` |
| Server SHA256 | `de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58` | `02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0` |
| SPV-tree SHA256 | `cf403d9d6a68c7cf9b9304af2937365fd210a91b12c52039c1536eed2a1d22b6` | `5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633` |
| Source-matched quality helper SHA256 | `9774aa189e260f794675c3b614c4aca697ea5ee8bc06cb3f4f747ae520afdf0d` | `c5972ffd22bb261383986829773173fdd2e7a688c31bed1853e81af52af2eced` |
| Sessions / wide | 2 / 0 | 2 / 1 |

Both use the sealed
`/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm`,
SHA256 `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
Neither the previous production release nor either sealed/original pack
was modified. New helper/build releases were staged separately on all eight.
Only .42–.49 were accessed; no clock/governor/hardware changes were made.

Complete selected recipe, on **all** owners:

```text
FG_ROOFLINE=2047
FG_ROOF_DENSE=19
FG_ROOF_QSA=3
FG_ROOF_EXPERT=33024
FG_ROOF_HC=96
FG_ROOF_RECURRENT=3
FG_ROOF_GDN=1
FG_NGRAM_LOCAL_CACHE=1
FG_EXPERT_Q8_WIDE=1
FG_DECODE_SUBMIT_GROUP=2
```

The coordinator retains its existing `FG_RING_TRACE=1`. No intrusive decode
profiler, fusion, MTP, ring/fabric/generation profiling or preload shim was
enabled. Model name, 262144 context/index capacities, QSA hot=0, per-session
cache=700 MiB, and `--sessions 2` are unchanged.

## Validation and quality

Native server, every shader and both source-matched helpers built successfully.
Focused CPU sampler/chat/runtime tests, QSA boundary/cache tests and the
adaptive multi-session/API/pipeline/session suite passed. Offline harness,
identity/reuse, exact-control, pin-render and compact-evidence tests:
**25 passed**. Effective LF transport/pinned shell sources parsed successfully.
Windows checkout line endings are normalized before transport and pin hashing.

With serving stopped, `.42` passed all eight RADV selectors:
`roof_controls`, `roof_dense_exact`, `roof_qsa`, `roof_hc_up_mix`,
`roof_hc_down_inject`, `roof_gdn_gate_controls`,
`expert_decode_fused_q8_gates`, and `submit_group`.
These are component evidence, not a substitute for the real model gate.

One combined full-stack quality comparison, including wide, passed:

| Full-vocabulary gate | Measured | Limit |
|---|---:|---:|
| Frontiers / vocabulary | 480 / 248320 | >=400 / 248320 |
| Mean KL | 0.000246434840116538 | <=0.01 |
| p99 KL | 0.004250554284797628 | <=0.05 |
| Top-1 agreement | 479/480 = 99.7917% | >=97% |

Frozen corpus file SHA256:
`8087c73149ea3029e910eddd7e54b2b54d32d83876d467c7be6f95d14aa7a9f1`;
semantic corpus SHA256:
`153b30d2cc11f7488615b4add737a82b06a3c419e751e5649cc1b021a74ad2ca`.
Full teacher-forced frontiers ran over the actual eight-rank model with fresh
single-logical-session helper runtimes; this qualifies arithmetic, not
multi-session scheduler isolation. Serving performance used sessions 2.
All owners' shader trees were checked and receipts bind both binaries,
helpers, trees, complete flags, pack and corpus.

Global A/B greedy hashes differ on `capital`; numeric differences are reported
and covered by KL, not silently declared exact. Submission grouping has a
separate exact gate: otherwise identical group-zero B was freshly deployed
twice and all five canonical hashes matched. All 25 canonical B cases across
the five grouped performance arms matched that same-kernel control.

## Fixed paired experiment

Predeclared protocol is in
[`protocol.json`](stack-fleet-20261016/protocol.json).
Five complete pairs, order **AB, BA, AB, BA, AB**; each arm freshly deployed.
Fresh salt per pair, byte-identical salted requests across its A/B arms.
Stable user 0 retains prefix affinity; concurrency uses distinct users 0/1.
Cold 4K/32K reuse must be zero and cached decode reuse is checked explicitly.
All prefix primes actually returned `Ready`.

Performance ran **03:14:20–03:52:31 UTC−07 on 2026-10-16**, about 38.2 minutes
including its final baseline restore. The outer batch also restored A.
No completed pair was dropped, no sample extension/repeat-until-pass was
performed, and the conditional additional roofline-off set was not executed.

Intervals use 20,000 fixed-seed whole-paired-round bootstrap draws, shared
indices across metrics, marginal 95% percentile limits. No Bonferroni or
fixed-df t expansion. Benefit is B/A for TPS and A/B for latency.
Eligibility requires every primary median >=0.97, no CI wholly below 0.97,
quality, and at least one primary median >=1.03.

### All six primaries

TPS except chat TTFT in milliseconds. Spreads are observed min/max;
IQRs and all individual paired ratios remain in the raw result. The benefit
column is the **median of paired ratios**, not the ratio of arm medians.

| Primary | A median [min,max] | B median [min,max] | Benefit [marginal 95% CI] |
|---|---:|---:|---:|
| Chat TTFT | 1650.434 [1607.589,1691.169] | 1601.466 [1530.055,1648.090] | 1.03269 [1.02267,1.05067] |
| Concurrent aggregate | 37.696 [37.118,38.056] | 43.447 [43.067,43.552] | 1.15258 [1.13169,1.17335] |
| Decode 32K | 25.330 [24.507,25.723] | 28.487 [26.761,28.756] | 1.11106 [1.05648,1.17337] |
| Decode short | 28.888 [27.557,29.149] | 32.449 [30.695,34.818] | 1.11988 [1.05306,1.26349] |
| Prefill 32K | 369.031 [365.500,383.365] | 385.256 [382.081,408.657] | 1.05405 [0.99891,1.06597] |
| Prefill 4K | 325.719 [315.114,330.824] | 330.711 [326.027,333.078] | 1.01488 [0.99623,1.05701] |

All 20 concurrent stream responses emitted exactly 128 actual tokens.
Consequently equal-length normalization did not change these aggregate
samples. The reusable harness still labels normalization an approximation:
the API has no ignore-EOS support and other workloads may terminate early.

### Secondary tradeoff

Cached-prefix TTFT is the sole secondary with median benefit below 0.97:
paired B/A latency ratio **1.043496**, benefit 0.958317,
95% benefit CI [0.948669,0.971448].
Arm medians are 542.686 ms versus 569.476 ms. This is not hidden by the
aggregate improvement and is not a primary veto. See every secondary,
client-wall distinction and IQR in
[`ab-result.json`](stack-fleet-20261016/full/full-wide/perf/ab-result.json).
No 128K/near-262K memory forensic or loaded mixed-prefill requalification
was performed in this task.

## Persistence, rollback and evidence

Pinned operator:
`/home/xander/flash-gordon-live/20261015-abq-operator`.
`launch-current.sh` points to the exact selection represented by
`launch-stack-3bac55ceb9c2.sh`,
`ab-instance-stack-3bac55ceb9c2.sh` and `current.json`.
The launcher verifies body/binary/manifest/tree hashes; new experiment
environment is scrubbed before explicit selected flags are supplied.
Persistence is the original `nohup`/logout-safe semantics, not an automatic
host-boot service.

The complete selected ring was stopped and restarted, with all eight live
inode hashes, exact argv, FG environment, manifest and shader tree checked.
Post-restart canonical greedy hashes matched the group-zero control.
The **actual pinned-B to adaptive-A one-command rollback** was then exercised:

```powershell
pwsh -NoProfile -File tools\ab-rollback-pre-stack.ps1
```

Its eight-owner identity, health, model and real greedy checks passed.
The qualified full stack was subsequently re-promoted and restart-tested
again; final health/model, greedy and genuine two-request verification are
retained with the selected and final receipts. Exact launch commands and
fallback are in [`DEPLOY_CURRENT.md`](../../docs/DEPLOY_CURRENT.md).

Compact evidence is under `stack-fleet-20261016`, with an SHA256 index.
Full vocabulary `.fgl`/`.f32` captures and archives are retained locally under
the recorded `.build\stack-live` paths and hash-indexed, **not pushed as large
GitHub payloads**. Corpus frames, quality rows, source/build/component receipts,
all paired samples, both pinned restart proofs and the rollback proof remain
reviewable. Owned remote capture/staging scratch was cleaned before release;
immutable releases, selected launchers and sealed packs were retained.
