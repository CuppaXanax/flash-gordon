$ErrorActionPreference='Stop'
$work='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $work
$r=Get-Content .build\robust-override\result.json -Raw|ConvertFrom-Json -AsHashtable
if(-not$r.complete-or-not$r.lease_released){throw 'No finalized lease'}
$verified=([DateTimeOffset]$r.final_verified).ToUniversalTime().ToString('o')
$started=([DateTimeOffset]$r.started).ToUniversalTime().ToString('o')
$decision=if($r.promoted){'PROMOTED UNDER EXPLICIT OVERRIDE'}else{'RESTORED STACK-1'}
$text=@"
# Robustness promotion override — 2026-10-18

## Decision: $decision

The lease authority explicitly authorized promotion despite the earlier
five-pair **0.965996x short-decode median** (95% CI **[0.879970, 1.016834]**).
This remains an **open performance item**, not a reclassified statistical pass.
The earlier 32K decode, all TTFT guards, 25 canonical greedy comparisons and
five cached-32K continuation-text comparisons passed. No new performance
samples are pooled with that screen and no threshold is retroactively changed.

Candidate: unchanged runtime C tree ``182d47e`` / robustness source ``11c7e39``,
stack-1 flags, immutable mix-04, sessions 2, four prefill-only credits per slot,
and ordered watchdog. The prior baseline and all sealed files remain unchanged.

* Release: ``$($r.B.release)``.
* Binary SHA256: ``$($r.B.binary_sha256)``.
* Manifest SHA256: ``$($r.B.manifest_sha256)``.
* Shader-tree SHA256: ``$($r.B.shader_tree_sha256)``.

## Fresh functional gates in this lease

| Gate | Actual observation | Result |
|---|---|---|
| Agent/title/agent replay | $($r.replay.messages) public messages; $($r.replay.prompt_tokens) prompt tokens; separate 2-message title; $($r.replay.reused_tokens) reused tokens; no user hint | $($r.replay.pass) |
| Injected worker loss | Request HTTP $($r.transport.request_status); API PID $($r.transport.before_pid) -> $($r.transport.after_pid); recovery epoch $($r.transport.health.recovery_epoch); next request succeeds | $($r.transport.pass) |
| Coordinator SIGKILL | Exact numeric owned PID checked against executable/manifest; PID $($r.watchdog.killed_pid) -> $($r.watchdog.after_pid); watchdog alone restarts ring; next request succeeds | $($r.watchdog.pass) |

All eight selected executable, manifest, shader-tree, argv and FG-flag
identities are verified. Five nonempty real greedy API generations match the
production reference after the fault/restart checks; health/models verified.
The selected pinned launcher and current.json persist on all owners.
The watchdog remains enabled/active, with ``LimitMEMLOCK=infinity`` and
``KillMode=process``. SIGKILL exercises its cold process-restart path; no host
reboot or new near-262K capacity qualification is claimed.

Original spontaneous incident trigger: still unproven; this qualifies the
specific routing repairs and real transport/process failure recovery.
Physical ring failure invalidates both active computations; clients retry
failed requests. Healthy API listener preservation applies to the worker-fault
case, not coordinator death.

## Loaded-prefill QoS is not this override's gate

Production was directly measured in the previous window: chat **28.981 isolated
-> 8.374 loaded TPS**, p95 visible event gap **55.93 -> 200.07 ms**, max
**2283.81 ms**, TTFT **2495.2 ms**, with a cold 32K agent prefill.
Thus production itself missed the 1.5x objective.
Candidate loaded QoS is not requalified here; soft quanta cannot preempt GPU
work/already-issued frames. The prior screen also shows the cache-retention
tradeoff when three identities compete for two slots. These limitations remain
disclosed in ``MULTI_SESSION_ROBUSTNESS.md``.

## Lease, persistence and rollback

* Started: ``$started``.
* Final identity/API verification and release: ``$verified``.
* Elapsed: **$('{0:F2}'-f$r.lease_elapsed_minutes) minutes of 45**.
* Promoted: **$($r.promoted)**; selected **$($r.selected)**.
* Lease/pause sentinels cleared; independent 40-minute rollback timer stopped.
* No further fleet/API access after release. Follow-up decode investigation is
  offline only, and does not alter the installed candidate.

Exact prior stack-1 rollback helpers are retained on all owners:
``launch-pre-robustness-20261018.sh`` and ``pre-robustness-20261018.json``.
Do not overwrite these dated files with a future lease's current selection.
Pause/stop the watchdog before switching; stop coordinator/workers, copy both
rollback helpers to launch-current.sh/current.json on all eight, start workers
before coordinator, verify generation, then resume supervision.

Raw functional responses, all-eight identity receipts, watchdog journal,
bounded lease driver and SHA256 inventory:
[``multi-session-override-20261018``](multi-session-override-20261018).
Historical five-pair screen:
[``MULTI_SESSION_ROBUSTNESS.md``](MULTI_SESSION_ROBUSTNESS.md).
"@
[IO.File]::WriteAllText("$work\results\hillclimb\MULTI_SESSION_OVERRIDE.md",$text.Replace("`r",''),[Text.UTF8Encoding]::new($false))
if(-not$r.promoted){return}
$old=(Get-Content "$ab\docs\DEPLOY_CURRENT.md" -Raw).Replace("`r",'')
$body=$old.Substring($old.IndexOf('| Item | Current value |'))
$body=$body.Replace($r.A.release,$r.B.release).Replace($r.A.binary_sha256,$r.B.binary_sha256)
$body=$body.Replace('Runtime build `31508ee`; branch tip `7381f93` changes only build-product ignores',
  'Robustness source `11c7e39`; runtime C tree `182d47e`')
$body=$body.Replace('The old baseline needs a full coordinator restart on a'+
  "`n"+'transport fault; listener-preserving recovery belongs to the unpromoted code.',
  'Worker transport faults preserve the API while workers reconnect; coordinator'+
  "`n"+'death triggers the full ordered restart.')
$body=$body.Replace('launch-stack-3bac55ceb9c2.sh','launch-incident-r2.sh')
$body=$body.Replace('ab-instance-stack-3bac55ceb9c2.sh','ab-instance-incident-r2.sh')
$body=$body.Replace('[qualification archive](../results/hillclimb/stack-fleet-20261016/selected)',
  '[override archive](https://github.com/CuppaXanax/flash-gordon/tree/cuppaxanax/multi-session-int/results/hillclimb/multi-session-override-20261018)')
$body=$body.Replace('Clients need stable, distinct top-level OpenAI `user` values'+
  "`n"+'(for example `chat` and `agent`) for resident prefix affinity; omitting `user`'+
  "`n"+'does not provide durable conversation affinity.',
  'Routing uses longest matching public-message prefixes, independent of the optional'+
  "`n"+'`user` hint. Send exact assistant/tool transcripts for forward state reuse.'+
  "`n"+'A system prompt alone is not affinity; shortened replays reset GDN state.')
$head=@"
# Current production deployment

**Current selection: robustness source ``11c7e39`` on stack-1, sealed mix-04,
sessions 2, plus the ordered watchdog.** The lease authority explicitly
overrode the prior **0.965996x short-decode paired median**
(95% CI **[0.879970, 1.016834]**). This remains an **open performance item**,
not a statistical pass; no deployed binary/kernel change was made to hide it.

Fresh functional gates pass: a 44-message approximately 44K agent retains
**$($r.replay.reused_tokens) tokens** across a separate two-message title request;
worker loss fails the request while preserving the API PID and recovers;
coordinator SIGKILL triggers watchdog-only ordered restart. Next generations
succeed, all eight pinned identities match, and five real canonical greedy
outputs match production. Watchdog boot-enabled/active; memlock unlimited.

Production also misses loaded cold-32K chat QoS (8.37 TPS, p95 event gap
200.07 ms); the override does not claim that target is solved.
[Decision, functional receipts and open item](https://github.com/CuppaXanax/flash-gordon/blob/cuppaxanax/multi-session-int/results/hillclimb/MULTI_SESSION_OVERRIDE.md).

Final persistence/API verification: ``$verified``.
Exclusive lease released after **$('{0:F2}'-f$r.lease_elapsed_minutes) minutes** of 45.

"@
$rollback=@'
## Immediate rollback: preceding stack-1 baseline

Pause/stop supervision as above. Stop current coordinator, then workers.
On all eight owners:

```bash
op=/home/xander/flash-gordon-live/20261015-abq-operator
cp "$op/launch-pre-robustness-20261018.sh" "$op/launch-current.sh"
cp "$op/pre-robustness-20261018.json" "$op/current.json"
```

Start workers 1–7 before coordinator 0 with
`bash "$op/launch-current.sh" start`, verify real generation, then resume
supervision. This returns to immutable stack binary
`02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0`,
release `20261016-abq-stack-adaptive`, same mix-04 and sessions 2. It
reintroduces the old routing and listener-recovery defects.
Do not overwrite the dated rollback files during later leases.

The older pre-stack rollback below is historical, not the immediate target.

'@
$body=$body.Replace('## Roll back to pre-stack adaptive production (sessions 2)',
  $rollback+'## Historical pre-stack adaptive rollback (sessions 2)')
[IO.File]::WriteAllText("$work\docs\DEPLOY_CURRENT.md",($head+$body),[Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText("$ab\docs\DEPLOY_CURRENT.md",($head+$body),[Text.UTF8Encoding]::new($false))
