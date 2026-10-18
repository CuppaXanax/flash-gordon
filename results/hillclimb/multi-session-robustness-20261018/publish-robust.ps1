$ErrorActionPreference='Stop'
$work='D:\looking-glass-labs\fg-wt\multi-session-int'
Set-Location $work
$r=Get-Content .build\robust-promotion\result.json -Raw|ConvertFrom-Json -AsHashtable
if(-not$r.complete-or-not$r.lease_released){throw 'Fleet has not been finalized/released'}
$startUtc=([DateTimeOffset]$r.started).ToUniversalTime().ToString('o')
$deadlineUtc=([DateTimeOffset]$r.deadline).ToUniversalTime().ToString('o')
$verifiedUtc=([DateTimeOffset]$r.final_verified).ToUniversalTime().ToString('o')
$selected=if($r.promoted){'PROMOTED'}else{'RESTORED BASELINE'}
$table=foreach($m in $r.metrics){
  '| {0} | {1:F3} | {2:F3} | {3:F4} | [{4:F4}, {5:F4}] | {6} |' -f
    $m.metric,$m.A_median,$m.B_median,$m.benefit_ratio,$m.ci95[0],$m.ci95[1],
    $(if($m.primary){if($m.passes_primary_guard){'PASS'}else{'FAIL'}}else{'descriptive'})
}
$qos=foreach($arm in 'A','B'){
  $q=$r["qos_$arm"]
  if($q.concurrency.trials.Count){
    $t=$q.concurrency.trials[0];$iso=$t.isolated[0];$chat=$t.load.streams[0]
    '| {0} | {1} | {2:F3} | {3:F3} | {4:F1} | {5:F2} | {6:F2} | {7:F2} |' -f
      $arm,$chat.completion_tokens,$iso.client_decode_tps,$chat.client_decode_tps,$chat.ttft_ms,
      $iso.inter_event_p95_ms,$chat.inter_event_p95_ms,$chat.inter_event_max_ms
  }else{"| $arm | not run after the solo gate failed | | | | | | |"}
}
$decision=if($r.promoted){
  'The revised robustness gate passes. The persistent pinned selection is candidate B; the earlier full-QoS NO-GO is superseded only under this explicitly narrower user policy.'
}else{
  "All five fresh pairs completed, but the short-decode paired-median guard failed. The exact original stack baseline A was restored and remains selected. Error: $($r.error)"
}
$short=$r.metrics|Where-Object metric -eq 'decode-short.engine_tps'
$shortRatios=@($r.pairs|ForEach-Object{
  '{0:F6}'-f($_.B.metrics['decode-short.engine_tps'].value/$_.A.metrics['decode-short.engine_tps'].value)})
$shortAnalysis=@"
The failed veto is short decode: engine paired benefit **$('{0:F6}'-f$short.benefit_ratio)**,
below **0.970000**; client accounting independently fails the same point guard.
The five engine ratios are **$($shortRatios-join', ')**.
The marginal arm medians are **$('{0:F4}'-f$short.B_median) B** and
**$('{0:F4}'-f$short.A_median) A** (a ratio near 1.004), but a ratio of marginal
medians is **not** the predeclared paired-median estimand. The CI crosses 1:
this is not a demonstrated statistical regression, yet the explicit point guard
still fails. No outlier was discarded, no threshold changed, and no optional
extra sampling was used to reverse the decision.
"@
$functional=if($r.replay){
@"
* Agent replay: **$($r.replay.pass)**; $($r.replay.messages) messages,
  $($r.replay.prompt_tokens) actual prompt tokens, $($r.replay.reused_tokens)
  reused after a separate two-message title request, without a user hint.
* Injected transport failure: **$($r.transport.pass)**; request status
  $($r.transport.request_status), API PID $($r.transport.before_pid) ->
  $($r.transport.after_pid), next request succeeds.
* Coordinator SIGKILL/watchdog: **$($r.watchdog.pass)**; PID
  $($r.watchdog.killed_pid) -> $($r.watchdog.after_pid), next request succeeds.
"@
}else{
@'
**Not rerun in this window:** agent/title/agent replay, injected worker failure
and coordinator-SIGKILL gates. The driver deliberately failed fast when gate
(a) missed its explicit threshold; the user instructed restoration otherwise.
Earlier incident-window functional results are historical evidence, not a
substitute for fresh candidate qualification, and are not labeled passed here.
'@
}
$longHashes=@($r.pairs|Where-Object{$_.A.cached32k.visible_text-ceq$_.B.cached32k.visible_text}).Count
$text=@"
# Multi-session robustness qualification — 2026-10-18

## Decision: $selected

$decision

User-authorized candidate: **stack-1 + sealed mix-04 + sessions 2 + source
``11c7e39`` + ordered watchdog**. The runtime C tree is unchanged from
``182d47e``; ``11c7e39`` additionally contains tests, controller/deadline support
and the memlock-corrected unit. Production shaders, sealed weights, manifest,
submission grouping and explicit stack flags are unchanged. This is not a
numeric-kernel or quantization experiment.

* A release: ``$($r.A.release)``.
* A binary SHA256: ``$($r.A.binary_sha256)``.
* B release: ``$($r.B.release)``.
* B binary SHA256: ``$($r.B.binary_sha256)``.
* Manifest SHA256: ``$($r.B.manifest_sha256)``.
* Shader-tree SHA256: ``$($r.B.shader_tree_sha256)``.

## Predeclared gate and paired results

Five **new** fresh-salt alternating AB/BA pairs; none of the previous prototype
samples is pooled. Sustained decode-TPS requests use 64 actual tokens (minimum
32 enforced); TTFT probes and one-token cold primers are accounted separately.
The existing 4K/32K battery is retained, with a cached 32K SSE TTFT continuation
added immediately after its cached decode using the exact generated assistant
text. Five canonical greedy cases per arm per pair compare against production.

Protected metrics: short and 32K decode, engine and client; short chat TTFT,
cached-prefix TTFT and cached 32K TTFT. Benefit is B/A for throughput and A/B for
latency. Every protected paired median must be >=0.97, and its marginal paired
bootstrap 95% interval must not lie wholly below 0.97. Fixed seed 1729, 20,000
paired-round median resamples. Lower bounds below 0.97 remain **inconclusive for
strict statistical noninferiority**, even when this operational policy accepts
them. No performance-win requirement, aggregate gate or loaded-prefill QoS veto
is silently reintroduced.

| Metric | A median | B median | Benefit median | Marginal 95% CI | Gate |
|---|---:|---:|---:|---:|---|
$($table-join"`n")

Paired greedy identity: **$($r.hash_identity)**.
Protected solo/TTFT gate: **$($r.performance_pass)**.
All **$longHashes/5** cached-32K SSE continuation texts also match exactly.
The nonstream concurrent metric includes cold request overhead and potentially
three conversation identities competing for two retained slots; it is
descriptive, not a warm two-stream decode-only throughput measure.
It nevertheless drops to **0.5319x** baseline aggregate in this screen. Protecting
the third resident long conversation can force two unrelated short requests
through the shorter slot serially. That cache-retention tradeoff is disclosed,
not used as an additional veto or described as a warm-pair speed measurement.

$shortAnalysis

## Real-workload replay and failure gates

$functional

* Final selected executable/manifest/shader/argv/FG flags verified on all eight.
  Health/models and five nonempty canonical greedy API results verified.
  The watchdog remains enabled/active with ``LimitMEMLOCK=infinity``.
  No host reboot or new 262K capacity qualification is claimed.

The original spontaneous incident's first physical/protocol trigger remains
unproven. The proposed functional gates target the concrete bookkeeping/routing
corrections and service recovery; they were not rerun after the earlier veto
and do not retrospectively identify that lost diagnostic.

## Was loaded cold-32K chat QoS already failing on production?

**Yes.** Production A's chat falls from **28.981 to 8.374 TPS** (0.289x)
during the cold 32K agent prefill. Its p95 event gap rises from **55.93 to
200.07 ms** (3.58x), with a **2283.81 ms** maximum and **2495.2 ms** TTFT.
This already misses a 1.5x gap target on production.

This real concurrent SSE diagnostic starts from a fresh ring, with 128-token
chat/agent ceilings; both streams actually produce 128 tokens. Candidate B's
diagnostic was scheduled after the functional gates and was skipped on
fail-fast restoration. The earlier prototype's 8.42 TPS / 202.47 ms p95 point is
historical, not a new final-source A/B comparison. Actual completion counts,
isolated controls and visible event times are retained. Loaded QoS was **not**
used as this run's veto.

| Arm | Chat actual tokens | Isolated TPS | Loaded TPS | Loaded TTFT ms | Isolated p95 gap ms | Loaded p95 gap ms | Loaded max gap ms |
|---|---:|---:|---:|---:|---:|---:|---:|
$($qos-join"`n")

These are content-event gaps, not token-ID gaps. Small/generic prefill kernels
and already-issued GPU frames cannot be preempted by a coordinator priority
flag. No 1.5x inter-token-gap guarantee is asserted by this robustness release.

## Persistence, lease and reproduction

* Started: ``$startUtc``.
* Independent rollback deadline: ``$deadlineUtc`` (55 minutes).
* Final selected: **$($r.selected)**; promoted **$($r.promoted)**.
* Final verification/release: ``$verifiedUtc``.
* Exclusive lease used **$('{0:F2}'-f$r.lease_elapsed_minutes) minutes** of 60.
* Lease and pause sentinels removed; rollback timer stopped; watchdog resumed.
  No further fleet/API access after release.

The deadline inherits unlimited memlock and preserves serving child processes.
Its root-owned transient command drops to xander with setpriv for baseline
restoration, then resumes the watchdog; it does not run the model as root.
Pinned helper selection is verified before and after qualification.

Local validation: the focused native multi-session target, eleven supervisor
tests, six real PowerShell/local-SSE fixture tests, the one-frame wire variant
and flag audit passed before this lease. No runtime source was edited during it.

The fail-fast restore path initially checked health/models but omitted a real
generation. Before formal release, a separate bounded finalizer completed all
five canonical generations on restored A and updated the actual verification
time/lease duration. It did not restart the clock or run further experiments.
The dated pre-robustness rollback helpers were also preserved on all eight.
Main was **not** advanced because the promotion gate failed; the unrelated dirty
local main-worktree ``ROOFLINE.md`` was left untouched.

Exact historical drivers, raw paired samples, production mixed-prefill SSE
record, all eight identity receipts, final API receipts
and a SHA256 inventory are in
[``multi-session-robustness-20261018``](multi-session-robustness-20261018).
The historical driver is an audit record, **not permission to start another
fleet lease**. Deployment, supervisor maintenance and rollback are in
``docs/DEPLOY_CURRENT.md`` and ``docs/MULTI_SESSION.md``.
"@
[IO.File]::WriteAllText("$work\results\hillclimb\MULTI_SESSION_ROBUSTNESS.md",$text.Replace("`r",''),[Text.UTF8Encoding]::new($false))
