# Concurrent-prefill incident: 2026-10-18

## Scope and causal confidence

Baseline: stack/adaptive `31508ee`, mix-04, sessions 2; immutable release
`20261016-abq-stack-adaptive`, binary
`02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0`.
Candidate runtime: `2cf798a`, integrated with main `7381f93`; same sealed pack,
explicit stack flags, and **copied, unchanged production shader tree**
`5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633`.
Candidate binary:
`aba0f882f180a7585a63922b06475a576dacfb471aca50ca694bcc9a479ba9ba`.

The preserved incident's first error is **not recoverable from its log**:
request cleanup replaced it with the reset failure. A natural synthetic
short-tail/next-quantum transition did not crash. Do not mistake injected-fault
reproduction of the old fatal handling path for proof of the original wire/GPU
trigger.

Two actual bookkeeping defects were found:

* Ring-prefill sends did not mark the session transport pending. A reset could
  regard an unfinished ring operation as reusable. Issued work now stays pending
  until its complete ordered frontier drains; errors poison it.
* Shallow coordinator cloning preserved QSA transport callbacks pointing at slot
  zero's transport state. Slot one's failures could poison the wrong session.
  Each clone now initializes its own transport, callbacks and atomic state.

Frame validation additionally checks source, slot, nonce and sequence. Original
request errors survive cleanup. Failed pipelines forbid resets/new sends, unwind
both coroutine stacks, discard invalid public caches, and reconnect the fabric
without destroying the API listener/model. A physical ring fault invalidates
both active computations, not just the first reporting request; clients must
retry failed generations after recovery. New admissions receive 503 meanwhile.

## Prefix routing and utilization

Final routing (`182d47e`) chooses the longest **common complete-message public
prefix**, independent of `user`; merely sharing a system prompt is not affinity.
A replayed primer or edited history returns to its closest slot and resets it;
only a complete resident prefix can reuse the existing GDN frontier.
An active best match waits. Unused slots beat eviction. Otherwise
LRU applies, except a short request cannot displace the longer conversation
while a free/shorter slot is available or busy. In-flight admitted metadata remains
visible to routing even when generation invalidates the old cached transcript.
Two new unrelated short requests can serialize while protecting a third idle
long conversation; this is a deliberate cache-retention policy, not a ring lock.

The five-pair prototype binary predates this final replay/in-flight follow-up.
It solved the reported agent/title/continuation case but could misroute a
shortened replay, such as the concurrency harness's repeated agent primer.
Unit tests now cover both shortened replays and an invalidated busy-slot cache.
**Do not promote the final source using the prototype's paired measurements.**

Two prefills now use four 128-token frames each, rather than one each. This
fills the steady-state eight-frame ring instead of leaving six credits unused.
`FG_MULTI_PREFILL_ONLY_WINDOW=1..4` controls this; default 4. With a decoder active,
the existing one-frame/soft-service-budget policy and streamed decode priority
remain. Idle-peer prefill retains full microbatch/eight-frame behavior.
Already-issued work drains when limits shrink; this is not hard GPU preemption.

For isolated latency `L`, bottleneck stage service `S`, and `N` tokens circulating,
the ideal per-stream rate is `min(1/L,1/(N*S))`; aggregate is `N` times that rate.
The historical 37 ms / 3.7 ms ledger predicts about **27 TPS each / 54 aggregate**
for two streams, before a new bottleneck. Thus a 0.81x per-stream result is not
implied merely by having two tokens in flight: it indicates queueing, head-of-line
blocking or a larger serialized service than that old average-stage ledger.
That ledger is not a measurement of the current stack pack's maximum service.
There is no new host lock around whole generations.

## Offline tests

`make test-multi-session` exercises routing, sampler/session state, QSA namespace
isolation, the real coordinator/fabric coroutine wire with identity GPU mocks,
and eleven watchdog/controller/service tests. The wire fixture forces:

1. Unequal simultaneous prefills and a short last chunk.
2. The shorter request entering output/decode while the survivor changes from
   128-token to one-token quanta, then regains the idle-peer window.
3. An oversized output result while the survivor has in-flight work: reproduce
   the old poisoned-reset diagnostic, then unwind both jobs without new sends.

`FG_MULTI_PREFILL_ONLY_WINDOW=1 ./tests/test_pipeline_wire` also covers the old
one-frame setting. `bash tools/check-flags.sh` registers/audits the new knob.
The fake output owner waits for the one-token issue, rather than relying on a
timing-sensitive sleep. No spontaneous crash, real GPU parity or fleet capacity
claim follows from these mocks.
All six PowerShell concurrency-harness fixture tests also pass on Windows,
including real simultaneous local SSE streams and insufficient-usage rejection.

## Completed live incident probes

One exclusive validation window began **2026-10-18 20:31:18 UTC**.
An independent systemd deadline was armed for the exact pre-incident selection
at +43 minutes, reserving two minutes of the 45-minute lease for verification.
Manual finalization began before that deadline and stopped its timers;
the cold-start failure and resulting lease overrun are disclosed below.
The original release, pack and prior incident copy remain unchanged.

Concurrent, unequal **cold** prefills without a `user` key completed:

| Request | Prompt tokens | Generated tokens | Engine prefill TPS | Prefill seconds |
|---|---:|---:|---:|---:|
| Long agent | 46,113 | 1 | 328.760190 | 140.263332934 |
| Side request | 1,865 | 128 | 170.706956 | 10.925155245 |

These are whole-request rates, not a paired per-quantum throughput estimate.
The agent's fast solo tail contributes to its rate; do not compare 328.8
directly to the incident's fully contended 53 TPS. The side request completed
while the agent remained resident. A subsequent two-message title-style
request was followed by an agent continuation reusing **46,114 tokens**:
the long conversation was not evicted.

Rank 3 was deliberately stopped during an active generation. The affected
request returned HTTP 500 with the original diagnostic
`pipeline peer 3 disconnected`. The watchdog restarted workers, not the API;
coordinator PID remained **254265**, `/health` returned to `ok`, and
`recovery_epoch` advanced from 0 to **1**. A subsequent API request succeeded.
This is an actual listener-preserving transport-recovery test, not only a
watchdog process restart.

A staging-only error initially computed the pinned instance-script hash before
CRLF normalization. All starts refused the mismatched pin; no unchecked binary
ran. The launcher pin was recalculated from the installed bytes, then the
candidate was started and health verified under the **same original lease**.

## Paired gate and final selection: NO-GO

Five predeclared alternating A/B pairs completed on prototype `2cf798a`, with
fresh salt per pair, 4K/32K solo prefill and cached decode, short decode, chat
TTFT, actual-token concurrent generation, and five canonical greedy cases.
All **25 paired canonical hashes match**. Each sustained generation used 64
actual tokens; a token ceiling was not substituted for observed usage.

The existing 20,000-draw fixed-seed paired-round bootstrap gives marginal 95%
intervals. Benefit is B/A for TPS, A/B for latency. All protected point guards
pass: median benefit >=0.97 and no CI wholly below 0.97. Many intervals still
cross 0.97: **statistical noninferiority is inconclusive**, not proven by an
absence of significant regression. No robust A/B aggregate win is established.

| Metric | A median | B median | Paired benefit median | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| cached-prefix.ttft_ms | 569.431 | 567.148 | 1.0040 | [0.9249, 1.0631] |
| chat.ttft_ms | 1663.146 | 1650.892 | 1.0060 | [0.9360, 1.0126] |
| concurrent-agent.normalized_tps | 15.880 | 15.951 | 1.0073 | [0.9816, 1.0190] |
| concurrent-chat.normalized_tps | 15.811 | 16.033 | 1.0120 | [0.9802, 1.0268] |
| concurrent.normalized_aggregate_tps | 31.621 | 31.901 | 1.0096 | [0.9816, 1.0210] |
| decode-32768.client_tps | 28.473 | 28.460 | 1.0010 | [0.9439, 1.0573] |
| decode-32768.engine_tps | 28.567 | 28.550 | 1.0009 | [0.9436, 1.0577] |
| decode-4096.client_tps | 31.505 | 31.255 | 0.9865 | [0.8524, 1.0636] |
| decode-4096.engine_tps | 31.551 | 31.291 | 0.9862 | [0.8524, 1.0636] |
| decode-short.client_tps | 32.824 | 32.571 | 1.0012 | [0.9608, 1.1431] |
| decode-short.engine_tps | 33.121 | 32.880 | 1.0017 | [0.9621, 1.1448] |
| prefill-32768.client_wall_tps | 382.328 | 383.508 | 1.0003 | [0.8930, 1.0238] |
| prefill-32768.engine_tps | 382.835 | 383.958 | 1.0002 | [0.8927, 1.0238] |
| prefill-4096.client_wall_tps | 326.328 | 325.666 | 0.9918 | [0.9241, 1.1123] |
| prefill-4096.engine_tps | 329.734 | 328.990 | 0.9917 | [0.9287, 1.1067] |

The prototype's additional loaded SSE probe **does not meet the 1.5x gap
objective** during a cold 32K agent prefill:

* Chat: 128 actual tokens, 8.420 TPS versus 28.241 isolated (0.298x retention).
* Chat TTFT: 2510.0 ms versus 1007.6 ms isolated.
* Visible-event p95 gap: **202.47 ms versus 47.71 ms** (4.24x);
  maximum gap **2183.32 ms**. These are content-event gaps, not token-ID gaps.
* Agent: 26.95 client decode TPS after its prefill. Both compared hashes match.

The soft quantum cannot preempt a GPU block or already-issued frames. These
results do not establish whether the remaining gap is dominated by drain
backlog or fixed-cost small-prefill dispatch; further profiling is needed.
Increasing the prefill-only window must not be marketed as solving loaded
decode QoS.

Final replay/busy-slot routing `182d47e` was separately tested from a fresh ring
using the real warm SSE concurrency mode:

| Stream | Actual tokens | Loaded TPS | Retention vs isolated | TTFT ms | Reused tokens |
|---|---:|---:|---:|---:|---:|
| Chat, 64-token context target | 64 | 31.630 | 0.9750 | 926.207 | 113 |
| Agent, 4K context target | 64 | 30.577 | 1.0498 | 834.161 | 4156 |

Aggregate client decode TPS **60.46**, **1.9639x** mean isolated TPS, with
**1.9823 seconds of visible decode overlap**. Both loaded hashes match their
own isolated controls; repeated agent priming preserves the chat prefix.
This is one descriptive trial, **not five paired source-level qualification
rounds**, and it does not requalify 32K loaded QoS or 262K capacity.
Final binary: `3405c8976eca580ca2e8690bb336b0ca13e69ab761dbd7bffee590bb280cad17`.

**Do not promote:** the final routing follow-up is outside the five-pair
prototype gate, loaded-prefill QoS remains weak, and the original incident's
first physical/protocol trigger is unproven. The exact stack baseline was
restored, not historical Q8/pre-stack or a sessions-1 configuration.

## Persistent supervision

`flash-gordon-ring.service` is installed on rank 0 and enabled for boot.
`fg_ring_supervisor.py` uses a dedicated coordinator-only SSH key, forced
rank-control commands, source-IP restriction, strict provisioned host keys and
owned executable/manifest/PID checks. Its nonmutating `--check` passed all eight.
Cold recovery starts workers before rank 0; live transport recovery preserves
the listener, with a full ordered restart fallback after a failed handshake.
An operator pause sentinel prevents interference with future fleet leases.
No private key or credential belongs in the archived evidence.

The cold/boot-style restart test exposed a genuine deployment defect: systemd's
default memlock allowance made the baseline coordinator fail with
`register fixed buffer: Cannot allocate memory`. The unit now explicitly sets
**`LimitMEMLOCK=infinity`**, matching the SSH production allowance. It was
reloaded/restarted; the watchdog cold-started the baseline ring and logged
`ring_ready` with `kept_api:false` at **21:17:37 UTC**. No host reboot was done.

**Lease overrun:** emergency restoration extended the single window to
**47.17 minutes**, exceeding the requested 45 minutes by **2.17 minutes**.
No additional performance experiments were run during the overrun. Restoration
and ownership/API verification were finished before releasing the fleet;
there was no second validation window.

Final verification/release: **2026-10-18 21:18:28 UTC**:

* All eight baseline executable, manifest, shader-tree, argv and FG-flag
  identities verified; mix-04, sessions 2, wide 1, submission group 2 unchanged.
* Enabled and active watchdog; cold restart ordering exercised.
* Health/model API verified; all five real greedy outputs nonempty and
  identical to the baseline canonical hashes.
* Both qualification timers stopped; lease/pause sentinels and upload archives
  removed. Private SSH identity remains only in its retained operator directory.
* Fleet released; no further fleet/API access after release.

Raw paired samples, final warm/mixed SSE records, identity and supervision
receipts, corrected historical drivers, and SHA256 inventory:
[`multi-session-incident-20261018`](multi-session-incident-20261018).

## Exact local and retained-runtime test commands

```powershell
Set-Location D:\looking-glass-labs\fg-wt\multi-session-int
wsl.exe -d Ubuntu-24.04 --cd /mnt/d/looking-glass-labs/fg-wt/multi-session-int bash -lc 'export TMPDIR="$PWD/.build"; make test-multi-session && bash tools/check-flags.sh'
pwsh -NoProfile -File tools\hillclimb.ps1 -ConcurrentGeneration -ChatContextTokens 64 -AgentContextTokens 4096 -ConcurrentTokens 64 -ConcurrentRepeats 1 -PromptSalt final-routing-replay -ResultsDirectory .build\r2-warm
```

The second command requires a new explicitly authorized fleet lease and the
final candidate running; it must not be run against production without one.
Do not repeat the original window or silently pool a new source into its pairs.
Under a new lease, pause/stop the watchdog as documented in
`docs/MULTI_SESSION.md`. The final candidate and pinned helpers are retained on
all eight as release `20261018-abq-session-recovery-r2`,
`$op/launch-incident-r2.sh` and `$op/incident-r2.json`, where
`op=/home/xander/flash-gordon-live/20261015-abq-operator`.
Stop the current pinned instance on all ranks, copy that candidate launcher/JSON
to `launch-current.sh`/`current.json`, start ranks 1–7 with
`bash "$op/launch-current.sh" start`, then start rank 0. All flags/pack hashes
are pinned in those retained helpers. Qualification must include longest-prefix
replay, agent/title/agent reuse, busy-cache protection, warm matched-prefix
decode and explicit third-conversation retention behavior. Restore
`launch-lease-baseline.sh`/`lease-baseline.json` afterward, verify health/greedy
hashes, and resume the watchdog. This is a validation recipe, not permission
to promote the unqualified final runtime.
