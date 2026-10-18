# Multi-session robustness qualification — 2026-10-18

**Subsequently superseded for deployment only:**
[`MULTI_SESSION_OVERRIDE.md`](MULTI_SESSION_OVERRIDE.md).
The lease authority accepted the open short-decode point and the candidate was
promoted after fresh functional gates. The failed statistical screen below is
unchanged historical evidence.

## Decision: RESTORED BASELINE

All five fresh pairs completed, but the short-decode paired-median guard failed. The exact original stack baseline A was restored and remains selected. Error: Revised solo/TTFT/greedy gate failed

User-authorized candidate: **stack-1 + sealed mix-04 + sessions 2 + source
`11c7e39` + ordered watchdog**. The runtime C tree is unchanged from
`182d47e`; `11c7e39` additionally contains tests, controller/deadline support
and the memlock-corrected unit. Production shaders, sealed weights, manifest,
submission grouping and explicit stack flags are unchanged. This is not a
numeric-kernel or quantization experiment.

* A release: `/home/xander/flash-gordon-live/20261016-abq-stack-adaptive`.
* A binary SHA256: `02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0`.
* B release: `/home/xander/flash-gordon-live/20261018-abq-session-recovery-r2`.
* B binary SHA256: `3405c8976eca580ca2e8690bb336b0ca13e69ab761dbd7bffee590bb280cad17`.
* Manifest SHA256: `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
* Shader-tree SHA256: `5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633`.

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
| cached-32768.ttft_ms | 580.064 | 573.419 | 1.0109 | [0.9864, 1.0275] | PASS |
| cached-prefix.ttft_ms | 550.505 | 539.656 | 1.0150 | [0.9919, 1.0350] | PASS |
| chat.ttft_ms | 1695.822 | 1706.291 | 1.0239 | [0.9939, 1.0353] | PASS |
| concurrent-agent.normalized_tps | 15.780 | 8.362 | 0.5307 | [0.5229, 0.5332] | descriptive |
| concurrent-chat.normalized_tps | 15.849 | 17.078 | 1.0723 | [1.0488, 1.1292] | descriptive |
| concurrent.normalized_aggregate_tps | 31.560 | 16.724 | 0.5319 | [0.5241, 0.5332] | descriptive |
| decode-32768.client_tps | 28.347 | 28.268 | 0.9963 | [0.9479, 1.0767] | PASS |
| decode-32768.engine_tps | 28.448 | 28.365 | 0.9971 | [0.9483, 1.0769] | PASS |
| decode-4096.client_tps | 31.381 | 31.363 | 1.0558 | [0.9525, 1.1579] | descriptive |
| decode-4096.engine_tps | 31.419 | 31.395 | 1.0559 | [0.9526, 1.1576] | descriptive |
| decode-short.client_tps | 32.537 | 32.665 | 0.9661 | [0.8808, 1.0166] | FAIL |
| decode-short.engine_tps | 32.811 | 32.942 | 0.9660 | [0.8800, 1.0168] | FAIL |
| prefill-32768.client_wall_tps | 374.423 | 384.086 | 1.0290 | [0.9712, 1.1012] | descriptive |
| prefill-32768.engine_tps | 374.903 | 384.543 | 1.0289 | [0.9710, 1.1013] | descriptive |
| prefill-4096.client_wall_tps | 326.641 | 331.128 | 1.0565 | [0.9452, 1.0915] | descriptive |
| prefill-4096.engine_tps | 330.009 | 334.554 | 1.0485 | [0.9500, 1.0861] | descriptive |

Paired greedy identity: **True**.
Protected solo/TTFT gate: **False**.
All **5/5** cached-32K SSE continuation texts also match exactly.
The nonstream concurrent metric includes cold request overhead and potentially
three conversation identities competing for two retained slots; it is
descriptive, not a warm two-stream decode-only throughput measure.
It nevertheless drops to **0.5319x** baseline aggregate in this screen. Protecting
the third resident long conversation can force two unrelated short requests
through the shorter slot serially. That cache-retention tradeoff is disclosed,
not used as an additional veto or described as a warm-pair speed measurement.

The failed veto is short decode: engine paired benefit **0.965996**,
below **0.970000**; client accounting independently fails the same point guard.
The five engine ratios are **0.965996, 1.016127, 1.016834, 0.879970, 0.953039**.
The marginal arm medians are **32.9416 B** and
**32.8106 A** (a ratio near 1.004), but a ratio of marginal
medians is **not** the predeclared paired-median estimand. The CI crosses 1:
this is not a demonstrated statistical regression, yet the explicit point guard
still fails. No outlier was discarded, no threshold changed, and no optional
extra sampling was used to reverse the decision.

## Real-workload replay and failure gates

**Not rerun in this window:** agent/title/agent replay, injected worker failure
and coordinator-SIGKILL gates. The driver deliberately failed fast when gate
(a) missed its explicit threshold; the user instructed restoration otherwise.
Earlier incident-window functional results are historical evidence, not a
substitute for fresh candidate qualification, and are not labeled passed here.

* Final selected executable/manifest/shader/argv/FG flags verified on all eight.
  Health/models and five nonempty canonical greedy API results verified.
  The watchdog remains enabled/active with `LimitMEMLOCK=infinity`.
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
| A | 128 | 28.981 | 8.374 | 2495.2 | 55.93 | 200.07 | 2283.81 |
| B | not run after the solo gate failed | | | | | | |

These are content-event gaps, not token-ID gaps. Small/generic prefill kernels
and already-issued GPU frames cannot be preempted by a coordinator priority
flag. No 1.5x inter-token-gap guarantee is asserted by this robustness release.

## Persistence, lease and reproduction

* Started: `2026-10-18T21:29:35.6188252+00:00`.
* Independent rollback deadline: `2026-10-18T22:24:35.6188252+00:00` (55 minutes).
* Final selected: **A**; promoted **False**.
* Final verification/release: `2026-10-18T22:05:49.2949213+00:00`.
* Exclusive lease used **36.23 minutes** of 60.
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
local main-worktree `ROOFLINE.md` was left untouched.

Exact historical drivers, raw paired samples, production mixed-prefill SSE
record, all eight identity receipts, final API receipts
and a SHA256 inventory are in
[`multi-session-robustness-20261018`](multi-session-robustness-20261018).
The historical driver is an audit record, **not permission to start another
fleet lease**. Deployment, supervisor maintenance and rollback are in
`docs/DEPLOY_CURRENT.md` and `docs/MULTI_SESSION.md`.