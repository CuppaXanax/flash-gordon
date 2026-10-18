# Robustness promotion override — 2026-10-18

## Decision: PROMOTED UNDER EXPLICIT OVERRIDE

The lease authority explicitly authorized promotion despite the earlier
five-pair **0.965996x short-decode median** (95% CI **[0.879970, 1.016834]**).
This remains an **open performance item**, not a reclassified statistical pass.
The earlier 32K decode, all TTFT guards, 25 canonical greedy comparisons and
five cached-32K continuation-text comparisons passed. No new performance
samples are pooled with that screen and no threshold is retroactively changed.

Candidate: unchanged runtime C tree `182d47e` / robustness source `11c7e39`,
stack-1 flags, immutable mix-04, sessions 2, four prefill-only credits per slot,
and ordered watchdog. The prior baseline and all sealed files remain unchanged.

* Release: `/home/xander/flash-gordon-live/20261018-abq-session-recovery-r2`.
* Binary SHA256: `3405c8976eca580ca2e8690bb336b0ca13e69ab761dbd7bffee590bb280cad17`.
* Manifest SHA256: `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
* Shader-tree SHA256: `5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633`.

## Fresh functional gates in this lease

| Gate | Actual observation | Result |
|---|---|---|
| Agent/title/agent replay | 44 public messages; 44397 prompt tokens; separate 2-message title; 44398 reused tokens; no user hint | True |
| Injected worker loss | Request HTTP 500; API PID 263332 -> 263332; recovery epoch 1; next request succeeds | True |
| Coordinator SIGKILL | Exact numeric owned PID checked against executable/manifest; PID 263332 -> 263691; watchdog alone restarts ring; next request succeeds | True |

All eight selected executable, manifest, shader-tree, argv and FG-flag
identities are verified. Five nonempty real greedy API generations match the
production reference after the fault/restart checks; health/models verified.
The selected pinned launcher and current.json persist on all owners.
The watchdog remains enabled/active, with `LimitMEMLOCK=infinity` and
`KillMode=process`. SIGKILL exercises its cold process-restart path; no host
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
disclosed in `MULTI_SESSION_ROBUSTNESS.md`.

## Lease, persistence and rollback

* Started: `2026-10-18T22:10:59.9097960+00:00`.
* Final identity/API verification and release: `2026-10-18T22:16:59.0970135+00:00`.
* Elapsed: **5.99 minutes of 45**.
* Promoted: **True**; selected **B**.
* Lease/pause sentinels cleared; independent 40-minute rollback timer stopped.
* No further fleet/API access after release. Follow-up decode investigation is
  offline only, and does not alter the installed candidate.

Exact prior stack-1 rollback helpers are retained on all owners:
`launch-pre-robustness-20261018.sh` and `pre-robustness-20261018.json`.
Do not overwrite these dated files with a future lease's current selection.
Pause/stop the watchdog before switching; stop coordinator/workers, copy both
rollback helpers to launch-current.sh/current.json on all eight, start workers
before coordinator, verify generation, then resume supervision.

Raw functional responses, all-eight identity receipts, watchdog journal,
bounded lease driver and SHA256 inventory:
[`multi-session-override-20261018`](multi-session-override-20261018).
Historical five-pair screen:
[`MULTI_SESSION_ROBUSTNESS.md`](MULTI_SESSION_ROBUSTNESS.md).

## Offline decode investigation after lease release

**Finding: the dual-prefill credit calculation is not in the decode loop.**
Its only runtime caller is `coordinator_prefill_pipeline_ring`
(`src/runtime.c:4233`); the extra 1-versus-4 window decision is reached only
while prefilling with an active peer. `fg_pipeline_observe_prefill` likewise
has only a prefill caller. No decode-token arithmetic, GPU/shader kernel,
sampler, or normal `fg_pipeline_pick`/`fg_pipeline_turn` code changed versus
stack `31508ee`.

Other new work is bounded outside steady decode:

* Socket liveness probes require **no active job** and run at most every
  250 ms (`src/api.c:4309`), not per token or busy-poll iteration.
* Routing is parsed once at admission. Prefix scoring runs for queued entries
  (`src/api.c:4367`), not when the queue is empty. A waiting long request can
  be rescored on successive scheduler turns: that is a separate potential
  loaded-workload cost, not ruled out or measured by this solo CPU bench.
* Transport-ready validation runs when a request finishes.
* The healthy receive path adds a predictable `pipeline->failed` branch.
  Private QSA transport construction is session setup, not per token.

An actual-source CPU microbench compares matching `pipeline.c` **and headers**
from `31508ee` and `11c7e39`: nine alternating pairs in local WSL, `gcc -O2`,
200,000 real coroutine receives or 10,000,000 budget calls per process.
Fabric is an immediately ready mock; this measures neither GPU nor network:

| CPU operation | Stack median ns | Candidate median ns | Paired median delta ns |
|---|---:|---:|---:|
| Complete receive/yield/pick/resume | 1468.534 | 1458.543 | -18.514 |
| Both-prefilling budget calculation | 2.962 | 3.644 | +0.735 |

There is no detected receive-path CPU increase here. The budget's sub-nanosecond
increment cannot directly explain a **1.073 ms/token** paired latency increase,
and it is not called by decode in the first place. These results do not exclude
GPU/per-slot cache behavior, clock variation, sleep/poll phase effects or other
hardware-dependent differences; no new fleet profile was taken.

The old five-pair engine ratios were **0.9660, 1.0161, 1.0168, 0.8800, 0.9530**.
Their paired median is 0.965996, whereas the ratio of arm medians
**32.941580 / 32.810614 = 1.003992**. Thus the record does **not** establish a
repeatable fixed 3% per-token code cost. Short two-second samples, ordering and
unmeasured run-state variability remain confounded. The precise hardware cause
is **unattributed**, rather than invented or declared proven noise.

No performance patch, flag adjustment or binary replacement was deployed.
The credit-loop hypothesis is ruled out as a direct decode cost; the numerical
open item remains. CPU sources, exact compile commands, raw samples and timing
reanalysis are in the override archive. Any follow-up hardware profiling needs
a new lease and a predeclared longer, balanced decode battery.

Main delivery uses a non-force fast-forward push from the clean incident
branch. The unrelated dirty local main-worktree `ROOFLINE.md` is preserved;
that shared worktree is not reset or checked out.