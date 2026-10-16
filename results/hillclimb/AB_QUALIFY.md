# Paired 8-rank production qualification

## Decision

**Keep the original production configuration.** No candidate passed every
protected metric, its numerical gate and a corrected improvement gate.
The current-launch runbook is `docs\DEPLOY_CURRENT.md` on
`cuppaxanax/ab-harness`; final restart/identity/API receipts are in `ab-current`.
Only .42–.49 were used. Original binary, weights and manifest were not patched;
no MTP, fusion, CU-routing or other sibling branch was merged.

## Predeclared method

Three fresh-salt paired rounds, deployment order **A,B / B,A / A,B**.
Both arms of each pair use identical serialized cold/continuation/cached bodies;
canonical numerical cases remain fixed. A is always the original release
`/home/xander/flash-gordon-live/20260926-abl2` and original pack.
Full identities, per-arm observations, salts, request hashes, medians, min/max,
IQR and intervals are retained in `AB_QUALIFY.json` and `ab-*/ab-result.json`.

The estimator is the median paired log B/A, with 20,000 fixed-seed whole-pair
bootstrap resamples. Acceptance expands the 95% interval to include a
conservative df=2 paired-log Student-t interval. Its interpretation assumes
independent normal log ratios; three pairs are low power, not distribution-free
95% coverage under arbitrary drift. The guard was committed as `9a64f0f`
before the first complete paired result. Raw bootstrap intervals remain visible.
Throughput requires lower bound >=0.97; latency B/A requires upper bound <=1.03.
An interval crossing a margin is **inconclusive, not noninferior**.
Win intervals reserve 19 metrics × four planned inferential candidates
(Bonferroni 76 comparisons); every protected metric must pass independently.
Tokens, SSE events, layers and within-round requests are not replicates.

**Power limitation, retained at the user's wrap-up instruction:** “with 3
pairs, a df=2 t-guard and Bonferroni across 76 comparisons, this gate can't pass
anything. The original binary scored 12/19 inconclusive against itself.”
The recorded original-pack main-binary control indeed has 12/19 inconclusive
metrics; its A/B binary identities are separately preserved below rather than
represented as identical files. Whole-candidate acceptance is impractically
low-power here. **The gate was not changed after observing results. Redesign
power and an adequately precise control experiment before the next run.**
Inconclusive results are not evidence of a regression or equivalence.

Cold prefill is one non-stream response per arm/context: actual prompt tokens
divided by full HTTP wall, separately from engine prefill TPS. It is **not**
SSE TTFT. Cached follow-on decode uses that exact response's usage/headers;
chat/cached TTFT is time to first visible SSE content. `FG_DECODE_PROFILE`
was off. No engine continuation timing is mislabeled as cold/SSE performance.

The original API has no `ignore_eos`. Concurrent non-stream requests use actual
response usage, raw tokens/s and equal-length normalization retaining observed
queue/non-decode time. Every primary A/B stream emitted **128 actual tokens**,
so the normalization did not substitute a shorter EOS workload. This is not
SSE-event throughput. New schema-2 evidence is refused by the historical
`q4_qualify.py` point-baseline evaluator rather than compared with old events/s.

## Results

NI includes corrected wins; regressions exclude the 3% margin in the wrong
direction. Inconclusive metrics are not counted as passing.

| B vs original A | NI / protected | Regressions | Inconclusive | Corrected wins | Numerical control | Eligible |
|---|---:|---:|---:|---:|---|---|
| Main `a63a166`, original pack | 7/19 | 0 | 12 | 0 | 15/15 original hashes | No |
| Main, mix-04 | 13/19 | 0 | 6 | 0 | Existing manifest-bound KL passes; 12/15 original hashes | No |
| Main, mix-04, `FG_EXPERT_Q8_WIDE=1` | 15/19 | 0 | 4 | 0 | 12/15 same-pack hashes; capital differs | No |
| Repaired integration, mix-04, `--sessions 2` | 2/11 | 5 | 4 | 2 | 15/15 same-pack hashes | No |

Main release: `20261015-abq-main`, binary
`1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789`.
The final integration was built from `79b0efc`, release
`20261015-abq-multi-fix`, binary
`062fcd70c96006a5eb3d762533f7ae131afd80b0f26832593131a8c8d9663e7f`.
Later integration commits change harness/docs only.

Mix-04 manifest SHA256:
`0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
The prior full-vocabulary receipt covers 480 frozen frontiers, vocabulary
248320: mean KL **0.003128578**, p99 **0.031049242**, top-1 **98.9583%**.
Thresholds remain mean <=.01, p99 <=.05, top-1 >=97%, >=400 frontiers.
This is reused sealed-pack evidence, not a new quality capture.
Mix spelling changes are reported, not rejected merely for weight-format hash
identity. The wide flag additionally changes the `capital` hash in all three
rounds relative to the same-pack control. Existing mix KL does not certify
that arithmetic change; no flag-specific KL was measured. It remains
numerically **unqualified**, not proven semantically bad. A future legitimate
FP-reassociation candidate can use component/model numerical gates.

The original main control's 128K engine prefill interval is approximately
**0.814–1.172**, despite median B/A about 1.002. Mix median 128K prefill is about
1.151, but interval **0.923–1.360** remains inconclusive. Historical 3% point
comparisons would hide this uncertainty. No outlier was removed: repaired
integration round 0's original 4K decode is **5.724 TPS**, versus 26.442 and
26.551 in its other original arms; its broad decode interval is retained.

Full three-pair durations: main **71.37 min** including operational
interruption, mix **62.83 min**, wide **62.72 min**, integration screen
**25.68 min**. The full battery marginally exceeded the 60-minute target;
metrics were not dropped retrospectively to meet it. Main's unsupported
optional `gpu_busy_percent` sensor interrupted an operator status read.
The original was restored, complete pair 0 retained, and only the remaining
fixed pairs run with the tested resume path. Logs and initial analysis remain.

## Two-session integration

The first integrated build (`c59611b1…`, merge `4c646b5`) failed before completing
its first pair. A bounded warm reproduction identified rank 1:
`Vulkan batch exceeded 4096 descriptor sets`; other ranks then lost transport.
Slots 4–7 tag a separate static pool, but an old dynamic-range check rejected
their tags. Commit `79b0efc` limits that check to the dynamic pool and retains
the static slot/window bounds. An expanded eight-slot Vulkan test fails before
and passes after the repair. The obsolete near-limit HTTP client alone was
cancelled after this diagnosis; its parent captured evidence and restored
original. Failed-build data are retained, never pooled with repaired samples.

The repaired binary received three completely fresh pairs. Its 32-token,
one-in-flight prefill quantum regresses 4K engine prefill to median B/A
**0.1224**, guarded interval **0.0897–0.1902**. Chat TTFT is
**1.3527 [1.2728,1.4326]**, cached TTFT **1.1904 [1.1448,1.2305]**.
Concurrent chat wall throughput regresses
**0.8055 [0.7628,0.8609]**; aggregate improves
**1.6462 [1.5849,1.6998]** with a corrected win. Aggregate improvement cannot
hide the individual/single-stream failures.

Primary non-stream concurrency medians (actual 128-token responses):

| Deployment | Chat end-to-end TPS | Agent end-to-end TPS | Aggregate wall TPS |
|---|---:|---:|---:|
| Original, sessions 1 | 18.977 | 9.318 | 18.636 |
| Integration, sessions 2 | 15.215 | 15.258 | 30.429 |

Aggregate uses latest completion, not the sum of per-stream rates. The
predeclared hierarchical screen failed, so repeated 32K/128K two-session
performance was not measured or claimed noninferior. The one-cold near-limit
probe is a separate capacity/load diagnostic, not a replacement long-context
performance interval or isolated-agent greedy control.

The first repaired warm SSE diagnostic stopped on **one actual chat token**,
not the requested 128; it is not a sustained-throughput result. The harness
now preserves invalid partial trials before raising, covered by a new fixture.
One further forensic trial was declared in advance solely to retain its
isolated/loaded token counts and hashes; it cannot overturn qualification.
It also failed the sustained-generation check: chat emitted 128 tokens but the
agent emitted **one token (`"1"`) both isolated and loaded**, with identical
hashes. Chat likewise matched its isolated hash. Visible decode overlap was
zero. The observed loaded chat TTFT was 1394 ms versus 1156 ms isolated; the
one-token agent's 1304 ms is not a sustained decode measurement. Raw partial
rates are retained for diagnosis, not accepted or pooled into performance
intervals. This does not establish a concurrent-only numerical defect.

## Capacity/load and final operations

The user stopped the final capacity diagnostic at **23:03 on 2026-10-15
(UTC−07:00)** and requested immediate restoration/publication. The coordinator
recorded **178400 / 260075 prompt tokens** prefilled, frontier 178400,
4878.0 seconds, followed by `client_failed 1` from the explicitly cancelled
client. **No near-260K completion, terminal agent SSE usage, post-long resident
chat result or full-context memory pass exists.** This interrupted diagnostic
cannot establish two populated 262K sessions or their functional correctness.
No additional capacity retry was made.

The initial old-build probe and an earlier repaired-build probe were also
interrupted: the former after the descriptor fault was diagnosed, the latter
during transfer to stdin-only fleet transport. Both cancellations and all
eight-owner archive groups are retained. None contributes to paired CIs.
The final forensic trial and final capacity attempt used the stdin-only
adapter, which retains the central runner's credentials/transfers without
creating a remote script file. The shared central runner was not modified.

Final interrupted probe observations, **not full-context qualification**:

| Owner | Minimum MemAvailable GiB | Peak process RSS GiB | Peak DRM GTT+VRAM GiB |
|---|---:|---:|---:|
| .42 | 2.195 | .119 | 11.867 |
| .43 | 3.454 | .191 | 10.836 |
| .44 | 4.132 | .191 | 10.133 |
| .45 | 3.519 | .193 | 10.641 |
| .46 | 3.736 | .191 | 10.614 |
| .47 | 3.281 | .193 | 10.885 |
| .48 | 4.409 | .192 | 9.975 |
| .49 | 3.052 | .194 | 11.135 |

Each owner has 2457–2458 samples spanning about 4922–4924 seconds; maximum
sample gap is below 2.007 seconds. All have zero new OOM kills, zero process
swap, zero system-swap growth, zero swap-out pages, zero process major-fault
delta and no 512 MiB reserve violation or sampler error. Rank 0 has **14
global swap-in pages**; others have none. Historical counters/preexisting
swap are not interpreted as task OOMs. QSA backing files are preallocated:
allocated disk bytes do not prove a populated 262K context. Cache-hit/miss
and storage-throughput instrumentation was not collected.

Original production was restored, then its **retained pinned launchers were
installed and actually restart-tested on all eight owners**. Verification
checks the live executable inode SHA, exact argv/environment, original
binary/manifest SHA256, `/health`, exact model ID and real greedy visible chat.
No automatic boot-start service was installed; the original `nohup` semantics
are retained. See `docs\DEPLOY_CURRENT.md` and `ab-current`.

All **48 compressed owner archives** were secured before cleanup. All
task samplers were stopped by owned numeric PID plus exact script argv;
`/home/xander/hc-abq-20261015-memory` was removed on all eight after checking
live references. Only stopped, task-created mix-pack slot-1 state files were
removed after FD/mapping checks. Sealed candidate weights/manifests and staged
release binaries were retained; original weights/build/manifest were untouched.
Post-cleanup `/health` and real visible greedy chat passed. **Production
lease .42–.49 is released; no further fleet work is scheduled by this task.**

## Validation and evidence

Harness/historical-compatibility/stdin-transport fixtures: **16 pass**. Integrated concurrency
fixtures, including failed-short-trial persistence: **4 pass**. The expanded
static-slot regression passes on llvmpipe after reproducing the pre-fix failure.
Targeted native multi-session transport/API/QSA/session and decode-batch tests,
flag audit, PowerShell parsing, LF-normalized Bash parsing and source/docs whitespace checks
pass. This does not claim the full native baseline suite passes: existing
ordinary fabric/prefill fixture failures remain documented in
`docs\MULTI_SESSION.md`.

`ab-ops` retains staging, controls, operational failures and before/after
regression logs. `ab-memory` retains per-owner compressed sampler/log archives;
RSS and DRM allocations overlap on unified memory and must not be added.
Two-second sampling can miss transient peaks; major faults/swap are not a
complete QSA-cache hit/miss or disk-throughput instrument. No absence of
uninstrumented cache thrashing or two fully populated 262K contexts is claimed.
Rebuild the compact summary from the worktree root with
`python results\hillclimb\ab-ops\summarize.py`; `ab-ops` also retains the
manifest-bound quality receipt and its SHA-verified 480-frontier report.
