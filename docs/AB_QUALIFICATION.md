# Paired production qualification

The historical `results\hillclimb\BASELINE.json` and greedy reference are retained as historical evidence, not performance acceptance thresholds. Run-to-run changes in the original binary are larger than the former 3% single-sample margin.

```powershell
pwsh -NoProfile -File tools\hillclimb.ps1 -ABConfig .build\abq\main-config.json -ABRounds 6 -ABOrder ABBA -ABOutput results\hillclimb\ab-main
```

`ABConfig` is a JSON object with `A`, `B`, `deploy_script`, `restore_script`, `change_kind` (`exact`, `weight-format`, or `numeric`), and optional `contexts` (default 4096,32768). Five complete pairs are the minimum; six is the default. Deployment hooks receive `-Deployment <hashtable> -Arm A|B`; they must block until the same model is healthy. The restore hook runs even after benchmark or deployment failure. There is no automatic promotion. Never use `-WriteReference` in A/B mode.

`-ABResume` can continue an **incomplete** policy-v2 output directory after an operational failure. It retains every complete paired round, repeats only an uncommitted interrupted pair with a new salt, and rejects changes to the configuration digest, quality receipt digest, order or total rounds. It rejects old schemas. Partial arm files are retained with an interrupted suffix. It refuses completed runs: resuming is not permission to add favorable observations or repeat until pass. An unsupported optional sensor read is recorded as unavailable, not a serving/deployment failure.

The included `ab-deploy.ps1`/`ab-instance.sh` operator accepts only the eight production owners, original/mix-04 manifests, and isolated `YYYYMMDD-abq*` releases. It verifies executable/manifest hashes, stops only owned numeric PIDs with exact executable and manifest arguments, waits after TERM, and refuses force/name kills. Original logs/PIDs remain outside the immutable original release. `.build\abq\live.json` records the last attempted deployment before launch so partial starts can be rolled back. The fleet runner receives an explicit target array. Do not run two operators concurrently.

`ab-fleet.ps1` retains the user's fleet runner, credential source and file
transfers, but feeds the decoded Bash payload on stdin instead of creating a
remote script file. It accepts only explicit .42–.49 targets, has no sudo mode,
and fails closed if the upstream runner's transport layout changes. The shared
fleet runner itself is not modified.

Starting rotates the preceding rank log/PID to `previous-rank-N.log`/`.pid`
inside the operator directory. Download these immediately after a failed
candidate restores: the next deployment rotates again. A runtime failure is
unqualified, not an inconclusive confidence interval or permission to discard
an unfavorable completed pair.

## Workload and accounting

Each pair gets a fresh random salt; all arm requests within that pair use the same deterministic salts and seeds. `ABAB` runs A,B for every pair; `ABBA` reverses odd-numbered pairs, yielding A,B,B,A,A,B,B,A,A,B,B,A for six pairs. Each arm is freshly deployed, including adjacent equal arms. Long cold and decode request hashes, and cached-TTFT body hashes, must match. Canonical greedy cases remain unsalted. Cached continuations use the fixed assistant text `Ready`, not either model's generated prefix response; numeric changes cannot silently change the paired input. Actual prime responses remain in diagnostics, and long decode must retain all but eight tokens of its cold prompt.

The `paired-affinity-v3` battery supplies a stable OpenAI
`user=hillclimb-stream-0`; concurrent requests use distinct users 0 and 1.
This is required for retained-prefix routing with sessions 2. Resume also
binds this battery identity and the same-kernel greedy receipt digest.

One cold **non-streaming** request per context collects actual prompt usage, engine prefill TPS and client prompt-tokens/full-response-wall TPS. The latter includes HTTP/rendering and one generated token; it is not SSE TTFT or pure engine prefill. Actual reused tokens must be zero. Default contexts are 4K/32K; expensive 128K is optional and secondary, not duplicated by default. Reserve sufficient lease time for six pairs, quality captures, cleanup and rollback; no elapsed-time guarantee has been measured for the revised battery.

Long decode immediately continues the cold prompt with the canonical assistant stub. It must retain the prefix and emit at least 32 actual tokens. Engine decode TPS and actual completion tokens divided by (HTTP wall minus engine prefill seconds) are reported separately; the client interval includes HTTP overhead, not SSE-event counts. Short decode uses the same accounting. Chat and cached-prefix TTFT use first visible SSE content, not the role-only event.

The original API has no `ignore_eos`. Both concurrent non-stream HTTP requests are submitted before waiting, and each response contributes its own actual usage and completion timestamp. Raw per-stream/aggregate tokens/s are retained. Equal-length normalization retains observed queue/non-decode time and scales each engine decode interval to the minimum actual completion count; report this approximation explicitly. It is not a measurement of fixed-token execution, and does not correct content-dependent cost or the other stream's EOS-dependent queue delay. A/B sustained concurrency requires at least 32 actual completions per stream. Never count SSE events as tokens.

The legacy full/quick harness now emits schema 2 and labeled actual-token
concurrency rows. `q4_qualify.py` retains schema-1 historical replay, but refuses
the new actual-token and paired schemas with a structured failure: historical
visible-event baselines must never be compared to token throughput.

## Policy v2: primary metrics and acceptance

The six predeclared primary operating points are:

| Goal | Metric | Benefit ratio |
|---|---|---|
| Short decode | `decode-short.engine_tps` | B/A |
| 32K decode | `decode-32768.engine_tps` | B/A |
| 4K prefill | `prefill-4096.engine_tps` | B/A |
| 32K prefill | `prefill-32768.engine_tps` | B/A |
| Chat TTFT | `chat.ttft_ms` | A/B |
| Concurrent aggregate | `concurrent.normalized_aggregate_tps` | B/A |

Everything else is secondary/descriptive, including client TPS, 4K decode,
128K, cached-prefix TTFT and individual concurrent streams. Secondary wins
cannot select a candidate and secondary regressions cannot veto one.
Missing any primary means ineligible. Legacy `long_only` and
`requires_long_qualification` runs remain explicitly partial/ineligible.

Predeclare at least five pairs, normally six, and the complete candidate set.
Resample complete paired rounds, never tokens/layers/SSE events. Report each
arm's median/min/max/IQR and the arithmetic median of paired benefit ratios
(mean of the two middle values for even samples), with a fixed-seed,
20,000-replicate percentile bootstrap **marginal 95% CI**. All metrics use the
same resampled round indices; adding secondaries cannot change primary CIs.
Raw B/A remains available separately for latency.

A configuration is **eligible** only when:
1. Every primary median benefit is **>=0.97**, with no primary CI wholly below
   0.97 (upper bound must be >=0.97).
2. Quality passes.
3. At least one primary median benefit is **>=1.03**.

There is **no Bonferroni adjustment**, fixed-df t expansion, or CI-lower-bound
requirement for the operational promotion decision. `noninferior` still
describes a lower bound >=0.97, but is not the eligibility predicate.
`accepted-uncertain` explicitly labels median-acceptable metrics whose CI
crosses 0.97. A median win does not imply statistically established
improvement. This is the user's risk policy, **not proof of noninferiority or
family-wise error control**. Five/six pairs can still give unstable bootstrap
coverage, especially with drift, dependence or identical samples. Do not
repeat until pass or silently discard unfavorable completed pairs.

Historical three-pair reports, including the original-vs-original control
with 12/19 inconclusive metrics and its df=2/76-comparison guard, are unchanged.
They cannot satisfy the new minimum or be retroactively relabeled passes.

Exact-math changes require paired canonical greedy text hashes to match.
Numeric/reassociation changes report those differences and require the
full-vocabulary `q4_quality.py` gate: >=400 fixed frontiers, 248320 logits,
mean KL <=.01, p99 KL <=.05, top-1 >=97%. The `numeric` quality receipt binds
both releases, serving/helper hashes, manifests, sessions, sorted complete
flag sets, corpus digest and raw quality-file digest. Old pack-only receipts
do not qualify new flag combinations. `weight-format` remains for legacy
manifest-bound receipts; optional exact control/reference constraints remain
binding when configured.

The harness reports eligibility only; it always invokes the configured
restore hook and records `restored_baseline`. A may be the current mix-04
deployment, not the historical original. Candidate-stack operation and
lease-gated execution are documented in [CANDIDATE_STACK.md](CANDIDATE_STACK.md).
