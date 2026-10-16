# Paired production qualification

The historical `results\hillclimb\BASELINE.json` and greedy reference are retained as historical evidence, not performance acceptance thresholds. Run-to-run changes in the original binary are larger than the former 3% single-sample margin.

```powershell
pwsh -NoProfile -File tools\hillclimb.ps1 -ABConfig .build\abq\main-config.json -ABRounds 3 -ABOrder ABBA -ABOutput results\hillclimb\ab-main
```

`ABConfig` is a JSON object with `A`, `B`, `deploy_script`, `restore_script`, `change_kind` (`exact` or `weight-format`), and optional `contexts` (default 4096,32768,131072). Deployment hooks receive `-Deployment <hashtable> -Arm A|B`; they must block until the same model is healthy. The restore hook runs even after benchmark or deployment failure. There is no automatic promotion. Never use `-WriteReference` in A/B mode.

The included `ab-deploy.ps1`/`ab-instance.sh` operator accepts only the eight production owners, original/mix-04 manifests, and isolated `YYYYMMDD-abq*` releases. It verifies executable/manifest hashes, stops only owned numeric PIDs with exact executable and manifest arguments, waits after TERM, and refuses force/name kills. Original logs/PIDs remain outside the immutable original release. `.build\abq\live.json` records the last attempted deployment before launch so partial starts can be rolled back. The fleet runner receives an explicit target array. Do not run two operators concurrently.

## Workload and accounting

Each pair gets a fresh random salt; all arm requests within that pair use the same deterministic salts and seeds. `ABAB` runs A,B for every pair; `ABBA` reverses the second pair, yielding A,B,B,A,A,B for three pairs. Each arm is freshly deployed, including adjacent equal arms. Long cold and decode request hashes, and cached-TTFT body hashes, must match. Canonical greedy cases remain unsalted to retain their role as a fixed numerical reference. A changed one-token prefix response makes a paired continuation invalid rather than silently comparing different workloads.

One cold **non-streaming** request per context collects actual prompt usage, engine prefill TPS and client prompt-tokens/full-response-wall TPS. The latter includes HTTP/rendering and one generated token; it is not SSE TTFT or pure engine prefill. Actual reused tokens must be zero. This avoids the old two-cold-prefill-per-context cost; the full battery aims for about 50-60 minutes per three-pair experiment, including deployments, not a hard timeout.

Long decode immediately continues each exact cold prefix. It must retain the prefix and emit at least 32 actual tokens. Engine decode TPS and actual completion tokens divided by (HTTP wall minus engine prefill seconds) are reported separately; the client interval includes HTTP overhead, not SSE-event counts. Short decode uses the same accounting. Chat and cached-prefix TTFT use first visible SSE content, not the role-only event. A cached prefix includes the exact assistant text from its immediately preceding prime.

The original API has no `ignore_eos`. Both concurrent non-stream HTTP requests are submitted before waiting, and each response contributes its own actual usage and completion timestamp. Raw per-stream/aggregate tokens/s are retained. Equal-length normalization retains observed queue/non-decode time and scales each engine decode interval to the minimum actual completion count; report this approximation explicitly. It is not a measurement of fixed-token execution, and does not correct content-dependent cost or the other stream's EOS-dependent queue delay. A/B sustained concurrency requires at least 32 actual completions per stream. Never count SSE events as tokens.

## Inference and acceptance

The unit of resampling is a complete paired round, not a token, layer, SSE event, or request within the round. Report arm medians, min/max and IQR, the median paired B/A ratio, and a fixed-seed 20,000-replicate percentile bootstrap 95% interval on paired log ratios. Higher-is-better metrics require the entire interval >=0.97; latency requires the entire B/A interval <=1.03. An interval crossing the margin is **inconclusive**, not a pass. Requiring every protected metric to pass is an intersection-union noninferiority test; a fast short context cannot hide a slow long context.

A win requires the interval to show improvement. To avoid selecting a spurious win among metrics/candidates, the win interval uses a Bonferroni tail adjustment across all protected metrics and the four predeclared candidates. The ordinary 95% intervals remain visible. Three rounds are a minimum, not evidence of reliable asymptotic coverage: a small-n bootstrap is exploratory and can be unstable, especially if samples are identical or correlated. Thermal drift/order effects and salt-dependent routing are not eliminated by pairing. Predeclare the number of rounds/metric subset; do not repeat until a desired result appears.

Exact-math changes require every paired canonical greedy text hash to match. Weight-format changes report hash differences, but use the existing `q4_quality.py` full-vocabulary gate: >=400 frozen frontiers, 248320 logits, mean KL <=.01, p99 KL <=.05, top-1 >=97%. A quality receipt binds `manifest_sha256`, `quality_path`, and `quality_sha256`; the raw record and its digest are checked before deployment. The receipt is a provenance assertion based on the prior capture, not a new quality measurement. Existing pack quality does not excuse an unqualified new arithmetic change.

Promotion requires complete samples, quality/hash approval, noninferiority on every protected metric, and at least one corrected win. Incomplete, inconclusive or failed candidates leave the original serving. Preserve JSON and deployment identity receipts with the final summary.
