# Offline gate and stack preparation

2026-10-16. **No fleet access, production HTTP, staging, deployment, or lease
acquisition.** Current main/mix-04 production and immutable originals are untouched.

## Gate

Policy `primary-median-v2`: default six pairs, minimum five. The six primaries
are short/32K engine decode, 4K/32K engine prefill, chat TTFT, and normalized
concurrent aggregate. Every primary median benefit must be >=.97, no primary
95% paired-round bootstrap CI may be wholly below .97, quality must pass, and
at least one primary median must be >=1.03. No Bonferroni or df=2 t guard.
Secondary results remain descriptive. Uncertain acceptance is labeled, not
misrepresented as statistical proof. Historical three-pair reports are unchanged.

## Integrated code and prepared batch

`cuppaxanax/stack-1`: main `a63a166` + roofline `e073c5d` + latency `8c4f9be`
+ host-overhead `198583f`; MTP Q8 addressing fix only in `48c6bff`.
The host/roofline dispatch conflict retains both guards/ledger behavior.
All experimental defaults remain off. No full MTP or multi-session merge.

The concrete batch on `cuppaxanax/ab-harness` declares all selected roofline,
n-gram-cache and wide flags, followed by the identical stack with
`FG_DECODE_SUBMIT_GROUP=2`. Each complete flag combination gets one fresh
480-frontier full-vocabulary comparison and one six-pair performance run.
Teacher capture is shared. Group 2 additionally requires unchanged greedy
hashes against the completed same-binary/shader/other-flags group-zero control;
KL cannot waive that requirement. Receipts now include actual shader-tree
identity, because the Q8 fix is in SPIR-V, not necessarily the serving ELF.

## Offline validation

* Native C17/server, all shaders and both source-matched quality helpers built.
* Flag gate passed: code 50, test 33, docs 78, removed 18.
* Three native CPU tests passed: sampler, chat, chat runtime.
* Four local llvmpipe selectors passed: roof controls, independent two-block
  Q8 generic/fused oracle, byte-exact n-gram cache lookup, submit grouping.
* QSA causal/128K-boundary/cache/ordered-upload/persistence suite passed.
* Harness/stack/quality/transport: 40 Python cases, 39 passed, one existing
  native quantizer case skipped because a built ggml library is unavailable.
* Roofline/fusion ledger: 18 Python tests passed.
* PowerShell AST parsing and the concrete default-offline example plan passed.

A separate pre-host-merge broader llvmpipe run printed three first-shape dense
parity passes and then aborted with heap corruption; HC up/mix rejected its
unsupported capability profile. These are **not passes**. Root cause of the
dense abort was not established; targeted RADV dense/HC/GDN hardware oracles
remain required. The original log is preserved compressed, along with the
later selected-test success logs. Full `make test` was not run because unrelated
legacy fixtures use prohibited non-project scratch.

Build hashes and raw validation logs are in `STACK_OFFLINE.json` and
`stack-offline\`. Prepared tarballs persist under the respective worktrees'
`.build\stack`; no files were uploaded. Rebuilds require refreshed ELF hashes.

**Pending, not claimed:** real-model stack KL/top-1, same-kernel grouped greedy
parity, paired serving performance, and promotion. Execute only after the
multi-session and MTP leases end and a new exclusive .42–.49 lease is granted.
See `docs\CANDIDATE_STACK.md`; planning is offline, execution is explicit and
lease-gated, and cleanup always restores configured current main/mix-04 A.
