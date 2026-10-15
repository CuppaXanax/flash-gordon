# Single-stream decode: measured overhead

2026-10-15. Branch `cuppaxanax/decode-climb`, based on `1863c95`.

**No overall winner was promoted. Original production is restored and healthy.**

## Scope and reproducibility

Eight production ranks only, `192.168.42.42` through `.49`; no access to
`.50` through `.53`. Pack `/home/xander/fg-ring-pack-abliterated`, model
`Qwen3.8-Flash-Next-Uncensored`, API `.42:8080`. All measurements retain
`--context-tokens 262144 --gpu-index-tokens 262144 --qsa-hot-tokens 0
--qsa-page-cache-mib 700`. No weight/shader formats or machine settings changed.

The original executable SHA256 is
`e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2`;
the manifest file SHA256 is
`35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`.
The manifest-file digest is not the abbreviated runtime ledger's manifest digest.
The original `20260926-abl2` directory and production pack were never modified.
Experiments use isolated `20261015-decode*` releases.

No serving systemd unit was found. The existing `start-*-ring-split4trace.sh`
scripts select the newest release, use a stale pack, and kill by name, so were
not executed. `tools/decode-instance.sh` instead follows the existing operator's
manifest-scoped `/proc` stop mechanism: owned numeric PID, exact manifest
argument, release executable path, TERM, bounded wait, no forced/name kill.
Its rank/IP allowlist is deliberately specific to this experiment. Start only
after stop; logs and PID files live in the isolated release, not scratch.
The PowerShell wrapper requires an explicit fleet-runner path, validates ranks
0--7, checks every remote exit code, and removes its generated local script.
For example, from the worktree:

```powershell
& .\tools\decode-instance.ps1 -Action status `
  -Release /home/xander/flash-gordon-live/20260926-abl2 `
  -FleetScript D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1
```

Original-build logs/PIDs go into `20261015-decode-operator`, never into the
protected original build directory.

## Measurement traps and additive timeline

Production-equivalent capture: `FG_DECODE_RING_TRACE=1 FG_FABRIC_PROFILE=1
FG_GENERATE_TRACE=1`, with static replay still enabled. The original binary's
quick harness gives **26.723 client TPS**, essentially identical to the
immediately preceding uninstrumented **26.722 TPS**.

**`FG_DECODE_PROFILE=1` is intrusive.** It disables static replay and serializes
profiled Vulkan flushes. The same original binary falls to **20.931 TPS**;
its captured token window is **50.450 ms**, versus **37.437 ms** normally.
Do not call the extra 13.013 ms production overhead. GPU timestamps below are
diagnostic measurements of this alternate execution, not simultaneous
measurements of the normal static-replay run.

`decode-original-trace.json` summarizes the final 64 matching token positions
60--123, the fixed logic correctness case. The offline parser checks consecutive
token indices and identical windows on all ranks.

| Serial critical-path accounting | Original normal, ms/token | Original detailed-profile, ms/token |
|---|---:|---:|
| Coordinator embedding, n-gram and initial send preparation | 2.122 | 1.595 |
| Eight rank services, including enqueue and final HC/fanout | 29.971 | 43.404 |
| Transport, head completion and remaining unattributed time | 5.344 | 5.451 |
| **Sum = observed ring decode** | **37.437** | **50.450** |

The old trace's `embed_ms` already includes n-gram and send preparation:
adding its three labels double-counts approximately 2.1 ms. The patch makes
embedding-only `embed_ms` additive. A refined candidate trace measures:

| Coordinator preparation | ms/token |
|---|---:|
| Embedding | 0.017 |
| N-gram fetch/assembly | 1.886 |
| Initial send/preparation | 0.182 |
| Total | 2.084 |

That separate normal capture totals **38.212 = 2.084 + 30.710 + 5.419 ms**.
It is not evidence of a short-context speedup; the optimization is inactive
for the small-window attention path.

## Rank GPU, host and idle measurements

All entries are milliseconds/token. Service is the original normal run;
GPU/host/fence columns are the original intrusive profile. Input poll p50
comes from the refined normal capture, after fixing the previously zero
worker `wait_ms`. These columns must not be added across different captures.

| Rank | Normal service | Profile GPU span | Host record | Queue submit | Fence wait | Input poll p50 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3.646 | 4.012 | 0.485 | 0.443 | 4.337 | n/a |
| 1 | 4.167 | 4.279 | 0.454 | 0.353 | 4.543 | 1.348 |
| 2 | 3.575 | 3.324 | 0.430 | 0.348 | 3.561 | 10.661 |
| 3 | 3.722 | 3.682 | 0.480 | 0.441 | 3.992 | 32.950 |
| 4 | 3.450 | 3.824 | 0.440 | 0.351 | 4.087 | 17.901 |
| 5 | 3.637 | 3.508 | 0.479 | 0.457 | 3.850 | 32.966 |
| 6 | 3.454 | 3.828 | 0.444 | 0.365 | 4.128 | 32.809 |
| 7 | 4.320 | 3.561 | 0.464 | 0.452 | 3.887 | 32.357 |

Rank order is 1,0,2,3,4,5,6,7; rank 0's service is `own_ms`, not its whole
coordinator wall time. Rank 7's normal service includes HC/fanout.
Fence wait overlaps GPU execution; it is **not another 32.385 ms/token**.
Input poll is the wait immediately before a work frame, interrupted by other
control/head traffic, not a rank's complete idle interval. First-token waits
include between-request startup, so p50 is more useful than the mean.
Never sum the ranks' concurrent idle waits into the serial token budget.
Worker fallback poll intervals use existing local realtime timestamps;
fabric service and other profile intervals use monotonic clocks.

The detailed run sums to 30.017 ms GPU span, 28.821 ms scoped kernels,
1.196 ms inter-dispatch gaps, 3.677 ms host recording and 3.207 ms submission.
Those numbers describe **72 submissions / 1,675 dispatches** across the
profiled blocks, not the normal static command-buffer replay.
The added normal block counters expose the difference: worker blocks commonly
show 5--8 submissions and 40--85 dynamically counted dispatches, with recorded
static work not represented as newly recorded dispatches.

Largest diagnostic GPU kernel scopes, summed over the eight serial blocks:
experts 7.517 ms; GDN projections 5.327; GR attention/FFN reads 4.644;
GDN output 2.904; QSA projections 2.607; GDN recurrence 1.321;
QSA state attention 1.293; shared experts 1.112; QSA output 0.961;
router 0.802. These are GPU costs, not host/network overhead.

## Network, head and sampling

Work-frame enqueue averages approximately **0.028--0.035 ms/send**, not
end-to-end propagation. Each ordinary hyper payload is 40,960 bytes plus
protocol metadata; n-gram-carrying work is larger. A payload already queued
can report a short receive service even though propagation happened earlier.
Do not multiply the receive syscall's duration by hop count and call it link
latency. The additive residual above deliberately remains partially unattributed.

The refined trace measures four concurrent head shards on ranks 4/0/1/2 at
**0.768--0.785 ms each**, and rank 7 HC at **0.157 ms**. The refined detailed
profile measures each head shard at **0.674--0.685 ms GPU**, and HC at
**0.074 ms GPU**. Do not sum concurrent head shards, or charge the 675 MB head
as a serial single-rank read. HC is already in rank 7's service.
Trace-only JSON's zero GPU fields mean unmeasured, not zero execution time.
The normal residual includes roughly one shard wall time plus fanout,
partial-result return, sampling and coordinator waits; these are not all network.
The old coordinator `output_ms=0` means handoff, not a free output head.

For the original 64-token logic capture, generation-loop decode is 2397.349 ms
and total is 2397.487 ms. Interrupt/tokenizer/callback/render together are only
0.138 ms/request, approximately **0.0022 ms/token**. Host text handling is not
the missing 18 ms/token.

The 6.67 GB / 450 GB/s estimate is a lower bound on ideal weight reads, not a
proof that everything above 14.8 ms is CPU/network. Actual mixed kernels have
issue, reduction and occupancy costs; the detailed GPU work is already large.
Keep the normal wall ledger and intrusive GPU diagnostics separate.

## Implemented change and correctness

The selected optimization removes the QSA selection CPU round-trip when
capacity covers **all pages of all local QSA layers for the session**, and a
conservatively tracked contiguous resident prefix covers the visible tokens.
The same GPU scan, top-k, slot resolution and attention arithmetic remain.
It omits the forced batch end, host miss-flag read, resolved-slot read and
unnecessary LRU touches when eviction is impossible. The caller no longer
restarts a batch that the resolver left active. `FG_QSA_DEVICE_SELECT=0`
restores the old path; it is not needed for the candidate operating point.

Eviction truncates the prefix; reset clears it; sparse/restored caches cannot
claim residency until established. Undersized and locality-traced caches retain
the original host resolution. The regression test compares selected GPU slots
exactly against reference, immediately gathers nonzero records plus a partial
tail in the same batch and checks every byte, asserts zero submissions through
the gather, and verifies eviction/reset return to the draining path.
Existing causal, 128K boundary, cache eviction and deduplicated-fetch tests pass.

A separate projection-fusion attempt was **rejected**: four of five harness
output hashes changed (`20261015-061031.json`). Disabling both changes on
that same binary restores all five; disabling only fusion also restores all
five. Fusion was removed from the final source, not merely hidden behind an
environment flag. Its hazard was not diagnosed. No bf16 codec or weight-format
change was attempted, so no KL claim is made.

Final isolated binary `20261015-decode-r2` SHA256:
`c57e043095b15da22808c949efceb4f7f22bc5403615de63df62fc7288e93246`.
The final binary's trace and detailed-profile quick runs both match all five
fixed greedy-output hashes. The earlier full selection-only experiment
(`20261015-061729.json`) also matches all five, with client decode
25.272 TPS at 32K and 22.679 TPS at 128K. Long performance prompts are salted
independently; their EOS lengths can differ, so these are throughput comparisons,
not claims of identical long-prompt token trajectories.

Additional validation: owner reduction; Vulkan QSA gather/attention, flush and
batch parity, output argmax, four-way combine, and cooked quarter-view all pass.
The flag/documentation gate passes. Four local ledger-accounting tests pass.
`test_fabric` fails at the existing output-config assertion, line 186; recompiling
with unmodified HEAD `fabric.c` reproduces the identical failure. It is not
reported as a passing suite and was not changed as part of this work.

## Final harness and deployment

Final uninstrumented full run: **`20261015-064339.json`**, 1066.00 seconds,
no harness errors, **5/5 fixed greedy-output hashes identical**. This build
does not require an optimization environment variable.

| Client decode | BASELINE TPS | Candidate TPS | BASELINE ms/token | Candidate ms/token |
|---|---:|---:|---:|---:|
| Short | 26.760 | 26.606 | 37.369 | 37.585 |
| 32K | 24.440 | 25.279 | 40.917 | 39.559 |
| 128K | 22.320 | 22.119 | 44.803 | 45.209 |

32K saves **1.358 ms/token (+3.43% TPS)**, but 128K loses
**0.406 ms/token (-0.90% TPS)**. The earlier selection-only full run's
128K improvement did not reproduce. The worst-band roadmap gate is not met.
The five correctness prompts are short and do not exercise the new resident
selection path; that path additionally has the exact GPU-slot/gather regression
test, not a separate long-prompt greedy-identity or KL claim.

Engine TPS is respectively 25.197 / 25.402 / 22.229, versus baseline
25.80 / 23.84 / 21.77. Long-context engine headers measure a **cached follow-on
continuation**, not the exact SSE stream; do not use them to hide the client
128K regression.

| Cold prefill, client TPS | BASELINE | Candidate |
|---|---:|---:|
| 4K | 339.91 | 331.31 |
| 32K | 357.02 | 361.10 |
| 128K | 415.58 | 353.40 |

128K prefill is 15.0% slower and the full run takes 14.6% longer than the
930.04-second baseline. No contemporaneous full opt-out run isolates whether
this is source-build or machine/run variation, so it is **not attributed
causally to the resident-selection change**. All eight ranks' complete SPIR-V
sets are byte-identical to original production. Regardless of cause, these
measurements do not establish a strictly better full-harness deployment.

**Deployed: original `/home/xander/flash-gordon-live/20260926-abl2` on all
eight ranks.** The candidate remains an isolated, non-serving `decode-r2`
build and branch experiment. Restored-original quick run
`20261015-070323.json` passes all five hashes: **26.595 client TPS,
37.601 ms/token**. `/health`, `/v1/models` and a chat-completion request respond.
Numeric-PID inspection confirms the original executable, exact pack/arguments,
and unchanged executable/manifest hashes on every blade. All
`/home/xander/hc-decode*` directories, the copied-original profiling release,
and the rejected fused release were removed. Persistent operator logs remain
outside the original build.

## Evidence and replay

Tracked summaries: `decode-original-{trace,profile}.json`,
`decode-r2-{trace,profile}.json`, `decode-result.json`, `decode-validation.json`,
`decode-deployment.json`, `decode-api-check.json`; harness artifacts under `runs/`.
Local raw captures are retained in `decode-captures.tar.gz`.

From an extracted capture directory:

```text
python tools\decode-ledger.py CAPTURES --suffix trace --legacy-embed --output original.json
python tools\decode-ledger.py CAPTURES --suffix r2-trace --output candidate.json
python -m unittest discover -s tests -p test_decode_ledger.py -v
```

Use `--legacy-embed` only for the old binary's trace. Profiling flags are off
for qualification and final serving; only the original coordinator's
`FG_RING_TRACE=1` launch setting is retained.
