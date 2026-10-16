# Decode submission grouping: offline gate

Base: main `a63a166`. Branch: `cuppaxanax/host-overhead`.
Only `.50` and `.51` were built/tested; `.50–.53` were inventoried.
**No access to `.42–.49`, no production API requests, no production deployment.**
The original production release and mix-04 pack remain untouched.

## Finding and implementation

`PROD_GAP.md` identified 3.054 ms of submit calls, 1.263 ms of recording,
0.064 ms of resets and 3.653 ms of residual service wall across eight ranks.
The 23.266 ms of fence-call blocking is not an additional CPU cost or a
measurement of GPU busy time. Rank-7 egress is a separate 0.557 ms.

The old resident-QSA experiment established two important restrictions:
`FG_DECODE_PROFILE` disables static replay and is not representative, and
removing QSA drains without preserving mutable uploads is unsafe. Multiple QSA
layers reuse the same slot-ID tensor. CPU writes for a later layer can replace
the inputs of an earlier queued gather.

The new flag defaults **off**:

* `FG_DECODE_SUBMIT_GROUP=1`: queue existing static GDN and dynamic QSA command
  buffers, then submit/wait once. No arithmetic kernels or weight formats change.
* `FG_DECODE_SUBMIT_GROUP=2`: publish the first GDN run asynchronously before
  recording QSA, then submit/wait for the suffix. This retains recording/GPU
  overlap while eliminating most submissions.

Groups use bounded private command buffers, monotonic descriptor epochs, explicit
cross-primary memory barriers, the existing semaphore chain, and ordered
`vkCmdUpdateBuffer` snapshots of mutable slot uploads. Explicit host-readback
ends remain synchronization points. Capacity spills drain before buffer reuse.
Abort waits for any published prefix and discards unsubmitted work.

Serving eligibility requires static replay, session zero, no timestamp profile,
full-session QSA cache capacity, a contiguous resident prefix, matching committed
positions, and an already allocated index segment. Sparse/restored/undersized
caches, other sessions and segment-allocation boundaries retain the old path.
Skipping CPU selection/LRU touches is safe only when the entire session fits.

## Offline results

Evidence: `host-overhead/summary.json`, native helper output, exact binary hashes,
and the compressed raw CPU-only trace in `host50-results.tar.gz`.

The replay streams 1,002,700,800 cooked-Q8 weight bytes per synthetic block using
the existing 2560×12288 kernel. Two static runs surround six dynamically recorded
pieces, totaling 30 dispatches. Five rotating-order rounds use distinct inputs;
each mode has 128 warmups and 128 measured blocks per round. Outputs are compared
byte-for-byte, not just approximately. This is **not a model decode benchmark**.

| Blade | Baseline, 7 submits | One submit | Early publish, 2 submits |
|---|---:|---:|---:|
| .50 | 3.129 ms | 3.169 ms | 3.110 ms |
| .51 | 3.129 ms | 3.171 ms | 3.110 ms |

Early publish is a small native win; the one-submit variant is slower despite
reducing CPU call time. Reducing host wall is not equivalent to reducing token
latency: delaying all GPU execution until recording finishes loses overlap.
Paired-round bootstrap bounds and individual samples are retained in JSON.
These synthetic bounds do not establish production noninferiority.

Separate CPU-hook means on `.50`, ms/synthetic block:

| CPU wall ledger | Baseline | One submit | Early publish |
|---|---:|---:|---:|
| Submit calls | .358 | .089 | .136 |
| Recording intervals | .055 | .048 | .050 |
| Command resets | .007 | .005 | .005 |
| Other wall | .192 | .088 | .105 |
| Fence-call blocking | 2.548 | 3.056 | 2.828 |
| **Sum** | **3.160** | **3.286** | **3.124** |
| Outside fence calls | .611 | .230 | .296 |

The CPU-only preload hook is copied from the archived `prod-vk-wall-v1.c`, not
the fdinfo/engine variant. Recording intervals are unioned, not double-counted.
`vkResetFences` is included in residual wall. Hook logging perturbs recording
and scheduling; only the separate uninstrumented runs support latency claims.
Do not extrapolate this table to eight-rank TPS or subtract it into “GPU time.”
The requested approximately 1 ms/token total host-overhead goal is **not achieved**.

## Validation and next gate

On both blades:

* Group baseline parity, ordered-upload reuse, cross-command dependencies,
  bounded spills, nested batches, invalid states, empty scopes, explicit
  readback, published-prefix waits, and abort/recovery passed.
* Existing batch, pipelined flush, static replay and GPU-profile tests passed.
* Real QSA orchestration tests passed: resident selection and partial-tail
  gathers, reused slot uploads across queued gathers, readiness/fallback guards,
  128K segment boundary, causal/split attention, eviction, fetch and persistence.
* Flag-budget audit passed. Offline helper rejects forbidden blades/releases;
  the shell physical-host guard was tested with a simulated `.42` interface.

Full model greedy hashes, 32K/128K behavior, routing/head integration and harness
performance still require a fleet slot. No pack exists on the tested offline
blades. **Do not promote either mode until all greedy hashes are unchanged and
the paired production harness passes.** Mode 2 is the first fleet candidate;
mode 1 is retained as an explicit counterexample/control.

## Reproduce

Build `all tests/test_fg_vk tests/test_qsa_prefill` in an isolated
`/home/xander/flash-gordon-live/YYYYMMDD-host-overhead` release; use a project-local
`TMPDIR`. The experiment build/capture scripts are saved beside the evidence.
From Windows:

```powershell
& .\tools\decode-instance.ps1 -Action replay `
  -Release /home/xander/flash-gordon-live/20261016-host-overhead `
  -FleetScript D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1 `
  -OfflineBlades @(50,51) -CaptureTag final
python .\tools\analyze-host-replay.py `
  --native .\results\hillclimb\host-overhead\operator-replay-plain-final.json `
  --trace .\results\hillclimb\host-overhead\host50-results.tar.gz `
  --output .\results\hillclimb\host-overhead\summary.json
```

`replay` never launches a ring rank, opens a fabric connection, reads a pack, or
stops another process. Its defaults target `.50`, never the production ring.
