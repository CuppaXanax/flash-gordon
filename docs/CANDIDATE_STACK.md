# Candidate stacks: offline preparation, leased execution

The initial preparation was offline. Run only after
both multi-session and MTP leases have ended and a new exclusive .42–.49 lease
has been granted. Planning is the default; nothing is scheduled automatically.
The leased execution baseline is adaptive `2d0f8a7` plus the sealed mix-04 pack,
sessions 2. The older sessions-1 example is illustrative, not its deployment.

## Prepared integration

As originally prepared, `cuppaxanax/stack-1` contained main `a63a166`, roofline `e073c5d`, latency
`8c4f9be`, host-overhead `198583f`, and the three MTP Q8 activation-address
fixes from `a6344e6` adapted to roofline's shared gate/up GLSL. That preparation did not
merge the MTP implementation or multi-session branch. Experimental flags
default off. The always-active Q8 correctness fix means flag-zero stack-1
must not be assumed numerically identical to the old main shaders.

The integration was subsequently merged with adaptive `2d0f8a7` in `31508ee`,
retaining per-session static bindings and separate static descriptor-pool limits.
It includes multi-session scheduling but not MTP serving. New experimental
flags remain default off.

The example declares two complete flag combinations, not nine separate flag
experiments: `stack-all`, then identical kernels with submission group 2.
Each combination gets **one fresh full-vocabulary KL/top-1 comparison and one
six-pair performance experiment** versus current main/mix-04. Teacher logits
are captured once per batch. Failed quality skips performance. Every performance
run must independently satisfy [policy v2](AB_QUALIFICATION.md).

Supported flags are `FG_ROOFLINE`, `FG_ROOF_DENSE`, `FG_ROOF_QSA`,
`FG_ROOF_EXPERT`, `FG_ROOF_HC`, `FG_ROOF_RECURRENT`, `FG_ROOF_GDN`,
`FG_NGRAM_LOCAL_CACHE`, `FG_EXPERT_Q8_WIDE`, and `FG_DECODE_SUBMIT_GROUP`.
Values are canonical decimal uint32; cache/wide are 0/1, grouping is 0/1/2.
Other names, shell text, unsafe releases/manifests, and contradictory `wide`
values fail before transport.

### Submission grouping: exact greedy control is mandatory

Grouping changes submission/fence behavior, **not kernels**. A candidate
selecting group 1/2 must supply either an earlier `hash_control` candidate or
a `greedy_control` deployment with the
same binary, shader tree, pack, helper, sessions and all other flags.
The example uses `stack-all` as the group-2 control. The completed control
must be quality-approved, have at least five policy-v2 pairs, and have stable
canonical B hashes across its rounds. Every grouped B hash must match that
control; missing/duplicate/changed cases fail. Passing KL never waives this.

The cheaper `greedy_control` alternative deploys the otherwise identical
group-zero configuration twice, capturing all five canonical hashes after
each fresh deployment. Both captures must agree; every grouped B round must
match them. This control is not a performance experiment and supplies no
grouping speedup claim. The combined candidate still needs its own full
KL/top-1 gate and all predeclared paired rounds.

This control is intentionally same-kernel, not old-main-vs-roofline:
legitimate roofline/Q8-fix numeric changes may differ from old main.
`same_kernel_hash_identity` is recorded separately from global A/B identity.
Both performance experiments compare against main; their difference alone
is not a paired causal estimate of the grouping increment.

## Build artifacts and quality provenance

From the stack-1 worktree:

```powershell
wsl -d Ubuntu-24.04 -- bash tools/test-stack-offline.sh
```

This builds the native server, every shader, source-matched `q4-eval-batch`,
and targeted local tests, with all scratch/cache files under `.build`.
The frozen 480-frontier corpus is `tools\corpora\stack-480.json` on the harness
branch (copied unchanged from the aligned-480 Q4 qualification corpus).
The 80-frontier built-in screening corpus is not accepted.

Local prepared bundles:

| Worktree-relative file | Contents |
|---|---|
| stack-1: `.build\stack\stack-1.tar.gz` | `flash-gordon`, `vulkan\*.spv`, `q4-eval-batch` |
| ab-harness: `.build\stack\main-quality.tar.gz` | main-source `q4-eval-batch` only |

Their digests and validation limits are in
`results\hillclimb\STACK_OFFLINE.json`/`.md`. B executable/helper hashes and
shader-tree digest in `candidate-stack.example.json` match this local build.
Rebuilding elsewhere can change ELF debug-path/build bytes: update hashes
from the actual immutable staged artifacts, never invent ready hashes.

After receiving the new lease, stage the bundles with `ab-fleet.ps1` and
explicit `-Targets @('192.168.42.42', ... '192.168.42.49')`:

1. Create a fresh owned `/home/xander/hc-abq-stack-stage-*` on all eight.
   Upload archives there; verify the recorded archive SHA256 **before**
   extracting. Reject existing destination releases.
2. Extract the stack bundle into
   `/home/xander/flash-gordon-live/20261016-abq-stack-1`, preserving its root
   layout. Verify server/helper SHA256 and runtime library availability.
3. Extract only the teacher helper into the **new**
   `/home/xander/flash-gordon-live/20261016-abq-quality-main` on all eight.
   Do not add files to either immutable original release or current main.
4. Retain sealed packs on their owners. Remove only the owned staging scratch.
   No production build, pack, launcher, or deployment is modified by planning.

The capture helper must be built from its corresponding serving source;
both helper and serving ELF hashes are pinned. All owners' complete SPV-tree
digests are checked before capture and must agree. The discovered A digest
is attached to the teacher identity; B's supplied digest is enforced.
Every later A/B deployment rechecks the tree. This is important because the
Q8 fix changes shaders without necessarily changing the serving ELF.

Matched A/B sessions may be 1 or 2. The affinity-aware battery uses stable
OpenAI `user=hillclimb-stream-0` for each cold/continuation sequence and
distinct users 0/1 for concurrency. Its `paired-affinity-v3` identity prevents
resuming older traffic as though it used the same cache-routing workload.
Quality frontiers use fresh single-logical-session helper runtimes; they
qualify arithmetic, not concurrent scheduler isolation.

An optional `teacher_directory` reuses an existing teacher only when its
captured deployment identity exactly matches A, including the discovered
shader digest. Each candidate still needs a fresh capture and quality receipt.

Receipts bind both deployment identities, complete flags, helpers, shader
trees, manifest, corpus and raw full-vocabulary KL report. Capture uses a
fresh resident runtime per frozen frontier, rank-4 logit dumps and regular-file
only archive extraction. Coordinator logs and transport receipts survive
failure. Stops are exact PID/executable/manifest/prompt scoped; cleanup refuses
foreign/symlink scratch and never name-kills or force-kills.

## Plan and execute

Run from the ab-harness worktree:

```powershell
pwsh -NoProfile -File tools\candidate-stack.ps1 -ConfigPath tools\candidate-stack.example.json -Output .build\stack\plan-new
```

This only validates input and writes a plan. It never calls deployment,
capture, quality comparison, or HTTP tools.

For execution, provide a current operator assertion, for example:

```json
{
  "multisession_released": true,
  "mtp_released": true,
  "targets": ["192.168.42.42", "192.168.42.43", "192.168.42.44", "192.168.42.45",
              "192.168.42.46", "192.168.42.47", "192.168.42.48", "192.168.42.49"],
  "expires_at": "REPLACE_WITH_THE_GRANTED_LEASE_EXPIRY"
}
```

```powershell
pwsh -NoProfile -File tools\candidate-stack.ps1 -ConfigPath tools\candidate-stack.example.json -Output results\hillclimb\stack-run-new -Execute -LeaseFile .build\stack\lease.json
```

The lease is checked before capture transport, each candidate, and every
performance deployment. Reserve time for quality, all predeclared pairs,
cleanup and restoration. Restore is allowed even if the lease expires.
Do not run concurrent operators.

The batch always restores configured **A/current adaptive+mix-04**, not the
historical original, even on capture/performance failure. Its output contains
`restore-baseline.ps1`, a one-command restore hook. No automatic promotion
occurs; eligible does not mean deployed. If a same-kernel control fails quality,
dependent grouping candidates are reported blocked rather than silently
waiving hashes. Preserve all raw captures, paired samples and receipts.

For an eligible adaptive-stack result, `stack-pin-selected.ps1` separately
pins the complete flags and hash-checked launcher, restarts the whole ring,
checks all eight live executable/manifest/shader identities and exact argv/FG
environment, and repeats the five canonical hashes. `-PlanOnly` renders
LF-only pinned scripts without transport. Execution requires the current
lease and a complete primary-gate/quality/exact-grouping winner. Failure
restores A and its retained pre-stack launch pointers. Persistence matches
the original `nohup` semantics, not an automatic host-boot service.
`ab-rollback-pre-stack.ps1` is the one-command fallback to adaptive
`2d0f8a7`/mix-04/sessions 2, not the older sessions-1 rollback.

`stack_collect_evidence.py` copies compact receipts and framed prompts,
hash-indexes full-vocabulary payloads retained locally, and refuses an existing
destination. Raw logits and their large archives must not be pushed to GitHub.

`CaptureScript`, `PerformanceScript`, `QualityScript` and `DeploymentScript`
overrides exist for trusted local fixtures. Tests replace every mutator and
never contact a blade. Production operation should use the defaults.
