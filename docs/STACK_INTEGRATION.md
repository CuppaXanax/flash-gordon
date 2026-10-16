# Stack 1 integration

Prepared offline; **no fleet access or deployment**.

Base main: `a63a166`.
Integrated roofline `e073c5d`, local n-gram cache `8c4f9be`, and host-overhead
`198583f`. Merge commits are `7676d38`, `07a08bd` and `ddd0146`.
All experimental flags remain default off, including `FG_DECODE_SUBMIT_GROUP`.
The sole host/roofline conflict in `dispatch_impl` retains both the submit-group
active-batch guard and roofline's layer-dispatch ledger; invalid group dispatches
are rejected before ledger recording.

Commit `48c6bff` carries only the MTP `a6344e6` Q8 activation-address corrections
and their independent two-block scalar regression, not MTP serving/drafting:

* Shared normal/roof gate-up GLSL uses `act_base + 2 + lane*8` in both paths.
* Generic cooked gate-up uses `2 + lane*8`.
* Q8_K has two header words. Activation LDS is local to the current 256-value
  block; the global subblock index must not address it.
* The Q8 fused test now spans 512 inputs and checks generic and fused mid
  results against independently dequantized scalar FMA/SiLU reference values.
  GPU-vs-GPU common-mode agreement alone cannot pass.

The always-active Q8 correction can change results even with flags off.
Do not claim old-main bit identity. Full-model frozen-corpus KL/top-1 is still
required for roofline/numeric stacks. Submission grouping itself must additionally
preserve canonical hashes against the identical stack with grouping disabled.

## Offline validation

From this worktree:

```powershell
wsl -d Ubuntu-24.04 -- bash tools/test-stack-offline.sh
```

The script uses project-local compiler/cache scratch and disables core files.
It builds native C17 with warnings-as-errors, all shaders and the resident
quality helper; runs the flag-budget gate, sampler/chat/chat-runtime CPU tests,
software-Vulkan roof controls, two-block Q8 oracle, byte-exact n-gram cache
lookup, submit-group correctness and QSA readiness/ordered-upload/persistence
tests. All these selected tests passed on local llvmpipe.
Python roofline/fusion ledger tests also passed (18 tests).

This is not RDNA2 qualification. A separate broader software-Vulkan attempt
printed three first-shape dense parity passes, then aborted with heap corruption;
HC up/mix rejected the unsupported roof capability profile. These are recorded,
not counted as passes or attributed to a proven root cause. Target-RADV dense/HC/
GDN hardware oracles, model KL, grouped same-kernel greedy hashes and serving
performance remain pending under the next lease. Full `make test` was not run:
unrelated legacy tests contain prohibited non-project scratch paths.

The paired gate and candidate batch live on `cuppaxanax/ab-harness`, not merged
here. Use that branch's `docs\CANDIDATE_STACK.md` and concrete pinned example;
default planning never touches the fleet. Execution must wait for both the
multi-session and MTP leases to release and a new exclusive .42–.49 lease.
Current production and immutable original releases/packs were not modified.
