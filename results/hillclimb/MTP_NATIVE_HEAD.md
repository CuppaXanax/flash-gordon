# Native MTP head: .51 execution milestone

Branch `cuppaxanax/mtp`. **Not a serving rollout or end-to-end exactness result.**

## Measured

Pinned abliterated native weights execute through the complete HC-aware
Vulkan head. Cooked Q8/F32 weights: 2,772,584,448 bytes; independent F32 KV:
4096 bytes/cell. Placement API requires the manifest's final target owner.
The benchmark's default manifest assigns that owner to rank 7.

Three runs, each 16 warmups plus 32 timed cells (positions 16..47):

| Run | Core ms/draft | Local full-vocabulary + argmax ms |
|---:|---:|---:|
| 1 | 3.063 | 5.748 |
| 2 | 2.983 | 5.698 |
| 3 | 3.140 | 5.726 |

Means: **3.062 ms core**, **5.724 ms full local head**, **8.786 ms total**.
The HC input is synthetic; token-198's embedding and all weights are real.
No operating point was changed. Sysfs reports 1000/1500/2000-MHz states and
idle 1000 MHz before/after; loaded fixed-2000-MHz operation was not measured.

## Correctness evidence

* Independent raw-HF-order NumPy full-cell oracle: HC scaled error 2.87e-6,
  contracted error 1.84e-5, full-vocabulary KL(reference || GPU) 3.10e-12,
  equal first-cell argmax. **Synthetic parity, not real-model Q8 quality.**
* Dense attention: 16 causal rows, max absolute error 5.53e-8; hostile future
  KV invisible. Full native last-cell truncate/rebuild is byte-exact in all runs.
* Retained tensor/context lifetime and wrong-owner rejection pass.
* Q8 expert routing exposed old activation-header/global-offset bugs in both
  generic and fused gate/up. Fixed both. Regression now uses two activation
  blocks and an independent CPU dot/SwiGLU oracle, not GPU-vs-GPU alone.
  Existing Q4/Q5 fixtures retain their original shape and pass.
* Small-microbatch owner scratch now includes the fixed selection/live tail.
* Werror build, flag registry, MTP/sampler/depth-B host suites, native tests and
  Q4/Q5/Q8 routed-expert selectors pass. Invalid benchmark counts are rejected.
* All 414 archived source-file hashes match the uploaded source archive;
  local files match exactly (350) or differ only in line endings (64).

## Scope and next gate

Serving remains default-off/fail-closed. Native embedding/head-fanout hookup,
causal all-rank verification/commit and MTP-on/off same-kernel greedy equality
are **not implemented/qualified end-to-end**. Previous small-N block ratios
remain **1 / 1.519 / 2.199 / 2.505**; N=2 <=1.3 is not claimed.

Need a production-ring lease or true final-HC captures of **prompt plus greedy
continuations**, chat and code: 4x2560 F32 pre-final-mixer HC, tokens/positions,
next embeddings, target labels and exact pack/binary hashes. Partial-layer
states or cold decode-only draft KV are not normal acceptance measurements.
No production ring was accessed; only .51 was used.

Strata E=2.4–3.2 remains a published estimate, **not measured FG acceptance**.
A conditional future scenario (40–45 baseline TPS, N=4 target ratio 1.8, k=3,
measured native core, ideal four-way division of local head time) gives
**41–60 TPS before fanout/commit**. The measured monolithic proposer gives
34–48 under those same assumptions. Neither is current ring TPS or MTP's
ceiling; the distributed head and remaining verifier work must be measured.

Raw timings, input/source/binary hashes, clock observations, independent
oracle, diagnostic first cell/trace and test logs:
`mtp-native-head-evidence.tar.gz` (SHA256
`9729e8021084a420a0c7af016a48543fd24b30b90e2fc98ec455aef4855dd64f`).
See `mtp-native-head.json` and `docs/MTP.md` for contracts and reproduction.
