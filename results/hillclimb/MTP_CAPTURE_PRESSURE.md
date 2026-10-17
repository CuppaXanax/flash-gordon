# MTP capture failure, CLI repair and compiled-kernel pressure

Default off; **no promotion or new serving TPS/quality/acceptance claim**.
Base: `9eea42f10b02044798d10327927094a9c80ff8d3`; source-input hashes and
the exact CLI repair patch are sealed in the evidence archive.
Machine-readable data: [mtp-capture-pressure.json](mtp-capture-pressure.json).

## Authorized capture-only lease: no authentic union result

The user granted exclusive `.42–.49` access at
**2026-10-17T00:54:23Z**, limited to15minutes and real chat/code draft-route
capture. Neither attempt reached an actual verifier round:

1. `union-20261017-0058`: a single nested PowerShell phase flattened into
   individual elements; the worker argument guard rejected it. Restored,
   verified and scratch absent twice at01:02:23–24Z.
2. `union-20261017-0102`: fixed phase nesting, but the coordinator exited with
   `flash-gordon: unknown api option: --mtp-capture-dir`.
   Seven workers had started; no requests or verification rounds completed.
   Every original service/binary/shader tree/manifest and pinned FG environment
   was restored and independently verified; all eight scratch roots were
   absent twice at **01:06:07Z**,11m44s after the grant.

There was **no further production access**. Restored source is
`31508eea70025a14fde14340c1b66cb4867536d9`, binary SHA256
`02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0`,
sessions2/wide1. `FG_SMALL_KERNELS` is absent and unsupported by that source.
Production uses native prefill followed by resident-state N1, not repeated
whole-prefix cold replay.

**Real-draft union-byte ratios and a measured-union op-weighted floor remain
unavailable.** The existing optimistic six-layer proxy is unchanged:

```
T4_us = 437.84*R_GU + 296.16*R_down + 1.1*(854.20+434.44) + 671.24
N1_us = 2693.88
```

An assumed GU/down ratio3 gives1.593x, or1.735x retaining current N4-other
cost. These are conditional planning values, **not measured unions, hardware
lower bounds or attainable serving ceilings**. No101ms-round rescaling.

## Repair and capture readiness

`api` has a separate option parser from `rank`; only the latter accepted the
diagnostic capture option. The API parser now configures it identically.
The new regression exercises both commands with existing/missing directories,
without a model or GPU. `make test-mtp-capture` builds the executable first.
The Linux C17/Werror build, eight capture tests, CPU MTP suite, two pinned
restoration-identity tests, flag registry and guard syntax checks pass.

Repaired Linux binary SHA256:
`772cf9c5805ca6d70558919728364eefc8d1b5509a9abd70df341ee96c96f7c4`.
The private replacement release preserves all stock shaders and the existing
embedding exporter, with regenerated release SHA manifest. The prepared
capture-only launcher has a single fixed-k3 phase and a9-minute work budget
(maximum10),13/14-minute restoration watchdogs and15-minute wall budget.
It probes **API and rank capture parsers before stopping any production
service**. A fresh explicit grant and fresh lease timestamp are required.
No claim is made that the repaired binary has already run on the full ring.

## Profile before variants: shared weights, no spills

Offline `.51`, RADV GFX1013; mask393546 and prior replay-faithful synthetic
six-layer mixed-expert geometry. `RADV_DEBUG=shaders,shaderstats` and disabled
shader cache were used **only for compiler captures**, never wall benchmarks.
The final machine-code lines, not duplicate NIR/ACO listings, were counted.

| Selected kernel | N | Weight-load sites | Activation-load sites | VGPRs | Compiler subgroups/SIMD |
|---|---:|---:|---:|---:|---:|
| HC IQ4 |1 |16 |2 |40 |24 |
| HC IQ4 |2 |16 |4 |64 |16 |
| HC IQ4 |4 |16 |8 |84 |12 |
| Dense reuse8 Q4 |2 |18 |4 |64 |16 |
| Dense reuse8 Q4 |4 |18 |8 |100 |10 |

HC load counts cover the weight loop between its first two machine barriers;
later descriptor reuse is excluded. Dense N1 uses a different native l2
kernel,64 VGPRs/compiler subgroup field16. Sampled kernels have **zero
VGPR/SGPR spills and zero scratch**.

This disproves per-token independent weight scans in these selected loops.
It does **not** measure DRAM traffic, achieved occupancy or establish the
dominant bottleneck. Compiler subgroup fields must not be converted into
hardware occupancy ratios. Increased arithmetic, activation issue, register
pressure and epilogue predication remain plausible contributors.

## Bounded, profile-led alternatives

Wall summaries use process sample medians and `B/mean(A0,A1)`, then the median
of brackets. Warmup/timed token budgets are264/216, N4 calls66/54; N1
calls264/216 where present. All use existing reassociation tolerance, without
loosening any numerical threshold.

| Alternative | Resource change | Whole-block B/A | Decision |
|---|---|---:|---|
| Half-wave HC row partition |VGPR84→64; instructions1571→838 |1.005606,3 brackets |Reject; terminal hashes change |
| Dense row4 vs row8 |VGPR100→72; twice as many workgroups |1.049470,3 brackets |Reject; hashes equal |
| Parallel-token HC epilogue |VGPR84 unchanged; VMEM sites46→31 |0.989994 initially,3 brackets |Confirm rather than select |
| Same epilogue, held-out confirmation |No further shader changes |0.995044,6 brackets |Hold unselected; modest signal |

N1 row4/control median1.000901; epilogue/control confirmation0.999463.
Using process **means**, the epilogue confirmation median is0.997768, with one
bracket slower. The original~1% saving does not reproduce at that magnitude.
All12 HC component cases pass for each prototype; the token-parallel
epilogue preserves terminal output/state hashes. No prototype is shipping.

For compact Q4 `in=2560, rows=6144`, isolated reuse8 N1/N4 costs are
41.039/64.463us (1.5708x); row4 is36.178/75.166us (2.0777x).
These are separate unbracketed component runs, not the weighted dense stage.
The initial strict component invocation omitted existing
`--allow-reassociation`; it failed at~7e-7 and is not a valid new regression.
The corrected run passes with the unchanged established tolerance.

Shipping shaders remain9eea42f. Its previously published synthetic
whole-block ratio **2.166612x** is unchanged, not a fresh serving result.
No precision or full draft vocabulary change; no adaptive-policy promotion.

## Cleanup and retained receipts

Offline `.51` released **2026-10-17T01:28:40Z**: governor inventory unchanged,
root-visible owned processes zero and `/home/xander/hc-mtp` absent twice.
The five bounded parent redirections from the earlier CRLF child-script
failure were separately removed and authenticated in `newline-cleanup.json`.
No `.50/.52/.53` access in this segment; no clocks/governors changed.

Evidence: [mtp-capture-pressure-evidence.tar.gz](mtp-capture-pressure-evidence.tar.gz),
**8,055,064 bytes**, SHA256
`a747dde7f3ed23b81c4ab39c07b842cb3deaf06c4a147df2c52da8acd38d7714`.
It contains raw compiler/replay archives, both failed fleet attempts and
restoration receipts, source/prototype scripts, exact CLI patch, repaired
binary and offline release proof. Large model/head/embedding files are
excluded. Next prerequisite is a fresh short capture lease; derive real
union physics before revising the whole-block ceiling.
