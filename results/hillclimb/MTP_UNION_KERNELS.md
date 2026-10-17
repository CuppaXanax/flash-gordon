# MTP: expert-union measurement gap and HC kernel repair

## Status

Offline research only, on .51; production was not accessed. Target weights and
the full draft vocabulary are unchanged. Whole-block N4/N1 is **2.167x**, not
the requested 1.5x. There is no new authentic rank7 replay, chat/code TPS,
acceptance, complete-model quality result or promotion.

**Actual real-draft target expert-union bytes remain UNMEASURED.** Retained
captures contain native drafts, teacher tokens and final HC, but no N4
per-layer target routes. Sliding teacher windows cannot substitute for rejected
drafts. A fresh explicit diagnostic capture lease is needed.

## Selected repair and replay result

`gr_read_batch_into` previously disabled native HC up+mix fusion for every
`N>1`. Default-off `FG_SMALL_KERNELS` bit **262144** now enables N2/N3/N4
compact-IQ4/cooked-Q8 variants: load/decode each packed fragment once, accumulate
N FP32 columns, then fuse sigmoid mix and injection reduction. No new persistent
GPU tensor or target precision reduction is introduced. N1 is unchanged.
Requires the existing native-HC roofline enable and subgroup64; unsupported
formats retain the previous fallback. HC down is not changed.

Warm, static-replay, synthetic six-layer mixed-expert block, fixed prefixes
264..479, same binary/shaders and process-only unlimited memlock. No clocks or
governors changed. Baseline mask131402; candidate393546.

| A/B/A bracket | N4 baseline mean of A medians, us | N4 B median, us | B/A | N1 B/A |
|---:|---:|---:|---:|---:|
|1|5097.084|4936.846|0.968563|1.000492|
|2|5124.621|4973.580|0.970527|0.994303|
|3|5113.575|4961.364|0.970234|0.999426|

Median N4 cost falls **2.9766%**; N1 is unchanged within noise
(median B/A0.999426). Candidate N4/N1 median **2.166612x**. Terminal output
and GDN-state hashes match baseline in each bracket. Synthetic hash agreement
does not establish real-model greedy exactness or quality.

Independent component-oracle corrected-mode microbench medians at K320:
IQ4 N2/N3/N4 **35.175/48.775/64.110 -> 24.211/38.136/46.690 us**;
Q8 **38.429/55.003/56.436 -> 26.617/37.276/46.049 us**.
The first microbench used the wrong small-N mode; its timings are excluded.

## Remaining op gaps and rejected experiments

Three additional intrusive A/B/A profile brackets, reported only for
attribution, give candidate within-bracket median N4/N1 ratios:

| Operation | Candidate N1 us | Candidate N4 us | N4/N1 |
|---|---:|---:|---:|
|Dense/shared|858.68|2079.48|2.426|
|Expert gate/up|437.56|1519.48|3.473|
|Expert down/reduce|295.68|768.36|2.624|
|HC/projections/small ops|436.32|1023.08|2.356|
|Routing/union|166.28|300.00|1.804|
|GDN conv/recurrence/norm|105.76|216.32|2.045|
|QSA attention/cache/index|277.28|350.76|1.268|
|Inter-dispatch gaps|131.92|179.56|1.361|

Component medians do not necessarily sum, and profile synchronization changes
the workload. Unmodified operation ratios also drift across brackets; use the
nonintrusive replay result above for the block gain, not profile totals.
The synthetic GU/down ratios are **not evidence of authentic draft unions**.
Dense and HC still miss1.1–1.3x; this is a partial HC repair, not a closed dense
gap or a proof that same-weight-byte work must attain that ratio.

* Shared-LDS HC dequantization passed component checks but lost to register
  fusion and changed terminal hashes; removed.
* Dense K256 shared-LDS tiling passed Q4/Q5/IQ4 component checks but representative
  N4 projections cost2.2–2.5x the existing implementation; whole-block cost
  regressed about29%. Removed, with rejected sources retained in evidence.
* A private nominal FP32 arithmetic probe is not an ISA-validated quantized-kernel
  ceiling and is not used for a feasibility claim.

## Actual-verifier capture and byte-weighted floor

Existing `--mtp-capture-dir` now appends
`verify-routes-SESSION-LAYER.jsonl` after each owned target block drains. Rows
preserve N, first position, replay classification and all ten expert IDs/token;
overlapping rejected branches are not overwritten by position-addressed writes.
Traced runs must not be used for performance timing.

`tools/analyze_mtp_union.py` requires the exact manifest SHA, source commit and
session-to-chat/code case bindings. It rejects incomplete48-layer windows,
duplicate window/layer rows, unknown sessions and malformed selections; restore
replays are excluded. It computes GU and down ratios separately using actual
cooked-bank strides, byte-weighting layers rather than averaging cardinalities.
These are logical unique-bank bytes, not measured DRAM traffic.

The stale CPU route-count hint previously rejected an N4 read after cached N4
replay following N1 recording. Readback now uses the drained frame's exact shape,
bounded by allocated capacity and initialization. A GPU regression proves the
recording callback was skipped, the cached N4 IDs match, and the old strict
guard reproduces the failure before the corrected guard passes.

The [roadmap](../../docs/MTP_TO_PROD.md#expert-union-physics-versus-the-kernel-gap)
gives the op-weighted conditional floor: assuming **3x** GU/down union bytes,
1.1x dense/HC and ideal/current-other pricing gives **1.593/1.735x N4/N1**.
That is an optimistic planning scenario, not a measured physical floor. The
authentic union, cooked format mix and per-layer N1 service ledger are required
before changing the realistic target or projecting fleet TPS.

## Validation and provenance

* C17 `-Werror` builds; CPU MTP, wire, native, embedding, vocabulary and capture
  tests pass;15 analyzer tests,2 lease-identity tests, flag registry and shell
  syntax pass.
*12 independent HC oracle cases, N2..4, two formats, K320/640: maximum normalized
  error **6.70552e-8**, with poisoned-tail and malformed-input checks.
* Existing N1 HC and Q8 down/union checks;12 union scheduling fixtures;
  N2..4 whole-block tolerance and bit-exact causal-prefix checks pass.
* New cached-N4/N1 route-shape regression passes; old guard fails as expected.
*24 state retain/rollback fixtures, two-slot independence/N1 transition, and
  eight localhost TCP ring configurations pass.
* All716 existing local compiled-source inputs match the closing source hash
  list, allowing only CRLF-to-LF source packaging normalization.

Baseline source `4b002778f21d9c1e63a5f246be572638637f827b`; exact source,
binary and shader lists are sealed in the closing raw evidence. Machine-readable
results and raw-archive SHA256 values: `mtp-union-kernels.json`.
Raw scripts, regressions, profiles, replay samples and rejected prototypes:
`mtp-union-kernels-evidence.tar.gz`.

.51 was cleaned/released **2026-10-17T00:48:08Z**: no owned process/listener,
governor inventory unchanged, owned root absent twice. No .42–.50/.52/.53
access occurred in this experiment. Full draft head retained; no unsupported
sessions2 residency or end-to-end speedup is claimed.
