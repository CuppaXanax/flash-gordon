# Authorized local-chain N4 diagnostic: incomplete, restored

Lease19:56:06Z–20:41:06Z,2026-10-16; final independent all-eight release
verification **20:23:07Z**. No promotion. Original current sessions2/wide1
launcher/binary/shaders/manifest/argv/FG environment verified; API/model/greedy
restored, own processes absent, helper port closed, scratch absent twice.
No hardware/governor settings were changed.

## Actual measurements

The first attempt measured40 validated protocol6 codec/CRC TCP records:
N1..4 ×ordinary/initial ×five rounds,100 timed samples plus10 warmups each.
Actual N4 frames are163,936 /204,896 bytes; ordinary/initial mean RTT/2 is
**1.728931 /2.113174 ms**. The7ordinary+1initial pair-to-eight-work-hop model
is **14.215694±0.440437 ms** (95% CI across five paired rounds), versus the
old unmeasured proportional18.242224 ms.
This is measured application-frame RTT/2, **not isolated wire-only latency,
synchronized one-way time or separately measured eight-hop ring transport**.

Retry MTP-off A0 completed12 perf requests:

| Category |Requests |Median decode TPS |Median client-wall TPS |
|---|---:|---:|---:|
| Chat |6 |31.978319 |24.243704 |
| Code |6 |31.988146 |26.710363 |

F3 failed on its first cold N4 pass; A1 was not run. **MTP-on TPS, acceptance,
paired speedup and completed-round/stage timing are unavailable.**
The retained first-pass owner compute times, ranks0..7, are
225.732,168.665,143.724,144.032,143.480,146.429,144.534,147.992 ms
(sum1,264.588 ms). Rank7 first k3 chain is87.328 ms.
These are **cold failed-pass values**, not the warmed47.493 ms owner proxy
or8.989 ms native-chain benchmark. Vocabulary timing rows and owner
input/read/forward rows remain in the machine-readable report and raw logs.
No missing `MTP_STAGE` ledger or transport/head residual was manufactured.

## Failures and scope

Attempt1: rank7 backing-file allocation ENOSPC. Retry: coordinator and worker
cache ceilings both440 MiB; own stopped QSA files removed between phases;
owned native overlay/head source copies deleted after successful loading.
This changes disk peak, not trained weights. One-layer owners still clamp to
their complete309 MiB cache. Worker limit is default640 unless the new
registered `FG_QSA_WORKER_CACHE_MIB` override is supplied.

Retry reached eager full-head admission: rank7 requested-live14,457,438,040B,
38,076,584B below13.5 GiB cap. No dual-session/long-context fit claim.
All eight target owners completed first N4 compute, and all seven remote
retained-prefix handlers logged success. The coordinator then failed its
retain/restore transaction (visible restore status5); the original error is
not established because the rollback can encounter queued acknowledgments.
The distributed runtime refused reuse and the controller restored production.

Both outcomes have `productionVerified=true`, `scratchRemoved=true`,
`complete=false`, no finalization errors. Full same-prefix divergence and
same-geometry original-Q8 N4 KL/top1 remain pending. A diagnostic greedy
identity comparison is not a substitute for the new gate.

## Reproduce analysis

`tools/mtp_live_analyze.py` normally requires completed diagnostic off/F3/off.
Its explicit `--allow-incomplete` mode reports this single failed cold pass
and refuses to invent on-arm TPS/acceptance. Raw run receipts are weight-free
in the supplemental evidence archive; retain both first failure and retry.
See [JSON](mtp-local-n4-serving.json) and
[production roadmap](../../docs/MTP_TO_PROD.md), including the newly allowed
draft-only lower-bit formats and speculative rank7 future-window drafting.
