# a038dec: completed local-head N4 paired diagnostic

**Diagnostic only, default off, no promotion.** User-direct exclusive lease
on .42–.49; conservative start **2026-10-16T21:03:03Z**,45-minute absolute
deadline21:48:03Z. All eight current sessions2/wide1 services were independently
restored/verified/cleaned at **21:15:32Z**. No lease remains active.
Future requests go directly to the user/delegated authority, not sol-abqual.

Candidate **`a038dec46287439a2b2f63d725fe13aab815b661`**,
binary SHA `6c23c7b5e0b0510d9154cc8c61639efa1f7449e7461d5be41b30d0cc60c57868`,
release SHA `309f906513dfb19098e1382317771e7fe0df0ba457262da71786e25fc85e3f3f`.
Same sealed mix04 target, shaders and binary for A0/F3/A1; full authenticated
Q8 draft head, not the new smoke-test vocabulary subsets. F3: fixed k3,
single session, mask330,440 MiB actual worker QSA cache and mandatory eager
admission. No FP16/BF16 transport, clocks/governor changes or serving promotion.

## Paired throughput

Each arm has two chat/two code prompts, three repeats: six requests/category.
For each F3 request, off baseline is the mean of matching A0/A1 case+repeat.
Report median off baseline, median F3, and separately median paired ratio.
The ratio of medians need not equal the median of ratios.

| Category | Off decode TPS | F3 decode TPS | Median paired ratio | Off client-wall TPS | F3 client-wall TPS |
|---|---:|---:|---:|---:|---:|
| Chat |31.063354 |22.378404 |**0.735812×** |23.875554 |18.311201 |
| Code |31.199628 |33.409332 |**1.074378×** |26.172023 |27.500518 |

Code gains modestly on this short-context set; fixed-k3 chat regresses.
Client-wall includes HTTP/prefill overhead. Decode headers are cross-checked
against retained per-request round/stage and token-hash receipts, not presented
as independent wire/GPU throughput. No long-context/concurrency or universal
speedup claim follows from six requests/category.

## Measured round ledger

Means over **performance requests only**, excluding cold greedy warmup and
stream tests. Chat246 rounds; code225. N2/N3 generation-tail rounds remain
included rather than silently filtered.

| Stage, ms | Chat | Code |
|---|---:|---:|
| Draft/coordinator chain span |9.851325 |9.853120 |
| Checkpoint preparation span |9.830561 |9.832129 |
| **Subtract overlap** |**−9.830809** |**−9.832307** |
| Verify |90.176415 |89.967760 |
| Target retained-prefix repair |0.560862 |0.259204 |
| Native repair |0.527350 |0.884311 |
| Commit |0.260419 |0.284796 |
| Host |0.024602 |0.031071 |
| **Total stage time** |**101.400756** |**101.280018** |

Total=draft+prepare−overlap+verify+target/native repair+commit+host, within
printed-timer rounding. Preparation queues behind rank7's native chain;
its9.83 ms span includes that wait. It is not another independent9.83 ms
expense or evidence of9.83 ms checkpoint compute saved. This is checkpoint/
draft overlap, **not draft execution overlapping target transformer verify**.

`MTP_ROUND` mean101.393089/101.272342 ms differs by a few logging microseconds
from the later stage total. Round p95:104.582/104.619 ms. Repaired-round
means: **101.219683 ms chat (186),101.024947 ms code (75)**; full-retention
means101.930650 ms (60)/101.396040 ms (150). The older158 ms figure was a
different candidate/cohort, not a new paired control for a percentage claim.

Observed yield **2.219512 chat /3.386667 code tokens per verification**.
Accepted draft fraction **0.418033 /0.811659**, using the recorded accepted/
proposed counts. Generation limits/EOS and sequential acceptance mean this
is not a single independent per-position acceptance probability.

### N4-only target compute and residual

Chat243/code219 N4 rounds; do not combine these N4-only averages with the
all-shape stage table as if populations were identical.

| Actual N4 quantity, ms | Chat | Code |
|---|---:|---:|
| Owner compute sum |65.625058 |65.159553 |
| Owner input+compute+read sum |68.136477 |67.666822 |
| Verify wall minus that service sum |22.445123 |22.629151 |
| N4 stage total |101.886160 |101.690799 |

The residual includes head, transfer, control/host and overlap, **not isolated
wire time**. The earlier measured-frame .48/.49 pair→eight-work-hop model
**14.215694±0.440437 ms** is reused with its original receipt SHA; no fresh
wire-only or synchronized one-way measurement is claimed this lease.

| Rank | Chat N4 compute ms | Code N4 compute ms |
|---|---:|---:|
|0 |8.055436 |8.206795 |
|1 |9.255695 |9.246151 |
|2 |7.391683 |7.254055 |
|3 |8.434399 |8.357320 |
|4 |7.490584 |7.472096 |
|5 |8.400255 |8.115804 |
|6 |7.922321 |7.835342 |
|7 |8.674683 |8.671991 |

Vocabulary ranks0/1/2/4 each still perform four serial head calls per N4.
Their compute sum≈12.203 ms is **overlapping work, not extra verify wall**;
mean per-round maximum head compute≈3.096 ms for code. Batch-head savings
should target the critical span, not subtract the sum as a serial latency.
Native-owner rank7 requested-live admission: **14,457,438,040 bytes**,
38,076,584 bytes below13.5 GiB; no dual-session/deferred long-context fit claim.

## Correctness scope

The authentic first cold N4 now completes. Its stage total1404.942 ms
(verify1315.223 ms) is cold-first-use evidence, excluded from warm perf.
**775 complete stage ledgers** align with all eight owners; every owner
verification reports status0. **2,167 actual MTP choices** equal their pass-local
target argmax. SSE parity, custom stop, cancellation and subsequent sequential
chat/code requests pass. A0/A1 off identity passes; diagnostic F3 identity is
6/8, with chat-1/chat-2 differing. Bitwise N1/N4 identity is not the numerical
gate, and complete same-prefix margin/quality qualification is not supplied.

The original-Q8/current-mix04 **N1** comparison on80 forced-first frontiers
reports KL0.051191953/top1 85%; all-hop FP16 and mask224 cumulative controls
also fail. This is not N4 attribution. Precision branch
`9531f7e4130c582cb52bef1f4e14c7f4cff6ad5a`,
`results/hillclimb/FP16_HOPS.md`/`fp16-hops.json`, supplies authenticated control
provenance. The later480-raw reconciliation reproduces mix04/Q8
KL0.003128578/top1 98.9583% and direct Q8/stack KL0.003084267/top1 98.75%.
The stack0.000246435/99.7917% figure is **incremental versus mix04**, not Q8.
These cohorts/paths are different; native-prefill versus frozen N1 conditioning
is being reconciled. No threshold changes, KL subtraction or revocation of
the older480-cohort pass is implied.

## Next work

Use observed≈65 ms owner compute, not the old47.493 ms proxy. Prioritize dense
verify weight reuse/GDN spill reduction and a **real serial fallback for chat**.
Curated restricted vocab remains a draft-cost/residency experiment with unknown
acceptance, not the head used here. Preserve the original-Q8 gate and resolve
the80/480 execution/conditioning discrepancy before qualification.

The roadmap is rebased on actual101.280 ms code rounds and observed yield.
Its aggressive, conditional≤60 ms budget needs≈25 ms owner-compute reduction,
head critical-span reduction, faster drafting and measured disjoint overhead
savings. Optional all-hop FP16+scale remains unimplemented/unqualified for
MTP and blocked by explicit codec/native-owner interlocks;≈7 ms saving is a
conditional model, not an enabled lever or achieved result.

## Restoration and retention

Before and after sanity use the original API/model/greedy response.
All eight original executable/shader/manifest SHA values and exact argv/FG
environment are verified. Watchdogs were cancelled only after restoration.
Controller exported all eight weight-free evidence archives, removed owned
scratch, and independent verification proves root absence twice, no owned
processes and helper port26840 closed. Final independent release **21:15:32Z**,
well before21:48:03Z. Original packs, overlay and launchers remain unchanged.
Owned local copied head/embedding are removed after hash retention; original
source archives remain available.

Machine report `mtp-retain-serving.json`; bundle
`mtp-retain-serving-evidence.tar.gz`; seals
`mtp-retain-serving-publication.json`. These are new artifacts; prior failed
diagnostics and offline-repair evidence are unchanged.
