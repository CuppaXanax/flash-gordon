# MTP to production: measured blockers and the ≤60 ms path

## Status and binding policy

**Default off; no promotion.** The latest user-granted a038dec diagnostic
completed A0/F3/A1 and released all eight owners after independent verification
at **21:15:32Z**, within its conservative21:03:03Z–21:48:03Z lease envelope.
Current production is restored to the pinned **sessions2 / wide1** stack.
The cold retain fix is confirmed on the authentic ring. Median paired
off/on decode TPS: **chat31.063/22.378, code31.200/33.409**; median
per-request paired ratios0.736×/1.074×. Mean warm round stages total
101.401/101.280 ms. Chat regresses; this is diagnostic, not qualification.
See [paired receipts and stage ledger](../results/hillclimb/MTP_RETAIN_SERVING.md).
Future lease requests go directly to the user/delegated authority, not
sol-abqual.

The earlier authorized 2026-10-16 diagnostic lease
ended with all eight original services restored, pinned and verified, and
owned scratch/processes removed. Final independent release verification was
**20:23:07Z**, before the original **20:41:06Z** absolute deadline. Restored
production is the current stack, **sessions2 / wide1**, not the older rollback
configuration. Both attempts, including their failures, remain evidence.

**User ruling: the Q4 floor applies to target weights only. MTP draft-layer
and draft-head weights may go below Q4.** This supersedes historical blanket
floor wording; it does not authorize IQ3/Q2 target weights or weaker target
quality gates.

Sources: [diagnostic measurements](../results/hillclimb/MTP_LOCAL_N4_SERVING.md),
[machine-readable ledger](../results/hillclimb/mtp-local-n4-serving.json),
[previous local-chain measurements](../results/hillclimb/MTP_LOCAL_CHAIN.md).
Ranking also incorporates the user's uncommitted
`D:\looking-glass-labs\flash-gordon\results\hillclimb\STRATA_AUDIT.md`
(2026-10-16; Strata9259cad4, FG serving bc75b750). Its blanket prohibition
on Q2 draft experts predates the user ruling above and is superseded.

## Offline follow-up: retain repair and restricted vocabulary

The retain failure is **reproduced and fixed offline on .51**. Replicated
coordinator executors intentionally own all48 layers, but a ring pass
advances only the coordinator's actual block. Retention incorrectly required
every QSA layer, including untouched foreign layers, to reach the verified
frontier. It now validates only the journal's executed layer interval.
The old code fails with `MTP retained QSA frontier mismatch` when N1-only
warmup is followed by the first cold N4 at prefix22. The fixed GPU regression
passes24 cases over frontiers22/127, both private slots, all proper retained
prefixes for N2–4, full-state reconstruction, exact rollback and continuation
tolerance. Foreign frontiers remain unchanged. This fixes the originating
local condition; it does **not** add ACK-draining/network-error recovery.
That offline prerequisite has since received authentic full-ring confirmation
in the completed a038dec lease above.

Selected item2 before further dense-kernel tuning on expected ms/effort:
`--mtp-local-head` now accepts an authenticated `FGDVOC1` draft-only subset.
Full target weights/vocabulary are unchanged. Runtime authenticates every
original cooked source byte against sealed `output.weight`; corrupting an
unselected row also fails. Only selected rows become GPU resident, while
the full raw source remains in the container on disk. See the packer,
container layout and regression commands in [MTP.md](MTP.md).

Paired-process, replay-warmed measurements on **.51, reported24 CUs**
(8 warmups/100 timings;8 synthetic hidden vectors):

| Draft rows | Mean/head | Requested-live bytes | Saving vs full | Three-head-call saving proxy |
|---|---:|---:|---:|---:|
|248,320 |2.238927 ms |676,920,320 |0 |0 |
|40,525 |0.430886 ms |110,479,316 |566,441,004 |5.424124 ms |
|106,299 |1.018502 ms |289,784,684 |387,135,636 |3.661277 ms |

All IDs/scores match the independent existing raw-row loader exactly.
These are **contiguous low-ID smoke/performance sets, not curated
English/CJK sets or acceptance-qualified drafts**. The three-call difference
is not a measured full-chain or serving-round improvement; do not subtract
it directly from the older40-CU chain's8.989 ms. Both memory savings exceed
the historical0.24 GiB overrun, but complete new dual-session/deferred fit
still requires eager measurement. Subset language/tool coverage, acceptance
and live TPS remain unmeasured. See
[retained follow-up evidence](../results/hillclimb/MTP_RETAIN_SUBSET.md).

### Latest offline complete rank7 admission

The actual Strata ID sets, rather than the earlier contiguous smoke sets,
now pass exact independent raw-row GPU comparison. On the same reported24-CU
.51, full/English/CJK means are **2.237300/0.427269/1.016979 ms/head**.
Three-call differences are **5.430094/3.660964 ms proxies**, not measured
serving savings or guaranteed acceptance preservation.

All152 trained rank7 tensors were reconstructed and read back against sealed
production tensor SHAs, without production access. With the authentic native
Q8 overlay, authenticated head/embedding, target buffers and **both private
target/native sessions**, final N1..4 target and k3 native graph warmup gives:

| Head / worker QSA cache |Complete graph-warmed requested bytes |13.5 GiB headroom |Result |
|---|---:|---:|---|
|106,299 CJK /320 MiB |14,443,303,740 |52,210,884 |**PASS offline admission** |
|40,525 English /416 MiB |14,465,327,940 |30,186,684 |**PASS offline admission** |
|106,299 CJK /440 MiB |14,634,324,460 before second native session |−138,809,836 |Target admission rejects |
|40,525 English /440 MiB |14,455,019,092 before second native session |40,495,532 available vs50,348,032 reserved |Native admission rejects |

The cap is unchanged at14,495,514,624 bytes. These are **different explicit
cache configurations**: neither subset alone proves the440 MiB configuration
fits. Graph warmup adds further allocations, so an earlier
`target_verify_ready`/`native_all_ready` is not the final fit certificate.
This closes the bounded short-context two-slot allocation experiment,
**not production disk-space, long-context cache pressure or coexistence gates**.
The legacy serving controller still denies dual-session local heads at its
440 MiB configuration; a powered controller must bind the new cache/receipt.

Retained authentic **40-case N1-teacher** replay now measures the acceptance
tradeoff, with no new production capture. Both subsets produce the same
accepted-frontier schedule on this English chat/code cohort:

| Head |Chat tokens/round |Code tokens/round |Warm local code k3 draft |
|---|---:|---:|---:|
|Full248,320 |2.392523 |3.240506 |8.966019 ms |
|English40,525 |2.283675 |3.104912 |3.591989 ms |
|CJK106,299 |2.283675 |3.104912 |5.291276 ms |

These frontiers advance by `accepted+bonus`, clipped at256 tokens/case, with
native state repaired from retained target HCs. **136,386/136,386 draft IDs**
match an independent direct GPU/raw-row reference across all replay modes.
Code yield drops **4.184%**; target next-token coverage is96.270% code and
92.852% chat for either subset on this cohort. Do not present head latency
alone as a serving win. Transferring only the measured relative code yield
and local draft delta onto101.280 ms full-head rounds gives a **cross-cohort
planning model**, English≈1.012×/CJK≈0.994× versus full-head MTP; neither is
a measured new ring TPS result. New N4 acceptance, multilingual/tool coverage,
adaptive chat noninferiority and the six-pair performance gate remain open.
See [sealed offline results](../results/hillclimb/MTP_GOODPUT_LOCAL.md).

## Earlier failed lease: historical measurements

| Observation | Measured result | Limitation |
|---|---:|---|
| Ordinary N4 application frame |163,936 bytes; RTT/2 **1.728931 ms** |Codec/CRC/TCP pair .48↔.49, not synchronized one-way wire latency |
| Initial N4 frame, including ngram |204,896 bytes; RTT/2 **2.113174 ms** |Same pair; five rounds,100 timed samples/shape/round |
| Eight-work-hop transfer model |**14.215694 ±0.440437 ms** |7 ordinary+1 initial;95% CI across paired rounds, still a pair→ring model |
| MTP-off chat/code decode TPS |**31.9783 /31.9881** medians |Six requests/category; A0 only, not a completed paired bracket |
| MTP-off chat/code client-wall TPS |**24.2437 /26.7104** medians |Includes HTTP/prefill overhead |
| MTP-on TPS / acceptance |**Unavailable** |First N4 pass failed before a completed round or emitted MTP result |
| First cold N4 owner compute sum |**1,264.588 ms** |One cold pass; cannot replace the warm47.493 ms proxy with a steady result |
| Cold rank7 k3 draft chain |**87.328 ms** |One first-use chain; previous warmed native-only result remains8.989 ms |
| Single-session rank7 requested-live |**14,457,438,040 bytes** |Eager440 MiB worker cache; only38,076,584 bytes below13.5 GiB cap |

The direct N4 measurement replaces the old **unmeasured N1×4 transport
proxy**,18.242224 ms, with the actual-frame pair-transfer model above.
It does **not** measure all-hop ring transport separately. No completed
`MTP_STAGE` ledger exists for the failed round, so verify-wall residual,
steady repaired-round time, live yield and paired chat/code speedup remain
unmeasured. Per-owner input/compute/read/forward-or-head and vocabulary rows
are retained; a missing completed ledger is not synthesized.

Attempt1 failed on rank7 QSA backing-file allocation: **ENOSPC**, not a
numerical error. Retry1 bounded worker cache to440 MiB, removed only stopped
candidate QSA files, and deleted owned overlay/head source copies only after
successful loading/readiness. Its first N4 target pass reached all eight
owners; remote retained-prefix handlers reported success, then the coordinator
failed in the retain/rollback transaction. The visible restore status5
does not establish the original failure: queued replies can obscure it.
The matching original owner error has since been reproduced and repaired
offline above; retain the failed lease's evidence unchanged.
Do not describe either failed attempt as a completed serving comparison.

## Gate status and blockers

| Gate | Status / required next evidence |
|---|---|
| Target quantization floor |Unchanged: target≥Q4; draft-only exception above |
| Pass-local self-consistency |2,167 actual a038dec MTP choices all match the pass-local target argmax;775 stage ledgers align across all eight owners; complete qualification capture still pending |
| N1/N4 divergence |No bitwise alignment requirement. At identical prefixes, differing winners must exchange the same top2 with **both raw margins<0.10**; complete serving capture pending |
| Full-model N4 quality |Same-geometry original-Q880-vector control now retained by the precision sibling; N4 candidate comparison still pending: `KL(P_Q8||P_N4)≤0.005`, top1≥0.985, hence≥79/80 |
| Reference provenance |New Q8-control archive SHA `d4ca5b1351204674082c580e3623e90bb97fb518cd0c577efbf81b3b67fd48c3`; historical80 geometry discrepancy is not N4 attribution. Separate480-natural-context/p99 gate is not interchangeable |
| N1 target prerequisite |**Reported FAIL** on matched80 forced-frontier contexts: current-Q8→current-mix04 mean KL0.051191953/top1 85%; this is already an N1 target/profile gap, not evidence against N4 |
| State correctness |Cold coordinator failure fixed offline and confirmed on authentic a038dec ring:775 completed aligned rounds;24 GPU fixtures/8 TCP cases also pass |
| Residency |Offline complete two-session/deferred graph fit now passes with CJK320 MiB or English416 MiB caches; both440 MiB admissions reject. Production disk/coexistence/long-context and any future extra speculative banks remain unqualified |
| Performance |Paired A0/F3/A1 completed: code1.074×, chat0.736× median paired decode ratio; warm≈101.3 ms rounds, not60 ms |
| Temperature sampling |Standard rejection-sampling primitive exists; production path/gates here are greedy, not temperature qualification |
| Lease hygiene |PASS: latest a038dec A0/F3/A1 restored/verified/cleaned; independent all-eight verification21:15:32Z, no promotion |

### Matched-Q8 update: N1 target baseline fails

The precision sibling reports the final same-geometry control comparison:

| Reference → candidate, same current frozen N1 profile |Mean full-vocabulary KL |Top1 |Original-Q8 gate |
|---|---:|---:|---|
| Original Q8 → F32-hop mix04 |0.051191953 |85% |**FAIL** |
| Original Q8 → all-hop scaled-FP16 mix04 |0.056666349 |85% |**FAIL** |
| Original Q8 → mask224 scaled-FP16 mix04 |0.054000836 |85% |**FAIL** |

Source: precision branch `cuppaxanax/bf16-hops`,
`9531f7e4130c582cb52bef1f4e14c7f4cff6ad5a`,
`results/hillclimb/FP16_HOPS.md` and `fp16-hops.json`; its canonical report was
read here. No new quality run or artifact copy was performed here. The sibling reports
all1,223 original-Q8 trained spans authenticated/read back and80 logits
validated against the already shared corpus/prepared frames.
Historical-r3 Q8 versus current-profile Q8 also differs
(KL0.010931423/top1 92.5%), so historical geometry/profile/conditioning is a
real confound, **but does not explain away the matched cumulative failure**.
The control does not yet separate dense quantization from dtype-dependent
kernel arithmetic.
Do not subtract KLs or attribute this existing N1 gap to speculative verify.
The80 forced-first frontiers and the older480-natural-context Q4 gate remain
different cohorts; neither replaces the other.
The sibling's subsequent raw480 reconciliation reproduces mix04/Q8
KL0.003128578/top1 98.9583% and direct original-Q8/stack
KL0.003084267/top1 98.75%, both passing. The often-cited stack
KL0.000246435/top1 99.7917% is **incremental versus mix04**, not versus Q8.
Native-prefill/current-N1 cross-running and conditioning reconciliation are
still in progress; do not revoke the older cohort's pass or relabel an
incremental statistic as cumulative.

Promotion is blocked on diagnosing the N1 target gap as well as the N4 gates.
Prioritize offline target-format/kernel/profile attribution using matched
prefixes; do not resume bitwise arithmetic alignment or weaken thresholds.
The≈7 ms all-hop half-precision lever remains conditional, and mask224's
relative-to-F32 success is **not** an original-Q8 quality pass.
Any requested performance-only lease must explicitly acknowledge this known
baseline failure; no active lease or production access follows from this update.

## Ranked experiments

Savings below are **conditional planning budgets, not measured improvements**.
Correctness/retain reproduction is a prerequisite, not a throughput experiment.
The offline coordinator reproduction/fix above closes that local prerequisite,
not full-ring qualification. Preserve the first error as well as any restore
error and exercise all keep1..4 frontiers.
Warm every N1..4 graph/pipeline before timing, without publishing dummy tokens;
prove state restoration after warmup. No further production access is implied.

1. **Goodput/serial fallback: implemented offline; serving gate still pending.**
   The chooser now includes true k=0, compares complete-round expected yield
   against a measured target-only serial price with0.5% headroom, and removes
   forced periodic speculative probes. Acceptance is per-stream1/16 EWMA;
   initial priors are optimistic estimates, not calibrated confidence.
   Losing streams skip native ADVANCE RPC/GPU work, latching serial until a
   successful prefix synchronization; exact-prefix reuse cannot draft from
   stale KV. Cold/new conversations reset acceptance, while prefix-hit
   continuations retain their learned policy. Both private slots, serial
   fallback without native RPC, restart, cancellation and EOS have bounded
   eight-rank regression coverage. Fixed-width diagnostics remain overrides.
   First-use compilation is not a valid warm price; one extreme first-shape
   stall is excluded from price learning but remains charged in raw timings.
   **Chat≥1.0× and retaining the code win are not yet measured for this policy.**
   Initial probing can still incur regret; correctness tests are not a
   noninferiority result. Suffix proposals and calibrated probability stops
   remain future work; subset-normalized confidence is not acceptance.
2. **Dense verify weight reuse: target5–15 ms/round research range;
   combined owner-compute research budget≈25 ms.**
   The authentic code N4 owner compute sum is65.160 ms, not the old47.493 ms
   proxy; all-shape code mean is64.979 ms. Aim for40 ms using dense
   N-column weight reuse, expert-union/per-token gates, persistent replay and
   dispatch amortization. Preserve causal GDN recurrence and QSA masking.
   Reuse the unpacked weight fragment across columns, not just coalesced
   addresses; select rows/K splits by actual VGPR/spill/occupancy measurements.
   Keep target≥Q4. Require authentic warmed whole-block measurements and N4
   model-quality gates; the cold lease pass is not evidence this target is met.
   The new size-guarded cooked-Q8 reuse bit4096 avoids the small-HC
   regression caused by blanket bit1. On24-CU .51,2560×6144 Q8 reaches
   N4/N1≈1.44; representative Q4 remains≈1.66. Synthetic whole-block
   replay measurements still miss1.5× and do not demonstrate an end-to-end
   win. Forcing wave32 made representative Q4 worse and was rejected.
   This is a measured partial kernel-path improvement, not the requested
   whole-rank result or a credited5–15 ms serving gain.
3. **Restricted draft-only vocabulary: target1–7 ms/round;387–566 MB
   rank7 saving.**
   Keep a locally resident authenticated subset of target Q8 rows, mapping
   selected IDs back to full vocabulary. The audit's40,525-ID English/code
   and106,299-ID CJK-inclusive examples are≈110.5/289.8 MB instead of676.9 MB.
   Default-off subset loading/authentication is now implemented and measured
   above, without weakening full-head source authentication.
   **The full target vocabulary is unchanged.** Measure multilingual/tool/code
   coverage and acceptance; English-only subsets can damage CJK acceptance.
   Raw row savings alone did **not** solve full440 MiB dual admission.
   The measured complete-fit configurations above additionally reduce the
   worker cache to320 MiB (CJK) or416 MiB (English). Include every private
   bank, alignment, deferred peak and driver reservation; do not count a
   requested-byte headroom measurement as a long-context service gate.
   The actual Strata sets have now been located:40525 English/code IDs from
   `f2a08d4b2da99bb0638dd03dc1d097d132f0af68` and106299 CJK-inclusive IDs
   from its current data file (binary SHA
   `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`).
   These are not the earlier contiguous-ID smoke fixtures. The packer accepts
   little-endian binary IDs; benchmarks can authenticate caller-supplied
   containers.    Complete rank7 graph fit and retained40-case N1-teacher acceptance are
   now measured above. Actual N4 serving acceptance/performance and
   multilingual/tool coverage remain separate evidence requirements.
4. **Partition GDN recurrent state across cooperating threads: target2–8
   ms/round research range, included in the target-service bundle.**
   Replace each lane's128-float private column with four cooperating groups
   owning32 values each. The audited FG kernel's256 VGPR/130 spilled registers
   and33,792 scratch bytes/wave are a concrete target, not successful
   register retention. Requalify reassociated reductions. Saved-input
   state-only retention and remote-first fanout already exist; do not count
   their implementation again or regress to whole-transformer repair.
   This overlaps item2's≈25 ms bundle and is **not an extra2–8 ms credit**.
5. **Batch target vocabulary: target≈2 ms critical-path saving.**
   The current verify head performs four serial calls/slice. Reduce the
   measured≈3.10 ms maximum per-head-rank compute to≈1.1 ms by loading each
   Q8 weight tile once for N columns and performing batched argmax.
   Head ranks0/1/2/4 overlap: their≈12.20 ms compute **sum is not wall time**.
   Do not retain the old5.66 ms proxy saving as a measured critical-path gain.
   Draft-chain heads remain sequential because each conditions the next.
6. **Lower-bit draft head, with Q2 draft experts for fit: target5.35 ms saving.**
   Aim for observed coordinator chain span9.853→4.5 ms, subject to acceptance.
   The repeated full-vocabulary head reads dominate more than selected expert
   reads; expert-only Q2 is primarily a residency lever. Native packed storage
   and kernels are required—dequantizing whole weights back to Q8 would erase
   the residency benefit. See the format/acceptance requirements below.
   Restricted vocabulary is the lower-integration-risk alternative in item3;
   **do not add its1–7 ms to the same head's5.35 ms low-bit saving**.
7. **Actual N4 framing/amortization: target3.22 ms saving.**
   Improve the measured pair-transfer model14.216→11 ms with frame/header
   coalescing, sendmsg/batching and fewer host/CRC passes. The N4 payload
   serialization term at negotiated1 Gb/s is about10.83 ms/eight work hops;
  11 ms is an aggressive application budget, not achieved bandwidth.
   Reduced-precision transport receives no credit: the published sibling's
   full-hop scaled-FP16 result fails KL and must be separately qualified.
   **MTP + FP16-with-scale hops is now a separate conditional lever:**
   sol-roofline is qualifying that transport. If its model-quality result
   passes and the codec covers the actual N4 payload, budget roughly
   **14.2→7.1 ms/round (≈7 ms saved)** for wire-dominated work frames.
   Measure N4 framing, scale/codec CPU cost and initial-ngram scope explicitly;
   passing an N1 quality run alone is not an N4 end-to-end gate. The older
   failed precision policy does not decide this new candidate's result.
   This replaces, rather than adds to, the11 ms F32 transport milestone.
   Latest sibling evidence: all-hop scaled FP16 fails versus frozen F32
   (KL0.00987737/top1 98.75%); destination5–7-only mask224 passes that relative
   comparison (KL0.000678952/top1 100%). Neither result qualifies all-hop N4
   transport or cumulative N4/original-Q8 quality, so the≈7 ms credit remains
   conditional rather than an available saving.
   The published precision branch deliberately excludes MTP verification and
   rejects either low-precision codec with experimental MTP before readiness.
   Combining them requires an explicit batched-codec/native-owner validation
   change and fresh quality tests, not merely enabling its flags.
8. **Input/read/head-control/host amortization: target3.8 ms total saving.**
   Budget3.6 ms inside verify, distinct from owner compute, wire and the
   parallel-head compute saving, plus0.2 ms from measured post-verify
   retain/native-repair/commit/host≈1.46 ms. These are hypotheses requiring
   new disjoint timing, not inherited old RPC proxies. Current N4 verify
   residual after owner input/compute/read is22.63 ms for code and includes
   head, transfer and control; it is not independently measured wire latency.
   Checkpoint span≈9.83 ms includes waiting behind rank7's draft chain and is
   overlapped almost completely; do not count it again as9.83 ms saved.
9. **Rank7 future-window drafting during ranks0–6 verify: target2 ms
   average critical-path saving after low-bit drafting.**
   Rank7 is idle until its target block arrives. Extend the native draft
   branch assuming current drafts are fully accepted, in a separate
   speculative KV/replay bank. Generate a predicted bonus/pending token too:
   full acceptance alone does not reveal the target's bonus token.
   Reuse only when accepted prefix, actual pending/bonus and branch key match.
   Otherwise discard the speculative branch and rebase/repair from actual
   accepted target HCs. Target GDN/QSA checkpoint/retention remains authoritative.

   Give arriving target work priority; stop/yield between bounded draft steps,
   measure target queueing and wasted GPU work, and do not occupy rank7 past its
   target arrival. Preserve native-state/version ownership when actual HCs
   replace projected draft conditioning. Proposals may be approximate; emitted
   tokens may not bypass target verification. The maximum hidden interval is
   bounded by rank7's measured idle window and useful remaining draft work.
   Expected saving is further reduced by rejection/bonus-mismatch probability.
   **Do not add a9 ms overlap win to a4.49 ms low-bit win:** the planning credit
   is only2 ms of the remaining4.5 ms chain. Additional private banks must pass
   full eager dual-session residency; lower-bit shared weights can enable them.

Native repair's existing0.601 ms mean proxy and the historical158 ms
*whole repaired round* do not justify treating158 ms as removable repair work.
Keep state-only retention; prioritize repair further only when a complete
authentic stage ledger identifies a real remaining bubble.

## Draft-only low-bit implementation and memory lever

Read-only precedent: `Strata\docs\ORCA.md` and `Strata\tools\mtp_pack.py`.
That packer keeps dense MTP matrices BF16/norms F32 and supports **Q2_0 routed
experts**. Its custom GGUF type42 is **64 weights/18-byte block**, FP16 scale
plus2-bit values on `{-1,0,1,2}×scale` (2.25 effective bits/weight).
**This is not GGML Q2_K** and must not be relabeled as an existing FG type.
Strata's target IQ3 configuration is outside FG's target policy.

For2,516,582,400 native expert parameters, ideal packed sizes are:

| Draft expert format |Bytes |Delta from Q8 |
|---|---:|---:|
| Q8_0 |2,673,868,800 |— |
| Q4_0 |1,415,577,600 |−1,258,291,200 |
| Q2_0 |707,788,800 |**−1,966,080,000 bytes (1.831 GiB)** |

Even Q4→Q2 saves0.659 GiB of experts, larger than the historical
**0.24 GiB dual-session overrun**. Shared weights save once, not once/session.
The local full head adds another676,920,320 requested-live bytes and future
banks add private memory, so this is a promising byte budget, **not proof
that the new complete two-session configuration fits**.

For the635,699,200-weight vocabulary, ideal Q8→Q2 packed bytes fall from
675,430,400 to178,790,400, another0.463 GiB before cooker/scratch overhead.
This head change is separate from Strata's expert-format precedent and needs
its own kernel and acceptance experiment.

Implement a **separate authenticated draft artifact** with source/target-parent
SHA, tensor inventory, shape, quantizer and format identities. Current
`--mtp-local-head` authenticates full Q8 target-output bytes and must not be
silently weakened. Add draft-only packer/loader/native GPU kernels and explicit
policy checks; keep target loaders/head and target quality unchanged.
Measure expert/head Q8/Q4/Q2 combinations on identical captured prefixes with
correct skipped-frontier simulation, then actual N4 serving. Record acceptance,
tokens/verify, draft ms and eager peak memory. Reconstruction MSE alone cannot
select the format. Throughput improves only if
`new_yield/old_yield > new_round_ms/old_round_ms`.

## Conditional path to ≤60 ms/round

**Rebase on measured a038dec code rounds:101.280018 ms.** This supersedes the
95.789182 ms transport-adjusted offline proxy and its cross-cohort calibration
allowance. All-shape owner compute averages64.978676 ms; N4-only compute is
65.159553 ms. The tiny N2/N3 generation tails explain the population difference.
Below, owner compute, head critical-path compute, draft span, new future-window
overlap, wire model, remaining verify overhead and post-verify control are
separate proposed reductions. Head compute is parallel, not its12.20 ms sum.
The14.215694 ms transport anchor is the **prior** measured-frame pair→ring model,
not newly isolated wire timing. Reduction attribution needs new measurements;
the table is an aggressive research budget, not a demonstrated path.

| Conditional milestone |Round ms |Code TPS |Chat TPS |
|---|---:|---:|---:|
| Measured code round; yield/round TPS model |101.280 |33.44 |21.91 |
| Combined owner compute64.979→40 ms |76.301 |44.39 |29.09 |
| Parallel target-head critical compute−2 ms |74.301 |45.58 |29.87 |
| Restricted-vocab **or** low-bit draft span9.853→4.5 ms |68.948 |49.12 |32.19 |
| Useful future-window overlap2 ms average |66.948 |50.59 |33.15 |
| N4 work transport budget11 ms |63.733 |53.14 |34.83 |
| Other verify input/read/control−3.6 ms |60.133 |56.32 |36.91 |
| Post-verify retain/native/control−0.2 ms |**59.933** |**56.51** |**37.03** |
| Optional qualified FP16+scale transport≈7.1 ms instead of11 ms |≈56.033 |≈60.44 |≈39.61 |

TPS=`1000×tokens_per_verify/round_ms`, using **observed**
code3.386667/chat2.219512 throughout, assuming each prospective round budget
for both categories. This yield/mean-round model differs from the actual
per-request median TPS, code33.409/chat22.378. Draft restriction/quantization,
lookahead or a serial-fallback policy can change yield. This is not an additive
measured speedup or confidence interval; chat should first recover its serial
baseline rather than be forced onto fixed k3.
The newly measured subset counterfactual loses4.184% code yield relative to
its full-head control. Therefore the unchanged-yield restricted-vocab rows
are **not supported for these curated subsets**. Applying that cross-cohort
ratio merely as a sensitivity model gives≈54.14 code TPS at59.933 ms and
≈57.91 at56.033 ms, not60+. Better subset coverage/acceptance or a different
draft format is still needed; no such correction is credited as measured.

**60 ms rounds do not imply60 TPS.** At code yield3.386667,60 TPS requires
≤56.444 ms/round; at60 ms it requires yield≥3.6. After the≤60 ms budget is
actually met and qualified, the remaining path is another≈3.5 ms reduction or
better draft acceptance (e.g. head quality/calibration/adaptive windows), with
the same target gates. Chat needs a larger acceptance improvement.
If qualified half-precision hops reach≈7.1 ms, the conditional56.033 ms budget
would narrowly exceed60 code TPS at unchanged yield. Neither those wire
savings nor that MTP/transport combination is implemented/qualified here.

The user directly granted the exclusive≤45-minute **diagnostic-only A0/F3/A1** window,
single session, mask330, fixed k3,440 MiB worker cache, mandatory eager
admission, watchdogs and exact current sessions2/wide1 restoration. The
offline retain fix was confirmed in the completed authentic run, including
paired chat/code, SSE/stop/cancel and warm-stage logs, using the authenticated
**full Q8 head** to isolate that fix. All eight owners were independently
restored/verified/cleaned and released21:15:32Z. No lease remains active.
The controller now validates either raw Q8 or typed subsets against the
complete pinned raw source and cooked-parent declaration, then verifies the
actual container SHA on staging. It still rejects dual-session local heads
at its440 MiB configuration. It must bind a complete graph-warmed receipt
and the explicitly smaller cache before a two-session subset lease.
Its old non-head700 MiB argument was invalid and silently resolved to640;
it now declares640 directly, preserving actual behavior/provenance.
Future requests must go directly to the user/delegated authority, not sol-abqual;
do not treat prepared artifacts as an active lease.

Next address owner compute and real serial fallback using this completed ledger.
Complete same-geometry Q8/N4 quality and emission capture before any
qualification/promotion; an explicitly unqualified diagnostic may precede
that separate gate. Test full two-session eager/deferred/cancel/restore peaks
separately. No serving ref, launcher or default is promoted by this plan.

The requested powered promotion experiment is **six balanced arm pairs**
(three AB, three BA), not six requests inside one arm: A=MTP off,
B=adaptive k0..3 with the same binary/target/profile. Bind the curated head,
actual cache and full-rank fit receipt; warm both slots and N1..4 graphs
without changing published history. Record paired chat/code decode and
client-wall TPS, acceptance, complete stages, coexistence and cancellation.
The arm pair is the statistical unit. Require chat noninferiority at1.0×
and a retained code win, in addition to every model-quality/state/memory gate.
No promotion lease is requested while whole-rank reuse, numerical
qualification or powered-controller preparation is missing. A subsequently
authorized six-pair lease can establish the new policy's on/off performance;
it does not waive those prerequisites.
