# Cold coordinator retain repair and restricted draft vocabulary

**Offline only; default off; no promotion or new production access.**
Branch `cuppaxanax/mtp-serving`, source parent `4ce396b5880a789961ca71b46412532004b330af`.
Only .51 was used; its Vulkan query reported **24 CUs**, not the earlier
40-CU geometry. It was cleaned/released at **2026-10-16T20:57:43Z**.

## Retain failure: negative reproduction and fix

Replicated coordinator executors intentionally report ownership of all48
layers. A ring pass advances only its actual block. Retain incorrectly
checked every QSA layer's frontier, including foreign layers left at0.
Workers use rank-local ownership, so previous worker-only fixtures missed it.
Validate only `journal.first_layer..journal.last_layer`, matching the state
that the journal actually reconstructs. Do not alter global ownership
semantics or advance/truncate foreign QSA layers.

The committed old `owner.c`, rebuilt against the strengthened fixture, fails
with **`MTP retained QSA frontier mismatch`** after N1-only prefix warmup and
the first cold N4 at frontier22. The fixed binary passes **24 GPU cases**:
frontiers22/127, both private sessions, N4-first then N2/N3, every proper
retained prefix, full retention, continuation and rollback. Every case checks:

- saved-input GDN reconstruction and full journal state exactness;
- continuation within the existing component tolerance;
- exact rollback, untouched HC output and untouched other session;
- stale-retain rejection and preserved static replay.

Maximum observed absolute state/continuation differences versus the serial
fixture were0.000205993652 /0.0000103116035; all component-tolerance checks
passed. Full journal reconstruction and rollback were exact. This is not a
bitwise N1/N4 arithmetic-alignment gate.

The fix removes the reproduced originating owner condition. It does not
implement ACK draining or general network-error recovery: secondary queued
replies can still obscure an earlier failed transaction. Authentic full-ring
confirmation remains the next diagnostic, not an offline-test claim.

Negative evidence is retained. The first run needed unlimited **process
memlock**, not a hardware/governor change, for fixed-buffer registration.
An initial strengthened fixture carried frontier22 into its later127 warmup
without resetting; that harness error was corrected with an explicit session
reset. Old/fixed logs, the failing harness attempt and final source are
included, rather than discarding these attempts.

## Restricted vocabulary: implemented, source-authenticated

The existing default-off `--mtp-local-head` accepts a raw full Q8 head or
`FGDVOC1`. The new container holds sorted global IDs, the sealed cooked
`output.weight` parent SHA, and the **complete original raw Q8 source**.
Runtime streams/cooks/hashes that entire source and checks each selected GPU
tile against the authenticated stream. A claimed parent hash plus unauthenticated
rows would not suffice. Corruption outside the selected set also fails.
Only selected/padded rows and maps become GPU resident; the container is
**not smaller on disk**. Full target weights, full target argmax and token
IDs are unchanged.

`tools/mtp_vocab_pack.py` creates containers with explicit ID lists and
refuses overwrite. `make test-mtp-vocab` checks mapped GPU argmax/replay,
wrong parent, duplicate IDs, unselected-source corruption and truncation.
The3-row GPU fixture uses43,548 requested-live bytes and returns global ID17,
not subset ordinal1.

### Authentic head, replay-warmed timings

Source raw SHA:
`2801d79db74bcae38b47fd5051f03136d8ca7715abeba7d92b49ba46c696bc33`.
Sealed cooked parent:
`fa90643ca673bf032655f6c645c2a5718bc1eac3b1b29bd3ff90eb13d14e19af`.
Same process/.51/reported24 CUs;8 synthetic hidden vectors,8 warmups and100
timings per shape. Scores/IDs match the independent existing raw-row loader
exactly for all8 vectors/shape.

| Rows | Mean/head ms | Median | p95 | Requested-live bytes | Saving bytes | Three-call saving proxy ms |
|---|---:|---:|---:|---:|---:|---:|
|248,320 |2.238927 |2.241487 |2.252524 |676,920,320 |0 |0 |
|40,525 |0.430886 |0.438086 |0.442210 |110,479,316 |566,441,004 |5.424124 |
|106,299 |1.018502 |1.019804 |1.027259 |289,784,684 |387,135,636 |3.661277 |

These subsets are **contiguous low-ID smoke/performance examples**, not the
audit's curated English/code/CJK lists. No acceptance, semantic coverage,
whole-chain speedup, dual-session fit or serving-round gain is claimed.
Both byte savings exceed0.24 GiB under a same-baseline comparison, but full
private-state/deferred/driver peaks still need admission measurement.
Do not subtract these24-CU head deltas from the older40-CU8.989 ms chain.
Restricted vocabulary was selected before another dense-kernel tuning pass
for expected ms/effort; no target arithmetic kernel changed.

## Gates and next lease request

**29 targeted Python tests**, strict C17/`-Werror` build, environment-flag
registry, GPU vocabulary test,24 coordinator state cases, native/cycle/wire/
embedding tests and8 TCP ring cases pass. All existing target numerical gates
remain: pass-local argmax; identical-prefix top2 swaps with both margins<0.10;
same-geometry original-Q8 full-vocabulary KL≤0.005/top1≥0.985. The last
full-model gate and complete actual-serving receipts remain pending.
The sibling has retained the authentic same-geometry80-vector Q8 control;
no new GPU work or copy was requested from it.

A fixed **full-head-first A0/F3/A1** candidate is sealed and the existing
controller's no-production-access plan has been exercised. Request a fresh
exclusive≤45-minute diagnostic lease,30-minute work budget, single session,
mask330/k3/440 MiB cache, eager admission and watchdogs; restore exact current
sessions2/wide1, verify, clean and release. The controller intentionally
continues pinning the raw full-head SHA. Subset serving needs curated IDs and
explicit typed-artifact staging validation before extending that controller.
No lease is active and none was consumed for this follow-up.

The existing transport-updated model remains95.789 ms/round, not a new live
measurement. The roadmap now includes **conditional** FP16-with-scale
14.2→≈7.1 ms transport,≈7 ms saved, dependent on sol-roofline's quality result
and actual N4 codec/framing measurement. The older all-hop precision policy
fails KL; destination5–7-only relative-quality success is not an all-hop N4
qualification. Avoid double counting the existing14.2→11 ms F32 milestone.
See [the updated roadmap](../../docs/MTP_TO_PROD.md) for the conditional
59.956 ms path and optional≈56.1 ms qualified-half-hop extension; neither is
achieved performance or60 TPS.

## Retention and hygiene

- Final binary SHA `6c23c7b5e0b0510d9154cc8c61639efa1f7449e7461d5be41b30d0cc60c57868`.
- Diagnostic release SHA `309f906513dfb19098e1382317771e7fe0df0ba457262da71786e25fc85e3f3f`.
- Final offline archive SHA `2491672dc08eea6e165991952d6a0a0d85d611949a8dc1537a8b35bb2617f091`.
- Head/before-after archive SHA `51c7476618c0fe37f3866243785403299f0619918dda9c10e359d6ef56e13f64`.
- Initial passing archive SHA `e3dec5155f3dd2a9a813c0049cff5b268d0a09cf49094a2fe85123b19dd4abc3`.

All archives were downloaded and independently SHA-checked before cleanup.
The owned `/home/xander/hc-mtp` root is absent twice, no owned executable
remains and every reserved test listener is closed. CPU-governor nodes were
unexposed both before and after; the cleanup guard initially failed closed on
the empty checksum list, then explicitly verified continued node absence.
No hardware/governor/service/production-launcher settings were changed.
.52/.53/.50 and production were untouched. The authenticated local full-head
copy is intentionally retained for the pending lease; original archives and
retained trained packs are unchanged.

Machine ledger: `mtp-retain-subset.json`. Weight-free reproducibility bundle:
`mtp-retain-subset-evidence.tar.gz`; publication hashes:
`mtp-retain-subset-publication.json`.
