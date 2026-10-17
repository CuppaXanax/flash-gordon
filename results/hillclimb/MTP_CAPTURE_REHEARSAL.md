# Renewed MTP capture: insufficient rehearsal, timeout and scale-hoist A/B

Default off; **no promotion, real union measurement or new TPS/quality claim**.
Base `69ca796144ca33fa126735cddcc521016d94aa0b`.
Data: [mtp-capture-rehearsal.json](mtp-capture-rehearsal.json).

## Rehearsal scope and failure

Before accessing production, `.51` passed:

- The repaired binary's API/rank parsers with all candidate option values.
- A real eight-rank runtime/fabric/native-chain transaction fixture with
  deterministic compute/expert doubles:960 appended route records spanning
  all48 layers, including N4.
- The unchanged four-case `capture-client.py` against an HTTP protocol double,
  validating request bodies, response accounting and result-file production.

**This was not a full trained-model end-to-end rehearsal of the exact API/
worker/cold-prefill path. The component rehearsals were insufficient and
should not have been treated as the requested readiness gate.**
The new persistent `make test-mtp-capture-ring` covers complete per-window
all48-layer multiplicities, N4 shapes and valid top10 IDs; its synthetic IDs
are explicitly **not acceptance or expert-union physics evidence**.

The renewed user grant was2026-10-17T01:37:00.395Z, maximum15minutes.
The private controller used a12-minute absolute work deadline
(01:49:00Z), preserving restoration time. Source staging and offline rehearsal
left only a short first-request window. Candidate source/binary and stock
shaders were unchanged from the repaired69ca796 release; fixed-k3, full Q8
draft head, sessions1,440MiB cache and mask330.

The API/worker capture-parser preflights passed before stopping production.
The first `chat-0` request then exceeded the remaining work deadline.
The coordinator logged chunk0 and its6..11 prefill block; no verification
route records or completed HTTP cases were retained on **any** owner.
Shutdown reported `distributed transport is not reusable; reopen the runtime`.
That is not a causal attribution: there are no measured busy counters,
completed prefill ledger or first-error evidence proving a particular
kernel, rank or transport defect. In particular, a time-budget expiration
must not be relabeled as a diagnosed kernel deadlock.

All eight original services/binary/shaders/manifest and exact pinned FG
environment were restored and independently verified, with matching API
greedy sanity. Scratch was absent twice and governors unchanged at
**01:50:21Z**,13m21s after grant; there was no later production access.
Restored source31508ee, sessions2/wide1, ordinary native prefill plus cached N1.

**Real chat/code union ratios and the measured-union op-weighted floor remain
unavailable.** Earlier conditional3x-union planning values1.593x/1.735x remain
hypothetical; no serving ceiling or60TPS projection is newly justified.
The private launcher now reports `ready=false` and blocks `Run`.
Do not request another lease until genuine trained cold-prefill/capture
end-to-end rehearsal is complete; more parser/protocol doubles are not enough.

## Is dequantization repeated per token?

Dense `fg_small_reuse_rows.glsl` calls `reuse_weight` once per row/weight
fragment, then consumes the same qlo/qhi in all token-column accumulations.
`reuse_weight` already performs bit extraction, scale decode and FP32 scaling
before those accumulations.

Compiled dense N2 and N4 both contain64 `v_cvt_f32_i32_sdwa`,8
`v_cvt_f32_ubyte0_e32` and8 `v_cvt_f32_ubyte1_e32` instruction sites.
HC IQ4 similarly has32 bit-extract and8+8 half-to-float conversion sites at
both N2/N4, restricted to the weight loop. These are **static instruction
counts**, not executed counters or memory transactions.
The original N2/N4 compiler captures are retained in
[MTP_CAPTURE_PRESSURE.md](MTP_CAPTURE_PRESSURE.md)'s evidence archive;
the new hoisted-shader dump is in this report's archive.

HC decodes quants/LUT values and `d` outside the token loop. Its per-token
`fma(d, dot(q0,a)+dot(q1,b), acc)` scales the dot result, not every weight
element again. Moving all eight q0/q1 element scales outside the loop can
increase arithmetic relative to four per-column dot-result scale FMAs.

## Timestamp A/B: explicitly hoisted HC scale

No `rocprof`, `rocprofv2`, `rocprofv3` or `rgp` executable was found on
`.51`'s PATH. The requested timestamp alternative was used; **no VALU/memory busy
percentage or achieved hardware occupancy was measured**.

Private variant:

```
q0 *= d; q1 *= d;
for (t in columns)
    acc[t][r] += dot(q0,a[t]) + dot(q1,b[t]);
```

Target formats stay Q4-or-higher; no draft-head/vocabulary change.
This A/B times mask393546 synthetic six-layer target blocks, not the draft
head or transport.264 warmup/216 timed token budgets, process medians,
`B/mean(A0,A1)` per bracket. Compiler debug/cache-disable were used only for
ISA dumps, not wall measurements.

| Bracket | N1 B/A | N4 B/A |
|---:|---:|---:|
|1 |0.999219 |1.001426 |
|2 |0.999308 |1.000155 |
|3 |1.003267 |1.054137 |
|Median |0.999308 |**1.001426** |

The third bracket is materially slower; do not average it away or claim a
robust speedup. All12 independent HC component cases pass the unchanged
tolerance. Stock/hoisted N4 HC compiler statistics:

| Statistic | Stock | Hoisted |
|---|---:|---:|
|VALU instructions |1034 |1098 |
|VMEM instructions |46 |46 |
|VGPRs |84 |84 |
|Scratch |0 |0 |

The variant is **rejected**; stock source/SPVs were restored before cleanup.
This rules out that specific hoist as a useful optimization, not all ALU
bottlenecks. Per-column FMAs/reductions/activation work remain; the prior
2.167x synthetic whole-block result and dense-stage gap are not improved.

## Validation and release

C17/Werror fixture/benchmark build, eight Linux capture tests, the persistent
ring-capture regression and flag registry pass. The new fixture capture
argument is opt-in; no runtime environment flag was introduced.
`.51` governors matched reservation, root-visible owned processes were zero,
and owned scratch was absent twice at **01:50:29Z**.
No `.50/.52/.53` access or hardware changes.

The [raw evidence archive](mtp-capture-rehearsal-evidence.tar.gz), hash and size
are recorded in the JSON. It includes both
the executed live controller snapshot and the subsequently blocked launcher,
all eight restoration receipts/logs, rehearsals, component/ISA/timestamp
evidence and exact source patch. It excludes model/head/embedding downloads.
