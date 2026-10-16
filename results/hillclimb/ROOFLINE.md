# Small-N follow-up (2026-10-16)

The newer default-off expert-union, long-context QSA, dense K-split,
wave-size and lossless vector-layout research is documented in
[MTP_BANDWIDTH.md](MTP_BANDWIDTH.md), with machine-readable paired receipts
in `mtp-bandwidth.json`. Its current targets are N2/N1 <= 1.2 and N4/N1 <= 1.6.
Those experiments use unchanged **stock dynamic governors**, not the historical
pinned operating point below; synthetic replay is not real-pack serving
qualification. Native dual-session residency and model-quality gates remain
separate admission requirements.

Five-round final A/B: Q4 N2 vector+wave32 saves **5.25 +/- .19%** on .50
and **4.44 +/- 3.99%** on .53; the mixed long-N4 vector/Q8/hybrid-QSA
combination saves **15.53 +/- .36% / 8.52 +/- .37%**. Both ratio targets
remain missed; measured large multi-column GEMVs remain below 85% of the
historical roof. Default-off research only; full tables, all tested masks,
per-shape BW, source snapshots and raw receipts are linked above.
.50 live 40 CU is verified; .53 is stock/driver-reported24, **not an independently
verified 24-CU restriction**. No hardware change was made. Both workspaces are
removed, governors unchanged, and original .50 services restored/healthy.

# Remaining-excess follow-up: experts, HC and GDN

2026-10-15. Experimental selection: `FG_ROOFLINE=2047`,
`FG_ROOF_DENSE=19`, `FG_ROOF_QSA=3`, `FG_ROOF_EXPERT=33024`,
`FG_ROOF_HC=96`, `FG_ROOF_RECURRENT=3`, `FG_ROOF_GDN=1`.
**Defaults are unchanged; real-pack KL/top-1 qualification remains pending.**
All measurements use .50 only, at the same stock-safe pinned **2000 MHz GFX**
operating point. This is not a claim that sysfs MCLK was set to 2000 MHz.
Live CU routing, not Vulkan's unchanged reported count of 24, defines 40/24 CU.

## Definitive paired result

Each row is a fresh **five-round A/B/A** study: seeds 1-5, 256 warmup steps,
five 64-step windows per phase, alternating baseline/candidate/baseline.
Confidence intervals bootstrap whole rounds, not correlated individual tokens.
Incremental baseline is the previous `251/19/3/2` bandwidth selection;
cumulative baseline is mask zero. Candidate settings above are identical at
both CU counts. Intrusive profiles are separate from wall-time qualification.

| Comparison | CUs | Six-layer wall before -> after, ms | Saving | 95% saving CI, us |
|---|---:|---:|---:|---:|
| Incremental vs mask 251 | 40 | 2.406567 -> **2.277085** | **5.380%** | 125.268-133.097 |
| Incremental vs mask 251 | 24 | 2.676635 -> **2.505894** | **6.379%** | 168.649-173.227 |
| Cumulative vs mask 0 | 40 | 2.685077 -> **2.277203** | **15.190%** | 406.302-409.446 |
| Cumulative vs mask 0 | 24 | 2.939196 -> **2.503129** | **14.836%** | 433.675-438.600 |

No-clock controls preserve the incremental improvement: **2384.093 ->
2259.648 us** at 40 CUs, **2655.493 -> 2480.948 us** at 24. Separate
2048-warmup intrusive controls also improve, but their instrumented wall
times are not substituted for the table above.

### Production-shape bandwidth: complete operations

Five intrusive profile pairs. Cells are **before -> after us
(before -> after % of 450 GB/s)**, incremental comparison.
Weight bytes exclude activation/intermediate traffic consistently on both
sides. Recurrence uses the ledger's **6,382,848 logical state/read-write and
small-tensor bytes**, not a hardware DRAM read counter. HC rows include the
adjacent operations before fusion; they must not be compared with an isolated
old GEMV. The same 450 GB/s normalization is used at both CU settings, not
a separately measured 24-CU bandwidth ceiling.

| Operation and production K x M | Weight/state bytes | 40 CUs | 24 CUs |
|---|---:|---:|---:|
| Expert gate/up/SiLU, 2560 x 640 x ten, Q4_K | 18,432,000 | 56.803 -> **54.829** (72.1 -> **74.7%**) | 72.721 -> **71.756** (56.3 -> **57.1%**) |
| Expert down/reduce, 640 x 2560 x ten, Q5_1 | 12,288,000 | 38.123 -> **37.653** (71.6 -> **72.5%**) | 46.049 -> **41.631** (59.3 -> **65.6%**) |
| HC attention up IQ4 + mix, 320 x 10240 | 1,843,200 | 12.844 -> **11.428** (31.9 -> **35.8%**) | 15.700 -> **14.049** (26.1 -> **29.2%**) |
| HC FFN up IQ4 + mix, 320 x 10240 | 1,843,200 | 12.621 -> **11.133** (32.5 -> **36.8%**) | 15.313 -> **13.831** (26.7 -> **29.6%**) |
| HC attention down Q4 + inject + SiLU, 10240 x 320 | 2,058,240 | 15.771 -> **13.037** (29.0 -> **35.1%**) | 16.347 -> **12.553** (28.0 -> **36.4%**) |
| HC FFN down Q4 + inject + SiLU, 10240 x 320 | 2,058,240 | 15.412 -> **12.929** (29.7 -> **35.4%**) | 15.876 -> **12.472** (28.8 -> **36.7%**) |
| GDN gate Q4 + both F32 controls, 2560 x (6144+48+48) | 10,076,160 | 38.880 -> **35.902** (57.6 -> **62.4%**) | 42.432 -> **37.690** (52.8 -> **59.4%**) |
| GDN cached recurrence, 48 heads x 128 x 128 | 6,382,848 | 28.896 -> **20.582** (49.1 -> **68.9%**) | 29.066 -> **19.360** (48.8 -> **73.3%**) |
| GDN QKV Q4, 2560 x 10240, unchanged geometry | 15,155,200 | 50.070 -> 50.002 (67.3 -> 67.4%) | 57.752 -> 57.670 (58.3 -> 58.4%) |
| GDN output Q4, 6144 x 2560, unchanged | 9,093,120 | 25.890 -> 26.212 (78.0 -> 77.1%) | 29.246 -> 29.114 (69.1 -> 69.4%) |
| QSA query Q4, 2560 x 12288, unchanged | 18,186,240 | 59.352 -> 59.308 (68.1 -> 68.1%) | 68.652 -> 68.788 (58.9 -> 58.8%) |
| QSA output Q4, 6144 x 2560, unchanged | 9,093,120 | 26.496 -> 26.512 (76.3 -> 76.2%) | 29.500 -> 29.460 (68.5 -> 68.6%) |

**The >=85% large-shape target is still not attained.** Expert improvements
are modest, especially gate/up at 24 CUs. The largest reliable new gains are
recurrence and removal of HC/control passes. Unchanged rows illustrate profile
noise; their small differences are not credited as independent optimizations.
Final recurrence profiles are slower than the exploratory 18-us observations;
the table deliberately uses the fresh five-pair result.

### Low-K shape check, not a substitute for replay

Five paired 64-iteration component profiles at each CU setting, H2560,
four HC groups. Values are complete up+mix before -> fused after, us.
Expanded-Q8 rows use identical dequantized IQ4 values but more bytes.

| K / format | Weight bytes | 40 CUs | 24 CUs |
|---|---:|---:|---:|
| 320 / IQ4 | 1,843,200 | 13.401 -> 14.056 | 16.081 -> 15.030 |
| 640 / IQ4 | 3,686,400 | 21.458 -> 20.863 | 22.522 -> 21.852 |
| 320 / expanded Q8 | 3,481,600 | 16.327 -> 17.950 | 17.980 -> 17.900 |
| 640 / expanded Q8 | 6,963,200 | 26.797 -> 27.922 | 27.960 -> 28.173 |

The isolated 40-CU K320 IQ4 case **regresses**, despite the gain in the
production replay's two HC scopes. This is not a universal fusion win:
working-set/order matters, and real-pack shader coverage and quality remain
mandatory. K640 is shape-only evidence, not a second whole-model replay.
Lossless Q8 expansion is not selected at either K.

### Conditional eight-rank arithmetic

Using the fresh cumulative layer deltas, 36 GDN + 12 QSA layers save
**3.417316 ms/token at restricted 24 CUs**, against the historical 37.437-ms
anchor: **34.019684 ms/token, 29.395 TPS** (round-only 95% interval
34.008791-34.032713 ms). Incremental arithmetic saves another **1.393667 ms**
against the earlier 35.407-ms projection, yielding **34.013333 ms**.
These agree closely but are **not measured eight-rank throughput**: fabric,
context, quality, flag coverage and operating-point assumptions remain.
The 40-CU cumulative arithmetic is 34.299502 ms/token, not a production-CU
forecast. Confidence intervals do not include those systematic uncertainties.

## What changed

* **Expert gate/up was already fused with SiLU, and both expert stages already
  process all ten selections in one dispatch each.** No credit is taken for
  inventing that fusion. The selected gate/up variant caches the K2560 Q8_K
  activation once per WG and prefetches four weight blocks. Down uses two
  selected-slot cohorts in WG64 with true wave32, retains sixteen output rows
  per WG, and merges cohort results in LDS: no extra dispatch or global
  partial buffer. Selector `33024 = 256 + 32768`.
* **HC up + mix** computes the four groups of logits and their sigmoid-weighted
  residual mix together, also finalizing injection partials. It avoids the
  intermediate logit write/read and a dispatch. IQ4 weights remain IQ4.
* **HC down + injection + scaled SiLU** shares the normalized input in one
  WG256 dispatch. `FG_ROOF_HC=96` selects the unsplit path and WG256.
  The activation is `SiLU(0.25 * low)`, not `0.25 * SiLU(low)`.
  The owner decode path uses both HC fusions; prefill/unsupported formats
  retain the existing paths.
* **Cached recurrence** assigns all 128 output columns and four K partitions
  to WG512/head. Each lane retains 32 decayed state values, computes the
  memory/prior terms, and updates state without rereading it. Final RMS/gating
  is in the same dispatch. This is compared against the existing **~29 us
  algebraic** kernel, never the ~139 us non-algebraic oracle implementation.
* **GDN gate + alpha/beta** appends 96 one-row F32-control workgroups to the
  true-wave32 compact-Q4 gate GEMV. Three projections become one dispatch.
  One control row/WG avoids the substantial register-pressure hypothesis
  observed with eight rows; register allocation itself was not instrumented.

## Bounded hypothesis sweeps and rejected alternatives

Exploratory numbers below are three seeded intrusive replay profiles, not
confidence-qualified wall-time wins. Times include every merge/normalization
kernel where needed. All rejected variants remain opt-in and documented.

| Hypothesis | Measured outcome, us | Decision |
|---|---|---|
| Serialize all ten selected experts per row/WG | Gate/up **128.882**, versus 57.393 original | Occupancy loss; reject |
| Serialize two/five experts, shared activation | 40-CU gate/up 54.569 / 75.502; two-slot path **76.967 at 24 CUs** versus 72.707 | Two-slot wins at 40 only; not selected |
| Cache one expert's activation, prefetch4, native64 | Gate/up **54.802 / 71.511**, 40/24 CUs | Balanced selected gate/up geometry |
| Expert down full row, one/two lanes | 40-CU **99.384 / 71.080**, versus 38.240 | Reject |
| Expert down WG64 with twice the output rows | **53.164**, versus 38.240 | Smaller grid loses; reject |
| Parallel expert slots, global partials + merge | **36.593 + 3.676 = 40.269**, versus ~38.2 | Partial-only timing would be misleading; reject |
| Two/four internal slot cohorts, native64 | 40 CUs **40.742 / 40.342**; 24 CUs **47.116 / 52.040** | Reject |
| Two/four internal cohorts, true wave32 | 40 CUs **37.498 / 35.980**; 24 CUs **41.522 / 43.611** | Select two for both settings; four remains an explicit 40-CU alternative |
| IQ4 two/one lanes per group | 40-CU HC up **12.111 / 13.480**, versus 10.536 | Reject |
| Losslessly expand IQ4 to Q8, native / row-lane / fused mix | 40 CUs **13.278 / 32.576 / 14.984**; original IQ4 native/fused **10.536 / 11.389** | Expansion is exact in dequantized values, but slower; reject |
| HC down split4/WG64 + merge | **12.640 + 1.787 = 14.427**, versus 15.760 combined old path | Small win, superseded by single dispatch |
| HC unsplit WG64 / true-wave32-WG32 | 40 CUs **25.909 / 42.513** | Too few waves in flight; reject |
| HC unsplit WG256 / WG512 | 40 CUs **13.069 / 13.369**; 24 CUs **12.478 / 14.053** | Select WG256 |
| Q4 four-lane row cohorts, uvec4 x2/x4, wave64/32 | 40-CU QKV **79.430-85.237**, versus adaptive ~50.2; gate **48.567-49.837**, versus ~31.1 | Reassociation oracles pass, performance loses; reject |
| Cached four-column-split recurrence + RMS pass | 40 CUs **21.167 + 2.253 = 23.420**; 24 CUs **30.867 + 1.980 = 32.847** | Loses at 24; superseded by WG512/head |
| Vec4-state recurrence + RMS pass | 40 CUs **22.273 + 2.290 = 24.563** | Fewer load instructions alone do not ensure a win |
| GDN fused controls, 8/4/2/1 rows per WG | 40 CUs **71.997 / 49.453 / 40.683 / 36.000**; old gate + both controls ~38.75 | Select one row; do not enable the eight-row variant |

The exact IQ4 expansion preserves every original half scale and maps the
nonlinear four-bit integer grid directly to signed int8: **no requantization**.
Raw GGML and saved eight-row SoA/compact inputs have byte-identical dequantized
CPU tests, including padded rows and signed/zero scales. The persistent runtime
copy is cap-accounted and the source pack is unchanged. At K320/M10240 it grows
weight storage **1,843,200 -> 3,481,600 bytes**. Higher reported GB/s from this
larger numerator is not credited as a speedup. The selected profile does
**not** expand IQ4.

## Gates, accounting and scope

New HC up/mix CPU checks cover K320/640, H2560, IQ4 and expanded Q8.
HC down/injection covers eleven geometries and 1/7/24/40 injection pieces.
GDN gate/controls covers all 6144 gate outputs plus 48 alpha/beta values,
three input seeds including zero, and every control-row geometry.
Recurrence checks **all output and state elements over eight sequential
updates**, with component tolerance `1e-4*max(1,abs(reference))`; observed
cached-recurrence output/state maxima are **3.25e-6 / 7.45e-8**.
The new M640 x ten-selected-expert gate/up oracle uses the same approximately 1e-4
reassociation bound; the existing down oracle is unchanged. A fixture bug
was fixed by clearing regenerated raw Q4 rows before their bitwise-OR packer,
not by relaxing a failed shader's tolerance.

Full-block candidate dumps are finite/dimension diagnostics, **not** a
substitute for model quality or an arbitrary 3e-6 veto. Baseline A/A2 must
remain byte-identical. Every selected new individual path and the combination
is listed in `roofline-quality-candidates.json` for the unchanged real-pack
gate: **mean full-vocabulary KL(current || candidate) <=0.005, top-1 >=98.5%**,
same frozen Q4-qualification corpus and pack hashes. That gate has not been
run; no production promotion or rollout is claimed. Graph-only expert paths
still retain baseline kernels; fleet qualification must verify actual shader
coverage rather than silently measuring a flag that did not execute.

Integration note: sibling `cuppaxanax/mtp` commit `a6344e6` independently fixes
existing routed Q8 activation addressing. It is not merged into this Q4_K/Q5_1
experiment. Integration must carry both fused Q8 `act_word = act_base + 2 +
lane*8` corrections into the new shared `fg_moe_decode_gate_up.glsl`, as well
as the generic `fg_moe_kquant_cooked.comp` correction; patching only the `.comp`
wrapper would miss them. Re-run its independent two-block
`expert_decode_fused_q8_gates` CPU oracle and native-head oracle after integration.

The analyzer now compares HC up+mix, HC down+injection+SiLU, and GDN gate+both
controls as complete groups, preserving all unrelated dispatch multiplicity.
Expert split+merge and recurrence+normalization always sum both passes.
Grouped CSV byte counts are **baseline-equivalent logical traffic**, including
intermediates removed by fusion, not hardware DRAM counters. The bandwidth
table uses the stated weight/state numerator consistently on both sides;
a fused kernel's time cannot be split back into an invented isolated GEMV.
Historical 7996bbd statistics and every ranked dispatch reproduce unchanged
in the ungrouped analyzer.

## Follow-up evidence and restoration

* `roofline-excess-incremental-summary.json` and `-dispatches.csv` retain the
  fresh comparison against mask 251; corresponding `-cumulative-*` files retain
  comparison against mask zero. Each profile envelope lists executed shaders.
  All five profiles at both CU counts execute the selected cached expert,
  two-cohort down, both HC fusions, single-pass recurrence and one-row controls.
* `roofline-excess-captures.tar.gz` contains raw wall/profile JSONL, all-element
  diagnostics, component logs, exploratory sweeps and original/restored
  operating-point evidence. SHA256:
  `9dcf6069d9de415a854026df9364236555afc60e2124db204e31f3489d1ec519`.
  Replay binary SHA256:
  `76fb6d06c9a379f9ed84d2ddaeb9e14fd77b273e4c6e9654aecee165c11ae0e6`.
  Timed source archive SHA256:
  `8d453d984040625edc30ebe9bfd292a15d3b38b20a6a18dd4345b96474a184ac`.
* **64 retained selected-native invocation logs pass**, across both CU
  settings, plus five supplemental rejected-path logs (Q8 rows at K320/640,
  actual IQ4 dense23/24, expert6/7). The supplemental run used the restored
  governor and is correctness-only; its timings are not performance evidence.
  Native/GLSL builds, exact CPU repack tests, **18 Python ledger tests**, flag
  registration (`48 code / 32 test / 76 documented`) and shell syntax pass.
  The supplemental test binary adds K640 Q8 shape coverage; timed kernel and
  replay-binary sources did not change.
* The standalone diagnostic helper previously leaked its mask variable,
  retaining the last candidate expert log instead of a separate baseline log.
  Its variables are now local. Recorded selectors verify that the **60 wall
  captures and all profile pairs are unaffected**; no standalone expert A/B
  timing is claimed from the overwritten files.
* All **20 A/A2 output and state comparisons are byte-identical**.
  Candidate dumps inspect 3,276,800 output and 3,309,568 state elements per
  phase. Maximum incremental scaled-output/absolute-state drift is
  **1.33998e-3 / 2.19345e-4**; cumulative is
  **1.33622e-3 / 4.90379e-3**. These are accumulated synthetic diagnostics,
  not a real-pack model-quality pass or veto.
* Only **192.168.42.50** was used. Both `bc250-embed` and `bc250-rerank` were
  stopped for clean timing and restarted, with both health endpoints returning
  `{"status":"ok"}`. Original governor configuration is byte-identical,
  governor/service units are active, and original **40/40 CU routing** is
  restored. `/home/xander/hc-roof` was removed and absence tested.
  `roofline-excess-cleanup.txt` retains the final cleanup result.
  No production-ring or sibling blades were touched.

The earlier study follows for context; its numbers are not the new A/B result.

# Prior decode roofline: reassociation and bandwidth follow-up

2026-10-15; all flags remain **off by default**. Same .50-only, stock-safe
2000 MHz GFX operating point and replay protocol as the historical study below.
Each row is **five alternating A/B/A rounds**, independently seeded, at each
live CU setting. Intervals bootstrap rounds, not individual correlated tokens.

| Candidate | CUs | Six-layer wall before -> after ms | Saving | 95% saving CI, us |
|---|---:|---:|---:|---:|
| Readmitted: mask 219, dense 6, QSA 3 | 40 | 2.686937 -> **2.458141** | **8.515%** | 225.897-232.389 |
| Readmitted | 24 | 2.939492 -> **2.711450** | **7.758%** | 225.046-231.131 |
| Bandwidth: mask 251, dense 19, QSA 3, expert 2 | 40 | 2.685013 -> **2.406759** | **10.363%** | 276.285-279.937 |
| Bandwidth | 24 | 2.944236 -> **2.677356** | **9.064%** | 263.771-270.409 |

Bandwidth policy adds about **51.4 us / 34.1 us** to the readmitted saving
at 40/24 CUs, respectively. This difference compares separate five-round
studies, not a third directly paired experiment or its own confidence interval.

## Corrected quality gate

Floating-point reassociation is acceptable. **Real-pack full-vocabulary mean
KL(current || candidate) <=0.005 and top-1 agreement >=98.5%** are required,
with the same frozen corpus and pack bytes as Q4 qualification. Component
oracles remain bug checks, approximately fp32-reassociation tolerance; QSA
and alpha/beta component checks now use `1e-4*max(1,abs(reference))`.
The existing quant CPU sum-absolute-products bound and mixed-expert oracle
are unchanged; the latter reports up to **2.37e-4** in its existing mixed
Q4/Q5 fixture. No tolerance was loosened to mask a failing component.

The arbitrary `2e-6` full-block veto is removed. Baseline A/A2 is still
byte-identical; candidate dumps must be finite, nonempty and correctly sized.
All 3,276,800 output and 3,309,568 state floats per phase are inspected.
Across five seeds, readmitted max scaled output/state-absolute drift is
**1.3585e-3 / 2.9373e-4**; bandwidth is **1.1571e-3 / 5.1212e-3**, identical
at both CU settings. These synthetic accumulated drifts are diagnostics,
**not** a model-quality pass or automatic failure. Real-pack quality is
**not run / pending**, including the larger bandwidth-candidate drift.
`roofline-quality-candidates.json` lists dense 6, eight-split QSA, vec4
alpha/beta, both combinations, adaptive dense and expert prefetch for the
fleet gate. Verify the flagged shaders actually execute in real-pack paths;
graph-only expert execution currently retains its baseline path.

## Exact production-shape bandwidth results

Five intrusive profile pairs, separate from wall-time qualification.
Each cell is **before -> after us (before -> after % of 450 GB/s)**.
BW uses stored logical weight bytes, not hardware DRAM counters or an
arithmetic-free lower bound. HC rows show the attention-read scope; the
separate FFN-read scope remains in the dispatch CSV.

| Shape/type | Weight bytes | 40 CUs | 24 CUs |
|---|---:|---:|---:|
| QKV Q4, K2560 M10240 | 15,155,200 | 53.664 -> **50.106** (62.8 -> **67.2%**) | 61.826 -> **57.614** (54.5 -> **58.5%**) |
| GDN gate Q4, K2560 M6144 | 9,093,120 | 38.254 -> **31.104** (52.8 -> **65.0%**) | 42.494 -> **35.894** (47.6 -> **56.3%**) |
| GDN output Q4, K6144 M2560 | 9,093,120 | 30.032 -> **25.864** (67.3 -> **78.1%**) | 33.588 -> **29.222** (60.2 -> **69.2%**) |
| QSA query Q4, K2560 M12288 | 18,186,240 | 69.980 -> **59.580** (57.8 -> **67.8%**) | 74.092 -> **68.768** (54.5 -> **58.8%**) |
| QSA output Q4, K6144 M2560 | 9,093,120 | 30.872 -> **26.328** (65.5 -> **76.8%**) | 33.932 -> **29.564** (59.6 -> **68.4%**) |
| HC down Q4, K10240 M320, unchanged | 1,894,400 | 11.477 -> 11.565 (36.7 -> 36.4%) | 12.147 -> 12.053 (34.7 -> 34.9%) |
| HC up IQ4, K320 M10240, unchanged | 1,843,200 | 10.569 -> 10.565 (38.8 -> 38.8%) | 13.435 -> 13.397 (30.5 -> 30.6%) |
| Expert gate/up Q4, K2560 M640, top10 | 18,432,000 | 57.199 -> **56.643** (71.6 -> **72.3%**) | 74.181 -> **72.744** (55.2 -> **56.3%**) |
| Expert down Q5_1, K640 M2560, top10, unchanged | 12,288,000 | 38.171 -> 38.187 (71.5 -> 71.5%) | 46.128 -> 46.181 (59.2 -> 59.1%) |

**The >=85% large-shape target was not reached.** Best replay result is
78.1% (standalone output GEMV reached 80.3%). HC and experts remain major
gaps; unchanged-path noise is not credited as a win.

| Other component | 40 CU before -> after us | 24 CU before -> after us |
|---|---:|---:|
| QSA split, 257 records | 46.544 -> **13.732** | 46.696 -> **16.300** |
| QSA merge, now eight partials | 2.388 -> 5.928 | 2.428 -> 5.964 |
| Alpha/beta F32, K2560 M48, per dispatch | 5.207 -> **3.851** | 5.184 -> **3.230** |
| Router top10 | 9.259 -> **5.020** | 9.195 -> **4.984** |
| HC injection, attention-read | 6.677 -> **3.463** | 6.612 -> **3.451** |
| HC mixing, attention-read | 4.631 -> **2.329** | 4.544 -> **2.341** |
| Recurrence, unchanged | 28.936 -> 29.520 | 28.902 -> 29.248 |

Eight QSA splits save **29.272 us net of the more expensive merge** at
40 CUs, about **0.351 ms/model token** for twelve QSA layers, rather than
crediting the split-only ~0.39 ms opportunity. Router is 5.37 dependent-step
floors and HC mixing 2.49, using the sibling's 0.934 us reference; these are
comparisons, not measured universal limits. No floor microbench was duplicated.

## What was tried and why the selected policy helps

Read-only llama.cpp `mul_mat_vec_q4_k` and packed Q8 implementations informed
subgroup-only reductions and load granularity. Native packed INT8 dot remains
unavailable on this RADV target; no unsupported instruction assumption is made.
Every research choice is retained behind the registered roofline selectors.

* **Dense 15-18:** original eight-row layout, two/one lanes per 32-value group,
  aligned uvec2/uvec4 payloads, vec4 activations, prefetched metadata and a
  subgroup-only epilogue without LDS. Policy **19** selects two lanes/WG64
  for QKV/output, one lane/**true wave32** for gate/query, and baseline
  geometry for small/HC/IQ4 shapes. Subgroup-size-control is queried/enabled;
  this is not a half-populated wave64 approximation.
* **Dense 8-12:** one row per lane and one/two/four independent uvec4 streams,
  wave64 and true wave32. Large QKV regresses to roughly 61-79 us; HC down
  25-45 us. More register-level prefetch alone does not guarantee occupancy
  or decode-issue throughput. No ISA occupancy counter was collected.
* **Dense 13/14:** low-M HC down K-split four/eight, **including merge cost**:
  13.182/12.629 us versus 11.484 us baseline in the sweep. Extra dispatch
  and reduction outweigh the parallelism; not selected.
* **Dense 20:** true-wave32 IQ4 is 12.096 versus 10.533 us; not selected.
  **22:** two-lane WG128 regresses QKV to 67.673 us and output to 29.373 us.
* **Dense 21, actual repack/layout test:** expand compact scale/min metadata
  into wide F32 records, avoiding unpack work but increasing weight bytes
  **29.7%**. QKV/gate/output/query become 58.810/36.800/48.797/71.147 us.
  It loses even before accounting for larger traffic. Wide format tags and
  actual wide byte counts are used; no artificial compact-byte BW gain.
* **Expert:** two/four independent uvec4 block prefetches include scale/min
  words, wave64 and true wave32, plus true-wave32 down. Selector **2** keeps
  four prefetched blocks/wave64; gain is small (~0.6 us at 40 CU), unlike
  dense. Wave32 gate/up and down did not produce a credible improvement.

The archived sweep contains all component-oracle and three-seed exploratory
results; only final selected combinations have five-round 40/24-CU CIs.
Large-shape CPU quant oracles passed throughout; all 96 targeted test
invocations (both studies, both masks, both CU settings, twelve filters)
pass. Local C17/shader builds, 13 Python analysis tests, flag registration,
shell syntax and diff whitespace checks pass.

## Block accounting, projection and reproducibility

Final bandwidth candidate keeps **214 dispatches / ten profiled submissions**.
No-clocks controls: 2657.182 -> 2381.696 us (40 CU), 2914.826 -> 2646.305 us
(24 CU). Token-2048 intrusive controls: 4057.973 -> 3838.949 us and
4285.211 -> 4201.163 us. Ordered single pairs are not extra CIs or
long-context service qualification.

Type-weighted measured layer deltas, `36*GDN + 12*QSA`, project **2.0492 ms**
saved/model token (40 CU; 95% CI 2.0463-2.0523) and **2.0297 ms** (24 CU;
2.0246-2.0349). Conditionally subtracting the latter from the historical
37.437 ms eight-rank anchor gives **35.4073 ms/token**, arithmetic CI
35.4021-35.4124, or **28.24 TPS**. Readmitted-only 24-CU projection is
35.7092 ms/token. **Neither is measured fleet performance**; quality,
real-pack path coverage and operating-point transfer remain unqualified.

Evidence: `roofline-followup-captures.tar.gz`, SHA256
`e353dce74d9fdbff7847ec28b6115479942231d076d62912c31496ee7bac2e69`.
It includes `bandwidth/`, `readmitted/`, `requalified/` and operating-point
evidence, excluding large float dumps and governor logs. Summaries/dispatch
CSVs are separately named `roofline-readmitted-*` and `roofline-bandwidth-*`;
the original exact-study captures are preserved. Reproduce with
`tools/roofline-bandwidth.sh`, `tools/roofline-wide.sh`,
`tools/roofline-requalify.sh` through the .50-only runner, then
`roofline-ledger.py CAPTURES --mask 219 --dense 6` or `--mask 251 --dense 19`
with explicit `--output` and `--dispatch-csv`.
Services and original governor config were restored and health checked;
40/40 routed CUs restored. `/home/xander/hc-roof` was removed and its absence
independently asserted by the cleanup script.

---

# Historical first iteration

**Superseded numerical policy:** the first-iteration results below document
the historical bit-exact candidate, not the current promotion requirement.
An additional follow-up section accompanies the readmitted and bandwidth
captures. The old `2e-6` block veto was unnecessarily strict: the reported
`3.078e-6` QSA drift and other reassociation differences are **not model
quality failures**. Dense policy 6, eight QSA splits and vec4 alpha/beta are
readmitted behind their existing opt-in flags. The real gate is full-vocabulary
real-pack mean KL <=0.005 and top-1 agreement >=98.5% against current.
`roofline-quality-candidates.json` lists pending individual and combined gates.
No real-pack quality measurement or deployment is claimed here.

2026-10-15; branch `cuppaxanax/roofline`, base `origin/main a63a166` plus
`cuppaxanax/decode-fusion f207404`. **Opt-in synthetic replay win, not deployment.**

| Live dispatch configuration on .50 | Six-layer baseline -> candidate | Saving | Round-bootstrap 95% saving CI |
|---|---:|---:|---:|
| 40 routed CUs | **2.684742 -> 2.565510 ms** | **4.441%**, 119.232 us | **117.801-121.123 us**, 4.390-4.507% |
| 24 routed CUs | **2.943556 -> 2.821084 ms** | **4.161%**, 122.472 us | **120.198-124.644 us**, 4.083-4.233% |

Qualified combination: **`FG_ROOFLINE=27 FG_ROOF_DENSE=7 FG_ROOF_QSA=3`**,
with `FG_DECODE_FUSION` unset. All roofline flags remain off by default.
Every measured output float and final GDN-state float is **byte-identical**
in all ten qualified A/B/A rounds. The experiment does not reach the bulk
450 GB/s roof; quantized GEMVs still contain significant decode/issue work.

## Hardware, clock and floors

Only **192.168.42.50** was contacted. No access to .42-.49 or .51-.53.
The read-only live-CU manager reports SPI routing `0x1f` in all four rows,
**40/40 routed CUs**, while Vulkan and the boot topology report **24**.
These are different quantities. The existing saved 40-CU profile was unchanged.
For the second qualification, `bc250-cu-live-manager --yes stock-dispatch`
restricted routing to `0x07`, independently read back as **24/40**. Its saved
profile was reapplied afterward and read back as **40/40**. No reboot,
enumeration rewrite, voltage change, or saved-table change was made.

The normal governor was stopped only during timing. An experiment-local copy
of its config changed its existing stock-safe GFX minimum from 1000 to
2000 MHz, keeping the existing 2000 MHz maximum and voltage points. Endpoint
captures show 2000 MHz. **This is GFX clock, not a 2000 MHz MCLK setting**:
the firmware MCLK metric still reads 450 MHz. Original config bytes were
compared after restoration. No memory-clock change was attempted.

`PHYSICS.md` establishes approximately 450 GB/s logical read payload for large
vec4/WG128 streams. It does not make Q4 dequantization free. Here the existing
compact layouts, uvec4 metadata loads, vec4 activations and multiple streams
are retained; WG64/128/256 and rows/WG are tested instead of assuming the
streaming-read geometry transfers unchanged to GEMV.

The sibling's `cuppaxanax/latency 8c4f9be` report supplies the floors; no floor
microbench was duplicated. Useful dependent steps are about **0.934 us** and
subgroup-reduction work about **1.44 us** on its .53/24-CU fixed-clock run.
The **29.5-34 us submit/fence round trip is not a per-dispatch floor**.
Floor ratios below are reference comparisons, not universal hard bounds.
All timing here uses RADV GFX1013 **wave64**; wave32/other subgroup devices
retain the baseline rather than executing wave64-only variants. The portable split-count research bit 64 is independent
of that wave64 guard and is not part of the qualified mask.

## Recomputed priorities

`roofline-dispatches.csv` maps all **214 baseline replay dispatches per block**
to the separate intrusive scoped means, exact grids, and logical reference
bytes. There are 428 rows across the two CU configurations. The ranking uses
the actual model mix, **36 GDN + 12 QSA**, not eight copies of this 4-GDN/2-QSA
block. Excess is `max(0, scoped_us - max(bytes/450000, 0.934))`.
HC rows below combine the attention and FFN scopes, each occurring 48 times.

| Baseline component, 40 CUs | Occurrences/model token | Estimated excess ms/token |
|---|---:|---:|
| Expert gate/up | 48 | 0.786 |
| GDN QKV | 36 | 0.727 |
| HC Q4 down, both scopes | 96 | 0.692 |
| GDN gate | 36 | 0.652 |
| HC IQ4 up, both scopes | 96 | 0.609 |
| QSA split attention | 12 | 0.546 |
| HC injection, both scopes | 96 | 0.542 |
| GDN recurrence | 36 | 0.526 |
| Expert down | 48 | 0.515 |
| Router top-10 | 48 | 0.398 |
| QSA query projection | 12 | 0.358 |
| GDN output projection | 36 | 0.349 |
| HC mixing, both scopes | 96 | 0.342 |

These are prioritization estimates, not independent achievable savings or
memory-controller counters. Shared BF16/quantization scopes pool different
shapes and are excluded from this shortlist; the CSV preserves their estimates
and binding-specific byte references without inventing per-shape timings.
The ranking changes materially at 24 CUs: expert gate/up excess becomes
1.591 ms, QKV 1.016 ms and expert down 0.905 ms.

## Per-kernel results

Microseconds, means of five separately seeded diagnostic pairs at token 256.
These runs use intrusive profiling and disable static replay; they diagnose
kernels, **not** the authoritative block speedup. Unchanged kernels' small
differences are noise, not credited changes. BW is cooked-weight/logical-payload
efficiency against 450 GB/s, not measured DRAM traffic. Attention's unique-byte
reference omits per-head reloads and arithmetic; recurrence's one-read/write
reference omits possible rereads.

| Component | 40 CU before -> after us | 24 CU before -> after us | Candidate reference, 40 CU |
|---|---:|---:|---|
| GDN QKV, K=2560 M=10240 | 53.870 -> **52.204** | 61.894 -> **57.718** | 33.678 us ideal; **64.5% BW** |
| GDN gate, K=2560 M=6144, unchanged | 38.330 -> 38.344 | 42.610 -> 42.618 | 20.207 us ideal; 52.7% BW |
| GDN output, K=6144 M=2560 | 29.898 -> **29.168** | 33.654 -> **32.528** | 20.207 us ideal; **69.3% BW** |
| QSA query, K=2560 M=12288 | 70.228 -> **65.224** | 74.168 -> **70.164** | 40.414 us ideal; **62.0% BW** |
| QSA output, K=6144 M=2560 | 30.652 -> **29.856** | 34.184 -> **32.948** | 20.207 us ideal; 67.7% BW |
| QSA split attention, 257 records | 46.460 -> **44.952** | 46.560 -> **45.024** | 0.785 us unique-byte ideal; still strongly non-streaming |
| GDN recurrence, unchanged | 28.788 -> 28.678 | 29.232 -> 29.178 | 14.184 us logical-byte ideal; 49.5% |
| Router top-10, 512 logits | 9.219 -> **5.052** | 9.269 -> **5.005** | 3.51x the 1.44 us reduction reference |
| HC attention injection | 6.672 -> **3.441** | 6.604 -> **3.455** | 2.39x reduction reference |
| HC FFN injection | 6.487 -> **3.239** | 6.445 -> **3.252** | 2.25x reduction reference |
| HC attention mixing | 4.548 -> **2.313** | 4.552 -> **2.340** | 2.48x dependent-step reference |
| HC FFN mixing | 4.441 -> **2.176** | 4.396 -> **2.152** | 2.33x dependent-step reference |
| HC attention down, unchanged | 11.541 -> 11.404 | 12.167 -> 12.165 | 4.210 us ideal; 36.9% BW |
| HC attention IQ4 up, unchanged | 10.507 -> 10.535 | 13.385 -> 13.409 | 4.096 us ideal; 38.9% BW |
| Expert gate/up, unchanged | 57.328 -> 57.341 | 74.113 -> 74.041 | 40.960 us selected-weight ideal; 71.4% BW |
| Expert down, unchanged | 38.045 -> 38.192 | 46.167 -> 46.165 | 27.307 us selected-weight ideal; 71.5% BW |

HC RMS norm stays about 2.49 us (attention) / 2.05 us (FFN); 320-element
SiLU stays about 0.85 us. They are already much closer to the useful-work
reference than router or serial injection. No blanket barrier relaxation,
weight-format change, lower-than-Q4 quantization, FP16 narrowing or native
INT8-dot assumption is introduced.

## Hypotheses, variants and rejection

The following bits are registered in `tools/check-flags.sh` and `docs/FLAGS.md`.
Set flags before recording static command buffers. Existing graph-only expert
paths remain baseline.

* **Bit 1, compact Q4 GEMV:** excessive cross-wave reduction/metadata overhead
  and row geometry, rather than the streaming bus alone. Five geometries cover
  r4/WG64, r4/WG128, r8/WG256, r8/WG128 and r8/WG64, with unroll hints.
  The retained `FG_ROOF_DENSE=7` preserves the original accumulation partition:
  r8/WG64 for large matrices except the GDN gate, whose original WG128 remains.
  One-subgroup GEMV bypasses its unnecessary LDS write/barrier/final combine.
  M<2560 and non-Q4 formats keep their original paths. This is why HC down
  does not inherit a large-matrix geometry regression.
* **Bit 2, QSA geometry:** four 768-thread workgroups at 257 records leave
  little workgroup-level parallelism. `FG_ROOF_QSA=1/2/3/4` tests register-tiled
  one-head/WG64, LDS three-head/WG192, six-head/WG384 and one-head/WG64.
  At the existing two splits these take **55.82/58.04/45.22/126.76 us** in the
  initial sweep. The qualified six-head variant modestly improves the grid
  from 2x2 to 4x2 while retaining softmax partition/order and byte parity.
* **Bit 64, aggressive QSA splits, not qualified:** eight splits reduce the
  original geometry from about **46.6 to 14.26 us**; six-head geometry gives
  **13.72 us**. This confirms serialization/occupancy, not 46 us of inherent
  dispatch cost. Split/merge reassociation passes the existing component
  4e-4 attention tolerance, but seed 4 reaches **3.078e-6** scaled full-block
  error, above the extra 2e-6 gate. It is excluded from the qualified mask
  and all credited projection savings. Its approximately 0.39 ms/model-token
  opportunity is not counted as a qualified gain.
* **Bit 4, recurrence, rejected:** combine state update with output
  accumulation, removing a later global reread/barrier, and unroll the
  dependent loops. State remains exact in the existing oracle and output
  passes its 3e-4 tolerance, but **34.7 us versus about 28.8 us** loses.
  Code/register pressure and long dependent chains are hypotheses; no ISA
  register/occupancy counter was collected. Do not enable it.
* **Bit 8, router:** one wave owns eight logits per lane, with subgroup
  max/min-ID selection and no workgroup LDS/barrier per winner. Tie ordering,
  invalid-logit handling and final probability order are retained. The ten
  dependent selections remain real work: **9.22 -> 5.05 us**, not 0.005 us.
* **Bit 16, HC:** specialize the actual four-group case, share each input
  across four independent FMA streams, retain the first two cross-wave
  reduction stages and replace the remaining tree by same-order shuffles.
  Mixing unrolls the four groups and prefetches four independent partial
  loads before serial summation. Injection and mixing together save about
  **5.3 us per HC read**, without removing a dispatch.
* **Bit 32, expert gate/up, not selected:** same-order subgroup-XOR reduction
  replaces the eight-lane LDS epilogue. Random mixed-format and production
  shape tests pass, but **56.85 versus 57.09 us** in one diagnostic is too
  small to claim a useful win. Existing uvec4 payload loads and software
  integer-dot lowering remain; expert down is unchanged.
* **Bit 128, alpha/beta controls, not qualified:** vec4/WG256 F32 GEMV gives
  **3.83 versus 5.17 us** at K=2560/M=48, but changes the reduction partition.
  Its component CPU/current-kernel checks pass the existing 2e-4 tolerance.
  The aggressive mask 219, which includes this and reassociated dense paths,
  fails the extra full-block gate at **7.656e-5** scaled error. Its timing
  is not presented as the qualified result.

The original dense selection `FG_ROOF_DENSE=6` also reassigns the GDN gate
and output reduction partitions. Dense-only seed 1 reaches 1.932e-4 scaled
full-block error despite passing the existing per-GEMV oracle. The qualified
selection 7 deliberately forgoes its approximately 5 us GDN-gate saving.
No component tolerance was loosened after these failures.

## Replay and numerical qualification

The fusion benchmark constructs the **same synthetic, nonzero tensor formats
and production shapes** as its source experiment: owner-local layers 6-11,
4 GDN + 2 QSA, 512 experts, real top-10 routing, Q4_K gate/up, Q5_1 down,
compact/wide Q4 dense, IQ4 for K=320/640, F32 router/controls and BF16 index
projections. Expert matrices repeat at distinct physical expert addresses.
There is no real pack, expert diversity, model KL, PLE, head, ring transport,
MTP, multi-session or long-context quality qualification.

For **each CU configuration**, five fresh seeds use **A/B/A**, alternating
baseline/candidate/baseline, with 256 warmup tokens and **five windows x 64**
measured tokens, positions **256-575**. Both static GDN replay slots stay
recorded. QSA retains dynamic recording/flush behavior. Two timestamps per
layer are read after draining, never an intrusive profile inside the main run.
Each round's A is the mean of its two baselines. The deterministic bootstrap
resamples **five rounds**, not 1,600 correlated tokens: 10,000 resamples,
empirical 95% intervals. This is single-blade, fixed-clock qualification,
not an uncertainty interval over the fleet or all workloads.

Before deleting dumps, all **3,276,800 output floats** and **3,309,568 final
GDN-state floats per phase** are checked. A/A2 dumps are compared bytewise.
The extra full-block check allows at most `2e-6*max(1,abs(reference))` for
outputs and 3e-5 absolute state error; **the selected mask actually changes
zero bytes** in every seed on both CU configurations. Hashes also match in
paired intrusive, no-clock and token-2048 controls. Rejected attempts and
their measured errors remain in the archive; gates were not relaxed.

`roof_dense_exact` separately byte-compares **every output row** for three
input/weight cases, including zero input, at KxM =
2560x10240, 2560x6144, 6144x2560, 2560x12288 and 10240x320.
The earlier geometry sweep runs the existing CPU quant oracle at all nine
ordinary production shapes, sampling 32 distributed rows for timings.
`roof_qsa` compares all 6,144 outputs for seven record counts
1/7/8/31/257/513/2051, four geometries and 2/4/8 splits, using the actual
current attention path as reference and its unchanged 4e-4 tolerance.
Router tests include ties, finite logits, NaN, +inf and all -inf, with IDs,
gates and tile schedules byte-compared.

On both CU configurations, default and selected paths pass
`dense_quant_compact`, `decode_fusion_router`, `roof_qsa`, `roof_controls`,
`hc_inject_partial`, `gr_partial_boundaries`, `gdn_algebraic`,
`expert_decode_fused_random`, `q8_decode_shape_parity`,
`batch_submission_parity`, `pipeline_flush_parity`, `static_batch_replay`.
The final additional all-row dense oracle passes at 40 CUs.
Local WSL builds of serving/bench/test/shaders, the flag gate, 11 Python
analysis regression tests and `git diff --check` pass.
No full Vulkan-suite or real-model quality pass is claimed.

## Block accounting and conditional eight-rank projection

| Type mean | 40 CU before -> after us | 24 CU before -> after us |
|---|---:|---:|
| GDN layer GPU | 387.267 -> 369.742 | 433.527 -> 414.241 |
| QSA layer GPU | 510.509 -> 486.643 | 545.314 -> 523.068 |

Dispatches remain **214 -> 214**, with ten intrusive-profile submissions;
no global barriers were removed. Diagnostic envelopes are
2538.264 -> 2419.752 us (40 CU) and 2801.800 -> 2672.984 us (24 CU).
Intrusive attributed gaps are 106.272 -> 101.608 us and
104.848 -> 100.936 us respectively; the gain is mainly kernel work, not
eliminating a large hidden launch bucket.

A single no-layer-clock control pair gives 2664.833 -> 2546.588 us at 40 CU
and 2915.392 -> 2796.168 us at 24 CU, consistent with the repeated win.
Token-2048 intrusive controls give 4049.891 -> 3944.582 us and
4297.284 -> 4213.716 us wall time, respectively. These controls are ordered
single pairs, not additional CIs or long-context service qualification.

Type-weight only the measured layer GPU deltas:
`saving = 36*GDN_delta + 12*QSA_delta`.
At 40 CUs this is **0.9173 ms/model token**, round CI 0.9140-0.9214 ms.
At production-like **24 CUs**, it is **0.9613 ms**, CI **0.9469-0.9753 ms**.
These must not be obtained by multiplying this mixed block's wall saving by
eight.

Conditionally transferring the absolute 24-CU saving to the historical
29.971 ms rank-service sum yields **29.010 ms**. Holding its 2.122 ms prep
and 5.344 ms other terms fixed gives **37.437 -> 36.476 ms/token**,
or **26.71 -> 27.42 TPS**. The arithmetic-only CI is 36.462-36.490 ms/token;
it omits the much larger uncertainty from transferring synthetic weights,
short contexts and fixed clocks to production. The analogous 40-CU anchor
is 36.520 ms/token. **Neither is a measured eight-rank result.**
No production flag was enabled or pack changed; existing real-pack KL/top-1
and service/fleet gates remain mandatory.

## Reproduce, evidence and restoration

Run only against .50 while its experiment directory is available:

```powershell
Set-Location D:\looking-glass-labs\fg-wt\roofline
New-Item -ItemType Directory -Force .build | Out-Null
wsl -d Ubuntu-24.04 --cd /mnt/d/looking-glass-labs/fg-wt/roofline --exec env TMPDIR=.build make -s -j8 flash-gordon tools/bench-decode-fusion tests/test_fg_vk
tar -cf .build\roof-source.tar --exclude=*.o --exclude=*.d --exclude=*.spv include src shaders vendor tools tests docs Makefile glslc
& .\tools\Invoke-RooflineFleet.ps1 -Targets @('192.168.42.50') -ScriptPath .\tools\roofline-smoke.sh -Sudo -UploadPath D:\looking-glass-labs\fg-wt\roofline\.build\roof-source.tar -UploadDestination /home/xander/hc-roof/source.tar
& .\tools\Invoke-RooflineFleet.ps1 -Targets @('192.168.42.50') -ScriptPath .\tools\roofline-qualify.sh -Sudo
& .\tools\Invoke-RooflineFleet.ps1 -Targets @('192.168.42.50') -ScriptPath .\tools\roofline-release.sh -Sudo -DownloadSource /home/xander/hc-roof/captures.tar.gz -DownloadPath D:\looking-glass-labs\fg-wt\roofline\results\hillclimb\roofline-captures.tar.gz
& .\tools\Invoke-RooflineFleet.ps1 -Targets @('192.168.42.50') -ScriptPath .\tools\roofline-cleanup.sh -Sudo
```

The adapter invokes the canonical `bc-250-dbg\Invoke-BC250Fleet.ps1` in memory,
restricts targets to .50, and confines script staging to `hc-roof`, not OS
temporary directories. `roofline-sweep.sh` and `roofline-prequal.sh` preserve
the intermediate experiments; no floor microbench is part of these scripts.
`roofline-numerics.sh` diagnoses the historical failed masks and assumes those
rejected captures exist; it is not needed for a fresh qualification.

`roofline-captures.tar.gz` contains final/rejected JSONL, parity, build/test,
clock, CU routing and restoration records; large duplicate float dumps,
service-unit content and verbose governor logs are excluded.
Archive SHA256:
`82bf84bdc1f812693c95d338d71efed97f0900e92f70b81f4b739db59366b9bc`.

```powershell
New-Item -ItemType Directory -Force .build\roof-captures | Out-Null
tar -xf results\hillclimb\roofline-captures.tar.gz -C .build\roof-captures
python tools\roofline-ledger.py .build\roof-captures\final --output results\hillclimb\roofline-summary.json --dispatch-csv results\hillclimb\roofline-dispatches.csv
python -m unittest tools\test_roofline_ledger.py tools\test_fusion_ledger.py
```

Embedding/reranking services were stopped for clean timing and restarted
after every timed experiment, including failed gates. Final independent
release checks show both **8091/8092 `/health` responses `{"status":"ok"}`**,
all three services active, original governor bytes unchanged and 40 routed
CUs restored. After downloading the archive, cleanup independently repeated
those checks, removed `/home/xander/hc-roof`, and asserted its absence:
`hc-roof absent; services healthy; original governor and CU profile restored`.
No API or artifact on another blade was touched.

Read-only inputs: `flash-gordon\results\hillclimb\PHYSICS.md`,
`fg-wt\fusion\results\hillclimb\FUSION.md`,
`fg-wt\q4-dense\results\hillclimb\q4-compact.md`,
`flash-gordon\results\hillclimb\LATENCY.md`, and the local
`llama.cpp\ggml\src\ggml-vulkan\vulkan-shaders\mul_mat_vec*.comp`
subgroup/row-tiling reference. No llama.cpp file was modified.
