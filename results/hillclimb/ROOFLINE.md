# Decode roofline experiment

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
