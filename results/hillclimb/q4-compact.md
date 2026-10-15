# Exact compact dense Q4/Q5 follow-up

Branch `cuppaxanax/q4-dense`; follows wide-SoA commit `a9b56d6`.
**Kernel gate passes; not deployed; quality gates remain mandatory.**
Only `.51` (40 CU), `.52` (24 CU), and read-only artifact inventory on `.53`
were contacted. No production `.42`–`.50` access or service changes.
CU configurations are user-described, not an independent mask readback.

## Result and protocol

All **27 ordinary GEMV, 63 ordinary prefill, and nine protected-plan GEMV**
comparisons per blade beat same-run cooked Q8, with scalar parity: 198/198
across `.51` and `.52`. Each timing uses Vulkan GPU timestamps, 20 repetitions,
rotating candidate/wide/Q8 order. Warmup uses 64 continuous batches of 256
dispatches before measurement. Subgroup size is 64. No int8 dot, clock changes,
FP16 weight narrowing, or precision concessions.

The wide baseline is independently cooked from the **same quantized weights**
as the candidate; Q8 is independently initialized at the same shape. Small
correctness tests check all rows/tokens; benchmark parity samples 32 distributed
rows for every token. Retained-wide shapes alias the same timing, not a noisy
duplicate dispatch. See `q4-compact.json` for complete logs, rejected experiments,
clock-ramp evidence, validation, and inventory.

One earlier CPU-synchronous warmup run on `.52` had a narrow first-run
K=2560/M=6144 loss: 64.028 us compact versus 63.892 Q8 and 56.604 wide.
Its warmed repetitions won. This evidence is preserved, not silently removed.
The definitive continuously warmed rerun below passes all comparisons.
These are warm synthetic kernels, not a cold-clock or serving-latency promise.

## GEMV: selected layout against previous wide layout and Q8

Times are microseconds; bandwidth is decimal **source/cooked GB/s**.
Speedups are baseline time / selected time. Tables use the third warmed run;
all three runs are in the evidence. The two IQ4 shapes retain 18 B/32 but use
the new guarded small-shape shader. K=2560/M=640 and 512 retain wide storage.

### `.51`, 40 CU

| K x M | Old wide us | Selected us | Q8 us | vs Q8 | vs wide | Source/cooked GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 2560x10240 | 58.826 | 53.248 | 79.752 | 1.498x | 1.105x | 276.9/284.6 |
| 2560x6144 | 43.142 | 37.980 | 57.500 | 1.514x | 1.136x | 232.9/239.4 |
| 6144x2560 | 52.528 | 29.486 | 76.284 | 2.587x | 1.781x | 300.1/308.4 |
| 2560x12288 | 76.322 | 69.968 | 98.258 | 1.404x | 1.091x | 252.9/259.9 |
| 10240x320 | 12.742 | 10.664 | 27.082 | 2.540x | 1.195x | 172.8/177.6 |
| 320x10240 | 9.720 | 9.436 | 12.682 | 1.344x | 1.030x | 195.3/195.3 |
| 640x2560 | 6.032 | 5.396 | 7.256 | 1.345x | 1.118x | 170.8/170.8 |
| 2560x640 | 5.760 | 5.760 | 7.244 | 1.258x | 1.000x | 160.0/213.3 |
| 2560x512 | 5.350 | 5.350 | 6.638 | 1.241x | 1.000x | 137.8/183.7 |

### `.52`, 24 CU

| K x M | Old wide us | Selected us | Q8 us | vs Q8 | vs wide | Source/cooked GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 2560x10240 | 63.516 | 60.416 | 82.464 | 1.365x | 1.051x | 244.1/250.8 |
| 2560x6144 | 44.262 | 41.878 | 60.296 | 1.440x | 1.057x | 211.3/217.1 |
| 6144x2560 | 55.386 | 34.038 | 82.386 | 2.420x | 1.627x | 259.9/267.1 |
| 2560x12288 | 79.000 | 75.104 | 100.066 | 1.332x | 1.052x | 235.6/242.1 |
| 10240x320 | 13.300 | 11.316 | 34.718 | 3.068x | 1.175x | 162.9/167.4 |
| 320x10240 | 13.692 | 13.294 | 14.230 | 1.070x | 1.030x | 138.6/138.6 |
| 640x2560 | 7.102 | 6.118 | 8.674 | 1.418x | 1.161x | 150.6/150.6 |
| 2560x640 | 5.782 | 5.782 | 7.236 | 1.251x | 1.000x | 159.4/212.5 |
| 2560x512 | 5.090 | 5.090 | 6.592 | 1.295x | 1.000x | 144.8/193.1 |

Protected Q5_K projections, same final run:

| Blade | K x M | Q5 selected us | Q5 wide us | Q8 us |
|---|---|---:|---:|---:|
| .51 | 6144x2560 | 40.244 | 41.966 | 75.924 |
| .51 | 2560x12288 | 78.908 | 96.696 | 97.214 |
| .51 | 2560x512 | 5.930 | 5.930 | 6.572 |
| .52 | 6144x2560 | 45.258 | 59.518 | 81.818 |
| .52 | 2560x12288 | 88.824 | 100.864 | 99.876 |
| .52 | 2560x512 | 6.040 | 6.040 | 6.570 |

The other six protected-plan shapes also win against Q8. Protected prefill
was not rerun. The vocabulary head is **kept Q8**, not requalified as Q6.
Its earlier Q6 loss remains documented in `q4-kernels.md`.

## Prefill

N=8,16,32,64,128,256,512: all 63 cases per blade beat Q8. The table gives
the minimum Q8 speedup and the range of old-wide/selected speedups across N.
Values below 1 in the latter columns are an explicit regression versus wide,
**not** versus Q8. Full times and bytes are in the evidence.

| K x M | .51 min vs Q8 | .51 vs wide range | .52 min vs Q8 | .52 vs wide range |
|---|---:|---:|---:|---:|
| 2560x10240 | 1.135x | 0.929..0.968x | 1.308x | 0.986..1.011x |
| 2560x6144 | 1.112x | 0.952..0.986x | 1.240x | 0.983..1.011x |
| 6144x2560 | 1.505x | 1.099..1.837x | 1.359x | 1.142..1.541x |
| 2560x12288 | 1.205x | 0.939..0.972x | 1.349x | 0.989..1.011x |
| 10240x320 | 1.161x | 1.110..1.470x | 1.149x | 1.103..1.241x |
| 320x10240 | 1.100x | 0.995..1.005x | 1.088x | 0.997..1.004x |
| 640x2560 | 1.037x | 0.997..1.002x | 1.039x | 0.998..1.002x |
| 2560x640 | 1.154x | 1.000..1.000x | 1.188x | 1.000..1.000x |
| 2560x512 | 1.171x | 1.000..1.000x | 1.186x | 1.000..1.000x |

One physical representation per matrix is selected for GEMV. Compact prefill
dequantizes once into the existing **FP32 LDS tile**, not packed-FP16 MMQ.
For some K=2560 matrices on `.51`, wide prefill is up to 7.6% faster.
Keeping both layouts would defeat the single-copy memory requirement; this
tradeoff is explicit. No prefill case regresses against the requested Q8 gate.

## Exact compact encoding and diagnosis

Eight-row tiles, G=K/32 groups:

1. Original fp16 d/dmin pairs: `[256-superblock][8 rows][4 B]`, total G*4 B.
2. Expanded original six-bit sc/min codes as bytes:
   `[32-group][8 rows][2 B]`, total G*16 B.
3. Contiguous row/group quant streams: 16 B/32 for Q4, 20 B/32 for Q5.

Thus Q4 is **18.5 B/32 = 4.625 bpw**, versus 24 B/32 wide, a **22.9%**
reduction; Q5 is **22.5 B/32 = 5.625 bpw**, versus 28, a **19.6%** reduction.
Original GGML is respectively 18 and 22 B/32. Rows are padded to eight; all
measured shapes divide eight. IQ4 keeps its original 18 B/32 SoA layout.

No scale is narrowed: half d has at most 11 significant bits, d*sc at most 17,
and multiplying by a Q5 integer at most 22, all representable in FP32's 24.
The original min term is reconstructed the same way. Independent CPU/GPU
one-hot dequant checks cover exact weights; dot-product parity allows FP32
reassociation bounded by `2e-5*max(1,sum(abs(products)))`.

`fg_dense_prefer_compact` selects IQ4 always, and Q4/Q5 when M>1024 or K>=8192;
the other requested Q4/Q5 shapes retain wide metadata. IQ4 now skips the whole
inactive group and eliminates the final LDS reduction/barrier for one subgroup.
It retains the exact shared integer lookup table.

Compaction improves GEMV time but does not preserve wide's physical GB/s:
at 2560x10240 on `.51`, ~334 GB/s wide becomes ~285 GB/s compact.
Extra exact half/code reconstruction consumes some of the saved bandwidth;
the 1.84x ideal Q8/compact byte ratio is not fully achieved. This is consistent
with added instruction/register pressure, not evidence of an all-memory limit.
Only 6144x2560 reaches the ~300 GB/s source goal here.

Rejected: whole-superblock amortization (~339 us versus ~59 us wide on the
large shape), subgroup-shuffle IQ4 LUT (~2x slower), and WG32/WG128 IQ4 variants.
The flat-group representation and WG64/r4 IQ4 win. Forcing compact on the
small M=640/512 cases regresses, so it is not selected.

## Deployment memory: single-copy solved; actual eight-rank totals blocked

`q4-repack --cook-dense --keep-q8-head` cooks at repack time. Layout tags 6
(wide SoA) and 7 (compact), GGML type, shape, exact stored byte count, and
SHA256 are sealed into the manifest. The loader maps views of those bytes
directly into its verified arena, including replicated loads. No original
dense GPU allocation remains beside a cooked buffer. Legacy raw-pack runtime
cooking remains available but is not recommended for deployment.

**An honest numeric per-rank abliterated ledger cannot be supplied yet.**
`/home/xander/fg-ring-pack-abliterated` is absent on **all three permitted
blades**, as preserved in the inventory. `.53` has the verified 3,447,677,568-byte
ordinary overlay, not the full pack. No full cached local manifest was found;
the active production peer also confirmed it has none already downloaded.
No production read was made to get it. Expert placement/representation and
alignment cannot be inferred safely from the overlay's size.

The read-only `--plan` mode needs only that real sealed manifest plus the
existing overlay. It emits **all eight** rows, including six-layer counts:

```
PLAN_RANK rank=R layers=6 resident_weight_bytes=W common_bytes=C expert_bytes=E endpoint=... residency_budget_bytes=B
PLAN_COORDINATOR_RING rank=0 resident_weight_bytes=... foreign_global_bytes=...
```

W is the exact aligned owner-local weight arena, **with Q8 head and without
duplicate cooked buffers**. The separate coordinator-ring line accounts for
replicated foreign global common weights and local MTP. B includes unchanged
manifest host/transient/KV/scratch/driver reservations: a budget, not measured
live Vulkan memory. Planning creates no output and does not prove payload
hashes or lossless routers. The staging script runs the plan before copying
and again against the sealed result, saving exact per-rank totals for review.

## Exact staging commands — future lease only, never deployment

From this branch after committing, in PowerShell 7:

```powershell
Set-Location D:\looking-glass-labs\fg-wt\q4-dense
git archive --format=tar --output=.\q4-stage.tar HEAD

# ONLY after .42-.49 have been released to this task:
& .\tools\q4-stage-pack.ps1 -SourceArchive .\q4-stage.tar `
  -OutputDir .\results\q4-stage-compact -Label q4-compact -AfterLease
```

Without `-AfterLease`, the script refuses **before any network access**.
It retrieves `.53:/home/xander/hc-q4/dense-q4.gguf`, verifies SHA256 after
transfer, uploads/builds the exact archive on eight owners, and discovers the
source manifest under `/home/xander/fg-ring-pack-abliterated`. All work is under
`/home/xander/hc-q4/stage-q4-compact/`. It checks identical source manifest SHA,
eight unique endpoint-derived ranks and six layers/rank; then each owner runs:

```sh
./tools/q4-repack "$manifest" /home/xander/fg-ring-pack-abliterated \
  "$root/dense-q4.gguf" "$root/pack" "$rank" --keep-q8-head --cook-dense
./tools/q4-repack "$root/pack/manifest.fgm" "$root/pack" "$root/dense-q4.gguf" \
  "$root/unused" --keep-q8-head --cook-dense --plan
```

All eight resulting manifest hashes must match. Local `preflight.json` and
`staged.json` preserve the ledgers; remote `sealed-resident-ledger.txt` does too.
The output is a complete eight-owner **owner-local** installation, not eight
portable copies: each owner has local experts and foreign sparse common files.
Keep the baseline directory for absolute tokenizer/ngram/host-cache symlinks.
Existing output packs are refused. Nothing starts/stops workers, changes
services, or deploys. This full real-pack path cannot be executed until released.

After staging, preserve ordinary worker/rank/listen settings, use the staged
executable and manifest, and leave `FG_DENSE_SOA`/`FG_DENSE_K_COOKED` unset.
Baseline/candidate logits capture and KLD/top-1 commands are in
`docs/Q4_DENSE.md`; mean KL<=0.01 nat, p99<=0.05, top-1>=97% remain required,
along with existing reference/router/prefill/decode/long-context gates.

## Validation and reproduction

On `.51`, the final sources build and pass core, actual file-cooker bytes/SHA,
exclusive fixture preservation,
sealed ordinary/replicated loader/cap/corruption tests, compact/wide dense
Vulkan oracles, and existing cooked-Q8 GEMV/prefill parity. Local Python:
eight tests, seven pass and optional native-library test skipped. Offline
PowerShell staging tests pass lease refusal, transfer hash rejection,
source/candidate manifest disagreement rejection, and reversed endpoint/rank
mapping. No real future production staging command was executed.
Known unrelated full-suite blockers remain as recorded in `docs/Q4_DENSE.md`;
no full-suite pass or model quality result is claimed.

Reproduce on permitted idle blades only, using the safe fleet wrapper:

```powershell
tar -cf .\compact.tar .\include .\src .\tests .\shaders .\vendor .\Makefile .\glslc .\tools
& .\tools\Invoke-Q4Fleet.ps1 -ScriptPath .\tools\q4-compact-final.sh `
  -Targets @('192.168.42.51','192.168.42.52') `
  -UploadPath .\compact.tar -UploadDestination /home/xander/hc-q4/compact.tar -PassThru
& .\tools\Invoke-Q4Fleet.ps1 -ScriptPath .\tools\q4-compact-validate.sh `
  -Targets @('192.168.42.51') -PassThru
& .\tools\test_q4_stage.ps1
```

The wrapper uses the required shared fleet runner with staging confined to
`hc-q4`, not OS temporary directories. Verified `.53` artifacts are untouched.
