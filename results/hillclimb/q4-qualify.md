# Full-ring compact Q4 qualification

2026-10-15, `cuppaxanax/q4-dense`. Production lease: `.42`–`.49`; `.50` is
excluded. Preserved `.53` overlays supply the dense-only weights. This report
supersedes the earlier “production manifest unavailable” staging limitation.

## Immutable reference and isolated candidate

Original release: `/home/xander/flash-gordon-live/20260926-abl2`.
Original pack: `/home/xander/fg-ring-pack-abliterated`.

- Original executable SHA256:
  `e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2`.
- Original manifest file SHA256:
  `35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`.
- Ordinary candidate sealed manifest file SHA256, identical on all eight owners:
  `0b6be8d28ca31dee068a646bfabd6cf18b1ec7ce50e869453e8fcbdb8e5d0bff`.
- Candidate release: `/home/xander/flash-gordon-live/20261015-q4-foreign`.
- Candidate executable SHA256, identical on all eight owners:
  `de8e1067732baaf9bb49083fbdebc55a991c78619d4156624f6ec03ddfc2022b`.
- Same-engine resident evaluator SHA256:
  `a05d0beb5cbb4f7355808ddb945f74306a7ebe212a8b94726ada1a560fa889d2`.
- Ordinary durable pack: `/home/xander/fg-ring-pack-abliterated-q4`.

The Q8 vocabulary head and embedding are unchanged. Exact compact Q4_K uses
18.5 B/32; the existing per-shape policy retains the faster wide layout for the
small K2560/M640 and M512 matrices. Routers remain lossless. Cooked weights are
loaded directly from the sealed files: no additional resident raw dense arena.

All owners support XFS reflinks. Unchanged expert/head extents are cloned
copy-on-write, but their original bytes are still read and SHA256-verified.
The initial `.49` free space was only 12,638,507,008 bytes; ordinary duplication
would not have fit. Filesystem free-space deltas, allocated extents (which
double-count shared extents), and logical sparse sizes must not be conflated.

Actual source installations are owner-local: e.g. `.43` has rank 1 and foreign
head rank 4, not rank 0. An initial safe staging attempt refused a missing
foreign file before changing services. The repacker now emits owned records
plus the rank-4 globals needed by the coordinator/head/HC helpers, rather than
requiring unnecessary foreign per-layer files. All eight resulting sealed
manifest digests agree; native tests cover these owner roles and reflink/hash
identity. The first coordinator staging additionally retained harmless foreign
common extents; these are not made resident in ring mode.

Initial integration exposed a genuine foreign-output binding bug: model-local
views recognized persisted SoA/compact layouts, but the last-layer owner's
foreign HC loader tagged them as raw GGML. Activations remained finite through
the layer stack and then failed at foreign HC. Both loaders now share
`fg_vk_dense_saved_format`; native regression checks cover all three quant
types, both persisted layouts, Q8 cooked, and BF16 fallback. The candidate pack
bytes/seal do not change. A missing private-teacher tokenizer directory was
also repaired without touching source assets; both failed attempts restored
the original API before diagnosis.

## Ordinary resident weight ledger

Every rank owns six layers. Values below are exact aligned **owner-local weight
arena** bytes, not whole-process GPU memory or manifest reservation budgets.
The ring coordinator additionally retains 679,211,008 foreign-global bytes:
its total model weights are **11,078,316,032 bytes**. Four-way head helpers 1/2
add a 168,857,600-byte head slice each; the last layer owner also loads foreign
HC weights. Separate resident receipts record deduplicated DRM-client VRAM/GTT
and process RSS after starting each API, including runtime scratch.

| Rank / blade | Owner-local weight bytes | Common bytes | Expert bytes | Manifest residency budget |
|---|---:|---:|---:|---:|
| 0 / .42 | 10,399,105,024 | 961,921,024 | 9,437,184,000 | 11,761,860,608 |
| 1 / .43 | 10,478,870,528 | 307,683,328 | 10,171,187,200 | 11,876,392,960 |
| 2 / .44 | 9,725,636,608 | 288,452,608 | 9,437,184,000 | 11,056,050,176 |
| 3 / .45 | 9,723,674,624 | 286,490,624 | 9,437,184,000 | 11,086,430,208 |
| 4 / .46 | 10,404,847,616 | 967,663,616 | 9,437,184,000 | 11,735,261,184 |
| 5 / .47 | 9,985,818,624 | 286,490,624 | 9,699,328,000 | 11,348,574,208 |
| 6 / .48 | 9,725,636,608 | 288,452,608 | 9,437,184,000 | 11,056,050,176 |
| 7 / .49 | 10,247,962,624 | 286,490,624 | 9,961,472,000 | 11,610,718,208 |

Measured ordinary startup GPU allocations below include runtime scratch,
foreign weights and head slices; the deduplicated DRM resident counters equal
these allocation totals in this snapshot. They are **not** maximum-context
high-water marks. CPU RSS is reported separately because this is unified
memory. Runtime logs report zero cooked-at-load tensors.

| Rank | GPU allocated/resident bytes | Process RSS bytes |
|---|---:|---:|
| 0 | 11,883,655,168 | 83,439,616 |
| 1 | 11,140,792,320 | 187,457,536 |
| 2 | 10,385,514,496 | 187,351,040 |
| 3 | 10,549,493,760 | 187,904,000 |
| 4 | 10,900,504,576 | 187,486,208 |
| 5 | 10,811,637,760 | 187,940,864 |
| 6 | 10,215,993,344 | 187,158,528 |
| 7 | 11,077,713,920 | 187,772,928 |

## Qualification method

The lifecycle helper is adapted from `cuppaxanax/decode-climb`: owned numeric
PID, exact manifest argument and allowed release executable, TERM with a bounded
wait, never a process-name or forced kill. Original-release operator logs/cache
live in a separate Q4 operator directory, not inside the protected build.

Runtime QSA state lives beside its manifest. Therefore the Q8 teacher uses
`/home/xander/fg-ring-pack-abliterated-q4-teacher`, a byte-identical manifest and
links to immutable source artifacts, never links to `.state` files. Teacher
evaluation does not write into the original pack.

Quality uses the same resident `q4-eval-batch` executable and engine objects
for both packs: 20 fixed chat/code/multilingual prompts, four fixed textual
assistant prefixes each, independent reset before each of 80 frontiers. Exactly
one frontier is computed; its prediction is discarded, never fed into another
context. Numbered exclusive-created dumps contain all 248,320 finite FP32
logits. Stable full-vocabulary KL(teacher || candidate), mean ≤0.01, nearest-rank
p99 ≤0.05, top-1 ≥97%; no top-k renormalization.

The new binary also runs the five original greedy reference hashes on the Q8
teacher before candidate testing. Candidate performance uses the **unmodified**
`D:\looking-glass-labs\flash-gordon\tools\hillclimb.ps1` full battery and
`results\hillclimb\BASELINE.json`: all 30 comparable rates/latencies/wall times,
all five greedy hashes, valid cold prefills, and quality must pass. Rates must
be at least 97% of baseline; latencies at most 103%. Capture and intrusive
profile flags are cleared for performance; static replay remains enabled.

## Execution and evidence

Staging receipts: `q4-qualification/ordinary-stage/preflight.json`,
`staged.json`, and `resumed.json`. The protected overlay is preserved on `.53`,
3,711,132,288 bytes, SHA256
`5765933781d6ca366d90343f9123ecf94e68f284465ae39cf837652cfb18ad7a`.
It uses Q5 for GDN qkv/out and QSA projection and otherwise the ordinary plan.

The corrected binary's fresh Q8 control passes all five original greedy hashes.
Ordinary Q4 completes all 80 finite full-vocabulary captures, but **fails
quality**: mean KL **0.1043244150**, p99 **0.6921010355**, top-1 **83.75%**.
This is not a claim that the quantized-source kernels are inexact: scalar
dequant parity and cooked/raw source equivalence are separate from the
end-to-end difference between the published-BF16-derived overlay and the
deployed dense-swap weights. The earlier published-source sample check is not
an exhaustive deployed-pack weight-identity proof, so these results alone
cannot attribute every difference exclusively to quantization.
Corpus SHA256: `4fc3d274599c174a93de8b98e272dd6b9e5dc0c8ceb1eefba17271337c6a14d7`.

### Ordinary full harness

The complete, error-free run took **1,063.57 s** versus 930.04 s baseline.
Only **3/5** greedy hashes match (capital and spelling differ). **19/30**
performance gates pass; quality, correctness, and performance independently
reject this candidate.

| Metric | Baseline | Ordinary | Change |
|---|---:|---:|---:|
| Client decode, short / 32K / 128K, tokens/s | 26.76 / 24.44 / 22.32 | 30.645 / 27.759 / 24.479 | +14.5% / +13.6% / +9.7% |
| Engine decode, short / 32K / 128K, tokens/s | 25.80 / 23.84 / 21.77 | 28.987 / 27.643 / 24.304 | +12.4% / +15.9% / +11.6% |
| Client prefill, 4K / 32K / 128K, tokens/s | 339.91 / 357.02 / 415.58 | 369.500 / 356.874 / 349.123 | +8.7% / −0.04% / −16.0% |
| Engine prefill, 4K / 32K / 128K, tokens/s | 355.12 / 420.86 / 415.47 | 333.659 / 367.731 / 346.963 | −6.0% / −12.6% / −16.5% |
| Cold prefill TTFT, 4K / 32K / 128K, ms | 12,226.82 / 91,952.51 / 315,527.54 | 11,242.23 / 91,984.77 / 375,598.45 | −8.1% / +0.04% / +19.0% |
| Chat / cached-agent TTFT, ms | 2,096.32 / 713.04 | 1,904.34 / 681.31 | −9.2% / −4.5% |
| Concurrent aggregate visible events/s | 2.68 | 1.504 | −43.9% |

Concurrency stream wall times also regress 4.45%/4.49%. Event-rate comparisons
are additionally affected by changed generated output, so they are not a pure
GPU-kernel speed estimate. Nonetheless the unchanged harness gate fails. Small
batched microbenchmark wins do **not** establish full long-context prefill
parity; the 128K measurement is materially worse.

### Protected fallback

All eight protected packs are sealed with the identical manifest file SHA256
`3ee6bd13d3fab0747df021c0ae1a4140d0be23e757f771f794a185477bfd0e36`
at `/home/xander/fg-ring-pack-abliterated-q4-protected`. They use the same
corrected binary and the same teacher captures as ordinary.

Protected also **fails quality** on 80 complete finite frontiers: mean KL
**0.0725013513**, p99 **1.0109592110**, top-1 **81.25%**. The selective Q5
upgrade lowers average KL but does not fix the tail or top-1 gate. Neither
plan is eligible for promotion, regardless of its performance.

The complete, error-free protected run took **1,084.20 s** versus 930.04 s.
It passes **22/30** performance gates and **3/5** greedy hashes, again missing
capital and spelling. Its eight performance failures are engine prefill at
all three sizes, client 128K prefill and its TTFT, concurrent agent/aggregate
event rates, and full wall time.

| Metric | Baseline | Protected | Change |
|---|---:|---:|---:|
| Client decode, short / 32K / 128K, tokens/s | 26.76 / 24.44 / 22.32 | 29.483 / 26.675 / 24.105 | +10.2% / +9.1% / +8.0% |
| Engine decode, short / 32K / 128K, tokens/s | 25.80 / 23.84 / 21.77 | 28.034 / 26.571 / 24.109 | +8.7% / +11.5% / +10.7% |
| Client prefill, 4K / 32K / 128K, tokens/s | 339.91 / 357.02 / 415.58 | 344.903 / 352.985 / 341.625 | +1.5% / −1.1% / −17.8% |
| Engine prefill, 4K / 32K / 128K, tokens/s | 355.12 / 420.86 / 415.47 | 326.198 / 356.191 / 341.815 | −8.1% / −15.4% / −17.7% |
| Cold prefill TTFT, 4K / 32K / 128K, ms | 12,226.82 / 91,952.51 / 315,527.54 | 12,061.36 / 92,998.20 / 383,830.52 | −1.4% / +1.1% / +21.6% |
| Chat / cached-agent TTFT, ms | 2,096.32 / 713.04 | 1,882.86 / 680.73 | −10.2% / −4.5% |
| Concurrent aggregate visible events/s | 2.68 | 1.849 | −31.0% |

## Final decision and cleanup

**Reject both plans.** The original release and original pack are serving
`.42:8080`; no Q4 candidate remains active. All eight owned numeric PIDs have
the exact original executable and manifest arguments, and all eight original
executable/manifest SHA256 pairs remain unchanged. `/health` reports `ok`,
`/v1/models` lists the original model, and a real nonstreamed chat returns
`4` for the arithmetic smoke request. The smoke request needs more than 16
generation tokens because the original model emits a separate reasoning
channel first; the verified 128-token allowance completes in 32 tokens.

Both sealed candidate packs are retained on every owner, including rejected
protected. The corrected candidate release is retained for reproduction.
The private Q8 teacher view, inactive candidate QSA state, qualification-only
stage/build/upload trees, downloaded remote logit archives/dumps and prompt
frames, and the obsolete failed candidate release were removed. The `.52`
qualification test workspace/archive was removed separately. Verified `.53`
artifacts and pre-existing `.52` baseline evidence were preserved.

Final storage below is decimal GB. Allocated extents **double-count COW-shared
blocks** and therefore are not incremental disk cost. Exact byte counts and
logical apparent sizes are in `q4-qualify.json`.

| Blade | Ordinary allocated GB | Protected allocated GB | Disk available GB |
|---|---:|---:|---:|
| .42 | 14.493 | 11.379 | 125.086 |
| .43 | 12.231 | 12.263 | 55.559 |
| .44 | 10.672 | 10.705 | 58.268 |
| .45 | 9.725 | 9.758 | 16.879 |
| .46 | 10.406 | 10.438 | 67.874 |
| .47 | 9.987 | 10.020 | 49.081 |
| .48 | 9.727 | 9.759 | 51.290 |
| .49 | 10.993 | 11.027 | 12.963 |

The initial coordinator ordinary pack's harmless extra foreign extents explain
its larger allocation/logical size: **80.692 GB apparent**, versus protected's
20.871 GB. Holes are not resident GPU weights. `.42` has 125.086 GB available
after cleanup versus 129.295 GB at initial inventory, an observed net filesystem
delta of 4.209 GB, not an exclusive-extent accounting measurement.

`q4-qualify.json` preserves all 30 per-plan metric comparisons, quality
summaries, all five greedy results, per-rank resident receipts, seals, final
process ownership/storage, and capture-archive identities. Full raw captures
and source archives remain local ignored evidence, not Git blobs.

### Commands

From `D:\looking-glass-labs\fg-wt\q4-dense`, the corrected full attempt was:

```powershell
& .\tools\q4-qualify-live.ps1 `
  -SourceArchive .\results\hillclimb\q4-qualification\code-v3.tar `
  -OutputDir .\results\hillclimb\q4-qualification -Attempt r3 `
  -Release /home/xander/flash-gordon-live/20261015-q4-foreign
```

The source archive SHA256 is
`b719fc746ad29d94409c255c340724c38183e8796473968605a9d44b091db555`.
The orchestration invokes `q4-stage-pack.ps1 -AfterLease -Label
qualify-protected -OverlayName dense-protected` for fallback, using the
immutable original pack as its reflink/SHA-verified unchanged-weight source.
Capture output tags are exclusive; fresh captures require a new attempt tag.
These are historical commands, not permission to overwrite the retained
candidate directories on a rerun.

Validation completed: native cooker/owner-placement/saved-format regression
checks on all eight owners; real Vulkan model-loader tests on `.52`; twelve
Python checks (one optional native-library skip); mocked staging lease/hash/
seal checks; PowerShell parsing and `git diff --check`. The qualification-only
`.52` model-loader workspace/archive has been removed.
