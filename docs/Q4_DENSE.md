# Experimental Q4 dense overlay

**Kernel gate passed; source-aligned mixed Q4/Q5/Q8 passes 480-row quality.**
**Deployment NO-GO:** the best tested mix passes only 18/30 full performance
gates and 4/5 exact greedy references; immutable original serving is restored.
**Source-control correction:** the extracted BF16 differs from deployed Q8 in
96 residual-output matrices; 400 other dense matrices match byte-for-byte.
The old ordinary/protected overlays therefore mix quantization error with
source drift. See [`q4-ablation.md`](../results/hillclimb/q4-ablation.md).
Do not stage the historical `dense-q4.gguf` as a deployment candidate.
The aligned Q8 control is 100% top-1 with mean KL 3.96e-14. The best tested
mixed plan (`mix-04`) keeps QSA/head/HC-up Q8, GDN-out/HC-down Q5, and the
remaining eligible dense families Q4; joint mean KL .003129, p99 .031049,
top-1 98.9583%. Its owner weights save 1,279,074,304 bytes versus Q8.
The fresh same-binary full Q8 control also misses the historical 128K prefill
baseline (engine −14.50%); real-N=128 paired dense kernels all favor Q4.
This does not waive the unchanged deployment gates or explain the full
historical cross-run deficit. See the ablation report for controls and limits.
The selected exact compact/wide implementation beats paired cooked Q8 on all
nine non-head GEMV shapes in three warmed runs on **both `.51` and `.52`**,
and all 63 prefill comparisons per blade at N=8,16,32,64,128,256,512.
The protected Q5 variant also passes its nine GEMV comparisons on both blades;
protected prefill was not requalified. The Q6 head still loses; retain Q8.
See [`q4-compact.md`](../results/hillclimb/q4-compact.md) for the follow-up,
single-copy sealed packs and historical staging estimates; the measured
eight-owner resident ledger is now in `q4-ablation.md`.
Historical wide-layout results remain in
[`q4-kernels.md`](../results/hillclimb/q4-kernels.md) and its JSON evidence.
The older ~93 GB/s cooked Q4/Q5 and ~65 GB/s Q6 results describe the superseded
interleaved kernels, not the new SoA path. These are synthetic GPU timestamp
microbenchmarks, not wall-clock serving throughput or a model-quality result.

## Contents and compatibility

The existing Q8 pack path remains supported. New dense GGML types are Q4_K (12),
Q5_K (13), Q6_K (14), and IQ4_NL (20). K-quants require row width divisible by
256; IQ4_NL requires 32. The ordinary variant uses Q4_K, IQ4_NL for K=320/640,
and historically offered Q6_K for the complete output vocabulary. Deployment
plans keep the original Q8 head. The protected variant promotes
GDN qkv/output and QSA q/k/v/output to Q5_K. No weights are below Q4.

Routers narrow from F32 to BF16 only if every value is bit-exact on expansion;
the repacker independently proves this against the original pack. Otherwise
retain F32 (`--keep-f32-routers`). Other F32 gates, norms, and injections,
BF16 indexers, token embedding, all routed experts, and ngram storage retain
their original pack representation. Historical GGUF overlays include a freshly
quantized embedding that the repacker does not install; aligned
`--deployed-dense-only` overlays exclude embedding and head entirely.

Single- and multiple-token shader parity is checked against independent scalar
oracles. The new SoA path stages dequantized FP32 weights once per LDS tile
for batched dispatch, with token-adaptive 8/16/32-column tiles. It is **not
packed-FP16 MMQ**: no potentially lossy FP16 weight narrowing is performed.
No native int8 dot is used.
`FG_DENSE_K_COOKED=1` optionally cooks eligible Q4/Q5 matrices into the existing
8-row interleave, without changing on-disk GGUF bytes. The sliced vocabulary
head remains raw. Default Q8 cooking is unchanged.

### Exact persisted SoA path (recommended)

`q4-repack --cook-dense --keep-q8-head` writes the selected GPU layout directly
into the pack, with sealed tensor hashes and layout tags. The loader creates
views into its verified weight arena: **no raw dense GPU copy, second dense
allocation, or load-time recooking**. This remains true with legacy
`FG_DENSE_SOA=1` accidentally set. Old executables reject these new layouts.
Embedding and output weights are excluded.

Eight-row tiles retain contiguous quant streams and separate metadata.
Compact Q4_K uses **18.5 bytes/32 weights (4.625 bpw)**; Q5_K uses **22.5
(5.625 bpw)**. Original half d/dmin bits are shared by each 256-value block,
with byte scale/min codes per 32-value group. Products are reconstructed
exactly in FP32. IQ4_NL keeps its 18-byte layout and improved small-shape
kernel. K=2560/M=640 or 512 retains wide Q4_K 24-byte/Q5_K 28-byte metadata,
which wins those GEMVs. Selection is per stored matrix, not duplicated storage.
Compact prefill remains faster than Q8, but can be slower than wide prefill.

Legacy `FG_DENSE_SOA=1` on **raw** packs still cooks into additional buffers;
it is an experimental compatibility path, not the deployment solution.
Use the persisted path and leave both runtime cooking flags unset.
Loader tests cover all three types, sealed roundtrips, every small-case
row/token, ordinary and replicated loads, exact arena accounting/caps, and
hash corruption. Do not extrapolate the historical 4.4–4.9 GB/token ledger.

The measured ordinary plan uses Q4_K for every 256-aligned projection and
IQ4_NL for K=320/640. It is not permission to downgrade protected projections
without the existing quality gates. `q4-repack --keep-q8-head` copies the
original output weight rather than installing the overlay's slower Q6 head.

## Source and quantization

Source repository: `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`,
revision `06a318de4011598a142ee54bcfec47c6921c958e`.
`tools/q4_source.py` range-reads GGUF headers and only selected nonexpert tensor
payloads from the eight BF16 shards. It does not download complete shards.
It excludes routed experts, the huge ngram table, vision, and MTP tensors.
GGUF-native names and layouts are retained.

`tools/q4_check_source.py` checks shapes and 60 sampled Q8 blocks from 20 dense
families against the published abliterated UD-Q4_K_XL source. Its bounds include
Q8 scale rounding. **This is not a deployed-pack identity proof.** Full-ring
qualification against the actual deployed pack is recorded in
`results/hillclimb/q4-qualify.md`: ordinary and protected overlays both fail
the numerical quality gates. Do not deploy either on the strength of kernel
parity or published-source sampling.

Historical published-source reproduction only (not deployment qualification):

```sh
cd /home/xander/hc-q4
python3 q4_check_source.py --bf16 dense-bf16.gguf --output source-parity.json
python3 q4_imatrix.py
python3 q4_quantize.py --source dense-bf16.gguf --output dense-q4.gguf \
  --ggml-library llama-build/bin/libggml-base.so --imatrix dense-imatrix.gguf
python3 q4_quantize.py --source dense-bf16.gguf --output dense-protected.gguf \
  --ggml-library llama-build/bin/libggml-base.so --protected --imatrix dense-imatrix.gguf
sha256sum dense-*.gguf
```

Quantization calls llama.cpp's exact `ggml_quantize_chunk` implementation from
revision `58367713a6935c0810103378144008df32e3d5db`, in bounded row chunks,
normally directly from BF16/F32. The explicitly aligned ablation instead uses
`--q8-overrides DEPLOYED.gguf --override-map COMPARISON.json` for the 96 differing
residuals: their deployed Q8 values are dequantized exactly to FP32, not claimed
to be recovered BF16. Complete payload hashes and shapes are checked before
using an override. BF16 conversion also uses ggml's
native row conversion. Quantization defaults to eight CPU threads with at most
16 row chunks in flight; tests prove byte identity against `--threads 1`.
`q4_imatrix.py` range-fetches matching dense sums/counts from Unsloth's
`imatrix_unsloth.gguf_file`, revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`, without downloading its expert
payloads. Quantization normalizes sums by counts and passes the full input-width
importance vector to ggml. Missing entries (often the head) retain ordinary
quantization and are listed in the output inventory. This calibration is from
the matching architecture, **not the abliterated checkpoint**; quality remains
unproven until the actual baseline comparison.
Each output's JSON inventory records type, bytes, and tensor SHA256.

Before quantizing a deployment candidate, extract and hash-verify actual
deployed common Q8 tensors with `tools/q4-extract-pack`, generate a
`--q8-control`, and run `q4_compare_source.py --require-identical`.
Unexplained differences fail this provenance gate. After explicit alignment,
all 496 complete Q8 payloads match; numerical capture is still required.
`--deployed-dense-only` emits only the reference's nonhead Q8 matrix set and
leaves original routers/norms/embedding/head untouched in the repack.
`--q5-dense`/`--q6-dense` make 256-aligned matrices Q5_K/Q6_K; other rows
remain IQ4_NL in those raw artifacts. `q4_mix.py` uses an explicit family/tensor
Q4/Q5/Q6/Q8 policy,
copies payload bytes without requantizing, and falls back to original Q8
where a requested Q5_K/Q6_K cannot represent K=320/640. Its inventory records
actual types, source hashes, and changed families.
`--changed-families-only` omits untouched families from the transfer overlay,
but retains Q8 fallback tensors inside any changed family. Pass the inventory's
`changed_families` as native `--family` selectors.
Optional `--routers GGUF` imports **only** BF16 `.ffn_gate_inp.weight` entries,
never residual projections from that file. The native repacker must independently
prove every imported router value against the deployed F32 record; do not use
`--keep-original-routers` for that explicitly requested optimization. Family
isolation captures retain original F32 routers; any final narrowed-router mix
must pass a fresh joint capture.

## Repacking — orchestrator lease required

The initial kernel-only experiment did not contact production. Full-ring
qualification is recorded separately in `results/hillclimb/q4-qualify.md`.
The `.53` artifacts are dense overlays, **not standalone serving packs**.
Use a new directory; never overwrite the baseline.
Keep the original pack: external tokenizer/ngram/host-cache files are symlinked
to its absolute paths.

The following commands are historical layout-staging examples only: use a
provenance-aligned, numerically qualified overlay from the ablation report,
not the old default artifact. After `.42`–`.49` are released, they stage all eight
owner-local packs without starting/stopping workers or changing services:

```powershell
Set-Location D:\looking-glass-labs\fg-wt\q4-dense
git archive --format=tar --output=.\q4-stage.tar HEAD
& .\tools\q4-stage-pack.ps1 -SourceArchive .\q4-stage.tar `
  -OutputDir .\results\q4-stage-compact -Label q4-compact -AfterLease
```

**Run only while holding the production lease.** Without `-AfterLease`, it refuses before any network
access. It retrieves the preserved `.53` overlay, checks transfer hashes,
discovers the baseline manifest, builds on every owner, validates identical
source manifests and six layers/rank, derives ranks from manifest endpoints,
cooks/seals each pack, and requires identical candidate manifest hashes.
`preflight.json` and `staged.json` retain per-rank byte ledgers. Offline tests
mock every network call and cover mismatched hashes and non-IP-order ranks.
Staging does not constitute quality qualification or promotion.
The remote staging basename is always `stage-LABEL/dense-q4.gguf`, regardless
of `-OverlayName`. Once cooking/sealing completes and no builder uses it, remove
that staging GGUF before allocating candidate QSA state; `.45`/`.49` have
limited disk. Never delete the preserved `.53` source artifacts or original
pack. A failed 480-row capture caused by stage-file disk pressure is discarded,
not silently treated as a smaller quality sample.

For manual single-owner staging only, substitute its manifest-derived rank:

```sh
cd /home/xander/hc-q4/code
make -j8 flash-gordon tools/q4-repack shaders
old=/home/xander/fg-ring-pack-abliterated
# Set these from the actual manifest/inventory:
manifest="$old/manifest.fgm"
rank=0
./tools/q4-repack "$manifest" "$old" /home/xander/hc-q4/dense-q4.gguf \
  /home/xander/hc-q4/pack-q4 "$rank" --keep-q8-head --cook-dense --reflink
# Read-only planning: no output created; does not verify source payloads/routers.
./tools/q4-repack "$manifest" "$old" /home/xander/hc-q4/dense-q4.gguf \
  /home/xander/hc-q4/unused --keep-q8-head --cook-dense --plan
```

The manifest basename above is an example, not a claim about the deployment.
The repacker emits a new sealed `manifest.fgm`.
It verifies hashes of copied originals and computes replacement hashes.
Per-owner mode includes owned layer tensors and experts. The coordinator and
four-way head/HC helpers additionally retain the required rank-4 global tensors
in a sparse foreign rank file; other foreign per-layer files are unnecessary.
This also accepts the actual original owner-local installations, which do not
contain every rank file.
Other owners' expert extents are intentionally sparse; this is an owner-local
installation, not a complete portable pack. Run it on **every** owner and require
identical new manifest SHA256 across owners. To make a complete portable pack
from a complete original pack, omit the rank argument.

`--reflink` clones unchanged, aligned extents on supporting filesystems while
still reading and SHA256-verifying every copied tensor. Unsupported ranges fall
back to ordinary copying: check free space first. Shared extents remain
copy-on-write; neither original tensor bytes nor their manifest are changed.
`du` counts shared extents in both directories, so use filesystem free-space
deltas as well as allocated/apparent sizes. Mutable `.state` artifacts are never
linked into the new pack.

Do not promote the new manifest until all gates below pass. Runtime QSA state
is written beside its manifest, so even the Q8 teacher must use a private pack
view: byte-identical manifest plus links to immutable original artifacts,
excluding state files. `q4-qualify-ready.sh` constructs that view.

## Full-vocabulary quality capture

`FG_EVAL_LOGITS_DIR=/home/xander/flash-gordon-live/RELEASE/operator/logits-TAG`
must be set when starting the **output-owner worker** for resident capture.
`make tools/q4-eval-batch` links the same engine objects as `flash-gordon`.
`q4-capture.ps1 -ResidentBatch -CaptureTag TAG` uploads a length-framed exact
UTF-8 corpus and evaluates every frontier in one resident runtime
(80 default contexts, or the explicitly supplied corpus up to 1024 rows),
resetting before each frontier. Its discarded prediction never enters the next
context. Numbered dumps use exclusive creation; import requires every finite,
complete vocabulary row and refuses stale destinations.

The lifecycle wrapper's `quality` mode sets the directory on the output owner.
Its `plain` mode explicitly clears capture/profile flags before performance
measurement. `q4-qualify-live.ps1` combines teacher/candidate capture, the full
unmodified main-worktree harness, all comparable baseline metric gates, and a
`finally` rollback to the immutable original release unless promotion passes.
Use a fresh `-Attempt` after an interrupted capture; do not reuse dump files.

For the slower, one-process-per-context compatibility flow,
`FG_EVAL_LOGITS=/home/xander/hc-q4/eval-latest.fgl` must be set when starting
the **output-owner worker**, not merely the coordinator. Start baseline workers
with this branch's executable and the private Q8 teacher manifest, preserving their
normal rank/listen/peer arguments. Then repeat with the candidate manifest.
The dump is taken by the ordinary greedy output path after GPU completion;
it is not supported for sampled/penalized output or split-head greedy decode.
`eval --generate 0` requests the ordinary frontier output path.

From the worktree in PowerShell 7, with no concurrent eval/generation traffic:

```powershell
& .\tools\q4-capture.ps1 `
  -BuildDir /home/xander/hc-q4/code `
  -Manifest /home/xander/fg-ring-pack-abliterated-q4-teacher/manifest.fgm `
  -OutputDir .\results\q4-teacher

# Restart candidate workers with its manifest, same dump environment and flags.
& .\tools\q4-capture.ps1 `
  -BuildDir /home/xander/hc-q4/code `
  -Manifest /home/xander/hc-q4/pack-q4/manifest.fgm `
  -OutputDir .\results\q4-candidate

python .\tools\q4_quality.py --teacher .\results\q4-teacher `
  --candidate .\results\q4-candidate --output .\results\q4-quality.json
```

Adjust `-Coordinator` and `-OutputOwner` if placement differs; defaults are
`.42` and `.46` for the current eight-rank layout. Adjust the actual manifest
basename in both examples. The capture helper deletes stale worker dumps,
refuses existing local captures, runs coordinator eval, and retrieves each
fresh output through the fleet runner. It does not start/stop services.
The runner wrapper stages only under `hc-q4` if the shared runner uses files,
and leaves the shared runner and its credential file unchanged.

The fixed set has 20 chat/code/multilingual prompts and four explicit textual
assistant prefixes each, yielding 80 teacher-forced frontier rows. Neither model
feeds its own generated answer back. Dumps contain magic `FGLOG01\n`, little-
endian uint32 vocabulary size, then all 248320 little-endian F32 logits.
Comparison rejects incomplete/nonfinite dumps and uses stable full-vocabulary
KL(teacher || candidate), not renormalized top-k. Gates: mean KL ≤0.01 nat,
p99 ≤0.05, top-1 agreement ≥97%. These 80 rows are an initial screening set,
not long-answer, router-top10, or long-context qualification.

### Aligned-source ablation and prefill profiling

`q4_make_contexts.py` freezes complete UTF-8 prefixes from original-Q8 teacher
continuations. The current ablation uses 480 distinct frontiers, with the same
frozen text and engine on both sides; source generation token positions are
informational because both captures reencode the text.
Comparison requires each capture's `corpus.json` to match the exact requested
UTF-8 text, not merely its row IDs; a verified subset of a larger teacher corpus
is allowed. The supplied imatrix maps 494/496 selected matrices by name and width;
the two global `output_hc_*` matrices have no calibration entries.
`q4-ablation-capture.ps1` rejects the original manifest as a candidate, stops
only manifest-scoped PIDs, removes only stopped private QSA state, and attempts
original-service restoration even after capture, stop, or cleanup failure.
`q4-ablation-sweep.ps1` refuses family attribution until the aligned Q8 control
passes near-zero KL and 100% top-1.

`q4-ablation-profile.ps1` uses `FG_PREFILL_PROFILE`, **not decode profiling**.
It warms at 4K, records all eight log byte boundaries, then captures only the
matched 128K request. `PREFILL_ROUTE` logs exact pair counts, occupied experts,
tile counts and width; padded tile occupancy is pairs/(tiles×width).
`q4_profile_report.py` rejects unequal paired workloads or nonfinite GPU times.
It also rejects different serialized request-body SHA256s. `-WallOnly` repeats
the identical requests with profiling disabled, to measure instrumentation cost.
Route statistics cover grouped expert schedules; GPU queries cover owner-layer windows
on workers and one coordinator chunk, **not a whole-model GPU total**.
Overlapping worker RPC durations cannot be added into wall time.
Profiling is intrusive and never substitutes for the unmodified performance
harness. Neither this flow nor mixed packs retain a Q8 prefill-only weight copy.
`q4-compare-packs` compares sealed tensor order/type/shape/layout/owner/bytes and
payload hashes, holding storage policy fixed in source-only experiments.
`q4-ablation-harness.ps1` accepts only passing ≥400-row full-vocabulary quality,
runs the unmodified full canonical harness, and promotes only when all gates pass;
failure paths attempt original restoration and record end-state readiness.

## Validation and remaining gates

```sh
bash tools/q4-safe-tests.sh make -j8 test test-vulkan
python3 -m unittest tools/test_q4_tools.py
python3 -m unittest tools/test_q4_qualify.py
FG_Q4_GGML_LIBRARY=/home/xander/hc-q4/llama-build/bin/libggml-base.so \
  python3 -m unittest tools/test_q4_tools.py
DS4_REMOTE_TEST_FILTER=router_bf16_exact ./tests/test_fg_vk
FG_BENCH_DENSE_QUANT=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk
FG_BENCH_DENSE_COOKED=1 FG_BENCH_DENSE_QUANT=1 \
  DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk
FG_BENCH_DENSE_SOA=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
  FG_BENCH_DENSE_QUANT=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk
```

PowerShell lifecycle/promotion/failure mocks (no fleet/network access):
`pwsh -NoProfile -File .\tools\test_q4_ablation.ps1`.

Each benchmark prints subgroup size (64 on `.52`), logical weight bytes,
GPU microseconds, decimal weight GB/s, same-shape Q8 latency, break-even, and
scalar parity. There are 20 timed repetitions per candidate and Q8, alternating
in the same profiled command batch and reversing order each pair. Benchmarks
check 32 evenly sampled rows for every token; smaller correctness cases check
every row/token. The Q8 baseline uses an independently initialized, same-shaped
payload, not numerically identical weights. Separate experiment runs may still
have different clocks/cache behavior; compare each candidate to its own paired
baseline. Cooked GB/s counts the expanded SoA bytes, not the original compressed
payload; both rates are reported in the new results. No privileged
hardware/power changes are made.

Observed regression limits on `.52`: the full make targets hit existing
`test_hc_down_split` missing-topology link dependencies and a
`test_prefill_dispatch` GCC `-Werror=nonnull` diagnostic. Standalone core passes.
The native Vulkan suite passes Q8 dense/cooked/prefill, new quantized dense,
router, embedding, HC, ngram, and PLE oracles before crashing in
`qsa_record_commit`. That crash also reproduces using the original baseline
Vulkan, loader, output, and test sources (exit 139). Therefore **no full-suite
pass is claimed**. The current 17 Python/native-quantizer tests pass on `.53`;
local Windows runs skip only the optional ggml-library case. Expert-prefill
Vulkan parity passes with route profiling both disabled and enabled.

**Remaining gates/limitations:** the compact ordinary latency gate passes on
both 40-CU `.51` and 24-CU `.52`; Q5 GEMV passes, but the Q6 head stays Q8.
Compact reaches ~308 GB/s cooked and ~300 GB/s source on K=6144/M=2560
on `.51`; most shapes remain below the 300 GB/s goal. The original pre-lease
manifest-access limitation is superseded: `q4-qualify.md` and `q4-ablation.md`
record real sealed eight-owner packs, native resident ledgers, live captures and
end states. Native weight budgets are not live scratch/KV/driver measurements.
Packed-FP16 MMQ remains future work. Deployment still requires
real-pack source/identity and exact-router proof;
complete ring startup and logits capture; KLD/top-1, router-top10, existing
reference/Vulkan/prefill/decode/long-context gates. Nothing here authorizes
deployment or claims the projected 4.4–4.9 GB/token is achieved serving speed.
