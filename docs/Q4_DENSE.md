# Experimental Q4 dense overlay

**Kernel gate passed for ordinary Q4; NO-GO for deployment pending quality.**
The exact SoA implementation beats paired cooked Q8 on all nine non-head
GEMV shapes in three runs on `.51`, and all 63 prefill comparisons at
N=8,16,32,64,128,256,512. The Q6 vocabulary head still loses; retain Q8 there.
The protected Q5 plan is not a universal win. See
[`q4-kernels.md`](../results/hillclimb/q4-kernels.md) and its JSON evidence.
The older ~93 GB/s cooked Q4/Q5 and ~65 GB/s Q6 results describe the superseded
interleaved kernels, not the new SoA path. These are synthetic GPU timestamp
microbenchmarks, not wall-clock serving throughput or a model-quality result.

## Contents and compatibility

The existing Q8 pack path remains supported. New dense GGML types are Q4_K (12),
Q5_K (13), Q6_K (14), and IQ4_NL (20). K-quants require row width divisible by
256; IQ4_NL requires 32. The ordinary variant uses Q4_K, IQ4_NL for K=320/640,
and Q6_K for the complete output vocabulary. The protected variant promotes
GDN qkv/output and QSA q/k/v/output to Q5_K. No weights are below Q4.

Routers narrow from F32 to BF16 only if every value is bit-exact on expansion;
the repacker independently proves this against the original pack. Otherwise
retain F32 (`--keep-f32-routers`). Other F32 gates, norms, and injections,
BF16 indexers, token embedding, all routed experts, and ngram storage retain
their original pack representation. The GGUF overlay includes a freshly
quantized token embedding, but the repacker intentionally does not install it.

Single- and multiple-token shader parity is checked against independent scalar
oracles. The new SoA path stages dequantized FP32 weights once per LDS tile
for batched dispatch, with token-adaptive 8/16/32-column tiles. It is **not
packed-FP16 MMQ**: no potentially lossy FP16 weight narrowing is performed.
No native int8 dot is used.
`FG_DENSE_K_COOKED=1` optionally cooks eligible Q4/Q5 matrices into the existing
8-row interleave, without changing on-disk GGUF bytes. The sliced vocabulary
head remains raw. Default Q8 cooking is unchanged.

### Exact SoA runtime path

`FG_DENSE_SOA=1` selects the new layout for eligible common Q4_K, Q5_K, and
IQ4_NL matrices; leave the older cooking flag unset. Embedding and output
weights are excluded. Eight-row tiles store metadata separately from contiguous
row-local quant streams. Q4_K expands to 24 bytes/32 weights, Q5_K to 28, and
IQ4_NL retains 18. K-quant group scale/minimum products are exact FP32; IQ4
keeps the original half-scale bits. No source dequantized value changes.

This is load-time cooking into **additional** Vulkan buffers, not an on-disk
SoA pack. The verified raw arenas remain allocated; `fg_model_weight_bytes`
includes the extra buffers, and exceeding the manifest's persistent cap is
refused. Do not extrapolate the historical 4.4–4.9 GB/token ledger to this
expanded format or its resident memory. Loader tests cover all three types,
untouched embedding/head, cooked bytes/accounting, and cap refusal.

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
Q8 scale rounding. **This is not a deployed-pack identity proof.** The current
abliterated pack still needs manifest/provenance and numerical qualification.

On the artifact blade:

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
directly from BF16/F32, not dequantized Q8. BF16 conversion also uses ggml's
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

## Repacking — orchestrator lease required

No production blade was contacted and no complete expert-bearing pack was
created by this experiment. The `.53` artifacts are dense overlays, **not
standalone serving packs**. Use a new directory; never overwrite the baseline.
Keep the original pack: external tokenizer/ngram/host-cache files are symlinked
to its absolute paths.

After obtaining a fleet lease, transfer this branch/build and the chosen dense
GGUF to every owner. Stop candidate workers before repacking. On each owner,
substitute its actual rank (derive ranks from the manifest, not IP arithmetic):

```sh
cd /home/xander/hc-q4/code
make -j8 flash-gordon tools/q4-repack shaders
old=/home/xander/fg-ring-pack-abliterated
# Set these from the actual manifest/inventory:
manifest="$old/manifest.fgm"
rank=0
./tools/q4-repack "$manifest" "$old" /home/xander/hc-q4/dense-q4.gguf \
  /home/xander/hc-q4/pack-q4 "$rank" --keep-q8-head
```

The manifest basename above is an example, not a claim about the deployment.
The repacker emits a new sealed `manifest.fgm`.
It verifies hashes of copied originals and computes replacement hashes.
Per-owner mode includes local experts plus common tensors in foreign sparse
rank files (needed for replicated common weights and foreign output slices).
Other owners' expert extents are intentionally sparse; this is an owner-local
installation, not a complete portable pack. Run it on **every** owner and require
identical new manifest SHA256 across owners. To make a complete portable pack
from a complete original pack, omit the rank argument.

Do not point production at the new manifest until all gates below pass.
Repacking against the real pack, per-owner sparse behavior, and ring startup
remain untested without a lease.

## Full-vocabulary quality capture

`FG_EVAL_LOGITS=/home/xander/hc-q4/eval-latest.fgl` must be set when starting
the **output-owner worker**, not merely the coordinator. Start baseline workers
with this branch's executable and the original manifest, preserving their
normal rank/listen/peer arguments. Then repeat with the candidate manifest.
The dump is taken by the ordinary greedy output path after GPU completion;
it is not supported for sampled/penalized output or split-head greedy decode.
`eval --generate 0` requests the ordinary frontier output path.

From the worktree in PowerShell 7, with no concurrent eval/generation traffic:

```powershell
& .\tools\q4-capture.ps1 `
  -BuildDir /home/xander/hc-q4/code `
  -Manifest /home/xander/fg-ring-pack-abliterated/manifest.fgm `
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

## Validation and remaining gates

```sh
bash tools/q4-safe-tests.sh make -j8 test test-vulkan
python3 -m unittest tools/test_q4_tools.py
FG_Q4_GGML_LIBRARY=/home/xander/hc-q4/llama-build/bin/libggml-base.so \
  python3 -m unittest tools/test_q4_tools.py
DS4_REMOTE_TEST_FILTER=router_bf16_exact ./tests/test_fg_vk
FG_BENCH_DENSE_QUANT=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk
FG_BENCH_DENSE_COOKED=1 FG_BENCH_DENSE_QUANT=1 \
  DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk
FG_BENCH_DENSE_SOA=1 FG_BENCH_DENSE_Q4_ONLY=1 FG_BENCH_DENSE_SKIP_HEAD=1 \
  FG_BENCH_DENSE_QUANT=1 DS4_REMOTE_TEST_FILTER=dense_quant_bench ./tests/test_fg_vk
```

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
pass is claimed**. All eight Python/native-quantizer tests pass on `.53`.

**Remaining gates/limitations:** the ordinary kernel latency gate now passes
on `.51`, not the protected Q5/head variants or a final 24-CU validation.
The ≥300 GB/s goal is reached on only two large shapes using expanded cooked
bytes; no Q4 shape reaches it using original compressed bytes.
Packed-FP16 MMQ remains future work. Deployment still requires
real-pack source/identity and exact-router proof;
complete ring startup and logits capture; KLD/top-1, router-top10, existing
reference/Vulkan/prefill/decode/long-context gates. Nothing here authorizes
deployment or claims the projected 4.4–4.9 GB/token is achieved serving speed.
