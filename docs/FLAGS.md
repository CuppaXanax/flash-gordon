# Flash Gordon environment flags

Every `FG_*` environment variable read by the runtime, the packer, or the test
binaries. Unknown `FG_*` variables are ignored.

Status values:

- **validated default** - part of the qualified configuration; the reference
  start configuration sets it and the numbers are measured with it on.
- **opt-out** - on by default; setting it to `0` restores the alternate path.
  Kept as a control so a regression can be bisected on the same binary.
- **A/B toggle** - same-binary switch used to measure a variant against the
  default path.
- **variant (opt-in)** - off by default; only enable for a specific experiment.
- **profiling** - diagnostics only; no numeric effect when off (some traces
  force a GPU synchronization and are not throughput-representative).
- **test/bench** - read by test binaries or bench tools only; the serving
  binary ignores them.

## Execution and routing

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_PREFILL_RING` | unset (off); reference config sets `1` on rank 0 | Layer-ring prefill. The ring executor activates only when both ring halves are non-zero. | validated default |
| `FG_DECODE_RING` | unset (decode half enabled); reference config sets `1` | Layer-ring decode. `0` restores the legacy expert-parallel decode replay. | validated default |
| `FG_WORKER_OWNER` | unset (off); reference worker config sets `1` | Worker-side owner executor. | validated default |
| `FG_DECODE_CHAIN` | unset (on) | Chained ring blocks: one submission and one fence per block owner. `0` falls back to the per-layer owner machine. | opt-out |
| `FG_DECODE_DIRECT_OUTPUT` | unset (on) | Final block owner hands the hidden state straight to the output owner. `0` restores the rank-0 relay. Must match on every rank. | opt-out |
| `FG_PREFIX_CONT` | unset (on) | Prefix continuation on exact frontier hits. `0` forces cold resets. | opt-out |
| `FG_PACK_EMBED_RANK` | rank 7 | Packer placement override for `token_embd.weight` (0-7). | packer option |
| `FG_MTP_DRAFT_ECHO` | unset (off) | Draft/verify harness for the MTP scaffolding; no trained MTP weights exist, so this echoes the target token. Decode stays bit-identical. | test harness |

## Correctness / performance controls (validated opt-outs)

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_DECODE_PIPELINE` | unset (on) | Async decode submission pipeline. `0` disables it. | opt-out |
| `FG_DECODE_STATIC` | unset (on) | Static recorded replay for token-invariant layer runs. `0` disables it. Forced off while profiling or numeric tracing is active. | opt-out |
| `FG_QSA_SELECT_GPU` | unset (on) | Device-side QSA selection resolve with an exact-miss host fallback. `0` forces the old host selection path. | opt-out |
| `FG_QSA_TOPK_V2` | unset (on) | Wide chunked top-k control. `0` selects the legacy top-k kernel. | opt-out |
| `FG_DECODE_EXPERT_LEGACY` | unset (off) | `1` forces the legacy five-dispatch batch-1 expert path instead of the fused pair. | opt-out (inverted) |
| `FG_VK_HOLD_STATIC` | unset (hold on) | Holds static submissions behind the pending fence queue. `0` submits them immediately. | opt-out |
| `FG_FABRIC_DIRECT_SEND` | unset (off) | Direct fabric send path. | variant (opt-in) |
| `FG_FABRIC_DIRECT_RECV` | unset (off) | Direct fabric receive path. | variant (opt-in) |
| `FG_EXPERT_BATCH_SEND` | unset (off) | Batches expert work messages on the remote fan-out. | variant (opt-in) |

## Output split (opt-in 4-way head handoff)

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_OUTPUT_SPLIT` | `0` (off) | `1`/`2` select the 2-way head split; `4` selects the 4-way split. Requires a pack which supports the split layout. | variant (opt-in) |
| `FG_OUTPUT_SPLIT_HIDDEN` | unset (suppress on) | Hidden-state suppression in the split path: when the output owner already has the sampling config, the 40 KiB hidden message is dropped (bit-identical result). `0` restores the hidden send for A/B. | A/B toggle |
| `FG_OUTPUT_SPLIT_TIMEOUT_MS` | `4000` | Liveness bound for waiting on split partials; clamped to 100-60000. | variant (opt-in) |
| `FG_OUTPUT_SPLIT_TRACE` | unset (off) | Split handoff state trace. | profiling |

## Fabric and numeric traces

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_FRAME_TRACE` | unset (off) | Per-frame send/receive traces (fabric, owners, n-gram, QSA). | profiling |
| `FG_FABRIC_PROFILE` | unset (off) | JSON fabric service records to stderr. | profiling |
| `FG_DECODE_PROFILE` | unset (off) | Per-token decode phase timings and Vulkan profile dump. | profiling |
| `FG_PREFILL_PROFILE` | unset (off) | Per-layer prefill kernel profile dump. | profiling |
| `FG_PREFILL_PROFILE_CHUNK` | unset (profile all) | Restricts `FG_PREFILL_PROFILE` capture to one microbatch index. | profiling |
| `FG_OUTPUT_TRACE` | unset (off) | Per-token greedy output trace on the output owner. | profiling |
| `FG_NUMERICS_TRACE` | unset (off) | Hyper-state digests per layer/phase. Forces static replay off. | profiling |
| `FG_GDN_DIAG` | unset (off) | GDN state diagnostics. | profiling |
| `FG_DECODE_MS` | unset (off) | Per-token decode timings. | profiling |
| `FG_TRACE_ROUTES` | unset (off) | Router/expert route traces. | profiling |
| `FG_PREFIX_TRACE` | unset (off) | Prefix reuse diagnostics (counts, mismatch positions and reasons only). | profiling |
| `FG_GENERATE_TRACE` | unset (off) | Generation loop breakdown (interrupt/tokenizer/callback/render/decode). | profiling |
| `FG_DECODE_RING_TRACE` | unset (off) | Ring decode issue/stage/done traces. | profiling |
| `FG_RING_TRACE` | unset (off) | Ring prefill issue/stage/done traces. | profiling |
| `FG_PROFILE_TOKEN` | unset (off) | Selects a single token index; enables the detailed per-token captures below. | profiling |
| `FG_NGRAM_VERIFY` | unset (off) | Verified n-gram row comparison for the `FG_PROFILE_TOKEN` token. | profiling |
| `FG_QSA_WARM_TRACE` | unset (off) | QSA cache warming trace. | profiling |
| `FG_NGRAM_LOCALITY_TRACE` | unset (off) | n-gram locality summary. | profiling |
| `FG_QSA_LOCALITY_TRACE` | unset (off) | QSA locality report: `summary` at reset/close, `token` adds one keyed digest per QSA layer/token. Allocates nothing when off. | profiling |
| `FG_QSA_LOCALITY_MIB` | `16,32,64,128,256,512,1024` | Comma-list of curve budgets used by the locality report. | profiling |
| `FG_BLOCK_BENCH` | unset (off) | Offline block-service bench; exits after printing per-layer wall times. | test/bench |
| `FG_SHADER_DIR` | `vulkan` | Directory containing the SPIR-V files. | test/ops |

## Test and bench flags (test binaries only)

| Flag | Effect |
|---|---|
| `FG_BENCH_PREFILL` | Prefill kernel benchmark output. |
| `FG_BENCH_PREFILL_SHAPES` | Prefill shape sweep. |
| `FG_BENCH_Q8_SUBGROUP` | Q8_0 subgroup vs generic dense benchmark. |
| `FG_BENCH_Q8_COOKED` | Cooked Q8_0 dense benchmark. |
| `FG_BENCH_HC_INJECT` | HC-inject partial vs two-pass benchmark. |
| `FG_BENCH_GDN_ALGEBRAIC` | GDN algebraic vs classic recurrence benchmark. |
| `FG_BENCH_Q5_1_COOKED` | Cooked Q5_1 down benchmark. |
| `FG_BENCH_KQUANT_COOKED` | Cooked K-quant benchmark. |
| `FG_BENCH_EXPERT_DECODE` | Fused expert decode benchmark. |
| `FG_EXPERT_PARITY_DUMP` | Dumps expert parity reference data. |
| `FG_RMS_HASH` | Prints a group-norm output hash for bit-exactness checks. |

Test selection uses `DS4_REMOTE_TEST_FILTER=<name>` (not an `FG_*` flag).

## Removed flags

These no longer exist in the code; setting them has no effect.

| Flag | Why it was removed | Re-addable? |
|---|---|---|
| `FG_FABRIC_HOP_BF16` | Parked bf16 hop codec for the hyper-state and hidden handoff messages. Precision trades are policy-excluded, so the encoder, wire tags and tests were deleted. | Yes - as a new tagged wire encoding if the no-precision-trade policy changes. |
| `FG_DENSE_R8_WAVE_SPLIT` | Gated rows8 dense variant; lost its fleet A/B (regression). | No; the validated rows8 kernel is unconditional. |
| `FG_DENSE_R8_PAIR` | Gated 32-block rows8 pair variant; lost its fleet A/B (regression). | No; the validated rows8 kernel is unconditional. |
