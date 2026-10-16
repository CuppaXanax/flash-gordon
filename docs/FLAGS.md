# Flash Gordon environment flags

Every `FG_*` environment variable read by the runtime, the packer, or the test
binaries. Unknown `FG_*` variables are ignored. `tools/check-flags.sh` enforces
this file against the code: every flag read in `src/`, `include/` or `tests/`
must have a row here, a live row must name a flag the code reads, and the read
set must equal the checked-in expected set in that script. A new flag cannot
merge without a row here and an explicit expected-set update.

Policy (PRD, "Canonical build and execution"): performance behavior must be a
default or a CLI option. Environment variables are reserved for
diagnostics/profiling, bisection opt-outs that restore an alternate path on the
same binary, and test/packer tooling. No environment variable may be required to
reach the validated operating point.

Status values:

- **validated default** - unset runs the validated configuration; the flag is a
  bisection opt-out whose `0` value restores the alternate path. The reference
  numbers are measured with it unset.
- **variant (opt-in)** - off by default; only enable for a specific experiment.
  A variant must not be required for the validated operating point.
- **profiling** - diagnostics only; no numeric effect when off (some traces
  force a GPU synchronization and are not throughput-representative).
- **test/bench** - read by test binaries or bench tools only; the serving
  binary ignores them.
- **packer option** - packer-time placement/artifact knob.
- **test/ops** - deployment/test plumbing (for example the shader directory).

## Execution and routing

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_PREFILL_RING` | `1` | Layer-ring prefill. `0` restores the non-ring prefill path. | validated default |
| `FG_DECODE_RING` | `1` | Layer-ring decode. `0` restores the legacy expert-parallel decode replay. | validated default |
| `FG_WORKER_OWNER` | `1` | Worker-side owner executor. `0` opts out. | validated default |
| `FG_NGRAM_PAGEABLE` | `1` (on) | Worker n-gram shard mode. The shard is mapped (`mmap` + `MADV_RANDOM`) and served from the kernel page cache with a 128 MiB mlocked hot prefix and bounded `posix_fadvise(WILLNEED)` read-ahead; `0` restores mlock of the whole shard. Measured equal-or-better at 4K-151K and returns 3.7-3.9 GiB of host headroom per worker. | validated default |
| `FG_PACK_EMBED_RANK` | rank 7 | Packer placement override for `token_embd.weight` (0-7). | packer option |
| `FG_DENSE_K_COOKED` | unset (off) | Cook supported Q4_K/Q5_K weights to the dense cooked layout at load. | variant (opt-in) |
| `FG_DENSE_SOA` | unset (off) | Convert supported dense GGML weights to SoA layout at load. | variant (opt-in) |
| `FG_EXPERT_Q8_WIDE` | unset (off) | `1` opts into subgroup-shuffle wide Q8 MoE decode; off pending fleet A/B. | variant (opt-in) |

Prefix continuation (resuming exact token-prefix extensions from the recorded
owner-state frontier) is the default. Disable it per run with
`--no-prefix-cont` on `flash-gordon chat` or `flash-gordon api`; the
session-level cold reset (`fg_runtime_reset`) is unchanged.

`flash-gordon api --sessions 2` enables two independent ring sessions with
adaptive prefill quanta; omitted/`--sessions 1` retains FIFO execution. This is
a CLI capacity option, not an environment variant. See [MULTI_SESSION.md](MULTI_SESSION.md).

### Experimental two-session scheduler tuning

The scheduler repair is **not fleet-qualified**. These overrides tune/bisect
the opt-in `--sessions 2` path; none is needed for its default policy. They are
read once by the coordinator when creating the scheduler and ignored by
`--sessions 1`. Invalid values fail initialization rather than silently falling
back. Idle resident slots do not count as contenders.

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_MULTI_PREFILL_TOKENS` | `128` | Contended prefill quantum, integer 32–256, capped by the sealed microbatch. With no other active request, use the full sealed microbatch instead. | variant (opt-in) |
| `FG_MULTI_PREFILL_WINDOW` | `1` | Contended frames in flight, integer 1–8, capped by the full frame window. With no other active request, restore all eight frames. Raising this increases decode head-of-line blocking. | variant (opt-in) |
| `FG_MULTI_PREFILL_SLICE_US` | `15000` | Soft rank-service budget while another session decodes, integer 0–1000000 microseconds. Shrinks the token cap using the worst observed coordinator prefill-block cost per token, including its outgoing send. Starts conservatively at one token without a sample; never below one. `0` disables service-time sizing for fixed-cap bisection. Other ranks, fixed costs and already-issued frames can exceed the estimate; this is not a deadline guarantee. | variant (opt-in) |
| `FG_MULTI_DECODE_PRIORITY` | `1` | Ready streamed decode, then other ready decode, then prefill; round-robin ties. `0` bisects to ready round-robin scheduling. Only `0` and `1` are accepted. | variant (opt-in) |

## Output split (4-way head handoff)

The 4-way head split is unconditional: ranks 4/0/1/2 each reduce one
vocabulary slice and the final block owner runs the HC chain. A pack/deployment
must carry the owning rank's `output.weight`/`output_hc_*` extents on the
slice ranks (the foreign loader names the exact missing span at startup).

| Flag | Default | Effect | Status |
|---|---|---|---|
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
| `FG_EVAL_LOGITS` | unset (off) | Write greedy evaluation logits to this file. | profiling |
| `FG_EVAL_LOGITS_DIR` | unset (off) | Write numbered greedy evaluation logits (`%04u.fgl`) in this directory. | profiling |
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
| `FG_QSA_LOCALITY_TRACE` | unset (off) | QSA locality report on every rank: `summary` at reset/close, `token` adds one keyed digest per QSA layer/token. Append `+prefill` to include prefill selections (slow and noisy at long context). Models the decode page stream by default; allocates nothing when off. | profiling |
| `FG_QSA_LOCALITY_MIB` | `16,32,64,128,256,512,1024` | Comma-list of curve budgets used by the locality report. | profiling |
| `FG_QSA_PREFETCH_TRACE` | unset (off) | QSA read-ahead staging trace (pages staged/served/dropped). | profiling |
| `FG_BLOCK_BENCH` | unset (off) | Offline block-service bench; exits after printing per-layer wall times. | test/bench |
| `FG_SHADER_DIR` | `vulkan` | Directory containing the SPIR-V files. | test/ops |

## Vision tower

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_TOWER_FAIL_ALLOC` | unset (off) | One-shot fault injection: fails the nth tower buffer allocation with `VK_ERROR_OUT_OF_DEVICE_MEMORY` to exercise the vision fail-soft path. The failure uses the real allocation-failure code path. | test/ops |

## Static replay

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_DECODE_STATIC` | `1` | Static recorded replay for chained ring blocks. `0` disables it; a Vulkan profile or numeric trace also disables it. | validated default |
| `FG_STATIC_RERECORD` | unset (off) | Re-record the replay run every N tokens (0/unset keeps the recorded run) to bisect the staleness horizon. | profiling |
| `FG_STATIC_CHECK` | unset (off) | Per-token finite-output check of the replayed static run. | profiling |
| `FG_SET_TRACE` | unset (off) | Trace the first dynamic descriptor-epoch crossing into the static set range. | profiling |

## Test and bench flags (test binaries only)

| Flag | Effect |
|---|---|
| `FG_BENCH_IQ_VARIANT` | Select the compact IQ4 shader variant (`r4w32`, `r8w32`, `r4w128`, or `r8w128`). |
| `FG_BENCH_DENSE_COMPACT` | Enable compact dense storage for the dense benchmark. |
| `FG_BENCH_DENSE_COOKED` | Benchmark cooked Q4_K/Q5_K dense weights. |
| `FG_BENCH_DENSE_GGUF` | GGUF input path for the dense benchmark. |
| `FG_BENCH_DENSE_N` | Set the dense benchmark token count. |
| `FG_BENCH_DENSE_ONLY_HEAD` | Benchmark only the output-head shape. |
| `FG_BENCH_DENSE_Q4_ONLY` | Use Q4_K for the Q5_K dense benchmark shape. |
| `FG_BENCH_DENSE_QUANT` | Run the quantized dense benchmark. |
| `FG_BENCH_DENSE_SELECTED` | Limit compact runs to selected shapes and print prior-path comparisons. |
| `FG_BENCH_DENSE_SKIP_HEAD` | Skip the output-head shape in the dense benchmark. |
| `FG_BENCH_DENSE_SOA` | Enable SoA dense storage for the dense benchmark. |
| `FG_BENCH_EXPERT_FORMAT` | Select expert benchmark format (`q5`, `q4q8`, or `q4q5`). |
| `FG_BENCH_IQ_ONLY` | Run only the IQ dense benchmark shape. |
| `FG_BENCH_Q8_GGUF` | GGUF input path for the Q8_0 dense benchmark. |
| `FG_BENCH_ROUTER` | Run the router benchmark instead of the parity check. |
| `FG_BENCH_ROUTER_GGUF` | GGUF input path for the router benchmark. |
| `FG_BENCH_ROUTER_N` | Set router benchmark token count (1-512). |
| `FG_EXPERT_BOUND_DUMP` | Append expert down-projection bounds to this file. |
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
| `FG_DECODE_PIPELINE` | Async decode submission pipeline is unconditional; the descriptor-epoch discipline stays. | No. |
| `FG_DECODE_CHAIN` | Chained ring blocks are unconditional; the per-layer owner machine remains only as the eligibility fallback for packs without whole-slab experts. | No. |
| `FG_DECODE_DIRECT_OUTPUT` | Direct final-block handoff is selected from the topology; the rank-0 relay remains the fallback when the route is not eligible. | No. |
| `FG_DECODE_EXPERT_LEGACY` | The fused expert pair is unconditional; the generic projection path remains for layouts that cannot fuse. | No. |
| `FG_VK_HOLD_STATIC` | Static submissions always coalesce with the pending fence queue. | No. |
| `FG_QSA_SELECT_GPU` | Device-side QSA selection resolve with the exact-miss host fallback is unconditional. | No. |
| `FG_QSA_TOPK_V2` | The chunked top-k is unconditional; the legacy wide kernel and shader were deleted. | No. |
| `FG_FABRIC_DIRECT_SEND` | Never-validated forced direct send; the qualified QSA page append keeps its explicit direct call. | No. |
| `FG_FABRIC_DIRECT_RECV` | Never-validated direct receive; receives always use io_uring. | No. |
| `FG_EXPERT_BATCH_SEND` | Never-validated batched expert fan-out; the send-batch API and test were removed. | No. |
| `FG_MTP_DRAFT_ECHO` | Harness scaffolding for the parked MTP head; the draft/verify scaffold and speculative-accept helpers were deleted. | Yes - with the trained MTP pack and kernels. |
| `FG_OUTPUT_SPLIT_HIDDEN` | The finished 4-way A/B round promoted suppression; the 40 KiB hidden is always dropped under 4-way. | No. |
| `FG_OUTPUT_SPLIT_TIMEOUT_MS` | The split partial wait is a fixed 4000 ms liveness deadline, not a tunable: the value only matters when a helper rank is dead or wedged. | No; the bound is compiled in. |
| `FG_OUTPUT_SPLIT` | The 4-way head split is the unconditional operating point; the former opt-in modes (off / 2-way / 4-way) were removed. | No; the topology is fixed. |
| `FG_PREFIX_CONT` | Replaced by the discoverable `--no-prefix-cont` CLI option on `chat`/`api`. | N/A - use the CLI option. |
