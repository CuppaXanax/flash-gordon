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

MTP uses the existing `--experimental-mtp` **CLI** flag, not an environment
variable. It is default-off and still fails closed: the GPU-routed replay core
and explicit vocabulary-shard argmax are benchmarked, but serving integration
and exact batched ring verification are not linked. See [MTP.md](MTP.md) for the audited weights and qualification
gates. The source exporter and
causal benchmark use command-line arguments; they introduce no new `FG_*` flags.

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
| `FG_NGRAM_LOCAL_CACHE` | unset (off) | `1` opens the coordinator's existing sealed `ngram.iq4nl` through the OS page cache instead of `O_DIRECT`, retaining bounded io_uring batching and the 8 MiB application cache. Single-token IQ4_NL dequantization stages 1,440 packed bytes and 10,240 output bytes in CPU-cacheable stack buffers to avoid scalar reloads from the uncached UMA mapping; arithmetic is unchanged. In legacy expert-parallel decode it also replaces worker scatter/gather with that local lookup; ring decode already reads locally. Requires the full 28.8 GB table on coordinator storage. Experimental, not fleet-qualified; set before startup, not during a session. | variant (opt-in) |
| `FG_PACK_EMBED_RANK` | rank 7 | Packer placement override for `token_embd.weight` (0-7). | packer option |
| `FG_DENSE_K_COOKED` | unset (off) | Cook supported Q4_K/Q5_K weights to the dense cooked layout at load. | variant (opt-in) |
| `FG_DENSE_SOA` | unset (off) | Convert supported dense GGML weights to SoA layout at load. | variant (opt-in) |
| `FG_EXPERT_Q8_WIDE` | unset (off) | `1` opts into subgroup-shuffle wide Q8 MoE decode; off pending fleet A/B. | variant (opt-in) |
| `FG_DECODE_FUSION` | unset (off) | Chained decode fusion bitmask: `1` router + tile schedule, `2` shared add + HC residual, `4` supported shared gate/up + SwiGLU; `7` enables all. Invalid masks disable it. Synthetic parity/performance only; model KL qualification required. | variant (opt-in) |
| `FG_ROOFLINE` | unset (off) | Decode research mask: `1` large compact Q4 GEMV geometry, `2` QSA head geometry, `4` recurrence variants, `8` one-wave router, `16` HC injection/mixing reductions, `32` expert GEMV variants, `64` QSA eight-way split for at least eight records, `128` vec4/WG256 alpha/beta, `256` fused HC up/mix, `512` fused HC down/injection/SiLU, `1024` fused Q4 GDN gate/controls. Unsupported devices/formats fall back (split count is portable). No production promotion; see `results/hillclimb/ROOFLINE.md`. Set before static recording; graph-only expert paths retain baseline. | variant (opt-in) |
| `FG_ROOF_DENSE` | `7` | With roofline bit `1`, compact Q4 GEMV geometry: `1` r4/WG64, `2` r4/WG128, `3` r8/WG256, `4` r8/WG128, `5` r8/WG64, `6` shape-selected WG64 for K=2560 and WG128 otherwise, `7` retain the baseline gate geometry and use r8/WG64 elsewhere to preserve accumulation order. Selections 1-7 leave M<2560 unchanged. Research selections 8-28 are described below; unrecognized selections fall back. | variant (opt-in) |
| `FG_ROOF_QSA` | `2` | With roofline bit `2`: `1` register-tiled one-head/WG64, `2` LDS-tiled three-head/WG192, `3` six-head/WG384, `4` one-head/WG64. Other values retain baseline. | variant (opt-in) |
| `FG_ROOF_EXPERT` | `0` | With roofline bit `32`, low three bits select gate/up: `0` original shuffle epilogue, `1/2` two/four prefetched blocks wave64, `3` one block wave32, `4/5` two/four prefetched blocks wave32, `6/7` WG128/256 with two-block prefetch. Add `8` for true wave32 down. Bits `16/32/48` select Q5_1 down WG64 shuffle / one lane per row / two lanes per row. Gate/up overrides `64/128/192` batch 2/10/5 experts per workgroup and cache activation in LDS; `256/512/1024` cache one expert / two with prefetch4 / one with true wave32. These overrides require Q4/Q4, K2560 and ten slots. Down `2048/4096` parallelizes ten slots in WG32/64 plus a merge, for M2560/ten slots. Wave32 requires subgroup-size-control support. Pending real-pack quality gate. | variant (opt-in) |
| `FG_ROOF_HC` | `0` | Low two bits: `1` losslessly expand HC IQ4 integer-grid weights to cooked Q8 (same half scales), `2` also select one-row-per-lane vector-prefetched low-K Q8 GEMV. Expansion occurs at model load; raw and saved SoA/compact IQ4 are supported, with persistent-cap accounting. Bit `4` chooses eight rather than four HC-down splits, `8` WG64 rather than WG128, `16` true-wave32/WG32 (requires subgroup control), for roofline bit `512`. No weight quantization error is introduced by expansion; model-level qualification remains mandatory. | variant (opt-in) |
| `FG_ROOF_RECURRENT` | `0` | With roofline bit `4`, `1` selects four column partitions/head, four K partitions/column and register-cached state, plus a separate RMS/gate dispatch. `2` uses vec4 state loads/stores, sixteen K parts and subgroup shuffles. `3` caches state in WG512/head and fuses final normalization. Restricted to production 48x128 state; other values retain the original bit-4 research kernel. | variant (opt-in) |
| `FG_SMALL_KERNELS` | `0` | Small-N verification research bitmask, fixed before Vulkan pipelines are created. `1`: two rows/32-lane cohort, dequantize once across N2..4 Q4/Q5/IQ4 and cooked Q8 columns. `2`: WG512/head recurrence, 32 state values/lane across N causal updates. `4`: experimental union-down plus per-token gates in one dispatch, no global partials. `8`: one shared causal QSA prefix gather plus batched split/merge when the entire prefix is resident and at most 2051 tokens; indexed/cold paths retain the existing fallback. `16`: subgroup GU reductions; `32`: add next-block weight prefetch; `64`: scalar FMA GU/down plus subgroup GU reductions (`64` takes precedence over `32`, then `16`). `128`/`256`: four/eight rows per cohort on Q4/Q5/IQ4 with M,K >=1024, otherwise retain `1` or the legacy path; `256` takes precedence. Cohort/state variants require 32/64-lane subgroups and WG512 capability. N1 and ordinary prefill are unchanged. Component and N-versus-sequential-N1 gates are required; no model-quality, greedy-token equivalence, or serving promotion is implied. | variant (opt-in) |
| `FG_ROOF_GDN` | `8` | With roofline bit `1024`, `1/2/4/8` selects F32 control rows per workgroup appended to the fused GDN gate GEMV, trading register pressure against workgroup count. Other values use eight rows. | variant (opt-in) |

Small-N bit `4096` enables cooked-Q8 load/dequantize-once column reuse only
when both M and K are at least1024, retaining legacy small/HC geometry.
It does not enable bit1's two-row Q4/Q5/IQ4 path. With the existing330
profile, the combined research mask is4426. A representative2560×6144
Q8 matrix measured N4/N1≈1.44 on the24-CU .51; this is not a whole-rank
or model-quality pass. N1 and ordinary prefill remain unchanged.

Additional default-off bit `131072` reuses cooked-Q8 expert-down weights across each GPU union's token
pairs, then uses the existing per-token gated four-part reduction. Its
FP32 reassociation requires the component tolerance gate; no precision drops.
Scheduling remains GPU-only, existing scratch is reused, and per-token
gates/IDs are unchanged. N1 is unchanged. This is not a whole-block≤1.5×,
full-model quality or serving qualification; `330+131072=131402`.

Additional default-off bit `262144` restores native HC up+mix fusion for N2..4:
IQ4 compact or cooked-Q8 weights are loaded/decoded once into the N-column
accumulators, followed by the per-token sigmoid mix and injection reduction
in the same dispatch. Requires `FG_ROOFLINE` bit256 and native 64-lane
subgroups; other formats/shapes retain the old path. N1 is unchanged.
This is **not** the earlier rejected compact-HC two-row guard using a
transient experimental bit of the same value. The combined union-down/HC
research mask is `393546`; no full-model or serving promotion is implied.

Small-N bit `512` is a **rejected precision experiment**, not a qualified
selection: packed FP16 GU accumulation enables `shaderFloat16` only when
requested and supported, but fails the independent `1e-4` CPU component gate.
Exact integer operands do not make FP16 accumulation exact. Bit `1024` instead
tests exact software-packed integer GU products: two signed 16-bit lanes,
at most eight Q5-by-Q8 products per partial, with the low-lane borrow restored.
It uses the existing lossless weights and FP32 scale/minimum accumulation,
requires the subgroup capabilities above, and takes precedence over the other
GU variants. Its four-pair activation cache stores packed words, not FP32 vectors.
The production-shape `expert_small_component` CPU oracle is enabled only when
`FG_SMALL_KERNELS` is explicitly present (including `0`); it allocates full
512-expert-addressable GPU slabs, so run it as a targeted test on an idle device.

Roofline dense selections `8/9/10` additionally test one/two/four independent
uvec4 streams per lane with wave64; `11/12` use two/four streams with **true
wave32**, not half-populated wave64. These stream variants cover compact Q4
and IQ4, including HC up/down and small outputs. They do not change pack bytes.
All reassociated variants, including dense `6`, split bit `64` and control bit
`128`, are candidates for the real-pack KL/top-1 gate, not bit-exactness gates.

Further dense research selections: `13/14` use four/eight K splits for compact
Q4 with K>=8192 and M<=1024 (including an extra merge dispatch), and selection
`6` elsewhere. `15/16` use two/one lanes per K-group with aligned uvec2/uvec4
loads and WG64; `17/18` use those geometries with true wave32. Non-Q4 retains
selection `6`. `19` selects `18` for the K=2560 gate/query shapes, `15` for
other large Q4 matrices, and the baseline small/HC geometry. `20` tests IQ4
wave32, `21` tests expanded-F32-metadata wide SoA Q4 with two lanes/group
(requires a matching wide layout; the replay bench recooks large Q4 when
this flag is selected), and `22` tests two lanes/group with WG128. These
are off-by-default experiments, not a commitment to their performance.

Roofline mask `256` fuses HC IQ4/cooked-Q8 up projection with four-group
mixing and injection-partial finalization (batch1 only). Mask `512` fuses
compact-Q4 HC down K-splits with injection partials, then merges and applies
SiLU **after scaling its argument by 0.25**. Both are wired into owner decode;
prefill and unsupported formats retain the existing paths.
Mask `1024` appends twelve F32-control workgroups to the Q4 GDN gate GEMV,
producing gate, alpha and beta in one dispatch; it requires true wave32 and
the production compact-Q4 K2560/M6144 shape.
Expert down high bits `8192/16384` split selected slots into two/four
parallel in-workgroup cohorts (WG64/128, still sixteen output rows per
workgroup). `32768/65536` use the same geometries with true wave32 when
supported. Unlike `2048/4096`, they reduce partials in LDS and require no
second dispatch; these paths require Q5_1, K640 and ten selected experts.
HC bit `32` disables K splitting and fuses down, injection and scaled SiLU
in one dispatch. With it, bits `8/16/64/128` choose WG64/true-wave32-WG32/
WG256/WG512 instead of WG128; precedence is 128, 64, 16, 8.
Recurrence selector `3` keeps all 128 columns and four K parts in one
WG512/head, caches state in registers, and also fuses the final RMS/gate.
Dense `23/24` test two-lane IQ4/WG64 and one-lane IQ4/true-wave32;
`25-28` test compact-Q4 four lanes/row, two/four prefetched groups, WG64
then true-wave32/WG32. These retain adaptive `19` on other types/shapes.

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
| `FG_DECODE_MS` | unset (off) | Per-token decode timings; experimental MTP also emits per-owner verification, native draft/repair, local-chain and vocabulary timings, plus pass-local `MTP_CHOICE` token/argmax records. Successful MTP rounds always emit a disjoint `MTP_STAGE` ledger for draft, checkpoint, verify, state-only target repair, native repair, commit and residual host time; `overlap_ms` is subtracted from draft+checkpoint, not counted twice. | profiling |
| `FG_TRACE_ROUTES` | unset (off) | Router/expert route traces. | profiling |
| `FG_PREFIX_TRACE` | unset (off) | Prefix reuse diagnostics (counts, mismatch positions and reasons only). | profiling |
| `FG_GENERATE_TRACE` | unset (off) | Generation loop breakdown (interrupt/tokenizer/callback/render/decode). | profiling |
| `FG_DECODE_RING_TRACE` | unset (off) | Ring decode issue/stage/done traces. | profiling |
| `FG_RING_TRACE` | unset (off) | Ring prefill issue/stage/done traces; one API request-ID/count/SHA256 receipt over emitted little-endian u32 token IDs for same-build greedy qualification. | profiling |
| `FG_PROFILE_TOKEN` | unset (off) | Selects a single token index; enables the detailed per-token captures below. | profiling |
| `FG_NGRAM_VERIFY` | unset (off) | Verified n-gram row comparison for the `FG_PROFILE_TOKEN` token. | profiling |
| `FG_QSA_WARM_TRACE` | unset (off) | QSA cache warming trace. | profiling |
| `FG_QSA_WORKER_CACHE_MIB` | `640` | Worker QSA record-cache ceiling in MiB, clamped to each rank's complete page set. Integer overrides16..640 are accepted; invalid values retain the default. Use the same limit on both arms of memory-constrained diagnostics. Does not change logical context or the coordinator CLI cache. | variant (opt-in) |
| `FG_NGRAM_LOCALITY_TRACE` | unset (off) | n-gram locality summary. | profiling |
| `FG_QSA_LOCALITY_TRACE` | unset (off) | QSA locality report on every rank: `summary` at reset/close, `token` adds one keyed digest per QSA layer/token. Append `+prefill` to include prefill selections (slow and noisy at long context). Models the decode page stream by default; allocates nothing when off. | profiling |
| `FG_QSA_LOCALITY_MIB` | `16,32,64,128,256,512,1024` | Comma-list of curve budgets used by the locality report. | profiling |
| `FG_QSA_PREFETCH_TRACE` | unset (off) | QSA read-ahead staging trace (pages staged/served/dropped). | profiling |
| `FG_BLOCK_BENCH` | unset (off) | Offline block-service bench; exits after printing per-layer wall times. | test/bench |
| `FG_SHADER_DIR` | `vulkan` | Directory containing the SPIR-V files. | test/ops |
| `FG_TEST_Q8_DOWN_TENSOR` | unset (synthetic fixture) | `small_q8_down_native` component test: read an authenticated, already-cooked 512-expert Q8_0 down slab (640 inputs, 2560 outputs, exactly 891289600 bytes). Compare N2..4 bytewise against native N1 on dense/cancellation/basis inputs and check the basis against an independent CPU decoder. Does not enable serving or assert whole-model quality. | test/bench |

## Vision tower

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_TOWER_FAIL_ALLOC` | unset (off) | One-shot fault injection: fails the nth tower buffer allocation with `VK_ERROR_OUT_OF_DEVICE_MEMORY` to exercise the vision fail-soft path. The failure uses the real allocation-failure code path. | test/ops |

## Static replay

| Flag | Default | Effect | Status |
|---|---|---|---|
| `FG_DECODE_STATIC` | `1` | Static recorded replay for chained ring blocks. `0` disables it; a Vulkan profile or numeric trace also disables it. | validated default |
| `FG_DECODE_SUBMIT_GROUP` | unset (off) | `1`: one-submit static GDN/dynamic QSA block. `2`: publish the first GDN run asynchronously, then submit the remainder to retain CPU/GPU overlap. Existing kernels; mutable uploads are snapshotted; sparse/undersized QSA caches and unallocated index segments fall back. Experimental; fleet greedy/hash and performance gates remain required. | variant (opt-in) |
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
| `FG_BENCH_HC_INJECT` | HC-inject partial vs two-pass benchmark; with `roof_hc_up_mix`, five paired profiles at K320/640 for original IQ4 and its lossless Q8 expansion. |
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
