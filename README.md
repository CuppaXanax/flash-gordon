# Flash Gordon

Flash Gordon is a Linux-only C17 inference appliance specialized for Qwen3.8-Flash-Next on eight BC250 blades. It deliberately has no generic-model compatibility layer.

Its distributed execution model keeps the sequential common graph, including QSA projection/search/attention, on the coordinator. Each layer's routed experts still fan out across their owning blades; ranks 3 and 7 are the authoritative long-record owners for six QSA layers each. Whole-layer and pipeline parallelism are not part of the design. The immediate raw single-stream decode contract is a 10 tok/s floor and a 20 tok/s engineering target before MTP or multiple sessions.

The current eight-blade LKG qualifies at 99.647 ms/token, or 10.035 tok/s, over the final 20 unprofiled greedy frames with exact response parity and a complete 48-layer route trace. See [PERFORMANCE_TRACE_10_035TPS.md](PERFORMANCE_TRACE_10_035TPS.md) for the qualification record.

The implementation owns its complete runtime boundary: artifact format, rotating expert topology, memory ledger, GGUF parser and repacker, raw io_uring storage, fail-closed wire protocol, quantization primitives, Vulkan allocation/dispatch, and Qwen3.8-specific shaders. There is no linked or vendored inference runtime. Code adapted from another project is copied into Flash Gordon, renamed and maintained here, and admitted to a production path only after model-specific reference and Vulkan parity tests pass. Qwen's published architecture and processor behavior are the semantic authority; behavior inherited from another model runtime is not. Rank and text evaluation refuse a pack until text weights, the n-gram tensor, and tokenizer are sealed into the manifest. Vision and MTP are separately flagged overlays and are not prerequisites for the sealed text profile. The runtime also exposes a resident interactive chat frontend and a deliberately
single-threaded OpenAI-compatible HTTP frontend. Both use the exact Qwen ChatML
template and the same greedy distributed generation path as evaluation.

## Build and test

```sh
make
make test
make test-vulkan
```

The build requires Linux headers with io_uring support, Vulkan headers and loader, and `glslangValidator`. It does not require liburing. The Vulkan suite executes production-dimension Qwen grouped-RMS, gated-residual, dense Q8_0, and routed-expert Q5_1 parity oracles. The core suite also validates the machine-readable 48-layer expert-parallel fleet-trace contract and proves corrupted expert coverage fails closed.

## Interactive chat

```sh
./flash-gordon chat --manifest /srv/flash-gordon/manifest.fgm --max-tokens 512
```

The model runtime stays resident. Initial and divergent turns canonically render
and tokenize the full in-memory transcript. Exact API and interactive
continuations preserve the authoritative tokens the model actually evaluated,
then fully tokenize the new suffix beginning at the pending `<|im_end|>`
boundary. This avoids assuming that decoded generated text re-tokenizes to the
model's original token sequence.
Hits prefill only that boundary and new turn; divergence resets and prefills the
full prompt. The current boot-safe serving profile uses an 8,192-token working
context so QSA records stay on the qualified resident hot path; this is a
temporary qualified profile, not the model's context target. Tokens stream directly to
the terminal. `/clear` clears the
transcript and runtime session; `/quit` exits. `SIGINT` or `SIGTERM` requests a
stop at the next boundary between completed distributed tokens. Each turn ends
with compact prefill, generation, and context-usage metrics.

Both `chat` and `api` accept the runtime-shape contract
`--context-tokens`, `--gpu-index-tokens`, `--qsa-hot-tokens`, and
`--qsa-page-cache-mib`.
Experimental component contracts use `--experimental-context`,
`--experimental-mtp`, and `--experimental-vision`; those component flags remain
disabled. Manifest v5 seals logical/index/compatibility-hint/cache defaults as
8,192/8,192/8,192/0, and omitted runtime options preserve that qualified
profile. A staged tiered-QSA request must specify all four budgets. It accepts
32,768, 65,536, 131,072, or 262,144 logical tokens, requires equal logical and
GPU-index coverage and requires a whole-MiB unified record cache from 16
through 512 MiB. Other combinations fail before fleet
startup. One-shot `eval` uses the same resolved profile, rejects prompt plus
generation beyond the logical limit, and bounds owner state to that limit.

## OpenAI-compatible API

```sh
./flash-gordon api --manifest /srv/flash-gordon/manifest.fgm \
  --host 127.0.0.1 --port 8000
```

The server implements `GET /v1/models` and `POST /v1/chat/completions`.
The model response includes a fail-closed capability extension reporting the
active native context and tool support. Experimental context, MTP, image, and
video remain reported as unavailable until their runtime paths are qualified.
Completions accept string-content system/developer, user, assistant, tool, and
function messages; `max_tokens` or `max_completion_tokens`; and `stream`.
Non-streaming responses are JSON and streaming responses use SSE. The runtime
stays loaded, and its single authoritative session reuses only exact canonical
token prefixes across sequential requests. Shorter or divergent transcripts
reset before a full prefill. The server handles one connection at a time and
closes it after one request.

The server also retains the exact public assistant content and structured tool
calls it returned. A normal OpenAI round trip can therefore continue the raw
runtime transcript, including server-private reasoning tokens, without sending
`reasoning_content` or an opaque cache identifier. Any difference in the echoed
public history fails closed to a reset and full canonical prefill.

Flash Gordon currently performs greedy decoding only. Requests that select
non-greedy sampling, custom stop sequences, or log probabilities receive a
clear `400` response. The API accepts OpenAI function `tools`, `tool_choice`,
historical assistant `tool_calls`, and tool results linked by `tool_call_id`.
Generated native Qwen tool tags are translated into OpenAI `tool_calls` for
both JSON and SSE responses, with `finish_reason` set to `tool_calls`; native
tool syntax is not exposed as assistant content.

## Inspect a prospective pack

```sh
./flash-gordon pack --dry-run --output /tmp/flash-gordon-q38 \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
```

The production pack path is bound to the four official Unsloth `UD-Q4_K_XL` shard sizes and SHA-256 identities. A dry run verifies the complete real GGUF metadata and canonical sizes without reading 111 GB of tensor payload; the full pack hashes every shard and fails before writing a deployment manifest if any payload differs.

`--router-profile FILE` accepts whitespace-separated `layer expert frequency` rows. Optional `--expert-map FILE` accepts one `layer=N ranks=R0,...,R511` row per layer for an explicit placement; the two options are mutually exclusive. Without either option, expert residency is round-robin. Every mode enforces exactly 128 experts on each of the four ranks participating in a layer. The map is a pack-time input only: ownership is sealed into the manifest, and rank/eval runtime never reads or requires the source map file. Maps derived from qualification prompts are oracle-only diagnostics and cannot qualify a release.
The `--profile native-262k-microbatch-128` switch seals a separate native-262K
deployment profile for `pack`, or upgrades a legacy manifest with
`upgrade-manifest`; it never rewrites an existing pack.
Without `--profile`, `upgrade-manifest` only performs the format/protocol
upgrade and preserves the legacy 256-token prefill and default 8,192-token
session budgets.

The sealed memory ledger is architecture-derived. GDN recurrent state and the
complete searchable QSA index are Vulkan-resident on the coordinator. The
coordinator holds one budgeted GPU record-page cache for newly committed and
fetched historical pages, and performs Q/K/V/index projection, record commit,
selection, attention, and output projection locally. Cache-hit decode therefore
has zero QSA request/response traffic and no token-age boundary. Ranks 3 and 7 each
own one bounded rank-specific session file containing six layer regions; their
page service never executes QSA projections. Session rotation and shutdown
remove only those two known paths, preventing nonce-named state-file leaks.

The coordinator's transient PLE, GDN, and QSA projection/attention buffers share
one phase-exclusive Vulkan scratch allocation. At the sealed 256-token
microbatch this is 76,021,760 bytes (72.5 MiB), rather than 142.517578 MiB of
simultaneously reserved family scratch; the 70.017578 MiB reclaim does not
change any tensor shape, index capacity, cache policy, or persistent state.
Coordinator routed-expert tensors remain dedicated allocations. Aliasing fixed
expert graphs into owner scratch changed hardware placement and reduced hot
decode consistency for only 22,993,960 bytes of savings in the native profile;
that cross-executor alias is therefore not part of the production geometry.
The coordinator's QSA selection arrays and expert-output scratch are views of
the same family arena. Expert prefill results consumed by CPU reduction remain
in a dedicated cacheable host arena; reading mapped write-combined Vulkan memory
here reduced short-prompt prefill by roughly four times on BC250. Prefill result
wires, cold-page staging/cache, and n-gram
decode tensors are created only when their phase first needs them. Startup emits
`COORDINATOR_*_LEDGER` records with final requested/allocated bytes and a
physical-memory estimate that counts host-visible Vulkan UMA exactly once. The
estimate reports startup usage separately from a conservative peak: lazy
n-gram tensors/cache, QSA staging/cache, the prefill result wire, and QSA page
transport are all allowed to coexist. The concrete deferred peak ledger is
68,028,066 bytes at the sealed 256-token prefill size. It includes the
788,496-byte concurrent prefill work wire and keeps the 524,288-byte persistent
n-gram I/O buffer in startup host usage; the n-gram cache is a full deferred
allocation. The telemetry also prints the 68,552,354-byte diagnostic sum when
that persistent I/O buffer is shown alongside deferred allocations; it is not
added twice to the projected peak. The 51,817,084-byte Vulkan reclaim and the
separate 1,024-byte host
position reclaim (4,096 to 3,072 bytes) are never combined. Readiness adds the
manifest's rank-0 driver reserve to the larger requested or allocated peak and
reports process RSS and available memory. Fabric, io_uring, thread-stack, and
kernel socket costs that cannot be measured from the runtime are labeled
`unknown_os_overhead=unmeasured`; a positive raw UMA margin alone never
produces `ready`.

`pack --profile native-262k-microbatch-128` creates a separate sealed profile.
The same switch on `upgrade-manifest` converts a legacy manifest to that
profile without changing tensor or rank weight metadata. It has a
262,144-token logical/index context, a unified 16 MiB record cache, and a
128-token microbatch. It does not alter the default 8,192 profile
or an existing pack. The profile halves the prefill working chunk, so
QSA, GDN, PLE, and owner append batches use the same 128-token boundary; decode
shapes and record-cache policy are unchanged. Prefill has twice as many chunk
boundaries for the same prompt and can lose fixed per-batch throughput, while
decode is unaffected by the chunk size. Rank-0 sealed scratch falls from
335,544,320 to 201,326,592 bytes (134,217,728 bytes reclaimed).

Only complete four-record pages are replicated. One record is exactly 1,236
bytes: 544-byte Q8 key, 544-byte Q8 value, 136-byte Q8 index key, and 12-byte
text position. A wire page is therefore 4,944 record bytes plus an 8-byte
layer/block header. Every fourth decoded token sends one one-way batch of six
pages to each owner: `12 + 6*(8 + 4,944) = 29,724` payload bytes per owner.
Including two 32-byte fabric headers, this is 59,512 bytes per four tokens, or
14,878 bytes/token. There is no append acknowledgement. A 512-token prefill
batch sends 768 pages per owner, exactly 3,803,148 payload bytes each. QSA
transport buffers are allocated on first page service. A depth-two all-or-none sender
queue moves socket writes off the token critical path, and each owner copies
accepted batches into a bounded background-writer queue so record-file I/O does
not block its expert dispatch loop. Queue saturation or an asynchronous send or
write failure is sticky and fails the session explicitly; pages are never
silently dropped.

Protocol v6 binds append, barrier, fetch, and result frames to the distributed
session nonce. Owners require exact append and fetch sequence numbers,
manifest-owned layers, contiguous four-token page frontiers, and fetches no
newer than the committed frontier. Duplicate, skipped, stale, future, or
misrouted work fails closed. Before reset or session rotation, the coordinator
sends a barrier on each owner bulk channel and waits until the background
writer drains; only then does it replace owner files via the control channel.
An exact live-prefix hit preserves both local QSA state and owner frontiers.

Cold selection first checks the coordinator hot ring, then its bounded hashed
LRU page cache. A miss batch includes unique next-page prefetches that remain
cold and fit the fixed 512-page request. Only a true miss sends one batched page
fetch to the authoritative owner; index selection is the only point where the
Vulkan batch is submitted early. The staged profiles activate this path, while
the default remains 8,192/8,192/8,192/0. Longer profiles are implemented but
still require the serialized fleet qualification gate.

Startup reports exact requested and Vulkan-required bytes for the QSA index and
for all live Vulkan allocations. The exact index uses 131072-token segments per
QSA layer, so the 262144-token profile has two independently allocated segments
per layer. Score dispatches add the segment block base, while commit and restore
use the segment-local token offset; this keeps global top-k IDs and record
ordering unchanged. Before use, every 4 KiB page of every segment is write/read
verified and restored, so a staged profile proves touched residency rather than
relying on nominal allocation success or host `MemAvailable`.

`FG_QSA_LOCALITY_TRACE=summary` enables a best-effort locality report at reset
or close; `token` also emits one keyed selection digest per QSA layer/token.
Reports contain no raw page IDs. They include selected and deduplicated pages,
hot-tail and cold references, previous-token overlap, observed host-cache
hits/misses, exact reuse distance, and projected LRU hit curves. Curve budgets
default to 16,32,64,128,256,512,1024 MiB and may be replaced with a comma-list
in `FG_QSA_LOCALITY_MIB`. The trace allocates no state when disabled. Because
reading GPU-selected IDs makes the diagnostic submit the active Vulkan batch,
trace runs are for locality sizing rather than throughput qualification.

Both per-layer remote designs are rejected. Candidate `3673156` measured
8.681585 tok/s with twelve serialized RTTs and 983,424 payload bytes/token.
The compact block candidate reduced payload to 246,144 bytes/token but still
measured only 8.787463 tok/s because the twelve dependency-ordered RTTs
remained. Their QSA-block messages stay decodable only as dormant compatibility
contracts; runtime decode and prefill never dispatch them. The hybrid page
architecture still requires a parent fleet gate, and one-way socket
backpressure/encoding cost remains the principal latency risk to measure.

Manifest v5 appends a versioned session contract to the v4 tensor layout:
protocol compatibility, text/four-axis position mode, explicit logical/index/hot/page
budgets, and deterministic component and per-rank state-format fingerprints.
The v5 contract requires protocol v6. The reader continues to accept sealed
v4/protocol-v5 manifests with synthesized text and 8K defaults, while unknown
combinations fail closed. Routed experts, token
embeddings, and narrow common projections preserve their GGML quantized bytes.
Measured winning Q8_0 common matrices are cooked offline into 16-row supertiles
with block-major FP16 scales and row-major quant planes. Both planes are 64-byte
aligned and production matrices retain their original byte count. Runtime tensor
metadata selects the matching kernel and rejects cross-layout use. A 512-expert
GGUF tensor is split into four rank-local segments in ascending global-expert
order according to the manifest map. All output segments begin at 4 KiB
boundaries.

Protocol v6 adds deterministic session identity/frontier encodings, owner
begin/prepare/commit/restore controls, and the QSA page append/barrier/fetch
service. Protocol-v5 execution remains available
only to legacy v4 manifests, which keep the empty session-begin exchange and
exact legacy layer, prefill-layer, and dormant QSA-block payload layouts. Protocol v6
payloads carry the explicit position-axis contract. Fabric receive paths reject frames whose
version differs from the version negotiated by the manifest handshake. The default runtime remains the qualified 8,192-token text profile. Explicit
staged tiered-QSA profiles are admitted but not fleet-qualified; MTP and
multimodal execution remain unavailable, and four-axis manifests are rejected
before model or fleet startup.

## Qualification contract

The manifest fixes the production prefill choice; runtime requests cannot change it. `bench` enumerates microbatches 128/256/512 and windows 1–4 against the 32,768-token, three-warm-run gate. Promotion defaults to 24 CUs. Rank startup rejects corrupted manifests, topology drift, protocol mismatches, and memory-cap violations.

The checked-in [qualification baseline](qualification-baseline.json) freezes the
current fleet, decode, and prefill gates. After deploying a candidate, run the
OpenAI/tool/decode acceptance harness from PowerShell:

```powershell
.\tools\qualify-openai.ps1 -BaseUrl http://192.168.42.42:8080/v1
```

Non-streaming completion responses include `X-Flash-Gordon-*` timing and
live-prefix headers: cache hit/miss, reused and prefilled token counts,
exact-frontier continuation, and reset reason. These diagnostic extensions let
the harness compare engine-reported work rather than HTTP wall time; the JSON body remains
OpenAI-compatible. The harness reads the checked-in baseline itself; callers
cannot lower the frozen LKG threshold. It parses and reconstructs SSE deltas
before checking structured calls and native-tag leakage. A fleet candidate
passes only when this harness succeeds after all eight blades also pass the
build, test, and homogeneous fingerprint gates recorded in the baseline.

`FG_PREFIX_TRACE=1` enables best-effort prefix diagnostics that report only
token counts, mismatch positions, and mismatch reasons. It is disabled by
default so normal continuation hits tokenize only the pending boundary and new
turn, and it never logs token IDs or transcript text.

No fleet command, deployment, CU unlock, or model download is performed by the local build.
