# Flash Gordon: product and implementation design

## Product

Build a readable, model-specific Qwen3.8-Flash-Next inference appliance on **eight
BC-250 blades**, currently 24 CUs with 40 CUs planned. Four additional blades are
reserved for smaller or diffusion models. Twelve-blade support is outside scope.

Serve one active coding/creative session well: Q4-class weights, fast cold and
append prefill, native 262,144-token context, image/video input, trained MTP,
sampling, tool calls and exact transcript reuse. This document supersedes the
earlier recovery checklist and amendments. It describes the chosen implementation,
not completed work. Initial findings are in AUDIT_2026-09-07.md.

## Architecture

Rank 0 executes all 48 common layers in order: embeddings/n-grams, gated residual
operations, GDN/QSA, routing, shared experts and routed reduction. Workers execute
selected experts and serve stored data. Keep the existing output helper.

Remove pipeline parallelism from orchestration, manifests, profiles and protocols.
Workers never execute contiguous transformer stages. Salvage useful math kernels
under operation-specific names, without preserving stage ownership assumptions.

| Area | Decision | Reason |
|---|---|---|
| Placement | Keep eight ranks and rotating four-owner groups initially | Wider fan-out trades compute for communication; existing evidence does not justify repacking |
| Decode experts | Keep cooked layouts and fixed command graphs | Retained EP results support them and document failed alternatives |
| Fabric | Keep TCP/direct send/receive and validation | Existing traces do not justify a transport rewrite |
| Decode n-grams | Keep resident worker shards | Already removes per-token NVMe lookup from decode |
| Prefill dense math | Integrate token-tiled cooked kernels | Generic calls miss weight reuse across tokens |
| Prefill experts | Adapt grouped cooked kernels to local shards | Cooked EP currently schedules individual token/expert pairs |
| Prefill scheduling | Separate routing, dispatch, shared/local work and collection | Shared computation currently finishes before remote dispatch |
| QSA | Keep EP index/page machinery; batch queries and fetches | EP attention loops per token despite batched projections |
| Measurement tools | Reuse | New qualification infrastructure does not repair inference |

Keep the actual mixed Q4_K_XL formats initially: routed Q4_K/Q5_K and Q5_1 with
higher-precision common tensors and state. Record source revision, hashes and
formats using existing metadata. No silent requantization or uniform-Q4 claim.

## Canonical build and execution

There is one canonical product implementation on main. A normal `make` with the
repository's default release configuration must build the complete, optimized,
correct runtime end to end. Required correctness and performance improvements
become the default implementation, not optional features users must discover.

- Delete pipeline-parallel codepaths, orchestration, profiles, build targets and
  feature switches. Do not retain them behind compile-time guards, environment
  variables, runtime branches or experimental launch scripts. Minimal detection
  that rejects a retired artifact is allowed; executing it is not.
- When replacing an incorrect implementation, remove the superseded production
  path and its selector. A reference implementation may remain in tests when it
  provides a useful independent oracle, not as a selectable product fallback.
- Minimize bespoke compiler, Clang/developer and experimental build configurations.
  No special compiler invocation, preprocessor define, environment variable or
  wrapper script may be required to obtain correct routing, full supported
  context, real batched prefill, supported sampling, multimodal or MTP behavior.
- Keep necessary device/pack configuration and explicit user choices distinct
  from implementation switches. Artifact availability and hardware capabilities
  must be validated clearly, not used to silently choose an incorrect path.
- Debug symbols, sanitizers, diagnostic tracing and test-only fault injection
  may remain where useful. They must not enable otherwise-missing product
  semantics or be required for the optimized production path.
- Validate and report the default build and ordinary launch path. A result that
  depends on an undocumented build variant or a special feature-enabling recipe
  does not establish product completion.

Implementation branches are temporary delivery work. They do not define alternate
supported products. As each change lands on main, its intended behavior is the
ordinary build's behavior. Review includes deleting obsolete flags, wrappers and
branches, rather than accumulating compatibility with mistaken requirements.

## Real prefill

Use the existing 128-token native profile initially, with partial-chunk support.
Each chunk traverses all common layers on rank 0, distributing its expert work.

1. Execute compatible common projections with existing token-tiled cooked Q8
   kernels. Preserve correct paths for other formats, tiny shapes and single-token
   decode. Preserve causal GDN recurrence and gated residual semantics.
2. Prepare router logits, exact top-10 IDs/gates and packed activations together.
   Reuse the GPU router after matching the CPU oracle. Read compact routing
   metadata and packed activations for transport.
3. Send remote requests promptly, before waiting for shared or local experts.
   Keep activation storage valid until sends finish.
4. Execute shared and local experts while remote owners work. These jobs share
   the coordinator GPU; asynchronous requests do not create independent compute.
5. Collect ready responses, validate every slot exactly once, reduce in stable
   order (routing slot within each shard, then fixed binary-tree child order), and write the residual.
   Prefill replies contain preweighted per-token shard sums and complete route
   metadata; regrouping FP32 sums is tolerance-tested against the scalar oracle. Drain or invalidate outstanding work
   before reusing buffers after failure.

Split fg_owner_moe_prepare_batch into named routing/packing/shared operations.
Give prefill a start/finish dispatch interface like existing EP decode. A generic
task scheduler, transport thread pool or extra architecture mode is unnecessary.

Replace the cooked one-pair schedule in fg_expert_prefill with expert-major
16-pair tiles consumed by existing grouped cooked kernels. Translate global IDs
to shard-local indices and preserve token/route slots. Remove the stage-local
512-expert restriction from the mathematical helper; validate actual tensor
counts. Cover partial tiles, noncontiguous tokens and supported down formats.
Group with histogram/prefix offsets rather than rescanning pairs for every expert.
Allocate bounded schedules once. Keep decode's proven fixed expert graph.

## Native-context QSA

EP already scores blocks in parallel and selects hierarchically through
score_index_segments/select_blocks in src/qsa.c. The separate resident selector
scores 4,096 blocks serially per workgroup. Do not transplant it into EP or
attribute its latency to EP. The EP reducer still sorts padded 4,096-entry groups;
retain it initially as an exact reference while fixing per-query scheduling.

Keep full index coverage and bounded record cache on rank 0, with complete records
on page owners. Page owners serve bytes, never attention. No hidden sliding window
or approximate selection is allowed.

Implement QSA prefill in bounded query tiles:

- Project/prepare the chunk once and make its records available.
- Score across queries and blocks, masking every query to its own visible prefix
  with correct partial-block behavior.
- Preserve exact top-512 selection and ties using bounded scratch, not a full
  prompt-by-context score matrix.
- Deduplicate selected pages across a tile, group owner fetches and pin pages
  until consumers finish. If the union exceeds cache capacity, split the work
  or gather into bounded scratch; never evict in-use pages or truncate selection.
- Attend with per-query selected IDs and causal tails, then run the tiled output
  projection. Commit progress only after successful execution.

Decode uses the same state and rules with one query. Page-fetch readbacks remain
real dependency boundaries; fuse GPU work between them. Remove unconditional
cache-miss logging from ordinary execution and retain existing opt-in diagnostics.

Current records use 1,236 bytes/token/QSA layer, about 3.62 GiB across twelve QSA
layers at native context before padding, duplicate indexes and caches. Full
coordinator record residency is therefore not the initial design. Native context
counts prompt/media/output tokens; reserve output room when testing generation.

## Memory and storage

Keep expert weights resident. NVMe supports cold data and capacity; adding reads
to already-resident expert decode is not an optimization.

Update the existing ledger as allocations change: common/local expert weights,
cooked layouts, recurrent state, full QSA index, bounded cache, prefill buffers,
transport and OS/driver reserve. Reuse scratch only across documented disjoint
lifetimes. Avoid duplicate decoded weights or permanent old/replacement arenas.
Aggregate fleet memory does not establish rank-0 fit.

Preserve resident distributed n-gram decode and source verification. EP prefill
uses the local NVMe store; retain it during compute repairs. Known prompt rows
permit batching/prefetch without prediction. Reuse io_uring block deduplication
and cache. Historical evidence already found network overhead could defeat
storage-only sharding; do not repeat that approach.

N-gram hashing uses only the current token and two predecessors. Prefill validates
and hashes the new chunk plus those two predecessors; resident decode uses the
same bounded lookup. EOS boundaries outside that window cannot affect the hash.
No cached boundary state is needed, so divergence, rollback and reset cannot
leave it stale. Earlier tokens were validated when the runtime processed them.
Grouped resident prefill lookup remains a possible small protocol extension if
the remaining storage cost warrants it.

Publish QSA records before evicting their only current copy. Distinguish visible
records from durable checkpoints using existing append/barrier semantics. Abort
or reconstruct after failure. Do not add per-token forced durability requirements.

Publication releases cache pins only after the replica queue accepts copied
pages. Fetch drains coordinator sends on the ordered bulk channel and the owner
drains its writer before reading pages. Session rotation waits for owner barriers.
These barriers establish live read visibility, not a power-loss-safe model-state
checkpoint; ordinary restart reconstructs state from conversation tokens. A
poisoned transport requires reopening the runtime rather than reusing its state.

## Decode and API

Keep asynchronous EP dispatch and deferred residual writes. Remove unused
alternative orchestration after callers migrate. Compact GPU routing still needs
host synchronization for networking. Preserve proven shape-specific decode kernels.

Keep output-helper GPU vocabulary selection, temperature/top-k/top-p, penalties,
seeded sampling, tool calls, structured text, streaming and exact prefix reuse.
Bind existing tests to EP execution where they currently use mocks or pipeline
fixtures. Numerical parity uses the same quantized reference; text similarity is
insufficient. Quality relative to source precision is a separate check.

## Multimodal and MTP

Both are required and do not wait for raw 50 TPS.

Implement image/video preprocessing, vision encoder and projection from the pinned
Qwen processor/model contract. Feed embeddings into rank 0's ordinary prefill with
correct multimodal positions and token accounting. Chunk vision work and reuse
temporary memory outside text-layer execution. Persistent vision weight placement
is a pack decision based on actual inventory and free memory; the four supporting
blades are not an implicit dependency.

Add the trained MTP head after base state transitions are correct. Start with one
proposal step and target verification through repaired batched EP. Separate
committed/speculative frontiers. Snapshot or reconstruct GDN/convolution state at
the proposal boundary, isolate uncommitted QSA records and restore n-gram/history/
penalty state on rejection. Replay an accepted prefix from the boundary when
necessary; never retain recurrent state from rejected tokens.

Use exact speculative acceptance/residual sampling with actual draft and target
distributions after sampling transformations. Greedy tests do not establish
stochastic correctness. Report accepted output TPS, raw TPS, acceptance and
draft/verification cost separately. A trained head does not guarantee speedup.

## Implementation order and ownership

Current checkout: step 1 is implemented. The default release build includes
shaders and automatically selects supported cooked weight layouts. Local core,
session, API, storage, model-placement and eight-process fabric checks pass;
Vulkan checks here use llvmpipe, not the BC-250 fleet. This is a source cleanup
and correctness checkpoint, not a new performance result.

Step 2 now uses the GPU top-10 router for prefill and reads compact IDs/gates.
Cooked shard-local experts use 16-pair tiles, launch only populated tiles and
submit gate/up, SwiGLU and down projection together. Global-to-local mapping and
schedule construction are linear in expert count plus routed pairs. Remote work
is sent before shared/local expert computation; all issued replies are drained
on failure. These are ordinary runtime behavior, without new feature switches.
Local tests cover grouped/decode parity, non-default placement, partial tiles,
single-token tails, routing edge cases, dispatch order and failure cleanup.
Common Q8 projections now automatically use size-preserving cooked layouts and
the existing token-tiled kernels through the ordinary dense API for multi-token
calls (64 output rows by 32 tokens per workgroup). GDN prefill uses the
column-parallel recurrence with state kept in registers across the chunk.
Prefill expert results are reduced on the GPU in bounded tiles, preserving
routing-slot order; staging reuses dead attention scratch and stops before live
outputs. Single-token decode retains its existing dispatch. PLE residuals use protected
ping-pong storage through the following GR/attention operations.
QSA prefill commits a chunk's projected records,
scores queries in bounded causal tiles, reduces each query to its exact top-512
complete blocks, and deduplicates cold-page fetches across the tile. Cache hits
are gathered into private query slices before eviction; partial pages expose
only the query's visible tokens. New pages remain pinned until owner publication.
The native 128-token chunk uses four-query tiles within the existing attention
scratch arena; smaller chunks use proportionally smaller tiles. No feature flag
enables this behavior. Selection scratch is included in sealed allocation sizing.
Local checks cover scalar-selection parity across the 128k index boundary, causal
attention with precommitted future records, partial query tiles, deduplicated
fetches, cache exhaustion/eviction and scratch lifetimes. Step 3 now
uses bounded n-gram suffix lookup for both prefill and resident decode. The
storage review tightened cache-pin release to follow successful queue commit;
failure-injection checks cover that boundary. Default packing now seals native
262,144-token context, a 16 MiB record cache and 128-token chunks without a
profile switch. Existing packs retain their sealed budgets until explicitly
upgraded. Step 4 and actual fleet usability remain outstanding. No fleet TPS
result is claimed from these local tests.

1. **One EP runtime:** salvage math helpers, delete stage/pipeline orchestration,
   reject retired packs, reserve retired message IDs, remove profiles and update
   README. Preserve source tensor compatibility.
2. **Real prefill:** tiled common projections, grouped shard-local experts,
   compact routing and shared/remote overlap. Refactor touched long functions
   into named operations with explicit buffer lifetimes.
3. **Native context:** tiled causal QSA, grouped fetches, incremental history and
   complete allocation accounting.
4. **Product completion:** EP API coverage, vision/video and trained MTP rollback.
   Record actual performance after working changes.

Each change builds and receives focused existing reference/integration checks for
affected behavior: route mapping, grouped parity, chunk composition, causality,
failure cleanup or session state. No new benchmark framework, tuning matrix or
mandatory context sweep per commit. New measurements answer a specific unresolved
question or check completed work; public TPS claims require actual results.

The agent owns implementation, technical review and result interpretation. The
user chooses product tradeoffs and need not review Vulkan code. Fleet operations
remain manual and explicitly authorized. Work forward in the current checkout:
no historical reset/revert, cherry-pick or parallel worktree. Do not move tags.
Pipeline throughput cannot qualify EP; useful component evidence remains available
with its limitations. Keep diagnostics concise and remove benchmark-specific
token-number behavior from normal execution. Avoid unnecessary environment switches.

## Performance and completion

The first interactive-usability checkpoint includes a representative 131,072-token
conversation: cold filling, continuing an existing conversation with prefix reuse,
and sustained generation at that history length. Short prompts are smoke tests;
they cannot qualify the user's normal workload. Finish the required causal QSA
and history path before that checkpoint. Native 262,144-token support remains the
product requirement. Use the completed harness for this check; do not introduce a
separate benchmark framework or an open-ended tuning campaign.

Retain requested 250/600 tok/s prefill and 50/100 raw-decode targets as ambitions,
not demonstrated ceilings or feature prerequisites. At 50 TPS the whole token has
20 ms; at 100 TPS it has 10 ms. Coordinator work, networking and storage do not
scale with aggregate fleet bandwidth or CU count. No silent target reduction or
unsupported speedup claim is acceptable.

Completion means the sole EP runtime serves representative text/image/video
sessions through native context with sampling/tools/prefix reuse and correct MTP,
within eight-blade memory limits. Report achieved cold/append prefill, TTFT, raw
and speculative output rates with context length and CU configuration. Explain
remaining costs when targets are missed instead of changing the workload. The
million-token extension is separate work requiring official scaling and validation.

## Prior art and evidence

- [Qwen model contract](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): semantic
  authority; pin model and processor revisions during implementation.
- [DeepEP](https://github.com/deepseek-ai/DeepEP): dispatch/combine and expert
  layout reference. CUDA/NVLink/RDMA code is not a BC-250 dependency, and its
  throughput does not transfer to this fleet.
- [FlashInfer attention interfaces](https://docs.flashinfer.ai/api/attention.html):
  prefill/decode and paged-cache patterns, while preserving Qwen sparse semantics.
- PERFORMANCE_TRACE_10_035TPS.md and PERFORMANCE_20TPS.md: existing EP operation
  ordering, cooked-kernel evidence and failed approaches. Instrumented slices
  guide choices but are not additive runtime forecasts.

### 2026-09-09 fleet prefill checkpoint

The ordinary eight-rank release now dispatches token-tiled common projections,
column-parallel GDN prefill and bounded GPU expert-result reduction by default.
The same 244-token correctness prompt improved from 8.65 to 13.15 prefill tok/s
(28.22 to 18.55 seconds); arithmetic, prefix reuse and Paris smoke checks pass.
Decode remains approximately 9–10 tok/s. These short-context measurements do not
qualify 128k usability or the requested performance targets.

A pre-reduction-fix profile measured 7.71 seconds in expert dispatch/collection
and 4.29 seconds in CPU reduction per 128 tokens, versus 2.06 seconds of coordinator
GPU time. The CPU reduction was moved onto the GPU. Further prefill work must
address communication volume: rank zero's observed link is 1 Gb/s, and the current
protocol returns one 2560-element FP32 vector for every routed expert selection.
At a representative 7/8 remote share this is about 4.30 MB per prompt token,
bounding those result bytes alone to approximately 29 tok/s at ideal line rate.
Shard-local/distributed reduction and an explicit communication budget are needed;
200 end-to-end tok/s is not established by the existing compute microbenchmarks.
Evidence: `results/prefill-integration/RESULT.md` and its saved response/trace files.

### 2026-09-09 prefill communication repair

The default expert prefill implementation now applies routing gates and reduces
local expert outputs on each blade's GPU. Responses contain one FP32 sum per token
per blade, together with every route slot for coverage validation. Each layer uses
four blades; this reduces aggregate returned vector count from ten to at most four.
The response version rejects the old per-expert format. The coordinator accumulates
in fixed ascending rank order; scalar-oracle tests cover the FP32 regrouping.

The same 244-token smoke prompt improved from 13.154880 to 21.441903 prefill TPS
(18.548250 to 11.379587 seconds), returning `12`. Paris and prefix reuse also pass.
Decode remains approximately 9–10 TPS. This is neither a 128k validation nor target
attainment. Full evidence: results/shard-reduction/RESULT.md.

Group broadcast and distributed reduction are now implemented in the default
prefill path. The coordinator sends one activation/routing message to a binary
tree of active remote ranks and receives a single stream of reduced token tiles.
Parents validate exact subtree route coverage and merge cached host arrays before
forwarding each 16-token tile. The coordinator's local experts overlap remote work.
A failed subtree invalidates the mesh rather than allowing partial results to be
reused. See results/collective/RESULT.md for validation and fleet results.

### 2026-09-09 streamed collective fleet checkpoint

The default broadcast/reduction tree reaches 23.272471 prefill TPS on the same
244-token prompt, up from the preceding shard-local implementation's 21.441903 TPS.
It returns `12`; Paris and prefix reuse also pass. Decode remains approximately
9–10 TPS. The initial full-frame tree regressed and was replaced with streamed
16-token replies and merges of cached host buffers; no alternate feature path is
retained. End-to-end targets and long-context qualification remain outstanding.
See results/collective/RESULT.md for exact timings, failure tests, and artifacts.
