# Flash Gordon Expert-Parallel Recovery PRD

## Authority and status

This is the authoritative architecture and delivery contract for Flash Gordon. It
supersedes every earlier plan that permitted an eight-stage layer pipeline, stage-local
decode or prefill, or a pipeline deployment profile.

Flash Gordon is an expert-parallel inference appliance for Qwen3.8-Flash-Next on eight
AMD BC250 blades. Pipeline parallelism is not a supported mode, fallback, experiment,
qualification target, or future option in this repository.

Recovery is forward-only from current `main` on `codex/expert-parallel-forward` in the
root checkout. Recovery may add commits and merge reviewed branches. It may not rewrite
history, revert commits, reset to an old tree, check out an old revision, cherry-pick an
old implementation, or use worktrees.

## Incident statement and evidence status

The intended system kept the sequential common graph on the coordinator and fanned each
layer's routed experts out to their owning blades. A separate pipeline runtime was later
introduced. Pipeline-specific API, sampling, prefill, qualification, and promotion work
was then presented as the production LKG even though the harness explicitly reported
`execution_mode=pipeline`.

That promotion violated the architecture. Its end-to-end throughput, TTFT, context-curve,
and stage-service results are invalid as Flash Gordon qualification evidence. These labels
may remain in immutable Git history but must be identified as invalidated experiments in
the current tree:

- `lkg-10.49tps-pipeline-decode`
- `lkg-15.18tps-106.07prefill-p4`
- Results derived from `/home/user/flash-gordon-pipeline-candidate`
- Qualifications that required `execution_mode=pipeline`

No historical tag will be deleted or moved. A forward commit will record the invalidation
and prevent these artifacts from satisfying a current release gate. All prior fleet
performance data is excluded from the new EP baseline. Component observations may inform
test design, but every number used for prioritization, release, or public claims must be
remeasured by the EP-only harness.

## Product objective

Deliver one Qwen3.8-Flash-Next runtime that:

- Runs the complete ordered 48-layer common graph on the coordinator.
- Routes every layer's top-10 experts to their sealed owners.
- Executes routed experts concurrently across participating blades.
- Overlaps the coordinator's shared expert with routed experts where dependencies allow.
- Validates and reduces all routed results before advancing to the next layer.
- Supports batched prefill and raw decode through the native 262,144-token context.
- Supports nonzero temperature, `top_k`, `top_p`, presence/frequency penalties, and
  repetition penalty without full-vocabulary CPU readback.
- Provides an OpenAI-compatible API suitable for Pi coding sessions, including tools,
  structured content, transcript reuse, and context-sized completion budgets.
- Fails closed on invalid topology, routing, protocol, artifacts, context, and sampling.
- Measures the real EP critical path and cannot qualify another architecture through a
  manifest, command switch, or benchmark wrapper.

Flash Gordon is a creative-studio appliance, not a generic inference-provider platform.
Generic model abstractions, multi-tenant scheduling, and benchmark-only execution modes
are out of scope unless they directly improve this model's correctness or single-session
performance.

## Required architecture

### Coordinator common graph

Rank 0 owns and executes all 48 ordered transformer layers:

1. Token embedding and PLE/n-gram injection.
2. Hyperconnection attention read.
3. GDN or QSA attention-family work.
4. Hyperconnection attention write.
5. Hyperconnection FFN read.
6. Router logits and deterministic top-10 selection.
7. Shared expert.
8. Expert fan-out, collection, weighted reduction, and FFN write.
9. Final norm and output sampling, whether the sealed output bundle remains on a helper
   rank or becomes coordinator-resident in a later measured change.

Layer dependencies remain sequential. The coordinator never sends a generic hidden-state
boundary to another blade so that blade can execute a contiguous layer range.

### Expert workers

Ranks 1 through 7, plus rank 0 where sealed ownership requires it, provide expert service
rather than transformer-stage service. At every layer:

- The top-10 route is partitioned by the manifest's `(layer, expert) -> owner` map.
- One bounded request per participating owner contains only the required activation and
  selected expert slots.
- Independent owners execute concurrently.
- Responses identify request, session, token/prefill range, layer, experts, slots,
  quantization contract, and payload length.
- The coordinator rejects missing, duplicate, stale, misrouted, truncated, or non-finite
  results before advancing.
- Every routed expert is reduced exactly once using its model-defined gate.

Workers must not own contiguous transformer layers, execute attention as stages, receive
stage credits, or forward hidden states to another worker.

### Placement

The packer seals one owner for every `(layer, expert)` pair while respecting persistent
and transient memory limits. Placement is prompt-independent. Oracle maps learned from a
qualification prompt are diagnostic-only and can never qualify or ship.

The initial recovery retains balanced rotating ownership unless an EP trace proves a
better prompt-independent map. A placement change requires identical routing semantics,
a complete memory ledger, representative-corpus evidence, and no context or route-shape
regression.

### QSA and native context

QSA is part of the coordinator's ordered common graph. Durable records may be stored or
served by designated blades, but those blades are page owners, not transformer stages.
Page fetch, cache, selection, and attention must be causally correct and measured.

Native context means 262,144 real accumulated tokens with all required records and
positions. It may not mean a smaller hidden resident window, repeated filler, skipped
records, undeclared approximate selection, or precomputed state.

### Prefill

Prefill is a true token-batched EP graph:

- The coordinator executes common projections in token tiles.
- Router selection is computed for every token.
- Routed `(token, expert)` pairs are grouped by owner and expert.
- Owners process grouped expert batches concurrently.
- Results return in bounded grouped payloads and reduce in token order.
- GDN scans, QSA commits, n-gram injection, and hyperconnections preserve causality.

Prefill must not run decode once per prompt token. It must not distribute contiguous
layers across blades to manufacture aggregate throughput.

### Decode

Decode carries one autoregressive token through the coordinator's complete common graph.
At each MoE layer, otherwise-idle blades become useful through concurrent expert fan-out.
The critical path is common-graph time plus the maximum participating owner time, not the
sum of eight transformer stages. MTP/speculation is out of scope until raw EP decode clears
its gate.

## Pipeline-parallel removal contract

Pipeline parallelism is removed, not hidden behind a feature flag.

### Runtime and public surface

The forward implementation removes:

- `FG_EXECUTION_PIPELINE` and every runtime branch selected by it.
- `fg_pipeline_*`, `fg_pipeline_runtime_*`, and `fg_stage_*` model orchestration APIs.
- Contiguous layer-owner execution as a deployment mode.
- Pipeline activation, credit, result, drain, drained, and abort messages.
- Pipeline slots, frontiers, admission, stage state, and terminal forwarding.
- Pipeline CLI profiles, pack modes, API branches, tests, and fleet targets.

The expected deletion inventory, subject to a dependency audit, is:

- `include/fg_pipeline.h`, `include/fg_pipeline_runtime.h`, `include/fg_stage.h`
- `src/pipeline.c`, `src/pipeline_runtime.c`, `src/stage.c`
- `tests/test_pipeline.c`, `tests/test_stage.c`, `tests/pipeline_manifest_fixture.h`
- `tools/pipeline-canonical-dry-run.sh`

The build must stop compiling or exposing these components.

### Manifest and protocol migration

New manifests encode one fixed architecture identity: Expert Parallel. They offer no
execution-mode choice. The reader may retain the minimum code needed to identify an old
pipeline manifest, but must reject it with a stable error such as `retired pipeline
manifest is unsupported; repack for expert-parallel`. It must never reinterpret old
weight placement.

Retired message numbers remain reserved and produce a protocol error if received. Their
payload types, codecs, and handlers are deleted.

### Kernel salvage

A mathematically useful GPU kernel is not deleted solely because it arrived with pipeline
work. Each is classified:

- Keep and rename it if it serves EP model math, has a reference oracle, and contains no
  stage or contiguous-layer assumption.
- Adapt it to EP when it improves coordinator token tiling or grouped expert service.
- Delete it if its only consumer is stage transport, ownership, credit, or termination.

No dead shader, public function, or compatibility wrapper remains. Every retained kernel
has an EP-oriented name, caller, and test.

### Static anti-regression gate

A source-tree test fails if production/build/tool/test code reintroduces the removed
files, `FG_EXECUTION_PIPELINE`, `FG_MSG_PIPELINE_*`, model-execution `fg_pipeline_*` or
`fg_stage_*` symbols, or a contiguous-layer pack profile. Incident documents may name the
retired design, and manifest parsing may contain one stable rejection message. Generic
Vulkan compute-pipeline terminology is allowed.

## User-visible feature preservation

The EP runtime must own and test:

- GPU-bounded vocabulary selection and temperature-zero greedy generation.
- Finite positive temperature, `top_k`, `top_p`, and their interaction order.
- Presence, frequency, and repetition penalties over accepted history.
- Deterministic seeded tests and bounded production RNG.
- Supported string/structured content and system/developer/user/assistant/tool roles.
- Assistant tool calls, tool-call IDs, and tool results.
- Prefix reuse without divergent or unevaluated transcript acceptance.
- Completion budgets up to remaining native capacity.
- OpenAI-compatible validation errors.

Tests must instantiate an EP manifest/runtime fixture. A mode-neutral parser test does not
prove that a feature reaches EP inference.

## Correctness contract

Qwen's architecture and processor behavior are semantic authority. llama.cpp's Qwen code
is prior art and an oracle, not permission to alter semantics.

Required parity evidence includes exact tokenizer IDs; checkpoints at layers 0, 1, 3, 7,
8, 15, 31, 40, and 47; GDN/QSA state transitions; router logits and ordered top-10 routes;
per-expert outputs; reduced MoE output; final logits; sampled-distribution checks; prefix
reuse/full recompute equivalence; and cold/append prefill equivalence. Tolerances are
specified per operation. Text similarity alone is not a gate.

## Observability and EP conformance

Every request has a stable correlation ID. Measured records include coordinator GPU time
by layer/family, routing, fan-out timestamps, anonymous route counts, worker queue/GPU/
response time, shared/routed overlap, collection/reduction, QSA selection/cache/page
traffic, n-gram work, sampling, correlated fabric bytes, TTFT, prefill TPS, and raw decode
TPS.

An EP conformance trace proves:

- Rank 0 enters and exits all 48 ordered layers.
- Every routed slot has exactly one sealed owner and returned result.
- Independent owners overlap when a route spans owners.
- Shared/routed work overlaps when enabled.
- Workers execute experts and never transformer stages.
- No retired message type appears.

Crashes leave request ID, last completed layer, outstanding owners, transport state,
Vulkan result, and exit signal in durable logs. Core dumps supplement structured logs.

## Measurement and anti-reward-hacking rules

No throughput is valid until EP conformance passes. Representative coding/chat corpora and
append-only sessions are required. Repeated filler, prompt-trained placement, cached
answers, skipped operations, reduced dimensions/expert count, truncated attention, hidden
context caps, state snapshots, and synthetic-only product evidence are prohibited.

Kernel benchmarks are labeled component measurements and never model TPS, fleet TPS, TTFT,
or release results. Every performance report includes commit/dirty state, artifact and EP
identity, eight device identities/CUs, corpus and token counts, all repetitions, median,
p95, failures, tokenizer/prefill/TTFT/decode separation, useful-work/idle accounting, and
all context points: empty, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 192K, 256K, and 262K.

Discontinuities trigger investigation and are never excluded from an average. No public
claim comes from an unmerged branch, an instrumented run represented as uninstrumented,
or a candidate failing semantic gates.

## Performance requirements

These are requirements, not present claims:

| Workload | Minimum | Directional target |
|---|---:|---:|
| 4K cold graph prefill | 250 tok/s | 600 tok/s |
| Raw single-stream 4K decode before MTP | 50 tok/s | 100 tok/s |
| Native 262K prefill | Complete and maximize | Derive from EP roofline |
| Decode context curve | No unexplained cliff | Smooth decline through 262K |
| OpenAI/Pi session | Representative coding session completes | Daily-driver stability |

A lower engineering baseline may be recorded honestly but cannot replace these gates.
Before optimization, construct a first-principles budget from sustainable bandwidth,
quantized matrix throughput, bytes/operations for common and selected-expert work, maximum
concurrent owner time, correlated fabric cost, QSA context work, and required submissions.
Comparisons with llama.cpp validate ambition but do not define the fleet ceiling.

## Forward recovery plan and commit boundaries

Every leg is a reviewable forward commit that compiles and passes relevant local tests.
There is no fleet computation or deployment during R0-R5.

### R0 — Specification and evidence invalidation

- Install this PRD as the single architecture authority.
- Add an invalid-results ledger and README warning.
- Separate last verified EP evidence from invalid pipeline evidence.
- Change no runtime behavior.

Exit: the current tree cannot plausibly describe pipeline mode as a product objective.

### R1 — Architecture lock and manifest migration

- Make EP the sole new-manifest architecture.
- Remove pipeline profile creation/selection.
- Reject historical pipeline manifests without reinterpretation.
- Make architecture reporting constant and explicit.
- Add manifest and CLI regression tests.

Exit: no command creates or launches a new pipeline deployment.

### R2 — Remove transport and orchestration

- Delete stage/pipeline runtime files, public headers, tests, and build targets.
- Remove retired protocol codecs/handlers while reserving numeric IDs.
- Remove contiguous layer-owner branches and pipeline-only repository tools.
- Add the anti-regression scan.

Exit: the tree builds with no pipeline execution implementation or selectable path.

### R3 — EP runtime closure

- Repair ownership assumptions exposed by R2.
- Make coordinator/worker construction unconditionally EP-shaped.
- Verify top-10 fan-out, concurrent owners, shared overlap, and exact reduction.
- Salvage useful tiled/grouped kernels behind EP-oriented interfaces.
- Add eight-process synthetic EP success and failure tests.

Exit: local integration proves EP is the sole topology.

### R4 — EP OpenAI and sampler closure

- Bind API, chat, tool, history, budget, and sampling tests to EP fixtures.
- Add a Pi-shaped multi-turn/tool transcript integration test.
- Verify GPU sampling against a CPU reference without full-vocabulary readback.

Exit: every user-visible feature previously exercised on the wrong runtime is proven on EP.

### R5 — EP measurement suite

- Add correlated EP critical-path tracing and representative context collectors.
- Assert architecture/routing conformance before recording throughput.
- Emit raw machine-readable records and deterministic summaries.
- Keep measurement and promotion as separate operations.

Exit: the suite attributes coordinator, expert, QSA, n-gram, fabric, and synchronization
cost without deploying or mutating the operational service.

### Mandatory manual pause

After R0-R5, stop for user review of the complete diff and local test output. Code review
does not authorize fleet action.

### R6 — Isolated manual fleet correctness

Only with explicit direction: build the exact commit on all blades in an isolated location,
leave the operational pack untouched until the test window, start one EP coordinator and
seven workers, prove conformance before evaluation, run bounded parity/API/Pi checks,
collect failure evidence, stop the candidate, and restore normal service manually.

### R7 — Valid EP baseline

In a separate approved window: run bounded 4K prefill and short decode, then the context
curve only after correctness and tracing pass. Produce the first valid EP baseline and
first-principles budget. Select one bottleneck from evidence.

### R8+ — Evidence-driven optimization

Possible families are tiled coordinator projections, grouped EP prefill, QSA selection and
locality, expert kernels, fabric overlap, and submission reduction. Their order is not
fixed before R7. Each commit requires reference parity, EP conformance, affected curve
non-regression, median/p95 improvement, and no workload or semantic change.

## Local test matrix

The recovery branch covers:

1. Clean warnings-as-errors build and shader registration.
2. EP manifest creation/load and old-pipeline rejection/corruption/truncation.
3. EP protocol and retired-message rejection.
4. Complete `(layer, expert)` ownership and top-10/gate parity.
5. Expert success, timeout, duplicate, stale, wrong-owner, and partial failure.
6. Concurrent fan-out and deterministic reduction.
7. GDN, QSA, hyperconnection, PLE, MoE prefill/decode parity.
8. Sampling, penalties, OpenAI messages/streaming/tools/history/errors.
9. EP trace validation, anti-pipeline scan, and injected crash-report completeness.

CPU references, software Vulkan, loopback processes, and small fixtures are allowed for
local correctness. Their timings are never fleet evidence.

## Deployment and promotion policy

Fleet actions are manual and user-observed. Promotion requires one evidence bundle with a
clean identical commit on all blades, EP artifact identity, complete conformance trace,
numerical/API parity, representative 4K prefill and raw decode, the context curve required
for the claim, per-rank memory/health, zero unexplained failures, and explicit user approval.

The promotion tool independently validates evidence instead of trusting a filename or
mutable summary. It refuses every retired pipeline artifact.

## Multimodal and MTP boundary

Vision/video and MTP do not begin until text EP is the sole architecture, passes Pi-shaped
sessions, and has a trustworthy curve. Multimodal extends the EP contract and cannot
reintroduce contiguous stages. MTP always reports raw base decode separately.

## Definition of done

Recovery is complete only when:

- This architecture contract is merged into current `main`.
- Pipeline/stage runtime, profiles, protocol handlers, tests, and tools are absent.
- Old pipeline manifests fail closed and cannot promote.
- The anti-regression gate prevents reintroduction.
- The sole runtime executes all common layers on rank 0 and concurrently fans experts to
  sealed owners.
- Sampling, OpenAI tools/history, and native context are proven on EP locally and on fleet.
- A representative Pi coding session completes without workaround.
- The first valid EP context curve and critical-path budget are archived.
- Reports use only EP-conformant evidence and retain every curve point.
- The user manually approves candidate, merge, and promotion.

Correctness below required performance is an honest intermediate milestone, not completion.
Meeting a TPS number with the wrong architecture, reduced work, or selective harness is
failure.
