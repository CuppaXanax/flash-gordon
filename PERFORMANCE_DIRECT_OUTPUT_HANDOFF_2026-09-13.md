# Direct Output Handoff and Output-Head Fixed Cost (2026-09-13)

Worktree `fg-work-headfold`, based on `06cbe6c` (chained ring blocks, host
single-row embedding, row-8 output logits). Scope: the fixed per-token cost of
the ring tail on rank 0 (embed + hops + output relay) and rank 4's output head.
Changes stay inside `include/fg_protocol.h`, `src/protocol.c`, `src/runtime.c`,
`src/output.c`; no new kernels and no `src/vk.c` edits.

## 1. Direct final-block -> output-owner handoff (implemented, default on)

The old tail was `rank 7 --40 KiB--> rank 0` (`DECODE_LAYER_RESULT`), then
`rank 0 --40 KiB--> rank 4` (`OUTPUT_WORK`), 16-byte `OUTPUT_RESULT` back to
rank 0. The direct route removes the rank-0 relay:

- `FG_MSG_OUTPUT_CONFIG = 46` (control, 40 bytes): rank 0 ships the sampler
  config + uniform for the token at token start, before the chain runs.
- `FG_MSG_OUTPUT_HIDDEN = 47` (bulk, `FG_DECODE_LAYER_RESULT_BYTES`): the final
  block owner sends the 40 KiB hyper straight to the output owner.
- Rank 4 matches the two halves by `token_index` and runs the head as soon as
  both are present; `OUTPUT_RESULT` still returns to rank 0 on the bulk
  channel with `sequence = token_index*48+48`.

Env gate: `FG_DECODE_DIRECT_OUTPUT` (default on; `=0` restores the rank-0
relay). The route is selected from the manifest
(`fg_output_owner_rank(manifest) != 0`, `!= manifest->layer_owner[47]`) plus the
env, so all ranks agree **only if the env matches on every rank**. A mixed-env
deployment fails loudly at the first token instead of hanging: rank 0 rejects an
unexpected `DECODE_LAYER_RESULT`, and rank 4 rejects a config/hidden when its
own `decode_direct_output_eligible` is false. Documented fallback: set
`FG_DECODE_DIRECT_OUTPUT=0` on all ranks.

### Hazards handled (from the prior patch map)

- **Arrival order**: `fg_output_handoff` is a one-deep matcher
  (`src/protocol.c`). Config-first and hidden-first both become ready; a newer
  token replaces the pending pair; duplicates refresh in place; stale halves
  are dropped without error. Unit-tested for both orders, duplicates, stale
  configs, stale hiddens and newer-token replacement.
- **Stale token rejection**: the matcher never mixes token indexes; readiness
  requires `config.token_index == hidden.token_index`.
- **Frame sequencing**: rank 4 validates session, peer, source/destination,
  layer 47, `sequence = token*48+47` for the hidden and `token*48+48` for the
  config; rank 0 accepts only `FG_MSG_OUTPUT_RESULT` with `sequence =
  token*48+48`. `FG_MSG_DECODE_LAYER_RESULT` arriving on rank 0 while direct is
  active is a hard mismatch, not a silent hang.
- **Rank 4's own block still flows**: the worker loop is unchanged except for
  two new branches; a pending config/hidden simply sits in the state struct
  while rank 4 executes and forwards layers 24-29. Pending state is cleared on
  `SESSION_BEGIN` so a new session cannot inherit a stale pair.
- **Sampler state**: the uniform draw happens exactly once per token, at
  config-send time on rank 0 (same per-token order as the old relay path).
- **Wire support**: `message_type_supported` in `src/protocol.c` now admits
  ids 46/47 for protocol 6; without this the fabric would reject the frames.

### Expected saving

Removed per token: rank-0 `fg_vk_tensor_read` of the 40 KiB hyper, the 40 KiB
big-endian encode (26.2 us measured locally on 10,240 values), the 40 KiB
control send, the 7->0 bulk hop and its rank-0 receive; rank 4's decode cost is
unchanged (it decodes 40 KiB either way). One 40 KiB hop remains (7->4) instead
of two. The prior patch map estimated **0.3-0.6 ms/token** for exactly this cut;
the local encode alone is 26 us and the hop is expected to dominate.

## 2. Output head (rank 4)

The head is one `fg_vk_begin`/`fg_vk_end` batch: HC RMS norm, HC down, SiLU,
HC up, HC finalize, the 2560->248320 cooked-Q8 GEMM, then the two-pass
hierarchical argmax. Fleet data already established the shape of the budget:
the isolated projection was 2.625 ms (row-4) and rank-4 output wall 2.933 ms;
row-8 selects the same shape now. So the non-GEMM head is only ~0.3 ms and the
hierarchical argmax is already 0.020 ms.

Change made: the per-token `greedy:` debug line was an unbuffered `stderr`
write plus two tensor maps on the output critical path; it is now behind
`FG_OUTPUT_TRACE` (default off). No numerics changed; the sampled token and
logit are produced by the same kernels with the same operands.

Honest expected saving: **0.02-0.05 ms/token** (one unbuffered write syscall).
The GEMM is DRAM-bound on 675 MB and cannot be reduced from `src/output.c`
without kernel fusion or splitting the vocabulary; the HC passes are ~7 MB of
traffic and the argmax ~1 MB, both already minimal. Direct handoff, not
`src/output.c`, is what removes rank-0 fixed cost.

## 3. Stretch: per-token pipelining across the chain (analysis only, no code)

**Diagnosis: a single sequence cannot pipeline the ring without speculation.**
The dependency cycle is absolute: `embed(T+1)` needs `sample(T)`, which needs
the output head of `T`, which needs the last block of `T`, which needs every
earlier block of `T`. Each rank also advances token-serial GDN/QSA/PLE state, so
rank `r` cannot start `T+1` before finishing `T`. A hop cannot overlap the next
block of the same token because the next block consumes the payload; the
sender only has the payload after its own block ends.

What is legally overlap-able today:

- Hop *exposure*: rank `r+1` is idle while the 40 KiB is in flight
  (`egress_ms` ~0.09 per block in the trace). The receiver could be parked in
  `recv` (already effectively true) but cannot begin compute earlier.
- QSA page publishing and warm fetches already overlap block work.
- The output head cannot overlap anything: it is the tail and its result ends
  the token.

The real alternative that fills the ring is **two independent sequences
(continuous batching)**: while rank `r` computes sequence B, sequence A's
message is queued at rank `r+1`. In steady state each rank alternates blocks,
so per-token latency stays ~sum(blocks) but throughput approaches
`1/max(stage)`; rank 4's stage is block + output (~11 ms), giving ~2x current
TPS at batch 2 (bounded by DRAM bandwidth sharing). This is not MTP and needs
no new kernels: it needs the session/state layer to hold two token streams
(the runtime already allocates `FG_PREFILL_FRAMES` slots for prefill).

A second, larger option (already noted by the prior agent): **vocabulary-split
output**. All ranks hold `output.weight` in the replicated layout; rank 0 (or
the last owner) broadcasts the 40 KiB hidden to N idle ranks, each computes a
16-row-tile-aligned slice with the existing cooked kernels, and the partial
argmax candidates (16 bytes each) are combined. At N=4 the 675 MB stream
becomes ~169 MB per rank and the head falls from ~2.6 ms to ~0.8-1.0 ms plus
the 40 KiB gather, i.e. **1.4-1.8 ms/token**. This is a protocol change and
should be sequenced after the direct handoff is fleet-qualified.

## Local evidence (llvmpipe, correctness only)

```
make clean && make all -j8 ... test binaries          build_exit=0, no warnings
./tests/test_qsa_prefill                              PASS
./tests/test_expert_prefill                           PASS
./tests/test_owner_reduce                             PASS
./tests/test_chat                                     PASS
./tests/test_chat_runtime                             PASS
./tests/test_fabric                                   PASS
  (new: output config/hidden frame validation, one-deep handoff matcher
   for both arrival orders + stale/duplicate/newer-token cases, and a real
   eight-process wire round trip with hidden-first ordering)
tests/test_core                                       exit=1, same
  tests/test_core.c:457 stale_reserve==FG_ERR_LIMIT
  (confirmed pre-existing: identical failure at 06cbe6c with changes stashed)
```

Host micro-benchmark (this WSL box, 20k iterations):
`work_encode_us=26.212 config_encode_us=0.006 work_decode_us=10.633`.

## Fleet validation instructions

1. Quiesce the fleet; never build on a serving blade. In a clean copy of this
   worktree: `make clean && make all -j8` (must be warning-free), then stage
   the binary and `vulkan/` shaders to all eight ranks as usual. The shader set
   is unchanged from `06cbe6c`.
2. Same ring pack and env as the `cbb3aa54` session: `FG_PREFILL_RING=1` on
   rank 0, `FG_WORKER_OWNER=1` on workers, and `FG_DECODE_DIRECT_OUTPUT` left
   unset (default on) **on every rank**. Correctness gates first: `/no_think`
   arithmetic -> `12`, capital -> `Paris`, exact token continuity over the
   standard 32-token decode.
3. Instrumented A/B (one blade per configuration is enough for the fixed
   scopes; the orchestrator's eight-blade A/B owns TPS):
   - run `FG_PREFILL_RING=1 FG_DECODE_RING_TRACE=1` for a 4K prompt and 32
     decode tokens on the new binary and on `06cbe6c`;
   - `grep RING_DECODE` both logs; on the new binary every token must carry
     `handoff=1`, and rank 7's log must carry one `RING_DECODE_HANDOFF` line
     per token;
   - compare the mean `total_ms` over tokens 2..31 (token 0/1 include cold
     start). Accept if the mean improves by >= 0.2 ms with exact token/logit
     parity; `tail_ms` now includes the output head on the direct route, so
     compare `total_ms`, not `tail_ms`/`output_ms`.
4. If a token mismatches or the run stalls: set `FG_DECODE_DIRECT_OUTPUT=0` on
   **all eight ranks** and rerun the same battery. The relay route is byte-for-
   byte the old path; a pass on the fallback with a failure on direct isolates
   the handoff without a rebuild.
5. Whole-frame gate: eight-blade 4K/32-token ring battery against the `06cbe6c`
   log; require exact response/token parity, no gate regression, and a final-20
   mean improvement consistent with the instrumented delta. Treat 0.6 ms/token
   as the upper bound and 0.25-0.3 ms as the dependable part.
6. Do not deploy from this worktree; hand the commits to the orchestrator.
