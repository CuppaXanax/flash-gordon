# Depth-B Decode Ring Design (B=2 gated milestone)

Worktree `fg-work-depthb`, branch `feat/depth-b` off `main` `1caabc7`.
Status: **host-side core implemented and unit-tested locally; the owner/runtime
wiring listed in §8 is the first task of the fleet window** (`PLAN.md`).

## 1. Why

Decode is depth-1: `coordinator->decode_work_wire` / `decode_result_wire` are a
single `fg_layer_work` / `fg_layer_result`, so a ring step moves one token and
every rank re-reads its block weights for it. Prefill already batches through
`FG_PREFILL_FRAMES` slot arrays (`prefill_expert[]`, `prefill_layer[]`,
`ring_output[2]`).

Depth-B mirrors the prefill pattern: B independent sequences ride one ring step,
the weights are read once per step and B tokens are emitted. Reference byte
model (`PERFORMANCE_BYTE_BUDGET_2026-09-15.md`):

```
bytes/token = 6.34 GB / B + 0.36 GB + wire / B   (+ per-sequence QSA scan)
ideal speedup: B=2 ~1.9x, B=4 ~3.4x
```

The state and ring wire scale with B; the QSA record scan scales with B *and*
context, which is why the batch depth is a token-budget decision, not a
constant (§6). B=4 is out of scope for the gated milestone but the wire and the
policy are sized for it (`FG_DECODE_BATCH_MAX_SLOTS` is the single cap).

## 2. B=1 invariance

- `coordinator_decode_token` (`src/runtime.c:4574`) keeps dispatching a
  single-sequence step to the existing `coordinator_decode_token_ring` /
  `coordinator_decode_token_local` unchanged. The batch path is entered only
  when the scheduler has **at least two ready sequences**; a one-ready
  schedule returns `slot_count == 1` and the runtime does not use it.
- No new environment flag: `FG_DECODE_RING` stays the validated default /
  opt-out. Depth-B is armed by constructing the batch table with `depth > 1`
  (`fg_decode_batch_table_init`), which only the test-only harness and the
  future multi-session engine do. `tools/check-flags.sh` is untouched.
- `fg_decode_batch_choose_batch` clamps to the sequence count, so the scheduler
  can never assemble a batch larger than the work available.

## 3. Wire schema (protocol 6, message ids 51/52)

New message types `FG_MSG_DECODE_BATCH_WORK = 51` and
`FG_MSG_DECODE_BATCH_RESULT = 52`, admitted for protocol >= 6 in
`message_type_supported` (`src/protocol.c`). The single-token ids 44/45 and
their wire layout are untouched.

Work header, big-endian, 16 B:

| offset | field |
|---|---|
| 0 | layer |
| 1 | source_rank |
| 2 | destination_rank |
| 3 | flags (`FG_LAYER_WORK_HAS_NGRAM`, `FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY`) |
| 4..7 | reserved (0) |
| 8..9 | slot_count (1..2) |
| 10 | position_mode |
| 11 | axes (3 or 4) |
| 12..15 | reserved (0) |

Then `slot_count` slot records, each:

| field | bytes |
|---|---|
| token_index u32 | 4 |
| state_slot u32 | 4 |
| position[axes] u32 | 12/16 |
| hyper f32[FG_HYPER_WIDTH] | 40,960 |
| ngram f32[FG_NGRAM_EMBED_VALUES] (iff flag) | 10,240 |

Max work message: `16 + 2 * 51,224 = 102,464 B`. Result header is the same
16 B (`slot_count` at 8..9, reserved zero) followed by
`token_index u32 + hyper f32[10240]` per slot; max `16 + 2 * 40,964 = 81,944 B`.

Contract enforced by `fg_decode_batch_work_encode/decode`
(`src/protocol.c`):

- protocol < 6 is rejected (batch types are not in the v5 contract);
- exact size from header + slot_count * per-slot size, reserved bytes zero;
- `state_slot < FG_DECODE_BATCH_MAX_SLOTS` and **unique inside a message** (two
  tokens sharing an owner state namespace would corrupt state, so it is a
  format error, not a runtime check);
- `token_index < FG_MAX_CONTEXT`; text mode requires `position[3] == 0`;
- hyper and n-gram must be finite on both encode and decode;
- flags limited to the two known bits; `HAS_NGRAM` only on layer <= 1;
- caller-provided hyper/ngram storage capacities are checked.

The B=1 path never emits these ids; the batch wire is also *self-consistent*
at slot_count 1 for tests.

## 4. Per-session state

Host-side (`include/fg_decode_batch.h`): one `fg_decode_batch_sequence` per
conversation owns

- `sequence_id`, `state_generation` (bumped per committed step),
- `state_slot` — the owner-side state namespace, stable for the sequence's
  life and unique inside a batch,
- `token_index` (next decode position), M-RoPE `position[4]`,
  `committed_tokens`, `context_tokens`,
- `qsa_records[FG_LAYER_COUNT]` (per-QSA-layer record cursor),
- `sampler` / `sampler_generation`.

Owner-side (fleet wiring, §8): `gdn_state[state_slot][layer]` conv+recurrent
pairs, `ple_state[state_slot]`, and one QSA session per state slot
(namespace + cursor + record cache). Weights, arena, Vulkan context, command
infrastructure and the fabric stay singular. The batch message carries
`state_slot` per token so the worker selects the right namespace without any
per-step coordination.

Positions are copied into the batch slot and back out through the outcome, so
each sequence keeps its own M-RoPE cursor (a vision-advanced cursor survives a
batched step exactly as it does today).

## 5. Transactional step

`fg_decode_batch_step_begin/advance/commit/restore` implement the existing
`SESSION_PREPARE/COMMIT/RESTORE` semantics for a whole batch:

1. `begin`: schedule (FIFO, token budget), snapshot every slot's host state,
   then call the device `prepare` hook per slot. A prepare failure restores the
   already-prepared slots and puts every sequence back in `ready` without any
   state change.
2. run the ring step; `advance(slot, outcome)` records the sampled token,
   position, QSA cursor and sampler state per slot.
3. `commit`: validate that every slot has an outcome and that no sequence's
   generations moved during the step (fails closed on interleaving), then
   device-commit each slot. If any device commit fails, every slot is restored
   (device and host) and the sequences return to `ready` for retry.
4. `restore`: idempotent; device-restore every slot, roll the host table back
   to the snapshots, clear outcomes, set `table.restored`.

`table.restored` is the retry-frontier signal the runtime logs; `ready` is set
again so the same tokens are retried after the owner state is rolled back.

## 6. Scheduler and token budget

`fg_decode_batch_schedule` picks the oldest `ready_since` first (stable table
order on ties) and caps the batch with `fg_decode_batch_choose_batch`:

- QSA scan model per token (12 layers):
  `148 B/committed-token index scan + min(ctx/4,512) pages * 4944 B * 2
  (gather read+write) + min(ctx,2048) * 1088 B attention + 5080 B commit`,
  times `qsa_layer_count`. 94.8 MB at 4K, 553 MB at 262K — matches the byte
  budget doc.
- `FG_DECODE_BATCH_QSA_SCAN_BUDGET_BYTES` = 640 MiB per step. B shrinks while
  `B * scan(ctx) > budget`, using the **worst** context in the ready set. With
  the deployed geometry B=2 holds up to ~139.6K tokens and degrades to B=1 at
  262K, so the long-context case never swaps or stalls — it simply runs at the
  existing B=1 rate.
- `fg_decode_batch_bytes_per_token` / `_speedup` expose the model for logging
  and for the fleet gate (predicted 1.87x at B=2/4K including QSA, 1.90x on the
  weights+state+wire model alone).

## 7. Memory budget

- Extra GDN+PLE state per sequence: 36 GDN layers * (160 KiB conv + 3 MiB
  recurrent) + 360 KiB PLE ≈ 119 MB fleet-wide per extra sequence
  (≈ 6.5 MB/rank-block on the GDN owners).
- Extra QSA session per sequence: index + record cache. Rank 0's index is
  ~411 MiB at 262K; the second session is the largest single allocation. The
  rank0-mask work gave +3.43 GiB margin; the tower needs ~100 MB; the pageable
  n-gram agent frees ~3 GB/worker. B=2 fits with the mask work in place.
- No extra weights, arena, or command buffers: the ring reads the same blocks.

## 8. Fleet-window wiring (not in this local phase)

The host core is complete; the GPU/fabric half is deliberately left for the
fleet window so it can be validated against the real pack:

1. **Owner session slots** (`src/owner.c`): `gdn_state[slot][layer]`,
   `ple_state[slot]`, `qsa[slot]`, an `active_session` selector set around
   `owner_record_layer` / `fg_owner_gdn_decode` / `ple_decode_into` /
   `fg_owner_qsa_decode`; allocate slot 0 exactly as today and slot 1 only when
   the executor is created with two slots (`fg_owner_executor_create` gains a
   slot-count variant; existing callers keep 1).
2. **Session control handlers** (`src/runtime.c`): `FG_OWNER_SESSION_PREPARE`,
   `_COMMIT` and `_RESTORE` are admitted by the protocol but only `_BEGIN` /
   `_READY` are handled today. The batch step's device hooks are these
   messages: prepare snapshots the owner's per-slot GDN/PLE/QSA frontier,
   commit makes it permanent, restore rolls it back. The existing abort/retry
   frontier (`a8bc1ee`, merged) is the coordinator-side precedent.
3. **QSA namespace** (`src/qsa_owner.c`, `include/fg_qsa_owner.h`): per-slot
   `next_token[FG_LAYER_COUNT]` guards and a `session_slot` tag on
   `FG_MSG_QSA_PAGE_APPEND` / `BARRIER` / `FETCH` so a page belongs to one
   sequence's state file. Worker state paths become
   `qsa-owner-rank-%02u-s%u.state`.
4. **Runtime batch step** (`src/runtime.c`): a `coordinator_decode_batch_ring`
   that embeds B tokens (one `fg_owner_prefill_input_slot` per state slot),
   sends one `FG_MSG_DECODE_BATCH_WORK` per block owner, runs the local block
   per slot, and returns one `FG_MSG_DECODE_BATCH_RESULT`; worker
   `handle_decode_batch_work` loops the slots through the existing
   `fg_owner_decode_block*` with the slot's `active_session`. Call
   `step_begin` before the first send and `step_commit`/`step_restore` after
   the final result.
5. **Static replay**: chained blocks record against slot-0 tensors. For
   `state_slot != 0` the fleet wiring must either disable static replay or
   record per-slot static runs (`FG_VK_STATIC_SLOTS` exists).
6. **Output head / split**: the direct 4-way handoff is one token per message.
   At B>=2 either add slot arrays to `FG_MSG_OUTPUT_*` or fall back to the
   rank-0 relay; the relay is correct and only costs the existing 3.9 ms.
7. **Sampler history**: `fg_output_history` on rank 4 is a single session's
   token history (penalties). Per-session output history is required for
   isolation; until then B>=2 is limited to penalty-free sampler configs.
8. **Test-only harness**: a `flash-gordon depth-b-selftest` subcommand (CLI,
   not an env flag) that runs two canned conversations sequentially at B=1 and
   interleaved at B=2 and compares token ids, logits bits and per-session state
   digests. This is the executable B=2 parity gate.

## 9. Files in this phase

- `include/fg_protocol.h`, `src/protocol.c`: batch wire schema + validation.
- `include/fg_decode_batch.h`, `src/decode_batch.c`: table, scheduler, token
  budget, transactional step, device hooks.
- `tests/test_decode_batch.c`: policy, scheduler/fairness, transaction
  rollback/isolation, wire round-trip and rejection, batched-vs-sequential
  parity and failure rollback simulations.
- `Makefile`: `test_decode_batch` target wired into `make test`.
- `PLAN.md`: fleet validation steps, gates and expected numbers.
