# M3: session multiplexing (concurrent generation)

Design note for the multi-session engine milestone.  Status: **M3.1 landed**
(branch `feat/batch-block-r1`, commits `08448c0` + `723be95`, fleet-validated
2026-09-24); **M3.2 implemented and partially gated** - both blockers are
resolved (section 11), the engine scheduler and the `--concurrent` selftest are
on the branch, and the HTTP two-client/abort gates pass; the full promotion
gate set (depth-B + soak + battery + heartbeats on the final binary) is not yet
re-run, so it stays fail-closed.  It builds on M1 (front-end/engine split), M2
(bounded FIFO admission on `api_engine_queue`) and the depth-B ring decode
(`src/decode_batch.c`, `docs/DEPTH_B_DECODE_RING.md`,
`docs/BATCHED_DECODE_BLOCK.md`) that reached parity and measured 74.6 ms/step
for two sequences in the round that wrote this note.

Related: `docs/API_CONCURRENCY.md` (M1/M2, "What M3 still needs"),
`docs/BATCHED_DECODE_BLOCK.md` §10, `src/runtime.c`
(`handle_decode_batch_work`, `coordinator_decode_batch_step`,
`runtime_generate_tokens`), `src/owner.c` (owner session slots), `src/api.c`
(`api_public_session`, `api_engine_queue`).

## 1. Goal and non-goals

**Goal.** N concurrent chat sessions generate at the same time on one ring:
requests are admitted through the M2 queue, each session owns its transcript,
sampler, positions and state, and the engine pipeline (prefill, decode) is
shared across sessions by the depth-B batch step.  The public API keeps the
single-session contract and adds an explicit session identity for continuation.

**Non-goals (M3 phase 1).** Speculative/MTP decode, tensor-pipeline changes,
TLS/auth, WebSockets, per-session RNG streams that survive process restart, and
any change to the B=1 serving path.  A session's request is still answered by
exactly one engine worker (the existing engine thread); M3 does not add engine
threads.

**Hard constraints.**

- One ring: weights, arenas, fabric and the QSA page transport stay singular.
  Concurrency comes from slot batching, not from parallel rings.
- Owner state slots are the admission currency.  `FG_OWNER_SESSION_MAX` (= 2u)
  sets how many sequences can hold device state at once; everything else waits.
- B=1 stays on the existing single-token ring path; B>=2 uses the batch table.
- Every device state change remains inside a PREPARE/COMMIT/RESTORE
  transaction; a failed step must leave every participating session exactly at
  its pre-step frontier.
- No new env flags; concurrency is a serving property, not a knob.

## 2. Where M3 plugs in

```
front-end thread            engine thread                    ring (rank 0 + 7 owners)
----------------            -------------                    ------------------------
listener/parser   --queue--> session scheduler              prefill pipeline
api_engine_request           |  admission -> session          ring decode (B=1 direct)
                             |  prefill placement             depth-B batch step
                             |  decode step scheduling         owner state slots
                             |  output/abort/cancel
```

- `api_engine_queue` (M2) is unchanged: it is the *admission* queue, bounded at
  4 requests, FIFO, cancel-aware, 429 on overflow.
- `api_public_session` (M1) is currently one value embedded in the engine loop.
  M3 promotes it to a table of session objects.
- `fg_runtime` is currently the engine's whole state (history, frontier, next
  token, sampler).  M3 gives each session its own runtime-side state and keeps
  the ring-facing coordinator shared.
- The depth-B table/scheduler (`fg_decode_batch_*`) already implements the
  needed scheduling primitives: sequence enter/leave, readiness, oldest-ready
  first, QSA-budget batch choice, snapshot/advance/commit/restore.  M3 is the
  policy and lifecycle layer above it.

## 3. Session lifecycle

A session is created by the engine when it pops a request that does not name an
existing session, and destroyed when it finishes, aborts, is evicted, or the
engine stops.

| state | holds | leaves when |
|---|---|---|
| `PENDING` | queue entry + body, no owner slot | scheduler admits it and a state slot is free |
| `PREFILL` | owner state slot, QSA namespace, prefill cursor | prompt + first sample committed |
| `DECODE` | owner state slot registered in the batch table (`ready`) | EOS / max_tokens / abort |
| `DRAINING` | owner slot until the step that contains it commits | step commit/restore completes |
| `DONE` | nothing (response fully enqueued) | immediately |
| `FAILED` | nothing (transport may need reopen) | session freed |

Creation allocates: a session id (monotonic `uint64`, also the ring
`session_id`/request id), the transcript/rendered state (moved out of the
shared `api_public_session`), a sampler state seeded from the request, a state
slot (when admitted), and a QSA namespace (`fg_owner_qsa_open_*_slot`).  The
engine keeps `FG_SESSION_MAX` entries (suggested 8, matching
`FG_DECODE_BATCH_SEQUENCE_MAX`); beyond that the front-end queue bound (4)
already rejects with 429.

Sessions that are complete but whose client has gone are reaped by the
front-end sweep exactly like M2's `client_gone` handling; the engine owns the
slot release.

### 3.1 Public session identity

- Default: one session per connection turn.  A keep-alive connection that sends
  a second chat request continues the *same* session (and its prefix) when the
  request is a strict prefix extension - this is today's M1 continuation
  behaviour and must not regress.
- Explicit: request field `session_id` (opaque string; the server echoes
  `X-Flash-Gordon-Session`).  An unknown id creates a new session.  Tool
  continuations and agent frameworks use it to address a conversation across
  connections.
- Isolation: the session id never reaches the ring; the ring sees only the
  internal numeric `session_id` of the active step.

## 4. Admission and scheduling policy

### 4.1 Which sessions run

- **Concurrency bound** `active = min(FG_OWNER_SESSION_MAX, FG_DECODE_BATCH_MAX)`
  (`= 2` today).  Additional admitted requests wait in `PENDING` in FIFO order;
  they are not rejected while the M2 queue has room.
- **Slot grant**: when a slot frees, the oldest `PENDING` session is admitted,
  its state slot is prepared (owner `SESSION_PREPARE` for that slot), and its
  prefill is placed on the ring.
- **Prefill placement**: the prefill pipeline is exclusive and long.  M3 runs
  it chunk-aware: a prefill session owns the ring for one
  `prefill_microbatch` chunk at a time, then yields so that ready `DECODE`
  sessions can take a batch step.  This bounds decode inter-token latency by
  one prefill chunk instead of a whole prompt and makes a long prompt unable to
  starve the other session.
- **Decode placement**: at each engine scheduler pass, every `DECODE` session
  not in flight is marked ready; the depth-B scheduler picks the oldest-ready
  sessions up to `fg_decode_batch_choose_batch(policy, worst_context, depth)`.
  The policy already degrades B=2 to B=1 as the QSA scan budget is consumed
  (640 MiB: B=2 at short context, B=1 by 262K), so a long-context session
  shrinks the batch rather than exceeding the budget.

### 4.2 Timeouts and fairness

| condition | policy |
|---|---|
| queue wait | bounded by the M2 queue (4); 429 + `Retry-After: 1` on overflow |
| prefill chunk | existing ring timeout (split deadline + grace); on expiry the chunk is rolled back and the session fails |
| decode step | one step deadline (existing direct-handoff timeout as the model); on expiry RESTORE every slot and retry once, then fail the sessions involved |
| a `PENDING` session | wait bound (suggested 10 min) -> 503 `server_slow`; keeps a dead client from pinning a slot |
| idle `DECODE` prompt (client still waiting) | no timeout; SSE heartbeats are front-end-side (M1) |
| idle session with a gone client | front-end sweep -> engine cancel -> slot released at the next safe point |
| fairness | oldest-ready first (`ready_since`) at both the slot grant and the batch schedule; no priority inversion, no starvation: a ready session can wait at most one in-flight step + one prefill chunk |

## 5. Abort and isolation between sessions

- **Per-session cancel.**  `client_gone` remains a per-connection atomic; the
  batch step's assembly/sample/send points check the flags for the sessions in
  the step.  A slot whose client is gone is dropped from the *next* schedule and
  its owner state is RESTOREd at the next safe point (it never shares a
  transaction with live sessions beyond the step it was already in).
- **Mid-step abort.**  If a session aborts between PREPARE and COMMIT, the step
  completes for the remaining slots and then a per-slot `SESSION_RESTORE` rolls
  the aborted session back to its snapshot.  If the engine must abandon the
  step itself, the existing `fg_decode_batch_step_restore` restores every slot;
  the aborted session is then left/freed without committing a token.
- **Fault isolation.**  A ring/owner failure inside a step is a *shared*
  failure: RESTORE all slots and fail the transport (`transport_poison`), which
  M1/M2 already treat as engine-fatal (runtime reopen).  A session-local failure
  (bad sampler draw, context limit, client protocol error) fails only that
  session and leaves the other slot committed.
- **Output isolation.**  Each session keeps its own `api_generation`, response
  buffer and cancellation state; the front-end connection owns the sink, so a
  slow client aborts only its session (M1 backpressure contract).  Responses
  are written by the engine as tokens are emitted; a batch step produces one
  token per session and each is fanned out to its own sink.
- **Transaction invariants.**  State generations (`state_generation`,
  `sampler_generation`) are checked at commit; an interleaved mutation (e.g. a
  prefix-continuation that re-prefills a session while a step is in flight)
  fails closed instead of committing stale state.

## 6. State, QSA and prefix continuation

- **Owner state.**  One state slot per active sequence (`FG_OWNER_SESSION_MAX`
  slots).  The slot holds GDN conv/recurrent, PLE and the QSA session
  namespace.  Slots are created at admission (`fg_owner_qsa_open_*_slot`),
  reset/re-opened when a session restarts from a cold prefix, and released when
  the session is destroyed.
- **QSA.**  Depth-B stays ring-only: a session's QSA namespace and page stream
  are per slot, and the scan budget (not the page cache) is the batch limiter.
  The QSA page transport stays on the engine thread; page fetches for a session
  are only issued while that session owns the ring (prefill chunk or its slot
  in a decode step), so no two sessions can interleave page state.
- **Prefix continuation.**  `runtime_generate_tokens`' prefix plan becomes
  per-session: each session carries `history`, `state_frontier`,
  `state_position`, `next_token(_valid)` and the rendered transcript.  A
  continuation request on the same session proposes its own prefix; the plan
  either resumes from the session's frontier (`prefill_offset ==
  state_frontier`) or cold-resets that session's slot only.  A different
  session's frontier is never consulted, and the ring-wide `runtime_reset_state`
  is split into a per-slot reset plus a session-BEGIN barrier for the session
  being (re)started.
- **Rendered/tool state.**  `api_public_session`'s transcript and media
  identities move per session; boundary/EOS pending state is per session and
  flushed when that session's response completes.

## 7. Failure and rollback semantics

| failure | scope | action |
|---|---|---|
| sampler/context/format error | session | fail that session with 4xx/5xx; slot restored at the next safe point |
| client gone mid-decode | session | cancel; restore slot; no token committed |
| step timeout | step | RESTORE all slots, retry once (fresh snapshot), then fail the sessions |
| owner/fabric error | engine | RESTORE all slots, poison transport, fail every active session, require runtime reopen |
| prefill chunk error | session | roll back the chunk; fail the session; other sessions keep serving |
| engine shutdown (`SIGTERM`) | engine | stop admission, finish or abort in-flight sessions, restore slots, close fabric |

All rollbacks reuse `fg_decode_batch_step_restore` and
`fg_owner_session_rollback`; no new state-write path is introduced.  A session
that cannot be restored (owner reports a mismatch) fails the step and the
transport - the fail-closed rule from M2.

## 8. Milestones

1. **M3.1 - session objects (no batching).**  Split the shared
   `api_public_session` and `fg_runtime` token-path state into per-session
   objects; engine loops over sessions but still runs one at a time.  Public
   `session_id` + `X-Flash-Gordon-Session`.  Gate: single-session byte
   identity, M2 queue tests, `correctness64` `[12]`/`[Paris]`.
2. **M3.2 - two-session decode.**  Engine scheduler above the depth-B table:
   two decode sessions share batch steps, prefills chunk-yield.  Gate: depth-B
   selftest unchanged, plus a new two-client soak where both sessions decode
   concurrently; per-session parity against two separate runs.
3. **M3.3 - lifecycle and fault hardening.**  Timeouts, per-session abort,
   slot release/rebind, continuation across requests, soak with aborts and a
   slow client.  Gate: soak PASS with 2 sessions + abort churn, `pi-stability`
   PASS, battery, no regression in single-session TPS.
4. **M3.4 (optional) - depth beyond the owner slots.**  Time-slicing more than
   two sessions across two slots (a session swap costs a RESTORE + re-prefill
   of the frontier delta), or more owner slots.  Only if measurements show it
   pays.

## 9. Test plan

Unit (host, no fleet):

- `tests/test_decode_batch.c`: extend with a two-session scheduler model -
  interleaved readiness, oldest-first ordering, batch shrink as context grows,
  slot reuse after leave, commit/restore with one slot failing.
- New `tests/test_session_scheduler.c` (or `test_api.c` additions): session
  table create/continue/free, admission when slots are full, per-session cancel
  propagation, public `session_id` mapping, response fan-out ordering.
- Prefix: per-session continuation plans do not read another session's
  frontier; a cold reset of one session leaves the other's slot untouched.

Fleet gates:

- `depth-b-selftest --depth 2` parity/rollback (already green; must stay
  green).
- New `--concurrent` selftest mode: two scripted two-session workloads (same
  prompts as the depth-B selftest) driven through the public API with
  `session_id`; assert per-session tokens/logits byte-identical to the serial
  single-session runs, assert total wall time close to the batched estimate.
- `tools/api-concurrent-probe.ps1` style probe while both sessions decode:
  `/health` max latency and heartbeat spacing must stay in the M1 envelope.
- Abort churn: kill one client mid-decode, assert the other session's tokens
  are unchanged and the freed slot is reused by the next request.
- Soak: `tools/pi-stability.ps1` on a single session (unchanged) plus a
  two-session soak script; ledger `blocks/wire_hops/weights_total/logical`
  unchanged; 32K band intact with B=1 shrink.
- Battery + `tools/check-flags.sh` (no new env flags).

## 10. Risks and open questions

1. **Ring sharing is the wall.**  One ring step is ~65 ms of the 74.6 ms
   two-session step; multiplexing does not add throughput from thin air, it
   amortizes the weight read.  Aggregate is bounded by the batch policy and by
   the two head passes on rank 4 (~2.8 ms each).  If per-session latency
   matters more than aggregate, batch only when both sessions are ready.
2. **Prefill/decode interleaving.**  Chunk-yield needs the prefill pipeline to
   stop at a microbatch boundary without tearing down the session's QSA page
   stream; today `coordinator_prefill_pipeline` is written for one session at a
   time.  This is the largest code change in M3.
3. **Prefix continuation semantics across sessions.**  The current plan is
   global (one history); M3.1 must move it without changing the exact-prefix
   rules (`plan.prefill_offset == state_frontier`) that the gates pin.
4. **Cancellation races.**  M2 closes the queued-cancel race; M3 adds
   in-flight cancel, where a slot may be mid-transaction when its client
   disappears.  The rule above (complete the step, restore after) is simple but
   must be tested against the `restored` table flag.
5. **Slot/namespace leaks.**  A failed admit-after-prepare must not leak a QSA
   namespace or GDN state; the depth-B prepare-failure path already rolls back
   earlier slots, and M3 must extend the same pattern to session creation.
6. **Determinism.**  Two sessions' tokens must not depend on scheduling order
   (the sampler state is per session; the batch math is per token).  The
   depth-B selftest is the oracle; the concurrent mode must reproduce it
   exactly, which is also the strongest isolation test.

## 11. M3.1 status (landed) and M3.2 blockers (2026-09-24)

### M3.1 - landed and fleet-validated

Commits `08448c0` (session objects + public identity) and `723be95` (bootstrap
capture stays live; cold starts always BEGIN) on `feat/batch-block-r1`.
`fg_runtime_session` owns the token-path state (history, frontier, position,
sampler, rendered transcript); `api_session_table` resolves a request by
explicit `session_id`, by its connection (keep-alive default), or by strict
prefix extension of the most recent session; the server echoes
`X-Flash-Gordon-Session`.  The first session adopts the state `fg_runtime_open`
created, so the single-session path is byte-identical; a session whose ring
generation was reset by another session cold-starts instead of resuming a stale
frontier.  Fleet gates (bin `f049a224a72db397c031c1d4`):
byte identity `sha256(d7b851e9...)`, `correctness64` `[12]`/`[Paris]`,
`pi-stability` PASS (4k 323.11 / short 26.56 / 32k 23.73 vs main
323.08 / 26.93 / 24.00), battery + `check-flags` PASS, depth-B selftest PASS
(batch-2 decode 78.7 ms/step in that window, aggregate 25.4 tok/s).

### M3.2 - blockers resolved, gates in progress (2026-09-24 round 2)

Both structural blockers now have code and fleet evidence
(`bc-250-dbg/results/m3b-20260924-2314/EVIDENCE.md`):

1. **Per-slot cold start** - `fg_owner_reset_session_slot` (`src/owner.c`) plus
   the `FG_OWNER_SESSION_RESET/RESETTED` owner transaction carry a slot-scoped
   reset to all ranks; `runtime_reset_slot` (`src/runtime.c`) is the cold start
   (RESET + the existing PREPARE lazy QSA mirror open, no ring-wide BEGIN) and
   each live session reserves an owner slot.  The `--concurrent` selftest shows
   session X's slot-0 digest identical before and after session Y's cold start
   (`a0793599ac7b27c4`).
2. **API park/pump** - `handle_chat_completions` is now
   `api_turn_open`/`api_turn_run_legacy`/`api_turn_close`, and `api_scheduler_run`
   (at most two live turns, chunk-yield, depth-B batch, B=1 runner step for a
   lone session, client-gone abort with session drop) replaces the synchronous
   engine loop.  The runtime binding is taken per scheduler step, and worker
   owner-session selection rides `SESSION_SELECT/SELECTED` before each direct
   decode/prefill call.
3. **First-cut eligibility** - text-only, non-continuation, penalty-free,
   `temperature == 0`, identical sampler configs; everything else stays on the
   synchronous legacy path and runs strictly solo.
4. **Prefill chunking** - validated: the 2K-token chunk-yielded prefill
   reproduces the serial prefill's tokens and logits exactly.

Gates on the final binary still owed before promotion: depth-B selftest,
`pi-stability` soak bands, fleet battery, and the SSE heartbeat probe.

<details><summary>Original blocker list (historical)</summary>

### M3.2 - primitives on the branch, not wired; exact blockers

The incremental runner (`src/runtime.c`): `fg_runtime_session_runner_begin`
(plan + prompt/history setup), `runner_prefill` (one
`coordinator_prefill_pipeline` sub-range per call, final output head on the last
chunk), `runner_resume_decode` (continue a clean decode yield), `runner_batch`
(emit pending tokens, one `coordinator_decode_batch_step`, table bookkeeping
including EOS/max leave), `runner_finish` (rendered transcript/frontier/stats).
It mirrors the production phase order and is not called from the serving path
yet; it must be validated with a `--concurrent` selftest mode before wiring.

Blockers, in the order they must be solved:

1. **Per-slot cold start (the hard one).**  `runtime_reset_state` is ring-wide:
   `fg_owner_reset_state` zeroes every owner session slot and
   `coordinator_begin_session` recreates the single QSA namespace, so admitting
   a second session would wipe the first session's slot-0 state.  The depth-B
   selftest only ever runs both slots inside one BEGIN epoch.  Required:
   `fg_owner_reset_session_slot(executor, slot)` (zero that slot's GDN
   conv/recurrent + PLE, reset its QSA session and recreate its mirror file)
   and a slot-scoped cold start in the runner that skips BEGIN and uses
   `coordinator_owner_transaction(PREPARE, slot)` (whose lazy slot!=0 mirror
   open already exists in `depthb_owner_prepare`).  Session A keeps slot 0, the
   admitted session gets slot 1; `coordinator->output_session` and the batch
   table's `state_slot` follow the session.
2. **API engine refactor.**  `handle_chat_completions` is synchronous: it owns
   the request body, rendered transcript, SSE state and stats until the
   response is finished.  A queue-triggered yield must park that state
   (`api_parked_request`) without finalising the response, and the engine needs
   a scheduler pump that alternates prefill chunks and batch steps, finishes
   each parked request through the existing tail, and keeps the solo path
   untouched (park only when the queue has work, so a lone session never leaves
   the production `fg_runtime_generate*` path).
3. **First-cut eligibility.**  The batch step requires a penalty-free sampler
   and one shared `coordinator->sampler` config; media, tool continuations and
   prefix continuations stay serial until M3.3.
4. **Prefill chunking validation.**  `runner_prefill` passes the full prompt as
   the history and one frame-group sub-range per call; it needs the
   `--concurrent` selftest parity check against the serial prefill before the
   API path uses it.
5. **M3.3 hardening** (abort churn, slow-client backpressure, slot rebind on
   session replacement) as designed in sections 3-7.

Until 1 and 2 are done and the two-session gates in section 9 pass, M3.2 stays
fail-closed on `feat/batch-block-r1`; production remains `20260924-teardown`
(main `c238858`).

</details>
