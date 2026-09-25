# API concurrency and HTTP/engine decoupling

Design note for the front-end/engine split. Status: **M2 landed** (2026-09-24,
branch `feat/api-m2-admission`); M1 landed as `ee25e7e` (2026-09-24, branch
`feat/api-m1-frontend`, deployed to the 8-blade fleet as live dir
`20260924-api-m1`); **M3.1 landed** on `feat/batch-block-r1` (`08448c0` +
`723be95`: per-session runtime state, live session table, public `session_id` /
`X-Flash-Gordon-Session`, fleet-validated); M3.2 (two-session decode) is in
progress - runner primitives on the branch, blockers in
`docs/M3_SESSION_MULTIPLEXING.md` section 11.

## M2: bounded FIFO admission on the engine queue (landed)

M1's transport ring (`api_engine_queue`, capacity 4) already carried request
ownership; M2 turns the admission policy from "reject while busy" into a
bounded FIFO queue. `handle_chat_completions`, the engine loop and the
single-session contract are unchanged.

- **Admission**: `api_engine_queue_try_push` returns
  `API_ENGINE_ADMIT_QUEUED | _FULL | _STOPPING`. A chat request is admitted
  while a generation is running and waits in the ring; the engine services it
  FIFO via the existing `pop_wait` loop.
- **Bound**: `FG_API_ENGINE_QUEUE_CAPACITY = 4` counts every admitted request
  (running + queued + not yet completed), not just ring occupancy. The next
  request is answered `429 Too Many Requests` with `Retry-After: 1` and body
  `{"error":{"message":...,"type":"queue_full"}}`. `503 + server_busy` is now
  only the shutting-down path.
- **Cancellation**: the front-end tracks `client_gone`; the reap sweep calls
  `api_engine_queue_cancel` for a queued connection that disconnected before
  the engine picked it up - the ring entry is compacted out, its body freed
  and its admission slot released, and the connection is completed so the
  reaper closes it. The engine pop path re-checks `client_gone` for the race
  and drops the request without spending a token slot. The abort/retry
  frontier path is untouched (a *running* generation still aborts through
  `api_interrupted`).
- **Probes/keep-alive/heartbeats**: `/health` `busy` is `outstanding != 0`
  (anything admitted), so it reports `busy:true` while requests queue; probes,
  keep-alive and the 10 s heartbeats are unchanged.

### M2 tests (`tests/test_api.c`)

- `test_frontend_engine_split`: second chat while busy is admitted, served in
  order (200 on the queued connection), the admission bound answers `429` +
  `Retry-After: 1` + `queue_full`, keep-alive/probes/heartbeat behavior
  unchanged.
- `test_engine_queue_admission`: FIFO order over the full ring, `FULL` at the
  bound, slot reuse after completion, `STOPPING` after stop.
- `test_engine_queue_cancel`: cancellation compacts the ring, survivors keep
  FIFO order, double-cancel is a no-op.
- `test_frontend_cancels_queued_client`: end-to-end - three queued clients,
  the middle one disconnects; the front-end sweep releases its slot and the
  survivors are served in order.

## M1: dedicated HTTP front-end thread (landed)

### Threads and transport

- The **front-end thread** (`api_frontend_thread`) owns the listener and every
  client socket: accept, non-blocking read, incremental `api_http_parser_try`,
  routing, response buffering/flush, keep-alive and the idle/limits policy.
  It is a single `poll()` loop (listener + wake pipe + one pollfd per
  connection), so a slow client, a stalled body or a probe burst cannot stall
  another connection.
- The **engine thread** (the `fg_api_main_with_options` loop) opens the
  runtime, then blocks on `api_engine_queue_pop_wait` and runs exactly one
  `handle_chat_completions` per request. It never touches the listener.
- **Transport**: a mutex + condvar queue (`api_engine_queue`, capacity 4).
  M1 keeps the documented single-slot contract: the front-end rejects a chat
  request immediately while one is outstanding (`503` + `Retry-After: 1`) and
  only enqueues when the engine is idle. M2 will relax this to FIFO queueing
  with a bounded queue and `429` on overflow.
- **Abort**: the front-end publishes disconnect through the connection's
  atomic `client_gone` flag (POLLRDHUP/POLLHUP or a zero-length peek);
  `api_interrupted` is now only `api_stop_requested || client_gone` at the
  existing ring-prefill/decode boundaries. The abort/retry frontier path is
  untouched.
- **Heartbeats**: the front-end timer emits a chunk-framed `: keep-alive`
  comment every 10 s when a streaming response has nothing queued. This runs
  in the poll loop, so it covers long prefills and the whole vision tower; no
  socket I/O remains in the token path.
- **Keep-alive**: HTTP/1.1 persistent connections are the default.
  Content-Length responses (probes, non-streaming chat) are reusable as-is;
  streaming uses `Transfer-Encoding: chunked` and a terminating `0\r\n\r\n`
  chunk. `Connection: close` is honored; a malformed request gets `400` and
  the connection closes. Idle connections are reaped after 120 s.
- **Bounds**: per-connection outbound buffer capped at 8 MiB (a client that
  does not drain is marked gone and the generation aborts), 64 concurrent
  connections, 256 KiB header cap and the existing 32 MiB request cap.
  `configure_client_socket` now puts clients in non-blocking mode (the old
  blocking SO_RCVTIMEO/SO_SNDTIMEO pair is retained).

### Files

- `src/api.c`: `api_sink` abstraction; `api_http_parser_try` /
  `api_chunked_decode` incremental parser; `api_connection` output buffers
  (`api_connection_append/flush/complete_response`); `api_frontend_*`;
  `api_engine_queue_*`. `read_http_request` survives only for the unit-test
  build (`-DFG_API_TEST_BUILD`).
- `tests/test_api.c`: `test_frontend_engine_split` (probes + immediate 503
  while the engine is busy, chunked SSE, sequential keep-alive requests,
  `Connection: close`, malformed 400), `test_incremental_parser`
  (byte-by-byte, pipelined, chunked), heartbeat/client-gone unit tests.
- `tools/api-probe-during-generation.ps1`: long text or 512px vision turn
  while probing `/health` + `/v1/models` and one busy chat; prints probe
  latency stats.
- `tools/api-concurrent-probe.ps1`: same generation with/without 4 parallel
  probe workers; prints engine prefill/decode TPS delta.
- `tools/api-keepalive.ps1`: raw-TCP sequential requests on one connection.
- `tools/api-sse-keepalive.ps1`: raw SSE capture during a ~26K-token prefill;
  reports comment count and spacing.
- `tools/api-abort-retry.ps1`: hard client disconnect mid-prefill -> time to
  `busy:false` and the retry prefix-cache headers.

### Measured evidence (2026-09-24)

Fleet: all 8 blades, `20260924-final` (before, bin `7e836292df17efd9254dec269dfc4d10`)
-> `20260924-api-m1` (after, bin `ab79863f635b57b9089a46a3c48a61bd`), same
production config. Full record:
`bc-250-dbg/results/api-m1-20260924-0125/EVIDENCE.md`.

| probe | before | after |
|---|---|---|
| `/health` during 8.7K-token prefill+decode (59-136 samples) | max **2023.8 ms**, p95 1019.2 ms | max **6.7 ms**, p95 2.3 ms |
| busy chat `503` during a 512px vision turn | **2630.9 ms** | **0.7 ms** |
| second request on one kept-alive connection | 5 s timeout (server closed) | 200 on the same socket in 21.7 ms |
| `: keep-alive` during a 46 s streaming prefill | 2 comments (token-loop hook) | 4 comments, 10.00 s spacing |
| abort mid-prefill -> `busy:false` | 2.0 s | 2.2 s |
| retry after abort | prefix hit, reused 1664, prefilled 7004 | prefix hit, reused 1664, prefilled 7004 |
| decode TPS under 4 concurrent probe streams | n/a | +0.32% vs control (max probe 9.5 ms) |

M1 acceptance criteria from the original note are met.

## Original design note

Design note for the next agent. Status: **not started** - this is its own
workstream; do not bolt it onto the token loop.

### Current state (2026-09-17, commit `f1d8e8c`, historical)

- One process, one thread: a single accept loop (`src/api.c`, `open_listener`)
  and one in-flight generation.
- HTTP socket work is serviced **inside the generation loop** through the
  `api_interrupted` hook, which runs at ring-prefill chunk issues, legacy
  prefill microbatch boundaries and decode tokens (added in `cd1f629`). That
  hook currently: detects client-gone, emits `: keep-alive` comments when the
  stream is idle > 10 s, and drains the listener to answer `/v1/models` +
  `/health` or `503 + Retry-After: 1`.
- Responses use `Connection: close`; chunked request bodies are accepted;
  non-streaming requests receive no early headers.
- **The defect:** HTTP I/O is coupled to token cadence. Socket work shares rank
  0's frame time with the expert stream, HTTP latency is bounded below by the
  chunk/token interval, and a slow client or probe burst can perturb decode.

### Why this must become its own workstream

- Serving HTTP is orthogonal to servicing a token: framing, keep-alive,
  backpressure and multi-client fairness should not share frame time with the
  token engine.
- Scaling targets (multiple agent sessions, MCP probes, dashboards, CI) need
  concurrent connections, keep-alive, queueing, cancellation and eventually
  multi-session generation - none of which fit a single loop.
- `503 while busy` is a stopgap, not a contract to keep.

### Proposed direction

1. **Front-end / engine split.**
   - A dedicated HTTP front-end thread (pthreads are already linked; a single
     `poll()`-based event loop is enough - keep the hand-rolled parser, no
     external HTTP libraries).
   - Front-end owns: accept, keep-alive, multiple connections, per-client
     outbound buffering/backpressure, heartbeats (a front-end timer, not a
     token-loop hook), cancellation propagation.
   - Engine owns: request consumption, generation, cancellation checks at the
     existing boundaries (the abort/retry frontier path stays as-is).
   - Transport: SPSC/MPSC queue + condvar (or a pipe for wakeups); one request
     in flight today, an explicit queue for the future.
2. **Contracts.**
   - Backpressure: bounded per-client buffers; close slow clients without
     stalling tokens.
   - Cancellation: front-end `client_gone` -> engine abort (current semantics).
   - Queue: FIFO; `429` only when the queue bound is exceeded (documented),
     never merely because a generation is running.
   - Streaming: engine produces SSE frames; front-end frames/forwards them.
     Heartbeats are front-end-side only.
3. **Milestones.**
   - **M1** front-end thread + keep-alive + concurrent `/v1/models` `/health`
     with zero token-loop coupling; remove the listener work from
     `api_interrupted` (target: < 1% decode TPS delta under probe load). DONE.
   - **M2** request queue + cancellation + `429` queue-full semantics;
     single-session engine unchanged.
   - **M3** multi-session generation (ties into ring session multiplexing and
     the batched-verification path from `docs/MULTIMODAL_MTP_SCOPING.md`).
4. **Non-goals for the first pass:** TLS, auth (reverse proxy), WebSockets.

### Entry points

- `src/api.c`: `open_listener`, `handle_chat_completions`, `api_interrupted`
  (retired to the abort check), the `503` path, `api_send_sse_headers`.
- `src/runtime.c`: `runtime_generate_tokens` (engine loop) and the interrupt
  callback contract.
- `tests/test_api.c`: mock-runtime tests - extended for keep-alive, the
  front-end/engine split and the parser.

### Acceptance criteria

- No socket I/O in the generation path; `api_interrupted` is a pure client-gone
  check (or a flag set by the front-end). DONE.
- Keep-alive works; N concurrent probes during a 60K prefill leave prefill and
  decode timing within noise. DONE (4 concurrent probe streams, +0.32% decode).
- Preserved: continuation semantics (tool/system deltas), abort/retry frontier,
  gates `[12]`/`[Paris]`, battery in band, soak PASS. DONE.

## What M3 still needs

- **M3**: multi-session generation. The engine loop and `api_public_session`
  are still single-threaded; ring session multiplexing and the depth-B batch
  path (`src/decode_batch.c`, `docs/DEPTH_B_DECODE_RING.md`,
  `docs/BATCHED_DECODE_BLOCK.md`) are the upstream pieces. The front-end
  transport was built so request ownership (connection + body) moves through
  the queue, which is the prerequisite for multiple in-flight sessions; M2
  now admits more than one request and the engine still runs them one at a
  time.
- The M3 plan (session lifecycle, admission/scheduling on the owner slots,
  abort/isolation, state/QSA/prefix interactions, failure semantics, test
  plan) is `docs/M3_SESSION_MULTIPLEXING.md`; the depth-B batch step it builds
  on measured 74.6 ms/step for two sequences after the ring overhead cuts
  (`bc-250-dbg/results/overheads-20260924-2310/EVIDENCE.md`).
