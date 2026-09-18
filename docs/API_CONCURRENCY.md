# API concurrency and HTTP/engine decoupling

Design note for the next agent. Status: **not started** - this is its own
workstream; do not bolt it onto the token loop.

## Current state (2026-09-17, commit `f1d8e8c`)

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

## Why this must become its own workstream

- Serving HTTP is orthogonal to servicing a token: framing, keep-alive,
  backpressure and multi-client fairness should not share frame time with the
  token engine.
- Scaling targets (multiple agent sessions, MCP probes, dashboards, CI) need
  concurrent connections, keep-alive, queueing, cancellation and eventually
  multi-session generation - none of which fit a single loop.
- `503 while busy` is a stopgap, not a contract to keep.

## Proposed direction

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
     `api_interrupted` (target: < 1% decode TPS delta under probe load).
   - **M2** request queue + cancellation + `429` queue-full semantics;
     single-session engine unchanged.
   - **M3** multi-session generation (ties into ring session multiplexing and
     the batched-verification path from `docs/MULTIMODAL_MTP_SCOPING.md`).
4. **Non-goals for the first pass:** TLS, auth (reverse proxy), WebSockets.

## Entry points

- `src/api.c`: `open_listener`, `handle_chat_completions`, `api_interrupted`
  (the hook to retire), the `503` path, `send_sse_headers`.
- `src/runtime.c`: `runtime_generate_tokens` (engine loop) and the interrupt
  callback contract.
- `tests/test_api.c`: mock-runtime tests - extend for queue/backpressure.

## Acceptance criteria

- No socket I/O in the generation path; `api_interrupted` is a pure client-gone
  check (or a flag set by the front-end).
- Keep-alive works; N concurrent probes during a 60K prefill leave prefill and
  decode timing within noise.
- Preserved: continuation semantics (tool/system deltas), abort/retry frontier,
  gates `[12]`/`[Paris]`, battery in band, soak PASS.
