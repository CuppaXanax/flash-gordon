# PLAN: Depth-B decode ring (B=2) fleet validation

Local phase status: host core implemented and green in `fg-work-depthb`
(`feat/depth-b`, off `main` `1caabc7`). No fleet contact from this phase.
The orchestrator owns the window; this plan is the request.

## 0. Window preconditions

- Fleet at 192.0.2.42-49, deployed pack, `bc-250-dbg/Invoke-BC250Fleet.ps1`
  runner and `.env` auth, as used by `tools/start-smoke-chat.ps1`.
- Rank0 mask work (branch `fix/rank0-coordinator-mask`) landed: +3.43 GiB
  margin. Tower (~100 MB) and the pageable n-gram agent (~3 GB/worker freed)
  are in the same build.
- The window build includes the §8 wiring from
  `docs/DEPTH_B_DECODE_RING.md`: owner session slots, QSA per-slot namespace
  and page tags, runtime/worker batch step, output/sampler per-session
  handling, and the `depth-b-selftest` CLI harness.
- `make test` green on the build host (flag gate first; `test_decode_batch`
  included). `make test-vulkan` unchanged.
- Expected window: one build+deploy, one attach battery, one B=2 parity run,
  one long-context memory run, one performance run. Two sessions if a fix is
  needed.

## 1. Phase 0 - wiring acceptance (before any gate)

| # | Task | Acceptance |
|---|---|---|
| 0.1 | Owner session slots (`src/owner.c`) | `fg_owner_executor_create_slots(...,2)` allocates slot 1 GDN/PLE/QSA; existing callers allocate slot 0 only; `FG_GDN_DIAG` digests for slot 0 unchanged vs baseline |
| 0.2 | Owner PREPARE/COMMIT/RESTORE handlers + QSA per-slot namespace | session-control ops snapshot/commit/roll back the per-slot GDN/PLE/QSA frontier; two state files per worker rank (`...-s0.state`, `...-s1.state`); page append/barrier/fetch carry the session tag; per-slot guard frontiers |
| 0.3 | Runtime batch step | `depth-b-selftest` runs; batch messages observed in `FG_FRAME_TRACE` only at depth 2; no batch id at depth 1 |
| 0.4 | Static replay | disabled for `state_slot != 0` (or per-slot static runs); `FG_DECODE_STATIC=1` default still valid for slot 0 |
| 0.5 | Output/sampler | penalty-free sampler enforced at B>=2 until per-session output history lands; relay path (no direct split) used for B>=2 |

## 2. Phase 1 - B=1 no-regression (must be byte-identical)

Same binary, depth 1 (default), `FG_DECODE_RING` unset.

1. `make test` (flag gate + core suites).
2. Attach battery on the frozen baseline prompts:
   - `/no_think` arithmetic -> `12`, capital -> `Paris`.
   - 128-token prefill, 4K prefill, short decode, 4K decode (1 and 32
     tokens), prefix continuation.
3. Record for the B=1 run: greedy token-id sequence, per-token logit bit
   patterns, rendered output bytes, and the `FG_NUMERICS_TRACE` per-layer
   hyper-state digests (`FB_IN`/`FB_OUT`, `STATE_*`) for one fixed seed.

**Gate 1:** all of the above byte-identical to the baseline binary on the same
pack (logit comparison is exact f32 bits, not a tolerance); prefill TPS within
noise (>= 230 TPS at 4K, short-decode/4K-decode within +/-2%); memory ledger
unchanged.

## 3. Phase 2 - B=2 parity (the milestone gate)

`flash-gordon depth-b-selftest --manifest ... --depth 2`:

- Run A: conversation X alone, then conversation Y alone (B=1).
- Run B: X and Y interleaved through the B=2 ring, same prompts.
- Compare per-session: greedy token ids, logits bits, rendered output bytes,
  state digests at each frontier, and the final per-session QSA cursors.

Also run the same prompts through two concurrent API connections once the
multi-session engine is on (the selftest is the deterministic fallback).

**Gate 2:** X(B=2) == X(B=1) and Y(B=2) == Y(B=1) exactly, for at least:
the `12` prompt, the `Paris` prompt, a 4K-token prompt and a mixed
long+short pair. No batch-step error, no `restored` step in the passing run.

## 4. Phase 3 - isolation and mixed batches

1. Interleave two different conversations and assert neither stream changes
   when the other is introduced or removed (same comparison as Gate 2).
2. Mixed batch: one sequence at ~128K context and one at ~1K; assert the
   scheduler logs B=2 (both below the ~139.6K shrink point) and the long
   sequence's stream is unchanged from its solo run.
3. Failure injection: use the existing fault paths (`FG_TOWER_FAIL_ALLOC` is
   tower-only; use the session-control RESTORE path or a debugger breakpoint on
   one worker) to abort a B=2 step mid-flight; assert both sequences resume at
   the same frontier and produce the Gate 2 streams on retry.

**Gate 3:** isolation and retry pass; `SESSION_RESTORE` observed on the
injected run and no state digest differs from the pre-step snapshot.

## 5. Phase 4 - context shrink and memory

1. Run the selftest pair at 4K, 32K, 128K and 262K.
   Expected: B=2 at 4K/32K/128K; B=1 at 262K (policy threshold ~139.6K from
   the 640 MiB QSA scan budget). Log the chosen B and the predicted speedup
   from `fg_decode_batch_speedup`.
2. At 262K assert no swap (`/proc/<pid>/status` VmSwap unchanged), no page
   file growth, and decode TPS within +/-3% of the depth-1 262K baseline.
3. Memory ledger: record rank0 and worker Vk/host deltas at depth 2. Expected:
   +~6.5 MB/rank-block GDN+PLE, +one QSA session (~411 MiB index on rank 0 at
   262K plus worker record caches), inside the +3.43 GiB rank0 margin.

**Gate 4:** no swap, no OOM, memory delta within the margin, 262K decode not
worse than depth 1.

## 6. Phase 5 - performance

Measure decode TPS with `FG_DECODE_MS`/`FG_DECODE_PROFILE` at depth 1 vs 2,
same prompts, B=2 throughput counted as tokens across both sessions:

| Context | Predicted B=2 speedup | Notes |
|---|---|---|
| short (~1K) | ~1.9x | weights read once; state+wire small |
| 4K | ~1.85x | +94.8 MB QSA scan per sequence |
| 32K | ~1.6-1.8x | QSA scan grows |
| 128K | ~1.3-1.5x | B still 2; scan ~270 MB/token |
| 262K | 1.0x (B=1) | graceful degradation, no swap |

**Gate 5:** B=2 short-context aggregate TPS >= 1.6x the B=1 rate with the same
per-session outputs; per-step wall time (both tokens) <= 1.25x a single B=1
step; no prefill regression.

## 7. Rollback

- Depth-B is off unless the batch table is built with depth 2, so a production
  rank with the selftest never taken is unaffected; `FG_DECODE_RING=0` still
  restores the legacy expert-parallel decode replay.
- Per-phase rollback: revert the wiring commit(s); the host core
  (`decode_batch.c`, protocol ids 51/52) is additive and inert when unused.
- If B=2 parity fails only on QSA layers, fall back to B=2 on GDN-only layer
  ownership as a bisection (not a shipped configuration).

## 8. Risks and open questions

- **QSA page ownership**: a second session's pages must not advance or evict
  the first session's guard frontier. The session tag and per-slot guards are
  the critical review item of Phase 0.
- **Static replay**: replaying slot-0 recorded commands against slot-1 state is
  silent corruption; Phase 0.4 is a hard precondition.
- **Output penalties**: repetition/presence penalties need per-session output
  history on rank 4; until then B>=2 is restricted to penalty-free configs and
  the API must reject/fall back otherwise.
- **Rank0 n-gram O_DIRECT**: two sequences double the per-step lookup latency
  (16 reads each, 0.8-25 ms cold). It is serial on rank 0 and may dominate the
  B=2 step at short context; measure in Phase 5 and keep the n-gram agent
  (pageable, freeing ~3 GB/worker) in the build.
- **Token budget constant**: 640 MiB per step is a policy choice derived from
  the deployed geometry; if the fleet shows QSA scan headroom, raise it to keep
  B=2 beyond 139.6K, and re-run Gate 4.
