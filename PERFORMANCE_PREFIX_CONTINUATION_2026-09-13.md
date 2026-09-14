# Ring Prefix Continuation — 2026-09-13

Ring sessions now resume per-rank owner state across sequential requests. A request
whose rendered transcript extends the previous request's token history re-prefills
only the appended tokens; everything else is reused. The Pi "snappiness" killer
(20-40 s of re-prefill per turn, growing with context) is gone: turn-2+ prefill
tokens equal the new tokens and time-to-first-token is flat instead of linear in
context.

Branch: `perf/prefix-continuation` (worktree `fg-work-cont`, based on `daa2a60`).
Deployed binary: `ed94975831309243d1e989fb8c70398fa1beb0528f1898a32806823ca2695c85`
(runtime/qsa only; shaders untouched — the parallel shader work in
`fg-work-pref6` remains exactly as deployed).

## Why every turn re-prefilled

Two independent causes:

1. `runtime_generate_tokens` deliberately cleared the prefix plan for any ring
   request ("ring requests always prefill from a cold reset") because ring decode
   did not exist when that comment was written. With ring decode, each block owner
   keeps authoritative GDN/PLE/QSA state across requests, so the reuse is safe.
2. The QSA page cache is write-behind: decode records land in the page cache and
   never reach the per-rank state file. On the next request the state frontier
   lagged the session frontier, so a continuation prefill either failed
   (`QSA state write batch frontier is not page aligned`) or dereferenced an
   unwritten page (`stale, torn, or corrupt QSA state page`).

## Change (src/, no shader edits)

- `src/runtime.c`
  - `fg_runtime.state_frontier`: token count the owners have processed, set to
    `history_count` after every successful generation and 0 on reset.
  - Ring prefix reuse is enabled when `ring_decode` is active and
    `plan.prefill_offset == state_frontier`; otherwise the old cold reset stands
    (`FG_PREFIX_RESET_COLD_START` / `FG_PREFIX_RESET_FRONTIER_UNAVAILABLE`).
  - `FG_PREFIX_CONT=0` disables continuation for same-binary A/B without a rebuild.
- `src/qsa.c` — `persist_prefill_state` first flushes the range the state file
  lags (`[persisted, first_token)`) from the pinned page cache, then persists its
  own range. Depth is bounded at two calls.
- `src/qsa_state.c` — corrupt-page errors now name layer/block/frontier/header so a
  future failure is diagnosable from the log.

Invariant chain: `plan.prefill_offset == runtime->state_frontier == history_count`
for a resumable request; each rank's QSA state file frontier is advanced to the
session frontier before any continuation prefill writes its own range. Any
mismatch demotes to a full reset and cold prefill, so a violated assumption costs
latency, never correctness.

## Measurements (8x BC-250 ring pack, greedy)

### 4-turn growing conversation, ~2.5 K new tokens/turn

| turn | before prefilled | before TTFT | after prefilled | after reused | after TTFT | output |
|---|---|---|---|---|---|---|
| 1 | 2558 | 10.7 s | 2558 | 0 | 10.1 s | Seven plus three equals ten. |
| 2 | 5102 | 18.9 s | **2538** | 2564 | **10.4 s** | Fourteen plus three equals seventeen. |
| 3 | 7647 | 27.7 s | **2538** | 5109 | **10.5 s** | Twenty-one plus three equals twenty-four. |
| 4 | 10193 | 36.5 s | **2538** | 7655 | **10.4 s** | Twenty-eight plus three equals thirty-one. |

Every turn in both arms ends with EOS; the after arm logs
`prefix hit, reset none`. Turn 4's output was reproduced identically by a forced
cold run of the same conversation (full 10193-token prefill, 37.4 s).

### Pi-like turns (~240 new tokens/turn, 5 turns)

| turn | with continuation | `FG_PREFIX_CONT=0` |
|---|---|---|
| 2 | reused 270, prefilled 238, 3.33 s | reused 0, prefilled 508, 4.13 s |
| 3 | reused 521, prefilled 238, 3.37 s | reused 0, prefilled 756, 4.59 s |
| 4 | reused 772, prefilled 238, 3.36 s | reused 0, prefilled 1004, 5.07 s |
| 5 | reused 1023, prefilled 238, 3.39 s | reused 0, prefilled 1252, 5.73 s |

### Long generation then continuation

891-token EOS-terminated generation, then one follow-up turn:
prefilled **32** tokens, reused 943, wall **1.42 s**, answer correct. This
exercises the decode-lag flush at ~900 tokens of page-cache lag.

### Abort safety

Cancelling a stream mid-generation marks the session failed; the next request
cold-serves (`reset failure`, correct answer) and the request after that resumes
as a prefix hit (`reused 30, prefilled 30, reset none`).

## Gates and baseline comparison

- `correctness64.ps1`: `12` / `Paris` PASS.
- `pi-stability.ps1`: `stages=6 conversation=4 correctness=2 ranks=8 status=PASS failures=0`
  (4K prefill 278-306 TPS across runs, short decode 18.0-18.1, band 15).
- Attach battery (`Measure-FlashGordonAB -Attach -Build ep -Runs4k 1`):
  128 prefill 40.6, **4K prefill 282.1**, **short decode 20.8**, **4K decode 20.0**.
  Baseline state was 4K prefill 277-280, short decode 21.1, 4K sustained 19.4, so
  decode is unchanged within measurement noise (the change touches no decode
  kernel and no decode path).

## Known limits

- Greedy identity with a *fully cold* run is FP-level, not bit-level. Decode state
  updates and chunked GDN scans associate differently, so resuming from a decoded
  frontier can perturb logits; in one contrived 5-turn conversation it flipped a
  leading phrase ("The number 14…" vs "14…") while the 250-filler conversation and
  the soak conversations matched exactly. Continuation is run-to-run deterministic.
  Exact identity needs state snapshots at microbatch boundaries, which is what the
  unused `SESSION_PREPARE/COMMIT/RESTORE` transactions are for (see
  `MTP_FEASIBILITY_2026-09-13.md`); the continuation frontier is a clean hook for
  that work. `FG_PREFIX_CONT=0` restores full re-prefill if a workload needs the
  cold path.
- Rank 0's fileless QSA mirror never unpins pages (they are its only copy), so a
  single ring session's continuation context is bounded by the mirror record
  cache (~27 K pages, own layers only ≈ 2 layers × context/4; fine through ~100 K
  tokens, above the 34 K observed Pi sessions).
- Non-ring-decode deployments still cold-reset ring requests; the fast path is
  gated on `ring_decode` by construction.
