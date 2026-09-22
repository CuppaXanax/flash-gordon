# Performance Contract

This is the single citable operating-point record for the current checkout. Every
number below was measured on the production expert-parallel build at commit
`1f58c3a`, serving the sealed native-context pack described in
[PERFORMANCE_BYTE_BUDGET_2026-09-15.md](PERFORMANCE_BYTE_BUDGET_2026-09-15.md).
Early records (`lkg-10.035tps-cooked-experts` and the `PERFORMANCE_*` documents)
describe their measured revisions and are historical, not current claims.

The serving process derives this operating point from code: rank 0 prints one
`FG_LEDGER ...` line at startup from the sealed manifest, and the OpenAI endpoint
echoes that same line in the `X-Flash-Gordon-Ledger` response header.
`tools/pi-stability.ps1` asserts that live ledger against the sealed fleet
geometry and asserts the decode/prefill bands, so a stale or regressed deploy
cannot be reported as healthy.

## Measurement definitions

| term | definition |
|---|---|
| decode TPS | generated tokens divided by engine-reported decode seconds (`X-Flash-Gordon-Decode-TPS`), greedy unless stated; never HTTP wall time |
| warm | the process has completed at least one full request, so pipelines are compiled and the clock/thermal state has settled |
| cold | the first request after process start; clocks ramp, so a cold single turn reads ~21-22 TPS where the same turn warm reads ~25 |
| short decode | 32 generated tokens from a ~30-token prompt |
| N-context decode | sustained decode with N committed context tokens in the same session |
| prefill battery | repeated synthetic-prompt prefill throughput at the stated context depth |
| real turn (pi shape) | one complete multi-K-token attachment turn including rendering, tokenization and prefix handling; lower and noisier than the battery |

## Operating-point matrix

| operating point | context depth | decode TPS (engine) | notes |
|---|---:|---:|---|
| warm short decode | ~30 | 25.5-26.5 | gate floor 22; the 25-26 band is warm/short only |
| warm sustained 4K | 4,096 | 24.0-25.0 | not the production workload shape |
| warm sustained 16K | 16,384 | ~23 | |
| warm sustained 32K | 32,768 | ~19-21 | QSA page-miss cliff, see byte budget section 4; fix in flight |
| live single turn | ~43,000 | 17.3 | real turn, warm process |
| live long context | ~148,000 | ~14 | real turn, warm process |
| cold first turn | any | ~21-22 | clock state, not a regression |
| prefill battery 4K | 4,096 | 278-327 | run-to-run thermal spread |
| prefill battery 16K | 16,384 | 367.8 | |
| real turn 16K attachment | 16,384 | ~310 | pi-shaped turn |

Roofline context: the deployed byte chain is 6.671 GB/token (dense 62%,
experts 23%, state 5%), giving a practical warm band of 24-31 TPS and an
absolute 52.9 TPS ceiling at the measured 353 GB/s streaming peak. Decode is
bandwidth-bound; remaining kernel headroom is small. See
[PERFORMANCE_BYTE_BUDGET_2026-09-15.md](PERFORMANCE_BYTE_BUDGET_2026-09-15.md).

## Do not claim

* Do not present 25-26 TPS as production single-turn speed. It is warm,
  short-context decode.
* Do not present 4K as the workload. Production turns run at 16K-148K context,
  where decode is ~23 TPS down to ~14 TPS.
* Do not quote the early `lkg-10.035tps-cooked-experts` (10.035 tok/s) record as
  current performance.
* Do not compare prefill battery numbers across thermal states; the 4K battery
  alone spans 278-327 TPS between runs.
* Do not claim 50 TPS or 100 TPS raw decode: both are at or above the measured
  byte roofline for this pack.
* Do not cite a TPS number that the live `FG_LEDGER`/gate output does not carry.

## Verification

`FG_LEDGER` fields asserted by the gate: `ranks`, `layers`, `experts`, `topk`,
`layer_mode`, `blocks` (rank:layer-count ownership runs), `wire_hops`,
`batch`, `prefill_frames`, `window`, `logical`, `ring_prefill`, `ring_decode`,
per-rank `weights`/`dense`/`expert` bytes, `weights_total`, `qsa_cache_pages`
and `qsa_cache_bytes`. A deploy that changes ownership, sealed weight bytes or
the fast-path defaults fails the gate instead of silently changing the
operating point.
