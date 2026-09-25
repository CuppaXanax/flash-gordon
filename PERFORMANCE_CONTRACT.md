# Performance Contract

This is the single citable operating-point record for the current checkout. Every
number below was measured on the production expert-parallel build at main
`4b3e85d` (pageable n-gram default, full-context QSA record caches, paired
down/reduce expert kernels), serving the sealed native-context pack described
in [PERFORMANCE_BYTE_BUDGET_2026-09-15.md](PERFORMANCE_BYTE_BUDGET_2026-09-15.md).
Early records (`lkg-10.035tps-cooked-experts` and the `PERFORMANCE_*` documents)
describe their measured revisions and are historical, not current claims.

Current configuration defaults: 8-rank expert-parallel ring with ring prefill,
ring decode, the worker-owner executor and the 4-way head split all on by
default; a 640 MiB/rank worker QSA record window (rank 0 requests 618 MiB), the
pageable n-gram shard mode (mmap + bounded read-ahead; 3.7-3.9 GiB of host
memory returned per worker) and unconditional locality-hint record read-ahead.
Those are code defaults, not launch recipes: environment variables are
diagnostics, bisection opt-outs or test/packer tooling only, listed with a
status in [docs/FLAGS.md](docs/FLAGS.md) and enforced against the code by
`tools/check-flags.sh`.

The serving process derives this operating point from code: rank 0 prints one
`FG_LEDGER ...` line at startup from the sealed manifest, and the OpenAI endpoint
echoes that same line in the `X-Flash-Gordon-Ledger` response header.
`tools/pi-stability.ps1` asserts that live ledger against the sealed fleet
geometry and asserts the decode, prefill and 32K bands, so a stale or regressed
deploy cannot be reported as healthy.

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
| warm short decode | ~30 | 25.6-26.1 | 32-token sweep; gate floor 22 |
| warm sustained 4K | 4,096 | 23.8-24.4 | not the production workload shape |
| warm sustained 16K | 16,384 | 22.4 | |
| warm sustained 32K | 32,768 | 22.2 | gate floor 21.0; the pre-WS1 QSA page-miss cliff measured 19.80 |
| warm sustained 64K | 65,536 | 21.5 | page-cache bump took this from 21.2 |
| warm sustained 151K | ~151,000 | 20.1 | page-cache bump took this from 17.0 |
| cold first turn | any | ~21-22 | clock state, not a regression |
| prefill battery 4K | 4,096 | 280-330 | soak run measured 285-315 |
| prefill battery 16K | 16,384 | ~375 | |
| real turn 16K attachment | 16,384 | ~310 | pi-shaped turn |

The decode sweep rows are the 32-token sweep on the current build; the soak run
reports 4K prefill ~285-315 TPS, short decode ~25, 32K decode ~23 and 8/8 ranks
alive. The WS1 spread worker window and read-ahead raised 32K from 19.80 to
22.54 TPS; the later pageable n-gram shard mode returned 3.7-3.9 GiB of host
memory per worker and let the record caches reach full 262K coverage, taking
151K from 17.0 to 20.1 TPS.

Roofline context: the deployed byte chain is 6.671 GB/token (dense 62%,
experts 23%, state 5%), giving a practical warm band of 24-31 TPS and an
absolute 52.9 TPS ceiling at the measured 353 GB/s streaming peak. Decode is
bandwidth-bound; remaining kernel headroom is small. See
[PERFORMANCE_BYTE_BUDGET_2026-09-15.md](PERFORMANCE_BYTE_BUDGET_2026-09-15.md).

## Multi-session status

M3.2 two-session serving is implemented on `feat/batch-block-r1` but **not
promoted** (fail-closed): parked turns, the two-turn engine scheduler, per-slot
cold start and owner-session select. A lone eligible session runs the
production B=1 runner step and never pays a batch or select round trip; a
second session batch-steps through the depth-B table (74.6 ms/step measured,
~25.9 tok/s aggregate in the depth-B gate) and a long prefill chunk-yields one
128-token group at a time, only while another session has work. The
`--concurrent` selftest (short pair + 2K chunk-yielded prefill) passes with
per-session byte parity and an unchanged slot-0 state digest; two real clients
decode concurrently at ~11.4/13.4 tok/s per session with `/health` probes
<= 9.3 ms and abort churn leaving the survivor byte-identical. The solo path is
byte-identical to `main` on the pinned greedy requests (3 runs each), but the
full soak/battery/depth-B re-run on the final binary is still pending, so
production remains `20260924-teardown` and the numbers above are gate
evidence, not a production claim.  See
`bc-250-dbg/results/m3b-20260924-2314/EVIDENCE.md`.

## Do not claim

* Do not present 26 TPS short-decode as production single-turn speed. It is warm,
  short-context decode.
* Do not present 4K as the workload. Production turns run at 16K-151K context,
  where decode is ~22.4 TPS at 16K down to ~20.1 TPS at 151K.
* Do not report the pre-WS1 32K cliff (19.80 TPS) as healthy; the gate floor is
  21.0 TPS and the current build measures ~22.2-23.1.
* Do not quote the early `lkg-10.035tps-cooked-experts` (10.035 tok/s) record as
  current performance.
* Do not compare prefill battery numbers across thermal states; the 4K battery
  alone spans 280-301 TPS between runs.
* Do not claim 50 TPS or 100 TPS raw decode: both are at or above the measured
  byte roofline for this pack.
* Do not cite a TPS number that the live `FG_LEDGER`/gate output does not carry.
* Do not introduce an environment variable as the only way to reach a
  performance path: defaults or CLI options only, per [docs/FLAGS.md](docs/FLAGS.md).

## Verification

`FG_LEDGER` fields asserted by the gate: `ranks`, `layers`, `experts`, `topk`,
`layer_mode`, `blocks` (rank:layer-count ownership runs), `wire_hops`,
`batch`, `prefill_frames`, `window`, `logical`, `ring_prefill`, `ring_decode`,
per-rank `weights`/`dense`/`expert` bytes, `weights_total`, `qsa_cache_pages`
and `qsa_cache_bytes`. A deploy that changes ownership, sealed weight bytes or
the fast-path defaults fails the gate instead of silently changing the
operating point.

The gate also runs a 32K decode probe (32 generated tokens from a list-request
prompt on a warm process) and asserts 21.0 TPS; `-Skip32kDecode` opts out. The
probe fails closed if fewer than 32 tokens were generated, so first-token
latency can never be reported as decode TPS. The flag budget is enforced by
`tools/check-flags.sh` inside `make test`: every `getenv("FG_*")` in `src/` and
`include/` must have a documented status, the doc must not list a flag the code
does not read, and the read set must equal the checked-in expected set.
