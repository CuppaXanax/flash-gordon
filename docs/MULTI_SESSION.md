# Independent sessions through the ring

Latest revised robustness-only gate:
[`MULTI_SESSION_ROBUSTNESS.md`](../results/hillclimb/MULTI_SESSION_ROBUSTNESS.md).
Five fresh pairs pass 32K decode, TTFT and greedy checks, but short decode's
paired median is **0.965996x**, below the explicit 0.97 floor. Production was
restored; the robustness runtime remains unpromoted. Production itself also
fails the loaded cold-32K QoS target.

Latest production incident follow-up:
[`MULTI_SESSION_INCIDENT.md`](../results/hillclimb/MULTI_SESSION_INCIDENT.md).
Recovery/routing fixes are **not promoted**; stack-1 plus mix-04, sessions 2
remains selected. Ordered supervision is installed and restart-tested.

Current adaptive-scheduler qualification and deployment decision:
[`MULTI_SESSION_FLEET.md`](../results/hillclimb/MULTI_SESSION_FLEET.md).
That report distinguishes the user's revised solo/aggregate promotion rule
from the original full QoS/parity targets, including remaining limitations.

## Historical fixed-32 qualification: NO-GO

The numbers in this introduction describe the previous fixed-32 scheduler,
not the adaptive scheduler's new fleet measurements.
The main-based integration was tested on .42–.49 against immutable original
production with three fresh-salt paired rounds. The repaired `79b0efc` native
build and mix-04 pack pass all 15 same-pack single-session canonical hashes,
but the 11-metric screen has **2 corrected wins, 5 significant regressions and
4 inconclusive metrics**. Every protected metric must pass independently;
aggregate throughput does not excuse chat/prefill/TTFT losses.

The first build failed on static descriptor slots 4–7; `79b0efc` repairs the
dynamic-pool guard without weakening static window bounds. An eight-slot
llvmpipe regression reproduces the old failure and passes after the repair.
Three entirely fresh pairs were used for the repaired binary; failed-build
samples were not pooled. Aggregate non-stream wall TPS rises from 18.636 to
30.429, but chat falls from 18.977 to 15.215 and 4K engine prefill is only
0.1224x original. The long-context inferential follow-up was skipped after the
predeclared screen failed; no 32K/128K noninferiority claim is supported.

A separate warm SSE forensic trial also fails sustained generation: the agent
emits one actual token both isolated and loaded; chat emits 128, and both hashes
match their own controls. It is not valid sustained concurrent-throughput
evidence. The ~2x and memory gates below remain future acceptance criteria,
not a promotion claim. Full paired evidence and the final capacity/restore
receipts are published in `results\hillclimb\AB_QUALIFY.md` on
`cuppaxanax/ab-harness`.

The final capacity diagnostic was stopped at the user's wrap-up instruction
after **178400/260075 prompt tokens**, not after a completed near-262K session.
All eight sampled owners retained at least 2.195 GiB MemAvailable, with no new
OOM, process swap or system-swap growth, but this is **not full-context
qualification**. The original production release was restored, its pinned
launchers restart-tested, real API generation verified and task scratch
cleaned; .42–.49 are released. Three-pair df=2/76-comparison gating was
acknowledged as too low-power for practical whole-candidate acceptance and
left unchanged; redesign power before another qualification run.

## Contract

Start the API with `--sessions 2`; the default remains `--sessions 1` and its M2
FIFO engine. Requests are routed by the longest matching canonical public
message prefix, not by the optional OpenAI `user` field. A shortened replay or
edited history stays in its closest slot but resets that slot: GDN state cannot
rewind. Only a complete cached prefix qualifies for forward state reuse.
A shared leading system prompt alone does not establish affinity. `user` is not
an authentication/tenant-security boundary.
Clients must still send the complete canonical conversation, including exact
assistant/tool messages, to retain prefix reuse.

There are two resident state slots, not an unbounded cache. Requests for the
same resident conversation serialize; unrelated work can use the other slot.
An unused slot wins before eviction. With no reusable prefix, eviction is LRU,
except a short request waits for a free/shorter slot rather than evicting a
longer conversation. A busy best-prefix match waits instead of duplicating
resident state. Exact prefix reuse works without `user`, including side/title
requests between agent turns. Busy slots are scored from their admitted request
even while a mismatch has invalidated their old public cache. Protecting an idle
long conversation can serialize two unrelated short requests: those represent
three resident conversations, not the qualified two-user continuation workload.
The existing bound remains
four outstanding requests (running plus queued), with `429` on overflow.

Only protocol-6 ring prefill/decode with complete expert slabs on block owners
is admitted. Legacy EP fan-out is not a supported two-session configuration.
Deploy the same binary on every rank: the protocol version did not change, but
old workers ignore the new frame-slot convention and are unsafe to mix.
MTP and multimodal concurrency are not qualified by this text-serving change.

## Transport recovery and supervision

Ring-prefill sends mark their session transport pending until every issued
frame drains. Failed or incomplete transactions poison it; no reset may mutate
a failed pipeline. Each session owns its QSA transport callbacks and failure
state, even though model weights and physical sockets are shared. Header checks
include slot, nonce, source, token frontier and sequence. Ring traces identify
the slot and nonce; request cleanup preserves the original wire/GPU error.

A physical transport failure invalidates **both** in-flight computations and
resident worker caches: unaffected streaming requests cannot continue across
worker replacement. The coordinator unwinds both jobs, fails affected requests,
keeps its listener/queue, and reconnects after workers restart. New generations
and `/health` return HTTP 503 while recovering. Healthy multiplexed responses
include `ring_recovery:true` and a monotonic `recovery_epoch`.

Install `tools/fg_ring_control.py` on every owner beside the existing
hash-pinned `launch-current.sh`/`current.json`, and write its numeric rank to
`supervisor-rank`. Install `fg_ring_supervisor.py` and its JSON configuration on
rank 0. The configuration supplies `operator`, contiguous `nodes` with `rank`
and `host`, `user`, `identity`, `known_hosts`, `health`, `interval`, and
`startup_timeout`. Use a dedicated coordinator SSH identity: authorize only
`start`, `stop`, `status` (and active-lease rollback) through the forced control
entry point, `from="192.168.42.42",restrict`. Provision host public keys through
the trusted operator transport; strict host-key checking is mandatory. Never
copy the private identity into source or qualification evidence.

On rank 0, after provisioning, run:

```bash
op=/home/xander/flash-gordon-live/20261015-abq-operator
python3 "$op/fg_ring_supervisor.py" --config "$op/supervisor.json" --check
sudo install -m 644 tools/flash-gordon-ring.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now flash-gordon-ring.service
sudo systemctl is-enabled flash-gordon-ring.service
sudo systemctl is-active flash-gordon-ring.service
```

Keep `LimitMEMLOCK=infinity` in the unit: serving children must inherit the
same pinned-buffer allowance as the production SSH launcher. The systemd
default can make coordinator io_uring registration fail with `Cannot allocate
memory`, even though the model itself loaded.

Cold/failed API recovery stops rank 0, stops workers, starts workers, then starts
rank 0. A live recovery-capable API only restarts workers and waits for a new
healthy epoch; failed handshakes fall back to a complete ordered restart.
Service startup after reboot uses this same ordering, not separate competing
rank units.

**Before taking a fleet lease**, create `$op/supervisor.pause` on rank 0 and
stop `flash-gordon-ring.service`; check it is inactive before experiments.
Stopping this unit deliberately leaves serving processes alone. Remove the
sentinel and start the unit only after restoring/pinning the selected recipe.
Do not leave a qualification deadline timer armed after returning the lease.

Offline regressions (no model, GPU or fleet required):

```bash
export TMPDIR="$PWD/.build"
make test-multi-session
FG_MULTI_PREFILL_ONLY_WINDOW=1 ./tests/test_pipeline_wire
bash tools/check-flags.sh
```

The synthetic wire test executes production prefill/output coroutines,
framing and demultiplexing: unequal prefills, short tail, 128-to-1-token
quantum transition, fault injection, poisoned reset and survivor unwind.
Injected faults reproduce the old fatal **handling path**, not the unknown
first physical/protocol fault in the preserved production log.

## Scheduling and ownership

Two stackful cooperative jobs run on the existing engine thread. They execute
the normal generation loop, yielding **before** a fabric receive submits an
io_uring operation. The dispatcher peeks headers, then resumes the job whose
session/peer/channel is ready. One turn resumes at most one job. Ready streamed
decode has priority over other decode, which has priority over prefill;
equal-priority jobs alternate. Sleeping jobs are polled without selecting their
GPU state, and only a switch to a ready job drains/selects owner state. This
permits A at rank k while B is at rank j, unlike running
two complete generations serially or batching two tokens at the same position.

The high byte of frame flags carries the slot; request ID remains the
session-generation nonce and sequence remains the token/layer coordinate.
Workers select state before handling each frame. Nonce/route checks, per-slot
layer frontiers, and output-config/history slot checks reject stale or
misrouted state mutations. Session BEGIN resets only its own GDN, PLE, QSA and
output history once multiplexing is enabled.

Resetting a resident QSA namespace retains its backing inode: the page service
and owner executor both hold it open. Unlinking/recreating it on an epoch change
would leave the owner writing an invisible old file and split the two views.

| Private per slot | Shared, serialized on each rank |
|---|---|
| GDN recurrent/conv state, PLE/HC state | Weights, tokenizer, n-gram store |
| QSA state file, cache and GPU indexes | Vulkan context/queue, temporary GPU scratch |
| Token/transcript history and exact frontier | Embedding and prefill scratch tensors |
| Sampler configuration/RNG and output history | Transport connections |
| Decode host wires, prefill host wires and coroutine stack | Immutable topology/model metadata |
| Four static decode recordings and head handoff state | Stage-local expert kernels |

There is no concurrent Vulkan recording on one rank. Before switching jobs the
coordinator drains static work and rejects an active Vulkan batch. Outgoing
payloads are copied before yielding; shared transient tensors must never be
retained as live inputs across a receive. Ring output is consumed into the
output request before another job can overwrite it.

With only one active request, prefill uses the **full sealed microbatch and all
eight frames**, exactly as the single-session issue pipeline. An idle resident
conversation does not throttle another request. With two active requests, the
default is a maximum **128-token quantum**. Two prefills use **four frames
per session**, filling an eight-frame steady-state ring instead of exposing
round-trip latency with only two frames total. With a decoder active, prefill
is limited to **one frame**. A **15 ms soft rank-service budget** can shrink that
cap further, down to one token. It uses the worst observed coordinator
prefill-block wall cost per token (including its outgoing send), learned also
during idle/full-depth prefill. Without a sample it starts at one token rather
than assuming a 128-token chunk is cheap. Smaller chunks' fixed overhead can
raise the estimate on subsequent observations. This conservative estimate is
shared by the two slots and is reset on API restart.

The live policy is checked before every issue, not just at prompt start. Each
completed issue also yields to admit newly queued requests and run ready
streamed decode before issuing another quantum. When the other request finishes, full depth
and microbatch are restored without restarting prefill.

Already-issued frames are not canceled on the idle-to-contended transition:
the old window drains while no new frame is issued above the new cap. A quantum
change uses actual token offsets/counts, not a fixed microbatch divisor, for
return matching, final output-row selection and interrupted-prefix frontiers.
Decode tokens and prefill chunks advance independently around the ring.
A stage is not preempted in the middle of a GPU dispatch.
Tokenization, media preprocessing, filesystem stalls and a TCP frame transfer
are not preemptible either: this is bounded text-prefill scheduling, not a
hard real-time TTFT guarantee. The two sessions still share stage bandwidth,
page cache, output-head helpers and TCP channels.

The old 32-token one-frame policy averaged about 0.83 s/chunk in the first
4K qualification trial; blindly raising that quantum to 128 is not a latency
fix. Service-time sizing, rather than the token cap alone, targets the ~1.5x
gap objective: 15 ms is below half the measured 37.437 ms solo decode interval.
For example, an observed 2.5 ms/token rank cost selects six tokens, not 128.
This is still **not a measured gap guarantee**: other ranks, fixed dispatch
costs, filesystem stalls and the old in-flight window can exceed the estimate.

Fleet retesting must measure loaded token gaps and TTFT as well as throughput;
if a rank's quantum service time exceeds half the solo decode interval, the
~1.5x gap objective cannot be claimed. Keep window 1 and tune the service budget
against the loaded gap gate; `FG_MULTI_PREFILL_SLICE_US=0` is a fixed-cap
bisection, not the latency-protecting default. See `docs/FLAGS.md` for accepted
ranges. Coordinator launch examples (workers need no tuning flags):

```sh
# Defaults; prepend to the existing API launcher command below.
FG_MULTI_PREFILL_TOKENS=128 FG_MULTI_PREFILL_WINDOW=1 FG_MULTI_PREFILL_SLICE_US=15000 FG_MULTI_DECODE_PRIORITY=1 \
  ./flash-gordon api --manifest pack/manifest.fgm --sessions 2 --port 8080
# Priority-only bisection; keep the same binary, pack, cap and window.
FG_MULTI_PREFILL_TOKENS=128 FG_MULTI_PREFILL_WINDOW=1 FG_MULTI_PREFILL_SLICE_US=15000 FG_MULTI_DECODE_PRIORITY=0 \
  ./flash-gordon api --manifest pack/manifest.fgm --sessions 2 --port 8080
```

Client cancellation is observed at existing completed token/chunk boundaries;
already-issued work is drained before retaining the exact frontier. A transport,
GPU or protocol failure is fail-closed for the serving process, not an attempt
to continue the other session on a potentially corrupt ring. Restart all ranks.

### Decode diagnosis and ring prediction

The previous **0.81x chat metric is wall TPS**, not decode-only TPS. The three
paired qualification rounds contain the following chat measurements (seconds
outside decode include prefill/queue/HTTP work; they are not pure prefill):

| Round | Original engine TPS | Fixed-32 engine TPS | Original outside decode (s) | Fixed-32 outside decode (s) |
|---|---:|---:|---:|---:|
| 0 | 27.35 | 24.38 | 2.02 | 3.07 |
| 1 | 26.07 | 23.82 | 2.09 | 3.04 |
| 2 | 27.40 | 23.60 | 2.07 | 3.07 |

Thus decode-only retention was approximately 0.86–0.91x; about another second
of non-decode delay also depressed wall TPS. Source inspection found no
generation-long mutex: the admission mutex covers queue operations, the send
mutex one frame, and rank-local GPU execution is intentionally serialized.
There was unnecessary owner activation/static draining on every attempted
turn, including sleeping jobs; the ready-first picker removes that work.
This is not evidence that it explains the entire measured decode regression.
Shared head/GPU service, TCP head-of-line blocking and dispatch gaps remain
possible contributors and need profiling under the new binary.

From `DECODE_PROFILE.md`, normal solo latency is L=37.437 ms, with rank service
3.45–4.32 ms and roughly 0.8 ms per participating head shard. For N independent
circulating tokens, the ideal closed-ring bound is
`T_stream >= max(L, N * max_rank_service)`; include head service in a rank's
demand. At N=2, even `2*(4.32+0.8)=10.24 ms` is well below L. With tokens spaced
around the ring, the expected ideal rate is therefore **~26–27 TPS per stream,
~53 TPS aggregate**, not half solo per stream. Actual contention and scheduling
can lower it; this model is a prediction, not a new fleet result. Profiling
that disables static replay must not be compared with normal replay timings.

## Memory admission at 262K

Do not budget against nominal 16 GB. These blades expose about **14.84 GiB**
MemTotal; use the current pack's sealed per-rank ledger and live measurements.
No new quantization or weight/kernel schema is introduced here.

Two GDN/PLE slots were already allocated by the owner implementation. The new
increment is predominantly the second QSA cache/index and host histories, not
a second weight copy. At 262144 tokens the existing KV/index/GDN ledger is
approximately **48.7 MB per rank per session**, topology-dependent; that figure
does **not** include the record cache or state backing files.

Each rank logs `FG_MULTI_MEMORY` before opening slot 1. Admission requires:

```
extra_bound = QSA_cache_memory(cache_pages)
            + logical_tokens * 148 * owned_QSA_layers + 64 MiB
MemAvailable >= extra_bound + 512 MiB reserve
```

The bound covers a conservative auxiliary/index allowance plus histories and
stacks. It is not a proof of worst-case unified-memory pressure or kernel page
cache residency. The cache budget is per session: the measured rank-0 ledger's
131072 cache pages already occupy about **648,019,968 bytes** per cache, rather
than merely its nominal record payload. Inspect every rank; do not multiply one
rank's ownership by eight.

QSA backing is about **512 MiB per owned QSA layer/session** at 262K, approximately
**6 GiB fleet-wide per session**. Slot 0 retains `qsa-owner-rank-NN.state`; slot 1
uses `qsa-owner-rank-NN-s1.state`. Full-context qualification must record actual
RSS, GPU/unified allocations, MemAvailable, disk use, major faults and swap,
with both sessions live. Reject OOM, swap growth, cache thrashing or a reserve
below 512 MiB. A prefill smoke test alone cannot qualify this.

## Targeted local validation

From Windows:

```powershell
Set-Location D:\looking-glass-labs\fg-wt\multi-session-int
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/looking-glass-labs/fg-wt/multi-session-int && mkdir -p .build && export TMPDIR="$PWD/.build" && make -j8 && make -j8 test-multi-session tests/test_decode_batch && ./tests/test_decode_batch && bash tools/check-flags.sh'
python -m unittest tools.test_hillclimb_concurrency -v
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

The transport test uses eight real localhost processes, two different recurrent
integer sequences/positions and changing nonces, checks both are issued before
the first result, and exercises timeout, slot restart and peer-specific receives.
It does not execute model kernels. The Python fixture drives the real PowerShell
harness against concurrent local SSE streams, including role-only events,
terminal usage, a delayed cold agent and independent completion times.
The QSA test opens real state files for both slots, resets one nonce, checks
inode identity, writes through a retained owner handle, verifies the page-service
view sees it, and checks the other slot's records/nonce are unchanged.
Deterministic coroutine tests additionally cover full-depth idle policy and
restoration, contended caps, conservative uncalibrated service sizing, measured
service-budget shrinking, fixed-cap bisection, streamed/nonstream decode
priority and decoder fairness. Variable 128/32/1/6/256/19-token chunks verify
exact return matching and interrupted-prefix accounting. Invalid tuning values
fail parsing; the registered flag gate passes. The native Windows Python run
executes all four PowerShell fixtures (WSL Python alone skips them without
PowerShell on PATH).

Known baseline failures must not be reported as passes: ordinary
`tests/test_fabric` has an output-config protocol assertion failure;
`tests/test_prefill_dispatch` fails its existing route assertions and crashes
(reproduced with the unmodified main runtime/API); `test_prefill_collective`
reports invalid expert prefill arguments. The passing targeted transport mode
does not conceal or resolve those older fixtures.

## Future deployment: lease required

**Do not execute this section without the exclusive production fleet lease.**
First checkpoint the working API, obtain the exclusive .42-.49 lease, and stop
the previous deployment with its PID-scoped runbook. Preserve that deployment's
binary, launch arguments, environment and manifest for full revision rollback.
The commands below assume the selected sealed manifest is named
`manifest-native-262k.fgm`; change only that variable if the lease holder uses a
different manifest. Do not repack or silently select a different Q4 candidate.

Build/ship one isolated experiment directory; its pack files are symlinks, with
a regular manifest copy and private state files. Never run it simultaneously
with production on the same endpoints.

```powershell
Set-Location D:\looking-glass-labs\fg-wt\multi-session-int
$runner = 'D:\looking-glass-labs\fg-wt\ab-harness\tools\ab-fleet.ps1'
$targets = @(42..49 | ForEach-Object { "192.168.42.$_" })
$id = 'multisess-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$remote = "/home/xander/fg-experiments/$id"
$pack = '/home/xander/fg-ring-pack-abliterated'
$manifest = 'manifest-native-262k.fgm'
$local = New-Item -ItemType Directory -Force ".build\$id"
$archive = Join-Path $local.FullName 'runtime.tar.gz'
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/looking-glass-labs/fg-wt/multi-session-int && export TMPDIR="$PWD/.build" && make -j8 && tar -czf .build/multisess-runtime.tar.gz flash-gordon vulkan'
if ($LASTEXITCODE) { throw 'Build failed' }
Copy-Item .build\multisess-runtime.tar.gz $archive
git rev-parse HEAD | Set-Content (Join-Path $local.FullName 'commit.txt')
$step = 0
function Invoke-Step($target, $text, $transfer = @{}) {
    $script:step++
    $path = Join-Path $local.FullName "step-$step.sh"
    [IO.File]::WriteAllText($path, $text.Replace("`r",''))
    $r = @(& $runner -ScriptPath $path -Targets @($target) -PassThru @transfer)
    if ($r.Count -ne 1 -or $r[0].ExitCode -ne 0) { throw "Failed: $target" }
    $r | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $local.FullName "step-$step.json")
}
foreach ($target in $targets) {
    Invoke-Step $target "set -eu; test -r '$pack/$manifest'; mkdir -p '$remote/pack'"
    Invoke-Step $target 'true' @{ UploadPath=$archive; UploadDestination="$remote/runtime.tar.gz" }
    Invoke-Step $target @"
set -eu
cd '$remote'
tar -xzf runtime.tar.gz
cp -as '$pack/.' pack/
find pack -type l -name '*.state' -delete
cp --remove-destination '$pack/$manifest' pack/manifest.fgm
./flash-gordon inspect --manifest pack/manifest.fgm
sha256sum flash-gordon pack/manifest.fgm
"@
}
$sessions = 2
function Start-Experiment {
    foreach ($rank in @(1..7) + @(0)) {
        $launch = if ($rank -eq 0) {
            "api --manifest '$remote/pack/manifest.fgm' --host 0.0.0.0 --port 8080 --model-name Qwen3.8-Flash-Next-Uncensored --sessions $sessions"
        } else { "rank --manifest '$remote/pack/manifest.fgm' --rank $rank" }
        Invoke-Step $targets[$rank] "set -eu; cd '$remote'; nohup ./flash-gordon $launch >server.log 2>&1 </dev/null & echo `$! >server.pid"
    }
}
Start-Experiment
Invoke-RestMethod http://192.168.42.42:8080/v1/models -TimeoutSec 30
```

Allow weight loading before the final probe; retry it for up to 30 minutes.
Then inspect `server.log` on all ranks for READY and `FG_MULTI_MEMORY`; a `/models`
response alone is not evidence that both state slots have initialized. Compare
binary/manifest SHA256 on all eight machines and preserve the current memory
ledger and pack identity with the local results.

## Qualification commands and gates

With the exclusive lease still held, these commands collect descriptive load
diagnostics only. They do not replace the fresh-salt paired confidence-interval
gates in `docs\AB_QUALIFICATION.md` on `cuppaxanax/ab-harness`. Inconclusive
noninferiority is a rejection, not permission to promote.

```powershell
$url = 'http://192.168.42.42:8080/v1'
pwsh -NoProfile -File .\tools\hillclimb.ps1 -BaseUrl $url -ConcurrentGeneration -ChatContextTokens 64 -AgentContextTokens 32768 -ConcurrentTokens 256 -ConcurrentRepeats 5
foreach ($context in @(4096,32768,131072,260000)) {
    pwsh -NoProfile -File .\tools\hillclimb.ps1 -BaseUrl $url -ConcurrentGeneration -MixedPrefill -ChatContextTokens 64 -AgentContextTokens $context -ConcurrentTokens 256 -ConcurrentRepeats 5 -ChatDelayMilliseconds 200 -RequestTimeoutSeconds 3600
    if ($LASTEXITCODE) { throw "Mixed prefill failed at $context" }
}
pwsh -NoProfile -File .\tools\api-abort-retry.ps1 -BaseUrl 'http://192.168.42.42:8080'
```

260000 is a prompt-size hint leaving room for template/suffix/generated tokens;
the authoritative SSE `prompt_tokens + completion_tokens` must fit 262144.
Also run warm two-long-session tests with both context hints 260000, and
repeated cached suffixes of 32/128/256 tokens. Keep a second live stream during
agent abort, retry, divergence, identity eviction and queue-full tests.

For the requested one-near-262K-plus-chat **capacity** run, add `-CapacityProbe`
to `-ConcurrentGeneration -MixedPrefill`. It skips the isolated long-agent
generation, so the long context is cold-prefilled only once while chat is live.
It retains exact SSE usage, prefix counters, loaded TTFT/TPS, overlap, and the
isolated-versus-loaded chat hash. The long agent's isolated TPS, retention,
same-prompt hash comparison and aggregate speedup are explicitly null, not
invented from another request. This is a memory/load probe, not an independent
long-context throughput or greedy-parity qualification.
If sustained-generation validation succeeds after the long request completes,
it also sends a fresh chat on the existing
chat identity while leaving the agent identity assigned. The result records
that real chat response separately, exercising chat allocation/generation with
the completed near-limit agent still resident.

`-ConcurrentGeneration` primes separate identities with a complete Ready
answer (32-token ceiling), then asks for a sustained list continuation. It measures each isolated,
re-primes the identical prefixes, and submits **both** generation requests
asynchronously before waiting for response headers. `-MixedPrefill` leaves the
agent cold and delays chat submission into agent prefill. Per-stream TTFT ignores
role-only/heartbeat chunks. TPS normally requires terminal usage from those
exact streams, never counts borrowed from another request or SSE-event counts.
For a legacy FIFO baseline only, `-AllowSerializedBaseline` permits missing
usage/overlap/prefix metadata, but leaves accounting incomplete and rates null.
Reconcile the captured request IDs against completion lines in that same
coordinator's log before using those samples; exit 2 also signals pending
accounting. `-PromptSalt` makes request nonces reproducible across paired arms;
use a fresh salt for each new round.

The legacy decode-window aggregate divides actual completions by **latest DONE
minus earliest visible token**. `aggregate_wall_tps` instead starts at the
first request launch and includes all prefill/queue delay. Use the latter for
paired aggregate wall-throughput comparisons: the former omits FIFO prefill
before the first token, but includes prefill overlapped with a pipelined chat.
Neither is the sum of individual TPS. Warm samples require actual visible-generation overlap and at
least 32 completions per stream. Early EOS invalidates a short sample; do not
claim the requested ceiling was generated. A synthetic fixture's ~2x is not
model throughput evidence.

Invalid sustained-generation trials are retained under `concurrency.trials`
with `concurrency.complete=false` before the harness raises its error. Inspect
their actual token counts and isolated/loaded hashes; do not erase early-EOS
or failed-overlap observations by reporting only completed trials.

Report JSON's isolated/loaded per-stream TPS, aggregate speedup, TTFT p50/p95,
inter-event p95, raw event times, prefix counts and same-prompt text hashes.
Record commit/pack identity separately if the server lacks a Git SHA header.
The original full-design acceptance targets below remain distinct from the
latest user-authorized solo/aggregate decision rule in the fleet report:

* Warm aggregate at least **1.7x** the mean isolated TPS (target approximately
  2x); each stream retains at least **85%** of isolated TPS.
* Decode-only inter-event p95 no worse than **1.25x** isolated. SSE events are
  visible-text events, not necessarily one event per model token.
* Warm chat TTFT at most **1 s** for a suffix up to 256 tokens while the agent
  performs cold long prefill; chat must produce content before agent prefill
  finishes. Inspect every context level, not only the combined aggregate.
* Fixed prompts/seeds: default `--sessions 1`, `--sessions 2` with one active
  request, and each member of a concurrent pair must match greedy output
  hashes. Re-run the production reference set without overwriting it. Also
  test nonzero repetition/frequency/presence penalties with fixed seeds, tool
  continuations and session-local divergence/reset. Local mocks do not prove
  model parity, especially across different prefill chunk sizes.
* Both 262K resident contexts meet the memory/disk/no-swap conditions above;
  aborting/resetting A must leave B's next output and prefix frontier intact.

### Stop and rollback

Use the same PowerShell variables/functions from deployment:

```powershell
function Stop-Experiment {
    foreach ($rank in @(0) + @(1..7)) {
        Invoke-Step $targets[$rank] @"
set -eu
if test -r '$remote/server.pid'; then
  pid=`$(cat '$remote/server.pid')
  if kill -0 "`$pid" 2>/dev/null; then
    test "`$(readlink /proc/`$pid/exe)" = '$remote/flash-gordon'
    kill -TERM "`$pid"
    for i in `$(seq 1 30); do kill -0 "`$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "`$pid" 2>/dev/null; then echo 'Process did not stop' >&2; exit 1; fi
  fi
fi
"@
    }
}
Stop-Experiment
$sessions = 1
Start-Experiment
Invoke-RestMethod http://192.168.42.42:8080/v1/models -TimeoutSec 30
```

That disables this scheduler with the same binary. For full revision rollback,
stop the experiment again and restore the lease holder's saved production
launch commands/binary on **all ranks**, then health-check and run the known
greedy smoke gate. Only after a working API is restored, remove this exact
`$remote` directory using `Invoke-Step` on each target, and retain/download
its logs before deletion. The original pack, manifest and weights are unchanged.
