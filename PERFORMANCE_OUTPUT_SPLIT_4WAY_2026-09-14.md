# Output head: 4-way split + fabric hop payload probe (2026-09-14)

Worktree `fg-work-split4`, branch `perf/output-split-4way` (rebased onto main
`56f76bb`).  Two parts:

1. the 4-way output head split (round-3 `PERFORMANCE_DECODE_OVERLAP_2026-09-14`
   section 3.6 item 5), implemented behind a new mode so the fleet-validated
   2-way stays the default, and
2. the `FG_FABRIC_PROFILE` hop payload inventory and probe plan (section 3.6
   item 1), with per-candidate payoff estimates.

Round 1 (`73152f4`..`c45a50e`) came back from the fleet wedged: the diagnosis
in section 8 shows the wedge is **rank-0 memory exhaustion during 4K ring
prefill**, reproduced on the default config, not the split.  Round 2 adds the
mandated liveness deadlines plus split tracing and lands them on the fleet.

| commit | change |
|---|---|
| `ae3ce0b` | 4-way split mode, slice-hidden protocol, foreign loader, runtime plumbing |
| `b538b42` | layout/view-slice/combine oracles and the eight-process 4-way mesh |
| `52ce7d3` | uniform missing-slice-executor check and this document |
| `788b965` | comment-free style pass on the split code |
| `2c812c0` | round 2: bounded split waits, trace, dropped-partial mesh probe |

## 1. Design: the 4-way split

### 1.1 Mode contract

`FG_OUTPUT_SPLIT` is parsed once per rank at startup by
`fg_output_split_mode()` (`src/protocol.c`):

| value | mode |
|---|---|
| unset / empty / `0` | off (bit-identical default path) |
| `1` or `2` | 2-way (the fleet-validated mode; `FG_OUTPUT_SPLIT=1` unchanged) |
| `4` | 4-way (new) |
| anything else | startup error: `FG_OUTPUT_SPLIT=<v> is not a supported split; use 0 (off), 1 or 2 (2-way) or 4 (4-way)` |

Every consumer parses the mode at rank startup (rank 0 in
`coordinator_output_split_open`, workers in `fg_rank_main`) and passes the
parsed ways through the runtime; there is no per-token `getenv`.  The mode must
be set on every rank that owns a slice (0, 1, 2, 4) and on the final block
owner (7); a mismatch is rejected by the handoff/executor checks with a message
that names both ways, never by waiting.

### 1.2 Slice layout and rank map

`248320 / 4 = 62080` rows per slice, exactly `3880` cooked tiles of 16 rows.
`fg_output_split_span()`/`fg_output_split_rank()`/`fg_output_split_way_for_rank()`
define the layout (rows are tile aligned in both modes):

| way | rank | rows | bytes of `output.weight` (cooked, 43520 B/tile) |
|---|---:|---:|---:|
| 0 | 4 (output owner) | 0 .. 62080 | 0 .. 168,857,600 |
| 1 | 0 (coordinator) | 62080 .. 124160 | 168,857,600 .. 337,715,200 |
| 2 | 1 (helper) | 124160 .. 186240 | 337,715,200 .. 506,572,800 |
| 3 | 2 (helper) | 186240 .. 248320 | 506,572,800 .. 675,430,400 |

2-way stays exactly as qualified: way 0 = rank 4 `[0, 140048)`, way 1 = rank 0
`[140048, 248320)` (8753 + 6767 tiles).  The 4-way slice assignment was chosen
so the two foreign slices land on ranks 1 and 2: they are the first two chain
owners and therefore idle when the head runs, while rank 7 must stay free for
the next token's first hop.  Sealed-weight sizes on the recorded 4K session
(`ab-20260913-204953`) are 9.49/9.45/9.47/9.45/9.40/9.60 GiB for ranks
1/2/3/5/6/7, so any worker fits the +169 MB slice; 1 and 2 are the natural
first two.

### 1.3 Payload flow: hidden broadcast, not hyper fanout

A naive 4-way would fan the 40 KiB hyper out to four slice ranks.  The
BC-250s carry 1 GbE (RTL8111H), so four serialized 40 KiB sends would cost
~1.3 ms/token on rank 7's uplink and eat the whole GEMM win.  The implemented
flow instead runs the head's HC chain **once** on the final block owner and
ships the 10 KiB head input:

```
rank 7 (final block owner, after layer 47):
    read 40 KiB hyper (existing)
    -> fg_output_hc_run()                 # HC chain, ~0.10 ms
    -> FG_MSG_OUTPUT_SLICE_HIDDEN (10248 B) to ways 0..3   (4 sends)
rank 4 (way 0): store head input, run its slice, wait for 3 partials, combine
rank 0 (way 1): run its slice from the head input, return partial
rank 1 (way 2): run its slice from the head input, return partial
rank 2 (way 3): run its slice from the head input, return partial
```

Every slice therefore reduces the *same* head input bits, and no rank re-runs
the HC chain on the split path.  Wire bytes per token are unchanged from the
2-way (4 x 10 KiB replaces the 40 KiB slice frame), and rank 4 starts its GEMM
after 10 KiB instead of 40 KiB.  The 40 KiB `FG_MSG_OUTPUT_HIDDEN` is still
sent to rank 4 because rank 7 cannot see the sampler config; rank 4 falls back
to the full head for sampling/penalties exactly like the 2-way (the hidden
slice is then dropped).

Protocol additions (`include/fg_protocol.h`, `src/protocol.c`):

* `FG_MSG_OUTPUT_SLICE_HIDDEN = 50` (valid on protocol 6) and
  `FG_OUTPUT_SLICE_HIDDEN_BYTES = 8 + 2560*4 = 10248`; the frame is
  `source, destination, reserved0, reserved0, token BE, 2560 f32 BE`.  The
  decoder rejects a wrong size, any non-zero reserved byte, a self route,
  bad ranks, or a non-finite head value.
* `FG_OUTPUT_CONFIG_FLAG_SPLIT_4 = 2u` on the config frame; a config with
  flag 2 but not flag 1 is rejected, so the way count is authenticated on rank
  4 before the first token.
* The partial frame (`FG_MSG_OUTPUT_PARTIAL`, 12 B) is reused unchanged; the
  handoff now records up to `FG_OUTPUT_SPLIT_WAYS_MAX` partials with their
  source rank and validates each id against that source's slice span.

### 1.4 Foreign tensor loader

Ranks 1 and 2 do not hold the output bundle in their own shard.  A new
`fg_output_slice_create_foreign()` (and `fg_output_hc_create(...,foreign=true)`
for the final owner's HC chain) reads the needed byte spans directly out of
the **owning rank file** (`rank-04.fgw`) at the manifest offsets:

* helper slice: 168,857,600 B (161 MiB) for ways 2/3;
* final-owner HC chain: 7,004,160 B (6.7 MiB: `output_hc_norm/down/up`);
* the file may be sparse: only the needed extents have to exist.  A missing or
  short file fails at startup with the exact offset/length and the fix:
  `open <dir>/rank-04.fgw for foreign output slice (offset N bytes M): ...;
  FG_OUTPUT_SPLIT=4 needs the owning rank file on every slice rank`.

The weights are byte-identical to the pack (the manifest SHA covers them), and
the foreign tensor keeps the record's format (`Q8_0_COOKED`), so the slice runs
the same `fg_dense_q8_0_cooked` kernel as the full head.

### 1.5 Exact combine semantics

The full-vocabulary argmax is a max under the total order implemented by
`fg_output_better` (the shader `before()` rule):

1. padding (`id == 0xffffffff`) loses to any real row;
2. a non-finite value (NaN or Inf) beats a finite one;
3. among non-finite candidates the lower id wins; among finite candidates the
   higher value wins and equal values go to the lower id.

Each slice returns its own max under that order, and the four spans partition
`[0, 248320)`, so folding the four slice maxima with `fg_output_combine` is the
exact global argmax.  The fold is order independent (total order); rank 4
folds `local, remote[0..n]` in arrival order, and the oracle checks three
different fold orders against the flat reference.  `remote_count == ways-1` is
required before the result is sent; each partial is validated against the
source rank's slice span before folding.

### 1.6 Failure modes (loud, never hang)

| misconfiguration | behavior |
|---|---|
| invalid `FG_OUTPUT_SPLIT` value | every rank that parses it exits at startup with the mode message |
| rank 0 says 4-way, rank 4 holds a 2-way slice | rank 4: `output config declares a 4-way split but this rank holds a 2-way slice executor ...` on the first token |
| a slice rank holds no executor | uniform check `rank N needs a 4-way slice executor but holds none (FG_OUTPUT_SPLIT must match on all ranks)`; rank 4 then sees the peer close |
| a slice rank holds the other mode's executor | `rank N holds a X-way slice executor for a Y-way output slice ...` |
| final owner has no HC executor | `rank N is the final block owner without a 4-way output HC executor` |
| missing `rank-04.fgw` extent | startup error with the exact offset/length |
| stale/duplicate partial after the token completed | dropped (no error) |

`FG_OUTPUT_SPLIT=0`/unset and `=1` are behavior-identical to main: no slice
executor is created, no branch is taken, the 2-way frames and sequence
numbers are unchanged.

## 2. Expected deltas

Head time on the critical path (measured pieces: full GEMM 2.620 ms, HC chain
0.101 ms, argmax 0.020 ms; 1 GbE floor 40968 B / 125 MB/s = 0.33 ms per
40 KiB frame, 0.082 ms per 10 KiB frame):

| | 2-way (today) | 4-way (this branch) |
|---|---:|---:|
| rank 4 slice rows | 140048 (1.477 ms) | 62080 (0.655 ms) |
| rank 0 slice rows | 108272 (1.142 ms) | 62080 (0.655 ms) |
| HC chain | 0.101 ms on rank 4 and rank 0 | 0.101 ms on rank 7, once |
| first payload to rank 4 | 40 KiB (~0.33 ms) | 10 KiB (~0.08 ms) |
| last slice start (rank 2) | - | ~0.25 ms of sends after the HC chain |
| **estimated head wall** | **~1.9-2.0 ms** | **~1.0-1.2 ms** |
| wire bytes/token (all bulk frames) | 410,000 B | 410,120 B |

Estimate: **-0.4 .. -0.6 ms/token** over the 2-way at short/4K, matching the
round-3 estimate.  The gain is the GEMM quartering (1.48 -> 0.66 on the
critical rank), minus rank 7's one extra HC chain and the four 10 KiB sends.
Rank 0's post-token GPU work also drops from 1.14 ms to 0.66 ms, which is not
on the token critical path but frees the coordinator between tokens.

The model is sensitive to what `FG_FABRIC_PROFILE` will measure.  If a 40 KiB
hop is wire-bound (0.33 ms) the estimate holds; if per-hop latency is dominated
by host wake/handoff (the recorded `first_hop_ms` ~7.8 vs rank-1 GPU ~6.6), the
4-way win is *larger* (less payload and one fewer 40 KiB frame on the critical
path).  If sends overlap GPU work more than modeled, the win is smaller.
Section 5 has the candidates that move the wire bytes themselves.

## 3. Local validation (llvmpipe, correctness only)

`make all -j8` warning-free (`-Werror`); `make tests/test_fg_vk tests/test_fabric
tests/test_qsa_prefill tests/test_expert_prefill tests/test_owner_reduce`.

| oracle | result |
|---|---|
| `output_split_layout` (new) | PASS - mode parse for unset/0/1/2/4 and the loud error for 3/four; way/rank map for both modes; spans tile aligned and exactly partition the vocabulary |
| `output_split_combine` | PASS - unchanged 2-way oracle |
| `output_split_combine_4way` (new) | PASS - four slice maxima folded in 3 orders equal the flat argmax for ties, NaN, Inf, all-equal and last-row maxima |
| `q8_cooked_view_slice` | PASS - unchanged |
| `q8_cooked_view_slice_quad` (new) | PASS - each of the four 32-row views of a cooked matrix is bit-identical to the full GEMM rows |
| `test_fabric` protocol selfcheck | PASS - `FG_OUTPUT_CONFIG_FLAG_SPLIT_4` round trip, flag 2-without-1 rejected, hidden-slice codec reserved/finite/route rejection, `FG_MSG_OUTPUT_SLICE_HIDDEN` frame validation, 4-way handoff readiness and partial bookkeeping, and the uniform missing-executor error (`fg_output_split_require_slice`) |
| `test_fabric` eight-process mesh (new `output_split4_roundtrip`) | PASS - rank 7 broadcasts the head input to ranks 4/0/1/2, three partials return from 0/1/2, rank 4 validates spans and sends the combined result |
| `test_qsa_prefill`, `test_expert_prefill`, `test_owner_reduce` | PASS |
| focused `test_fg_vk` (batch_submission_parity, pipeline_flush_parity, gpu_profile, output_argmax, output_topk, qsa_attention*, qsa_record_gather, qsa_quant_and_bf16, qsa_prefill_prepare, qsa_segmented_index_score, qsa_prefill_chunk_liveness, memory_telemetry_and_canary, tensor_view_rebind) | PASS |
| `test_core` | 1 pre-existing failure (line 457), unchanged |
| `test_session` | 2 pre-existing failures (254/257), unchanged |
| `test_prefill_dispatch` | pre-existing failures (178-183), unchanged |

The full `test_fg_vk` suite was not run (known llvmpipe `qsa_record_commit`
crash per the round brief).

## 4. Fleet A/B plan

### 4.1 Prep: the foreign `rank-04.fgw` extents

Rank 1/2/7 do not carry `rank-04.fgw`.  On the fleet pack
`/home/user/fg-ring-pack` the output bundle is contiguous at the start of
`rank-04.fgw` (`output.weight` off 0, 675,430,400 B; `output_hc_norm/down/up`
at 678,912,000/675,430,400/678,952,960).  The needed extents are:

| rank | way | extent in `rank-04.fgw` | file size to keep |
|---|---:|---|---:|
| 1 (.43) | 2 | `[337715200, 506572800)` (168,857,600 B) | 506,572,800 |
| 2 (.44) | 3 | `[506572800, 675430400)` (168,857,600 B) | 675,430,400 |
| 7 (.49) | HC chain | `[675430400, 682434560)` (7,004,160 B) | 682,434,560 |

Build one sparse copy per blade from a full source (`/home/user/fg-ring-pack`
on .42 or .46 has one) and scp it into the pack directory:

```bash
SRC=/home/user/fg-ring-pack/rank-04.fgw
# way 2 -> rank 1
truncate -s 506572800 /tmp/rank04.fgw
dd if=$SRC of=/tmp/rank04.fgw bs=8M iflag=skip_bytes,count_bytes \
   oflag=seek_bytes skip=337715200 count=168857600 seek=337715200 \
   conv=notrunc,sparse
scp /tmp/rank04.fgw xander@192.0.2.43:/home/user/fg-ring-pack/rank-04.fgw
# way 3 -> rank 2
truncate -s 675430400 /tmp/rank04.fgw
dd if=$SRC of=/tmp/rank04.fgw bs=8M iflag=skip_bytes,count_bytes \
   oflag=seek_bytes skip=506572800 count=168857600 seek=506572800 \
   conv=notrunc,sparse
scp /tmp/rank04.fgw xander@192.0.2.44:/home/user/fg-ring-pack/rank-04.fgw
# HC chain -> rank 7
truncate -s 682434560 /tmp/rank04.fgw
dd if=$SRC of=/tmp/rank04.fgw bs=8M iflag=skip_bytes,count_bytes \
   oflag=seek_bytes skip=675430400 count=7004160 seek=675430400 \
   conv=notrunc,sparse
scp /tmp/rank04.fgw xander@192.0.2.49:/home/user/fg-ring-pack/rank-04.fgw
```

`du` on each file shows only the copied extent (161 MiB / 6.7 MiB).  If the
pack is rebuilt, the loader's startup error carries the new offset and length;
substitute them.  The full manifest SHA is unchanged (`rank-01..07.fgw` are
untouched), so the other ranks boot exactly as today.

### 4.2 Command sequence

1. Deploy the standard chain (quiesce, patch tar, build on .42, distribute,
   restart rank 0 first, then workers) with `FG_OUTPUT_SPLIT` set to the mode
   under test on **all ranks** (`start-rank0-ring.sh` / `start-workers-ring.sh`
   currently export 1; make a `-split4` pair that exports 4).
2. Confirm each startup log contains the slice line:
   `OUTPUT_SPLIT rank=0 ways=4 way=1 rows=62080..124160`,
   rank 1/2 `slice=1`, rank 4 `way=0`, rank 7 `slice=0 hc=1`.
3. Correctness gates, both modes (same pack, drop caches first):
   `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"` -> `[12]` /
   `[Paris]`.
4. Like-for-like battery:
   `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 2`
   * 2-way control: `FG_OUTPUT_SPLIT=1`
   * 4-way: `FG_OUTPUT_SPLIT=4`
   * default control (no env) for the no-regression check.
   Promotion bar: 4-way short/4K decode clearly above the 2-way, gates and 4K
   prefill (>= 270) unchanged, default/2-way not below their bands.
5. Context sweep on the winner:
   `pwsh -NoProfile -Command "& 'D:\workspace\flash-gordon\tools\context-sweep.ps1' -Contexts 16384,32768"`.
6. Hop probe on the 4-way build (section 6).

If rank 4 exits with the split-mismatch message, the env did not reach every
rank; if a helper exits with "without a 4-way slice executor", its own start
script missed `FG_OUTPUT_SPLIT=4`; if a helper exits at startup with the
foreign-file message, the rank-04.fgw prep missed that blade.  Do not mix
modes across ranks.

## 5. Hop payload inventory (task 2)

### 5.1 What is in a 40 KiB hop

Every ring hop is one `fg_layer_result` frame (`FG_DECODE_LAYER_RESULT_BYTES`,
`src/protocol.c`):

| field | bytes | notes |
|---|---:|---|
| `layer`, `source_rank`, `destination_rank`, reserved | 4 | reserved must be 0 |
| `token_index` (BE) | 4 | |
| `hyper[10240]` (f32 BE) | 40960 | `FG_Q38_HYPER_WIDTH = 4 groups x 2560 hidden` |
| payload total | 40968 | |
| frame header (magic/version/type/bytes/request/sequence/flags/crc32c) | 32 | CRC32C over the payload |
| **framed** | **41000** | |

The 4-way head frame carries only `hidden[2560]`: 8 + 10240 = 10248 payload,
10280 framed.  The partial is 12 B on the control channel.

The hop payload is *only* the token's hyper state: no weights, no QSA pages
(those ride their own QSA messages during prefill), no router/expert data (the
decode block does expert routing locally).  Precision is fp32 and there is no
padding; the 40960 B is exactly 4 groups x 2560 x 4 B.

### 5.2 How many hops per token

With the contiguous ownership map (rank1 0-5, rank0 6-11, rank2 12-17, rank3
18-23, rank4 24-29, rank5 30-35, rank6 36-41, rank7 42-47):

| # | frame | bytes |
|---|---|---:|
| 1-8 | rank0->1, 1->0, 0->2, 2->3, 3->4, 4->5, 5->6, 6->7 | 8 x 41000 |
| 9 | rank7->4 `FG_MSG_OUTPUT_HIDDEN` | 41000 |
| 10 | 2-way: rank7->0 `FG_MSG_OUTPUT_SLICE` / 4-way: rank7->4,0,1,2 `FG_MSG_OUTPUT_SLICE_HIDDEN` | 41000 / 4 x 10280 |

Per token: **9 chain+output 40 KiB frames** (not 7) plus the split frame;
default 369,000 B, 2-way 410,000 B, 4-way 410,120 B.  At the 1 GbE line rate
that is ~2.95 / 3.28 / 3.28 ms of wire per token if serialized, which is why
the 4-way was designed byte-neutral rather than as a hyper fanout.

### 5.3 What can be reduced or overlapped

| # | candidate | bytes saved/token | est. ms/token (1 GbE) | risk / work |
|---|---|---:|---:|---|
| 1 | drop the redundant 40 KiB `FG_MSG_OUTPUT_HIDDEN` when the token is greedy and the split covers the head | 41,000 | -0.33 | needs a "greedy split active" flag propagated with the decode work (rank 0 knows the sampler; rank 7 does not today). Sampling requests must keep the hyper fallback |
| 2 | bf16 hyper hops (8 chain + output frames) | ~164,000 | -1.3 | changes decode numerics (not bit-identical); needs gates + a per-request A/B; fp16 is the same size, only bf16 halves |
| 3 | q8_0-quantize the hyper hops | ~300,000 | -2.4 | larger error than bf16; plausible only with a correctness gate |
| 4 | reorder rank 7's sends: HC chain -> slices -> 40 KiB hidden | 0 | -0.3 .. -0.4 on the split path | one-line ordering change, but delays the hyper for sampling fallback by ~0.3 ms |
| 5 | split the hop payload so the receiver starts early | 0 | 0 | not applicable: the first op (`group_rms_norm` over all 4 groups) needs the whole hyper; there is no natural split |
| 6 | send-before-fence | 0 | small | rank 7 already sends immediately after the block read; the 40 KiB memcpy to the frame buffer is ~10-20 us |
| 7 | 2.5 GbE link | 0 (faster wire) | wire time /2.5 (~-2.0 total) | hardware |
| 8 | eliminate the 2-way's duplicate 40 KiB (already done by 4-way's 10 KiB slices) | -0.4 payload but +3 control partials | ~0 | implemented |

Items 1-4 are host/protocol changes in the split path; 2 and 3 change numerics
and need the same gate/A-B discipline as any kernel change.  Item 4 is the
cheapest measurable follow-up if the probe confirms rank 7's send order is on
the critical path.

## 6. `FG_FABRIC_PROFILE` fleet probe (orchestrator runs; vk-scheduler owns the fleet)

The probe is compiled in (`src/fabric.c`) and disabled unless the env is set
**before process start** (the flag is cached at first use).  Every send and
receive prints one JSON line:

```json
{"schema":"flash-gordon.profile","version":1,"record_type":"fabric_service",
 "local_rank":7,"peer":4,"fabric_class":"bulk","direction":"send",
 "message_type":47,"payload_bytes":40968,"framed_bytes":41000,
 "wall_ms":0.31,"mode":"io_uring","batch_count":1,"shared_batch":false,"status":0}
{"...","direction":"receive","message_type":44,"payload_bytes":40968,
 "framed_bytes":41000,"wall_ms":0.35,"mode":"io_uring","wait_ms":0.02,
 "header_ms":0.01,"payload_ms":0.31,"validation_ms":0.01,"status":0}
```

Probe procedure (one session, no code change):

1. Start rank 0 with `FG_FABRIC_PROFILE=1` **and** the ring envs
   (`FG_PREFILL_RING=1 FG_DECODE_RING=1 FG_RING_TRACE=1`), the 4-way split env
   if the split is under test, and the usual start script.  Start workers with
   `FG_WORKER_OWNER=1 FG_FABRIC_PROFILE=1` (at minimum ranks 1, 4 and 7; all
   eight is fine, the lines are small).
2. Run one 4K request with 32 decode tokens, e.g. the attach battery
   (`Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 1`) or a direct
   `/v1/completions` with `max_tokens=32`.
3. Collect: `grep -h fabric_service ep-rank-*.log | python3 tools/...` or
   `jq -c 'select(.record_type=="fabric_service")'` into one file per rank.
4. Read it against the frame table:

| what | where | how to read |
|---|---|---|
| hop wire time | `payload_ms` on `message_type` 44/45/47 (40 KiB bulk frames) | compare with the 1 GbE floor 0.33 ms.  ~0.33 = wire-bound; much less = chunks are pipelined; much more = host/copy |
| hop latency | `wait_ms` on the receiver | host wake/poll, not wire |
| chain hop overhead | `RING_DECODE first_hop_ms` minus rank 1 `DECODE_PROFILE gpu_ms`/`own_run_ms` | isolates payload+host per hop (recorded ~1.2 ms in round 3) |
| split frames | `payload_bytes` 10248 (`message_type` 50) vs 40968 (`48`) | confirms the 4-way flow; partials (12 B) show as control |
| tail | `RING_DECODE tail_ms` | total post-own-block time; A/B 2-way vs 4-way |

A/B: run the same 32-token request with `FG_OUTPUT_SPLIT=1` and `=4` (same
profile envs) and diff the `tail_ms`/`output_ms` plus the per-frame
`payload_ms` sums.  The probe answers the one question the 4-way design
assumes: whether the 40 KiB hops are wire-bound (then items 1-3 in 5.3 pay)
or latency-bound (then item 4 and the hidden broadcast pay).

Expected probe outcomes and what they would change:

* `payload_ms ~ 0.33` for 40 KiB: wire-bound at 1 GbE.  bf16 hops (item 2)
  become the largest single decode lever (~-1.3 ms/token) and worth a numerics
  A/B; link upgrades (item 7) stack on top.
* `payload_ms << 0.33` with large `wait_ms`: latency/host-bound; the wire work
  in 5.3 is not worth the numerics risk, and the remaining hop floor is the
  per-hop host turnaround (send-before-fence, receiver wake), not bytes.

## 7. Risks / open items

* The 4-way split's only unqualified assumption beyond the 2-way is that the
  HC chain output is identical wherever it runs; it is the same kernel
  sequence on the same weights and bits, and the slice GEMMs are covered by
  the bit-identity oracles, but the fleet gate is the proof.
* Helper ranks 1/2 pay ~169 MB RSS and a one-time sparse-file pread at startup;
  rank 7 pays ~7 MB.  The 2-way fleet headroom numbers say this fits, but
  watch rank 1 (9.49 GiB sealed + 3.91 GiB n-gram) at first boot.
* The 40 KiB `FG_MSG_OUTPUT_HIDDEN` is redundant on the greedy split path;
  dropping it (5.3 item 1) needs the greedy marker to ride the decode work
  chain and is deliberately left out of this round.
* The foreign extents in 4.1 are computed from the current ring manifest
  snapshot; a repack changes them and the loader error is the source of truth.

## 8. Round 2: the fleet wedge - root cause, liveness deadlines, evidence

### 8.1 Root cause: rank-0 memory exhaustion, not the split

The 4-way fleet run (`FG_OUTPUT_SPLIT=4` on all ranks, slices planted) passed
the correctness gates twice and then the battery's 4K request wedged rank 0.
Rank 0's log shows where: every stalled stage is **rank 0's own prefill block**
(`RING_OWN_BLOCK layers=6..11`), a code path the split never touches:

```
RING_STAGE chunk=0 layer=6 in_flight=8 t=798.514
RING_OWN_BLOCK chunk=0 layers=6..11 ms=55448.4     <- 55 s
RING_OWN_BLOCK chunk=2 layers=6..11 ms=135882.5    <- 136 s
RING_OWN_BLOCK chunk=11 layers=6..11 ms=690023.3   <- 11.5 min
...
double free or corruption (!prev)                  <- rank 0 dies
```

The same pathology reproduces **without the split**: the default-config
recovery run served a 21-token prompt with `RING_OWN_BLOCK ... ms=93510.1`
(93.5 s).  `.42`'s dmesg shows global OOM kills during both runs
(systemd-userwork/crond/systemd-userdbd at 02:01, NetworkManager at 02:59);
`free -m` during the run shows rank 0 at ~160 MB free (`used 15180` of
15198) with the 8 GB zram swap in active use.  Rank 0's Vulkan ledger is
`final_requested=15.53 GiB` on a 15.98 GiB UMA device, so the 4K/16K prefill
scratch pushes the box into swap thrash; blocks take tens of seconds to
minutes, SSH starves (sshd cannot fork), and rank 0 eventually crashes with
heap corruption.  The 4-way slices change rank 0's memory by less than 1 MB
(rank 0's slice shrank from 108272 to 62080 rows; the weight is a view of the
replicated arena), and no split timeout fired because the stall never reached
the split path.

Conclusion: the incident was a pre-existing rank-0 memory/swap pathology
triggered by the 4K ring prefill.  It is not a 4-way partial/combine deadlock.
It belongs to the memory workstream (rank-0 headroom; zram pressure); the
mandated liveness deadlines below are what keep any *split* wait from ever
looking like this again.

### 8.2 Liveness deadlines (`2c812c0`)

`FG_OUTPUT_SPLIT_TIMEOUT_MS` (default 4000, clamp 100..60000) bounds every
4-way wait; the default and 2-way paths keep the untimed calls unchanged.

| wait | owner | behaviour on expiry |
|---|---|---|
| partial wait | output owner (rank 4) | the worker loop polls with the remaining budget; on expiry it logs `OUTPUT_SPLIT_TIMEOUT rank=4 token=N waited_ms=M ways=4 output split timed out after M ms waiting for partials from ranks X,Y` and returns `FG_ERR_LIMIT` (the request fails; no wedge) |
| result wait | coordinator (rank 0) | the 4-way decode loop receives with a deadline of `timeout + 4000 ms`; on expiry it logs `RING_DECODE_TIMEOUT rank=0 token=N waited_ms=M expected=OUTPUT_RESULT peer=4` and fails the request |

`fabric.c` gains `fg_fabric_wait_ready_timeout()` and
`fg_fabric_recv_any_timeout()` (additive; `-1` is the old infinite wait), and
`fg_output_split_wait_remaining_ms()` / `fg_output_split_timeout_error()`
build the deadline math and the missing-rank message.  Diagnostics are behind
`FG_OUTPUT_SPLIT_TRACE=1`, which logs every split hop
(`OUTPUT_SPLIT_STATE rank=R token=T what={hc,slice,local,partial,combined}`),
so a future stall names its hop even when no timeout fires.  Unit proof:
`test_fabric` now runs the 4-way mesh over 4 tokens with per-token acks and a
dropped-partial probe where rank 4's bounded wait fires, names rank 1, and
rank 0's backstop ends its result wait; the protocol selfcheck covers the
budget helper and the timeout message.

### 8.3 Fleet evidence (round 2)

* Deployed binary `11ee2352bb063c842636c15d...` (patch `019de49b...`,
  5 files) on all eight blades.  Startup confirms `OUTPUT_SPLIT rank=0 ways=4
  way=1`, rank 1/2 `slice=1`, rank 4 `slice=1`, rank 7 `slice=0 hc=1`.
* Correctness gates with the 4-way active: `[12]` / `[Paris]`.
* Split trace on the gate traffic: rank 7 `what=hc`, rank 4 `what=local` +
  three `what=partial` (ranks 0,1,2) + `what=combined` per decode token; no
  timeouts.
* Short decode, same session, 32 tokens, identical prompt: 4-way
  **20.20 / 20.44 / 20.35 TPS** (wall 2.78/2.69/2.61 s) vs default
  **19.23 / 19.90 / 19.87 TPS** (wall 2.80/2.65/2.65 s).  Not a controlled
  battery A/B (the 4-way runs came first, so thermal drift can only favour the
  later default runs); it bounds the 4-way at or above the default here.
* The mandated battery (`Measure-FlashGordonAB.ps1 -Attach -Build ep
  -Runs4k 2`) could **not** be completed: its 4K/16K prefills wedge rank 0 on
  both configs (section 8.1), so pi-stability was skipped for the same reason.
  The fleet was recovered and left on the **default** config with gates
  `[12]`/`[Paris]` (rank 0: `FG_PREFILL_RING=1 FG_DECODE_RING=1
  FG_RING_TRACE=1`; workers: `FG_WORKER_OWNER=1`; no `FG_OUTPUT_SPLIT`).

### 8.4 Fleet A/B commands (unchanged, plus round-2 envs)

The 4.1 foreign-extent prep and the 4.2 restart/gate/battery sequence stand.
Until rank 0 has headroom, treat the 4K/16K battery as a memory experiment:
`sync; drop_caches` on all ranks first, run one case at a time, and watch
`.42`'s `free -m`; a wedge is recovered by killing the workers so rank 0 exits
(never leave it wedged).  For split diagnosis set
`FG_OUTPUT_SPLIT_TRACE=1` and optionally `FG_OUTPUT_SPLIT_TIMEOUT_MS=3000` on
all ranks: the trace names the stalled hop and the timeout names the missing
rank instead of hanging.
