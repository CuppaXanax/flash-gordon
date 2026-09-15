# Decode hop payload: inventory, cuts and ceiling (2026-09-15)

Worktree `fg-work-hop13`, branch `perf/hop-payload` (main `5fbeb29`).  Two
changes:

1. the final block owner no longer sends the 40 KiB `FG_MSG_OUTPUT_HIDDEN`
   when the output owner runs the greedy 4-way split (the frame the split
   union never reads), and
2. an opt-in bf16 hop encoding (`FG_FABRIC_HOP_BF16`) for the decode work,
   layer-result and 4-way slice-hidden frames.

The hidden-state suppression is bit-identical by construction; the bf16 path
changes numerics and is off by default (see 4).

## 1. What is in a hop: exact inventory

The decode chain ships one `fg_layer_work` per block boundary and one
`fg_layer_result` for the head input.  The payload is the token's hyper state
only: `FG_HYPER_WIDTH = 10240` f32 = 4 groups x 2560 hidden, no weights, no
QSA pages, no padding.  Protocol 6 text-position layout:

| frame | id | payload | framed | contents |
|---|---|---:|---:|---|
| decode layer work (block boundary) | 44 | 40,984 | 41,016 | 24 B header (`layer`, src, dst, flags, token BE u32, mode, axes, 2 reserved, 3x position u32) + 10,240 f32 hyper |
| decode layer work, layer 0/1 (with n-gram) | 44 | 51,224 | 51,256 | + 16x160 f32 n-gram embedding (10,240 B) |
| decode layer result (final block owner -> rank 0, non-direct) | 45 | 40,968 | 41,000 | 8 B header + token + hyper |
| output hidden (`FG_MSG_OUTPUT_HIDDEN`, direct) | 47 | 40,968 | 41,000 | same `fg_layer_result` layout, dst = output owner |
| output slice (`FG_MSG_OUTPUT_SLICE`, 2-way) | 48 | 40,968 | 41,000 | same layout, dst = 0 |
| 4-way head input (`FG_MSG_OUTPUT_SLICE_HIDDEN`) | 50 | 10,248 | 10,280 | 8 B header + token + 2,560 f32 post-HC hidden |
| output config (control) | 46 | 40 | 72 | sampler route + flags |
| output partial (control) | 49 | 12 | 44 | `(token, value, id)` |
| output result | - | 16 | 48 | `(token, logit)` |

The receiver consumes the whole hyper: the block's first op is
`fg_vk_group_rms_norm(width=2560, groups=4)` over all 10,240 values, then the
HC/GR chain mixes all four groups.  There is no unused lane, no hidden padding
and no precision mix on the wire (all f32 big-endian).  The only provably
redundant bytes are the three text positions, which are three copies of the
token index in decode (8 B/hop, 64 B/token) - not worth a protocol change -
and the `FG_MSG_OUTPUT_HIDDEN` frame on the greedy 4-way path (section 2).

### 1.1 Frames per token

The production contiguous map (rank 1 owns 0-5, rank 0 owns 6-11, rank 2
12-17, rank 3 18-23, rank 4 24-29, rank 5 30-35, rank 6 36-41, rank 7 42-47;
the code falls back to `layer % 8` without a route map) gives **8 work hops**,
not 7, plus one output handoff:

| path | frames | bulk bytes/token |
|---|---|---:|
| default (no split) | 8 work + 1 `OUTPUT_HIDDEN` | 379,368 |
| 2-way split | 8 work + `OUTPUT_HIDDEN` + `OUTPUT_SLICE` | 420,368 |
| 4-way split, before this branch | 8 work + `OUTPUT_HIDDEN` + 4 slice-hidden | 420,488 |
| **4-way split, greedy, this branch** | 8 work + 4 slice-hidden | **379,488** |

At the 1 GbE line rate (125 MB/s) that is 3.04 ms/token of pure wire for the
default and the suppressed 4-way, versus 3.36 ms before - the redundant frame
is 41,000 B = **0.328 ms/token** on a serial chain, plus one send/receive
service (local loopback probe: 0.05 ms send p50, 0.6-0.9 ms receive wall p50).

## 2. The redundant 40 KiB hidden (bit-identical fix)

`worker_output_handoff_flush` only reads `state->hidden` when the sampler is
not greedy (full-head fallback) or the split is inactive.  On the greedy
4-way path the local slice runs from `hidden_slice` and the result is the
folded slice argmax, so the 40 KiB `OUTPUT_HIDDEN` rank 7 sends to rank 4 is
never read - and worse, it serializes before the four 10 KiB slice sends on
rank 7's uplink, delaying rank 4's start by its full wire time.

Change: the coordinator marks the token's first decode work with
`FG_LAYER_WORK_FLAG_OUTPUT_4WAY_GREEDY` when the output slice is 4-way and
`temperature == 0` with no penalties (the same predicate rank 4 evaluates
from the config frame).  Each hop owner copies the bit into the next work
frame, so rank 7 sees it before it runs layers 42-47 and skips the hidden
send.  With the bit absent the path is byte-for-byte today's behaviour
(hidden + HC chain + slices), so an old or mixed peer only loses the saving,
never correctness.  `FG_OUTPUT_SPLIT_TRACE=1` logs
`OUTPUT_SPLIT_STATE ... what=hidden-suppressed`.

Greedy is decided from the same `coordinator->sampler` bits that are sent in
`FG_MSG_OUTPUT_CONFIG`, so rank 7's suppression and rank 4's split branch
cannot disagree; a stale/duplicated token is still bounded by the existing
split waits (loud `OUTPUT_SPLIT_TIMEOUT`/`RING_DECODE_TIMEOUT`, no wedge).

Expected: **-0.33 ms wire, -0.4 ms with the send service**, and rank 4's
slice (and the way-3 slice that gates the combine) leaves ~0.33 ms earlier
because the hidden no longer occupies the uplink first.

## 3. Evaluated, not implemented

| candidate | verdict |
|---|---|
| partial-consume: start the receiver earlier | **not applicable** - the first kernel normalizes all 4 groups and the HC down projection reads the whole normalized vector; no prefix of the 40 KiB is independently consumable. Per-group RMS workgroups exist but the next kernel needs every group. |
| send-before-fence | **no unnecessary fence found** - the hyper read is a mapped memcpy that must follow the block's `vkWaitForFences` in `fg_vk_end`; the frame encode then copies into the send buffer. The one real ordering defect was hidden-before-HC-chain on rank 7, removed by section 2 for greedy. |
| position triple | 8 B/hop, 64 B/token; protocol churn not justified. |
| receiver pre-posted recv | `wait_ms`/`header_ms` on the fleet probe will show whether host wake is material; candidate for the next round if `wait_ms` dominates `payload_ms`. |
| 2.5 GbE | halves all wire bytes (~-1.5 ms) but is hardware. |

## 4. bf16 hop packing (gated, `FG_FABRIC_HOP_BF16=1`)

One env flag on the sending rank switches the decode hop codecs to bf16
round-to-nearest-even with self-describing tags: `FG_LAYER_WORK_FLAG_BF16_HYPER`
(bit 2 of the work flags), reserved byte 1 in `fg_layer_result`, reserved byte
2 in `fg_output_slice_hidden`.  Decoders accept both sizes; older receivers
reject the tag loudly (unknown flag/reserved byte), never silently
mis-parse.  The result/slice encoders now return the wire length; the senders
use it.  The n-gram embedding stays f32 (10,240 B on one frame).  Prefill
frames are untouched.

Byte effect per token (4-way greedy): 8 work frames 338,368 -> 174,528 B,
4 slice-hidden 41,120 -> 20,640 B, total 379,488 -> **195,168 B** =
1.56 ms at 125 MB/s, **-1.47 ms wire/token**.  With the hidden present
(default/non-greedy) the same hop saving applies to the work chain plus
20,480 B on the hidden frame.

Numeric contract: bf16 keeps the fp32 exponent range; round-to-nearest-even
has unit roundoff 2^-8 (3.90625e-3) relative per converted value, so each hop
perturbs every hidden coordinate by at most 0.39% (absolute <= 2^-8 |x|, and
bf16 subnormals below 2^-126).  The per-hop error is bounded; the end-to-end
effect on logits/tokens is **not** analytically bounded and must be gated on
the fleet with the standard gate plus a token/logit comparison against the
f32 control.  The local oracle checks the round-trip bound over the full
10,240-wide vector and exactness for bf16-representable values (0, 1.5, -2.5,
multiples of 1/8).

## 5. Ceiling

If the fleet probe confirms `payload_ms ~ 0.33` per 40 KiB (wire-bound), the
token wire floor is 3.0 ms (default/suppressed 4-way) of the ~40 ms/token at
24.5 TPS; bf16 takes it to 1.56 ms.  Ranked expected ms/token:

| # | candidate | expected | risk |
|---|---|---:|---|
| 1 | drop 4-way greedy `OUTPUT_HIDDEN` (**this branch**) | -0.33 .. -0.45 | bit-identical |
| 2 | bf16 hops (**this branch**, env-gated) | -1.4 .. -1.6 | numerics; needs fleet token gate |
| 3 | receiver wake / pre-posted recv | ? | needs `wait_ms` probe |
| 4 | 2.5 GbE | ~-1.5 | hardware |

If instead the probe shows `payload_ms << 0.33` with large `wait_ms`, the
chain is host-bound, only #1 and #3 pay, and bf16 should stay off.

## 6. Local validation

`make all -j8` warning-free (`-Werror`).  All runs on llvmpipe / loopback.

| oracle | result |
|---|---|
| `test_fabric` protocol selfcheck | PASS - greedy flag round trip, unknown flag/reserved-byte rejection, bf16 work/result/slice-hidden round trip, bf16 tag/size mismatch rejection, error bound over the full hyper vector, env-off byte-identical sizes |
| `test_fabric` eight-process mesh | PASS - decode chain now carries the greedy bit through all 8 hops (rank 7 consumes it), 4-way mesh without a hidden frame (the suppressed shape), bounded-timeout probe unchanged |
| `test_core` | 1 pre-existing failure (line 458), unchanged |
| `test_session` | 2 pre-existing failures (254/257), unchanged |
| `test_qsa_prefill`, `test_prefix`, `test_expert_prefill`, `test_owner_reduce` | PASS |
| focused `test_fg_vk` (memory_telemetry_and_canary, tensor_view_rebind, batch_submission_parity, pipeline_flush_parity, gpu_profile, output_argmax, output_topk, qsa_attention, qsa_attention_single, qsa_record_gather, qsa_quant_and_bf16, qsa_prefill_prepare, qsa_segmented_index_score, qsa_prefill_chunk_liveness) | PASS |
| `FG_FABRIC_PROFILE=1` loopback mesh | 40,984 B frames: send p50 0.050 ms; receive wall p50 0.86 ms.  Loopback host-service only; the fleet probe is the wire measurement. |

## 7. Fleet A/B plan (ops details live in the local workspace)

1. Deploy the same binary to all eight ranks (lockstep; protocol 6, no version
   bump).  Keep `FG_OUTPUT_SPLIT=4` on all ranks so the 4-way path is active.
2. Gates: `[12]` / `[Paris]` with `FG_OUTPUT_SPLIT=4` (suppression active,
   no env needed) and `FG_OUTPUT_SPLIT_TRACE=1` to see
   `what=hidden-suppressed`.
3. Battery: 4-way default vs 4-way with `FG_OUTPUT_SPLIT=0` (or unset) for
   the no-regression control; promotion bar as in
   `PERFORMANCE_OUTPUT_SPLIT_4WAY_2026-09-14.md` section 9.
4. Hop probe: one 32-token request with
   `FG_FABRIC_PROFILE=1 FG_PREFILL_RING=1 FG_DECODE_RING=1 FG_RING_TRACE=1`
   on rank 0 and `FG_WORKER_OWNER=1 FG_FABRIC_PROFILE=1` on workers; confirm
   message type 47/48 disappears from rank 7's send log on greedy tokens and
   read `payload_ms` for 40 KiB frames to settle wire-vs-host.
5. bf16 candidate (separate A/B, after 2-4): rerun the gates and battery with
   `FG_FABRIC_HOP_BF16=1` on **all** ranks.  Gate: gates pass and per-token
   logits/tokens match the f32 control within the run's sampling tolerance;
   any divergence rejects the candidate.
