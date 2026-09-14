# Ring Decode: Shared-Tile QSA Attention and Expert Dot Cleanup (2026-09-13)

Worktree `fg-work-dec5`, base `bc037de`. Scope: batch-1 decode latency only.
Two shader changes and one diagnostic landed:

1. `fg_qsa_decode_attention_split` - a new decode-specialised split-K QSA
   attention kernel that reuses the prefill batch geometry: one workgroup per
   (kv head, split), 768 threads, twelve query heads share one shared-memory
   copy of each decoded record tile, subgroup-private online softmax, two
   barriers per eight-token tile.  The selected count and split count arrive as
   push constants, so no host-written count tensor is needed.
2. `fg_moe_decode_gate_up` accumulates the integer nibble dot in fp32 instead
   of int (the partials are exact integers, so results are bit-identical) and
   drops the `int(dot(...))` conversion boundary.
3. `runtime.c` can profile the coordinator's own block when
   `FG_DECODE_PROFILE=1` (diagnostic; no default-path change).

Binary lineage: `0fdb5209` (before) -> `23926a51` (QSA kernel only) ->
**`c15ff019`** (final, all eight blades, `start-rank0-ring.sh` /
`start-workers-ring.sh`, ring pack, `FG_PREFILL_RING=1` / `FG_WORKER_OWNER=1`).

The MTP scaffold in the base tree is fully gated behind `--experimental-mtp`
and `FG_MTP_DRAFT_ECHO` plus a sealed manifest component; the deployed pack
seals none, so default decode is unchanged.  No MTP path was enabled.

## TL;DR

The QSA decode attention kernel was the second named decode item at
~5.7 ms/token of GPU time across the twelve QSA layers.  The old kernel scanned
each selected record with one lane per element, three barriers per token and a
serial lane-0 online-softmax update; the new kernel decodes each record once
per kv head for the twelve heads that share it, runs the softmax inside
subgroups, and drops the per-token barrier chain.  Vulkan timestamps:

| kernel | before | after | speedup |
|---|---:|---:|---:|
| `fg_qsa_attention_split` (per call) | 0.43-0.51 ms | **0.10-0.14 ms** | 3.8-4.2x |
| QSA attention, all 12 layers/token | ~5.7 ms | **~1.4 ms** | ~4.1x |

Fleet, same ring pack and env:

| metric | before (`0fdb5209`) | after (`c15ff019`) |
|---|---:|---:|
| 4K prefill (battery) | 278-304 TPS | 285.4 TPS |
| short decode 32 tok (battery) | 20.50 / 20.91 | **21.14** |
| 4K first decode token | 18.53 / 18.79 | 11.26 (cold first token) |
| 4K sustained 40-token listing | 15.55 / 17.43 (thermal band) | **19.45 / 19.32** |
| `tools/pi-stability.ps1` | PASS | **PASS** (soak 128..16387, 4 turns, all ranks) |
| gates | `[12]` / `[Paris]` | `[12]` / `[Paris]` |

The 4K sustained listing was also measured on the *before* binary in this
session's thermal band with the same request and scripts: 15.01-15.76 TPS
(profiled) and 17.14 TPS (trace session).  The new binary measures 19.32-19.45
clean.  The per-kernel deltas are the reproducible part; the fleet band drifts
with soak heat.

## Per-kernel before/after (Vulkan timestamps, `FG_DECODE_PROFILE=1`)

4K steady state, mean of the last 24 profiled blocks, per call:

| kernel | rank | before ms | after ms | delta |
|---|---:|---:|---:|---:|
| `fg_qsa_attention_split` -> `fg_qsa_decode_attention_split` | 1 | 0.508 | **0.136** | -73% |
| (same) | 2 | 0.491 | **0.120** | -76% |
| (same) | 3 | 0.483 | **0.119** | -75% |
| (same) | 4 | 0.448 | **0.118** | -74% |
| (same) | 5 | 0.448 | **0.109** | -76% |
| (same) | 6 | 0.434 | **0.100** | -77% |
| (same) | 7 | 0.433 | **0.105** | -76% |
| `fg_moe_decode_gate_up` | 2 | 0.737 | 0.718 | ~0 (dot4f) |
| `fg_moe_decode_down_reduce` | 2 | 0.469 | 0.468 | ~0 |
| `fg_topk_reduce` (QSA select) | 2 | 0.163 | 0.167 | unchanged |
| block GPU (rank 2) | 2 | 5.14 | 4.70 | -0.44 |

`fg_moe_decode_gate_up` was rewritten to accumulate the nibble dot in fp32
(bit-identical: every partial is an exact integer below 2^24) which removes the
`FToS`/`SToF` round trip in the SPIR-V.  Fleet-measured effect is within noise
(-1%, rank-to-rank sign flips): the kernel is not bound by that conversion, so
the expert pair stays at ~150 GB/s and is left for a geometry rewrite.

## Per-token decode budget (4K steady, clean session)

Rank-0 `RING_DECODE` trace and worker `RING_DECODE_BLOCK` means (392 blocks):

| phase | ms |
|---|---:|
| embed + ngram (rank 0) | 0.26 |
| rank 1 block (layers 0-5) + hop | 5.85 + ~1.0 |
| rank 0 own block (layers 6-11) | 6.20 |
| ranks 2-7 blocks | 5.55 / 6.12 / 5.29 / 5.92 / 5.13 / 5.76 (~34.4 total) |
| output head on rank 4 + hops | ~5.5 |
| **token total** | **~53.5** (18.7 TPS steady) |

The new kernel moves QSA GPU from ~5.7 to ~1.4 ms/token; the remaining QSA
path cost is the selection: `fg_qsa_index_score` 0.02 + `fg_topk_reduce` 0.17
per layer, plus the host readback of the 2 KiB top-512 ids that forces one
submit + fence per QSA layer (`QSA_ATTEND_TRACE select_ms` 0.23-0.45).  Twelve
QSA layers make that ~2 ms of top-k GPU plus ~2.5-4 ms of fence/read wall.

## Remaining gap to 28 TPS / 10 ms per token

The 4K steady token is ~51.5 ms (19.4 TPS); 28 TPS is 35.7 ms (1.44x) and 10 ms
is 5.2x away.  The measured remaining budget, in attack order:

1. **QSA selection, ~4-6 ms/token.**  `fg_topk_reduce` sorts 2048 padded slots
   with one workgroup and 66 barrier-separated bitonic passes (0.167 ms/layer).
   The 2 KiB id readback then forces a fence that drains the block's queued
   GPU work at every QSA layer.  Next steps, in increasing size: (a) a
   partial-selection kernel (top-512 of ~1090 without a full sort); (b) a
   direct-mapped GPU record mirror so the gather consumes block ids straight
   from the selection and no host readback/fence is needed at contexts whose
   records fit the cache (4K fits today: `missing=0 fetched=0` in every
   trace).
2. **Expert pair, ~8.3 ms/token at ~150 GB/s.**  `fg_moe_decode_gate_up` and
   `fg_moe_decode_down_reduce` are instruction/latency bound, not weight
   bandwidth bound (both ~150 GB/s against a 350 GB/s stream ceiling).  The
   fp32-accumulator change was neutral, so the next experiment is a geometry
   rewrite (two rows per lane to halve activation unpack/convert, or
   deeper per-lane unrolling for ILP) with the same cooked Q4_K/Q5_1 layouts.
3. **GDN projections + output, ~11 ms/token at 172-324 GB/s** - documented as
   roofline-adjacent; not retuned here.
4. **GR read chains, ~7.4 ms/token** across the 88 small `dense_q8_0_cooked_r8`
   / `cooked_split` calls; small-K shapes run at 174 GB/s.
5. **rank-0 fixed, ~4-6 ms/token**: output head ~3.4 ms (1.93 ms DRAM bound)
   plus hops.  Splitting the vocabulary GEMM across block owners is a
   protocol-level change.
6. **Host/fence residual ~0.4-1.0 ms per block** (4-8 ms/token), dominated by
   the QSA break.  Item 1(b) attacks most of it.

## What changed

- `shaders/fg_qsa_decode_attention_split.comp` (new): the prefill
  `fg_qsa_attention_split_batch` geometry specialised to a single query.
  `selected_count`/`splits` are push constants; records keep the 1236-byte
  gathered layout; partials keep the `(head*splits+split)*258` layout, so the
  existing `fg_qsa_attention_merge` combines them unchanged.
- `src/qsa.c` `decode_attention`: routes every multi-split decode through the
  new kernel (the serial `fg_qsa_attention` path still covers <=256 selected
  tokens).
- `src/vk.c` + `include/fg_vk.h`: kernel registration, wrapper
  (`fg_vk_qsa_decode_attention_split`, dispatch `(2, splits, 1)`), destroy.
- `shaders/fg_moe_decode_gate_up.comp`: `dot4f` fp32 accumulation.
- `src/runtime.c`: coordinator own-block `DECODE_PROFILE` output (diagnostic
  only; gated on `FG_DECODE_PROFILE`) so rank 0's block time is attributable
  for the first time.

Local oracles: `expert_decode_fused(12)`, `expert_decode_fused(13)`,
`expert_decode_fused_q8_gates`, `qsa_attention`, `qsa_attention_single` all
pass; full-tree build warning-free (`make all -j8`, `-Werror`).

## Shared-file note for the prefill agent (`fg-work-pref5`)

Only two shared files are touched and both edits are decode-only:
`src/vk.c` (QSA decode kernel registration/wrapper) and `src/runtime.c`
(the coordinator `DECODE_PROFILE` block).  `fg_moe_decode_gate_up` and the new
QSA decode shader are not used by prefill.
