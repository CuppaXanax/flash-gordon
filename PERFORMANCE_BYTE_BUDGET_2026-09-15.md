# Performance Byte Budget for Decode (2026-09-15)

Byte-ledger audit of the pre-MTP ring decode: what each rank actually reads and
writes per token, the implied time floor, and whether 50-60 TPS is reachable
without changing the pack. Computed from the **deployed pack manifest** and the
source of record, not from prior docs. All pre-existing figures were treated as
hypotheses and re-derived; two of them are corrected below.

Worktree: branch `perf/byte-budget` off `main` (682e326). Fleet was read-only
throughout (manifest hash, pack listing, log greps); nothing was restarted,
rebuilt, or written on the blades.

## 0. TL;DR

* **True per-token traffic is 6.67 GB** of DRAM/UMA bytes (weights 6.34 GB,
  state 0.33 GB) plus 379,532 B of ring wire frames and ~64 KB of n-gram
  O_DIRECT reads on the coordinator.
* **At the measured 353 GB/s clpeak the absolute floor is 18.9 ms/token
  (52.9 TPS)**, and that assumes every kernel runs at 100% of peak.
* **At the measured per-shape kernel rates (92-298 GB/s, expert pair
  133-217 GB/s) the practical floor is ~32-42 ms/token (24-31 TPS)** before
  hop-wire time.
* Current measured decode is **39-43 ms/token (23.4-24.6 TPS)** - about 46% of
  the clpeak-equivalent floor and roughly 78-100% of the practical floor band
  (central ~86%). The binary is already close to its own byte band; remaining
  headroom is bytes, not scheduling.
* **Answer: 50-60 TPS pre-MTP is above the byte roofline.**
  - 60 TPS = 16.7 ms => 6.67 GB / 16.7 ms = 400 GB/s needed, *above clpeak*.
  - 50 TPS = 20 ms => 334 GB/s sustained on every category (94% of clpeak);
    measured categories run at 130-260 GB/s, so ~40% of the bytes must go
    (~4.0 GB/token) before 50 TPS is even arguable.
  - Absolute ceiling at clpeak: **52.9 TPS** (unreachable). Practical ceiling
    at measured rates: **~24-31 TPS**.
* Dominant byte category is **dense weights (62%)**, not experts (23%).
  The biggest single lever is dense requantization, not expert work.

## 1. Method and provenance

* The deployed pack manifest is byte-identical (sha256
  `1368cd6d12592b8f04c969a5c2a881d133d1ff1ad6c405e8c4d7d83ee0bc1956`,
  870,984 B, mtime 2026-09-12) to a local copy that was parsed offline with a
  struct-exact reader for `fg_manifest` (`include/fg_manifest.h`; offsets magic
  0, format 8, tensor_count 60, rank records 184, tensors 50,536 stride 200,
  ngram shards 870,472). The pack shard files equal `ranks[r].persistent_bytes`
  exactly (10,623,664,128 / 10,723,561,472 / 9,954,455,552 / 9,948,233,728 /
  10,636,890,112 / 10,210,377,728 / 9,954,455,552 / 10,472,521,728), so the
  manifest is the deployed pack.
* Model facts from the manifest: 48 layers, 512 experts/layer, `top_k = 10`,
  max context 262,144, expert-parallel ring with **6 whole layers per rank**
  (rank 1 owns layers 0-5, rank 0 owns 6-11, ..., rank 7 owns 42-47). All 512
  experts of a layer live on that layer's owner (layer_groups all equal to the
  layer owner); the "experts per rank = 128" constant in `fg.h` is not the
  deployed layout.
* QSA schedule: every 4th layer is QSA (`(l&3)==3`, `src/q38_schema.c:325`):
  3,7,...,47 = 12 QSA layers, 36 GDN layers.
* Byte constants verified in code: hyper width 10240
  (`include/fg_protocol.h:18`, `FG_HIDDEN_SIZE*4` with `include/fg.h:19`),
  n-gram 16 heads x 160 wide, 90 B/row (`include/fg_ngram.h:10`), QSA token
  record 1236 B = key 544 + value 544 + index 136 + position 12
  (`include/fg_q38_schema.h:30-34`), QSA page = 4 records = 4944 B, GDN conv
  state 163,840 B and recurrent state 48*128*128*4 = 3,145,728 B
  (`include/fg_protocol.h:81-82`), frame header 36 B
  (`typedef fg_frame_header`, `include/fg_protocol.h:207`).
* Context for the headline numbers: T = 4096 committed tokens (the 4K warm
  decode battery). T-dependence is given where it matters.

## 2. Per-rank byte ledger (T = 4096), MB/token

Weight categories come from summing the manifest common tensors of the rank's
own 6 layers; expert bytes are `10 * (gate+up+down)` per layer with the manifest
per-expert sizes; state is read+write per token; transport is the rank's own
sends including 36-B fabric headers.

| rank | layers | dense weights | experts (top-10) | GDN+PLE state R/W | QSA state (T=4K) | output bundle | other | total MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6-11 | 511.0 | 184.3 | 26.5 | 15.85 | 0.0 | 0.232 | 737.8 |
| 1 | 0-5 | 552.3 | 198.7 | 33.8 | 7.92 | 0.0 | 0.041 | 792.7 |
| 2 | 12-17 | 517.2 | 184.3 | 33.1 | 7.92 | 0.0 | 0.041 | 742.6 |
| 3 | 18-23 | 511.0 | 184.3 | 26.5 | 15.85 | 0.0 | 0.041 | 737.6 |
| 4 | 24-29 | 517.2 | 184.3 | 33.1 | 7.92 | 684.4 | 0.041 | 1427.0 |
| 5 | 30-35 | 511.0 | 189.4 | 26.5 | 15.85 | 0.0 | 0.041 | 742.8 |
| 6 | 36-41 | 517.2 | 184.3 | 33.1 | 7.92 | 0.0 | 0.041 | 742.6 |
| 7 | 42-47 | 511.0 | 194.6 | 26.5 | 15.85 | 0.0 | 0.041 | 747.9 |
| **fleet** | 48 | **4147.7** | **1504.3** | **239.1** | **95.1** | **684.4** | **0.5** | **6671.0** |

* Rounding: rank totals are the exact byte sums (737.84, 792.74, 742.57,
  737.65, 1426.99, 742.77, 742.57, 747.89 MB); fleet total is exactly
  **6,671,014,220 B = 6.671 GB**. Weights alone: 6.337 GB (dense layers 4.148,
  experts 1.504, head bundle 0.684).
* `experts`: per-layer per-expert triples are 3,072,000 B (Q4_K gate + Q4_K up
  + Q5_1 down) for most layers, 3,993,600 (Q5_K/Q5_K/Q8_0) at layer 2,
  3,584,000 (Q4_K/Q4_K/Q8_0) at layers 4, 30, 46, 47 - read from the manifest
  ggml types (`ffn_gate_exps` g12/g13, `ffn_down_exps` g7/g8). Top-10 is exact:
  `fg_vk_router_top10(...FG_TOP_K)` then `fg_vk_moe_decode_gate_up` and
  `fg_vk_moe_decode_down_reduce` read exactly the selected experts
  (`src/expert.c:403-412`).
* `dense weights`: every common tensor of the rank's layers is consumed once per
  token by the layer chain - GDN `attn_qkv`/`attn_gate`/`ssm_out`/`ssm_*`,
  QSA `attn_q/k/v/output` + bf16 `indexer.*`, shared-expert trio,
  router `ffn_gate_inp`, all 8 `hc_*` tensors, norms
  (`src/owner.c:847-895`, `src/qsa.c:1063-1096`). A GDN layer is 87.2 MB,
  a QSA layer 81.0 MB, plus the layer-1 PLE weights 36.85 MB on rank 1.
  (The old "61.2 MB/layer GDN" figure is the three q8 projections only:
  27.85 + 16.71 + 16.71 MB; the full GDN layer is 87.2 MB.)
* `output bundle` on rank 4 with the default no-split config: `output.weight`
  675,430,400 B + `output_hc_norm/down/up` 7,004,160 B + logits write 993,280
  and argmax read 993,280 B. With the qualified 4-way split (`FG_OUTPUT_SPLIT=4`)
  the same 675.4 MB is read as four 168,857,600 B slices on ranks 0/1/2/4; the
  fleet total is unchanged, only the critical rank shrinks by ~505 MB.
* `GDN+PLE state`: 36 GDN layers x (conv 163,840 + recurrent 3,145,728) x 2
  (read+write) = 238.3 MB plus PLE state 368,640 x 2 on rank 1. The recurrent
  algebraic kernel reads and rewrites the full `48*128*128` f32 state per token
  (`shaders/fg_gdn_recurrent_algebraic.comp:10,48,73`).
* `QSA state at 4K` per layer = index scan 148 B/committed-token (index keys
  136 + positions 12, `shaders/fg_qsa_index_score.comp` bindings 1/3) plus
  top-512 selection, gather of 512 pages x 4944 B read+write
  (`shaders/fg_qsa_record_gather.comp`), attention read of the selected records
  (1088 B/token, key+value only, `shaders/fg_qsa_decode_attention_split.comp`),
  and the per-token commit (4944 + 136 B). At T=4096 the 512-block selection is
  2048 of 4096 tokens: 7.92 MB per QSA layer, 95.1 MB fleet.
* `other` rank 0: embedding row gather 2,720 B + 40,960 B dequantized write,
  16 x 4 KiB n-gram O_DIRECT reads (65,536 B) + 3 x 10,240 B VRAM moves.
  n-gram disk I/O is measured in traces as exactly 16 reads / 65,536 B per
  decode token (`NGRAM_TRACE first=28 tokens=1 rows=16 reads=16 bytes=65536`).
* Transport per token, from `src/runtime.c:4149-4154`, `:862-886`,
  `:3835-3854` and `src/protocol.c:381-416`: one 51,224 B layer-work frame
  (layer 0 carries the n-gram embedding), seven 40,984 B work frames, one
  40,968 B output-hidden frame to rank 4, a 40 B output config (rank 0) and a
  16 B result. With 36 B headers: **379,532 B total**; the commonly quoted
  **379,368 B** equals the eight work frames with headers plus the output-hidden
  payload (it omits the output-hidden header and the two small control frames).
  The 4-way split adds four 10,248 B hidden-slice frames plus three partials:
  the complete code-derived 4-way frame set is **420,812 B**, and the quoted
  **420,488 B** is the same set under the payload-plus-work-headers accounting.

## 3. Cross-checks against measured kernel times

| category | bytes (fleet) | measured anchor | implied rate | verdict |
|---|---:|---|---:|---|
| expert pair | 184.3 MB/rank-block | 1.2-1.4 ms/block (mission), 0.851-1.385 ms in round-12 A/B | 133-217 GB/s | plausible (<= clpeak) |
| dense rank-0 block | 511 MB | 2.947 ms sum of profile scopes (round-12 sec 7.1, experts excluded) | 173 GB/s | plausible; matches the 92-298 GB/s shape table weighted by bytes |
| output head (rank 4) | 675.4 MB | 2.620 ms full GEMM (4-way doc) | 258 GB/s | plausible |
| GDN recurrent state | 6.62 MB/layer R+W | gdn_recurrent 0.148 ms / 4 layers | ~179 GB/s | plausible |
| QSA records+scan | 7.92 MB/layer (T=4K) | QSA GPU ~1.4 ms/token over 12 layers | < 100 GB/s | **latency-bound**, not byte-bound (small grids); bytes are not the issue at 4K |
| n-gram lookup | 65.5 KB/token | `io_ms` 0.8 ms warm to 25 ms cold in traces | n/a | storage latency, not DRAM bandwidth; a real per-token serial cost on rank 0 |

No category implies a bandwidth above 258 GB/s, so no byte count in the ledger
is inflated. The measured 39-43 ms/token is also what the ledger predicts:
510-552 MB dense at ~173 GB/s + 184-199 MB experts at ~150-190 GB/s + state
+ hops gives ~4.3-4.6 ms per rank block, x8 ranks = 34-37 ms, plus the 5-7 ms
of host/fence/first-hop overhead documented in the round-13/14 docs.

## 4. Floors

| model | ms/token | TPS |
|---|---:|---:|
| All 6.671 GB at clpeak 353 GB/s (absolute physics floor) | **18.90** | **52.9** |
| Measured per-shape rates, best observed (dense 20.92 + experts 6.84 + head 2.65 + state 1.33) | 31.7 | 31.5 |
| Same with experts at the slow end (220 -> 140 GB/s) and head 170: dense 20.92 + 10.74 + 4.03 + 1.33 | 37.0 | 27.0 |
| Measured rates de-rated 0.8x (thermal band) + slow expert/head | 42.3 | 23.7 |
| Ring wire alone, serialized at 1 GbE for 379,532 B | 3.04 | n/a |
| **Current measured decode (round 12)** | **39-43** | **23.4-24.6** |

Efficiency: 18.90 / 40.8 = **46% of the clpeak-equivalent floor**;
35 / 40.8 = **~86% of the central practical floor** (32-42 ms table band, plus
0-3 ms of hop wire). The remaining ~5-7 ms is host/fence/hop overhead, not
bytes.

Context scaling (same model, QSA term): T=32K adds ~1.3 MB/layer/token of
state-file page fetches once the 4096-page worker cache (16,384 tokens) is
exceeded, i.e. ~15 MB/token of NVMe+DRAM traffic fleet-wide, which matches the
measured 32K drop (19.36 vs 24.2 TPS). At T=4K there are no page misses.

## 5. Answer: is 50-60 TPS pre-MTP above the roofline?

**Yes.**

* 60 TPS is impossible at any efficiency: it needs 400 GB/s aggregate, above
  the 353 GB/s best measured memory bandwidth of the platform.
* 50 TPS (20 ms) needs 334 GB/s sustained across weights, experts, state, and
  hops - 94.5% of clpeak. The measured kernels run at 92-298 GB/s with the
  weighted dense average ~170-210 and experts 133-217. At those rates the byte
  budget for 50 TPS is ~4.0 GB/token, i.e. the ledger must shrink ~40%.
* The absolute pre-MTP ceiling is **52.9 TPS** at infinite kernel efficiency
  and zero overhead; the practical ceiling with the measured rates is
  **~28-31 TPS**, and the current build is already at 23.4-24.6 TPS.

Uncertainty: the byte ledger is good to a few percent (manifest-exact weights
and experts; QSA model assumes 512 selected blocks at T>=2048 and 1088 B/token
attention reads; small arenas may be L2-resident, which makes the practical
floor *lower* but never higher). Kernel rates carry the fleet's documented
+/-10-15% thermal spread and the 353 GB/s figure is a clpeak best case. The one
measurement that could move the answer is the n-gram O_DIRECT read latency
(0.8-25 ms measured cold/warm), which is a serial rank-0 cost not in the DRAM
ledger; it makes the practical ceiling lower, not higher.

## 6. Ranked byte-reduction candidates

| # | cut | bytes saved/token | est. ms at measured rates | precision/quality cost |
|---|---|---|---:|---|
| 1 | Dense linear weights q8_0 -> Q5_K (~5.5 bpw) | ~1.45 GB | ~7.6 | attention/GDN projections are precision-sensitive; pack rebuild + new kernels + full gates |
| 1b | same -> Q4_K (4.5 bpw) | ~1.95 GB | ~10.3 | higher risk on attention outputs; gate battery required |
| 2 | Output head q8_0 -> Q5_K | 238 MB | ~1.1 (rank-4 critical path) | LM head only; 4-way split already qualified but does not reduce bytes |
| 3 | Expert down projection Q5_1/Q5_K/Q8_0 -> Q4_K | ~150 MB | ~1.0 | down projection is the most sensitive expert matrix; repack of layers 2,4,30,46,47 already differ |
| 4 | GDN recurrent state f32 -> bf16/f16 | 113 MB | ~0.6 | recurrent state drift over long contexts; needs long-context gates |
| 5 | QSA index-key scan at long context (4-bit keys or shorter index) | ~27 MB at 32K; ~215 MB at 262K | 0.1-1.1 | indexer recall quality; the short/4K battery sees only ~7 MB |
| 6 | QSA record pages: fetch/pack K+V only in decode | ~6.7 MB at any T (index key is 11% of pages) | < 0.2 | protocol/pack layout change; low risk |
| 7 | bf16 hyper hops (8 work + output frames) | ~190 KB (379,532 -> ~190,000; wire 3.04 -> ~1.5 ms) | ~1.4 | not bit-identical; touches the chain codec on all ranks |
| 8 | Fold rank 0's own block into a neighbor | one 40,984 B frame + turnaround | 0.3-0.6 | structural; changes the ring schedule |

Notes: the 4-way output split changes no total bytes - it moves 505 MB off
rank 4's critical path onto ranks 0/1/2 (already qualified). Requantizing the
*experts* alone caps at ~150 MB; the dense weights are 62% of the budget and
are the only category with GB-scale headroom. No combination short of a
different quantization of dense weights gets the ledger under ~4.0 GB.

## 7. Verified / assumed / falsified

Verified from the deployed manifest or code:

* All per-rank dense and expert byte counts; expert quant types per layer;
  layer ownership and QSA schedule; output bundle and PLE sizes; top-10 routing;
  state sizes and read/write structure; hop frame sizes and counts; embedding
  and n-gram per-token byte patterns; the 4-way split row spans and foreign
  slice sizes.

Assumed (stated uncertainty):

* QSA decode selection is 512 blocks at T>=2048 and the attention reads
  1088 B/selected token; record arenas may be L2-resident, which can only lower
  the floor. L2 residency of small activations is likewise not modeled.
* Per-shape dense rates are from the round-12 fleet sweep (92-298 GB/s) with
  bytes-weighted mixing; ~61% of the dense bytes use a directly measured shape,
  the remaining ~39% (`attn_gate`, `hc_*`, the f32 router, bf16 `indexer.*`,
  small SSM/norm params) are priced at the nearest measured shape or a
  conservative 150-250 GB/s and are marked as estimates here rather than
  measured. No new profiling run was taken (read-only mandate).
* 353 GB/s is the recorded clpeak, not a sustained model-mix number.

Falsified or corrected:

* The "3.3 GB active weights/token" figure is not the deployed reality; the
  actual per-token weight read is **6.34 GB** (dense 4.83 including the head,
  experts 1.50).
* "61.2 MB/layer" for GDN is only the three q8 projections; the full GDN layer
  is **87.2 MB** dense, and the QSA layer is 81.0 MB.
* "379,368 B/token" holds only under the narrow frame-accounting definition
  above; the complete sent bytes are 379,532 B (and 420,812 B for the full
  4-way frame set).
* The "195 KB bf16 variant" is not in the tree: `fg_layer_work_encode` /
  `fg_layer_result_encode` write big-endian f32 (`src/protocol.c:381-393`), and
  no bf16 hop flag exists. A faithful bf16 version of the 4-way hop payload
  recomputes to ~190-195 KB and the default ring to ~175 KB, so the figure is
  plausible as a proposal only.

## 8. Local evidence (sanitized commands)

Offline, on the local copy of the deployed manifest:

```
git worktree add <worktree> -b perf/byte-budget main
node fgm_parse.js <manifest>            # header, owners, per-rank records
node fgm_parse.js <manifest> dump       # 1,225 tensor records
node fgm_ledger.js <manifest>           # per-layer dense/expert sizes
node fgm_ledger_bytes.js <manifest> 4096# per-rank ledger at T=4096
node fgm_floor.js <manifest> 4096       # measured-rate floor model
```

Fleet (read-only, small): `ls -l` and `sha256sum` of the pack manifest and
shard listing; `grep` of `NGRAM_TRACE`, `QSA_ATTEND_TRACE`, `RING_DECODE` and
rate lines in the existing logs; `pgrep` of the running ranks. No quiesce, no
writes to the blades, no `/tmp` usage.

Manifest identity: local copy sha256 `1368cd6d...` equals the deployed
`manifest.fgm`; every `ranks[r].persistent_bytes` equals the corresponding
pack shard size.
