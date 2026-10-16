# Real-pack MTP / small-N format audit

This is the **entire sealed mix-04 + stack-1 serving inventory**, not a
synthetic Q4/Q5-only checklist. The manifest SHA256 is
`0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
It contains **1,225 records and 51 distinct (family, quantization, layout)
groups**, including the n-gram and tokenizer artifacts. The native Q8 overlay
SHA256 is
`536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b`.

`tools/mtp_format_audit.py` reads every record, rejects unknown families or
format/layout pairs, requires full owned expert slabs, checks the exact
manifest identity, and accounts for all native aliases and unused tensors.
The current compiled manifest ABI is explicitly checked; this is not a
promise to accept a different model or deployment without a new audit.
The serving controller requires a hash-sealed audit tied to the candidate
release and successful real-Q8 gate **before any production access**.

## Dispatch and validation keys

* **D**: `fg_vk_dense_quant`, `fg_vk_dense_q8_0_cooked`, `small_plain_dispatch`.
  GPU N2/3/4 against native N1 at every actual projection shape; both SOA and
  compact storage for K/IQ4 types. BF16 general and BF16-router are distinct
  test cases. The four-way head is tested at its actual 62,080-row shard shape.
* **E**: `fg_expert_decode_small_n` → `fg_vk_small_n_experts`. Full owned
  512-expert K gate/up plus Q5_1 or Q8 down. Mixed Q5/Q5/Q8 and Q4/Q4/Q8
  six-layer tests include causal-prefix, replay, sessions and rollback checks.
* **R**: exact real-pack Q8 down test, described below.
* **S**: F32 group norms, HC injection/finalization, GDN controls/recurrence,
  PLE convolution. Dedicated existing scalar/GPU oracles plus owner-block
  session/causal/rollback tests cover these operators.
* **Q**: QSA BF16 projections, indexer, quantization/record storage and causal
  resident attention oracles. Verification retains per-token causal selection;
  a shared prefix is only an optimization, never permission to see future rows.
* **N**: n-gram direct/resident/prefill IQ4_NL lookup; PLE consumes the resulting
  F32 rows. MTP verification reuses this ordinary lookup, not the native head.
* **H**: unchanged N1 target final-HC contraction and four-way shared vocabulary
  head, reused for each verifier row and native proposal. The vocabulary is not
  an independent lower-quality draft head.
* **B**: exact raw/cooked Q8 target embedding and exported sidecar GPU oracle.
  Native drafting reads the target's exact embedding sidecar.
* **T**: tokenizer artifact; not a floating-point weight or a GPU dispatch.

All GPU gates use .51's unchanged stock governor and the complete current
recipe: roofline2047, dense19, QSA3, expert33024, HC96, recurrent3, GDN1,
local n-gram cache1 and submit-group2. Both small-kernel masks0/330 are
covered; R additionally covers expert-wide0/1. Production uses wide1.
No new quantization or precision reduction is involved.

## Every target family/format

The `blk.<layer>.` and `.rank<owner>` decorations are removed only for grouping.
Counts still include every individual record.

| Family | Count | Stored quantization / layout | Supported dispatch |
|---|---:|---|---|
| attn_gate.weight | 36 | Q4_K / compact | D; GDN gate |
| attn_k.weight | 12 | Q8_0 / GGML | lossless runtime Q8 cooking → D,Q |
| attn_k_norm.weight | 12 | F32 / GGML | S,Q |
| attn_output.weight | 12 | Q8_0 / cooked | D,Q |
| attn_q.weight | 12 | Q8_0 / cooked | D,Q |
| attn_q_norm.weight | 12 | F32 / GGML | S,Q |
| attn_qkv.weight | 36 | Q4_K / compact | D; GDN QKV |
| attn_v.weight | 12 | Q8_0 / GGML | lossless runtime Q8 cooking → D,Q |
| ffn_down_exps.weight | 43 | Q5_1 / cooked expert | E; existing Q5 down unchanged |
| ffn_down_exps.weight | 5 | Q8_0 / cooked expert | E,R; exact native-lane Q8 down |
| ffn_down_shexp.weight | 48 | IQ4_NL / compact | D; shared expert down |
| ffn_gate_exps.weight | 47 | Q4_K / cooked expert | E |
| ffn_gate_exps.weight | 1 | Q5_K / cooked expert | E |
| ffn_gate_inp.weight | 48 | BF16 / GGML | D; `BF16_ROUTER`, not an F32 reinterpretation |
| ffn_gate_inp_shexp.weight | 48 | F32 / GGML | D,S; shared scalar gate |
| ffn_gate_shexp.weight | 48 | Q4_K / SOA | D; shared expert gate |
| ffn_up_exps.weight | 47 | Q4_K / cooked expert | E |
| ffn_up_exps.weight | 1 | Q5_K / cooked expert | E |
| ffn_up_shexp.weight | 48 | Q4_K / SOA | D; shared expert up |
| hc_attn_down.weight | 48 | Q5_K / compact | D; HC reduction |
| hc_attn_inject.weight | 48 | F32 / GGML | D,S; HC injection |
| hc_attn_norm.weight | 48 | F32 / GGML | S; group RMS |
| hc_attn_up.weight | 48 | Q8_0 / cooked | D,S; HC expansion |
| hc_ffn_down.weight | 48 | Q5_K / compact | D |
| hc_ffn_inject.weight | 48 | F32 / GGML | D,S |
| hc_ffn_norm.weight | 48 | F32 / GGML | S |
| hc_ffn_up.weight | 48 | Q8_0 / cooked | D,S |
| indexer.k_norm.weight | 12 | F32 / GGML | S,Q |
| indexer.k_proj.weight | 12 | BF16 / GGML | D,Q; explicit BF16 dense |
| indexer.q_norm.weight | 12 | F32 / GGML | S,Q |
| indexer.q_proj.weight | 12 | BF16 / GGML | D,Q; explicit BF16 dense |
| output.weight | 1 | Q8_0 / cooked | D,H; four existing head shards |
| output_hc_down.weight | 1 | Q5_K / compact | D,H |
| output_hc_norm.weight | 1 | F32 / GGML | S,H |
| output_hc_up.weight | 1 | Q8_0 / cooked | D,H |
| per_layer_token_embd.weight | 1 | IQ4_NL / GGML | N; pageable rows, not dense GEMV |
| ple_conv1d.weight | 1 | F32 / GGML | S; causal per-token convolution |
| ple_key.weight | 1 | Q4_K / compact | D,S |
| ple_norm_conv.weight | 1 | F32 / GGML | S |
| ple_norm_key.weight | 1 | F32 / GGML | S |
| ple_norm_query.weight | 1 | F32 / GGML | S |
| ple_value.weight | 1 | Q4_K / compact | D,S |
| ssm_a | 36 | F32 / GGML | S; GDN recurrence |
| ssm_alpha.weight | 36 | F32 / GGML | D,S |
| ssm_beta.weight | 36 | F32 / GGML | D,S |
| ssm_conv1d.weight | 36 | F32 / GGML | S; causal convolution |
| ssm_dt.bias | 36 | F32 / GGML | S; recurrence control |
| ssm_norm.weight | 36 | F32 / GGML | S |
| ssm_out.weight | 36 | Q5_K / compact | D,S |
| token_embd.weight | 1 | Q8_0 / GGML | B; lookup, not dense GEMV |
| tokenizer/tokenizer.fgt | 1 | artifact / layout0, type field0 | T; not F32 weight data |

Storage cooking in `src/model.c` is an exact byte reordering, not requantization.
K/V narrow Q8 tensors acquire their runtime cooked format. BF16 router metadata
becomes `FG_VK_TENSOR_FORMAT_BF16_ROUTER`; QSA index projections use the
separate BF16 API. No other format is inferred from byte length.

## Every native overlay family/format

All names below have the `mtp.` prefix in the GGUF. The importer in
`src/mtp_head.c` maps **28 used source tensors to29 runtime aliases**: the
fused gate/up tensor is split into two aliases. Native draft cells are N1;
they do not route Q8 gate/up through the target's K-only small-N GU kernel.

| Source family | Source format | Runtime support |
|---|---|---|
| fc_embedding.weight | Q8_0 | cooked Q8 dense |
| fc_hidden.weight | Q8_0 | cooked Q8 dense |
| hyper_connection_mixer.hc_norm.weight | F32 | group RMS, gamma +1 once |
| hyper_connection_mixer.input_mix_weight_down.weight | Q8_0 | cooked Q8 dense |
| hyper_connection_mixer.input_mix_weight_up.weight | Q8_0 | cooked Q8 dense |
| layers.0.attn_hyper_connection.block_inject_weight.weight | Q8_0 | importer decodes to F32; HC injection |
| layers.0.attn_hyper_connection.hc_norm.weight | F32 | group RMS, gamma +1 once |
| layers.0.attn_hyper_connection.input_mix_weight_down.weight | Q8_0 | cooked Q8 dense |
| layers.0.attn_hyper_connection.input_mix_weight_up.weight | Q8_0 | cooked Q8 dense |
| layers.0.mlp.experts.down_proj | Q8_0 | cooked expert Q8, native N1 inline |
| layers.0.mlp.experts.gate_up_proj | Q8_0 | split cooked expert Q8 gate/up, native N1 inline |
| layers.0.mlp.gate.weight | Q8_0 | importer decodes to F32 router |
| layers.0.mlp.shared_expert.down_proj.weight | Q8_0 | cooked Q8 dense |
| layers.0.mlp.shared_expert.gate_proj.weight | Q8_0 | cooked Q8 dense |
| layers.0.mlp.shared_expert.up_proj.weight | Q8_0 | cooked Q8 dense |
| layers.0.mlp.shared_expert_gate.weight | Q8_0 | importer decodes to F32 scalar gate |
| layers.0.mlp_hyper_connection.block_inject_weight.weight | Q8_0 | importer decodes to F32; HC injection |
| layers.0.mlp_hyper_connection.hc_norm.weight | F32 | group RMS, gamma +1 once |
| layers.0.mlp_hyper_connection.input_mix_weight_down.weight | Q8_0 | cooked Q8 dense |
| layers.0.mlp_hyper_connection.input_mix_weight_up.weight | Q8_0 | cooked Q8 dense |
| layers.0.self_attn.indexer.index_qk_proj.weight | Q8_0 | intentionally not dispatched; see below |
| layers.0.self_attn.indexer.k_layernorm.weight | F32 | intentionally not dispatched; see below |
| layers.0.self_attn.indexer.q_layernorm.weight | F32 | intentionally not dispatched; see below |
| layers.0.self_attn.k_norm.weight | F32 | Q/K normalization, gamma +1 once |
| layers.0.self_attn.k_proj.weight | Q8_0 | cooked Q8 dense |
| layers.0.self_attn.o_proj.weight | Q8_0 | cooked Q8 dense |
| layers.0.self_attn.q_norm.weight | F32 | Q/K normalization, gamma +1 once |
| layers.0.self_attn.q_proj.weight | Q8_0 | cooked Q8 dense |
| layers.0.self_attn.v_proj.weight | Q8_0 | cooked Q8 dense |
| pre_fc_norm_embedding.weight | F32 | group RMS, gamma +1 once |
| pre_fc_norm_hidden.weight | F32 | HC normalization, gamma +1 once |

The three native learned-indexer tensors are **explicitly unused**, not
silently treated as supported compute. The experimental native head attends
over its full/bounded KV window rather than running learned sparse selection.
That draft approximation can change acceptance, never bypass target causal
verification. The target's learned QSA indexer **is used and covered** above.
Long-context native acceptance remains unqualified.

## Real Q8 identity, numerical contract and reproduction

The layer2 down tensor was range-fetched on .51 directly from the original
abliterated GGUF, revision `7e3bfc316b880fefeb049596f11c49d6a18e05fb`.
No production owner was contacted. `tools/mtp-real-cook` performs the ordinary
lossless cooking and requires the **entire 891,289,600-byte slab** to match
the retained real-pack tensor SHA256:
`065b86394f68fad0a08f4156ce64fa00d8265572bbc0aaa5b8b4908c0713b3b7`.
This is identical pack data, not newly quantized BF16 or a synthetic substitute.

`small_q8_down_native` tests N2/3/4 with dense, ±4096 cancellation and sparse
basis activations, varying gates and expert IDs spanning the full slab.
Both native lane geometries (wide0/wide1) and masks0/330 produce **36 byte-exact
N1 comparisons**; independent CPU basis maximum relative error is below
`3.2e-10`. Invalid stride, undersized ownership spans and incorrect tensor
formats are rejected. Cooker corruption and overwrite-protection tests are
also part of the qualification.

The Q8 fix preserves each native kernel's lane/parity walk, slot order and
reduction order. Existing Q5 down is unchanged. No research variant is promoted.
Do not generalize this exact down result to every floating-point operation:
the production dense19 N1 recipe can differ from small-N compact Q4 by
`9.54e-7` even with small-kernel mask0. Dense/full-block components therefore
use the explicit `1e-4 * max(1,abs(reference))` gate; causal-prefix, repeat and
rollback checks remain exact. **Same-build full-model greedy identity is
still a mandatory ring gate for both masks**, not inferred from tolerance.

Reproduction entry points:

* `tools/mtp_real_tensor.py`: bounded HTTP ranges, pinned public source and
  manifest geometry; records provenance without touching the ring.
* `tools/mtp-real-cook MANIFEST TENSOR RAW COOKED`: lossless cooking and exact
  full tensor hash check; refuses to overwrite an existing qualified slab.
* `tools/mtp-format-qualify.sh`: unchanged stock .51, production flags, real
  tensor and native overlay, all actual matrix shapes and operator/session
  oracles. All work stays under `/home/xander/hc-mtp` and is removed afterward.
* `results/hillclimb/mtp-format-audit.json`: complete machine-readable groups,
  source hashes, GPU receipts, numerical limits and sealed release identities.

This audit qualifies **format coverage and components**. It is not a claim
of trained-model greedy equivalence, end-to-end MTP speedup, long-context
acceptance or production promotion.

The subsequent exclusive ring lease reached a real55-token K1 completion
with the exact target token hash, then failed on lazy native-session residency
when the next request needed another session. The exhaustive format coverage
remains valid; it does **not** prove owner+native+all-session memory fits.
The13.5GiB policy cap must not be bypassed. See
[the separate serving/memory gate](../results/hillclimb/MTP_SERVING.md).
That lease restored/verified all eight owners, cleaned and released at14:12:25Z.
