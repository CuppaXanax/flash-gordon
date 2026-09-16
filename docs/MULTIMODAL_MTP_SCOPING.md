# Multimodal (vision/video) and MTP scoping — decision review

Date: 2026-09-16. Scope: what it takes to add (a) image/video input and
(b) trained-MTP speculative decoding to the Flash Gordon ring running
Qwen3.8-Flash-Next, ranked by effort and value. All claims below are from
primary sources: the llama.cpp tree at master (2026-09-16), the Qwen
checkpoint/processor configs, published GGUF artifacts, the open upstream
pull requests for MTP, and a read-only tensor audit of the deployed source
set. Details in §5.

Verdict in one paragraph: we do **not** have the vision or MTP tensors today
(the deployed source is text-only by design), but both are cheaply obtainable
as ready-made GGUF files (vision tower 0.6-0.9 GB, MTP head 1.9-2.8 GB).
Vision is fully supported by mainline llama.cpp and has no dependency on the
ring scheduler work; it is the higher-confidence deliverable (5-7 rounds).
MTP for this architecture is **not** in mainline yet (two open draft PRs; the
published heads require a fork), and its payoff on our serial ring depends
entirely on a batched K-token verify whose cost is owned by the
concurrent-sessions workstream (7-11 rounds, 1.3-1.7x *if* verify is cheap).
Recommendation: **vision first**; stage the MTP head now; execute MTP after
batched decode/verify lands.

---

## 1. Inventory: what we have, what exists

### 1.1 Deployed source set (audited, read-only)

`Qwen3.8-Flash-Next-UD-Q4_K_XL`, 4 shards, ~111.3 GB (~103.7 GiB), an
Unsloth Dynamic text quant. Full tensor schema dump: **1224 tensors**.

| family | count | notes |
|---|---|---|
| text decoder (common) | 1073 | GDN/QSA/HC/PLE-layer/indexer/output |
| routed experts | 144 | 48 layers x 3 (`gate/up/down_exps`) |
| PLE / N-gram | 7 layer tensors + 1 table | `blk.1.ple_*`, `per_layer_token_embd.weight` (160 x 320,001,536 = 51.2B params, IQ4_NL) |
| **vision** | **0** | none |
| **MTP** | **0** | none |

The earlier audit conclusion is confirmed: this quant is a text-only export
(no mmproj companion file was present in the audit). The schema dump hash is
recorded in the work log.

### 1.2 Official checkpoint (`Qwen/Qwen3.8-Flash-Next`)

180B params BF16, 131 safetensors shards, ~360 GB. It **does** carry:

- vision: `model.visual.*` — 333 tensors (patch_embed, pos_embed, 27 blocks,
  merger; see §2.2);
- MTP: `mtp.*` — 31 tensors (see §3.1).

No separate mmproj or MTP file in the HF repo; both live inside the main
shards. The `config.json` declares `image_token_id: 248056`,
`video_token_id: 248057`, `vision_start/end: 248053/248054`, and a built-in
MTP module (`mtp_num_hidden_layers: 1`, `mtp.hybrid`).

### 1.3 Community artifacts (ready to use, no conversion)

| artifact | file | size |
|---|---|---|
| Tower F16 | `unsloth/Qwen3.8-Flash-Next-GGUF` `mmproj-F16.gguf` | 904,004,000 B (~862 MiB) |
| Tower BF16 | same repo `mmproj-BF16.gguf` | 907,542,944 B |
| Tower Q8_0 (official) | `ggml-org/Qwen3.8-Flash-Next-GGUF` `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` | 616,703,104 B (~588 MiB) |
| MTP head (recommended) | `unsloth` `MTP/mtp-...-shared-Q8_0.gguf` | 2,786,568,256 B (~2.6 GiB) |
| MTP head (self-contained) | `MTP/mtp-...-Q8_0.gguf` | 4,137,429,120 B |
| MTP head (small) | `MTP/mtp-...-shared-Q4_K_M.gguf` | 1,907,151,936 B |

The mmproj and MTP files are quant-independent companions to the UD-Q4_K_XL
text weights; both are already in GGUF form, so "acquisition cost" is a
download plus a pack/format decision — not a conversion project.

### 1.4 Runtime surface today

| area | exists | missing |
|---|---|---|
| Vision tensor classification | `src/gguf.c:86` maps `vision`/`visual` names to `FG_TENSOR_VISION`; manifest component flags exist | no loader, no graph, no preprocess, no API parts; `--experimental-vision` still rejected (`src/runtime_options.c:289`) |
| MTP pack schema | `FG_TENSOR_MTP`, `FG_MANIFEST_HAS_MTP`, rank-0 routing, byte ledger (`MTP_FEASIBILITY_2026-09-13.md` §2) | converted heads use `blk.48.*`/`nextn` names not classified today; see §3.2 |
| MTP gating/capability | seal-aware option gate (`src/runtime_options.c:292`), `fg_runtime_mtp_capability` (`src/runtime.c:5008`), MTP tensor load on the sealed rank (`src/model.c:116`) | no MTP layer execution, no draft/verify loop (scaffold dropped in `39e116e`) |

---

## 2. Vision: verified architecture, format, and preprocessing

### 2.1 Tower architecture (config + llama.cpp graph)

`config.json` vision_config: `depth 27`, `hidden_size 1152`, `num_heads 16`
(d_head 72), `intermediate_size 4304`, `hidden_act gelu_pytorch_tanh`,
`patch_size 16`, `temporal_patch_size 2`, `spatial_merge_size 2`,
`out_hidden_size 2560`, `num_position_embeddings 2304` (= 48x48),
`deepstack_visual_indexes: []`.

llama.cpp implements it as an **unmodified Qwen3-VL ViT**
(`conversion/qwen4exp.py`: `class Qwen4ExpVisionModel(Qwen3VLVisionModel)`),
with the graph in `tools/mtmd/models/qwen3vl.cpp`:

- Conv3D patch embed (temporal 2) -> `v.patch_embd.weight` +
  `v.patch_embd.weight.1` + bias; patch bias added.
- Learned pos embedding (48x48) bilinearly interpolated with align-corners to
  the target grid, then spatially merged (2x2) in the same layout as the
  patches.
- 27 blocks: LayerNorm1 -> fused QKV (with bias) -> **M-RoPE** on Q/K
  (`GGML_ROPE_TYPE_VISION`, rope theta 10000, `n_dims = d_head/2`, 4 sections
  of `d_head/4`) -> full attention (no window pattern, no qk-norm) ->
  residual -> LayerNorm2 -> GELU MLP (up then down; no gate tensor) ->
  residual. No deepstack tensors for this model.
- Post-LN (`v.post_ln`), then merger: reshape 4x1152=4608 per merged token ->
  `mm.0` (linear_fc1) -> GELU -> `mm.2` (linear_fc2) -> 2560 (decoder width).

M-RoPE with vision positions is the subtle part: each merged patch group gets
`(t,h,w)` position triples and the rope type is visual (see
`tools/mtmd/mtmd.cpp`, `MTMD_POS_TYPE_MROPE`).

### 2.2 mmproj GGUF format and tensor list

Arch `clip`, projector `qwen3vl`, `vision.use_gelu=true`,
`vision.spatial_merge_size=2`, attention LN eps 1e-6, image mean/std 0.5.
Converted tensor names (334 tensors, ~449M params, ~0.9 GB F16 / ~0.6 GB
Q8_0):

```
v.patch_embd.weight          v.patch_embd.weight.1     v.patch_embd.bias
v.position_embd.weight       v.post_ln.weight          v.post_ln.bias
for i in 0..26:
  v.blk.i.ln1.{weight,bias}        v.blk.i.ln2.{weight,bias}
  v.blk.i.attn_qkv.{weight,bias}   v.blk.i.attn_out.{weight,bias}
  v.blk.i.ffn_up.{weight,bias}     v.blk.i.ffn_down.{weight,bias}
mm.0.{weight,bias}           mm.2.{weight,bias}
```

(Converter mapping: `visual.patch_embed.proj` -> `v.patch_embd` (Conv3D split
on the temporal axis), `visual.pos_embed` -> `v.position_embd`,
`visual.merger.norm` -> `v.post_ln`, `merger.linear_fc1/fc2` -> `mm.0/mm.2`,
blocks -> `v.blk.*`. This matches the official `convert.log` from the
ggml-org conversion exactly.)

### 2.3 Preprocessing and prompt/token semantics

From `preprocessor_config.json` / `video_preprocessor_config.json`
(processor class `Qwen3VLProcessor`, image processor
`Qwen2VLImageProcessorFast`) and llama.cpp `tools/mtmd/mtmd-image.cpp`,
`mtmd.cpp`:

- **smart_resize**: resize preserving aspect ratio, both sides rounded to a
  multiple of `patch_size * merge_size = 32`, subject to pixel limits:
  images `shortest_edge 65536 .. longest_edge 16,777,216` px; video
  `4096 .. 25,165,824` px. Resampler is Pillow-compatible bicubic.
  (llama.cpp's *default* for qwen3vl is 8..4096 image tokens =
  8,192..4,194,304 px unless overridden — the official runtime should follow
  the HF bounds, llama.cpp's defaults are narrower.)
- **Normalize**: RGB in [0,1], `(x - 0.5) / 0.5`.
- **Patchify**: 16x16 patches, groups of 2 frames merged on the temporal axis
  (`clip_model_n_temporal_merge() == 2`); a video with an odd frame count
  repeats the last frame.
- **Token count**: image `nx * ny = (W/32) * (H/32)`; video
  `ceil(F/2) * nx * ny`. `mtmd_image_tokens::n_tokens()` implements exactly
  this.
- **Prompt**: chat template emits
  `<|vision_start|><|image_pad|><|vision_end|>` for an image and
  `<|vision_start|><|video_pad|><|vision_end|>` for video. The single pad
  token (id 248056/248057) is expanded by the runtime to the computed token
  count; the tower embeddings replace the expansion. llama.cpp binds
  `img_beg/img_end` to `<|vision_start|>/<|vision_end|>` and preprocesses via
  `mtmd_image_preprocessor_dyn_size`.
- **PLE interaction**: the N-gram/PLE module hashes tokens host-side; image
  positions arrive as an embeddings-only batch with no token id, so the
  converter stores `ple_image_token_id` (248056) and the decoder hashes that
  id for embedding-batch positions (`src/models/qwen4exp.cpp`,
  `llm_graph_input_ple::set_input`). Our runtime must reproduce this when
  vision lands.

### 2.4 Integration plan (Flash Gordon)

Per PRD `Multimodal and MTP`: preprocess + tower + projection run outside the
ring text layers, then embeddings feed **rank 0's ordinary prefill** with
correct multimodal positions and token accounting. Weight placement is a pack
decision taken from actual inventory and free memory; supporting workers are
not an implicit dependency for vision.

- **V1 — Preprocess + parity harness** (1-2 rounds). Image decode
  (PNG/JPEG), smart_resize, normalize, patch/merge ordering; standalone CPU
  output compared against llama.cpp `mtmd` embeddings for 2-3 fixture images.
  Risk: resampler exactness (scores are sensitive at the last 1e-2; compare
  embeddings cosine, not pixels).
- **V2 — Tower kernels + numeric parity** (2-3 rounds). Conv patch embed
  (can be a matmul), LayerNorm, GELU, fused QKV MHA with bias, visual M-RoPE,
  bilinear pos-embed interpolation, spatial merge, merger MLP. Gate:
  embedding block cosine >= 0.999 vs llama.cpp on the fixtures.
  Risk: M-RoPE vision variant is new ground for our kernels; pos-embed
  interpolation flags (align-corners) must match.
- **V3 — Decoder integration** (1-2 rounds). `<|image_pad|>`/`<|video_pad|>`
  expansion, embeddings-only batch into the prefill path (today it takes
  token ids), M-RoPE positions for image tokens, PLE image-token-id hashing.
  Gate: single-image chat output matches llama.cpp (same mmproj) on 2-3
  prompts. Risk: the prefill path's position/history plumbing assumes text
  tokens; this is the main decoder-side change.
- **V4 — API + video + limits** (1 round). OpenAI content parts
  (`image_url`, `video_url`), request limits and token caps, error paths.
  v1 should accept base64 payloads only (URL fetching introduces SSRF/egress
  policy); videos sample frames, merge pairwise, and enforce a per-request
  token budget. Risk: video token growth (below) needs product-level caps.

Placement: run the tower on rank 0 alongside the API (one-shot per image,
~0.6-0.9 GB weights, outside the 48-layer ring). Later, it can move to a
worker for overlap; no ring protocol change is needed for v1.

### 2.5 Cost per image / video

Tower compute (order of magnitude): ~449M params; a 1024x1024 image is 4096
patches -> ~3.4 TFLOP linear + ~2.1 TFLOP attention ~= **5-6 TFLOP per
1024-token image**, i.e. roughly 0.5-1.5 s on one blade-class GPU depending
on utilization (to be measured in V2). Context/prefill cost (ring prefill
reference ~279 TPS, round-3 measurement):

| image | resized | merged tokens | prefill time @279 TPS | % of 262K ctx |
|---|---|---|---|---|
| 256x256 | min clamp | 64 | ~0.2 s | 0.02% |
| 512x512 | 512x512 | 256 | ~0.9 s | 0.10% |
| 1024x1024 | 1024x1024 | 1,024 | ~3.7 s | 0.39% |
| 2048x2048 | 2048x2048 | 4,096 | ~14.7 s | 1.56% |
| 4096x4096 | max clamp | 16,384 | ~59 s | 6.25% |

Video: `ceil(F/2) * W/32 * H/32` tokens per clip. 720p (1280x736), 2 s at
24 fps = 48 frames -> 22,080 tokens (~8.4% of context); 10 s -> 110,400
tokens (~42%). Frame sampling and token caps are mandatory product policy.

### 2.6 Vision estimate

**5-7 rounds** (one round = one measured, mergeable work cycle). Values: high
product value (a headline model capability; also image/video evaluation
content), no decode-TPS value. Key risks: M-RoPE vision positions, resampler
parity, embeddings-only prefill plumbing, video budget policy. No dependency
on the scheduler/concurrency workstream.

---

## 3. MTP: verified format and integration

### 3.1 Checkpoint tensors (31) and shape of the head

The official checkpoint carries a complete hybrid draft layer:

```
mtp.fc_embedding.weight          mtp.fc_hidden.weight
mtp.pre_fc_norm_embedding.weight mtp.pre_fc_norm_hidden.weight
mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight  {q,k}_norm.weight
mtp.layers.0.self_attn.indexer.{index_qk_proj,k_layernorm,q_layernorm}.weight
mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}   mlp.gate.weight
mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight  shared_expert_gate.weight
mtp.layers.0.{attn,mlp}_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}
```

It is a mini-MoE layer: ~2.5B params (512 experts, top-k), one full-attention
block (the QSA indexer tensors exist but the draft attends **dense**,
`compress_ratio 0`). That is why the Q8 head is ~2.6 GB.

### 3.2 Converted GGUF format (published heads, verified from the file)

The published `mtp-...-shared-Q8_0.gguf` header shows the converter output
naming: the head is a trailing block plus `nextn` tensors.

```
blk.48.{hc_attn_*,hc_ffn_*,attn_q/k/v/output,attn_q_norm,attn_k_norm,
        indexer.q/k_proj,indexer.q/k_norm,
        ffn_gate_inp,ffn_gate_exps,ffn_up_exps,ffn_down_exps,
        ffn_gate_inp_shexp,ffn_gate/up/down_shexp}
blk.48.nextn.eh_proj.weight      (concat of fc_embedding/fc_hidden)
blk.48.nextn.enorm.weight        (from pre_fc_norm_embedding)
blk.48.nextn.hnorm.weight        (from pre_fc_norm_hidden)
blk.48.nextn.hc_head_{norm,down,up}.weight  (from mtp.hyper_connection_mixer)
```

Self-contained heads additionally carry `blk.48.nextn.embed_tokens.weight`
and `blk.48.nextn.shared_head_head.weight`; the `shared-*` variants omit
them and borrow the target's token embedding/output head at load time.
GGUF gotcha for us: `fg_gguf_tensor_kind` matches only `mtp`/`draft`
substrings and `fg_gguf_tensor_layer` accepts layers `0..47`, so `blk.48.*`
and `nextn` names are **not** packable today (hazard already flagged in
`MTP_FEASIBILITY_2026-09-13.md` §2).

### 3.3 Semantics

Draft step (from the upstream patch): take the target's last hidden
hyper-connection stream `h` and the target token, normalize both
(`hnorm`/`enorm`), concatenate per stream, project with `eh_proj`, run the
single hybrid layer (HC mixer -> full attention -> HC MoE with shared
expert), collapse with `hc_head_*`, and project with the **shared output
head** -> draft logits. The wide hidden state is exported again as the next
step's `h` (feedback). The combiner **must run per hyper-connection stream**;
mean-pooling first collapses acceptance (upstream PR note). Verify is one
target forward over `[t, d1..dK]` from the committed frontier; rejected
tokens must not retain GDN/conv or QSA state.

### 3.4 Upstream status and measurements

- Mainline conversion currently **drops** the MTP head for this arch
  (`conversion/qwen4exp.py`: `supports_mtp_export = False`, `no_mtp = True`).
- **PR #27836** (`qwen4exp: add NextN/MTP draft head (--spec-type draft-mtp)`,
  open, draft) and **PR #28243** (`models: Qwen3.8-Flash-Next MTP`, open,
  draft, builds on #27836, adds `--mtp-shared-embd` borrowing). Both open and
  dirty as of 2026-09-16; stock llama.cpp cannot use the published heads.
  Unsloth also publishes prebuilt binaries from a fork (`b10715-mix-86bd2d3`
  or newer).
- Reported numbers (greedy, single stream): unsloth B200 + UD-Q4_K_XL,
  shared-Q8_0: acceptance 66.1%, **83.2 -> 138.8 tok/s (1.67x)**; PR #27836
  on M3 Max + UD-IQ4_XS: acceptance 85.7-89.2%, **+36-42%**. Both note the
  win is **not** from sequential drafting but from one batched K-token verify.
  Unsloth: at concurrency 8 the same config is a net loss (~0.81-0.87x).

### 3.5 Integration plan (Flash Gordon)

- **M1 — Pack/loader naming** (1 round). Classify `nextn` and layer-48 names
  as MTP, allow the trailing block, keep the `HAS_MTP` seal and rank-0
  placement; verify inspect/repack round-trip with the published head.
  Risk: low (schema already models MTP).
- **M2 — MTP layer execution + proposal** (2-3 rounds). HC mixer,
  `eh_proj/enorm/hnorm`, full attention (with its own context-length KV for
  the draft layer), MoE top-k with shared expert, `hc_head`, shared output
  head; propose `d1..dK` (start K=2). Reuses dense/owner primitives but is
  new graph work. Risk: numerical parity of the HC stream; draft KV state.
- **M3 — Batched K-token verify** (3-5 rounds, **shared with the
  concurrent-sessions workstream**). One target forward over `[t, d1..dK]`
  through the ring, transactional GDN/conv state and QSA pages
  (`SESSION_PREPARE/COMMIT/RESTORE` protocol exists and is unused), plus the
  direct-output fallback so rank 0 sees the final hyper while MTP is active.
  This is the critical path: without cheap batching there is no speedup.
- **M4 — Qualification and policy** (1-2 rounds). Greedy byte-identical gate,
  acceptance/telemetry, A/B (`a`, `beta_K`), stochastic residual sampling,
  and a policy to disable MTP under concurrency (per the hardware data).

### 3.6 Cost and payoff on our ring

Cycle cost = `1 + K*m + beta_K` target-forward equivalents; committed tokens
per cycle = `1 + a + ... + a^K`. With the observed `a ~ 0.66` and the
scaffold's planning values (`m = 0.15`, `beta_2 = 0.15`): `E_2 = 2.10`,
`S_2 ~ 1.45x`; K=1 cannot beat ~1.2x. The B200 result (1.67x) implies a cycle
near 1.26 forwards, i.e. `beta_2` much closer to 0 than 0.15. **The ring's
`beta_K` is the single deciding measurement**; if a K=2..3 verify costs more
than ~1.5 forwards, MTP is a regression on a latency-dominated serial ring.
Memory: +2.6 GiB on rank 0 (recommended shared-Q8_0) or +1.8 GiB (shared
Q4_K_M, ~2 points less acceptance), plus a small draft KV.

### 3.7 MTP estimate

**7-11 rounds**, of which M3 (3-5) belongs to the concurrency workstream
either way. Value: conditional 1.3-1.7x at greedy/low concurrency, likely
negative at high concurrency unless gated. Risks: upstream PRs unmerged,
shared-head borrowing, verify transaction correctness, acceptance below
vendor figures on a 4-bit main model.

---

## 4. Ranking and recommendation

| | vision | MTP |
|---|---|---|
| data acquisition | 0.6-0.9 GB download, no conversion | 1.9-2.8 GB download, no conversion |
| runtime work | ~5-7 rounds, one blade, no ring changes | ~7-11 rounds; verify path shared with concurrency |
| dependency | none beyond existing prefill | batched decode/verify + state transactions |
| value | product capability (image/video), no TPS | 1.3-1.7x conditional at low concurrency; negative at c=8 per vendor |
| upstream health | mainline-supported, official mmproj exists | two open draft PRs; published heads require a fork |
| primary risk | M-RoPE fidelity, video token budget | `beta_K` on the serial ring |

**Recommendation: pursue vision first (V1-V4).** It is self-contained
(rank 0 + prefill), has no dependency on the scheduler workstream, and is the
larger product step. In parallel, treat the MTP head as a data problem (M1
can be done immediately: one round to teach the packer `blk.48`/`nextn`
names), then schedule M2+ only after the concurrent-sessions workstream can
show a measured K-token batched verify; MTP's speedup is entirely determined
by that cost, not by the availability of the trained head. Leadership's
premise ("tensors we ignore") is half right: we don't have them, but they are
cheap to acquire — the cost center is runtime execution and batched verify,
not data.

---

## 5. Primary source references

llama.cpp (master, 2026-09-16):

- `src/models/qwen4exp.cpp` — text graph, HC, QSA, PLE; PLE image-token
  stand-in for embeddings-only batches.
- `src/models/qwen3next.cpp`, `src/models/qwen3vl.cpp`, `tools/mtmd/models/qwen3vl.cpp`
  — vision graph (patch embed, pos interp, M-RoPE, merger).
- `tools/mtmd/clip.cpp` — mmproj loader, qwen3vl hparams (merge 2, bicubic,
  image token limits), tensor naming; `tools/mtmd/mtmd.cpp` — projector
  dispatch, `<|vision_start|>/<|vision_end|>`, temporal merge, M-RoPE
  position type; `tools/mtmd/mtmd-image.cpp` — smart resize and normalization.
- `conversion/qwen4exp.py` — `Qwen4ExpTextModel` (drops MTP on master),
  `Qwen4ExpVisionModel(Qwen3VLVisionModel)`; `conversion/qwen3vl.py` —
  mmproj parameters and tensor mapping; `conversion/base.py` — mmproj GGUF
  keys; `convert_hf_to_gguf.py` — `--mmproj`, `--mtp`, `--mtp-shared-embd`.
- PR #28243 (qwen4exp MTP + shared embeddings; diff reviewed: `blk.48.nextn.*`
  tensors, `nextn_predict_layers`, draft graph, `--spec-type draft-mtp`).
- PR #27836 (qwen4exp NextN/MTP draft head; per-stream combiner note).
- `ggml-org/Qwen3.8-Flash-Next-GGUF` `convert.log` — exact mmproj tensor list
  and KVs; `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`.

Model artifacts:

- `Qwen/Qwen3.8-Flash-Next` `config.json`, `preprocessor_config.json`,
  `video_preprocessor_config.json`, `model.safetensors.index.json`
  (vision 333 tensors; `mtp.*` 31 tensors; token ids).
- `unsloth/Qwen3.8-Flash-Next-GGUF` — `mmproj-F16/BF16.gguf`, `MTP/` heads
  (sizes and header verified; shared-Q8_0 recommended; 1.67x/66.1% greedy
  single-stream; concurrency caveat).
- `ggml-org/Qwen3.8-Flash-Next-GGUF` — official mmproj Q8_0.

Flash Gordon:

- `MTP_FEASIBILITY_2026-09-13.md` — pack schema, gate/capability, acceptance
  math, A/B plan; `PRD.md` §"Multimodal and MTP" (rank-0 prefill, proposal
  boundary, state isolation).
- Deployed-source audit: schema dump of the 4-shard UD-Q4_K_XL (1224 tensors;
  0 vision, 0 MTP) — recorded under the work log with hash.

Open questions to pin during V1/M1: exact HF video sampling policy (fps) and
whether to honor the HF image bounds (65,536..16,777,216 px) rather than
llama.cpp's narrower defaults; bilinear pos-embed interpolation flags;
embeddings-only batch support in the prefill pipeline; and the upstream
merge state of the two MTP PRs.
