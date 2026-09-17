# Vision tower: pack decision and rounds 2-3 worklist

Date: 2026-09-16. Companion to `docs/MULTIMODAL_MTP_SCOPING.md` (architecture
verified against the Qwen3.8-Flash-Next checkpoint, llama.cpp master, and the
official mmproj). Round 1 landed the acquisition and the packer split; this file
pins the layout decision and the exact kernel/preprocessing/integration work
that rounds 2-3 must execute.

## 0a. Round-3 partial: streaming tower and image decode (local, verified)

Landed but not yet wired into the serving runtime:

- `vendor/stb_image.h` (+ `vendor/stb_image_impl.c`): single-header PNG/JPEG
  decode, isolated in one translation unit. `fg_tower_image_decode` converts to
  the planar RGB layout the preprocessing expects.
- `fg_tower_vision_forward` (`src/tower_vk.c`): one-call vision path. It
  decodes the image, runs smart_resize (image pixel bounds), normalize, bicubic
  resize, patchify, then runs the tower with **per-stage weight streaming**:
  `tower.fgm` plus an mmap of `tower.fgw`, one tensor dequantized into a
  transient host buffer (~20 MiB peak), uploaded to a transient device buffer,
  used, and freed after the stage submission. The merger (fc1/fc2) is
  dispatched in 1024-row chunks (peak device ~19 MiB per chunk) instead of
  holding the 85 MiB fc1 matrix. This keeps the tower inside rank 0's ~200 MiB
  Vulkan slack when the ring is loaded; the round-2 full-upload path needed
  ~1.9 GiB and only ran with the ring quiesced.
- `shaders/fg_tower_matmul.comp`: 4x4 register tile plus `stride`/`m_offset`
  push fields so chunked weight slices can write into a larger output.
- `src/tower.c`: fixed a resample-plan read past the row end (weighted by zero,
  but undefined and caught by ASan); `fg_tower_dequantize` moved out of the
  test.
- Verified on llvmpipe: `tests/test_tower --stream-cpu` runs the streaming path
  in a process with no other tower context and matches the CPU reference at
  cosine 1.000000000 (320x224 image, 280 patch tokens, 70 merged). Stage parity
  tests unchanged (patch/pos 1.0, block 1.0 rel 6e-8, merger 1.0).

Caveat found while testing: running the streaming path while a second tower
Vulkan context is alive in the same process gave a small numeric divergence
under lavapipe; with the stream as the only tower context it is exact. The
serving runtime will have the ring Vulkan context plus the tower context in one
process, so the first real-GPU deployment must re-verify this path before
trusting outputs (the two contexts are independent devices; this may be a
lavapipe-only artifact).

## 0b. Round-4: image path end-to-end (deployed)

Implemented and validated:

- **API content parts** (`src/api.c`): `content` arrays accept `text` and
  `image_url` parts. `image_url.url` must be a base64 data URL with
  `image/png` or `image/jpeg`; http(s) URLs, other media types and malformed
  base64 fail as 400 with a precise message, as does any image content when the
  tower pack is absent (capability gate: `${pack_dir}/tower.fgm` exists).
  Text-only requests take the unchanged code path.
- **Prompt expansion** (`fg_runtime_generate_vision`): the API renders each
  image part as `<|vision_start|><|image_pad|><|vision_end|>` in the message
  content; the runtime splits the rendered transcript on that marker, encodes
  the text segments (BOS only once), and appends the 248053/248056/248054 ids.
  The single placeholder expands to `(W/32)*(H/32)` image tokens (the tower's
  merged token count). Splitting avoids relying on the tokenizer's
  piece-boundary special-token matching.
- **Embeddings override**: after `fg_vk_embedding_q8_0_batch`, the image-token
  rows of the owner prefill input are overwritten with the tower embeddings for
  every hyper-connection copy (`[token][copy][2560]` layout). PLE needs no
  change: the expanded stream carries token id 248056, so the N-gram hashing
  already uses the image pad id.
- **Image M-RoPE positions**: image tokens get `t = pos0`,
  `h = pos0 + i/grid_w`, `w = pos0 + i%grid_w`, and the running position
  advances by `max(grid_w, grid_h)` after the span, matching the reference
  decoder positions for Qwen-VL style images.
- **Capability gating**: vision requests force a cold start and do not commit a
  continuation session (image embeddings cannot be re-derived from text
  history), so a later text-only request resets cleanly.
- **Tower placement**: the tower runs in the rank-0 API process with its own
  Vulkan context and per-stage weight streaming (~20 MiB transient host/device,
  1024-row chunked merger), which fits the rank-0 slack with the ring loaded.

Validated on the fleet: a 256x256 red-square/blue-border PNG answered "Red"
and "Blue" through `/v1/chat/completions`; tower 3.0 s for 64 image tokens,
ring prefill 93 tokens at 39-44 tok/s, decode ~16 tok/s. Text-path gates,
battery and the 6-stage soak passed unchanged.

v1 limits: one image set per request; images are not persisted across turns
(a multi-turn request that resends a past image placeholder without its bytes
is rejected); http(s) and non-PNG/JPEG payloads are rejected; video is out of
scope.

## 0c. Remaining performance work

- Quant-native tower kernels (Q8_0/F16 in-kernel) to drop the host dequant and
  the per-request weight streaming (~3.0 s vs the 1.1 s full-resident path).
- Production attention kernel for >4096 patch tokens, occupancy tuning.
- PNG/JPEG variants beyond the common subset and exact mtmd preprocessing
  parity; video handling and token budget policy.

## 1. Round-1 pack decision: separate tower pack

**Decision: the tower is a separate pack** (`tower.fgw` payload + `tower.fgm`
manifest), produced by `flash-gordon pack-tower` from the mmproj GGUF. The ring
text pack (`manifest.fgm`, `rank-00..07.fgw`, `ngram.iq4nl`) is not touched,
not re-sealed, and not repacked by this feature.

Evidence for separate over merged:

- Rank 0's qualified resident slack is tens to low hundreds of MB; the tower is
  588 MiB (official Q8_0) and 617 MB as an artifact. Merging it into the ring
  pack would blow the rank residency ledger and force a text repack of a
  qualified pack that is currently serving.
- The runtime has no vision loader, graph, or preprocess yet (`--experimental-vision`
  is rejected). A separate artifact lets V2/V3 land incrementally; merging would
  couple tower delivery to text-pack requalification.
- The tower has its own upstream lifecycle (mmproj revision, projector digest).
  A separate manifest keeps its versioning independent.
- `flash-gordon pack` now rejects vision tensors with an explicit error; the
  only writer of vision records is `pack-tower`.

Pack layout:

- `tower.fgw`: tensor payloads copied byte-for-byte from the mmproj GGUF in
  header order, 4 KiB aligned, no cooking (the tower reads GGUF-native
  Q8_0/F16/F32 layouts outside the ring).
- `tower.fgm`: a standard `fg_manifest` v6 with `FG_MANIFEST_HAS_VISION`, 334
  records (kind `FG_TENSOR_VISION`, rank 0, layer `UINT16_MAX`, layout `GGML`),
  rank-0 ledger charged with the payload high-water. Its contract component
  digest for VISION is the value the runtime binds as
  `vision_projector_sha256`.
- `flash-gordon verify --manifest tower.fgm --pack-dir DIR --source mmproj.gguf`
  re-hashes every tensor against both the source ranges and the artifact
  payload, and checks the artifact size against the ledger.

Classification (round 1, exact names verified from the official Q8_0 mmproj
dump: 334 tensors, 324 `v.blk.0..26` + `v.patch_embd.weight/.1/bias` +
`v.position_embd.weight` + `v.post_ln.weight/bias` + `mm.0/mm.2` weight+bias):

| pattern | kind | effect |
|---|---|---|
| `v.*`, `mm.*`, `*vision*`, `*visual*` | VISION | tower pack only; text pack rejects |
| `blk.48.*`, `*nextn*`, `*mtp*`, `*draft*` | MTP | classification only (no routing/execution); `blk.48.*` is matched before the expert suffix, so `blk.48.ffn_*_exps` is opaque MTP, not a routed split |
| `*_exps*`, `*experts*` | ROUTED_EXPERT | unchanged |
| `per_layer_token_embd` | NGRAM | unchanged |
| everything else | COMMON | unchanged |

The text source schema has zero names matching the new prefixes (`v.`/`mm.`/
`nextn`/`blk.48`), so the classifier change is text-neutral.

## 2. Round 2: preprocess parity + tower kernels (V1+V2)

### 2.1 Preprocessing (exact)

- Image decode: PNG/JPEG into RGB.
- `smart_resize`: preserve aspect ratio; round both sides to a multiple of
  `patch_size * merge_size = 32`; pixel bounds image `65,536..16,777,216`,
  video `4,096..25,228,800` (use the HF bounds, not llama.cpp's narrower
  defaults). Resampler must be Pillow-compatible bicubic.
- Normalize RGB in [0,1] with `(x - 0.5) / 0.5` (mean/std 0.5).
- Patchify 16x16; temporal groups of 2 frames; odd video frame count repeats
  the last frame.
- Token count: image `(W/32) * (H/32)`; video `ceil(F/2) * (W/32) * (H/32)`.
- Parity harness: compare against llama.cpp `mtmd` embeddings for 2-3 fixture
  images; gate on cosine >= 0.999 (not pixel equality).

### 2.2 Tower kernel worklist

Tower geometry: depth 27, hidden 1152, heads 16 (d_head 72), MLP 4304,
`gelu_pytorch_tanh`, conv patch 16, temporal 2, spatial merge 2, out 2560,
learned position embedding 48x48, no deepstack. LN eps 1e-6. Weight dtypes in
the official Q8_0 mmproj: F32 for norms, biases, patch embed and position
embedding; Q8_0 for `attn_qkv`, `attn_out`, `ffn_up`, `mm.0`, `mm.2`; F16 for
all 27 `ffn_down`.

1. Conv3D patch embed as matmul: two temporal halves
   (`v.patch_embd.weight`, `v.patch_embd.weight.1`, 16x16x3x1152 F32) + bias.
   Output ordering must match mtmd patch/merge grouping before anything else
   consumes it.
2. Learned position embedding (`v.position_embd.weight`, 1152x2304 F32):
   bilinear interpolation from the 48x48 grid to the target grid with
   align-corners, then the same 2x2 spatial-merge layout as the patches.
3. Per block (x27): LN1 -> fused QKV (bias) -> M-RoPE on Q/K
   (`GGML_ROPE_TYPE_VISION`, theta 10000, `n_dims = d_head/2 = 36`, four
   sections of `d_head/4 = 18`) -> full attention (no qk-norm, no window
   pattern) -> residual -> LN2 -> GELU(tanh) -> `ffn_up` (1152->4304) ->
   `ffn_down` (4304->1152) -> residual.
4. Post-LN (`v.post_ln`) then merger: reshape 4x1152 = 4608 per merged token ->
   `mm.0` (4608->4608) -> GELU -> `mm.2` (4608->2560).
5. Execution envelope: tower runs on rank 0 alongside the API, one-shot per
   image/clip, outside the 48-layer ring; no ring protocol change. Load
   `tower.fgw` by mmap and bind records by name from `tower.fgm`.
6. Kernel reuse: dense matmul/attention primitives exist for text shapes but
   the ViT tile geometry (1152/4304/4608, 72-wide heads) needs its own
   dispatch configuration; GELU(tanh), visual M-RoPE, and bilinear pos-embed
   interpolation are new kernels. Reuse the existing Q8_0/F16/F32 load paths
   rather than introducing a second quant format.
7. Gate: per-block and final embedding cosine >= 0.999 against llama.cpp mtmd
   on the fixtures; report per-layer max abs error.
8. Cost reference: ~449M params, ~5-6 TFLOP per 1024-token image (~0.5-1.5 s on
   a blade-class GPU); to be measured in V2.

## 3. Round 3: decoder + API integration (V3+V4)

- Prompt rendering: expand the chat template's
  `<|vision_start|><|image_pad|><|vision_end|>` into the computed token count:
  ids 248053 (start), 248056 (image pad), 248054 (end); video uses 248057.
  The expansion positions are replaced by tower embeddings.
- Prefill plumbing: today rank 0's prefill consumes token ids; add
  embeddings-only positions (the graph already anticipates an embeddings batch
  input). Visual M-RoPE triples `(t, h, w)` per merged token must flow through
  the position plumbing, and token/history accounting must count the expanded
  length.
- PLE / N-gram: image positions must hash the image pad token id (248056) for
  embedding-batch positions, matching the decoder's converter contract.
- Session binding: compare the pack's VISION component digest against
  `vision_projector_sha256`; mismatch fails closed.
- API: accept base64 `image_url`/`video_url` content parts only (no URL
  fetching); enforce per-request image/video token budgets; error paths for
  oversized or undecodable inputs.
- Gates: single-image chat output matches llama.cpp with the same mmproj on 2-3
  prompts; image budget table (256..4096 px -> 64..16,384 tokens) and video
  frame-sampling policy pinned and enforced.
