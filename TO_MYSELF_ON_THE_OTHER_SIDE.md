# To myself, on the other side: 50 → 200+ TPS prefill

> **Historical index (2026-09-21).** This file is a development log: every
> number below describes the revision its section was written on, not the
> current checkout. Citable current performance is
> [PERFORMANCE_CONTRACT.md](PERFORMANCE_CONTRACT.md), derived from the live
> `FG_LEDGER` line and asserted by `tools/pi-stability.ps1`. Treat all TPS,
> prefill, latency and memory figures below as historical, especially the
> pre-ring/ring-era decode (9-21 TPS) in sections 0a5-0c and the early
> qualification records they cite. Capability notes (vision tower,
> thinking/reasoning, CU unlock) remain accurate as of their dates, and the
> flag purge in 0a9 remains in force.

You are the post-compaction me. Read this top to bottom before touching anything.
Everything here is measured, not hoped. The fleet is healthy right now; keep it that way.

## 0ab. VISION TOWER READY (2026-09-17)

Images AND video are live end-to-end on the fleet (deployed SHA `11f406ff`, merged
`a850a43`). `/v1/models` reports `image/video/video_frames`; API takes OpenAI-style
content parts - `image_url` (base64 PNG/JPEG), `video_frames` (ordered frames +
fps/max_frames), and `video_url` MP4 gated on ffmpeg (8.1.2 already on the blades).
The tower pack (`tower.fgw`/`tower.fgm`, 334 tensors, HAS_VISION) streams from disk;
the serving ring pack is untouched.

- **Tower perf:** 3.0 s -> **0.71-0.88 s streaming** per 64-token image, **0.30 s
  resident**; the streaming bottleneck was the name-sorted pack read order
  (`blk.0, blk.1, blk.10...` vs numeric acquisition: 217 backward jumps); fixed with
  offset-sorted `pread` into mapped device buffers. Bounded LRU residency keeps
  58-160 MB on rank 0. Parity cosine 1.000000000, repeats byte-identical
  (image/temporal/pairs all run the same pair pipeline).
- **Video numbers:** 8 frames @2 fps (256 tokens): wall 9.1 s, tower 3.2 s;
  MP4 testsrc 6 s: 9.4 s wall; **Big Buck Bunny 360p 10 s clip: 880 tokens, wall
  19.2 s, tower 9.7 s, prefill 904 @163 TPS** - coherent scene descriptions.
- **Text path untouched:** gates/battery/soak all in band; soak PASS.
- **Limitations (future):** per-pair tower cost scales with pair token count
  (long/high-res clips are tower-bound; cross-pair batched attention is the fix);
  media requests cold-start (no image/video history persistence across turns).

## 0aa. THINKING CRASH FIXED + REASONING_CONTENT EMITTED (2026-09-16)

Two birds, one root cause hunt (`fix/thinking-and-reasoning`, merged `e7fb96d`):

**The crash.** "Thinking" prompts (no `/no_think`) killed the ring: rank 1 fatal
`non-finite layer input at 0`, others `peer closed fabric socket`, rank 0
`distributed transport is not reusable`. NOT a token/embedding issue (the full
248K-row embedding scan was clean) and NOT concurrency. Root cause: **static
replay in `fg_owner_decode_block_chained`** - a command recorded ~80 tokens
earlier replayed with bit-identical input/state produced NaN, while
re-recording the same dispatches stayed finite (context token 81, id 283, same
in both modes). Static replay is **parked** (correctness over its ~0.25-0.45
TPS); the verified dynamic per-token recording path is used. Permanent safety
checks added: non-finite block-output/input rejection with rank/token/element
naming, `FG_NUMERICS_TRACE` per-layer/state diagnostics, token ids under
`FG_GENERATE_TRACE`. Follow-up: re-enable replay only after the replay-only
divergence is root-caused.

**Reasoning.** The API parsed `<think>` into `generated->reasoning` but never
emitted it - Pi (configured `"reasoning": true`) saw only final content. Now:
non-streaming messages carry `reasoning_content`, streaming emits incremental
`delta.reasoning_content` frames (UTF-8-safe, partial `</think>` held back)
before `delta.content`; `/no_think` omits it. Input-side continuation of
assistant `reasoning_content` unchanged. Fleet-verified: probes 200, gates
[12]/[Paris], soak PASS.

## 0a9. 40 CU UNLOCK + FLAG PURGE (2026-09-16)

**40 CU unlock (runtime UMR path, per duggasco/WinnieLV; driver-level, no firmware).**
Registers CC `0xfff80000->0xffe00000` + SPI `0x07->0x1f` written live and persisted
by a boot service. Result: 7 blades at 40 CU, `.43` at 38 (one defective WGP on
SE1.SH1 masked; pre-mask ring crashes - rank1 non-finite router probs, rank0
double-free - were traced to it). `dmesg active_cu_number`/`vulkaninfo num_cu`
stay 24 under the UMR path (driver caches topology at init); evidence is UMR
register readback + compute parity/scaling.

**Measured 24 -> 40 CU (same ring/build):** 16K prefill 307 -> 377/379 TPS
(+23%), 32K 307 -> 334/392, 4K prefill 265-298 -> 282-328; decode flat
(24.9 warm, short 26.4) - bandwidth-bound as expected. Per-blade q8 microbench
+31-48% on CU-bound shapes. Thermals 76 C peak / 169 W at 2000 MHz; governor
left at 1000-2000 (1500 cap costs decode -5% for no prefill gain). Trap paid:
`dnf install umr` upgraded Mesa without libdrm -> RADV init failure, fixed with
`dnf update libdrm` (2.4.134). Revert steps + raw evidence in the local ops
workspace (`bc-250-dbg\results\cu40-*`, runbook).

**Flag purge + CU agnosticism** (branch `cleanup/flags-and-cu-agnostic`, local
validated): deleted `FG_DECODE_PIPELINE/STATIC`, `FG_QSA_SELECT_GPU/TOPK_V2`,
`FG_DECODE_CHAIN/DIRECT_OUTPUT`, `FG_VK_HOLD_STATIC`, `FG_DECODE_EXPERT_LEGACY`,
`FG_EXPERT_BATCH_SEND`, `FG_FABRIC_DIRECT_*`, `FG_MTP_DRAFT_ECHO`,
`FG_OUTPUT_SPLIT_HIDDEN`; `FG_PREFIX_CONT` env replaced by `api --no-prefix-cont`.
The 24-CU constants (`FG_HC_INJECT_PIECES`, coordinator copy, manifest
`required_cu` allowlist) now derive from
`VK_AMD_shader_core_properties2.activeComputeUnitCount` (fallback to the
properties product, else 0 = unknown). Fleet validation pending at the new CU
state.

## 0a8. ROUND 13 COMBINED A/B - DENSE + HOP MERGED, EXPERTS REJECTED (2026-09-15)

Three branches qualified in one fleet cycle (`perf/combined-round13` scratch build
`4c63f439`, results in the local ops workspace `bc-250-dbg\results\round13-combined-*`):

- **dense-streaming MERGED**: the "GR chain at 107 GB/s" was 2.6 MB/block of
  elementwise dispatches - `group_rms_norm` 8.5 -> 40.5 GB/s, `gr_attn_read`
  96.5 -> 127.3, `gr_ffn_read` 98.4 -> 129.9; split-reduce+SiLU fused, PLE fold,
  RMS register cache. Measured +0.85 TPS short / +0.75 4K; gates + soak PASS.
  New tool `tools/fg_dense_rates.py` (per-kernel achieved-GB/s from profile logs).
- **hop-payload MERGED (opt-in, `FG_OUTPUT_SPLIT=4` only)**: rank-7 hidden
  suppression is bit-identical; same-binary toggle `FG_OUTPUT_SPLIT_HIDDEN=0`
  measured +0.35..+0.73 TPS in the 4-way config; trace shows 73 x
  `hidden-suppressed` and zero hidden sends. bf16 hop packing is PARKED by the
  no-precision-trade policy.
- **expert-streaming REJECTED**: register-positive changes cost occupancy -
  gate_up 72 -> 84 VGPRs (14 -> 12 waves/SIMD), down_reduce 48 -> 64 (20 -> 16);
  pair 1.377 ms/block vs the <1.00 bar and 133.9 GB/s vs the 185 target (control
  ~1.283). Revert `1da0af1` first, then `61b1fe2`, if retried.

**Current best numbers:** default config 25.00/25.77 short, 24.13-24.53 4K warm;
4-way + suppression 26.51/26.89 short, 25.18-25.43 4K warm. The byte-budget
ceiling says the practical band is 24-31 TPS - we are near its top with these
kernels; the next real step is the structural item (row-interleaved cooked
layout for the 175 GB/s GDN-output shape) or new ideas, not more of the same.

## 0a7. BYTE BUDGET - THE PRE-MTP CEILING (2026-09-15)

Audited from the deployed pack/manifest (sha `1368cd6d`), not from prior docs:
**6.671 GB/token** (weights 6.34, state 0.33) + 379,532 B wire + n-gram reads.
Per-rank ledger: rank 4 is the fat rank (1427 MB: 684 MB output head), others
738-793 MB. **Dense is 62% of bytes, experts only 23%.**

- Absolute ceiling at clpeak 353 GB/s, zero overhead: **18.9 ms = 52.9 TPS**
  (unreachable).
- Practical ceiling at measured per-shape rates (92-298 GB/s): **32-42 ms =
  24-31 TPS**.
- Current 39-43 ms (23.4-24.6 TPS) is ~46% of the clpeak floor but **~86% of
  the practical floor** - the build is nearly at its own byte band.
- 50 TPS needs 334 GB/s sustained (94.5% of clpeak) with 4.0 GB/token; **60 TPS
  needs 400 GB/s > clpeak: physically excluded.**
- **Falsified**: "3.3 GB active weights/token" (true 6.34 GB); "61.2 MB/layer
  GDN" (full layer 87.2 MB). Dense requant (~1.45-1.95 GB, the 62% share) is
  the big bytes lever, not experts (~0.15 GB). Hop bf16 has no codec in tree
  yet; wire recomputes to ~175 KB default / ~190-195 KB 4-way.
- Unaccounted risk: n-gram O_DIRECT reads observed 0.8-25 ms cold.

See `PERFORMANCE_BYTE_BUDGET_2026-09-15.md`. Path to >31 TPS practical is byte
reduction (dense requant first); path past 50 is speculation on top.

## 0a6. ROUND 12 - DENSE OCCUPANCY HYPOTHESIS CLOSED (2026-09-15)

Two gated r8 variants measured on the fleet (`perf/dense-kernel-latency`,
merged as `fb87195`): wave-split (`FG_DENSE_R8_WAVE_SPLIT`) and 32-block pair
(`FG_DENSE_R8_PAIR`). Both regress: ws -1.4 short / -0.8 4K, pair -3.9 / -3.4;
prefill unaffected. VGPR ground truth via `RADV_DEBUG=shaderstats`: r8 = 36
VGPRs (7 waves/SIMD), ws = 40 (6), pair = 48 (5) - the r8 kernel is already the
lowest-register, highest-occupancy of the three, and TPS order matches wave
order. The LLVM-AMDGPU proxy (86) overestimated ACO by 2.4x. Occupancy and
trip-count hypotheses are closed; neither flag is promoted. The only remaining
structural item for the 320-workgroup output deficit is the row-interleaved
cooked layout, which needs a pack-format round (evidence in
`PERFORMANCE_DENSE_KERNEL_OCCUPANCY_2026-09-15.md`).

Current state after round 12: all 8 blades on `b47c12ba`, default config, gates
[12]/[Paris], soak PASS. Decode stands at ~24.5 short / ~23.6 4K warm.

## 0a5. RANK-0 MEMORY + DECODE ROUND 11 (2026-09-15)

**Rank-0 memory pathology (blocker, fixed).** Prefill blocks stalled 16s-561s and
double-freed at the memory edge: rank 0 started at 15.65/15.94 GiB used
(`conservative_peak_margin=-524 MB`, zram engaged). Fixes (`e5b2a1f`): QSA page
transport allocated lazily at microbatch size (was 246 MB, 232 MB fixed replica
slots), rank-0 QSA mirror index 204->34 MB and GDN/PLE owner state 114->12.6 MB
(owned layers only), rank-0 state file 6.0->1.0 GiB. Margin now +46 MB; 4K prefill
max 363 ms (n=34), three batteries + full soak PASS, no double-free in ~60 min.
The double-free root cause is open (follow-up: ASAN rank-0 build + FG_VK_ALLOC_TRACE).

**Dense-kernel latency round.** r8/generic dense scale-broadcast cut (210->123
result-ops, -41%), GDN algebraic 8-column update; bit-identical; ~+0.1-0.5 ms/token.

**QSA selection offload.** Device-side selection resolution (new
`fg_qsa_select_resolve`), chunked top-k (`FG_QSA_TOPK_V2`) now default, exact-miss
fallback to the old host path; ~1.1-1.7 ms/token expected at 4K.

**Measured (binary `cbcee93a`, default config):** gates [12]/[Paris]; short decode
24.28/24.68 (was 24.01/24.16), 4K warm 23.41-23.74 (was 22.57/22.76), 32K decode
19.68 (was 17.95), 64K decode 12.03; prefill 32K 303.2 / 64K 303.4; pi-stability
PASS.

**Ops lessons:** the full fleet runbook is kept locally in the `bc-250-dbg`
workspace (intentionally not in this public repo). Short version: /tmp on the
blades is tmpfs - never write large files there (it OOM-killed a blade and
sshd). A wedged rank 0 (oom_score_adj -500, no headroom) is SSH-unreachable;
recovery is killing the workers so rank 0's fabric transport becomes fatal and
it exits. The split4 branch adds `FG_OUTPUT_SPLIT_TIMEOUT_MS` liveness so a
missing partial can never wedge the coordinator again (fleet re-test pending).

## 0a4. DECODE ROUND 8 — EXPERT GEOMETRY + GDN/GR FUSION + CUTS (2026-09-14/15)

Three parallel rounds merged (main `ed99965`, fleet binary `9ec61a98`):
`perf/decode-expert-geometry` (gate_up unrolled/specialized: -58% result-ops,
bit-identical; down Q5_1 -32%, cooked-Q8 <=1.2e-5), `perf/gdn-gr-fusion` (r8
dense 64->128 threads / 8->16 splits, GDN algebraic 8-deep state pipeline, conv
vec4 views, GR write 4-feature/load), and `perf/decode-cuts` (FG_OUTPUT_SPLIT
protocol fix: the slice frame reused rank-4's destination encoding so rank 0
rejected it with `misrouted ring decode output slice`; topk v2 opt-in).

**Measured:** short decode 23.66/24.20/24.07/24.31 (was 22.52/23.01), 4K decode
22.38-22.81 (was 21.64/21.56), 16K decode 20.96, 32K 16.60; 16K prefill 350.35
(was ~313). Soak PASS. Net +1.0..+1.3 TPS: below the instruction-accounting
projections because the pair/dense kernels are latency/barrier bound, not issue
bound.

**Flags:** `FG_OUTPUT_SPLIT=1` now qualifies (gates + batteries green) at
+0.1-0.3 TPS, within spread — opt-in. `FG_QSA_TOPK_V2=1` read worst for 4K
prefill (257-268 vs 271-299 across configs) but thermal drift makes single-run
prefill comparisons unreliable; keep opt-in until re-measured.

**Round 9 lever:** dispatch/barrier overhead — 190-220 dispatches per decode
block, compute->compute barriers per dispatch in `src/vk.c`, GR at 107 GB/s
against a 353 GB/s byte floor. Standing: 4-way head split (foreign
`rank-04.fgw` loader) and hop payload work.

## 0a3. PREFILL ROUND 7 — LONG-CONTEXT SELECTION SCAN (2026-09-14, fg-work-pref7)

One shader-only commit (`89f0ee1`, binary `eb84975b`) on top of the decode-overlap
round-2 build `59b2f559`. No vk/pack/manifest/protocol/decode change.

**The measurement (FG_PREFILL_PROFILE=1 on workers, context sweep):** the 4K->256K
TPS loss is entirely the QSA selection scan. Per QSA layer per 128-token microbatch,
rank 1, GPU ms: 4K 37.1 (index_score 0.57, topk_merge 0.96, attention 5.30, q8 tile
8.77, gate/up 7.67) -> 64K 53.4 (11.95, 4.51, 6.92, 8.64, 7.53) -> 256K mean 96.3
(47.96, 11.27, 7.01, 8.65, 7.54) -> 256K last 33K depth 147.8 (91.84, 18.94, 7.07,
8.67, 7.54). Everything else is flat with context: GDN layer 30.0 ms at 4K and 30.0
at 256K; record gather 2.1 ms; attention 7.0 ms. Selection is 61% of the 256K QSA
layer mean and 75% at depth. Host wall is another ~5.5 ms/GDN layer and 14-30 ms/QSA
layer (32 per-tile staging fences + one selection readback per layer; QSA_PREFILL_TRACE
`select_ms` 94-99 of a 160 ms layer wall at the 256K tail). That host half is QSA
staging/residency work, documented in PERFORMANCE_PREFILL_ROUND7 section 5.

**The change:** `fg_qsa_index_score` was one workgroup per (block, query); the
block-scoped key transform (q8 decode, RMS tree, RoPE) plus four 128-lane shared
dot trees (28 barriers) ran per pair. It is now one workgroup per block with the
transform once, a `vec4` key in shared, four 32-lane head groups scoring every query
of the dispatch with a four-FMA chain and a five-step XOR butterfly, and per-query
head sums staged so the query loop has no barriers. Grid is now 1D; bindings/push
unchanged. Score values move in the last ulp (subgroup tree vs shared tree);
selection ids and tie-break are unchanged.

**Fleet A/B (same pack, gates first):** gates [12]/[Paris] in three sessions;
256K prefill **274.8 / 275.0 vs 237.9 TPS (+15.6%, two runs)**; 64K
318.9/312.0/309.3 (control's own 64K band today was 267-348); 4K first request
278.8/284.0 vs the 280.8-284.4 control band, warm repeats 325.5-328.6 (control
warm band 296-320) - 4K is neutral within spread, as expected (selection is
1.5 ms of a 37 ms 4K QSA layer). Short decode 22.32, 4K decode 21.1, 64K decode
11.6 - decode untouched.

**Local gates:** make all -Werror clean; test_qsa_prefill, test_expert_prefill,
test_owner_reduce PASS; focused test_fg_vk (qsa_indexer, qsa_segmented_index_score,
qsa_prefill_chunk_liveness, qsa_prefill_prepare, qsa_attention,
qsa_resident_hierarchical_topk, q8_cooked_prefill_parity/sweep, gdn_project_cooked)
PASS. Known pre-existing failures unchanged.

**Measured and rejected this round:** (1) `fg_dense_q8_0_cooked_tile` 4x4/128-thread
variant (vec4 both operands): neutral-to-slightly-negative at 4K (271-318 vs
325.5-326.6), reverted - the kernel is not LDS bound. (2) GDN recurrence wave-32
shuffle reduction (bit-exact on wave64, kills 14 barriers/token): fails
`gdn_chunked_prefill_parity_random` under llvmpipe (subgroup size 8), needs a
`gl_SubgroupSize >= 32` guard with the shared fallback; do it next.

**Ranked next steps:** (1) topk_merge partial selection (11.3/18.9 ms at 256K);
(2) index_score query-staging / multi-block workgroups - it re-reads 32 KiB of
queries per block per dispatch, ~2 GiB per dispatch at 256K, the likely bound on
the speedup; (3) GDN recurrence shuffle with the size guard; (4) host-side QSA
staging pipeline (2.5 MiB more rank-0 headroom needs a scratch decision);
(5) a 256-thread q8 tile variant (ROWS=64, TOKEN_TILE=64) if the LDS hypothesis
is to be given one more shot.

## 0a2. PREFILL ROUND 6 — EXPERT FMA CHAINS + QSA SCORE CUT (2026-09-14, fg-work-pref6)

Two shader-only commits on top of `daa2a60` (the post-revert control). No
vk.c/expert.c/owner.c edits, no pack change, no interface change: the same
pack/manifest A/Bs directly.

**`4dd37b0` — grouped expert kernels (gate/up + q5_1 down + q8_0 down raw and
cooked).** The reverted four-lane round proved that the instruction cut is real
but +48 accumulator VGPRs cost more occupancy than it bought. This keeps the
instruction cut and the scalar accumulator geometry:

- The q8 block delta multiplies the activation once per (pair, group) (4 scalar
  muls amortised over the four rows), so every product lane is a single scalar
  FMA into the existing sixteen scalar accumulators. Per (block, group):
  gate/up Q4 ~216 -> ~140 issued FP (1.54x), Q5 ~216 -> ~156 (1.38x); q5_1/q8_0
  down (fp32 input, no delta) 8 ops per (row,pair) -> 4 FMA (2x).
- Q4_K/Q5_K block loops are duplicated behind the uniform `pc.type`, so the Q4
  stream drops the 0x01010101 high-bit merge and four high-word loads per block
  (SPIR-V: 0 merge-mask uses in a Q4-only compile, 4 in a Q5-only compile).
- The shared `sums` staging, its barrier and the serial lane-0 tree are replaced
  by a 3-step `subgroupShuffleXor` butterfly over the eight K-segment lanes:
  epilogue ~270 -> ~110 issued wave instructions, 4.6 KiB shared freed, one
  barrier removed.
- Register budget: live set unchanged (16 acc + 16 weight + 16 activation + 12
  metadata + ~20 address/loop temps); the scaled activation is computed in
  place. Occupancy must stay at the control level — that is the promotion bar.

**`fbd82db` — QSA `fg_qsa_attention_split_batch`.** Three no-geometry cuts:
`dot()` -> 4-op FMA chain (0 OpDot left in SPIR-V), the 1/16 score scale folded
into the query load once (exact 2^-4 multiply, −8 muls/tile), and
`(s - next_max) * L2E` folded to one FMA against a shared bias (−8 subs/tile).
~18% off the per-tile issue with no register or shared-footprint growth
(16 KiB kept, splits kept at 8).

**Expected (instruction-derived, NOT measured; llvmpipe only locally):**
gate/up 21.5 -> ~15-18 ms/layer, down 8 -> ~5.5-7 ms/layer, QSA 20 -> ~16-17
ms/layer; roughly 45 ms off a 270-340 ms six-layer stage -> ~410-470 TPS if the
stage stays the bound. The honest test is the battery.

**Local gates (all PASS):** test_expert_prefill (Q4 gate/Q5 up/Q5_1 down vs
decode incl. the bit-identical single-token replay), test_owner_reduce,
test_qsa_prefill (5/100/2051 records, unbalanced splits), test_fg_vk
`grouped_kquant_prefill(12/13)` and `grouped_down_prefill` (q5_1 + raw/cooked
q8_0), `make all -j8` warning-free. Known pre-existing failures unchanged.

**Fleet A/B (orchestrator, after the fleet agent releases the blades):**
1. Correctness gate (64 max tokens; expect `[12]` and `Paris`):
   `pwsh -NoProfile -File "$env:TEMP\opencode\correctness64.ps1"`
2. Like-for-like attach battery against the same-pack control:
   `pwsh -File D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 1 -LogRunDir D:\workspace\bc-250-dbg\results\pref6-<hash>`
3. Ring stability gate:
   `pwsh -NoProfile -File D:\workspace\fg-work-pref6\tools\pi-stability.ps1`
Promotion bar: 4K prefill clearly above control, 12/Paris unchanged, short and
4K decode unchanged (these shaders are prefill-only), pi PASS. Also A/B `4dd37b0`
alone vs `fbd82db` alone if the pair regresses, then report the per-kernel
gate_up/down/QSA deltas from the attach profile.

**Ranked next steps if this round regresses or stalls:**
1. QSA cooperative scores (designed, not shipped): each lane accumulates one
   record over 32 dims and reduces over 8 lanes, broadcast via the existing
   key_tile dead space. On paper 2.5x on the score phase, but it needs either
   8 query vec4/lane (+32 VGPR) or a third barrier per tile; only try with a
   measured A/B slot and an occupancy check.
2. Split-count sweep at 128 queries (`FG_QSA_ATTENTION_SPLITS` 8 -> 4) now that
   the score phase is cheaper; also try 16 if the dispatcher cap is raised.
3. Shared-memory staging of decoded weights reused across pairs in gate/up:
   worth it only if the FMA-chain version shows issue-bound behavior; it buys
   no decode work (each weight is decoded once per tile already) so it is an
   LDS-latency play with a barrier cost.
4. `dense_q8_0_cooked_tile` (2.7-3.2 ms) and `gdn_prefill_recurrence` (~3 ms)
   are small; the tile kernel's inner loop is already 8 FMA per k with shared
   staging. Leave until the big two are settled.

## 0. RING STATUS 2026-09-13 NIGHT (commit series 2125ed0..652d59d — READ FIRST)

**The ring answers correctly now.** `correctness64.ps1` on the ring pack returns
`answer=[12]` and `answer=[Paris]`, short decode 9.5-10.0 TPS, 4K battery
prefill **243.7-254.7 TPS** (4322 tokens), 4K decode 2.06 TPS.
Binary hash `81105d0d...`, fleet on `/home/user/fg-ring-pack`.

**Root cause of the empty answers was decode state, not prefill numerics.**
With `FG_NUMERICS_TRACE=1` every prefill layer's hyper digest in the ring was
byte-identical to the single-owner reference (through layer 47, first sampled
token matched). The first decode step then diverged: ring prefill advances the
GDN conv/recurrent and PLE conv state on the block owners, but rank 0's decode
replay used its own unadvanced state for every layer it did not execute. Fixed
in `652d59d` with `FG_MSG_GDN_STATE_FETCH/RESULT` (protocol enum 42/43): owners
serve one layer's conv+recurrent (~3 MiB) plus PLE state for layer 1, rank 0
pipelines the fetches and writes them into its executor before decode; ring
requests force a cold reset (prefix reuse disabled until state push-back
exists). Cost: ~115 MB + ~36 messages per cold request (~1-2 s prefill; the
4K battery went 254.7 -> 243.7 TPS).

**Kernel rounds (commits 2125ed0/562b7ad/2c69599, 98ff525/2a2b693, 652d59d,
cc299df).** Per-kernel at 4K after round 2 (FG_PREFILL_LAYER_PROFILE, ranks
3/5): topk 31.7 -> 1.9 ms, QSA attention split_batch 35.1 -> 20.3 ms, moe
gate_up 29-31 -> 21.2-22.3, moe down 9-12 -> 6.4-10.1, QSA layer 110 -> 77-80,
GDN layer 58 -> 46-49, six-layer stage GPU 455 -> 343 ms. Batch-1 decode got a
fused gate/up/SwiGLU + down/reduce pair (cc299df, 2 dispatches/layer instead of
5; short decode 9.84 -> 10.00). Decode experts are still instruction-bound;
QSA decode at 4K is dominated by mirror record-cache thrash (32 MiB cache vs a
12-layer x 1024-block working set).

**Next decode lever:** allocate the mirror's QSA index segments lazily (segment
1 is 204 MB and unused below 131072 tokens) and use the freed rank-0 headroom
for a larger `--qsa-page-cache-mib` (128 MiB holds a 4K context's 12288 pages),
which should take 4K decode from 2.06 toward short-decode speed. After that,
ring decode (per-token chain with 40 KB hops, owners using their own QSA/GDN
state) is the remaining architecture piece.

## 0u. PREFIX CONTINUATION + PREFILL ROUND 6 (2026-09-14)

**Prefix continuation shipped** (`642fd55`, doc `PERFORMANCE_PREFIX_CONTINUATION`):
ring requests now resume owner state when the plan's prefill_offset equals the
runtime state frontier (`FG_PREFIX_CONT=0` restores cold resets); qsa state
persist flushes the range the state file lags after cache-only decode. Measured
on a 4-turn growing conversation: turn prefilled tokens 5102/7647/10193 ->
2538/2538/2538 (reused 2564/5109/7655), TTFT 18.9/27.7/36.5s -> 10.4/10.5/10.4s;
an 891-token generation followed by a continuation prefilled 32 tokens in 1.42s.
Run-to-run deterministic; not bit-identical to a fully cold run (FP
chunk-boundary noise, same class as pre-existing prefix reuse).

**Prefill shader round 6** (`cdaf95d`, `6e9c428`): scalar FMA chains with the
16-accumulator register budget unchanged (the deliberate answer to the round-5
occupancy regression), Q4/Q5 loop specialization, subgroup shuffle epilogue,
FMA-chain QSA scores with folded softmax bias. Same-pack A/B: 4K prefill
286.3/284.7 TPS vs 277.5/277.7 control (~+3%); decode unchanged.

**Final validated state** (binary ed949758 + pref6 shaders): gates [12]/[Paris];
4K prefill 284.7-286.3; short decode 20.7-20.9; 4K warm decode 19.9; soak gate
PASS (6 stages, 4-turn conversation, 8 ranks, 0 failures); continuation
validator turn 2-4 flat ~1.3s TTFT.

## 0v. DECODE ROUND 5 + PREFILL REVERT + SOAK GATE (2026-09-13/14)

**Ring decode round 5** (`81f50b5`): QSA decode attention rewritten to a shared
record tile across the 12 query heads of a KV head, subgroup online softmax,
push-constant selected/splits - 0.43-0.51 -> 0.10-0.14 ms/QSA layer; QSA
attention across the 12 layers ~5.7 -> ~1.4 ms/token. Measured: short decode
21.14 TPS, 4K sustained 19.32-19.45, 4K prefill 285.4 before the prefill
experiment. Explored but rejected: fp32 nibble-dot expert conversion (fleet
neutral; the expert pair stays ~150 GB/s and needs a geometry rewrite).

**Prefill shader round attempted and reverted** (`fc06ffb` then revert): the
four-lane vec4-accumulator expert GEMM rewrite regressed the 4K battery
prefill to 258.2/258.5 TPS vs the 277-304 control band; the ACO register
pressure risk materialized. Reverted the three expert shaders, kept the QSA
prefill register-query-staging change (shared 28->16 KiB). Final same-pack
measurements: 4K prefill 277.5/277.7, short decode 20.8, 4K warm single decode
20.2, sustained 4K ~19.5.

**Soak gate installed** (`tools/pi-stability.ps1`, `00d0236`): 6 escalating
contexts (131..16387), a 4-turn growing conversation, correctness [12]/[Paris]
and per-rank liveness checks. Final run: 0 failures, 4K prefill 279.55.

**MTP:** the current source GGUF has no mtp/draft tensors (schema check over
all 4 shards, zero matches), so MTP is not implementable with this pack even if
wanted. Scaffolding (`cb62f47`) is integrated but flag-off and default-false;
leadership asked us not to ship/default it.

**Gap to 100 TPS decode (no MTP, ~3.3GB/token floor = ~9.4 ms = ~106 TPS):**
steady 4K token ~51 ms. Remaining budget: expert pair ~8.3 ms at ~150 GB/s
(geometry-bound), GDN ~11 ms (roofline-adjacent), GR chains ~7.4 ms, rank-0
fixed ~4-6 ms (output head 3.4 DRAM-bound), host ~3 ms. Next levers: expert
tile geometry, GR chain fusion, output-head split. Prefill 1000 TPS remains
the 1.47 TFLOPS roofline; realistic band 400-600 after more stage-kernel work.

## 0w. PI SESSION + OOM LESSON + Q8 COOKED (2026-09-13)

The endpoint passed a real Pi coding session: 3 turns, multi-K prefills at
232 TPS, 71/169-token generations at 14-16 TPS, then rank0 was OOM-killed with
`cc1 invoked oom-killer` - a compile was running on .42 while it served. With
~90-190 MB headroom any build on the serving blade kills the model. Mitigation
in the helper scripts: `start-rank0-ring.sh` sets oom_score_adj -500 and the
worker scripts -300, so a stray compiler dies instead of the model. Do not
compile on .42 while serving, even with the guard.

`3a66f34` adds a size-neutral cooked Q8_0 expert layout: packer cooks type-8
3-D experts with the existing `fg_cook_q8_0_rows` (same bytes), new
fg_moe_q8_0_down_cooked(.grouped) kernels, fused-pair eligibility, and
`runtime_layout()` auto-cooks old manifests in place at load - so the raw-Q8_0
expert layers (2/4/30/46/47, ~37 GB/s) get cooked speed without a full repack.
Local tests pass; fleet deploy pending (the user may be poking Pi). A full
q8-requant repack (+53 GB, runbook in the packq8 agent report) remains the
optional phase 2.

## 0x. UPDATE 2026-09-13 NIGHT-FINAL (binary 930a76ff, ring pack)

Ring decode round 3 (commits 316ce5e, f19a180, a711e29): top-k bounded to its
real bitonic width, GR mix workgroups scaled, r8 scales per-lane, GDN
recurrent unrolled x4, and the direct rank7 -> rank4 output handoff
(`FG_DECODE_DIRECT_OUTPUT`, default on). Validated on the integrated binary:

| metric | mission start | now |
|---|---|---|
| gates | empty answers | **[12] / [Paris]** |
| 4K battery prefill | 47.4 | **279.1 TPS** |
| short decode | 9.2 | **19.68 TPS** |
| 4K sustained 40-token decode | ~2 | **16.62 TPS** |
| 4K single decode row | 1.96 | 10.36 (cold-process row, noisy) |

Remaining gap to 100 TPS (~54 ms/token): instruction/latency bound, not DRAM.
Gate/up/down are nibble-unpack kernels with no dp4a on GFX1013; raw-Q8_0
expert downs run at ~37 GB/s and need the cooked-layout repack (source GGUF is
not on this machine - it lives on the pack producer host). QSA attention
batching is the other measured lever (a 4-token batching experiment regressed
and was reverted). Single-sequence pipelining is impossible without MTP
(embed(T+1) depends on sample(T)); legal overlap is 2-sequence interleaving or
vocabulary-split output (documented in PERFORMANCE_DIRECT_OUTPUT_HANDOFF).

## 0y. UPDATE 2026-09-13 NIGHT (binary c8a95258, ring pack)

Ring decode round 2 (commits 31b17c2, 9338174): the block owner records all six
layers into ONE submission - GR chain + attention + GPU top-10 routing + fused
expert pair + GPU shared-scaled reduce + gr_write - one fence and 2-4 submits
per block instead of 21-22, no host router/expert/shared readback. Rank 0
fixed costs: single-row embedding gathered on the host straight from the
mapped arena (1.2 -> 0.05 ms) and output.weight routed to the row-8 cooked PSO.

Validated on the integrated binary: gates [12]/[Paris]; battery 4K prefill
276.4 TPS, short decode 17.35, 128 decode 10.70, 4K single decode 9.84 (noisy
single-token row); sustained 4K decode 15.3-16.1 TPS. Per-token 4K budget
83.5 -> ~64-67 ms (embed 0.55 | rank1 8.8 | rank0 7.5 | ranks2-7 44.0 |
output 3.4). Blocks are GPU-bound at 6-7 ms.

Gap: 20 TPS needs ~1 ms/block from the expert pair / GDN projections /
recurrent kernels; 100 TPS additionally needs cross-rank token pipelining and
the output head folded into the chain (rank7 -> rank4 direct handoff; patch map
in PERFORMANCE_RANK0_FIXED_COST_2026-09-13.md). PERFORMANCE_PERLAYER_CHAIN and
PERFORMANCE_RANK0_FIXED_COST documents have the full budgets.

## 0z. UPDATE 2026-09-13 LATE-LATE (binary 0dd3deda, ring pack)

**Blank short answers fixed.** The API never set `think_mode`, so `/no_think`
was plain text; the model spent ~29 tokens thinking, the stripper moved it to
reasoning, and content was empty while finish=stop. `src/api.c` now detects a
leading `/no_think` in user messages, strips the directive, sets THINK_OFF,
and parses generation as non-thinking. Measured: max_tokens 1/2/4 return
`1`/`12`/`12`; gates [12]/[Paris]; battery 4K prefill 237.6 TPS, short decode
10.72, 4K single decode 8.18.

Also integrated since 0a: the ring-decode architecture (`FG_DECODE_RING=1`,
default off; correct per gates but no speedup yet - the chain serializes the
same compute the legacy path overlapped with the expert wait; 8 ranks would
need ~2x faster expert kernels to clear 15 TPS; 4K ring request killed rank0
once and the transport is not reusable after a second ring request - both open)
and the vector-tiled batch-1 expert pair (short decode 10.0 -> 10.7, covers
raw-Q8_0 downs so layers 2/4/30/46/47 use the 2-dispatch path without a repack).
Repack plan for q8 experts is documented in the edec agent report: source shards
live on the pack producer host, not this machine.

## 0a. FINAL NUMBERS 2026-09-13 LATE (binary 7e47ddae, ring pack)

Validated on my own deploy, gates then battery, back to back:

| metric | mission start | final |
|---|---|---|
| gates 12/Paris | empty answers | **[12] / [Paris]** |
| 4K battery prefill | 48.81 / 47.4 | **236.2 TPS** (band 234-257) |
| short decode (32 tok) | 9.20 | **10.06 TPS** |
| 4K single decode row | 1.96 | **7.78 fresh proc / 9.5-9.8 warm** |
| 4K sustained decode (32 tok) | ~2 | **9.08-9.14 TPS** |

Same-night workstreams: kernel rounds 1-2 (prefill 150->255), numerics root
cause + GDN/PLE state handoff (`652d59d`), fused batch-1 expert pair
(`cc299df`), lazy index segments + 128 MiB mirror cache (`170f8f7`), overlapped
mirror warming during ring prefill (`9233146`), QSA decode split attention +
bounded top-k + descriptor reuse (`b9329ea`, `87e34ce`).

Remaining, in priority order:
1. **Short decode is ~10 TPS** (~100 ms/token): rank-0 GPU 39 ms, remote-expert
   wait 35 ms (worker expert kernels at 0.6 ms/layer, ~50 GB/s for cooked
   q5_1 - a pack re-cook or a better 4-bit kernel is needed), CPU/submission
   ~20 ms. Rank-0 common path hot spots: gdn projection/recurrent 12 ms, gr
   reads 9.4 ms, shared 4.2 ms, router 2.1 ms. Successive decode rounds took
   short 9.2 -> 10.1 and 4K 2.06 -> 9.1; reaching 20+ short needs the worker
   expert format/kernels or a ring decode architecture (token walks the 8
   blocks with 40 KB hops; owners use their own QSA/GDN state - no mirror).
2. **Layers 2/4/30/46/47 take the slow non-fused expert path** (0.97 vs 0.60
   ms) - needs a repack to make them fusable.
3. First 4K decode after a fresh process pays a ~90-100 ms lazy pipeline-compile
   for the new split/merge kernels; prewarm pipelines at session open.
4. A custom 4K prompt reportedly crashed rank 0 during ring prefill in the
   decode agent's testing (not reproduced here: a 2.2K and a 4K custom prompt
   both completed, rank 0 alive). If it reappears, capture the exact prompt and
   the rank-0 tail.
5. Prefix reuse stays disabled under ring until GDN/PLE state push-back exists
   (correctness > reuse).
6. rank 0 lives on ~25-90 MB available; drop_caches before heavy phases.

## 0b. QSA RING STATUS 2026-09-13 (commit 1dc1bc0 — historical)

**The QSA-in-block cost was the page cache never being created on state-backed
worker sessions.** `fg_qsa_session_open_state` only made the page cache when
`state_path==NULL` (the coordinator mirror), so every prefill token on a worker
ran `fg_qsa_state_write_block` + `select_blocks` + `fg_qsa_state_read_blocks`
with GPU syncs in between. Fixed in `1dc1bc0`:

- bounded worker record cache (`FG_QSA_WORKER_CACHE_PAGES=4096/layer`) for
  state sessions, gather misses served from the authoritative state file
  (`state_fetch_pages`);
- completed pages persist once per layer with a batched uring write
  (`persist_prefill_state`), not once per token.

Measured with the new `PREFILL_BLOCK_LAYER` / `QSA_PREFILL_TRACE` lines:

| point | before | after |
|---|---|---|
| QSA layer @1.3K | 260-430 ms | 84-100 ms |
| GDN layer @1.3K | 62-100 ms | unchanged |
| rank1/3/5 block @1.3K | 676/797/873 ms | ~420-480 ms |
| 4K battery prefill, depth 4 | 48.81 TPS | **95.14 TPS** |
| 4K battery prefill, depth 8 | n/a | **150.25-152.17 TPS** |
| 4K decode / short decode | 1.96 / 9.20 | 2.03-2.07 / 9.42-9.71 |

Gates: both correctness requests complete, decode >= 9 (answers still empty —
the distributed-prefill numerics defect in section 8 is unchanged and is still
the merge blocker; this work is speed-only).

**Depth 8 details.** `FG_PREFILL_FRAMES=8`. Worker owners allocate only owner
slot 0 (`create_decode_slots`), the coordinator processes its own block inline
on the shared base slot, and the final-result tensor is ping-ponged; rank 0's
Vulkan budget is therefore unchanged from depth 4. 128 prefill 21.4-22.0 TPS.
Deploy hash at this state: `a7f4134250c3dac5becdc0662baa42c23191ebe19a1f0ac5386f0f6a174501b4`.

**Why it is not 200 yet (measured, not guessed).** At 4K the chain still runs
in ~2.4 s / ~1.7 s waves; effective in-flight depth is ~6 of 8, and rank-0's
own inline block measures **518 ms mean (max 620 ms)**, which alone caps the
ring at ~171 TPS even with a perfect pipeline. QSA select is now the growth
term: `select_ms` 29.8 ms/layer at first=1024 (all blocks selected, fast path)
growing to 88-105 ms/layer at 4K where the top-512-of-1024 selection runs.

**Next levers, in order (each needs a battery):**
1. Pipeline rank-0's own block: use `fg_fabric_wait_ready` polling to
   interleave owner `begin/finish` with the message loop (the non-ring
   `coordinator_prefill_pipeline` already models begin/finish in flight).
   Expected 170-200 TPS because the serial own-block cap disappears.
2. Cut `select_prefill_tile`: 1.26 s of chain latency across the 14 QSA
   layers at 4K. Keeping selection ids on the GPU (block==cache-slot when the
   cache covers the context) removes the host read+fence per tile.
3. More depth only after socket buffers grow. `FG_PREFILL_FRAMES=12`
   regressed to **26.87 TPS** with 16 MiB SO_*BUF; the extra hops exceed the
   buffers and the chain locks step. Raise `socket_configure` (fabric.c) and
   the sysctl caps together, then re-test 10-12.

**Per-kernel GPU budget 2026-09-13 (new `PREFILL_LAYER_PROFILE` /
`PREFILL_LAYER_KERNEL`, FG_PREFILL_PROFILE=1 on workers, 4K battery, ranks
1/3/5). Kernel rewrite is authorized; this is the hit list.**

GDN layer ~58ms GPU: expert grouped GEMMs 40ms
(`fg_moe_kquant_cooked_grouped` 29-31 + `fg_moe_q5_1_down_cooked_grouped`
9-12), `dense_q8_0_cooked_tile` 2.8, `gdn_prefill_recurrence` 3.0, all other
kernels <1ms each.
QSA layer ~110ms GPU: `fg_qsa_attention_split` 35.1, `fg_topk_reduce` 31.7
(slow path only: 32 queries tiles/fences per layer), experts 37,
`qsa_index_score` 1.3, `qsa_record_gather` 1.6, `qsa_attention_merge` 0.7.

Per six-layer stage: GPU 455ms of ~650ms wall. Experts are 51% of GPU; QSA
attention+topk 29%. Utilization: expert GEMMs ~28% of the 1.47 TFLOPS prefill
primitive; QSA prefill attention ~6% (it is a batch-1 decode kernel called
per query). llama.cpp on a same-lineage 35B MoE (2 blades) is the control that
this silicon can do ~450 prefill TPS with real kernels.

Rewrite priority: (1) expert grouped GEMM (kquant gate_up + down) — biggest
term on every layer; (2) one batched QSA prefill attention dispatch for all
128 queries (replace split+merge per query) plus fused batched top-k with ids
left on GPU; (3) `dense_q8_0_cooked_tile` tiling. Target after (1)+(2):
six-layer stage ~350ms → pipeline bound ~350+ TPS.

**Dead ends already paid for (do not repeat):**
- `FG_PREFILL_FRAMES=12` at 16 MiB buffers: 26.87 TPS, wave stalls.
- Depth 8 with 8 owner slots + 8 ring-output tensors: rank-0 OOM within the
  first 4K request (33 s first chunk, then fleet down). Slot-0 sharing +
  ping-pong outputs fixed it.
- Query tile 4 -> 8 (halve selection fences): select got *worse*
  (88 -> 105 ms/layer), cost +28 MiB on rank 0 and tripped the frozen pack's
  scratch ledger (64 MiB alignment) in `fg_q38_runtime_scratch_bytes`; the
  manifest cannot be repacked cheaply. Reverted.

**KERNEL ROUNDS 1-2 (2026-09-13, commits 2125ed0/562b7ad/2c69599 + 98ff525/2a2b693):
4K battery prefill 150.2 -> 169.3 -> 254.7 TPS, short decode 9.84, 4K decode
2.06.** Per-kernel at 4K, rank 3/5, after round 2 (PREFILL_LAYER_PROFILE):

| kernel | before | after |
|---|---|---|
| qsa topk_reduce (slow path) | 31.7 ms | 1.9 ms (`resident_topk_merge`) |
| qsa attention (split_batch) | 35.1 ms | 20.3 ms |
| moe gate_up | 29-31 ms | 21.2-22.3 ms |
| moe down | 9-12 ms | 6.4-10.1 ms |
| QSA layer GPU total | ~110 ms | 77-80 ms |
| GDN layer GPU total | ~58 ms | 46-49 ms |
| six-layer stage GPU | ~455 ms | ~343 ms |

Round-2 changes: expert grouped kernels moved to 4-pair tiles with 32-row
workgroups, activation hoisting, folded K-quant min, no shared staging in the
float path (dp4a is not available on GFX1013 - RADV excludes it and RDNA1 has
no dot4; documented at the selection site); QSA prefill attention rewritten to
workgroup=(query, KV head, split) with head blocking so the 12 heads sharing a
KV head decode each record once (wave64 fast path).

Deploy note: the first kernel deploy OOM-killed rank 0 at the first slow-path
chunk (the lazy batch scratch needed ~54 MiB on a mirror with ~167 MB free).
Fixes: batch 8->4 queries, selection sides 8->4 MiB, mirror
`--qsa-page-cache-mib 32`. Even so rank 0 lives with ~25-60 MB available;
`sync; echo 1 | sudo tee /proc/sys/vm/drop_caches` before heavy test phases
prevents the global OOM killer from picking flash-gordon. A real fix is on the
numerics/memory workstream (mirror index keys are 408 MB, the largest reducible
item).

**Fleet state left behind:** ring pack `/home/user/fg-ring-pack`, workers
`start-workers-ringprof.sh` (FG_PREFILL_PROFILE=1) on .43-.49, rank 0
`start-rank0-ring.sh` (FG_RING_TRACE=1) on .42. Note the ringprof worker
script also starts a rank process if run on .42 — it killed the coordinator
once; target .43-.49 only.

## 0c. ORIGINAL TL;DR — the pre-ring plan (historical)

**The 200 TPS lever is the layer ring nothing else.** Rank 0 currently executes the
common path for all 48 layers; every other improvement is noise until that moves to
the layer owners. The reshard (single-owner experts) bought +5 TPS and the 4th frame;
the contiguous pack is staged plumbing for the ring, not a perf win by itself. If you
do only one thing: implement the ring. Expected result: **~200 TPS at 4K with 4 frames,
250-300 with 6**, versus 50.9 today. Math is in section 3.

## 1. Where everything is

Fleet: 8x BC-250 (192.0.2.42-49), UMA, 15.56 GB RAM each, budget ~14.5 GB safe.
Release dir: `/home/user/flash-gordon-live/20260910-230032-21d65c` on every blade.
Helper scripts (Windows): `%LOCALAPPDATA%\Temp\opencode\`
(`start-rank0-clean.sh`, `start-workers-clean.sh`, `quiesce.sh`, `fg-wait-listen.sh`,
`correctness64.ps1`, `fg-swap-contig.sh`, `fg-pack-fix` lives on .42).
Battery: `D:\workspace\bc-250-dbg\Measure-FlashGordonAB.ps1 -Attach -Build ep -Runs4k 1 -LogRunDir <dir>`.
Fleet tool: `D:\workspace\bc-250-dbg\Invoke-BC250Fleet.ps1` (creds in its `.env`).

**Live pack (running now):** `/home/user/flash-gordon-q38-cooked` — single-owner
`l%8` layout, 4 frames. Measured 4K **48.9-50.9 TPS**, 128 prefill 21-25, short decode
9.9, 4K decode 7.1, gates 12/Paris. Recovery copies: `/home/user/flash-gordon-q38-single`
on .42 and each worker's own file in its `q38-single` dir. Manifests backed up in
`q38-cooked` as `.bak-singleowner` and `.bak-contig2-failed`.

**Live binary (ring audit, 2026-09-12):** sha256
`032137d39b2cd8371f29748987b72933bb312007554e57fe9404ebe93c9c5043`, built from commits
through `5468c0b`. Fleet gates green, 4K 50.89, 128 prefill 24.25, short decode 9.79.
Worker instrument: `FG_BLOCK_BENCH=1` (bench mode, exits after printing BLOCK_BENCH
lines; rank 4 is the ideal subject: 6 GDN layers, no PLE/QSA). `FG_WORKER_OWNER=1`
enables the worker owner executor in the live loop (off by default; QSA layers need
section 4D before enabling in production).

**Staged pack (ring layout, NOT distributed):** `/home/user/flash-gordon-q38-contig2`
on .42 only. Rebalanced contiguous ownership map
(`/home/user/contiguous2.expert-map`):
- blocks: 0-5→rank1, 6-11→rank0, 12-17→rank2, 18-23→rank3, 24-29→rank4, 30-35→rank5,
  36-41→rank6, 42-47→rank7; all 512 experts per layer on the block owner; `token_embd`
  on rank7 (pack special case).
- Rank 0's replicated arena in this pack = 14,943,436,800 B — the exact size that has
  booted before. Rank 7 persistent 10.382 GiB (embedding), rank 4 9.906, all under cap.
- To deploy: upload `rank-01..07.fgw` from .42 to their blades (Windows relay, ~3 min
  each, **verify sizes; the relay has silently corrupted a shard once** — the runtime's
  per-tensor SHA check catches it at worker startup), upload `manifest.fgm` to all 8,
  run `fg-swap-contig.sh` (change its `contig=` path to `q38-contig2`), then restart
  rank 0 first, wait for `ss -ltn` to show 19100/19101, then workers.

  **DO NOT DEPLOY contig2 ON THE CURRENT BINARY.** Contiguous blocks put QSA layers
  (3, 7, 11, ... 47) on all eight ranks (1-2 each). The runtime hardcodes QSA
  ownership to ranks {3,7} with exactly 6 layers each:
  - `qsa_owner_runtime_create` (`src/runtime.c:842`) errors unless a rank owns
    exactly `FG_QSA_OWNER_LAYER_COUNT` (6) QSA layers — every worker would exit
    at startup on contig2.
  - `qsa_owner_index` (`src/runtime.c:1217`) maps only 3→0, 7→1; is used by
    `coordinator_publish_qsa_pages` (1614), `coordinator_fetch_qsa_pages` (1676)
    and `coordinator_qsa_barrier` (1737) — rank 0 would hard-error at the first
    chunk ("QSA page owner is not rank 3 or 7").
  Section 4D must land before contig2 can boot. Also commit the working-tree
  `src/pack.c` + `src/q38_schema.c` change first: contig2 was packed with
  `token_embd` on rank 7 and HEAD's validator rejects that manifest.

Disk on .42: 179 GB free. Do **not** delete `/home/user/flash-gordon-pipeline-candidate`
(131 GB, the ceiling harness's protected LKG pack). `q38-single` (104 GB) stays until
contig2 has served production for a while.

## 2. Why perf is flat today

Rank 0's coordinator loads **all** common tensors (5,506,252,800 B) plus its own six
layers' expert shards and executes all 48 layers' common path on its 24 CUs. Ownership
maps only change where weights live, not who computes. Measured per-128-token-stage
budget (rank 0, Sept 3 profile): experts 252 ms, GDN projections 153 ms, f32 gr 67 ms,
router 36 ms, QSA 28 ms, stage wall ~650 ms, GPU duty 88-94%. Four frames hide the
collect; the GPU sum is the wall. At 50 TPS the completion interval is 2.53 s per
128-token chunk, consistent with rank 0 serialising ~0.6 s of common work per chunk.

## 3. The ring: design and math

For each chunk, the token batch flows through **contiguous layer blocks**:
rank1 (0-5) → rank0 (6-11) → rank2 (12-17) → rank3 → rank4 → rank5 → rank6 →
rank7 (42-47) → rank0 (output projection). Each rank executes its six layers
end-to-end with local experts; the only cross-rank traffic is the hyper state at block
boundaries: 128 tokens x 10240 x 4 B = 5.24 MB per hop, 8 hops per chunk ≈ 42 MB.
Measured bulk rate ~114-150 MB/s → ~0.3 s per chunk transfer ceiling ~420-450 TPS.

Compute per rank per chunk after distribution: experts ~252 ms + common for six layers
~42 ms ≈ 300 ms. Chain latency for one chunk ≈ 8 x 330 ms ≈ 2.6 s; with 4 frames in
flight completion interval ≈ 0.65 s → **~200 TPS**; 6 frames → **~300 TPS**. Frames
must be issued without blocking on results (section 4C).

Chunk-coverage constant (why blocks are contiguous): with `l%8` ownership a token hops
every layer — 48 x 5.24 MB per chunk ≈ 250 MB, ~60 TPS ceiling. Contiguous blocks are
non-negotiable. Per-layer expert bytes measured (`EXPERT_LAYER_SUMS` in the probe):
44 layers 1,572,864,000 B; layers 2/30/46/47 1,835,008,000 B; layer 3 2,044,723,200 B.
Cheapest six-layer blocks are 6-11, 12-17, 18-23, 24-29, 36-41 at 9,437,184,000 B.
That is why rank 0 owns 6-11 in contig2 (its arena then equals the proven-boot size).

## 4. Implementation plan (in order)

**A. Protocol — DONE, do not rewrite.** `FG_MSG_PREFILL_LAYER_WORK/RESULT` (enum 15/16),
`fg_prefill_layer_work_encode/decode` and `fg_prefill_layer_result_encode/decode` exist
and are tested (`tests/test_core.c` round-trip, `tests/test_fabric.c:136-137` even has a
chain-hop test: rank r receives from r-1, runs layer r, forwards to r+1, rank 7 returns
to rank 0). Message carries layer, source/destination, flags, first_token, token_count,
positions (3*N u32), hyper (N*10240 f32), optional ngram embeddings
(`FG_LAYER_WORK_HAS_NGRAM`). 5.24 MB at N=128.

**B. Worker: owner executor + layer handler.**
- `fg_rank_main` (`src/runtime.c:1160`) currently creates `fg_model` (non-replicated),
  an expert executor, and for rank 4 an output executor. Add an owner executor:
  `fg_owner_executor_create(&owner, model, err)`. **Caveat:** that constructor
  hardcodes `executor->replicated=true` (`src/owner.c:216`), so `owns_layer` claims
  all 48 layers and `gdn_state` allocates for all 36 GDN layers (~120 MB wasted).
  Add a `replicated` parameter (coordinator passes true, workers false) and scale
  the gdn_state/ple_state allocation to owned layers. Keep the existing expert
  executor; the owner's fire/collect uses the local expert path (no tree, all 512
  experts of the block's layers are local).
- `rank_worker_loop` (`src/runtime.c:1149`) handles bulk messages at ~1156. Add:
  on `FG_MSG_PREFILL_LAYER_WORK`, decode with a reusable buffer sized
  `FG_PREFILL_LAYER_WORK_MAX_BYTES`, write hyper+positions into the owner's prefill
  input path, run `fg_owner_prefill_layer_begin/finish` for each layer from
  `work.layer` through the end of this rank's block, with a **local** expert
  fire/collect pair built from `fg_expert_prefill_enqueue` / `fg_expert_prefill_finish`
  (see `src/runtime.c:506-513` for the self-route pattern rank 0 already uses), then
  encode a result and forward it as the next block's work if another block follows,
  or as `FG_MSG_PREFILL_LAYER_RESULT` to rank 0 if this was the last block.
  Reuse the receive buffer for the next hop encode to avoid per-hop malloc.
- Layer 1 ngram: layer 1 lives in rank1's block. Rank 0 does the ngram lookup for the
  chunk (it owns the ngram store) and attaches embeddings to the first hop with
  `FG_LAYER_WORK_HAS_NGRAM`. Do not build ngram stores on workers.

**C. Coordinator: async chain dispatch.**
- Replace the per-layer `fg_owner_prefill_layer_begin/finish` calls in
  `coordinator_prefill_pipeline` (`src/runtime.c`, function near line 2018) for
  non-rank0 blocks with: begin = embed (rank 0) + ngram lookup + send work for the
  first layer of block 0 (rank1); finish = receive the result from the last block.
- **Issue frames without waiting**: keep the existing four frame buffers and loop, but
  sends must not block on the previous frame's completion. Use the fabric's direct
  send (or a small send queue) and collect results asynchronously by frame sequence
  (`fg_frame_sequence`). A simple v1: per outer iteration, send all `FG_PREFILL_FRAMES`
  chunks' first hops, then drain `FG_PREFILL_FRAMES` results in order before publishing
  QSA pages. That alone gives the chain overlap.
- Keep the local path for rank 0's own block (6-11) exactly as today.
- Preserve `coordinator_publish_qsa_pages` ordering per chunk.

**D. QSA state for all ranks (the hard part; blocks B and C).**
Status 2026-09-12: the owner generalization is implemented and live-parity verified
(dynamic owners from `layer_owner` in publish/fetch/barrier, per-rank sequences,
replica commit accepts any rank, `qsa_owner_runtime` accepts up to 6 owned layers;
fleet on the new binary measures 4K 50.89, gates green). Remaining for contig2:
worker QSA session/mirror for its 1-2 layers, owner-local page publish during the
block (local `qsa_owner_writer_enqueue`, bypass the `peer==0` append path), and rank
0's decode mirror page sidecar (or verified cold-fetch). Worker layer-work handler,
block bench, and non-replicated owner executor are in (`5468c0b`).
In the ring each block owner *computes* its own 1-2 QSA layers, so it must hold the
authoritative session + state file for those layers, and decode's cold-fetch must
route per layer instead of to ranks {3,7}. Changes, all in `src/runtime.c` plus a
small session open on the worker:
- `qsa_owner_runtime_create` (runtime.c:834): drop the
  `layer_count==FG_QSA_OWNER_LAYER_COUNT` requirement; size the layer array to the
  rank's actual ownership (0-2 for contiguous blocks); ranks with zero QSA layers
  keep `enabled=false`.
- `qsa_owner_index` (1217): replace with `manifest->layer_owner[layer]` everywhere
  (publish 1614, fetch 1676, barrier 1737). Barrier loops over owners {3,7} —
  generalize to the distinct set of QSA owners (or barrier only owners that
  received appends this session).
- Publish becomes owner-local: the block owner commits its own QSA pages during
  its block via its existing `qsa_owner_writer_enqueue` (runtime.c:790) — bypass
  the `peer==0` guard (handle_qsa_page_append 937, fetch 982, barrier 962) or call
  the writer directly. Rank 0 keeps publishing only its own block's layers.
- Rank 0 decode still runs all 12 QSA layers via its mirror. Its hot cache must be
  fed: either have each owner attach its page records for the block to the layer
  result message (a few KB), or accept cold-fetching the context from the eight
  owners on the first decode token (`coordinator_fetch_qsa_pages` generalized per
  above). Prefer the sidecar at first; cold fallback must be tested.
- Worker session: on the worker, open the QSA session for the block's layers in
  the owner executor (authoritative, `fg_owner_qsa_open`/`open_decode` in owner.c
  with a local state path) or mirror with a local fetch callback. Residency cost:
  index ~34 MiB/layer + selection scratch; the tightest ranks have ~1.4-1.8 GB
  headroom, so validate against the pack ledgers before enabling.
- Keep writer depth **8**; raise replica depth only after correctness gates pass.
- Gate: correctness 12/Paris, decode unchanged (9.9 short), then battery.

**E. Frames.** After C+D, re-test 5-8 frames. Keep `FG_PREFILL_FRAMES=4`,
`FG_OWNER_SLOT_COUNT=4`, replica depth 64 as the known-good baseline.

## 5. Memory budget (UMA)

Rank 0's replicated arena = shared (all common) + own experts. The split-arena commit
(`fc4fc4a`) allocates them as two buffers; the probe (`a0b64c2`) prints both sizes
(`REPLICATED_PROBE`) and per-layer expert sums (`EXPERT_LAYER_SUMS`). Known values:
shared 5,506,252,800 B; rank 0 experts must be <= ~9,437,184,000 B for a ~14.94 GB
total (boots by ~30 MB). The ring removes the shared-all-layers term entirely: rank 0
will load only its own block's common tensors + experts (~2-3 GB), which is the real
reason the ring is also the memory fix. Until then, any pack for rank 0 must use a
cheapest six-layer block.

## 6. Verification protocol

Every change: (1) build clean `make flash-gordon` in WSL; (2) deploy via the standard
chain (quiesce, upload patch, build on .42, distribute `runtime-fix.tar.gz`, extract);
(3) restart **rank 0 first**, wait for `ss -ltn | grep 191`, then workers; (4) run
`correctness64.ps1` (12 / Paris, >= 64 max tokens); (5) run the attach battery and
report 128/4K/short-decode like-for-like. Never claim perf without (5). Never enable an
unverified pack swap without a hash or a boot-before-swap check. Decode must never
regress: ring work is prefill-only until decode gets its own plan.

## 7. Operational pitfalls (learned the hard way)

- Start order: rank 0 first, wait for its fabric listener, then workers. A worker that
  connects twice kills rank 0 with `duplicate rank N channel 0`.
- Transfers: the Windows relay corrupted one 10 GB shard at identical size. The runtime
  verifies per-tensor SHA at worker startup; if it fails, re-transfer that file.
- `flash-gordon pack` refuses a non-empty output dir. Pack to a fresh dir.
- `common_owner` must use `m->layer_owner[layer]` (fixed in `9383009`); `token_embd`
  is placed on rank 7 for cap balance.
- Profiling mode (`FG_PREFILL_PROFILE`/frame trace) hung the fleet once and is parked.
  Do not restart the fleet into profiling mode without a soak test.
- Disk: delete a pack only after confirming nothing references it; `pipeline-candidate`
  is the harness's; keep one recovery pack (`q38-single`) while trying a new layout.
- **Never build on a serving blade.** `make -j8` on .42 next to live rank 0 caused an
  OOM livelock: SSH command execution stalled for ~20 min, rank 0 was OOM-killed, and
  the box only recovered after that. Quiesce all ranks first (`.42` free: 14.5 GiB).
- **The fleet tool's 15 s socket read timeout does not kill the remote command** — it
  keeps running detached from the client. Never run builds/packs in the foreground
  through it: `nohup <cmd> > /tmp/x.log 2>&1 &`, then poll with a short script.
- The 16 GB "blade" is 15.56 GB system RAM minus ~0.6 GB OS; the driver accepts ~15.5 GB
  total but a single allocation above ~15.5 GB fails with Vulkan result -2.

## 8. First actions, in order

**RING STATUS 2026-09-13 (read this first).** The layer ring now boots, handshakes,
runs a full chain prefill and decodes: first ring request = 29 tokens at 13.6 TPS
prefill, 64-token generation at 9.0 TPS, unoptimized and not yet numerically
correct. All eight ranks run the contig3 pack from `/home/user/fg-ring-pack`
(rank 0 carries `token_embd` again so rank 7 fits next to its n-gram shard; that
pack was built with `FG_PACK_EMBED_RANK=0`). Ring mode is gated: rank 0 needs
`FG_PREFILL_RING=1`, workers need `FG_WORKER_OWNER=1` (scripts
`start-rank0-ring.sh` / `start-workers-ring.sh`). Without the env gates the binary
is live-parity and the fleet runs the single-owner pack at 4K 50.89 with gates
green (current state after restore).

Faults found and fixed getting this far: non-replicated worker owner executor;
`HAS_NGRAM` allowed on a block-start work message; workers cannot open fileless
mirrors, so QSA sessions are state-backed (`fg_qsa_session_open_state`); owner
guard and mirror allow one-or-two owned QSA layers; per-token position map instead
of slot-zero writes; committed frontier advanced after ring prefill; cold fetches
read from the owner's authoritative session.

**DECODE EXPERIMENT 2026-09-13 (failed, reverted).** Ported the prefill grouped
pair-tile kernels into `fg_expert_decode` for batch-1 (`3c17717`). Measured:
worker expert GPU **0.77 ms -> 1.49 ms** per layer (sel=10) - a 16-pair tile with
one pair wastes lanes; the fixed per-slot graph is better at batch 1. Reverted in
`1198fb0`; binary hash restored to `0b91ca9c...`; fleet gates green at 9.1 TPS
decode. Baseline decode budget per token (48 layers, rank-0 log): total ~101 ms =
sync1 ~40 ms + collect ~48 ms + fire/shared/reduce ~13 ms. Worker expert GPU
0.77 ms for ~50 MB = 65 GB/s vs 350 GB/s roofline; rank-0 common path ~144 GB/s
vs roofline. Both are kernel-efficiency problems, not topology.
Next decode lever: a purpose-built batch-1 expert kernel (all 10 experts in one
X-flattened dispatch, vectorized loads), or MTP/spec after.
Bench caveat: `FG_BLOCK_BENCH` skips QSA layers, so an in-ring QSA stage time has
never been measured; extend the bench (open a state-backed session for the rank's
QSA layer) before tuning ring stage depth.

Remaining defects, in order:
1. **FIXED 2026-09-13**: `invalid QSA complete-page lookup` — worker fetches now read
   the session state directly (`fg_qsa_session_state_records`); no fetch errors, two
   consecutive requests complete without poisoning the transport.
2. **OPEN — numerics**: ring output is wrong (sampler picks control tokens 16/17,
   empty text) although requests complete. Distributed prefill is the suspect, not
   decode's expert path (unchanged). Isolate by comparing the ring's final prefill
   hyper against the single-owner reference, or by dumping per-layer hyper at each
   chain hop vs a local run.
3. **RING BATTERY 2026-09-13 (new binary ringC, wrong output but valid timing):
   128 prefill 21.40 TPS, 4K prefill 48.81 TPS (88.55 s), 4K decode 1.96 TPS,
   short decode 9.20 TPS.** Single-owner reference: 4K 50.89 / 4K decode ~7.
   So the ring currently gives **no prefill win and a 4x decode regression** — do
   not tune frames until stage time is understood.
4. **Why the benchmark block lied**: `FG_BLOCK_BENCH` skipped QSA layers. Real
   blocks carry 1-2 QSA layers whose state-backed session does state I/O +
   selection over a growing context every chunk; stage time is ~1.5-2.6 s, not
   0.4 s. 4K decode fetches cold pages from workers per token (cache misses).
5. **Path to 200, revised**: (a) fix numerics; (b) profile and kill QSA state I/O
   on the hot path (record cache residency, avoid per-chunk state writes/reads,
   confirm the selection path stays on GPU); (c) only then widen the chain to 6-8
   frames (Little's law: depth x 128 / stage time). Target stage ~0.5 s at depth 8
   ≈ 280 TPS; today's stage time is the blocker, not the topology.


Ops notes: each binary cycle is build on .42 (detached) -> package ->
download -> extract on 8 -> restart (~10 min). Keep the fleet quiesced while
developing; the user approved eviction on demand.

0. **Measurement gate — PASSED (2026-09-12).** `FG_BLOCK_BENCH=1` on worker rank 4
   (new binary, single-owner shard, local experts): tokens=128, 6 GDN layers,
   **mean block 400.42 ms (min 388.46, max 442.94), ~66-68 ms/layer.** Ring model
   `128 / (T_block + hop)` ≈ **260-300 TPS** at 4 frames; ~5.1-5.6x the current
   50.89. Proceed with the ring. Re-run the bench on a QSA-containing block before
   trusting QSA-layer service time.
1. Do **not** deploy contig2 until section 4D lands (section 1 warning). Keep the
   single-owner fleet healthy; commit the pack.c/q38_schema.c fix.
2. Implement B (worker owner executor with non-replicated create + layer handler +
   worker buffers) with a one-chunk round trip to the next block and back, under a
   flag, and gate on 12/Paris.
3. Implement C (chain driver in `coordinator_prefill_pipeline`, frames issued
   without waiting) using the current single-owner pack as a *correctness-only*
   fallback where each hop stays local. Gate: 4K >= 100 TPS once D+contig2 land.
4. Implement D (QSA authority per block owner, per-layer cold fetch, page sidecar).
   Then deploy contig2 and gate: 4K >= 150 TPS, decode unchanged.
5. Tune frames (5-6) and re-measure. Target: **200-300 TPS at 4K**, decode >= 9.9 short
   until MTP is unfrozen.

The user's goal, in their words: Qwen running at interactive coding speeds, Frontier-
class intelligence on their own hardware. The ring is the hill. Climb it.

## 0ad. PREFIX CONTINUITY V2 - DIAGNOSTICS, CUT TURNS, TOOL CHURN (2026-09-17)

Three long-standing continuation gaps closed (merged `75bf18d`, fleet build
`cd5bdba9`, live dir `20260917-tool-delta`):

- **Diagnostic**: a rejected client history now logs one `SESSION_MISMATCH <check>`
  line naming the first failing sub-check (`schemas=a->b`, `schemas[i]`,
  `choice=a->b`, `choice_name`, `messages=a->b`, `message[i].<field>`,
  `message[i].tool_calls[j].<field>`). Reproduced live for all four classes.
- **max_tokens continuation**: when the previous turn was cut by the token cap
  (no EOS), the runtime synthesizes the boundary instead of refusing; cut turns
  now continue with `prefix hit, reused N` (proved with chained cut turns).
- **Tool churn**: tool-set/choice changes no longer force a full re-prefill. The
  head stays stable; a tool-delta `<|im_start|>system` block is rendered into the
  continuation suffix only when metadata changes, and the session comparison
  keys on the message prefix (tool changes tolerated; message tampering still
  `public-history-mismatch`). Byte-identical renders when tools are unchanged.

**Measured (52K context, tool added + `tool_choice` flipped to named):**
baseline 52,451 tokens prefilled / 143.1 s / miss -> fix **131 prefilled,
52,373 reused / 5.25 s / hit**. Tool remove: hit. Tampered earlier message:
still rejected. Gates + battery + soak PASS.

## 0ae. ABORT/RETRY + SYSTEM-DELTA CONTINUATION (2026-09-17)

Merged `8c8cea3` (fleet build `03511029`, live dir `20260917-sysdelta`):

- **Client abort:** both prefill pipelines now check the interrupt between
  microbatches -> `FG_ERR_INTERRUPTED` with the contiguous frontier preserved;
  the API skips the failure reset for client aborts (genuine errors still
  reset). Dead-client stop latency 2.4-2.9 s (before: the ring kept prefilling
  a dead request for 6+ minutes and the retry restarted from zero). Retries now
  resume: `prefix hit, reused 4992/5248/10624` with suffix-only prefill.
- **System-message deltas:** the leading system run is delta-eligible
  (change/add/remove/reorder) and renders as a system block in the continuation
  suffix - Pi's MCP toggling no longer costs a full re-prefill. History shrink
  and message tampering still reset; unchanged-system renders are byte-identical.
- **Measured A/B (20K context):** system changed miss/54.4 s -> **hit/1.8 s**
  (reused 20032, prefilled 66); removed 62.8 s -> 1.5 s; added -> 1.6 s;
  combined system+tool delta -> 3.2 s. Tamper/shrink still `miss`.
- Gates + battery + soak PASS on the combined build.

## 0af. HTTP ROBUSTNESS - KEEP-ALIVE COMMENTS + BUSY-SERVE (2026-09-17)

Merged (binary `6ead9d2a`, live dir `20260917-http-robust`). The API was a
silent single-slot server: `Connection: close` on every response, no bytes
between the SSE role frame and the first generated token (minutes of silence at
60-70K context -> Node/undici body timeouts -> Pi `Error: terminated` mid
prefill), no listener service while generating, and chunked request bodies
rejected.

- **SSE keep-alives:** `: keep-alive` emitted between frames when a streaming
  request is idle >10 s (hooked into the existing per-chunk interrupt check;
  never mid-frame; send failure = client-gone abort path). Raw evidence:
  comments at 10.1/21.1/32.1/43.2 s during a 27K prefill.
- **Busy-serve:** the listener is drained non-blockingly at chunk/decode
  boundaries; `GET /v1/models` and `/health` answer 200 during generation, other
  requests get `503 + Retry-After: 1` (single-session semantics unchanged).
- **Chunked request bodies** accepted (was 400); CL+TE conflicts rejected.
- Also fixed a stale-error echo after aborts (400s now carry the real message).
- Abort/retry unchanged: kill -> abort in 2.5 s, retry `prefix hit, reused 14336`.
- Preserved: gates [12]/[Paris], battery in band, soak PASS.
- Deliberately skipped: HTTP socket keep-alive/reuse (a single-threaded accept
  loop would block on idle sockets; close-per-response retained).
- **Follow-up workstream:** `docs/API_CONCURRENCY.md` - front-end/engine split
  to take HTTP I/O off rank 0's token critical path (the current chunk-hook
  servicing means socket work and token servicing share frame time; that
  coupling is a defect, not a design).

## 0ag. CONTENT MISMATCH ROOT CAUSE - EMITTER/PARSER ASYMMETRY (2026-09-18)

Merged `50328b3` (fleet binary `1572e516`, live dir `20260917-cmpfix`). The
remaining `message[i].content` prefix misses - each costing a 78-99K full
re-prefill (6-7 minutes) in the owner's session - were OUR asymmetry:
`fg_chat_parse_generated` strips trailing whitespace (and leading CR/LF after
`</think>`) when storing the assistant message, but the streamer SENT those
characters. A streaming client echoing its reassembled content therefore
diverged by one trailing byte -> reset -> full prefill, repeatedly.

- **Root fix:** streamed deltas now byte-match the stored parse (test
  `test_streamed_content_matches_stored` pins sentinel/trailing, split-token,
  CRLF, no-sentinel, no-think cases).
- **Safety net:** `api_content_equal` tolerates leading/trailing whitespace,
  whitespace-only<->empty and CRLF/lone-CR vs LF; interior edits and any
  changed/added/removed/reordered text still reset. Rules in
  `docs/TOOL_CONTINUATION.md`.
- **Diagnostic upgraded:** `SESSION_MISMATCH message[i].content@off stored="..."
  echoed="..."` (48-byte escaped snippets) - the live t4 edit logged
  `stored="" echoed=" tampered"`.
- Measured: trailing-newline echo `hit, reused 52057` (was full re-prefill);
  real edit `miss`; abort/retry 1.96 s; tool/system deltas unchanged; gates +
  battery + soak PASS.

## 0ah. VISION CONTINUATION, WORKER MEMORY, DEPTH-B, KERNELS (2026-09-23/24)

Merged to main `4b3e85d` (fleet serves `20260924-final`; gates + soak + vision
PASS). The session's arc, in order:

- **Media prefix continuation** (`752aa63`/`62219ac`): vision turns now CONTINUE
  and COMMIT the public session. Before, `media_count==0` gated both, so any
  image turn cold-started and every following turn re-prefilled the whole
  context (~2 min at 28-30K). Media identity is a SHA-256 digest of the bytes in
  message order - a changed image resets, the same image hits. Mixed
  image/text conversations now hit 91.7-99.8% of the prefix; cold-vs-continued
  is byte-identical on a faithful replay (the `/no_think` closed-think render
  delta is documented in `docs/TOOL_CONTINUATION.md`).
- **Vision crash + rank0 mask** (`9000283`/`428bfad`): a 1 MP screenshot OOM'd
  rank0's Vulkan arena (`tower Vulkan allocate buffer memory failed`) and killed
  the API. Two fixes: the tower pixel cap now derives from the 4096-token
  attention budget (`ee4213b`), `buffer_append_text` keeps the API-inserted
  vision markers verbatim (`95cbb52`), a failed vision alloc returns 500 and the
  API survives, and rank 0 masks the common shards it does not execute
  (`COORDINATOR_MASK`, 940 tensors / 3.387 GiB) - margin +45 MB -> +3.43 GiB.
  The 2.6 MB screenshot now answers in ~40 s.
- **Pageable n-gram shard** (default since `f4aec44`): workers map the shard
  (`mmap`+`MADV_RANDOM`, 128 MiB mlocked hot prefix, bounded WILLNEED
  read-ahead) instead of mlocking all of it - equal-or-better at 4K-151K and
  **3.7-3.9 GiB returned per worker**. That headroom let the QSA record window
  go 320 -> 640 MiB (rank0 618 MiB) and took 151K decode from 17.0 to **20.1**
  TPS. Lesson recorded: the earlier 640 MiB attempt without the headroom
  swapped (pswpout ~100k, block_ms 4 -> 1200) and wedged `.49`; never bump
  caches without MemAvailable/swap monitoring.
- **Paired down/reduce kernels** (`d3cd06e`): Q5_1 expert down 61.8 -> 56.5 us
  (-8.5%, 217.6 GB/s), pair -3.3%, bit-identical (cross-compiler SPV parity).
- **Depth-B batch path** (`adf45dd` + `663236f`): owner session slots,
  PREPARE/COMMIT/RESTORE transactions, protocol 51/52. B=1 byte-identical; the
  first B=2 measured 506-532 ms/step (host-read `PREPARE` dominator) and after
  device-side `vkCmdCopyBuffer` checkpoints 100-132 ms/step = 1.25-1.46x vs a
  1.6x target. Below the gate, so **fail-closed/test-only**; the pivot is a
  batched decode block (one weight pass per layer for both tokens, projecting
  1.7-2.0x). The API is still single-session until the front-end/engine split
  (`docs/API_CONCURRENCY.md`) lands.
- **Flag purge 2**: `FG_OUTPUT_SPLIT`/`FG_OUTPUT_SPLIT_TIMEOUT_MS` removed (the
  4-way head split is unconditional); `FG_NGRAM_PAGEABLE` is now an opt-out
  default; `tools/check-flags.sh` still gates every read against `docs/FLAGS.md`.
- **Socket-buffer landmine**: a reboot drops the 16 MiB sysctls and long ring
  prefills deadlock deterministically; the four values are now persisted in
  `/etc/sysctl.d/99-fg.conf` on all 8 (verified effective). Never remove them.
- Ops lesson: three parallel long subagents plus a session restart left zombie
  task sessions the UI could not cancel; kill the opencode process to clear
  them, and checkpoint-commit agent work early (the killed kernel instance's
  uncommitted shader/test was recovered from its worktree).

## 0ai. TEARDOWN FIX + BATCH-BLOCK ROUNDS (2026-09-24)

- A rank-0 wedge during batched-block bring-up exposed a **pre-existing
  double-free** (on main since the flag purge): `fg_output_slice_create` /
  `fg_output_hc_create(foreign=false)` stored model-owned tensor views and their
  destructors freed them, then `fg_model_close` freed the same views again -
  every clean shutdown/restart of rank 0 or rank 4. ASan pinned it (`vk.c:310`
  <- `output.c:300` <- `coordinator_close`), fixed by an ownership flag
  (`b61db99`), proven with an ASan negative control plus five hardened stop/start
  cycles per affected rank, deployed as `20260924-teardown`.
- Batched decode block (branch `feat/batch-block-r1`, parked, fail-closed): two
  defects found and fixed - the pair GDN consumed the un-residual input, and the
  two slots' QSA sessions share one attention scratch so the held-back GR write
  read the wrong token's block; plus the B=1/B=2 logit-bit gap root-caused (r4
  slices vs the r8 relay head use different reduction trees; the depth-B selftest
  routes its B=1 reference through the relay for the gate, the permanent fix -
  adding slice widths to the r8 list - would change serving logits and needs its
  own round). Parity is now clean, but the measured 2-slot speedup is only
  ~1.19x (100.2 vs 119.6 ms/step): the ring is assembly/hop/wake-bound (assemble
  25 ms n-gram wake, sampling relays 6 ms), and the b2 dense saving is ~1.2 ms of
  a ~40 ms/token step - far under the 1.56x byte ceiling. Multi-session value now
  hinges on the overhead cuts (async n-gram prefetch, batched relay), not more
  kernel batching.
- API M1 (front-end split) and M2 (admission queue: FIFO, 429 + Retry-After at
  4, queued-client cancel) are live; probes answer in ~7-10 ms during prefill and
  the vision tower, keep-alive and frontier-preserving abort unchanged.
- Spare blades now carry a satellite: `.50` runs bge-m3 embeddings (:8091) and
  bge-reranker-v2-m3 (:8092) under llama.cpp Vulkan systemd units; `.52`/`.53`
  are mapped for a utility chat and SDXL; ops notes in FLEET_OPERATIONS.md §9.
