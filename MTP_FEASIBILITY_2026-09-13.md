# MTP / speculative decoding feasibility - 2026-09-13

Worktree: `D:\workspace\fg-work-mtp` (detached at `929bb62`). Scope of this
change: option plumbing, capability reporting, draft/verify scaffolding, acceptance
measurement. No shader/kernel/owner/expert changes.

## Verdict

1. `--experimental-mtp` was parsed and then deliberately rejected: the flag only
   reached `fg_runtime_options_resolve`, which hard-failed every experimental flag
   (`src/runtime_options.c`, old rejection at the resolve tail). Nothing downstream
   ever saw it, and `/v1/models` hard-coded `"mtp":false` (`src/api.c:2073` before
   this change).
2. The pack schema already models MTP (`FG_TENSOR_MTP`, `FG_MANIFEST_HAS_MTP`,
   `FG_COMPONENT_MTP`), and `flash-gordon pack` already routes any `mtp.*` /
   `*draft*` GGUF tensor to rank 0 and seals the flag. The runtime loader did *not*
   load MTP tensors (filtered in `model_open_replicated`), and no MTP layer can be
   executed (no kernels, no proposal path).
3. **A repack is required at minimum; a source export that retains the MTP tensors
   is required if the current GGUF dropped them.** Whether the current fleet source
   has them is a one-command fleet check (Stage 0 below).
4. The key performance finding: with this runtime's serial 48-layer ring, a
   *sequential* 1-draft verification cannot beat baseline (every committed token
   still needs one full forward; MTP work is pure overhead). Speedup requires
   K-draft batched verification (one target forward over `[t, d1, ..., dK]`) plus
   recurrent-state rollback on rejection. The rollback machinery exists as unused
   wire-protocol session transactions (`SESSION_PREPARE/COMMIT/RESTORE`); wiring it
   is owner/expert work owned by the parallel agent.
5. This commit lands: seal-aware option gating, `fg_runtime_mtp_capability`, MTP
   tensor loading in the coordinator arena, the draft/verify/accept loop with
   `FG_MTP_DRAFT_ECHO` harness, acceptance counters plumbed to `/v1/models` and
   completion metrics headers, and unit coverage.

## 1. Where `--experimental-mtp` flows

| Point | Location | Behavior |
|---|---|---|
| Help text | `src/main.c:34` | advertises the flag |
| Flag parse | `src/main.c:111-114` | sets `FG_RUNTIME_EXPERIMENTAL_MTP` in `fg_runtime_options.experimental_flags` |
| Flag bits | `include/fg_runtime.h:14-18` | `FG_RUNTIME_EXPERIMENTAL_MTP = 1u << 1` |
| Resolution | `src/runtime_options.c:290-296` (new) | allowed only when `manifest->flags & FG_MANIFEST_HAS_MTP`; otherwise precise `FG_ERR_UNAVAILABLE` telling the operator a repack is needed |
| Other experimental flags | `src/runtime_options.c:285-289` (new) | context/vision still rejected as before |
| Runtime open | `src/runtime.c:4044-4059` (new) | computes `mtp_enabled`/capability; fails closed if MTP was requested but the head is not linked |
| Capability API | `src/runtime.c:4517-4519` | `fg_runtime_mtp_capability` |
| Capability report | `src/api.c:2073,2083-2085` | `"mtp":true/false` in `/v1/models` |
| Metrics | `src/api.c:2027-2057` | `X-Flash-Gordon-Draft-Proposed/Accepted` only when nonzero |
| Stats | `include/fg_runtime.h:60-73` | `draft_proposed`, `draft_accepted` |

Pre-change, any non-zero `experimental_flags` produced
`"experimental context, MTP, and vision are not enabled in this runtime"`.
Default behavior with the flag off is unchanged (verified by the failing set below).

## 2. Pack and manifest audit

### The schema already supports MTP

- Tensor kind: `FG_TENSOR_MTP = 5` (`include/fg_manifest.h:24`), component flag
  `FG_MANIFEST_HAS_MTP = 1u << 3` (`include/fg_manifest.h:42`), component enum
  `FG_COMPONENT_MTP = 4` (`include/fg_manifest.h:54`).
- GGUF classification: any name containing `mtp` or `draft` becomes
  `FG_TENSOR_MTP` (`src/gguf.c:86`). **Exact name check matters** - see hazard below.
- Manifest storage: MTP is opaque to byte-size derivation
  (`src/q38_schema.c:19-27`), so GGML-raw and cooked Q8_0 MTP matrices both validate.
- Component digest: MTP tensors are hash-included in the manifest contract
  (`src/manifest.c:38-56,79-99`); re-sealing after a repack is automatic.

### What `flash-gordon pack` does today

- Pack loop: non-expert tensors go through `pack_common`
  (`src/pack.c:401-414`, `src/pack.c:206-231`).
- Placement: `common_owner` sends `token_embd` to rank 7 (or `FG_PACK_EMBED_RANK`),
  `output*` to rank 4, and **everything else - including `mtp.*` names - to rank 0**
  (`src/pack.c:184-201`).
- Ledger/flag: `record_segment` sets `FG_MANIFEST_HAS_MTP` and charges the bytes to
  rank 0's persistent ledger (`src/pack.c:166`).
- Validation: the packed-manifest check counts MTP separately and skips it when
  reconstructing the 1224-tensor text schema (`src/q38_schema.c:525-550`), so
  sealing MTP does not disturb the text contract.
- Deployment validation: only `FG_MANIFEST_COMPONENTS_TEXT_REQUIRED` is mandatory
  (`src/manifest.c:467-472`), so a text-qualified pack with an MTP overlay opens as
  before unless the flag is passed.

### Hazard: MTP tensor naming

`fg_gguf_tensor_layer` only accepts `blk.0..47.` (`src/gguf.c:84`). If a converter
ships the MTP block as `blk.48.*`, `fg_gguf_tensor_kind` sees no `mtp` substring,
classifies it as `FG_TENSOR_COMMON`, and the 1224-count check fails loudly
(`src/q38_schema.c:534-537`). If it ships as `mtp.0.*` (expected for a
`text_config.mtp` overlay), packing works with zero code changes. Confirm the exact
names in Stage 0; only `mtp.*`/`draft*` names pack cleanly today.

### What the runtime was missing (one fix landed here)

`model_open_replicated` included only `FG_TENSOR_COMMON` (shared) and the rank's own
`FG_TENSOR_ROUTED_EXPERT` tensors (`src/model.c:110-120` before this change). Even a
correctly packed MTP overlay was never mapped into the arena, so
`fg_model_tensor("mtp...")` returned NULL on rank 0. Fixed at `src/model.c:110-122`:
`FG_TENSOR_MTP` is loaded on the rank it is sealed to, and only there. This is
byte-neutral for packs without MTP records.

### Repack requirements, precisely

1. Source: a GGUF that actually contains the trained MTP tensors. The HF
   `text_config` declares one hybrid full-attention MTP layer
   (`"mtp": {"hybrid": true, "layer_types": ["full_attention"],
   "num_hidden_layers": 1}`, `mtp_num_hidden_layers: 1`). The UD-Q4_K_XL conversion
   in `/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/` may or may not retain them - Stage 0.
   If absent, re-export the MTP tensors (plus any `mtp.*` norm) into the GGUF source
   set; no Flash Gordon pack code change is needed for `mtp.*` names.
2. Run `flash-gordon pack` into a **new directory** (never overwrite the qualified
   pack). Expected effects: `HAS_MTP` set, MTP records on rank 0, component and
   quantization digests change, manifest SHA changes, rank 0 persistent ledger
   grows by the MTP bytes.
3. Run `flash-gordon verify` against the same sources; run `flash-gordon inspect`
   and confirm the new `components ... mtp=<n> flags=...` line.
4. Runtime memory: MTP weights are one hidden layer (~2,560 hidden). At Q4_K_XL that
   is hundreds of MB, not GiB; rank 0's 10.4 GiB persistent cap has to be checked by
   the pack ledger print. `--dry-run` reports it before writing.

## 3. Decode shape and integration points

### Current decode

- Token loop: `runtime_generate_tokens` (`src/runtime.c:4149`), emit/decode at
  `src/runtime.c:4295-4318`.
- Ring decode: `coordinator_decode_token_ring` (`src/runtime.c:3694`); local EP
  decode: `coordinator_decode_token_local` (`src/runtime.c:3885`); selector at
  `src/runtime.c:3934-3944`.
- Target head: `coordinator_output` ships `hyper` to rank 4
  (`src/runtime.c:3454-3500`); greedy argmax is `fg_output_greedy`
  (`src/output.c:95-110`, vocab 248320).
- Greedy with penalties is `fg_output_sample` fallback (`src/output.c:213-260`).
- Direct output handoff (`FG_MSG_OUTPUT_CONFIG`/`FG_MSG_OUTPUT_HIDDEN`) deliberately
  keeps the final hyper on rank 4 (`src/runtime.c:3505-3536`); rank 0 does not get
  it back. A real MTP head placed on rank 0 needs the final hyper, so the first
  implementation should disable direct handoff while MTP is active (fall back to
  `FG_MSG_DECODE_LAYER_RESULT`, which returns `decode_result.hyper`).

### MTP step (1 hybrid layer)

For greedy decode, let `h_p` be the last-layer hidden that produces target token
`t_{p+1}` and `d_1..d_K` be drafts:

```
draft step k: h'_k = MTP_k( RMSNorm(emb(x_k)) , h'_{k-1} )   with x_1 = t_{p+1}, h'_0 = h_p
logits_k       = output_head( RMSNorm(h'_k) )                (reuse output_hc_* + output.weight)
d_k            = argmax(logits_k)
verify         = one batched target forward over [t_{p+1}, d_1, ..., d_K] from the committed frontier
accept d_j     = (argmax target_j == d_j), accepted prefix commit + correction/bonus token
```

- The MTP weights need a norm (`mtp.norm` / `mtp.0.*` naming from the real export),
  an embedding of the drafted token, and the one attention layer. Executing it on
  rank 0 reuses existing owner/QSA/dense primitives but is new kernel/owner work -
  owned by the parallel agent.
- Verification is a K+1-token causal forward: the existing prefill pipeline
  (`coordinator_prefill_pipeline`, `src/runtime.c:3345`) can carry it, but the
  recurrent GDN/conv state and QSA record pages must be transactional. The wire
  protocol already defines the transaction - `FG_MSG_SESSION_PREPARE..RESTORED`
  (`include/fg_protocol.h:114-119`), `FG_OWNER_SESSION_PREPARE/COMMIT/RESTORE`
  (`include/fg_protocol.h:172-179`), wire codec and validation in
  `src/protocol.c:941-1023`. The runtime currently sends only BEGIN/READY
  (`coordinator_begin_session`, `src/runtime.c:2838`), so the transaction is the
  intended hook and is unused today.
- Rejected tokens must never be published: QSA pages are owner-published
  (`coordinator_publish_qsa_pages`, `src/runtime.c:2458`, called from the decode
  path at `src/runtime.c:3924`), and uncommitted pages must be isolated;
  PRD.md:200-205 requires exactly this ("never retain recurrent state from rejected
  tokens").

### What landed in this commit (scaffolding)

- `fg_runtime_options_resolve` seal-aware gate (`src/runtime_options.c:285-296`).
- `fg_runtime_mtp_capability` (`src/runtime.c:4517`), three states: unsupported
  (no sealed component), weights sealed (pack has MTP, not requested/executable),
  enabled (requested + sealed + executable backend).
- Open-time fail-closed: requesting MTP on an MTP pack without a linked head returns
  `FG_ERR_UNAVAILABLE` with the repack/kernels pointer (`src/runtime.c:4044-4059`).
- Draft/verify loop: after each target forward, the pending draft is verified
  against the fresh target token, then a new draft is proposed from `last_hyper` and
  the target token (`src/runtime.c:4300-4315`). `coordinator_decode_token_*` now
  optionally returns the final hyper (`src/runtime.c:3696,3872,3886,3922`).
- Proposal hook: `runtime_mtp_propose` (`src/runtime.c:3953`). The only executable
  backend today is the qualification harness `FG_MTP_DRAFT_ECHO=1`, which echoes the
  target token as the next draft. That exercises propose/verify/accept/reject with
  **bit-identical output to plain greedy decode**, because in sequential 1-draft
  mode the committed stream is always the target stream.
- Greedy-only guard: MTP + temperature or penalties fails closed
  (`src/runtime.c:4163-4167`); residual stochastic acceptance math is implemented and
  tested but not wired (`src/sampler.c:67-84`).
- Acceptance counters: stats (`include/fg_runtime.h:72-73`), completion headers
  (`src/api.c:2027-2057`), `/v1/models` capability (`src/api.c:2073-2085`).
- MTP tensor loading on the sealed rank (`src/model.c:110-122`).
- Inspect output lists component counts, including `mtp` (`src/manifest.c:687-699`).

## 4. Acceptance-rate math for 1-2 draft tokens

Let `a` be the per-position greedy acceptance probability (probability that the MTP
argmax equals the target argmax), assumed independent across draft positions. With
`K` drafts per target batch, expected committed tokens per cycle:

```
E_K = 1 + a + a^2 + ... + a^K = (1 - a^(K+1)) / (1 - a)
```

`K=1`: `E_1 = 1+a`. `K=2`: `E_2 = 1+a+a^2`.

Cycle cost, in units of one single-token target forward `F`:
`1 + K*m + beta_K`, where `m` is one MTP draft step (layer + head, plus proposal
overheads) and `beta_K` is the marginal cost of the K-draft batched verify over the
plain forward (`beta_0 = 0`, `beta_K` measured on the fleet; the ring is
latency-dominated, so `beta_2` is expected well below 1.0).

```
S(K, a) = E_K / (1 + K*m + beta_K)
```

Break-even needs `E_K > 1 + K*m + beta_K`; below that, spec decode is slower.

Speedup table (m = 0.15, beta_2 = 0.15, i.e. `K=2` denominator 1.45):

| a | E_1 | S_1 (K=1, /1.15) | E_2 | S_2 (K=2, /1.45) | TPS at 20 baseline (S_2) |
|---|---|---|---|---|---|
| 0.50 | 1.50 | 1.30 | 1.75 | 1.21 | 24.1 |
| 0.60 | 1.60 | 1.39 | 1.96 | 1.35 | 27.0 |
| 0.70 | 1.70 | 1.48 | 2.19 | 1.51 | 30.2 |
| 0.80 | 1.80 | 1.57 | 2.44 | 1.68 | 33.7 |
| 0.90 | 1.90 | 1.65 | 2.71 | 1.87 | 37.4 |

The task's 60-100 TPS target needs `S>=3`, i.e. `K>=3-4` with `a>=0.8` and small
`m`/`beta`, or a batched forward whose marginal cost is close to zero on the ring.
The measurements to take on the fleet are exactly `a` (from the new
Proposed/Accepted headers) and `beta_K` (decode seconds for a controlled K-token
batch vs single token). Note `K=1` cannot exceed ~1.6-1.9x even at `a=0.9`, and
cannot reach 2x because each accepted token still needs its own forward.

Stochastic decoding uses the residual rule implemented and tested here:
accept with probability `min(1, q(d)/p(d))` (`src/sampler.c:71-84`); greedy is the
degenerate `d == target` (`src/sampler.c:67`). Greedy spec output must be
byte-identical to greedy baseline; that is the first correctness gate.

## 5. Tests and verification performed

Build: `make all -j8` warning-free (`-Wall -Wextra -Wpedantic -Werror`).

Passing after the change (llvmpipe WSL):
`tests/test_qsa_prefill`, `tests/test_expert_prefill`, `tests/test_owner_reduce`,
`tests/test_chat`, `tests/test_chat_runtime`, `tests/test_sampler`, `tests/test_api`.

New coverage:
- `fg_sampler_spec_accept_greedy/stochastic` cases in `tests/test_sampler.c`.
- MTP gate: request on a pack without `HAS_MTP` fails, request on a sealed MTP
  manifest resolves, context/vision still fail (`tests/test_session.c`).
- `/v1/models` reports `mtp:false` and `mtp:true` from capability
  (`tests/test_api.c::test_model_capabilities`).

Pre-existing failures (confirmed identical at pristine HEAD, unrelated):
`tests/test_session.c:254,257` (retired profile-2 upgrade expectation) and the
already-known `test_core` replica-depth/SIGABRT and seed-dependent
`gdn_chunked_prefill_composition`.

Default behavior with `--experimental-mtp` absent: unchanged. Capability stays
`WEIGHTS_SEALED` or `UNSUPPORTED`, `/v1/models` says false, no generation-path
branch executes (the loop guard is `runtime->mtp_enabled`, set only when the flag,
the sealed component, and a backend all agree).

## 6. Fleet A/B plan for the orchestrator

**Stage 0 - source audit (no builds, blade 42):**

```sh
ls -l /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/
cd <build-dir> && ./flash-gordon schema --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/*.gguf > /tmp/q38.schema
grep -iE 'mtp|draft' /tmp/q38.schema || echo NO_MTP_TENSORS_IN_SOURCE
```

If `NO_MTP_TENSORS_IN_SOURCE`, stop and request an export with the trained MTP
tensors (names must contain `mtp` or `draft`; `blk.48.*` will fail the 1224-tensor
check by design). Report the exact MTP tensor names/shapes/types back to this
worktree.

**Stage 1 - repack into a new directory (never overwrite the qualified pack):**

```sh
./flash-gordon pack --output /home/user/flash-gordon-mtp-cooked \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/*.gguf \
  [--router-profile <same profile file as the current pack>] \
  --profile native-262k-microbatch-128
./flash-gordon verify --manifest /home/user/flash-gordon-mtp-cooked/manifest-native-262k.fgm \
  --pack-dir /home/user/flash-gordon-mtp-cooked --source /models/.../*.gguf
./flash-gordon inspect --manifest <manifest> | grep components
```

Expect `mtp=<n>` nonzero, `flags` with bit 3 set, and the rank 0 ledger delta equal
to the MTP bytes. Runtime opens on this pack without the flag exactly as today.

**Stage 2 - harness qualification (new binary, MTP pack, no real MTP head):**

- Start rank 0 with `--experimental-mtp` and `FG_MTP_DRAFT_ECHO=1` in the
  environment (inherited by `nohup`; export it in the rank-0 launch line).
- `GET /v1/models` must report `"mtp":true`; the other seven ranks need no env.
- Same prompt set as the baseline, greedy. Gates:
  1. completions byte-identical to the non-MTP run;
  2. `X-Flash-Gordon-Draft-Proposed/Accepted` present (proposed ~= generated-1);
  3. acceptance near 0 on prose, 100% on a repeated-token stream (echo harness);
  4. decode TPS within a few percent of baseline (harness overhead only).
- This validates the pack flag, option gate, capability, loop, and telemetry without
  claiming any speedup.

**Stage 3 - real MTP (after the MTP layer kernels and batched verify land):**

- Re-run Stage 2 without `FG_MTP_DRAFT_ECHO`. Record `a` = Accepted/Proposed per
  workload (short chat, code, 4K context).
- A/B decode TPS: same binary, same pack, `--experimental-mtp` on/off, same prompts,
  3 runs per arm; report raw TPS, accepted output TPS, `a`, and prefill parity.
- Gate on bit-identical greedy output vs the non-MTP arm, then the regression set
  (`test_qsa_prefill`, `test_expert_prefill`, `test_owner_reduce`, `test_chat`,
  `test_chat_runtime`, `test_sampler`) plus the known pre-existing failures.
- Only after that should `/v1/models` be allowed to keep `mtp:true` in production
  (it already follows the runtime capability, so this is automatic).
- Still unqualified after Stage 3: stochastic/residual MTP verification and
  concurrency; keep both out of the A/B until the PRD's transactional-state work
  (owner/expert) is in.

## 7. Open items owned by other work packages

1. MTP layer kernels + proposal on rank 0 (parallel agent: kernels/owner/expert).
2. Batched K-token target verify with `SESSION_PREPARE/COMMIT/RESTORE` rollback of
   GDN/conv and QSA page state (parallel agent: owner/expert; protocol codec is
   ready and unused).
3. Direct-output fallback while MTP is active so rank 0 can read the final hyper
   (`src/runtime.c:3505-3536` path), or ship rank 4's hidden back via
   `FG_MSG_OUTPUT_HIDDEN` (`include/fg_protocol.h:145`).
4. Real MTP tensor names/shapes from the source export, then wire
   `runtime_mtp_propose` to the head.
