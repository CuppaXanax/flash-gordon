# Flash Gordon — Qwen3.8-Flash-Next BC250 Appliance PRD

## Status and handoff boundary

Flash Gordon is a greenfield, Linux-only C17/Vulkan runtime specialized for the Unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL` artifacts and the eight-blade BC250 topology at `192.168.42.42` through `192.168.42.49`.

The implementation was originally committed at `430a250a2f43efa5ad451d22d930a9c9c07da1c0` (BC250 repository) and this PRD at `99e31edc50e51651f1dea3ab8ae95537d0c0f6f8`. Both were extracted into this standalone repository. The fleet has not been mutated.

The local implementation and component-level qualifications are complete enough for the first real-weight fleet deployment. Full-model semantic parity and throughput are not yet proven because the 111 GB source artifacts are available only on `192.168.42.42`, to which the current Codex environment does not have SSH authentication.

## Product objective

Produce a fleet-testable Flash Gordon build that:

- Consumes the existing official Unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL` four-shard GGUF artifact.
- Packs it into eight rank-local weight files plus the direct-I/O n-gram table and tokenizer.
- Starts one coordinator on blade 42 and seven workers on blades 43–49.
- Executes semantically correct batched text prefill and token decode across all 48 layers.
- Uses expert parallelism as its only distributed inference architecture: one sequential model graph per autoregressive token, with each layer's routed experts fanned out across their owning blades.
- Reaches at least 10 tok/s raw single-stream decode, with 20 tok/s as the engineering target, before MTP or multiple concurrent sessions enter scope.
- Uses every compute unit exposed by each blade's Vulkan driver. The runtime must not reject a blade merely because it exposes 24 versus 40 CUs; CU mode is telemetry and a benchmark label, not a correctness gate.
- Preserves hard pack-time limits of 10.4 GiB persistent weights and 13.5 GiB accounted Vulkan residency per rank.
- Fails closed on artifact, manifest, topology, protocol, session, rank, sequence, routing, truncation, checksum, direct-I/O, and allocation errors.

The immediate milestone is correct text prefill/decode and measured TPS on the real eight-blade fleet. HTTP serving, vision/video, and MTP remain later production milestones and must not be represented as working before their complete owned paths are implemented and qualified.

## Non-negotiable engineering rules

- Production implementations only. Do not add placeholder inference, fake success, stub kernels, minimal quant stand-ins, or simulated output.
- q36 is prior art only. Copy and adapt useful internals into owned Flash Gordon source; never link, vendor, submodule, or treat q36 as an upstream runtime.
- No worktrees, speculative branches, or competing parallel experiments. Use the current checkout.
- Pipeline parallelism is out of scope. Do not trade raw single-session expert-parallel throughput for inter-layer or inter-token pipeline scheduling.
- Subagents may implement bounded, non-overlapping components, but the primary agent owns architecture, reviews every accepted diff, reruns the full suite, and oversees commits.
- Preserve the specialized appliance design. Generic-model abstractions are out of scope unless they directly improve this model's correctness or TPS.
- Prefer io_uring and fixed registered files/buffers on steady-state storage and fabric paths.
- Do not add an artificial 24/40-CU startup gate. Vulkan should use all hardware exposed by the driver.
- Do not evict a working Ornith deployment until the exact fleet action is ready to use the released GPU memory immediately.

## Canonical artifacts

Repository: `unsloth/Qwen3.8-Flash-Next-GGUF`, directory `UD-Q4_K_XL`.

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `00001-of-00004` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `00002-of-00004` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `00003-of-00004` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `00004-of-00004` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |

`flash-gordon pack` is bound to these four ordered sizes and hashes. A dry run validates canonical sizes and the complete GGUF schema while intentionally deferring payload hashes; a full pack hashes every source shard and fails closed on any mismatch.

The real schema contains 1,224 tensors. Splitting each of the 144 routed-expert tensors into four rank-local segments yields 1,656 packed model records. The full pack adds one external tokenizer record.

## Implemented architecture

### Artifact and loading path

- Four-shard GGUF parser with exact Qwen3.8 tensor-name, shape, and quant-type validation.
- Byte-preserving Q4/Q5/Q8/IQ4_NL repacker.
- Rotating four-rank groups `{L, L+1, L+3, L+5} mod 8`.
- Exactly 128 of 512 experts per participating rank, with round-robin or frequency-profile assignment under equal residency.
- Common tensors owned by `L mod 8`; token embedding on rank 0; output bundle on rank 4; n-gram tensor retained as a rank-0 local-NVMe artifact.
- All segments and artifacts are 4 KiB aligned.
- Rank weights load through `O_DIRECT` and io_uring registered files/buffers directly into the final mapped Vulkan arena, then each tensor SHA-256 is verified.
- Tokenizer and n-gram artifacts use sealed direct-I/O paths.

### Text execution

- Owned Vulkan kernels for grouped RMS, hyperconnection gating/finalization, GDN, QSA, PLE, Q8/Q5 expert matrices, Q8_K activations, IQ4_NL n-gram dequantization, embedding, routing reduction, and output projection.
- Durable GDN recurrent state and QSA Q8-key/Q4-value/Q8-index state.
- Batched QSA prefill commits causally in token order without repeating the decode API.
- Batched n-gram lookup computes every prompt position, sorts/deduplicates/coalesces 4–8 KiB reads, uses a fixed 64 MiB cache, and performs one Vulkan IQ4_NL dequantization dispatch per microbatch.
- Prefill is a true microbatch graph, not decode repeated over prompt tokens.
- The coordinator holds the replicated common path and executes all 48 dependent layers for prefill and decode.
- At every layer, the router's top-10 experts are partitioned by expert owner and dispatched concurrently. Coordinator shared-expert work overlaps routed-expert execution; all routed results are validated and reduced before the next layer begins.
- Layer 1 alone carries the n-gram injection. Wire validation rejects missing or misplaced n-gram data.
- Protocol version is 5 so incompatible coordinator and expert-worker binaries cannot join silently.

### Current text deployment profile

A sealed text deployment requires text weights, n-gram tensor, and tokenizer flags. Vision and MTP are separately flagged future overlays and are not required to start the text evaluation profile. `serve` deliberately fails closed because its owned HTTP request path is not implemented. `eval` is the first-fleet correctness and performance entrypoint.

## Verified evidence

The following completed successfully from a clean build under WSL/Linux:

```sh
make clean
make shaders
make -j2
make test
make test-vulkan
```

Evidence:

- C17 build passes with `-Wall -Wextra -Wpedantic -Werror`.
- Core protocol, topology, quant, state, manifest, corruption, truncation, scratch-ledger, and pack tests pass.
- Production-dimension CPU/reference-versus-Vulkan kernel oracles pass on llvmpipe.
- Fixed-buffer `O_DIRECT` to final Vulkan arena test passes.
- Sealed direct-I/O tokenizer test passes.
- Eight-process dual-channel TCP/io_uring mesh test passes.
- The expert-parallel trace validator accepts a complete 48-layer coordinator/worker artifact and rejects corrupted expert-slot coverage.
- Real public shard headers were range-fetched into exact-size sparse fixtures and parsed as the actual 1,224-tensor schema.
- Canonical dry-run packing produces 1,656 model records and the following memory ledger:

| Rank | Persistent GiB | Accounted residency GiB | NVMe state file GiB |
|---:|---:|---:|---:|
| 0 | 9.967 | 11.298 | 0 |
| 1 | 9.492 | 10.948 | 0 |
| 2 | 9.448 | 10.778 | 0 |
| 3 | 9.474 | 11.657 | 6.000 |
| 4 | 10.034 | 11.365 | 0 |
| 5 | 9.448 | 10.778 | 0 |
| 6 | 9.399 | 10.730 | 0 |
| 7 | 9.596 | 11.779 | 6.000 |

All ranks are below the 10.4 GiB persistent and 13.5 GiB accounted-residency limits.

## First-fleet critical path

Do these in order. Do not claim semantic completion merely because startup succeeds.

1. Push the standalone Flash Gordon repository to its own `origin` after reviewing the public diff for secrets and unintended files.
2. On blade 42, locate the four existing GGUF shards under `/models` and run the same clean build/test suite. This build/test step should not require evicting Ornith unless its Vulkan allocation prevents the small kernel qualification suite.
3. Run `pack --dry-run` against the real shard paths. Compare the printed ledger to the table above.
4. Confirm at least roughly 110 GB of additional free NVMe space for the packed artifacts, plus 6 GB QSA state capacity on ranks 3 and 7. Do not delete or overwrite the source weights.
5. Evict Ornith only when ready to start the full pack/load sequence immediately.
6. Run the full pack. This reads and hashes all four source shards, writes `rank-00.fgw` through `rank-07.fgw`, writes `ngram.iq4nl`, writes `tokenizer/tokenizer.fgt`, and seals `manifest.fgm`.
7. Preserve the manifest, tokenizer, and rank-specific file on every destination. Rank 0 additionally requires `ngram.iq4nl`; ranks 3 and 7 require local space for their QSA session files.
8. Build the exact same Git revision on all eight blades. Do not mix protocol versions or manifests.
9. Start `flash-gordon rank --manifest ... --rank N` on blades 43–49 for ranks 1–7. Rank 0 is the `eval` coordinator on blade 42; do not start a separate rank-0 worker for the same session.
10. Run a one-token greedy text evaluation on rank 0. Capture every rank log, device name, load time, manifest hash, first token ID, first-token logit, and failure text.
11. Compare tokenizer IDs, first-token logits, and greedy continuation with the current llama.cpp/Qwen reference using the same GGUF, prompt bytes, BOS behavior, greedy sampling, and context position.
12. If parity passes, run short decode, then 32K prefill. Only after correctness is stable should optimization and TPS qualification begin.

## Required reference-parity vectors

Use at least these categories, recording raw prompt bytes, token IDs, first-token logit/token, and greedy continuation:

- ASCII factual prompt.
- Unicode and multi-byte punctuation.
- Chat-template prompt with system/user/assistant delimiters.
- Prompt crossing multiple 256-token microbatches.
- EOS-separated content to exercise n-gram segment reset.
- A prompt long enough to exercise both GDN and QSA durable history across every layer owner.

For the first investigation, compare layer outputs at layers 0, 1, 3, 7, 8, 40, and 47 if the final logits diverge. Do not paper over differences with loose output-text comparison; isolate tokenizer, embedding, GR gates, recurrent/QSA state, routing, expert output, PLE, and output projection in that order.

## Known remaining risks

- Full source-payload hashing, full packing, and loading of the real artifacts have not run in this environment.
- End-to-end model logits and greedy continuations have not yet been compared with llama.cpp on the real weights.
- Vulkan qualification here used llvmpipe, not RADV/GFX1013. Shader correctness and allocation behavior must be confirmed on BC250.
- Decode and prefill transport are direct, but steady-state paths still contain some per-operation heap allocations. Remove these using bounded startup arenas after correctness is established and before final TPS qualification.
- The manifest records `required_cu=24` as the original qualification profile, but startup intentionally does not gate on 24 versus 40. Vulkan uses all CUs exposed by the driver.
- The current prefill coordinator submits microbatches sequentially. The manifest records a window of two, but multi-microbatch overlap is not yet realized end-to-end. This is a likely limiter for the 250–300 tok/s target.
- `bench` currently prints the qualification matrix but does not execute the full benchmark sweep.
- HTTP serving, vision/video preprocessing, vision overlay loading, MTP, 1M YaRN qualification, and the 24-hour mixed soak remain unfinished product milestones.
- The one-time packer uses buffered stdio for tensor copying, although runtime loading and steady-state n-gram/session I/O use io_uring. This is not an inference TPS limiter but can be replaced later.

## Performance contract after semantic parity

Canonical measurements use three warm repetitions and the median:

| Workload | Release floor | Target |
|---|---:|---:|
| 32K prefill | 250 tok/s | 300 tok/s |
| Raw single-stream decode | 10 tok/s | 20 tok/s |
| Filled 1M sustained decode | 20 tok/s | 25+ tok/s |

Report tokenizer time separately from graph prefill. Graph prefill covers exactly 32,768 already-tokenized tokens from first graph submission until final prompt logits are ready. Also report end-to-end time to first streamed token.

Optimization order after correctness:

1. Capture and validate a real eight-rank, 48-layer expert-parallel integration trace for a correct decode token.
2. Account for the complete token wall time across coordinator GPU work, Vulkan submissions, route fan-out, worker execution, fabric transfer, straggler barriers, and reduction.
3. Choose system changes from that critical path and measure them end to end; isolated microbenchmarks are supporting evidence, not promotion gates.
4. Remove steady-state allocation, redundant host/Vulkan copies, and avoidable synchronization identified by the integration trace.
5. Realize the sealed two-microbatch prefill window and overlap routing/fabric/GPU stages.
6. Tune 128/256/512 microbatches and windows 1–4, preserving exact output equivalence.
7. Change kernels, layouts, or numerical formats only behind correctness and whole-appliance performance gates.

## Useful commands

From the repository root on Linux:

```sh
make clean
make shaders
make -j2
make test
make test-vulkan
```

Canonical dry run, substituting the actual shard directory:

```sh
./flash-gordon pack --dry-run --output /models/flash-gordon-q38 \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
```

Remove `--dry-run` only after confirming destination space and exact source paths.

## Definition of done for the immediate milestone

The immediate fleet milestone is complete only when all of the following are evidenced:

- The four real sources pass canonical full SHA-256 verification.
- The full pack is written and every rank artifact passes sealed load verification.
- All eight participating processes join protocol 5 with the identical manifest hash.
- A real prompt completes batched prefill and at least one decode step without fallback or simulated output.
- Tokenization, first logits, first token, and greedy continuation match the accepted reference within explicitly recorded numerical tolerances.
- Rank failure, wrong manifest, corrupted artifact, and truncated/stale frame tests fail closed on the fleet.
- Actual per-rank memory use remains within the sealed caps.
- Initial 32K prefill and decode TPS are recorded, even if further optimization is required.

Do not mark the broader appliance finished at this point; HTTP, multimodal, MTP, long-context, benchmark automation, performance gates, and soak qualification remain subsequent milestones.
