# Flash Gordon

Flash Gordon is a Linux-only C17 inference appliance specialized for Qwen3.8-Flash-Next on eight BC250 blades. It deliberately has no generic-model compatibility layer.

Its distributed execution model is expert parallelism only. The coordinator executes the sequential common graph for one autoregressive token while each layer's routed experts fan out across their owning blades; pipeline parallelism is not part of the design. The immediate raw single-stream decode contract is a 10 tok/s floor and a 20 tok/s engineering target before MTP or multiple sessions.

The current eight-blade LKG qualifies at 99.647 ms/token, or 10.035 tok/s, over the final 20 unprofiled greedy frames with exact response parity and a complete 48-layer route trace. See [PERFORMANCE_TRACE_10_035TPS.md](PERFORMANCE_TRACE_10_035TPS.md) for the qualification record.

The implementation owns its complete runtime boundary: artifact format, rotating expert topology, memory ledger, GGUF parser and repacker, raw io_uring storage, fail-closed wire protocol, quantization primitives, Vulkan allocation/dispatch, and Qwen3.8-specific shaders. There is no linked or vendored inference runtime. Code adapted from another project is copied into Flash Gordon, renamed and maintained here, and admitted to a production path only after model-specific reference and Vulkan parity tests pass. Qwen's published architecture and processor behavior are the semantic authority; behavior inherited from another model runtime is not. Rank and text evaluation refuse a pack until text weights, the n-gram tensor, and tokenizer are sealed into the manifest. Vision and MTP are separately flagged overlays and are not prerequisites for the sealed text profile. The runtime also exposes a resident interactive chat frontend and a deliberately
single-threaded OpenAI-compatible HTTP frontend. Both use the exact Qwen ChatML
template and the same greedy distributed generation path as evaluation.

## Build and test

```sh
make
make test
make test-vulkan
```

The build requires Linux headers with io_uring support, Vulkan headers and loader, and `glslangValidator`. It does not require liburing. The Vulkan suite executes production-dimension Qwen grouped-RMS, gated-residual, dense Q8_0, and routed-expert Q5_1 parity oracles. The core suite also validates the machine-readable 48-layer expert-parallel fleet-trace contract and proves corrupted expert coverage fails closed.

## Interactive chat

```sh
./flash-gordon chat --manifest /srv/flash-gordon/manifest.fgm --max-tokens 512
```

The model runtime stays resident. Each turn resets its single inference session
and prefills the full in-memory transcript, which favors correctness over prompt
cache reuse. The current boot-safe serving profile uses an 8,192-token working
context so QSA records stay on the qualified resident hot path; this is a
temporary qualified profile, not the model's context target. Tokens stream directly to
the terminal. `/clear` clears the
transcript and runtime session; `/quit` exits. `SIGINT` or `SIGTERM` requests a
stop at the next boundary between completed distributed tokens. Each turn ends
with compact prefill, generation, and context-usage metrics.

## OpenAI-compatible API

```sh
./flash-gordon api --manifest /srv/flash-gordon/manifest.fgm \
  --host 127.0.0.1 --port 8000
```

The server implements `GET /v1/models` and `POST /v1/chat/completions`.
Completions accept string-content system/developer, user, assistant, tool, and
function messages; `max_tokens` or `max_completion_tokens`; and `stream`.
Non-streaming responses are JSON and streaming responses use SSE. The runtime
stays loaded, while its one session is reset before every request. The server
handles one connection at a time and closes it after one request.

Flash Gordon currently performs greedy decoding only. Requests that select
non-greedy sampling, custom stop sequences, or log probabilities receive a
clear `400` response. The API accepts OpenAI function `tools`, `tool_choice`,
historical assistant `tool_calls`, and tool results linked by `tool_call_id`.
Generated native Qwen tool tags are translated into OpenAI `tool_calls` for
both JSON and SSE responses, with `finish_reason` set to `tool_calls`; native
tool syntax is not exposed as assistant content.

## Inspect a prospective pack

```sh
./flash-gordon pack --dry-run --output /tmp/flash-gordon-q38 \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
```

The production pack path is bound to the four official Unsloth `UD-Q4_K_XL` shard sizes and SHA-256 identities. A dry run verifies the complete real GGUF metadata and canonical sizes without reading 111 GB of tensor payload; the full pack hashes every shard and fails before writing a deployment manifest if any payload differs.

`--router-profile FILE` accepts whitespace-separated `layer expert frequency` rows. Optional `--expert-map FILE` accepts one `layer=N ranks=R0,...,R511` row per layer for an explicit placement; the two options are mutually exclusive. Without either option, expert residency is round-robin. Every mode enforces exactly 128 experts on each of the four ranks participating in a layer. The map is a pack-time input only: ownership is sealed into the manifest, and rank/eval runtime never reads or requires the source map file. Maps derived from qualification prompts are oracle-only diagnostics and cannot qualify a release.

The sealed memory ledger is architecture-derived. GDN recurrent state and the Q8 indexer history are Vulkan-resident. The much larger QSA Q8-key/Q4-value history is retained by its layer owner in an aligned local-NVMe state file and only selected tokens are staged to Vulkan; its required capacity is recorded separately as `state-file` and is never counted as free GPU memory.

Manifest v4 records each tensor's physical storage layout. Routed experts, token embeddings, and narrow common projections preserve their GGML quantized bytes. Measured winning Q8_0 common matrices are cooked offline into 16-row supertiles with block-major FP16 scales and row-major quant planes. Both planes are 64-byte aligned and production matrices retain their original byte count. Runtime tensor metadata selects the matching kernel and rejects cross-layout use. A 512-expert GGUF tensor is split into four rank-local segments in ascending global-expert order according to the manifest map. All output segments begin at 4 KiB boundaries.

## Qualification contract

The manifest fixes the production prefill choice; runtime requests cannot change it. `bench` enumerates microbatches 128/256/512 and windows 1–4 against the 32,768-token, three-warm-run gate. Promotion defaults to 24 CUs. Rank startup rejects corrupted manifests, topology drift, protocol mismatches, and memory-cap violations.

No fleet command, deployment, CU unlock, or model download is performed by the local build.
