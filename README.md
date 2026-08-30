# Flash Gordon

Flash Gordon is a Linux-only C17 inference appliance specialized for Qwen3.8-Flash-Next on eight BC250 blades. It deliberately has no generic-model compatibility layer.

Its distributed execution model is expert parallelism only. The coordinator executes the sequential common graph for one autoregressive token while each layer's routed experts fan out across their owning blades; pipeline parallelism is not part of the design. The immediate raw single-stream decode contract is a 10 tok/s floor and a 20 tok/s engineering target before MTP or multiple sessions.

The implementation owns its complete runtime boundary: artifact format, rotating expert topology, memory ledger, GGUF parser and repacker, raw io_uring storage, fail-closed wire protocol, quantization primitives, Vulkan allocation/dispatch, and Qwen3.8-specific shaders. There is no linked or vendored inference runtime. Code adapted from another project is copied into Flash Gordon, renamed and maintained here, and admitted to a production path only after model-specific reference and Vulkan parity tests pass. Qwen's published architecture and processor behavior are the semantic authority; behavior inherited from another model runtime is not. Rank and text evaluation refuse a pack until text weights, the n-gram tensor, and tokenizer are sealed into the manifest. Vision and MTP are separately flagged overlays and are not prerequisites for the sealed text profile. HTTP serving is not enabled until its full request path is owned and qualified; the command fails closed rather than simulating inference.

## Build and test

```sh
make
make test
make test-vulkan
```

The build requires Linux headers with io_uring support, Vulkan headers and loader, and `glslangValidator`. It does not require liburing. The Vulkan suite executes production-dimension Qwen grouped-RMS, gated-residual, dense Q8_0, and routed-expert Q5_1 parity oracles. The core suite also validates the machine-readable 48-layer expert-parallel fleet-trace contract and proves corrupted expert coverage fails closed.

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
