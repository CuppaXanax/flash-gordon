# BC250/Qwen primitive roofline benchmark

`make bench-bc250-roofline` runs a standalone Vulkan benchmark and writes one
JSON object per line to stdout. It does not load a model pack or enter the
Flash Gordon runtime. The executable uses the Qwen3.8 dimensions from
`fg_q38_schema.h` and the same Q8_0 dense dispatch entry points used by the
model, with synthetic resident buffers.

The report has three families:

* `linear_traffic` reads 64 KiB, 1 MiB, 16 MiB, 256 MiB, and 1 GiB resident
  allocations with coalesced 32-bit loads. Its copy variant reads and writes
  every word. These are primitive access ceilings, not model throughput.
* `q8_dense_shapes` covers the Qwen hyper/channel, shared-expert, QSA, GDN,
  and vocabulary dimensions for decode batch 1 and token batches 8, 16, 32,
  64, and 128. Both the source Q8_0 layout and cooked Q8_0 layout are
  measured. Its byte fields are explicitly logical useful traffic, not a claim
  about DRAM transactions. The legacy kernel counts one weight pass per token;
  the cooked prefill kernel counts one weight pass per 32-token tile because it
  stages and reuses weights within that tile. Input and output are counted once
  per token. Cache-line fetches, duplicate workgroup reads, and padding are not
  inferred from source code and require hardware counters to quantify.
* `command_overhead` measures empty submissions and one-word dispatch batches
  from 1 through 64 dispatches. GPU timestamps cover device execution while
  monotonic wall time includes submission and fence overhead.

Each record includes schema, case identity, shape, batch/tokens, iterations,
byte and operation accounting, GPU and wall time, derived rates where defined,
and the Vulkan device name. Unknown quantities are JSON `null`; no analytical
bandwidth or TPS constant is used.

The benchmark proves per-device primitive ceilings and exposes the gap between
GPU execution and host submission/fence time. It does not prove Flash Gordon
end-to-end prefill or decode TPS, pipeline balance, expert routing cost,
attention/QSA context behavior, fabric contention, thermal steady state, or
fleet throughput. Those require separate stage-replay, communication, and
native-context tests. In particular, a high linear or Q8 rate must not be
reported as an end-to-end model result.

The benchmark separates three warmup dispatches from measured repetitions and
uses Vulkan timestamp queries for GPU time. Allocation, mapping, and shader
creation happen before measured repetitions. Run it on each blade after the
device is idle and compare JSON records by device and case; do not average
across blades until their device identity and operating conditions are known.
