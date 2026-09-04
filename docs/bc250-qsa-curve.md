# BC250/Qwen resident-QSA context curve

`make bench-bc250-qsa-curve` is a standalone diagnostic for the resident QSA
path used by Qwen3.8. It emits JSON Lines records for the public resident
selection call, its hierarchical merge work, the fixed-budget resident
attention call, and the combined selector-plus-attention path. It does not
load a model pack and does not alter production inference.
Records declare the `fg.bc250.qsa_curve.v1` schema; the primitive P5A records
remain on their original `fg.bc250.roofline.v1` schema.

The decode curve preserves the native-context boundaries at 1, 511, 512,
2047, 2048, 2049, 2051, 2052, 4096, 8192, 16383, 16384, 16385, 16387,
16388, 32768, 65536, 131071, 131072, 131073, 196608, 261888, and 262144
visible tokens. The extra 16387/16388 points distinguish the 4096-block
selector-group boundary from the nearby token and complete-block boundaries.
The batch curve uses query batches 8, 32, 64, and 128 at 4K, 64K, 131072,
and 262144 visible tokens.

Synthetic resident state is bounded to the same two 131072-token segments as
the production API: two index segments and two record segments, plus the
query, gate, position, norm, selection, and output buffers for one case.
Initialization, shader creation, allocation, warmup, and mapped-output sanity
checks are outside the timed region. Every record reports the geometry that
formed the dispatch, including complete blocks, selector groups, candidate
count, merge passes, selected stride, scratch bytes, and segment capacity.

The curve shows where selector scan, hierarchical merge, and fixed-budget
attention spend GPU time, and whether query-batch scaling differs from decode.
It does not prove end-to-end TPS, memory traffic, quality, fleet saturation,
or a production promotion threshold. Smoothness is diagnostic only; no single
point or threshold in this benchmark is a promotion claim. Run the CPU-only
geometry/formatter checks with `make test-bc250-qsa-curve`; that target does
not open Vulkan or execute GPU work.
