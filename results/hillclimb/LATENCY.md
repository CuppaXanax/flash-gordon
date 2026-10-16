# Decode latency: hardware floors and n-gram locality

2026-10-15, `cuppaxanax/latency`, base `origin/main a63a166`.

## Measured floors (early shared result)

Measured on **192.168.42.53, 24 enabled CUs, RADV GFX1013**, existing stock-safe GFX clock fixed at 2000 MHz. This is not a 40-CU blade measurement. No production-rank load. Governor config was copied, not edited; original service is restored and active. Endpoint clock captures both show 2000 MHz. Each row has 32 warmups and 201 samples, pre-recorded command buffer, one queue submit and fence wait per sample, 10 ns timestamp period. GPU intervals include two outer timestamps; subtract the 0.36 us empty interval only where stated.

| Quantity | GPU p50 us | GPU p99 us | Interpretation |
|---|---:|---:|---|
| No dispatch, two outer timestamps | 0.36 | 0.36 | Measurement envelope, not kernel work |
| One empty dispatch, no barrier | 0.44 | 0.48 | Empty shader; does not imply a useful dependent kernel costs 0.08 us |
| One empty dispatch + full compute memory barrier | 1.00 | 1.12 | Lightweight dispatch plus synchronization |
| 256 empty dispatches, no barriers | 2.48 | 2.52 | Amortized front-end floor 0.0083 us/dispatch after envelope subtraction |
| 256 empty dispatches, full compute barriers | 74.60 | 74.80 | Amortized 0.290 us/dispatch; about 0.282 us additional versus no barriers |
| 256 barriers, no dispatches | 0.60 | 0.64 | Redundant barriers collapse; do not infer a 0.001 us useful-work barrier |
| 256 extra timestamp pairs, no dispatches | 33.88 | 33.92 | About 0.131 us/pair over envelope |
| One dependent read/add/write kernel + barrier | 1.68 | 1.80 | Real global-memory dependency |
| 8 dependent kernels + barriers | 9.60 | 9.80 | One launch chain, exact final counter verified |
| 64 dependent kernels + barriers | 64.12 | 64.48 | Same |
| 256 dependent kernels + barriers | 239.44 | 239.64 | About 0.934 us/dependent step, not 18 us |
| 64-thread sum, LDS tree + barriers | 1.60 | 1.72 | Sum is exactly 2016 |
| 64-thread sum, subgroupAdd + cross-subgroup combine | 1.44 | 1.52 | Sum is exactly 2016, 1.40 us minimum |

`vkQueueSubmit` + fence round trip for an otherwise empty recorded buffer: **29.5-34.0 us p50, 35.4-41.3 us p99** across the five empty controls. This includes the 0.36 us timestamp envelope; it is not a per-dispatch cost. At 1/24/128/256/1024 workgroups, subgroup-reduction GPU p50 is **1.44/1.44/1.52/1.56/1.96 us**, respectively. The analogous LDS-tree workgroups are in the raw ledger.

### Small reads are not steady-state 450 GB/s streams

| Working-set bytes | Parallel read/reduce, 256 WGs, us p50 | Single-WG read/reduce, us p50 | 1024 dependent random line loads, us p50 |
|---|---:|---:|---:|
| 4,096 | 2.12 | 6.88 | 91.00 |
| 40,960 | 2.20 | 50.40 | 160.52 |
| 358,400 | 3.84 | 425.24 | 275.64 |
| 1,048,576 | 7.04 | 1,242.68 | 287.40 |
| 8,388,608 | 42.44 | 9,926.20 | 293.00 |

The dependent load experiment follows a shuffled full-cycle permutation with one word per 64-byte line; 1024 loads per dispatch, same addresses warmed between samples. It is a cache/latency diagnostic, not an independent DRAM latency counter. The small parallel read includes reduction and dispatch overhead. A one-workgroup implementation can miss the bulk-bandwidth roof by orders of magnitude without any slow link or CPU launch in the kernel interval.

Evidence: `latency-gpu-fixed.jsonl` and `latency-fixed.tar` (clock/config restoration included). The earlier dynamic-clock `.52` sweep in `latency-gpu.jsonl` is diagnostic only and predates the shared-blade warning. Final timing-sensitive GPU measurements are on `.53`.

## Remaining evidence

Per-dispatch classification, network accounting, n-gram comparisons and implementation qualification follow in the next commit. The floors above are measured now and can be used independently; they are not an end-to-end decode speedup claim.
