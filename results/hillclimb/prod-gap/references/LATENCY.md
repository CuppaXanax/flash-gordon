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

## Per-dispatch classification

`latency-dispatches.csv` classifies **all 404 recordings** in the fusion ledger: 214 baseline and 190 mask-7 recordings, including geometry, push constants, original byte estimates and diagnostic scope means. Baseline layers 6-11 are four GDN and two QSA, not a representative block to multiply by eight. Input ledger SHA256: `6af1a7382ef44c83119ccebd326b042e73643ef6419ee4c8a6c2dc8e025fbdf3`, fusion branch commit `f207404`.

Operational labels, deliberately explicit and reproducible:

* **Bandwidth**: byte ideal >=3 us and diagnostic time <=1.6x ideal. This is >=281 GB/s logical payload, not a DRAM counter, and not proof of 450 GB/s sustained.
* **Floor**: byte ideal <3 us and diagnostic time <=2.52 us (1.5x the measured 1.68 us single dependent-dispatch envelope).
* **Excess**: above both envelopes. This is an investigation priority, **not a claim that all time above bytes/450 is removable**. Extra reads, reductions, occupancy, dequantization and serialization must be accounted for.

| Baseline six-layer category | Dispatches | Diagnostic us | Unique/reference-byte ideal us |
|---|---:|---:|---:|
| Bandwidth | 28 | 1,068.52 | 735.46 |
| Floor | 76 | 96.56 | 12.20 |
| Excess | 110 | 1,271.96 | 408.01 |
| Total | 214 | 2,437.04 | 1,155.67 |

Diagnostic means sometimes pool different invocations/shapes of a scoped shader. The same 17.12 us mean is attached to both 512-row and 128-row BF16 index projections; the same 11.41 us mean is attached to three QSA quantizations. Do not turn those entries into precise individual shader timings. The summed diagnostic 2,437 us also differs from the independently measured 2,583 us replay layer sum. Neither is the production service ledger.

Type-weighted to 36 GDN + 12 QSA, these diagnostic recordings total **19.160 ms**, with 600 floor-class, 852 excess-class and 228 bandwidth-class dispatches. Reference-byte time is **9.313 ms**; the heuristic sum of `max(byte ideal, 0.934 us dependent step)` is **9.987 ms**. This envelope is a target reference, not an achievable implementation or a proof that 9.17 ms can be deleted from the real ring. The fusion benchmark is synthetic Q4, short context and a different execution/weight operating point from the 29.971 ms production rank-service sum.

### Top implementation targets

Targets below are bounded engineering hypotheses, not measured wins. Savings use model type counts only where that operation occurs once per relevant layer.

| Item | Diagnostic vs byte ideal, us | Why not inherent launch cost | Candidate target / possible saving |
|---|---|---|---|
| QSA split attention | 46.72 vs 0.785 | Grid is only **2x2 workgroups** of 768 threads. On 24 CUs only four CUs can run a group; ~28 KiB LDS/group, per-tile barriers and online softmax matter. The byte ideal omits reload/decode arithmetic. | Sweep split count/geometry; 16-24 us would save **0.27-0.37 ms/token** over 12 layers. Not yet demonstrated. |
| GDN recurrence | 28.80 vs 14.18 | 48 workgroups, multiple LDS reduction trees, long dependent FMA chains; source rereads state before writing it. The one-read/one-write byte model omits a possible extra state read. | 20-24 us is a more defensible target than 1 us; **0.17-0.32 ms/token** over 36 layers. |
| Router top-10 | 9.23 vs 0.005 | One workgroup performs ten dependent selections, not one standalone 64-thread reduction. | 3-5 us is a research target, **0.20-0.30 ms/token** over 48 layers. Fusion's actually measured router+schedule saving is only ~1.66 us/layer, or **0.080 ms/token**. |
| HC inject/mix | 6.73 / 4.57 vs 0.456 / 0.206 | Small grids and reductions dominate. The measured useful-kernel floor is ~1-2 us, not the 0.01 us empty front-end slope. | A 3 / 2.5 us pair, used twice per layer, suggests **~0.56 ms/token**; fusion with adjacent HC work needs exact rounding tests. |
| HC compact IQ4 up/down | 10.48 / 11.41 vs 4.10 / 4.21 | Low-K reduction/row geometry and unpacking; logical bandwidth ~175-166 GB/s. | 6-8 us each would save **~0.57-0.95 ms/token** across two HC up/down pairs per layer. |
| GDN gate / QSA query GEMVs | 38.59 / 70.22 vs 20.21 / 40.41 | Weight traffic is real; 1.7-1.9x the roof still leaves geometry/dequant/reduction work, not dozens of us of driver launch. | Target 1.4x byte ideal, conditional **~0.53 ms/token** over 36 GDN + 12 QSA layers for these two projections. |
| Small QSA BF16/quantization scopes | Pooled 17.12 / 11.41 | Scope pooling prevents a reliable per-shape excess estimate. | First collect isolated replay-faithful timestamps; no numeric saving credited. |

The proposed savings are not independently additive: several target the same memory traffic and fusion boundaries. The existing fusion experiment removed 24 dispatches but saved only 16.42 us/block, consistent with the measured floors. The claim that all 1,675 dispatches cost an inherent ~18 us launch is false.

## Network and the 5.344 ms residual

Two kinds of evidence are kept separate. `latency-network.jsonl` is a Python TCP echo diagnostic. **`latency-fabric.jsonl` uses the real `fg_fabric_send`/`fg_fabric_recv` code, io_uring, protocol-6 framing, CRC validation, registered sockets and TCP_NODELAY**, with eight lightweight mesh endpoints on only .52/.53. Only rank 0 (.52) and rank 1 (.53) exchange the timed payload; other endpoints wait for a completion message. `pgrep -af` showed no competing benchmark/model jobs before this run. No production rank was used.

The same endpoint times send-to-return using one monotonic clock, then divides RTT by two; this is the requested RTT/2 alternative to cross-host clock-offset correction. It assumes symmetric legs and includes endpoint encode/validation, scheduling and echo work. It is not a direct synchronized-clock one-way measurement. 100 warmups, 1,000 measured round trips per size; all returned bytes and frame IDs/sequences match.

| Payload | Actual framed bytes | RTT/2 p50 us | RTT/2 p99 us |
|---|---:|---:|---:|
| Small control proxy, 64 B | 96 | 32.310 | 35.887 |
| bf16-sized hyper, 20 KiB | 20,512 | 274.884 | 291.947 |
| f32 hyper, 40 KiB | 40,992 | 441.694 | 475.457 |
| Ordinary ring work payload, 40 KiB + 24 B | 41,016 | 472.635 | 500.589 |
| Initial work with n-gram, 50 KiB + 24 B | 51,256 | 537.055 | 560.077 |

The 24-byte difference between adjacent tests does not cause 31 us of serialization: these are separate ordered runs and include endpoint/packetization/scheduling variation. Use ~**0.44-0.47 ms/hop**, not nanosecond precision, for the f32 design estimate. Python TCP gave 445.6 us at 40 KiB and 259.1 us at 20 KiB, a useful independent cross-check. No socket/sysctl/NIC tuning was installed. The bf16-sized test measures payload size only: no lossy codec is enabled or quality-qualified.

The production chain has **eight work transfers**, not 12: coordinator -> rank 1 -> rank 0 -> ranks 2,3,4,5,6,7. One initial 50 KiB payload plus seven ordinary 40 KiB payloads. Pure 1 Gb/s payload serialization for that set is **2.703 ms/token**, before Ethernet/TCP/frame overhead. Actual real-fabric estimates are **3.63-3.85 ms/token** from the measured p50 hop range. Around 0.25 ms of work-send enqueue time is already inside coordinator preparation/rank service, so the residual's net work-hop contribution is approximately **3.38-3.60 ms**, not another eight full blocking send intervals.

| Original-normal residual allocation | ms/token | Evidence strength |
|---|---:|---|
| Work-hop transport not already charged as send enqueue | 3.38-3.60 | Transfer from isolated real-fabric RTT/2; not new production-hop timestamps |
| Four concurrent head shards | ~0.785 | Historical refined-normal capture, max shard wall time, not four times this value |
| Remaining head fanout, partial/result delivery, greedy combine, polling and unassigned variation | **0.96-1.18** | Difference needed to close the original **5.344** ms residual |

That final line is deliberately not relabeled "network." Rank 7's HC (~0.157 ms) and its four hidden-slice sends are **already in rank 7 service**. It sends four 10,248-byte hidden slices to ranks 4/0/1/2; their link drain can overlap the first shard's work. Last-slice head start, three partial returns, the final result, waits and combine remain. Greedy combine is three CPU finalist comparisons, not a full-vocabulary CPU sampler. Historical tokenizer/render/callback cost is ~0.0022 ms/token. The supplied traces do not isolate a standalone fanout-drain or combine duration, and new .42-.49 instrumentation is forbidden during qualification, so **0.96-1.18 ms remains jointly bounded, not falsely measured as individual components**.

Using the matched refined capture instead gives 5.419 ms residual and an extra ~0.075 ms unassigned. Historical detailed GPU profiling is intrusive and must not be spliced into either normal wall ledger.

Halving each ordinary hyper payload saves about **0.17-0.20 ms/hop**, or **1.3-1.6 ms/token** for eight work hops if initial n-gram handling is also adapted. The physical payload-byte saving is ~1.31 ms/token. It cannot remove the whole 5.3 ms bucket; it needs a separate numerical/KL gate and is not implemented here.

## N-gram: premise, storage and options

**Important correction:** `resident_ngram_lookup` really does scatter/gather to ranks 1-7, but its caller is **legacy expert-parallel decode** (`coordinator_decode_token_local`). The default ring path (`coordinator_decode_token_ring`) directly calls `fg_ngram_store_lookup_prefill` on the coordinator's local `ngram.iq4nl`, as does prefill. This is true in base `a63a166` and the supplied decode-profile worktree. Historical ring traces report the n-gram interval, not `NGRAM_RESIDENT_TRACE`. The measured ~1.9 ms ring cost is therefore **local storage/assembly/dequantization**, not seven TCP round trips. The coordinator already opens the full table at startup.

Strata also has more than one path: `src/kernels/ngram.cpp` provides a mapped/page-prefetched table, whereas `src/ngram/ple_reader.cpp` provides threaded asynchronous **DirectFile** I/O plus a bounded row cache. It is not accurate to describe every Strata configuration as page-cache reads. The transferable ideas are batching misses, local storage, row/page caching, and issuing work before consumption.

### Capacity and storage measurements

Read-only `df`/`lsblk` on .42-.49 used a fleet-runner adapter that sends those commands directly, **without staging a remote script or writing files**. Coordinator .42 is a **1 TB TEAM TM8FP6001T NVMe**, with **119.67 GB free**. Other production blades are 256 GB Patriot P300 NVMe; free space .43/.44/.45/.46/.47/.48/.49 is **42.04/52.18/12.76/64.29/44.97/47.66/8.46 GB**. No SSD test ran on these blades. Full-table placement is feasible on .42, but not on .45/.49 without removing something; no removal is authorized by this experiment.

.52/.53 have the same 256 GB P300 model, with 52.46/61.49 GB initially free and ~15.94 GB physical RAM. Their idle RAM availability is not the coordinator's serving page-cache budget: production holds ~12 GB of GPU allocations in shared memory. Do not assume a 28.8 GB table can be made fully memory-resident on a blade.

For latency, a **fully written, fsynced, non-sparse synthetic file of exactly 28,800,139,264 bytes**, the real 320,001,536 x 90-byte row layout padded to 4 KiB, was created only under `/home/xander/hc-lat`. No model table was read from .42-.49. This is a P300 storage proxy, not model quality evidence or a measurement of .42's different NVMe. Cold random pages are evicted with per-range `POSIX_FADV_DONTNEED`, never global `drop_caches`. Reads span 8 KiB so a row straddling a 4 KiB boundary is covered. Sixteen random head rows per token, 32 warmups, 1,000 token batches; timed intervals exclude file setup and eviction hints.

The controlled hot fractions below use a reused per-head hot page versus new random rows. They are a **sensitivity sweep**, not measured production hit rates. Hash-based n-grams are workload-dependent; a repeated token alone is insufficient because the previous two tokens enter the hash. The 8 MiB application cache already catches exact repeated pages, so changing O_DIRECT to page-cache I/O need not help a warm application hit.

| .52 storage-only case | 0% hot p50/p99 us | 50% hot p50/p99 us | 75% hot p50/p99 us | 100% hot p50/p99 us |
|---|---|---|---|---|
| Buffered, serial pread | 886/1384 | 453/954 | 227/531 | 17/29 |
| Buffered, 16 persistent workers | 253/385 | 184/303 | 140/258 | 47/58 |
| O_DIRECT, serial pread | 1544/1583 | 1536/1581 | 1520/1563 | 1485/1507 |
| O_DIRECT, 16 persistent workers | 372/545 | 363/498 | 367/483 | 375/397 |

The 16-worker row test includes condition-variable wake/completion latency per batch (threads persist). It proves that page miss concurrency matters; it is not an io_uring speedup claim. O_DIRECT intentionally bypasses the OS cache even for the "hot" address stream. A second identical storage sweep on .53 is retained as `latency-storage53.jsonl`: cold buffered serial/parallel p50 is 1,553/455 us, direct serial/parallel 1,863/542 us. This cross-blade/run variation is material; do not promote .52's best SSD number to a universal disk floor. Both filesystems are XFS, without transparent filesystem compression.

### Actual lookup and implementation

The serving-path benchmark uses real `fg_q38_ngram_lookup` history hashes, the same bounded io_uring planner, 8 MiB application cache, packed assembly, IQ4_NL dequantization and `fg_vk_tensor_read` of all 10,240 embedding bytes. Random valid three-token histories are mixed with a repeated hot history. Each candidate output is `memcmp`-checked against its corresponding default-path output, including warmups. This is byte exactness of the synthetic file's embedding, not a claim that synthetic weights establish model quality.

The initial page-cache-only ablation on .53 was neutral:

| Repeated-history fraction | O_DIRECT p50/p99 us | Page-cache-only p50/p99 us |
|---|---|---|
| 0% | 815.8/1053.3 | 803.4/1039.4 |
| 50% | 335.3/999.1 | 337.4/956.4 |
| 90% | 332.3/922.0 | 332.5/893.2 |

Both modes still spent ~0.33 ms on a no-I/O warm application hit. The host dequantizer read individual packed bytes directly from the device-local HOST_VISIBLE mapping, despite `fg_vk_tensor_read` already having a streaming-copy path for slow uncached UMA reads. The selected opt-in path first copies 1,440 packed bytes into ordinary CPU stack memory, runs **the unchanged `fg_dequantize_iq4_nl` function**, then bulk-copies 10,240 result bytes back. Scratch is 11,680 bytes and only used for exactly one token; multi-token prefill keeps the existing dequantization path.

`FG_NGRAM_LOCAL_CACHE=1` selects buffered local I/O plus this cacheable CPU staging. It also selects local-table lookup in legacy expert-parallel decode, bypassing the worker scatter/gather there. **Unset/off remains unchanged**, including O_DIRECT and the old dequantizer. The flag is documented and registered in the flag gate; it must be set before startup.

Final .53 full-materialized-file run:

| Repeated-history fraction | Default p50/p99 us | Buffered + cacheable CPU staging p50/p99 us |
|---|---|---|
| 0% | 820.6/1039.5 | **447.1/688.3** |
| 50% | 336.9/1003.2 | **7.0/608.3** |
| 90% | 332.0/930.7 | **6.2/540.1** |
| 100% | 332.0/337.6 | **6.2/6.5** |

This is an ordered storage/assembly experiment, not a fleet qualification or a claim that the real coordinator's 1.9 ms becomes 0.45 ms. The cold result includes SSD/device variation. The large warm change was then isolated in **three A/B/A rounds**, 1,000 samples/phase and 32 warmups/phase. Default medians before/after are 332.413/332.009, 332.136/332.004 and 332.073/331.999 us; candidate medians are 6.151, 6.151 and 6.144 us. Paired p50 savings are **326.060, 325.919 and 325.892 us** (mean **0.325957 ms**). Candidate p99 is 6.75-8.63 us. All return embedding bytes match.

That warm control uses a separately labeled **sparse correctness fixture** containing the sixteen nonzero IQ4_NL rows for one repeated three-token history. No SSD latency conclusion uses it: measured samples are application-cache hits. It isolates uncached-UMA host dequantization from page misses and OS cache policy. The earlier full-table latency tests remain non-sparse, fully materialized files.

The existing `ngram_direct_lookup` oracle now checks default and candidate with store capacities 1 and 3, including borrowed tensor views, repeated application-cache hits, exact packed rows, and `memcmp` of every embedding byte against CPU reference. It passes on RADV; the existing `iq4_nl_dequant` GPU/CPU oracle also passes. Serving binary, both production-path benchmarks, shaders and the flag gate build/pass. No serving flag is enabled or production binary replaced. The new flag remains **off by default** because the actual .42 SSD, full model allocation pressure, long prefill and full-ring p99 are not qualified.

### Options not selected

1. **Overlap before consumption:** PLE consumes the embedding at **layer 1, on rank 1** in the current six-layer-block deployment. Layer 0 can run first, so a local asynchronous row lookup can overlap roughly one GDN layer (~0.4 ms in the synthetic replay), not all six layers or all 30 ms of rank service. Embedding-only work on coordinator is ~0.017 ms, too short to hide a cold lookup. Current work frames require the n-gram embedding before the six-layer call. Real overlap needs a separate row-result rendezvous at layer 1, or split layer-0 execution, with reset/cancel/session safety. It is promising for 0.2-0.4 ms concurrent SSD reads but is not a drop-in thread around the existing Vulkan store: that store owns shared packed/output tensors.
2. **Top-k speculative prefetch during the head:** for greedy four-way head splitting, the true next token must be one of the four local shard winners. Prefetching those four candidates means up to **64 rows, ~256 KiB of page reads**, not 16. With k candidates per shard, it is 64k rows and ~256k KiB, before page deduplication. Existing partial messages expose one local winner, not an early global top-k; those winners become available near the *end* of the ~0.78 ms head, leaving only shard skew/control-return time to overlap. Rolling/chunked early-head candidates are speculative and can miss. A hint-only cache prefetch can remain exact if final selection always performs the normal lookup, but it can amplify SSD/page-cache pressure. No measured hit-rate/early-head lead-time evidence here justifies enabling it.
3. **Owners send rows directly to rank 1:** useful only for the legacy resident-shard path or a redesigned ring path. Request rows with destination=consumer, let rank 1 gather/dequantize them while layer 0 runs, then rendezvous at layer 1. It removes coordinator result assembly plus the n-gram's 10 KiB detour (~0.082 ms raw wire plus a control hop). It still waits for the slowest shard SSD miss and needs new routing/session/cancel handling. Current ring decode does **not** incur that distributed detour, so this cannot explain or remove its 1.9 ms interval.

No speculative token changes, lower-than-Q4 weights, bf16 activation deployment, or unverified worker-shard format changes are made.

## Evidence-based decode ceiling: inherent versus fixable

**37.4 ms/token is not an inherent GPU launch floor. It is also not 14.8 ms of weights plus 22.6 ms of removable CPU overhead.** The supplied production-normal ledger is 2.122 ms preparation + 29.971 ms serial rank services + 5.344 ms transport/head/unassigned = 37.437 ms. Intrusive GPU profiling changes execution and raises it to 50.450 ms; it cannot be used to estimate normal submission overhead by subtraction.

There are three different levels of claim:

| Operating point | Byte/link reference ms/token | Meaning |
|---|---:|---|
| Preserve historical ~6.671 GB/token pack at 4K; four-way head | **~16.7-17+** | Correct the aggregate bytes for the three head shards that run concurrently: `(6.671 GB - 3/4 x 0.67543 GB)/450 GB/s = 13.70 ms`; add eight work hops' 2.703 ms raw serialization and ~0.328 ms head-fanout payload serialization. Ethernet/control costs and real kernel dependencies raise it. |
| Same single-stream eight-hop design with validated Q4 dense weights | **~13-15** | Synthetic layer ledger is 9.313 ms of reference traffic; PLE/exceptional layers/head and launch/reduction work are not all represented, then add ~3 ms of mandatory f32 ring/head wire. A conditional design target, not a measured pack-backed decode. |
| Measured replay-faithful synthetic Q4 kernel operating point | **~20.15 ms layer GPU alone** | Before ring, head, coordinator, PLE and exceptional formats. It does not establish a 13 ms working engine. |

The historical 6.671 GB budget is at 4K and predates the short, token-60..123 normal-profile window and latest deployment manifest. It is used here as an explicitly conditional reference, **not an exact byte ledger for that window**. Likewise 450 GB/s is a demonstrated bulk logical read roof, not a guarantee every kernel reaches it or a hardware counter proving physical DRAM traffic.

Against the conditional unchanged-pack ~16.7 ms byte/link reference, about **20.7 ms of the observed 37.4 ms lies above that reference**. That is the investigation budget, not 20.7 ms of proven removable time. It contains necessary unpack/reduction/softmax work and topology/driver overhead as well as fixable geometry, extra memory passes, scalar uncached host loads and redundant synchronization.

What is demonstrably fixable here is **~0.326 ms of warm n-gram host work**; simple substitution into the old ledger would be 37.437 -> 37.111 ms (~26.95 TPS), **a projection only**, not an observed ring result. Page cache by itself did not supply a material warm win. Existing qualified synthetic fusion adds only ~0.130 ms/model-token of conditional saving, not 30%. The ranked kernel hypotheses above offer another few milliseconds if each can be made exact and replay-fast; their full savings cannot be added blindly.

Eight-hop f32 serialization is inherent to the current 1 GbE payload/topology. Approximately 0.3-1 ms of endpoint/transport excess might be addressable by batching/protocol/scheduling, but deleting link serialization is impossible without smaller payloads or fewer crossings. bf16-sized payloads demonstrate a separate conditional ~1.3-1.6 ms opportunity, not an exact enabled feature.

Thus **60 TPS is not excluded by hardware after Q4 and substantial kernel work**, but it is not reachable merely by moving the existing ring n-gram lookup to disk: it is already on disk. **100 TPS (10 ms/token) is below these f32 eight-hop byte/link references**, before real kernel inefficiency, and is not supported for this pre-MTP design. A usable upper limit is conditional, not a promised achieved throughput: unchanged-pack byte/link-only ~58-60 TPS; Q4-design reference ~67-77 TPS. Measured implementations remain materially slower. MTP acceptance and multi-session pipeline throughput are separate metrics.

## Safety, persistence and reproduction

* Only .52/.53 received benchmarks. Final timing-sensitive GPU floors are on .53; .52 was checked with `pgrep -af` before the later actual-fabric timed run. .42-.49 received only direct read-only `df` and `lsblk`, no staged files or load. .50/.51 were not accessed.
* The existing `.53 /home/xander/hc-q4` artifacts were never read as table sources, overwritten, rebuilt or deleted.
* All remote experiment files, compiler scratch, shader cache, tables and binaries live under `/home/xander/hc-lat`. Materialized and warm-fixture tables are deleted after their tests. Cleanup verifies the original governor config/service, closed experiment firewall ports, no remaining benchmark jobs and removal of `hc-lat`. Production serving was never changed.
* Evidence JSONL files retain every measured configuration and quantiles. `latency-fixed.tar` retains fixed-clock config/start/end/restoration evidence. `latency-captures.tar.gz` contains final validation/build checks and storage/warm captures. No full-model quality or full-ring A/B qualification is claimed.
* `latency-production-disks.json` preserves the final read-only `df`/`lsblk` inventory. A subsequent **non-staging** check in `latency-cleanup-verification.json` confirms both experiment directories absent, stock governor active, no experiment firewall ports left, and .53's `hc-q4` directory present.

Local build and deterministic ledger replay:

```powershell
wsl -d Ubuntu-24.04 --cd /mnt/d/looking-glass-labs/fg-wt/latency --exec bash -c 'mkdir -p .build; export TMPDIR=$PWD/.build; make -s -j8 flash-gordon tests/test_fg_vk tools/bench-ngram-latency tools/bench-fabric-latency; bash tools/check-flags.sh'
python tools\latency-ledger.py D:\looking-glass-labs\fg-wt\fusion\results\hillclimb\fusion-dispatches.csv --output results\hillclimb\latency-dispatches.csv --summary results\hillclimb\latency-dispatch-summary.json
```

For a new **exclusive** .53 lease, archive the source tree into project-local `.build`, excluding objects/dependencies/SPIR-V, upload as `full-source.tar` using `tools\Invoke-LatencyFleet.ps1`, and run `tools\latency-build.sh`. The fixed-floor source tar must contain `tools/latency-vk.c`, `tools/latency.comp`, `tools/latency-storage.c` and `tools/latency-run.sh`; upload as `source.tar` then run `latency-fixed.sh -Sudo`. The adapter invokes the canonical fleet runner and relocates its remote staging under `hc-lat`; always use **absolute** upload/download paths.

`latency-store-run.sh` writes a full 28.8 GB synthetic table and needs that much free space; do not run it as a cheap status check. The standalone C storage tool deletes its file by default; its `keep` argument is only for the chained serving-path benchmark, whose shell trap deletes it. `latency-warm.sh` is a separately labeled sparse-fixture, application-cache-only A/B/A test. The actual-fabric driver needs both .52 and .53 and temporarily opens 19600-19615/tcp, then removes those rules. Run `latency-cleanup.sh -Sudo` only after retrieving evidence. These scripts are experiment tools, not a production launch recipe.
