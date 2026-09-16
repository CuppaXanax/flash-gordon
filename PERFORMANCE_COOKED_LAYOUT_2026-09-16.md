# Cooked layout micro-benchmark (2026-09-16): row-interleaved quant placement

Worktree `D:\looking-glass-labs\fg-work-layout`, branch `perf/cooked-layout` off
`main` `3f66f1d`.  This round builds the micro-benchmark that
`PERFORMANCE_DENSE_KERNEL_OCCUPANCY_2026-09-15.md` section 4 parked, runs it on
a BC-250 blade, and applies the pre-registered promotion rule before any pack
work.  Measured on one BC-250 (RADV GFX1013, 10 ns timestamp period) on
2026-09-16 with the ring quiesced; no pack was rebuilt and the live serving tree
was not modified.

| commit | change |
|---|---|
| `e75985f` | stride-parameterized cooked-layout bench shader |
| `a905020` | cooked-layout grid/stride table with bit-exact layout parity |
| `3456c9c` | 8-row group interleave candidate, parity for all physical layouts |
| docs update | measured table, decision (this file) |

## 0. TL;DR

**NO-GO.**  On the target shape (GDN output 6144 -> 2560, 192 blocks, 320
workgroups) the best layout measures +9.6% .. +11.7% over the current layout
across four independent measurements (mean +10.6%), against the pre-registered
bar of >= +15% (target +20%).  The round-12 design (`il_512`: row stride 32 B,
block stride 512 B) measures only +2.9% .. +5.6% and regresses the 2560-stride
controls by 2-6%.  `g8_256` is neutral on the 2560-stride shapes.
The pack round is declined.  The GDN output deficit is closed as a dispatch
ramp property, not a layout property: at steady state the current layout
already streams the target shape at 258.6 GB/s in a pipelined stream (the
production profile's 175 GB/s is per-dispatch cold contact), and the same
layout at 1280 workgroups with the same 6144 B row stride reaches 266.4 GB/s.
The instrument is retained; `g8_256` (8-row group interleave) is the best
measured variant if the +10% steady-state win or the -36% cold-contact win is
ever needed for another reason.

## 1. Instrument

New bench shader `shaders/fg_bench_cooked_layout.comp`: the production r8
instruction stream (same 128-lane workgroup, same 16-block trips, same
scale/input/reduction epilogue) with the quant address generalized to

```
quant(row r, block b) = tile_base + quant_offset
                      + (r&7)*row_stride + (r>>3)*group_stride
                      + b*block_stride + lane*4
```

Three physical placements are measured, plus the real production
`fg_dense_q8_0_cooked_r8.spv` as an anchor:

| layout | row_stride | block_stride | group_stride | address |
|---|---:|---:|---:|---|
| `prod` (current) | `blocks*32` | 32 | `blocks*256` | `r*blocks*32 + b*32` |
| `il_512` (round 12) | 32 | 512 | 256 | `r*32 + b*512` |
| `g8_256` | 32 | 256 | `blocks*256` | `(r&7)*32 + b*256 + (r/8)*blocks*256` |

`g8_256` was added because the r8 kernel consumes 8 rows per workgroup: each
workgroup's own trip touches one contiguous 4 KiB region and each wave
instruction scatters over 4 KiB instead of `il_512`'s 8 KiB, while a 16-row
reader still sees a regular placement.  The old "probe" strides (256/1024 with
row stride 32) that alias addresses were dropped after review; only physical
placements are benchmarked.

`fg_vk_bench_cooked_layout` (`src/vk.c`) runs 8 shapes x 4 configs x 2 passes x
10 no-barrier dispatches with GPU timestamps, and reports average microseconds,
achieved GB/s (cooked matrix bytes / time) and the first dispatch of each set.
Shape set is a stride x grid probe: 6144 B row stride at 80 / 320 / 1280
workgroups and 2560 B at 80 / 320 / 1280, plus `shared_down` and `gr_up`.
`flash-gordon bench` runs the section on synthetic buffers (no manifest data).

The parity oracle cooks real Q8_0 data, repacks each 16-row tile into `g8_256`
and `il_512`, dispatches production r8 and all three clone placements, and
compares with `memcmp`.  Both shapes (512 -> 32 with 16 blocks; 320 -> 48 with
10 blocks, i.e. cohort masking) are **bit-identical** on every layout, so the
interleave is a pure placement change with an exact addressing model.

## 2. Measured table (4 samples per cell: 2 passes x 2 invocations)

GB/s is cooked matrix bytes over GPU-timestamped kernel time; ratios are
`prod/config` time ratios, i.e. `>1` means the layout is faster (range over the
four sample pairs; all cells are reproducible within the range shown).

| shape | in -> out | blocks | grid | wt MB | prod GB/s | g8_256 GB/s | g8 ratio | il_512 GB/s | il ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate_2560 | 2560 -> 640 | 80 | 80 | 1.74 | 285.8 | 291.0 | 1.003-1.039 | 272.4 | 0.939-0.977 |
| ple_value | 2560 -> 2560 | 80 | 320 | 6.96 | 296.4 | 297.7 | 1.000-1.010 | 287.3 | 0.966-0.974 |
| gdn_qkv | 2560 -> 10240 | 80 | 1280 | 27.85 | 294.1 | 294.8 | 1.002-1.003 | 289.0 | 0.981-0.985 |
| probe6144_640 | 6144 -> 640 | 192 | 80 | 4.18 | 205.3 | 285.1 | 1.376-1.406 | 266.4 | 1.294-1.299 |
| **gdn_output** | 6144 -> 2560 | 192 | **320** | 16.71 | 258.6 | **286.0** | **1.096-1.117** | 268.8 | 1.029-1.056 |
| probe6144_10k | 6144 -> 10240 | 192 | 1280 | 66.85 | 266.4 | 278.5 | 1.044-1.046 | 269.4 | 1.008-1.013 |
| shared_down | 640 -> 2560 | 20 | 320 | 1.74 | 188.3 | 188.1 | 0.994-1.001 | 187.1 | 0.988-0.997 |
| gr_up | 320 -> 10240 | 10 | 1280 | 3.48 | 144.9 | 144.9 | 0.999-1.001 | 144.9 | 0.998-1.001 |

Readings: the interleave pays where the row stride is large (6 KiB: +4% .. +41%
for `g8_256`) and is neutral on 2560 B strides; `il_512` additionally costs
2-6% on 2560 B strides because per-instruction scatter is not recovered by page
locality when the 16-row span is already only ~40 KiB.  `g8_256` dominates
`il_512` on every shape measured.

## 3. Anchor and harness fidelity

`r8` (production SPIR-V) vs `prod` (clone with the production strides) agree
within +-0.7% on the three large 2560-stride shapes and +1.8% on the target
(r8 faster).  The gap is systematic +5.8-5.9% on `shared_down` and `gr_up`, whose
kernels run 1-2 trips and pay the clone's three uniform stride multiplies.
Clone-based ratios are therefore at most ~1.8% conservative on the target; the
gate verdict does not depend on it.

First dispatch of each set (4-sample mean, us) - the cold-contact diagnostic:

| shape | prod | g8_256 | il_512 |
|---|---:|---:|---:|
| gdn_output | 109.8 | 69.7 | 95.1 |
| probe6144_640 | 105.8 | 56.9 | 59.4 |
| probe6144_10k | 291.3 | 249.2 | 280.7 |
| gdn_qkv | 100.2 | 99.7 | 109.0 |

The interleave cuts cold contact on the disjoint-row shapes by 36% (`g8_256`,
target) while steady state moves 10%; the 2560-stride shapes are unchanged.
This column is diagnostic only; the promotion rule was deliberately written on
steady-state ratios because the cold cost cannot be priced from this instrument.

## 4. Ramp check (pre-registered rule 3)

`probe6144_10k` (same 192 blocks and 6144 B row stride, 4x the grid) runs at
266.4 GB/s in the current layout versus 258.6 GB/s for the 320-workgroup target,
while the production profile prices the target's real calls at 175 GB/s.  The
320-workgroup grid is therefore not grid-starved at steady state, and the
production 175 GB/s is the per-dispatch ramp/serialization cost (the bench's
cold dispatch, 106-112 us for 16.71 MB, is close to the profile's ~95 us per
call).  Rule 3 is satisfied: grid/ramp dominates the production deficit.

## 5. Decision

Pre-registered rule applied: GO required `gdn_output` `prod/config` >= 1.15
(target 1.20) for `il_512` and/or `g8_256`, no other shape regressing more than
5%, parity PASS, and a valid anchor.

* `g8_256`: 1.096-1.117 on the target (best 1.117, mean 1.106) - **below 1.15**;
  no shape regresses (worst -0.6%).
* `il_512`: 1.029-1.056 on the target - **below 1.15**; `gate_2560` regresses
  up to -6.1% (outside the 5% rule).
* Parity: PASS bit-exact.  Anchor: PASS on the decision shape.

**NO-GO for the 16-row row-interleaved pack round.**  The row-interleave
hypothesis is directionally real for `g8_256` but roughly two thirds of the
required size, and the production deficit it was aimed at is a dispatch ramp
problem, per section 4.  The remaining levers for the 320-workgroup shapes are
the round-12 ranked item 2: dispatch fusion or a persistent kernel, i.e.
`src/vk.c`/caller work, not a pack change.

If a future pack round is opened for another reason, `g8_256` is the measured
placement to carry: 8-row groups, block stride 256 B, group stride
`blocks*256` B, one-line writer change plus the reader addressing change.  A
pack-format version bump and all parity oracles would still be required, and
the expected end-to-end gain must be priced with a profile A/B on the rebuilt
pack before fleet deployment, because the steady-state +10% and the cold-contact
-36% cannot be combined from this instrument alone.
