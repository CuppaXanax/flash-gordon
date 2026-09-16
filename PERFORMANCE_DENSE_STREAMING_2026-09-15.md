# Dense streaming round 13 (2026-09-15): GR dispatch diet, PLE overlap, RMS loads

Worktree `D:\looking-glass-labs\fg-work-dense13`, branch `perf/dense-streaming`
off `main` `405c3f9`. Scope: GDN conv/recurrent/algebraic, PLE decode, the GR
read/write chain, the cooked Q8 r8/split dense kernels and the rms / inject /
silu elementwise kernels. Local-only round: no fleet access, no pack rebuild,
no precision trade. Every change is bit-exact against the qualified build or
provably equal arithmetic (FP association, operation order and operands
unchanged); the new oracles compare bitwise.

Commits:

| commit | change |
|---|---|
| `fa7e3be` | GR decode: split-reduce kernel with fused scaled SiLU; HC inject moved ahead of the down projection and overlapped with it |
| `1d7f070` | PLE decode: overlap key/value projections and key/query norms; fold the output add into the convolution |
| `1b9b600` | `tools/fg_dense_rates.py`: achieved-GB/s table from `FG_DECODE_PROFILE` logs + byte model |
| `dd8d1ce` | `fg_group_rms_norm`: register-cached input at width 2560 (redundant-load removal) |

## 0. TL;DR

1. **The GR read chain is dispatch-bound, not byte-bound.** Its elementwise
   kernels (rms 8.5, inject 29-31, reduce 3.7, mix 11.8 GB/s) move 2.6 MB/block
   and cost 40% of the read's scope time. Four of the seven dispatches per read
   cannot be merged without re-associating FP sums; the one that could (the
   scaled SiLU into the split-reduce) is now merged, and the HC inject is
   overlapped with the 640-workgroup down projection.
2. **The two remaining dense rate gaps are structural and documented, not
   fixable inside the kernels this round:** the GDN output shape at 175 GB/s
   needs the row-interleaved cooked layout (a pack round), and the GR up shape
   at 138 GB/s device-visible (plus 9.8 MB/block of L2 re-reads) needs a row
   tile change that re-associates the cohort sums. Occupancy (36 VGPRs, 7
   waves/SIMD), wave split and the uvec2 pair are already closed by rounds 9/12
   and are not re-litigated.
3. Expected fleet effect of this round: **-0.7 to -1.2 ms/token** across the
   GR and PLE chains (0.4-0.9 ms GR + 0.3-0.4 ms RMS + ~0.01 ms PLE), i.e.
   +0.4 to +0.8 TPS at the current ~24 TPS band. The fleet A/B decides; the
   local evidence is bit-exactness plus dynamic load/instruction counts.

## 1. Per-kernel achieved-GB/s table (dense chain)

Rank 0, one 6-layer decode block, `FG_DECODE_PROFILE=1`. The milliseconds are
the round-12 fleet profile scopes (`PERFORMANCE_DENSE_KERNEL_OCCUPANCY_2026-09-15.md`
§7.1, 0.471 ms `gr_attn_read` and 0.462 ms `gr_ffn_read`, 6 calls per read);
the bytes are the manifest-exact model from `PERFORMANCE_BYTE_BUDGET_2026-09-15.md`
priced by `tools/fg_dense_rates.py`. `L2` is the r8 input row re-read traffic
that stays inside L2.

| scope | kernel | calls | ms/block | MB/block | achieved GB/s | L2 MB |
|---|---|---:|---:|---:|---:|---:|
| gdn_projection | r8 qkv + z | 8 | 0.633 | 178.60 | 282 | - |
| gdn_recurrent | gdn_conv_decode | 4 | 0.012 | 2.29 | 191 | - |
| gdn_recurrent | recurrent_algebraic (3 state passes) | 4 | 0.148 | 38.11 | 257 | - |
| gdn_output | r8 6144->2560 | 4 | 0.382 | 66.99 | **175** | - |
| gr_attn_read | group_rms_norm | 6 | 0.087 | 0.74 | **8.5** | - |
| gr_attn_read | split down 10240->320 | 6 | 0.115 | 21.20 | 184 | - |
| gr_attn_read | reduce (+SiLU) | 6 | 0.021 | 0.08 | **3.7** | - |
| gr_attn_read | hc_inject_partial | 6 | 0.042 | 1.23 | **29** | - |
| gr_attn_read | r8 up 320->10240 | 6 | 0.152 | 20.96 | **138** | 9.83 |
| gr_attn_read | gr_mix_partial | 6 | 0.047 | 0.56 | **12** | - |
| gr_ffn_read | (same six kernels) | 6 | 0.455 | 44.76 | 98 | 9.83 |
| shared_expert | r8 gate/up/down | 18 | 0.164 | 31.56 | 192 | - |
| output (rank 4) | r8 head 2560->vocab | 1 | 2.620 | 675.43 | 258 | - |

Reads of the table: the dense GEMMs that have enough workgroups sit at
175-282 GB/s; every kernel between 3.7 and 31 GB/s is a 2-24 workgroup
elementwise dispatch whose bytes are irrelevant. That is the GR chain's 107
GB/s aggregate: 44.8 MB of weights at 184 (down) and 138 (up) GB/s plus 2.6 MB
of activations at single-digit GB/s.

The algebraic recurrence deserves a note: the audit counts it at 179 GB/s
(6.62 MB read+write); the kernel actually streams the 3.15 MB state twice and
rewrites it (9.44 MB, three passes, the first to accumulate `memory` and
`prior_result`, the second to apply `delta`). At 3 passes it is at 257 GB/s
and is not the GDN gap.

Fresh-fleet regeneration (orchestrator step):

```
FG_DECODE_PROFILE=1 <decode run> 2> rankN.profile
python3 tools/fg_dense_rates.py rank*.profile
```

`tools/fg_dense_rates.py` prices each kernel from the audit byte model and
prints one row per (rank, scope, kernel) plus scope totals; QSA rows are
marked unpriced because the audit's QSA numbers are not manifest-exact.

## 2. Changes, accounting, local evidence

### 2.1 `fa7e3be` GR: split-reduce + SiLU fusion, inject overlap

The decode read ran `fg_dense_q8_0_cooked_split` (split stage + a 2-workgroup
reduce) and then `fg_silu_scaled` over the same 320-value low vector. The new
`fg_dense_q8_0_cooked_split_reduce_silu` writes both the raw sum (so the
`FG_NUMERICS A_LOW|SYNC1_LOW` digests stay identical) and the scaled SiLU in
one pass. The HC inject dispatch, previously ordered only by the SiLU's
barrier, now carries its own barrier and is dispatched before the down
projection, whose barrier is elided with `fg_vk_next_dispatch_independent`;
the inject's 24 workgroups hide inside the split's 640.

Per read: one dispatch and one barrier removed, 2.56 KB of round trip removed,
and up to 6 us of inject time overlapped (bounded by the 16 us split). Per
6-layer block: about -0.05 to -0.10 ms on each read scope, i.e. 0.4-0.8
ms/token. For 12 reads/block and 8 blocks/token: -0.03 MB/token and 12 fewer
dispatches and barriers per block per rank.

Local: `q8_split_silu_parity` runs the production shape (10240x320, splits=8)
through the old sequence and the fused call and compares `low` and `active`
with `memcmp` - equal; the existing `gr_mix`, `gr_batch`, `hc_inject_partial`,
`gr_partial_boundaries`, `q8_decode_shape_parity` and the GDN chunked-prefill
parity set pass unchanged.

### 2.2 `1d7f070` PLE: projection and norm overlap, add fusion

The PLE decode chain serialized seven dispatches although `ple_key`/`ple_value`
read the same ngram embedding and the key/query norms read disjoint inputs.
Two `fg_vk_next_dispatch_independent` marks overlap those pairs. The trailing
`fg_vk_add_f32(hyper, conv_output)` becomes `fg_ple_conv_decode_add`, which
writes `hyper + (gated + silu(conv))` directly into the destination: one
dispatch removed and a 40 KB read plus 40 KB write removed per token. Rank 1
saves about 3-8 us/token of critical-path time.

Local: `ple_conv_add_parity` compares the fused output and the shifted state
against the old two-dispatch sequence with `memcmp` - equal; `ple_decode`,
`ple_prefill_scan`, `ple_prefill_t1_compat` pass.

### 2.3 `dd8d1ce` RMS: register-cached input

`fg_group_rms_norm` read `x` in the sum loop and again in the write loop. At
the production width (2560) the ten lane steps are now unrolled into registers
before the reduction and reused by the write; every other width keeps the
original loops. The accumulation order (`i = tid + 256k`, `fma(v,v,ss)`) and
the write expression (`(v*scale)*w`) are unchanged, so the result is exact.

Local: the `group_norm` oracle reports the same 64-bit output hash
`22ed354d9f59f1ab` for the old and the new SPIR-V (different binaries,
sha256 `ba8f2666…` old vs `6118c213…` new), i.e. bit-identical at the
production shape. Dynamic global loads per lane fall 30 -> 20; the rms scope
is 0.173 ms/block, so the expected cut is 0.035-0.05 ms/block (0.3-0.4
ms/token) if the kernel is issue-share limited.

### 2.4 `1b9b600` rate table tool

`tools/fg_dense_rates.py` plus `tools/test_fg_dense_rates.py` (make
`test-fg-dense-rates`). See section 1.

### 2.5 Expected before/after (rank-0 block, round-12 bytes)

| change | scope / kernel | before | after (expected) | expected ms/token |
|---|---|---|---:|---:|
| `fa7e3be` | `gr_attn_read` scope | 96.5 GB/s | 117-124 GB/s | -0.2 to -0.4 |
| `fa7e3be` | `gr_ffn_read` scope | 98.4 GB/s | 119-127 GB/s | -0.2 to -0.4 |
| `fa7e3be` | `silu_scaled` (removed) | 2.6 GB/s | - | - |
| `1d7f070` | `ple` chain (rank 1) | n/a | n/a | -0.003 to -0.010 |
| `dd8d1ce` | `group_rms_norm` | 8.5 GB/s | 11-13 GB/s | -0.3 to -0.4 |
| total | | | | **-0.7 to -1.2** |

The after numbers assume the same bytes over the shortened scope time; the
scope rates move even where a kernel's own timestamp barely changes, because
the overlap removes serialized wall time from the read. The combined estimate
is 1.7-2.9% of the 40.8 ms token, i.e. +0.4 to +0.8 TPS at the 24 TPS band,
before the fleet's thermal spread.

## 3. What is deliberately not touched, and why

* **GDN output 6144->2560 at 175 GB/s (0.382 ms/block, 3.1 ms/token).** The
  kernel reads every byte once and is not instruction-bound (qkv, same
  kernel, 282 GB/s at 4x the workgroups); the only remaining lever is the
  row-interleaved cooked layout, which changes the pack and needs a pack
  rebuild round. Occupancy (36 VGPRs), row-4, K-split, the wave split and the
  uvec2 pair are all measured and closed (rounds 11/12).
* **GR up 320->10240 at 138 GB/s device (0.152 ms/block x2 scopes).** 1280
  workgroups with a single trip and 6 of 16 dead cohorts; the per-cohort
  reduction order is fixed by the qualified kernel, and every geometry that
  changes it (wave split, pair, row tile) either re-associates sums or costs
  registers. The input row re-read (9.8 MB/block of L2) is the only redundant
  traffic and removing it needs a 16-row tile, which was not shown to win.
* **rms / inject / mix / reduce dispatches.** A merged rms+inject or
  inject+mix would have to re-associate the FP sums that feed `injection`
  (which scales the block output), so it is a precision change, not a merge.
  The remaining GR chain is six serialized dispatches per read and the two
  GEMMs dominate it.
* **Recurrent algebraic 3 passes.** The second state read is required because
  `delta` depends on the full `memory` reduction; it runs at 257 GB/s and the
  state bytes are the floor.
* **r8 prefetch / double buffering.** Any extra in-flight state raises VGPRs
  above 36 and the fleet already showed the register-limited wave order
  (7 > 6 > 5) dominates instruction-count wins (pair -3.4 TPS). Not
  re-litigated.
* **QSA projection elementwise.** Out of this workstream's scope and not
  manifest-exact in the audit model.

## 4. Local validation (llvmpipe, correctness only)

```
make all -j8                     warning-free (-Werror)
make test-fg-dense-rates         PASS (2 tests)
tests/test_qsa_prefill           PASS
tests/test_expert_prefill        PASS
tests/test_owner_reduce          PASS
focused tests/test_fg_vk (FG_SHADER_DIR=$PWD/vulkan), 30 oracles:
  q8_split_silu_parity, ple_conv_add_parity (new, bitwise memcmp)
  group_norm (FG_RMS_HASH old==new), gr_mix, gr_batch, hc_inject_partial,
  gr_partial_boundaries, q8_dense_cooked, q8_decode_shape_parity,
  q8_cooked_prefill_parity, q8_cooked_view_slice, ple_decode,
  ple_prefill_scan, ple_prefill_t1_compat, gdn_decode, gdn_algebraic,
  gdn_project_cooked, gdn_prefill_scan, gdn_prefill_decode_compat,
  hc_finalize, q8_dense_subgroup, q8_dense, q8_embedding, gpu_profile,
  batch_submission_parity, pipeline_flush_parity, static_batch_replay,
  memory_telemetry_and_canary, tensor_view_rebind,
  gdn_chunked_prefill_parity_{zero,random,extreme},
  gdn_chunked_prefill_{composition,decode_compat}                 PASS
known pre-existing failures unchanged (test_core:457,
test_session:254/257, test_prefill_dispatch:178-183, llvmpipe
qsa_record_commit crash)
```

## 5. Fleet qualification plan (orchestrator)

No new flags or environment variables; all changes are unconditional and
byte-identical on the wire. A/B the branch against `main` `405c3f9` on the
same ring pack and thermal state:

1. Gates and the standard battery from the round-12/13 protocol (same
   harness, runs 4k x2, short decode, 32k sweep), bracketed by controls.
2. Attribution: `FG_DECODE_PROFILE=1` on rank 0 and rank 1, one token, then
   `python3 tools/fg_dense_rates.py <logs>`; compare the changed rows against
   the section-1 baseline: `gr_*_read` totals, `fg_group_rms_norm.spv`,
   `fg_hc_inject_partial.spv`, `fg_dense_q8_0_cooked_split_reduce_silu.spv`
   (new; `..._reduce.spv` disappears), and the `ple` scope.
3. Revert points: each commit is independent. `dd8d1ce` is the only kernel
   binary change with an arithmetic-equivalent rewrite; `fa7e3be` carries the
   schedule change whose overlap depends on the blade's dispatch behaviour.

Promotion bar: gates green, pi-stability PASS, no prefill regression, short
and 4K decode at or above the same-session controls, and the per-kernel rows
of item 2 at or better than section 1 (within the fleet's +/-10-15% thermal
spread).

## 6. Honest blockers

* The GDN output shape (1.4-1.7 ms/token of headroom against the 300 GB/s
  floor) is blocked on the row-interleaved pack layout: a pack-format version,
  a rebuild and all parity oracles on the new pack. Round 12 already designed
  the addressing change; it needs its own fleet round.
* The GR up shape and the whole-cohort reduction order are locked by the
  bit-exact requirement; the remaining 0.9 ms/token there is unreachable
  without either a pack layout change or a tolerance budget.
* The GR elementwise floor is dispatch latency: 6 dispatches per read, 72 per
  block, ~576 per token. Further cuts are precision changes.
