# MTP: source audit, causal verification and remaining execution gates

2026-10-15. Branch `cuppaxanax/mtp`, based on `origin/main a63a166`.

**Not a serving MTP implementation and not a 60 TPS result.** The trained head
exists. This branch implements a Q8-only overlay exporter, a transactional
greedy cycle, residual rejection sampling, an owner-local causal verification
entry point, durable QSA truncation and single-blade rollback tests. The actual
hyperconnection MTP executor and batched ring orchestration remain unlinked.
The existing `--experimental-mtp` CLI gate stays off by default and fails closed;
it must not advertise MTP capability or execute a fake echo proposer.

## 1. Source facts, not the old feasibility assumptions

Range-only audit of these immutable Hugging Face revisions:

| Source | Revision | MTP tensors | BF16 bytes |
|---|---|---:|---:|
| `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` | `298f94632b784e26a7fe576114f82066689d5baa` | 31 | 5,214,301,696 |
| `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 31 | 5,214,301,696 |

Both complete configs are identical, as are the MTP tensor names, dtypes and
shapes. Tensor *data* equality was not tested. The original head is structurally
compatible as a draft head, but there is no need to borrow it: the abliterated
source retains its own. A less accurate draft still gives exact target output
only when verification, rollback and stochastic acceptance are correct.

`tools/mtp_source.py` fetches config/index JSON, then 8-byte safetensors lengths
and bounded JSON header ranges, not full shards. It pins the repository commit,
requires HTTP 206 with the exact Content-Range, and records offsets and header
SHA256. Audit records are in `results/hillclimb/mtp-{original-,}source.json`.

All following shapes are **HF row-major**, with the `mtp.` prefix omitted.
All source tensors are BF16:

| Name | Shape | Bytes |
|---|---|---:|
| `fc_embedding.weight` | 2560 x 2560 | 13,107,200 |
| `fc_hidden.weight` | 2560 x 2560 | 13,107,200 |
| `pre_fc_norm_embedding.weight` | 2560 | 5,120 |
| `pre_fc_norm_hidden.weight` | 10240 | 20,480 |
| `hyper_connection_mixer.hc_norm.weight` | 10240 | 20,480 |
| `hyper_connection_mixer.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 |
| `hyper_connection_mixer.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 |
| `layers.0.attn_hyper_connection.block_inject_weight.weight` | 4 x 10240 | 81,920 |
| `layers.0.attn_hyper_connection.hc_norm.weight` | 10240 | 20,480 |
| `layers.0.attn_hyper_connection.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 |
| `layers.0.attn_hyper_connection.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 |
| `layers.0.mlp_hyper_connection.block_inject_weight.weight` | 4 x 10240 | 81,920 |
| `layers.0.mlp_hyper_connection.hc_norm.weight` | 10240 | 20,480 |
| `layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | 320 x 10240 | 6,553,600 |
| `layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | 10240 x 320 | 6,553,600 |
| `layers.0.mlp.experts.down_proj` | 512 x 2560 x 640 | 1,677,721,600 |
| `layers.0.mlp.experts.gate_up_proj` | 512 x 1280 x 2560 | 3,355,443,200 |
| `layers.0.mlp.gate.weight` | 512 x 2560 | 2,621,440 |
| `layers.0.mlp.shared_expert.down_proj.weight` | 2560 x 640 | 3,276,800 |
| `layers.0.mlp.shared_expert.gate_proj.weight` | 640 x 2560 | 3,276,800 |
| `layers.0.mlp.shared_expert.up_proj.weight` | 640 x 2560 | 3,276,800 |
| `layers.0.mlp.shared_expert_gate.weight` | 1 x 2560 | 5,120 |
| `layers.0.self_attn.indexer.index_qk_proj.weight` | 640 x 2560 | 3,276,800 |
| `layers.0.self_attn.indexer.k_layernorm.weight` | 128 | 256 |
| `layers.0.self_attn.indexer.q_layernorm.weight` | 128 | 256 |
| `layers.0.self_attn.k_norm.weight` | 256 | 512 |
| `layers.0.self_attn.q_norm.weight` | 256 | 512 |
| `layers.0.self_attn.k_proj.weight` | 512 x 2560 | 2,621,440 |
| `layers.0.self_attn.v_proj.weight` | 512 x 2560 | 2,621,440 |
| `layers.0.self_attn.q_proj.weight` | 12288 x 2560 | 62,914,560 |
| `layers.0.self_attn.o_proj.weight` | 2560 x 6144 | 31,457,280 |

The two expert tensors alone contain 5,033,164,800 BF16 bytes. The old claim
that this head is merely hundreds of MB is wrong. With Q8_0 matrices and F32
vectors the overlay payload is **2,770,227,872 bytes (2.580 GiB)** before GGUF
metadata/alignment, excluding shared embedding/head and MTP KV allocations.
Q4-class formats could reduce residency later, but require a quality gate;
no format below Q4 is permitted. This branch exports only Q8_0/F32.

The exporter preserves HF names, fused expert/indexer tensors and **raw norm
gammas**. It is a source overlay, not an executable owner-layer alias. The
future importer must split expert gate/up along the 1280 axis, split indexer
query/key at row 512, cook supported matrices, and apply each Gemma `+1`
exactly once. Do not silently feed raw zero-centred gammas into ordinary RMSNorm.
Full-sized MTP weights were not downloaded or qualified on this branch; exporter
round-trip tests use small synthetic BF16 tensors.

## 2. Correct architecture

The actual config says `Qwen4ExpForConditionalGeneration`, hidden 2560,
HC count 4, HC low rank 320, 512 experts/top-10, one hybrid full-attention
MTP layer and `mtp_use_dedicated_embeddings=false`.

The relevant reference is SGLang's
[qwen4_exp_mtp.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen4_exp_mtp.py),
particularly `_fuse_residual_linear_shared`:

1. Embed the known next token using the target embedding.
2. Gemma-RMS-normalize its 2560-wide embedding and project with `fc_embedding`.
3. Gemma-RMS-normalize the **entire 10240-wide target HC hidden**. Reshape it to
   four 2560-wide streams and apply the same `fc_hidden` to each stream.
4. Add the projected embedding to every projected HC stream.
5. Execute one HC/QSA/MoE/HC layer, with no PLE and with separate draft KV state.
6. Contract through the **MTP's own** `hyper_connection_mixer`; apply the shared
   target vocabulary projection. Preserve the pre-contraction HC output for
   recursive drafting and accepted-target hidden-state refresh.

This is not the plain normalized embedding/hidden concat into a 5120->2560
projection used by
[vLLM Qwen3Next MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py)
and
[SGLang Qwen3Next MTP](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next_mtp.py).
Local read-only `llama.cpp\src\models\qwen3next.cpp` implements that NextN
graph, but the local `qwen4exp.cpp` has no MTP graph and its
`conversion\qwen4exp.py` explicitly disables MTP export. Nothing in llama.cpp
was modified.

## 3. Placement and causal ring design

Prefer the rank owning the final target block and embedding, currently rank 7,
not a hardcoded rank number. It already has the full final HC hidden before
direct output handoff. Keeping the draft layer there avoids returning each
10240-float hidden to rank 0. Reuse the existing four-way vocabulary-head fanout,
but feed it the **MTP-contracted** 2560 vector rather than the target HC mixer.
Account for a head/fanout on each draft step; those costs are not free.
Prove the extra 2.58 GiB plus separate draft KV, target state checkpoints and
verification scratch fit that rank before loading an overlay.

Use configurable `k=1..3`; initially qualify k=1. Once target token `t` has
already been emitted and is pending consumption, draft `d1..dk`, then send
**one causal sequence** `[t,d1,...,dk]` through every stage, requesting every
row's hidden/logits. Positions increase within this sequence.

The existing `decode_batch` protocol/scheduler is **not this operation**:
its slots are independent sessions with distinct recurrent namespaces.
`coordinator_decode_batch_step` and `handle_decode_batch_work` cannot simply
be handed repeated state slots. The existing prefill layer supports causal
rows, but has a very different execution/cost/numerical path.

Ring transactions must:

* checkpoint GDN convolution/recurrent state and PLE convolution on **each
  owner**, plus target and draft QSA frontiers;
* isolate speculative QSA pages, index metadata, positions and replication
  cursors; do not publish them to storage ranks until acceptance;
* return all k+1 target logits/HC rows, not just a prefill chunk's last row;
* commit the verified prefix on all owners, with generation/session identity
  checks and poisoned transport on a partial distributed failure;
* restore the initial recurrent checkpoint and replay the accepted input
  prefix on a rejection, until per-position recurrent checkpoints exist.
  Keeping only a final GDN state and truncating QSA is **incorrect**.

The low-cost endpoint should retain per-position GDN/PLE state during the
short sequential recurrence, so rejection selects a prefix checkpoint instead
of executing accepted tokens again. A full-state copy on every draft position
must be included in the bandwidth ledger.

The MTP QSA cache also needs prompt warmfill/teacher-forced updates using
shifted embeddings and target HC hidden states. After verification, repair
its accepted prefix with target HC rows, not rejected recursive draft hidden
states. There are no captured abliterated target HC streams here; offline
acceptance was consequently **not measured**, and hidden-feature argmax
agreement in the synthetic benchmark is not an MTP acceptance measurement.

## 4. Acceptance and state frontier

`fg_mtp_cycle_greedy` implements the cycle independently of a particular
transport/backend. Target row j predicts draft j+1. Emit the matching draft
prefix and the first target correction, or a target bonus if all drafts match.
If a cycle emits m new tokens, consume exactly m inputs: the already emitted
pending token plus the emitted prefix except its last token. The last emitted
token stays pending. EOS and output budgets shorten that frontier too.
No result is returned until checkpoint/verify/restore/replay/commit succeed.

Temperature sampling cannot use target/draft argmax equality. For proposal
distribution q, target p and proposed token d, accept with
`u < min(1,p[d]/q[d])`; on rejection draw from normalized `max(p-q,0)`.
`fg_mtp_rejection_sample` implements and tests this primitive, including
disjoint supports and an exactly integrated two-token distribution.
Distributions must include the same temperature, top-k/top-p and history
penalties as the baseline sampler, at each hypothetical prefix. The existing
sampler's filter/temperature ordering must be preserved. They are not raw
logits or only a proposal's scalar probability.

This stochastic primitive is **not connected to runtime logits or the cycle**.
Distributional equality does not imply identical seeded sample trajectories.
Greedy runtime equality also needs a target verifier with validated numerical
semantics; the current prefill verifier does not have bitwise hidden parity.

## 5. Measured causal batch costs on .51

Derived from `cuppaxanax/decode-fusion`'s `tools/bench_decode_fusion.c`,
retaining the production N=1 `fg_owner_decode_block_chained` static-replay
baseline. The imported fixture is adapted to build against main, with
nonintrusive wall timing and **no** fusion-branch clock API dependency.

N=2..4 calls `fg_owner_verify_block`, a causal block wrapper around the
existing prefill layers and local expert prefill. Same owner-local layers
6..11, 9,739,403,264 synthetic Q4/Q5 weight bytes, nonzero changing inputs,
full 512 expert slabs, real top-10 routing and QSA. These are not model
weights, full model logits, fleet transport or a fused small-N verifier.

Two fresh seeds, forward/reverse N order, 256-token warmup rounded down to a
whole batch and three windows of approximately 64 input tokens each.
N=3 starts at 255 and covers 189 measured inputs; others cover 192. Comparison
aligns overlapping absolute token positions. Stock governor is untouched:
endpoint samples range from 1000 to 2000 MHz, so these are coarse cost
measurements, not a tightly pinned qualification or confidence interval.

| N=k+1 | Mean block us | Mean per-seed cost ratio to N=1 | Individual ratios |
|---:|---:|---:|---|
| 1 | 2,675.68 | 1.000 | 1.000 / 1.000 |
| 2 | 22,862.45 | 8.545 | 8.506 / 8.583 |
| 3 | 23,946.70 | 8.950 | 9.028 / 8.871 |
| 4 | 25,066.67 | 9.369 | 9.151 / 9.587 |

**The available causal batch path does not cost approximately one decode.**
It includes different kernels, CPU routing/readback, expert staging/reduction,
QSA persistence and no GDN static replay. This does not prove an optimized
small-N kernel cannot work; it proves existing prefill is not that kernel.
All compared hidden-feature argmaxes match, but **zero** N>1 output vectors
are bit-identical to N=1; max absolute hidden differences reach 0.18512.
No inference about actual vocabulary argmaxes/quality follows.

Raw window samples, endpoint clocks and configs are retained in the local
`mtp-measure.tar`; tracked summary: `results/hillclimb/mtp-batch.json`.
`results/hillclimb/mtp-evidence.tar` additionally preserves raw measurements,
all rollback runs, the validation build log and tested binary SHA256 hashes.

Let F be baseline single-token latency, r the actual **whole-ring** verification
cost/F, m the draft-step cost/F, and C rollback/commit overhead/F. At constant
per-position acceptance a, the independence approximation is:

```
E = 1 + a + ... + a^k
TPS = baseline_TPS * E / (r + k*m + C)
```

Use measured prefix survival probabilities instead of powers of a once they
exist. From 27 TPS, 60 TPS needs a speedup of 2.222. For k=3:

| Assumed a (not measured) | E | Maximum total cycle/F to reach 60 TPS |
|---:|---:|---:|
| 0.7 | 2.533 | 1.140 |
| 0.8 | 2.952 | 1.328 |
| 0.9 | 3.439 | 1.548 |
| 1.0 | 4.000 | 1.800 |

If the six-layer ratio represented the full ring, even perfect acceptance,
zero draft cost and zero rollback overhead would yield only about **11.5 TPS**
for N=4 (27*4/9.369), not a speedup. That is a conditional extrapolation,
not a measured full-ring TPS. A 60+ projection is unsupported until an
exact, much cheaper small-N verifier and the real head are measured.

## 6. What is tested and what remains

* Host synthetic autoregressive model: k=1..3, every rejection position,
  output caps, EOS and injected verify/replay/commit errors. Speculative output
  tokens and retained mock GDN/PLE/KV state equal ordinary greedy decode.
* Actual six-layer Vulkan block: host and GPU snapshots, discard a four-row
  causal verification, restore and replay each retained prefix 1..4.
  Compare complete GDN state and last output bytes against serial decode.
  Change speculative MRoPE triples to prove rejected positions are reusable.
  Cover all four QSA page-boundary offsets. This tests exact rollback, not
  bitwise equivalence of prefill arithmetic and decode arithmetic.
* QSA truncation now repairs durable partial-page count/CRC and file frontier,
  reconstructs the retained tail, clears rejected MRoPE-written flags and
  unpins suffix pages. Prefix records remain append-only; suffix GPU bytes
  outside the frontier are inaccessible and overwritten before becoming visible.
* Source tool tests: ignored HTTP ranges rejected without reading the body,
  Q8 round-trip/zero blocks, raw F32 norm preservation, GGUF sizes and
  no overwrite of an existing overlay.

The .51 validation passed the full `-Werror` build, flag registry, MTP/sampler/
depth-B host suites and all **16/16** GPU rollback cases; see
`results/hillclimb/mtp-validation.json`. The three exporter tests passed locally.
The remote `/home/xander/hc-mtp/` experiment directory was removed after
downloading the evidence; no service or governor configuration was changed.

Reproduce from a Linux checkout with Vulkan build dependencies:

```
mkdir -p .build
make -j8 flash-gordon tools/bench-decode-fusion tests/test_mtp tests/test_sampler tests/test_decode_batch
./tests/test_mtp
./tests/test_sampler
./tests/test_decode_batch
bash tools/check-flags.sh
./tools/bench-decode-fusion --q4 --batch 1 --warmup 5 --iterations 2 --windows 1 --rollback-test
./tools/bench-decode-fusion --q4 --batch 4 --warmup 64 --iterations 16 --windows 3
python -m unittest discover -s tools -p test_mtp_source.py -v
python tools/mtp_source.py --audit results/hillclimb/mtp-source.json
# Large ranges: execute on the blade, not through Windows:
python tools/mtp_source.py --audit mtp-export.json --pack mtp-q8.gguf
```

Export requires NumPy; audit does not. Fleet scripts use only
`/home/xander/hc-mtp/`, and the PowerShell wrapper forwards the allowlisted
target set explicitly even when its default is used. An early wrapper call
omitted that forwarded default and attempted transfers to other blades;
those transfers failed before any file/script execution there. The wrapper
was fixed immediately; no production deployment or service was changed.

Remaining work before even an eight-rank correctness qualification:

1. Implement/import the real HC-aware MTP layer, aliases/cooking, separate
   draft KV warmfill and the MTP mixer/shared head handoff; verify its outputs
   against the Qwen4Exp reference. Qualify Q8 quality, not only file layout.
2. Implement a dedicated small-N verifier that preserves required target
   semantics without the old prefill's CPU/persistence overhead. Measure
   every rank's service, recurrence checkpoint and output-head costs.
3. Connect causal wire work/results, deferred page publication, all-rank
   prefix transactions, history/ngram cursors, cancellation/EOS and logits.
4. Connect stochastic distributions and per-prefix penalty histories, or keep
   temperature requests explicitly unavailable.
5. Obtain real acceptance on chat/code contexts and real target greedy
   identity, then run fleet A/B with fixed hashes and long-context/state gates
   when the production-rank lease is available. No fleet lease was taken here.
