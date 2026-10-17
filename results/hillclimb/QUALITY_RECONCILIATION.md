# Q8 / mix-04 / stack quality reconciliation

## Decision

**The historical numbers reproduce; their qualification scope was too broad
when carried over to the FP16 experiment.** The principal discrepancy is the
evaluation cohort, with additional execution-path/profile sensitivity.
It is not a broken Q8 reconstruction, mismatched token positions, or a
full-vocabulary KL calculation bug.

The original 480 natural-frontier qualification remains valid on that cohort.
It does **not** establish the same quality on the 80 forced-assistant-prefix
contexts: mix-04 fails those contexts even using the original native-prefill
method and original Q8 reference. This is a genuine broader quality limitation,
not grounds to discard the stress cohort or waive its failures.
**No FP16 policy, including mask 224, is promoted.**

## Same texts and positions, both methods

On isolated `.50`, independently execute all 48 trained layers, real routed
experts and the complete 248,320-logit head for:

* **Forced80:** the original 20 prompts, each with assistant prefix `""`,
  `"First"`, `"First, consider"`, or `"First, consider the"`.
  There are 1,808 supplied tokens, final positions 16–38.
* **Natural480:** the original frozen teacher-continuation frontiers used by
  Q4 and stack qualification. There are 15,408 supplied tokens, final
  positions 16–57.

The combined 560 supplied texts contain 20 deliberately repeated frontiers:
the empty assistant prefixes also occur in Natural480. Token counts and final
positions agree with **all 560 historical native capture records**. The first
80 token IDs, embedding bytes and n-gram bytes exactly reproduce the previous
FP16 experiment. These are re-encoded frozen texts, not an assumption that
original generation-step numbers equal subsequent tokenizer positions.

Run both original-Q8 and sealed mix-04 weights through each path:

1. **Legacy native prefill:** `fg_owner_prefill_layer`, sealed microbatch 128,
   research roof/small/wide/cache/group options disabled.
2. **Selected native prefill:** same native path and microbatch, selected
   research profile.
3. **Selected cold N1:** every supplied prompt token processed by the original
   offline decode evaluator; microbatch 2 is its decode scratch allocation.
   This is not native prefill followed by ordinary serving decode.

All work boundaries remain f32. Block-owner order is **1,0,2,3,4,5,6,7**;
rank 0 executes layers 6–11. Fresh two-slot state and owned QSA mirrors are used.
No production requests, fleet timing runs or other blades are involved.

### Direct cumulative comparisons, matched Q8 execution path

Every entry is direct `KL(P_Q8 || P_mix04)` from full-vocabulary vectors.
The strict bar is mean KL ≤ 0.005 and top-1 ≥ 98.5%.

| Cohort | Execution path | Mean KL | Top-1 | Strict gate |
|---|---|---:|---:|---|
| Natural480 | Legacy native prefill | 0.003128578 | 475/480 = 98.9583% | PASS |
| Natural480 | Selected native prefill | 0.003278703 | 474/480 = 98.7500% | PASS |
| Natural480 | Selected cold N1 | 0.003335419 | 472/480 = 98.3333% | FAIL top-1 |
| Forced80 | Legacy native prefill | 0.059809481 | 68/80 = 85.0000% | FAIL both |
| Forced80 | Selected native prefill | 0.053829783 | 68/80 = 85.0000% | FAIL both |
| Forced80 | Selected cold N1 | 0.051191953 | 68/80 = 85.0000% | FAIL both |

**The 85% result survives switching back to the original qualification method.**
Changing cold N1 to native prefill does not explain it away.
Conversely, using cold N1 on the original natural cohort brings KL back near
0.003, although its top-1 narrowly fails the strict bar.

For completeness, the explicit 560-row union also fails all three paths:
legacy native **0.011225850 / 96.9643%**, selected native
**0.010500286 / 96.7857%**, selected cold N1
**0.010172067 / 96.4286%**. These are ordinary sample means over the declared
union, not sums of incremental KLs or estimates for a serving population.

### Why the cohorts differ

The same 20 empty-prefix cases pass all paths with 100% top-1 and mean KL
between 1.26e-8 and 3.44e-8. The nonempty, fixed assistant prefixes account
for the stress failures. In the original-method legacy-native control:

| Assistant prefix | Samples | Mean KL | Top-1 |
|---|---:|---:|---:|
| empty | 20 | 0.000000024 | 100% |
| `First` | 20 | 0.035598624 | 95% |
| `First, consider` | 20 | 0.094168782 | 70% |
| `First, consider the` | 20 | 0.109470495 | 75% |

These fixed continuations are not sampled Q8 teacher continuations.
This observation identifies the sensitive cohort, not an invalid-input claim
or a causal attribution to a particular quantized tensor.

## Historical reference correction and exact reproduction

The stack number **0.000246435 / 99.7917%** compares optimized **mix-04**
against baseline **mix-04**, not against Q8. Both stack manifests are mix-04.
It must not be described as cumulative Q8 quality.

| Original captured comparison, Natural480 | Mean KL | Top-1 |
|---|---:|---:|
| Original Q8 → original mix-04 | 0.003128578 | 98.9583% |
| Stack A/mix-04 → stack B/mix-04 | 0.000246435 | 99.7917% |
| **Original Q8 → stack B, directly recomputed** | **0.003084267** | **98.7500%** |

The third line passes the strict bar, but is computed from the actual vector
pairs, **not** by adding the first two KL values.

New isolated executions reproduce the **entire logit-file bytes**, not only
aggregate statistics:

* Legacy-native Q8 versus original Q8: **480/480 natural + 80/80 forced**.
* Legacy-native mix-04 versus original mix-04: **480/480 natural**.
* Selected-native mix-04 versus original optimized stack B: **480/480 natural**.
* Cold-N1 Q8 and mix-04 versus the previous FP16 controls: **80/80 each**.
* Original stack A versus original mix-04: **480/480**.

This rules out a changed tokenizer, Q8 build/reconstruction, source/head
capture implementation, reference-file mixup or position mismatch as the
explanation for these historical comparisons.

Holding the **original**, rather than newly selected-profile, Q8 vectors fixed,
selected native mix-04 gives **0.003084267 / 98.75%** on Natural480.
Selected cold-N1 mix-04 gives **0.003212022 / 98.125%**. Reference geometry
must therefore remain explicit in future qualification.

## Additional numerical scope

Same-weight selected prefill versus cold N1 on Natural480:
Q8 **0.000264390 / 99.375%**, mix-04 **0.000236926 / 99.375%**.
On Forced80 those become Q8 **0.009128176 / 96.25%** and mix-04
**0.016218473 / 95%**. Same-weight legacy versus selected native prefill on
Forced80 is Q8 **0.009679231 / 96.25%**, mix-04
**0.014541187 / 100%**. Thus profile and batch arithmetic are themselves
more sensitive on the forced-prefix cohort; cohort sensitivity is not solely
a trained-weight quantization observation.

Native historical qualification resets and generates one first prediction
after prefill for each complete frozen text. It is not a test of long-running
autoregressive decode, hybrid prefill/N1 tails, N4 verification, long contexts,
or every deployed runtime revision. The cold-N1 top-1 failure must not be
rounded up to a pass. No such coverage or current live-ring requalification
is claimed here.

Mask 224's **incremental** 0.000678952 / 100% remains a promising research
result. Its matched cold-N1 cumulative Forced80 value is still
**0.054000836 / 85%**. No masked Natural480 replay was performed in this
reconciliation; a passing uncompressed baseline on Natural480 does not
qualify the compressed policy there.

## Authentication, controls and reproduction

* Mixed manifest SHA256:
  `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
* Original-Q8 manifest-file SHA256:
  `35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`.
* All **1,223** original-Q8 trained spans authenticated and read back:
  824 exact copies, 351 original dense-Q8 cooks, 48 provably lossless
  BF16-to-original-F32 router expansions.
* All **48 policy/owner stages** have byte-identical independently reset
  teacher/candidate frame files. Decode prefix oracles are zero; native
  prefill's independent full-frame SHA comparisons supply the stronger control.
* All **20 repeated-frontier controls per policy**, 120 total, produce
  identical full logit files despite different preceding cases.
* Six policies × 560 full-vocabulary vectors retained privately, along with
  prepared token/embedding/n-gram inputs and final hidden vectors.
  No weights, activations or logit payloads are committed.

The numerical helper SHA256 is
`b4efd7576d8c5677f3a0bf5825eea532de472593eb56f80b0766b9b19f1f1207`.
It uses the prior `9531f7e` source/shaders with the native-prefill offline-helper
extension committed alongside this report. `HOP_OFFLINE_PATH=prefill` is a
private tool control, f32-only; unset preserves existing cold decode.
There are no serving-default or new public flag changes.

Selected profile: `FG_ROOFLINE=2047`, `FG_ROOF_DENSE=19`, `FG_ROOF_QSA=3`,
`FG_ROOF_EXPERT=33024`, `FG_ROOF_HC=96`, `FG_ROOF_RECURRENT=3`,
`FG_ROOF_GDN=1`, `FG_SMALL_KERNELS=330`, `FG_EXPERT_Q8_WIDE=1`,
`FG_NGRAM_LOCAL_CACHE=1`, `FG_DECODE_SUBMIT_GROUP=2`; hop codecs disabled.
Legacy controls set those research options to zero.

`quality-reconciliation.json` contains compact metrics, per-context failure
IDs, restoration/stage proofs and artifact hashes.
`quality-reconciliation-evidence.tar.gz` contains the full weight-free
analysis, exact text/token/position identities, source/binary/environment
receipts and cleanup evidence. Private numerical archives are SHA-bound:

* Raw execution evidence:
  `30cbd6a17277f6458d770fde0eafebc172f7280207559a4cc600409c6ddebeda`.
* Full logits:
  `24ca48c7fe10c2744da2001da97ee0d9834231ecdbdc6ff47ede9477e0f131ce`.

`tools/reconcile_quality.py` independently computes float64 stable softmax KL,
requires identical corpora and rejects incomplete/nonfinite vectors.
`tools/reconcile_report.py` rebuilds all comparisons and fails on stale capture
links or failed stage/restoration controls. `tools/reconcile-run.ps1` documents
the owned `.50` rank-streaming protocol; its `-Resume` is specifically for
the initial pre-stage upload failure, not a general mid-run restart.

Validation: native and WSL helper builds, three comparator tests, eight
LF-normalized shell syntax checks, flag-registry gate, and real-model controls
above all pass. One initial CRLF checksum-format failure occurred before any
model stage; its receipt is retained and both packaging/parser were corrected.

`.50` cleanup verified at **2026-10-17T00:27:29Z**: owned scratch absent twice,
zero owned processes, both original embed/rerank services healthy, governor and
performance level unchanged. No `.42`–`.49`, `.51`, `.52` or `.53` access.
