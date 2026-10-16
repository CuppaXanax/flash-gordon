# MTP small-N verifier: .51

Branch: `cuppaxanax/mtp`. Software only, stock governor unchanged, only
192.168.42.51 used. Synthetic Q4-floor owner layers 6–11, 9,739,403,264
resident weight bytes, all 512 expert slabs distinct. Not a model/ring test.

| N=k+1 | Mean block ms | Mean per-seed ratio |
|---:|---:|---:|
| 1 | 2.684 | 1.000 |
| 2 | 4.078 | 1.519 |
| 3 | 5.903 | 2.199 |
| 4 | 6.725 | 2.505 |

Two seeds and reversed order, 256 input warmup, three ~64-input windows.
Historical prefill fallback ratios were 1/8.545/8.950/9.369. The fallback
regression is substantially reduced; **1.3x N=2 and 1.8x N=4 remain unmet**.
Do not interpret these implementation costs as a limit on MTP.

Implemented: causal small-N owner API, coalesced Q4/Q5/IQ4/Q8/F32/BF16
projection kernels, GPU top-10/expert union, shared gate/up/down weight
reads, ordered per-part gate accumulation, sequential GDN recurrence,
GDN-run static replay and per-row causal QSA cache/attention.
Runtime MTP remains default-off/fail-closed: native head/ring are unlinked.

All 1,144 aligned HC rows, 90 dense cases, N=2..4 whole-block outputs and
recurrent state are byte-exact against the same binary's N=1. Also passed:
12 scheduler cases, prefix causality, injected failure cleanup, 16/16 QSA
boundary/prefix rollbacks including replay-mode switching, Werror build,
MTP/sampler/depth-B host tests and flag registry.

Isolated 2560→6144 Q8 replay ratios: 1/1.193/1.541/1.797;
Q4: 1/1.512/1.820/2.042; IQ4: 1/1.552/2.436/3.010.
N=4 ISA statistics: all 30 pipelines have zero scratch/spills. Gate/up,
union-down and ordered-reduce use 64/64/16 VGPRs. Final-layer expert unions
contain 10–12 experts; this does not characterize every layer. Clock
endpoint samples span 1000–2000 MHz, not a pinned operating point.

Acceptance is unmeasured. Strata `9259cad`, `DETAILS.md:743` supplies the
labeled estimate E=2.4–3.2 committed tokens/pass, up to three drafts.
Projection: `TPS = baseline * E / (whole-ring r + draft/F + commit/F)`.
A future 40–45 serial TPS and qualified r=1.8 gives 53.3–80 TPS **before**
draft/commit. No production speedup or architectural ceiling is claimed.

Remaining implementation: Q4/IQ4 verifier tuning, HC-aware native head
import/execution on the derived final-stage owner, independent dense draft
KV/coupling, cheap retained-prefix GDN/conv/PLE commit, causal ring wire and
full-model exact greedy/acceptance tests. GPU layer-1 PLE is outside this
fixture. No work or rollout occurred on .42–.50, .52 or .53.

See `mtp-small-n.json`, `mtp-small-n-evidence.tar.gz` and `docs/MTP.md`.
