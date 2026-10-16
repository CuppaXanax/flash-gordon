# MTP offline goodput policy, dense reuse and complete rank7 fit

**Default off; no promotion or production access.** All GPU work used only
192.168.42.51, reporting **24 CUs**, under `/home/xander/hc-mtp/goodput`.
Source base is `83af9f3568b527dafe0e25d6b40b37c55e0f68a9`.
Target weights remain at least Q4; native draft weights remain Q8 here.
The user permits lower-bit drafts, but this experiment does not implement them.

Machine-readable results: [mtp-goodput-local.json](mtp-goodput-local.json).
Raw nonweight receipts: [mtp-goodput-local-evidence.tar.gz](mtp-goodput-local-evidence.tar.gz),
SHA256 `221a9aba787ba367842d88fbc9607ad6931c677b95f48d9a360b747bfae2eb37`.
No model tensor payloads or captured hidden-state payloads are published.

## 1. Actual serial fallback, not a speculative round disguised as N1

The adaptive chooser now includes k=0 and requires a measured ordinary-target
baseline before proposing drafts. It compares expected tokens/complete-round
time against serial goodput with 0.5% headroom. Conditional acceptance and
observed shape costs use 1/16 EWMA; optimistic initial acceptance priors are
explicitly **not calibrated confidence**. There are no forced periodic probes.

Losing streams latch serial and skip native ADVANCE RPC/GPU work. Target HCs
remain observed for a later synchronization, but stale native KV cannot resume
drafting. Successful prefix synchronization clears the latch; exact-prefix
continuations retain it. Cold/new contexts reset acceptance, while prefix-hit
continuations retain learned acceptance. A one-token generation tail uses N1;
research fixed-width overrides remain explicit.

Extreme first-use compilation is excluded from warm price learning once, not
from actual elapsed time. Serial timing is captured before native work.
The default-off path does not add a clock read.

**This is implemented/tested, not an established chat≥1.0×/code-win result.**
Initial probing can still cost latency. No new live-ring on/off TPS was measured.

## 2. Dense reuse: a partial result, not the requested whole-rank target

Mask330 does not enable cooked-Q8 reuse (bit1). Blanket331 improves large Q8
but regresses small HC projections. Existing `FG_SMALL_KERNELS` bit4096 now
enables that reuse only when M,K≥1024; 330+4096=4426. No new environment name
or default behavior is introduced.

Representative2560×6144 replay-warmed measurements:

| Matrix |Mask |N1 |N4 |N4/N1 |
|---|---:|---:|---:|---:|
|Q8 |4426 |59.434 µs |85.383 µs |1.437 |
|Q4 |4426 |33.323 µs |55.422 µs |1.663 |

Forcing wave32 worsened representative Q4 to≈2.15× and was discarded.
Existing Q4 reuse kernels already unpack once across columns; merely renaming
the current batching is not an optimization. Synthetic complete-block results
still exceed2× and do not establish a whole-block improvement.

New **authentic mixed-pack rank7** tests warm24 and time32 calls/shape/slot:

| Head/cache; mask; slot |N1 |N2 |N3 |N4 |N4/N1 |
|---|---:|---:|---:|---:|---:|
|CJK320;330;0 |3.139 |4.657 |6.167 |7.128 |2.270 |
|CJK320;330;1 |3.145 |4.679 |6.201 |7.156 |2.276 |
|English416;330;0 |3.445 |4.636 |5.673 |5.750 |1.669 |
|English416;330;1 |2.600 |3.784 |4.824 |5.507 |2.118 |
|English416;4426;0 |3.446 |4.640 |5.539 |5.901 |1.713 |
|English416;4426;1 |2.790 |4.067 |4.877 |5.543 |1.986 |

All times are median milliseconds, excluding input/output copies. These runs
use **synthetic sinusoidal HCs and zero ngram**, not captured model states.
Different cache/slot/process runs show material timing drift; they do not
support a claimed mask4426 whole-rank gain. **The≤1.5× rank target is unmet.**
No5–15 ms live-ring saving is credited.

## 3. Actual curated vocabularies and complete two-session graph fit

Curated IDs:

* English40,525: Strata commit `f2a08d4b2da99bb0638dd03dc1d097d132f0af68`,
  binary SHA `369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25`.
* CJK-inclusive106,299: Strata `data/draft_vocab.bin`,
  SHA `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`.

The source raw Q8 head SHA remains
`2801d79db74bcae38b47fd5051f03136d8ca7715abeba7d92b49ba46c696bc33`;
cooked parent is
`fa90643ca673bf032655f6c645c2a5718bc1eac3b1b29bd3ff90eb13d14e19af`.
`FGDVOC1` authenticates every original source row, including unselected rows.
Full sources stay on disk; only selected rows become GPU resident.

| Rows |Mean/head |Requested bytes |Saving vs full |
|---|---:|---:|---:|
|248,320 |2.237300 ms |676,920,320 |— |
|40,525 |0.427269 ms |110,479,316 |566,441,004 |
|106,299 |1.016979 ms |289,784,684 |387,135,636 |

Eight synthetic vectors,8 warmups/100 timings, exact independent raw-loader
argmax IDs/scores. Three-call latency differences5.430094/3.660964 ms remain
component proxies, not serving improvements.

All152 trained rank7 tensors were reconstructed/cooked/read back against
sealed manifest SHAs. Auxiliary embedding/norm spans are also verified.
The ngram allocation uses a pinned public same-format slice with a recorded
SHA, not a production-manifest ngram digest; target timing uses zero ngram.
No model-quality claim follows from this reconstruction.

Complete target/native **sessions2**, both target N1..4 and native k3 graphs
warmed, unchanged cap **14,495,514,624 bytes**:

| Configuration |Final requested |Final allocated |Headroom |
|---|---:|---:|---:|
|CJK320 MiB worker cache |14,443,303,740 |14,443,303,760 |52,210,884 |
|English416 MiB worker cache |14,465,327,940 |14,465,328,000 |30,186,684 |

Both finish `all_graphs_ready`; the guarded4426 English run has identical
final residency. **Neither440 MiB configuration fits admission**: CJK
already exceeds the target cap by138,809,836 bytes; English has40,495,532 bytes
left but needs a50,348,032-byte second-native-session reservation. Do not
claim subset bytes alone fixed440 MiB dual sessions. These smaller-cache
configurations need separate production disk, coexistence and long-context gates.

The first smaller-cache probe found a new **probe-only** attempt to write a
rank7 ngram tensor that does not exist (ngram injection belongs to layer1).
The helper now skips an absent tensor and both complete configurations were
rerun successfully. The initial failed receipt is retained. A shell-archive
CRLF staging failure was likewise corrected before final measurements.

## 4. Retained authentic-HC acceptance, not just a head-speed proxy

No production recapture: retained40 cases (20 chat/20 code),256 teacher
frontiers/case, authentic target HCs/tokens/embedding and Q8 native layer.
Full, English and CJK runs compare local-chain proposals against an independent
direct GPU head using the original raw-row loader. Every proposed ID matches:
**45,120+45,633+45,633=136,386 exact comparisons**.

Mode0 evaluates every teacher frontier; mode1 forces keep1..4 for repair
coverage; new mode2 advances by accepted+bonus and repairs from authentic
teacher HCs. Its final round clips at256 tokens. It is a counterfactual
against **N1 teacher outputs**, not a live N4 or adaptive-policy result.

| Head |Chat accepted/drafted |Chat tokens/round |Code accepted/drafted |Code tokens/round |Warm code k3 draft |
|---|---:|---:|---:|---:|---:|
|Full |46.682% |2.392523 |75.211% |3.240506 |8.966019 ms |
|English |42.953% |2.283675 |70.629% |3.104912 |3.591989 ms |
|CJK |42.953% |2.283675 |70.629% |3.104912 |5.291276 ms |

The same acceptance schedule for both subsets on this English cohort does
not qualify multilingual coverage. Next-token coverage is92.852% chat and
96.270% code for both. Relative code yield falls**4.184%**.

Transferring that relative yield and local code draft delta onto the old
101.280 ms serving round gives only a **cross-cohort planning sensitivity**:
English≈1.012× full-head MTP, CJK≈0.994×. Do not advertise the component-only
head saving as an equivalent TPS gain or substitute these teacher yields for
the actual ring's3.386667 tokens/round.

## Validation, remaining gates and release

* C17 `-Wall -Wextra -Wpedantic -Werror` build, MTP CPU/wire/native/embedding
  suites, GPU vocabulary auth negatives, flags checker: PASS.
* Eight bounded TCP-ring variants: both native placements7/6, both sessions,
  reject positions0..3, true N1 fallback, no native RPC while latched,
  exact-prefix continuation, safe prefix resynchronization, EOS/cancel/SSE: PASS.
* 64 Python MTP tests, including binary-ID packing, source authentication and
  no-access serving plans: PASS.
* The controller stages typed subsets only after complete raw-source/parent
  validation and checks their actual SHA remotely. Dual heads remain denied
  at its440 MiB profile. The previously invalid700 argument is now explicitly
  640, preserving actual behavior.

Dense whole-rank≤1.5, new-policy chat noninferiority/code win, current N4
original-Q8 quality/margin captures and the independent N1 reference
reconciliation remain open. No numerical threshold is weakened. The
six-balanced-pair powered controller must also bind the smaller-cache full
fit receipt. No promotion lease is requested before these prerequisites.
See [MTP_TO_PROD](../../docs/MTP_TO_PROD.md) for disjoint ms budgets, the
conditional FP16 lever and why60 ms rounds alone do not guarantee60 TPS.

.51 was cleaned/released at **2026-10-16T22:41:00Z**: owned root absent
twice, no owned executable, test ports closed, governor absence unchanged.
The exact326-file candidate release closure is retained locally and hash
verified; no preparation artifact or active lease authorizes production use.
