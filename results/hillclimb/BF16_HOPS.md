# BF16 decode-ring work hops

**Quality gate FAILED. Research-only, default off; no serving promotion.**
The measured transport saving does not override the trained-model quality gate.

The follow-up [scaled FP16 experiment](FP16_HOPS.md) keeps ten fraction bits
with per-hop or block scales and reports cumulative original-Q8 comparisons.

## Scope and safety

`FG_BF16_HOPS=1` enables round-to-nearest-even BF16 **at the send boundary**
for protocol-6 single-token and depth-B decode work. Hyper vectors and the initial
n-gram embedding are compressed; receivers expand them to f32 before computation.
All shader arithmetic, accumulation, prefill, output/head transfers, expert
results, native-head traffic and MTP verification remain unchanged f32.
Absent, `0`, or malformed values retain the original f32 wire format.
The wire marker is self-describing, so upgraded receivers can accept either mode
independently of their local send setting. Old receivers reject BF16 work.

MTP plus BF16 is deliberately rejected before readiness: MTP verification uses
the unchanged prefill codec, and mixing that unrounded path with rounded ordinary
decode/repair is not qualified. This is not an MTP greedy-equivalence result.
The separate `ac24b26` N2/N4 research variants were handed to sol-mtp with the
model-KL and MTP-greedy quality gates; they are not enabled here.

Only `.52`/`.53` were used for transport and `.50` for offline quality.
No production rank, serving reference, production model file, residency cap,
governor setting or kernel was changed.

## Five-round actual-codec transport

The benchmark uses actual `fg_fabric` io_uring bulk transport, protocol framing,
CRC and decode-work encode/decode. Rank 0 on `.52` exchanges work with rank 1 on
`.53`; the remaining six lightweight mesh endpoints are idle. Every returned
message is checked against its expected decoded vector, request and sequence.
These validation costs are included. This is **RTT/2**, not synchronized
one-way timing.

Five alternating A/B/A and B/A/B rounds, 500 measured RTTs and 50 warmups per leg;
ordinary/initial shape order also alternates. Each round averages the two outer
p50 RTTs and compares the center p50; divide by two for the hop estimate.
Intervals are Student-t 95% intervals over the five paired round observations,
not over 2,500 correlated packet samples.

| Decode work | f32 hop, µs | BF16 hop, µs | Paired saving, µs |
|---|---:|---:|---:|
| Ordinary, 10,240 values | 564.114 ± 2.594 | 418.841 ± 3.989 | 145.272 ± 2.204 |
| Initial + n-gram, 12,800 values | 649.819 ± 5.843 | 499.031 ± 0.442 | 150.788 ± 5.854 |
| Eight-work-hop model/token | 4,598.617 ± 13.123 | 3,430.922 ± 28.191 | **1,167.695 ± 18.281** |

The model is **one initial + seven ordinary** sequential work transfers, pairing
all eight terms within each round before forming its CI. Final output/head and
all other transfers are excluded. It is not measured eight-rank serving TPS,
and cannot be added to an independently measured overlap/pipeline speedup.

Ordinary frame bytes: 41,016 → 20,536; initial: 51,256 → 25,656.
Eight work frames: 338,368 → 169,408 bytes. Their raw 1-Gb/s serialization saving
is 1.35168 ms/token; actual-codec measurements model **1.168 ± 0.018 ms/token**.
The older `LATENCY.md` BF16-sized opaque-payload estimate is not a measurement
of this codec, and its absolute hop timings are not directly interchangeable.

## Offline real-pack quality

The authenticated all-48-layer evaluation completed on `.50`:

| Cohort | Positions | Mean KL | Maximum KL | Top-1 agreement | Gate |
|---|---:|---:|---:|---:|---|
| Primary: final position/context | 80 | 0.007226893 | 0.096033114 | 78/80 = 97.50% | **FAIL both** |
| Auxiliary: four trailing positions/context | 320 | 0.005065367 | 0.119094659 | 316/320 = 98.75% | **FAIL KL** |

The predeclared primary thresholds remain mean KL ≤ 0.005 and top-1 ≥ 98.5%;
neither a transport win nor the auxiliary top-1 result qualifies this variant.
Primary mismatches are case 25/position 35 (teacher token 11, BF16 token 248046)
and case 71/position 23 (248046 → 15346). Auxiliary mismatches additionally occur
at case 26/position 35 and case 75/position 22. Case indices are zero-based.
No thresholds, corpus, logits or cohort selection were changed after observation.

The evaluator uses the sealed mix-04 manifest
`0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252`.
It reconstructs every trained tensor from pinned public GGUF revision
`7e3bfc316b880fefeb049596f11c49d6a18e05fb` plus exact dense mix-04 overlay
`e742dbf593f03c5c608de229329755e7cb19138e7b1c4031632eaf49f1acf4d1`.
The existing cooker must SHA-match each cooked span to the sealed manifest;
the real loader re-verifies spans. No trained weight is requantized.

The first public GGUF shard contains **zero tensors but all tokenizer metadata**:
10,946,752 bytes, SHA
`290a31ca20b74ed2c21b257d378a5678d821ad9fdd8feb1fc52c719761668903`.
Its packed tokenizer is 8,949,760 bytes and must match sealed SHA
`b291b98670183858d414610cfe979e77a67ca53a1b0e977ed568548f8558c820`.
The complete 28,800,138,240-byte n-gram trained span matches
`dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3`;
its physical file is zero-padded to 4-KiB alignment for the existing store API.

The frozen Q4 corpus supplies 80 textual contexts (1,808 supplied tokens).
Separate causal owner sessions
execute the common prefix in f32, then four supplied trailing tokens with f32
versus BF16 at each of the eight real work boundaries:
`0→1→0→2→3→4→5→6→7`. The candidate round-trips the actual wire codec.
All 48 layers, trained routed experts, n-gram conditioning and the real output
HC/head execute on Vulkan, one rank at a time. Full 248,320-vocabulary logits
produce KL(P_f32 || P_bf16) and top-1 agreement. The primary gate is 80 final
positions; the 320 trailing positions are additionally reported.
Thresholds: mean KL ≤ 0.005 and top-1 ≥ 98.5%.
Common-prefix component tolerance is 1e-4 relative, not bit-exactness.
All eight common-prefix checks passed with observed maximum relative error zero.
All 1,223 trained spans passed source/loader authentication; the eight stages
each applied 320 rounded work boundaries (2,560 total).
The evaluator provisions microbatch-2 scratch (decode schedules require nine
words/pair versus five in prefill scratch), eagerly creates independent QSA
state files, and retains the entire 512-token owned-layer window in cache.
Rank 0 uses an owned-only coordinator mirror, with no remote/cold-page dependency.
These bounded evaluation allocations do not qualify production dual-session
residency, production prefill equivalence, or real-pack serving throughput.

## Validation and reproduction

Focused codec tests cover BF16 RNE ties, signed zero, subnormals, overflow,
nonfinite rejection, unchanged f32 bytes, text/four-axis layouts, optional
n-gram, all batch slot counts, mixed local settings, malformed/truncated frames
and the BF16/MTP interlock. Existing decode-batch tests and the real
eight-process/two-session fabric pipeline pass with f32 and BF16 enabled.
`make test-bf16-hops` also runs four Python tests covering paired arithmetic, quality thresholds, malformed
observations and the metadata-only tokenizer shard. Flag registration passes.

The full pre-existing fabric selfcheck fails at its output-config reserved-byte
check on both this worktree and a pristine `ac24b26` archive. That unrelated
baseline failure is disclosed, not fixed or counted as a passing full suite.

Use `tools/Invoke-BF16Fleet.ps1` only on its explicit `.50/.52/.53` allowlist.
It replaces the legacy runner's temporary staging with owned
`/home/xander/hc-bf16` staging. `bf16-build.sh` and `bf16-hop-run.sh` build/run
the pair; use `-Sudo` for its bounded temporary firewall rules.
`bf16_source.py` constructs authenticated rank sources; `bf16-tokenizer-source.sh`
preflights the tokenizer without loading GPU weights. `bf16-offline-run.sh`
runs with `-Sudo` for reversible service control and a process-local 256-MiB
memlock allowance for existing registered loader buffers. Failed attempts
restored both original `.50` services before retrying. No global memlock,
governor or residency-policy bypass is used.

## Evidence and reproducibility

`bf16-hops.json` contains every transport leg, paired round, full-vocabulary
quality observation and both logit-vector hashes. Reproduce it from the raw
receipt and extracted `.50` archive:

```powershell
python tools\analyze_bf16_hops.py --pair receipts\hop-pair-run.json `
  --quality quality-50\quality.jsonl --output bf16-hops.json
```

The definitive stage/head executable SHA256 is
`1931948e68d4a09c621e8e6388d3d1585409c0afd89c6b71ce7560f0910988d0`.
Prepared inputs were generated by
`8715051c73aeeafda8b269b296d2094afcbad3a8e94668a41d994d51755057cd`;
subsequent fixes changed stage scratch/cache setup, not preparation arithmetic.
Both binary identities and prepared-input hashes are in the evidence.
The actual measured hop executable was
`c7474f7e06e4c2b1fee97d899796b37a6bd5c9c9c709c8353d4994bf9781dc98`.
Source/shader hash lists and exact measured codec/benchmark sources are retained.

| Evidence archive | SHA256 |
|---|---|
| `bf16-evidence-50.tar.gz` | `1c2a6e91d358f788fa1b62aa8c2b0976b72b165e8b6097cc439ff299d40cf89f` |
| `bf16-evidence-52.tar.gz` | `dafdb3e5dbdcf0d761b48112c00b030f821e98f540092b76396ce505610703d3` |
| `bf16-evidence-53.tar.gz` | `0c808bdd1c4a429d0bdf94015ef4ff9bcced3d875511a49cee1a33e8faf8ded8` |
| `bf16-receipts.tar.gz` | `300b40745e5328ceac30f2b8981857f2e00e364e026ccc65dac7387d0e5c06af` |

The archives contain bounded logs, source/weight identity receipts and quality
observations, not trained weight payloads. Authenticated reconstructed rank
packs, the complete overlay and metadata are preserved locally, ignored by Git,
for sol-mtp's independent offline diagnosis. Rank-file SHA identities describe
reconstructed files with cooker-generated padding, not unavailable original FGWs.

`bf16-receipts.tar.gz` retains the raw paired recording, build/run/cleanup
receipts, focused validation logs, baseline-failure evidence and eight file-only
export receipts. Extract it before using the reproduction command above.

## Cleanup and release

The pair's owned `/home/xander/hc-bf16` roots were removed at
2026-10-16 16:55:08 UTC; temporary ports 19800–19815 were closed, original
inactive service states retained, and the original `.53` overlay preserved.
The `.50` root was removed at 17:52:04 UTC after exporting all evidence and all
eight reconstructed rank files. Both original `.50` services were active and
their health endpoints returned `status: ok`.
Each cleanup verified no owned numerical/transport process, unchanged governor
configuration/performance-level identities, and root absence twice.
No fleet helper was invoked on a host after its final cleanup.

All eight local retained packs were independently rehashed after download
(81,245,085,696 bytes total); the final hashes match the remote rank manifest.
Their local retention JSON and exact metadata/overlay remain available to
sol-mtp. No weight payload or production change is included in this branch.
