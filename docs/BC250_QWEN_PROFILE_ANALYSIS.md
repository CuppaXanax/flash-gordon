# BC250/Qwen service profile

This suite measures the BC250 fleet as the Qwen 3.8 runtime actually uses it.
It does not benchmark a substitute workload and it does not assert a TPS
target.

## Collection

Build the standalone roofline and resident-QSA tools, then run them once per
rank with their stdout captured as JSONL.  For a live pipeline run, set
`FG_PIPELINE_PROFILE_SEQUENCE` to one activation sequence on every rank.  The
stage executor emits one `stage_service` JSON record plus `kernel_service`
records.  Set `FG_FABRIC_PROFILE=1` to emit `fabric_service` records for the
real direct or io_uring send/receive path.  Redirect stderr separately from
ordinary human logs and combine only the JSONL files for analysis:

```text
python3 tools/analyze_fg_profile.py rank-*.jsonl --metadata run.json > analysis.json
```

The analyzer requires rank IDs 0 through 7.  `--allow-partial` is useful while
debugging one rank, but marks the report partial and withholds complete-fleet
stage bounds.  Synthetic records belong in tests only, never in a dogfood
report.

## What is measured

Stage service is timed around the activation callback, including the boundary
tensor write/read, while GPU timestamps cover only the profiled Vulkan work.
The stage report groups only equal execution kind, token batch, and sequence
family at the same `first_token`; different context positions are never
averaged together. Failed stage/fabric records remain available for diagnosis
but are excluded from performance calculations. Prefill's service upper bound is:

```text
tokens / max(stage service seconds)
```

for complete comparable stage coverage.  Pipeline fill/TTFT service is the
sum of representative stage service.  A perfectly balanced lower bound is
`sum(stage service) / stage count`; `max / mean` and its complement expose
mathematical imbalance headroom.  These are service bounds, not achieved
end-to-end TPS.

Decode has an autoregressive critical path.  The report keeps the sum of
stage service per comparable sequence separate from the steady-state
max-stage throughput.  It never sums bandwidth from independent ranks into a
token rate.

Fabric rows report payload and framed GB/s by class, direction, transfer mode,
and payload size, plus receive wait/header/payload/validation splits when
available.  A fabric share of stage wall is `null` unless records carry a safe
shared request correlation key; no correlation is inferred from timestamps.

The Qwen wire ledger in `include/fg_ledger.h` derives pipeline boundary,
activation, decode-work, expert-result, and prefill-work/result sizes from the
protocol constants.  It is overflow-checked and CPU-tested.  This keeps
communication traffic accounting tied to the bytes the runtime encodes.

`counterfactual` reports ideal stage balance and, when expert kernel scopes are
present, an ideal balance of those measured scopes by rank.  It is explicitly
counterfactual: it is not a placement recommendation or an expected speedup.
Per-expert placement, overlap, queue contention, memory residency, and true
end-to-end saturation remain unknown until a controlled fleet trace supplies
those correlations.
