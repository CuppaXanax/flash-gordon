#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
test -f fp16.complete
: > fp16-real-wire-oracle.jsonl
for policy in fp16 fp16-128 mixed-first-f32 mixed-no-rank0 mixed-screened single-0 single-1 single-2 single-3 single-4 single-5 single-6 single-7; do
    mode=1 mask=255
    [[ "$policy" != fp16-128 ]] || mode=128
    [[ "$policy" != mixed-first-f32 ]] || mask=253
    [[ "$policy" != mixed-no-rank0 ]] || mask=254
    [[ "$policy" != mixed-screened ]] || mask=$(python3 -c 'import json;print(json.load(open("fp16-screened-policy.json"))["mask"])')
    [[ "$policy" != single-* ]] || mask=$((1 << ${policy#single-}))
    input="$PWD/pack/input" dirty=false
    while read -r first last rank; do
        if ((mask & (1<<rank))); then
            hardware/tools/fp16-wire-oracle pack/manifest.fgm pack "$first" "$input" "$mode" \
                >> fp16-real-wire-oracle.jsonl
        fi
        if ((mask & (1<<rank))) || "$dirty"; then
            dirty=true; input="$PWD/pack/$policy-stage-$first"
        else
            input="$PWD/pack/f32-stage-$first"
        fi
    done < stages.txt
done
python3 - <<'PY'
import json
rows=[json.loads(s) for s in open("fp16-real-wire-oracle.jsonl")]
mask=json.load(open("fp16-screened-policy.json"))["mask"]
sites=38+mask.bit_count()
assert len(rows)==sites and sum(r["rows"] for r in rows)==sites*320
assert all(r["wire_and_decode_bit_exact"] for r in rows)
print(f"All{sites*320} authentic work vectors: optimized/portable wire and decoded f32 bit-exact")
PY
sha256sum hardware/tools/fp16-wire-oracle fp16-real-wire-oracle.jsonl
