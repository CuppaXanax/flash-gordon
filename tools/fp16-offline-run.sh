#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
prlimit --pid "$$" --memlock=268435456:268435456
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
export FG_SHADER_DIR="$PWD/source/vulkan"
export FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024
export FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 FG_SMALL_KERNELS=330
export FG_EXPERT_Q8_WIDE=1 FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2
export FG_BF16_HOPS=0 FG_FP16_HOPS=0
for i in $(seq 1 240); do
    ready=$(find pack -maxdepth 1 -name 'rank-*.fgw.verified' | wc -l)
    if [[ "$ready" == 8 && -f fp16-ngram.ready ]]; then break; fi
    [[ $i -lt 240 ]] || { echo 'waiting for immutable input timed out' >&2; exit 1; }
    sleep 30
done
sha256sum /etc/cyan-skillfish-governor-smu/config.toml > governor-before.sha256
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
printf '%s\n' "${active[@]}" > services-before.txt
restore(){
    local status=$?
    set +e
    for service in "${active[@]}"; do
        systemctl start "$service" || status=1
        local port=8091 healthy=false
        [[ "$service" != bc250-rerank ]] || port=8092
        for i in $(seq 1 30); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" 2>/dev/null; then healthy=true; break; fi
            sleep 2
        done
        "$healthy" || status=1
    done
    sha256sum -c governor-before.sha256 || status=1
    return "$status"
}
trap restore EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
if [[ ! -f fp16-prepared.sha256 ]]; then
    rm -f pack/cases.bin pack/ngram.f32 pack/input.teacher pack/input.candidate
    source/tools/bf16-offline prepare pack/manifest.fgm pack pack/public-header.gguf prompts.fgev \
        > fp16-prepare.jsonl 2> fp16-prepare.stderr
    sha256sum source/tools/bf16-offline pack/cases.bin pack/ngram.f32 \
        pack/input.teacher pack/input.candidate prompts.fgev > fp16-prepared.sha256
    rm pack/ngram.iq4nl
fi
sha256sum -c fp16-prepared.sha256
source/tools/bf16-offline plan pack/manifest.fgm > stages.txt
binary=$(sha256sum source/tools/bf16-offline)
policies=(f32 fp16 fp16-128 bf16 mixed-first-f32 mixed-no-rank0 mixed-screened single-0 single-1 single-2 single-3 single-4 single-5 single-6 single-7)
mkdir -p logits
for policy in "${policies[@]}"; do
    codec="$policy" mask=255 dirty=false
    if [[ "$policy" == single-* ]]; then codec=fp16; mask=$((1 << ${policy#single-})); fi
    if [[ "$policy" == mixed-first-f32 ]]; then codec=fp16; mask=253; fi
    if [[ "$policy" == mixed-no-rank0 ]]; then codec=fp16; mask=254; fi
    if [[ "$policy" == mixed-screened ]]; then
        codec=fp16
        mask=$(python3 - <<'PY'
import json
from pathlib import Path
mask=0;screen={}
for rank in range(8):
    rows=[json.loads(s) for s in Path(f"single-{rank}-quality.jsonl").read_text().splitlines()]
    final=[r for r in rows if r["final"]]
    assert len(rows)==320 and len(final)==80
    mean=sum(r["kl"] for r in final)/80
    top1=sum(r["teacher_top1"]==r["bf16_top1"] for r in final)/80
    selected=mean<=.005 and top1>=.985
    if selected:mask|=1<<rank
    screen[rank]=dict(kl_mean=mean,top1_agreement=top1,selected=selected)
Path("fp16-screened-policy.json").write_text(json.dumps(dict(
    mask=mask,selection=screen,method="In-corpus isolated-hop screen, not a holdout; combined policy is separately replayed."),indent=2)+"\n")
print(mask)
PY
)
    fi
    input="$PWD/pack/input"
    export HOP_OFFLINE_CODEC="$codec" HOP_OFFLINE_MASK="$mask"
    while read -r first last rank; do
        baseline="$PWD/pack/f32-stage-$first"
        if [[ "$policy" != f32 && "$dirty" == false ]] && ((!(mask & (1<<rank)))); then
            input="$baseline"
            continue
        fi
        dirty=true
        output="$PWD/pack/$policy-stage-$first"
        proof="$policy-stage-$first.sha256"
        if [[ -f "$proof" ]]; then
            [[ $(head -n 1 "$proof") == "$binary" ]]
            sha256sum -c "$proof"
        else
            rm -f "$output.teacher" "$output.candidate" "pack/qsa-$first-0.state" "pack/qsa-$first-1.state"
            if [[ "$policy" == f32 ]]; then unset HOP_OFFLINE_TEACHER
            else export HOP_OFFLINE_TEACHER="$baseline.teacher"; fi
            source/tools/bf16-offline stage pack/manifest.fgm pack "$first" "$last" "$input" "$output" \
                > "$policy-stage-$first.jsonl" 2> "$policy-stage-$first.stderr"
            sha256sum source/tools/bf16-offline "$output.teacher" "$output.candidate" > "$proof"
        fi
        input="$output"
    done < stages.txt
    mkdir -p "logits/$policy"
    if [[ ! -f "$policy-head.sha256" ]]; then
        rm -f "logits/$policy/"*.fgl
        HOP_OFFLINE_LOGITS="$PWD/logits/$policy" source/tools/bf16-offline head pack/manifest.fgm pack "$input" \
            > "$policy-quality.jsonl" 2> "$policy-quality.stderr"
        sha256sum source/tools/bf16-offline "$policy-quality.jsonl" "logits/$policy/"*.fgl > "$policy-head.sha256"
    else
        [[ $(head -n 1 "$policy-head.sha256") == "$binary" ]]
        sha256sum -c "$policy-head.sha256"
    fi
    echo "POLICY_COMPLETE $policy"
done
sha256sum source/tools/bf16-offline source/tools/bench-bf16-hops > fp16-binaries.sha256
touch offline.complete fp16.complete fp16-mixed.complete
