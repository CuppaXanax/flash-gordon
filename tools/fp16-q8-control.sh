#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
[[ "$(hostname -I)" == *192.168.42.50* ]]
test -f fp16-mixed.complete
test -f fp16-evidence.complete
[[ "$(sha256sum source/tools/bf16-offline | cut -d' ' -f1)" == 11ce46e6cb662c4d70509f28301c50a1bb3ff1dd9be49b1516cd9cd453f57239 ]]
prlimit --pid "$$" --memlock=268435456:268435456
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
export FG_SHADER_DIR="$PWD/source/vulkan"
export FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024
export FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 FG_SMALL_KERNELS=330
export FG_EXPERT_Q8_WIDE=1 FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2
export FG_BF16_HOPS=0 FG_FP16_HOPS=0 HOP_OFFLINE_CODEC=f32
unset HOP_OFFLINE_TEACHER HOP_OFFLINE_MASK
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
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
make -C hardware -j12 tools/mtp-real-place > q8-build.log 2>&1
sha256sum hardware/tools/mtp-real-place hardware/tools/fp16_q8_restore.py source/tools/bf16-offline > q8-source-binaries.sha256
input="$PWD/pack/input"
while read -r first last rank; do
    target="$PWD/pack/q8-rank-$(printf '%02d' "$rank").fgw"
    proof="$PWD/q8-rank-$rank-restored.jsonl"
    if [[ ! -f "$proof" ]]; then
        rm -f "$target"
        PYTHONPATH=hardware/tools python3 hardware/tools/fp16_q8_restore.py \
            --original "$PWD/original-q8.fgm" --mixed "$PWD/pack/manifest.fgm" \
            --dense "$PWD/original-dense-q8.gguf" \
            --source "$PWD/pack/rank-$(printf '%02d' "$rank").fgw" \
            --output "$target" --placer "$PWD/hardware/tools/mtp-real-place" --rank "$rank" \
            > "$proof.incomplete"
        mv "$proof.incomplete" "$proof"
        rm "$PWD/pack/rank-$(printf '%02d' "$rank").fgw"
    fi
    link="$PWD/pack/rank-$(printf '%02d' "$rank").fgw"
    [[ ! -e "$link" ]] || [[ "$(readlink "$link")" == "$(basename "$target")" ]]
    ln -sfn "$(basename "$target")" "$link"
    output="$PWD/pack/q8-stage-$first"
    if [[ ! -f "q8-stage-$first.sha256" ]]; then
        rm -f "$output.teacher" "$output.candidate" "pack/qsa-$first-0.state" "pack/qsa-$first-1.state"
        source/tools/bf16-offline stage original-q8.fgm pack "$first" "$last" "$input" "$output" \
            > "q8-stage-$first.jsonl" 2> "q8-stage-$first.stderr"
        sha256sum source/tools/bf16-offline "$output.teacher" "$output.candidate" > "q8-stage-$first.sha256"
    else
        sha256sum -c "q8-stage-$first.sha256"
    fi
    input="$output"
done < stages.txt
mkdir -p logits/q8
if [[ ! -f q8-head.sha256 ]]; then
    HOP_OFFLINE_LOGITS="$PWD/logits/q8" source/tools/bf16-offline head original-q8.fgm pack "$input" \
        > q8-quality.jsonl 2> q8-quality.stderr
    sha256sum source/tools/bf16-offline q8-quality.jsonl logits/q8/*.fgl > q8-head.sha256
else
    sha256sum -c q8-head.sha256
fi
tar -czf fp16-q8-control.tar.gz original-q8.fgm q8-*.jsonl q8-*.stderr \
    q8-*.sha256 q8-build.log logits/q8 hardware/tools/fp16_q8_restore.py
sha256sum fp16-q8-control.tar.gz
touch fp16-q8.complete
