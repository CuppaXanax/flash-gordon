#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
prlimit --pid "$$" --memlock=268435456:268435456
[[ "$(hostname -I)" == *192.168.42.50* ]]
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
export FG_SHADER_DIR="$PWD/source/vulkan"
export FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024
export FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 FG_SMALL_KERNELS=330
export FG_EXPERT_Q8_WIDE=1 FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2
export FG_BF16_HOPS=0
for i in $(seq 1 240); do
    if [[ -f pack/source-0.jsonl ]] && [[ $(wc -l < pack/source-0.jsonl) -eq 153 ]]; then break; fi
    [[ $i -lt 240 ]] || { echo 'initial source reconstruction timed out' >&2; exit 1; }
    sleep 30
done
prepared=false
if [[ -f prepared.sha256 ]]; then
    python3 - <<'PY'
import json
from pathlib import Path
proof={s.split("  ",1)[1] for s in Path("prepared.sha256").read_text().splitlines()}
assert {"pack/ngram.iq4nl","pack/cases.bin","pack/ngram.f32",
        "pack/input.teacher","pack/input.candidate","pack/tokenizer/tokenizer.fgt"}<=proof
assert json.loads(Path("prepare.jsonl").read_text().splitlines()[-1])==dict(
    kind="prepared_complete",cases=80,tokens=1808)
PY
    awk '$2!="pack/ngram.iq4nl"{print}' prepared.sha256 | sha256sum -c -
    prepared=true
else
    ngram_sha=$(head -c 28800138240 pack/ngram.iq4nl | sha256sum)
    [[ "${ngram_sha%% *}" == dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3 ]]
    python3 - <<'PY'
from pathlib import Path
p=Path("pack/ngram.iq4nl")
assert p.stat().st_size in (28800138240, (28800138240+4095)&~4095)
with p.open("r+b") as f:
    f.truncate((28800138240+4095)&~4095)
print("Full ngram trained span SHA exact; file zero-padded to sealed I/O alignment")
PY
fi
sha256sum /etc/cyan-skillfish-governor-smu/config.toml > governor-before.sha256
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
printf '%s\n' "${active[@]}" > services-before.txt
restore(){
    local status=$?
    set +e
    if ((status)); then
        for log in prepare.stderr stage-*.stderr quality.stderr rank*-source.log; do
            if [[ -f "$log" ]]; then echo "=== $log" >&2; tail -n 3 "$log" >&2; fi
        done
    fi
    for service in "${active[@]}"; do systemctl start "$service" || status=1; done
    sha256sum -c governor-before.sha256 || status=1
    for service in "${active[@]}"; do
        local port=8091 healthy=false
        [[ "$service" != bc250-rerank ]] || port=8092
        for i in $(seq 1 30); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" 2>/dev/null; then healthy=true; break; fi
            sleep 2
        done
        if ! "$healthy" || ! systemctl is-active --quiet "$service"; then
            echo "service restoration failed: $service" >&2
            status=1
        fi
    done
    return "$status"
}
trap restore EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
if ! "$prepared"; then
    source/tools/bf16-offline prepare pack/manifest.fgm pack pack/public-header.gguf prompts.fgev \
        > prepare.jsonl 2> prepare.stderr
    sha256sum source/tools/bf16-offline > prepared-binary.sha256
    sha256sum pack/ngram.iq4nl pack/cases.bin pack/ngram.f32 pack/input.teacher \
        pack/input.candidate pack/tokenizer/tokenizer.fgt prompts.fgev \
        pack/manifest.fgm pack/public-header.gguf > prepared.sha256
    rm pack/ngram.iq4nl
fi
source/tools/bf16-offline plan pack/manifest.fgm > stages.txt
input="$PWD/pack/input"
while read -r first last rank; do
    if [[ ! -f "pack/source-$rank.jsonl" ]]; then
        python3 source/tools/bf16_source.py "$rank" --manifest pack/manifest.fgm \
            --directory pack --overlay overlay.gguf --placer source/tools/mtp-real-place \
            > "rank$rank-source.log" 2>&1
    fi
    output="$PWD/pack/stage-$first"
    if [[ -f "stage-$first.sha256" ]]; then
        sha256sum -c "stage-$first.sha256"
    else
        rm -f "$output.teacher" "$output.candidate" "pack/qsa-$first-0.state" "pack/qsa-$first-1.state"
        source/tools/bf16-offline stage pack/manifest.fgm pack "$first" "$last" "$input" "$output" \
            > "stage-$first.jsonl" 2> "stage-$first.stderr"
        sha256sum source/tools/bf16-offline "$output.teacher" "$output.candidate" \
            > "stage-$first.sha256"
    fi
    input="$output"
done < stages.txt
source/tools/bf16-offline head pack/manifest.fgm pack "$input" \
    > quality.jsonl 2> quality.stderr
sha256sum source/tools/bf16-offline source/tools/mtp-real-place \
    pack/manifest.fgm prompts.fgev quality.jsonl > offline-inputs.sha256
touch offline.complete
