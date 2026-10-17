#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-reconcile
[[ "$(hostname -I)" == *192.168.42.50* ]]
prlimit --pid "$$" --memlock=268435456:268435456
export TMPDIR="$PWD/build" XDG_CACHE_HOME="$PWD/cache"
export FG_SHADER_DIR="$PWD/source/vulkan" FG_BF16_HOPS=0 FG_FP16_HOPS=0
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
        for i in $(seq 1 40); do
            if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" >/dev/null 2>&1; then
                healthy=true; break
            fi
            sleep 2
        done
        "$healthy" || status=1
    done
    sha256sum -c evidence/governor-before.sha256 || status=1
    return "$status"
}
trap restore EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
profile(){
    export FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024
    export FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 FG_SMALL_KERNELS=330
    export FG_EXPERT_Q8_WIDE=1 FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2
    if [[ "$1" == legacy ]]; then
        export FG_ROOFLINE=0 FG_ROOF_DENSE=0 FG_ROOF_QSA=0 FG_ROOF_EXPERT=0
        export FG_ROOF_HC=0 FG_ROOF_RECURRENT=0 FG_ROOF_GDN=0 FG_SMALL_KERNELS=0
        export FG_EXPERT_Q8_WIDE=0 FG_NGRAM_LOCAL_CACHE=0 FG_DECODE_SUBMIT_GROUP=0
    fi
}
profile current
if [[ ! -f prepared.ready ]]; then
    for i in $(seq 1 360); do
        [[ ! -f ngram.ready ]] || break
        sleep 10
    done
    test -f ngram.ready
    ln -sfn rank-00.mixed.fgw pack/rank-00.fgw
    source/tools/bf16-offline prepare pack/manifest.fgm pack pack/public-header.gguf prompts.fgev \
        > evidence/prepare.jsonl 2> evidence/prepare.stderr
    sha256sum pack/cases.bin pack/input.teacher pack/input.candidate pack/ngram.f32 \
        > evidence/prepared.sha256
    python3 - <<'PY'
from pathlib import Path
import hashlib,json,struct
raw=Path('pack/cases.bin').read_bytes()
count=struct.unpack_from('<I',raw,8)[0];offset=12;rows=[]
corpus=json.loads(Path('corpus.json').read_text())
assert count==len(corpus)==560
for key,text in corpus:
    n=struct.unpack_from('<I',raw,offset)[0];offset+=4
    ids=raw[offset:offset+4*n];offset+=4*n
    rows.append(dict(id=key,tokens=n,final_position=n-1,
        token_ids=list(struct.unpack('<'+'I'*n,ids)),
        tokens_sha256=hashlib.sha256(ids).hexdigest(),
        text_sha256=hashlib.sha256(text.encode()).hexdigest()))
assert offset==len(raw)
Path('evidence/positions.json').write_text(json.dumps(rows,indent=2)+'\n')
expected=json.loads(Path('expected-prefix80.json').read_text())
assert hashlib.sha256(raw[12:12+expected['cases_bytes']]).hexdigest()==expected['cases_sha256']
for name in ('input.teacher','ngram.f32'):
    with Path('pack',name).open('rb') as f:
        assert hashlib.sha256(f.read(expected[name]['bytes'])).hexdigest()==expected[name]['sha256']
print('First80 prepared tokens/embeddings/ngrams reproduce original frozen-decode inputs exactly')
PY
    rm pack/ngram.iq4nl
    source/tools/bf16-offline plan pack/manifest.fgm > stages.txt
    touch prepared.ready
fi
while read -r first last rank; do
    [[ -f "stage-$first.complete" ]] && continue
    mixed="pack/rank-$(printf '%02d' "$rank").mixed.fgw"
    test -f "$mixed"
    digest=$(awk -v r="$rank" '$1==r {print $2}' expected-ranks.txt | tr -d '\r')
    echo "$digest  $mixed" | sha256sum -c -
    for weights in mixed q8; do
        manifest=pack/manifest.fgm
        if [[ "$weights" == q8 ]]; then
            manifest=original-q8.fgm
            proof="evidence/q8-rank-$rank-restored.jsonl"
            if [[ ! -f "$proof" ]]; then
                rm -f "pack/rank-$rank.q8.fgw" "pack/rank-$rank.q8.raw"
                python3 source/tools/fp16_q8_restore.py --original original-q8.fgm \
                    --mixed pack/manifest.fgm --dense original-dense-q8.gguf \
                    --source "$mixed" --output "pack/rank-$rank.q8.fgw" \
                    --placer source/tools/mtp-real-place --rank "$rank" > "$proof.incomplete"
                mv "$proof.incomplete" "$proof"
            else
                digest=$(python3 -c 'import json,sys;print(json.loads(open(sys.argv[1]).readlines()[-1])["file_sha256"])' "$proof")
                echo "$digest  pack/rank-$rank.q8.fgw" | sha256sum -c -
            fi
            ln -sfn "rank-$rank.q8.fgw" "pack/rank-$(printf '%02d' "$rank").fgw"
        else
            ln -sfn "$(basename "$mixed")" "pack/rank-$(printf '%02d' "$rank").fgw"
        fi
        for method in decode prefill legacy; do
            profile current
            export HOP_OFFLINE_PATH="$method" HOP_OFFLINE_CODEC=f32
            if [[ "$method" == legacy ]]; then
                profile legacy
                export HOP_OFFLINE_PATH=prefill
            fi
            policy="$weights-$method"
            [[ ! -f "$policy-stage-$first.complete" ]] || continue
            env | LC_ALL=C sort | grep -E '^(FG_|HOP_OFFLINE_)' > "evidence/$policy-environment.txt"
            input="pack/$policy-current"
            if [[ "$first" == 0 ]]; then input=pack/input; fi
            output="pack/$policy-next"
            rm -f "$output.teacher" "$output.candidate" "pack/qsa-$first-0.state" "pack/qsa-$first-1.state"
            source/tools/bf16-offline stage "$manifest" pack "$first" "$last" "$input" "$output" \
                > "evidence/$policy-stage-$first.jsonl" 2> "evidence/$policy-stage-$first.stderr"
            sha256sum "$output.teacher" "$output.candidate" > "evidence/$policy-stage-$first.sha256"
            mv "$output.teacher" "pack/$policy-current.teacher"
            mv "$output.candidate" "pack/$policy-current.candidate"
            touch "$policy-stage-$first.complete"
            if [[ "$last" == 47 ]]; then
                # The trained head belongs to rank4, retained below.
                echo "$policy ready for head"
            fi
        done
        if [[ "$rank" != 4 ]]; then
            [[ "$weights" != q8 ]] || rm "pack/rank-$rank.q8.fgw"
        fi
    done
    if [[ "$rank" != 4 ]]; then
        rm "$mixed" "pack/rank-$(printf '%02d' "$rank").fgw"
    fi
    touch "stage-$first.complete"
    echo "STAGE_COMPLETE first=$first last=$last rank=$rank"
    break
done < stages.txt
if [[ -f stage-42.complete ]]; then
    for weights in mixed q8; do
        manifest=pack/manifest.fgm
        target=rank-04.mixed.fgw
        if [[ "$weights" == q8 ]]; then manifest=original-q8.fgm; target=rank-4.q8.fgw; fi
        ln -sfn "$target" pack/rank-04.fgw
        for method in decode prefill legacy; do
            policy="$weights-$method";profile current
            [[ "$method" != legacy ]] || profile legacy
            mkdir -p "logits/$policy"
            HOP_OFFLINE_LOGITS="$PWD/logits/$policy" source/tools/bf16-offline head \
                "$manifest" pack "pack/$policy-current" \
                > "evidence/$policy-quality.jsonl" 2> "evidence/$policy-quality.stderr"
            sha256sum "logits/$policy/"*.fgl > "evidence/$policy-head.sha256"
        done
    done
    sha256sum source.tar patch.tar controls.tar > evidence/source-archives.sha256
    tar -czf reconcile-evidence.tar.gz evidence corpus.json prompts.fgev patch.tar controls.tar
    tar -czf reconcile-logits.tar.gz logits
    sha256sum reconcile-evidence.tar.gz reconcile-logits.tar.gz
    touch complete
fi
