#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p paired-layout paired-mixed paired-combined-long round7b
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "paired-layout/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in paired-layout/*.stderr paired-mixed/*.stderr paired-combined-long/*.stderr; do
            if [ -s "$f" ]; then echo "=== $f"; tail -3 "$f"; fi
        done
    fi
    tar -czf paired-final.tar.gz paired-layout paired-mixed paired-combined-long evidence round7 round7b
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan FG_EXPERT_Q8_WIDE=1
sha256sum tools/bench-decode-fusion tests/test_fg_vk > ../paired-layout/binary-sha256.txt
expected=../round7/binary-sha256.txt
if [ -f ../round7b/binary-sha256.txt ]; then expected=../round7b/binary-sha256.txt; fi
cmp "$expected" ../paired-layout/binary-sha256.txt
run(){
    local flags=$1 n=$2 prefix=$3 seed=$4 path=$5
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 "${mix[@]}" --small-n --no-clocks \
        --seed "$seed" --batch "$n" --warmup "$((prefix/n))" \
        --iterations "$((72/n))" --windows 3 > "$path.jsonl" 2> "$path.stderr"
}
for kind in layout mixed combined-long; do
    prefix=264;mix=()
    if [ "$kind" = layout ]; then
        variants=(2097482 2105674 2621770 2130250)
    else
        mix=(--mixed-experts)
        if [ "$kind" = mixed ]; then
            variants=(4194634 2097482 6291786)
        else
            prefix=4096
            variants=(6291786 14684490)
        fi
    fi
    cp ../paired-layout/binary-sha256.txt "../paired-$kind/binary-sha256.txt.copy"
    FG_SMALL_KERNELS=330 ./tools/bench-decode-fusion --q4 "${mix[@]}" --ledger-only \
        > "../paired-$kind/ledger.jsonl"
    for round in 1 2 3 4 5; do
        seed=$((810+round));order=("${variants[@]}");ns=(2 4)
        if ((round%2==0)); then
            order=()
            for ((i=${#variants[@]}-1;i>=0;i--)); do order+=("${variants[i]}"); done
            ns=(4 2)
        fi
        run 0 1 "$prefix" "$seed" "../paired-$kind/r$round-n1-before-f0"
        for n in "${ns[@]}"; do
            index=0
            run 330 "$n" "$prefix" "$seed" "../paired-$kind/r$round-n$n-a$index-f330"
            for flags in "${order[@]}"; do
                run "$flags" "$n" "$prefix" "$seed" "../paired-$kind/r$round-n$n-b$index-f$flags"
                index=$((index+1))
                run 330 "$n" "$prefix" "$seed" "../paired-$kind/r$round-n$n-a$index-f330"
            done
        done
        run 0 1 "$prefix" "$seed" "../paired-$kind/r$round-n1-after-f0"
    done
    best=$(python3 - "../paired-$kind" <<'PY'
import collections,json,pathlib,re,statistics,sys
root=pathlib.Path(sys.argv[1]);gains=collections.defaultdict(list)
def med(p):
    return statistics.median(r["wall_us"] for s in p.read_text().splitlines()
        if s.startswith("{") and (r:=json.loads(s)).get("kind")=="sample")
for path in root.glob("r*-n*-b*-f*.jsonl"):
    match=re.fullmatch(r"r(\d+)-n(\d+)-b(\d+)-f(\d+)\.jsonl",path.name)
    if match is None:continue
    r,n,i,f=map(int,match.groups())
    a=(med(root/f"r{r}-n{n}-a{i}-f330.jsonl")+
       med(root/f"r{r}-n{n}-a{i+1}-f330.jsonl"))/2
    gains[f].append(1-med(path)/a)
print(max(gains,key=lambda f:statistics.mean(gains[f])))
PY
)
    printf '%s\n' "$best" > "../paired-$kind/selected-for-profile.txt"
    for round in 1 2 3 4 5; do
        seed=$((810+round))
        for n in 1 2 4; do
            flags_list=(330 "$best")
            if [ "$n" = 1 ]; then flags_list=(0); fi
            for flags in "${flags_list[@]}"; do
                FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 "${mix[@]}" --small-n --no-clocks \
                    --seed "$seed" --batch "$n" --warmup "$((prefix/n))" --iterations 1 --windows 1 --profile \
                    --routing-trace "../paired-$kind/routes-r$round-n$n-f$flags.jsonl" \
                    > "../paired-$kind/profile-r$round-n$n-f$flags.jsonl" \
                    2> "../paired-$kind/profile-r$round-n$n-f$flags.stderr"
            done
        done
    done
    if [ "$kind" = layout ]; then
        for round in 1 2 3 4 5; do
            for shape in "2560 10240 12" "2560 6144 12" "6144 2560 12" "2560 12288 12" \
                "10240 320 12" "320 10240 20" "2560 640 12"; do
                read -r k m t <<< "$shape"
                for phase in a0 b a1; do
                    flags=330;if [ "$phase" = b ]; then flags=$best; fi
                    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --no-clocks --dense-only \
                        --dense-type "$t" --dense-shape "$k" "$m" --allow-reassociation \
                        > "../paired-layout/shape-r$round-k$k-m$m-t$t-$phase-f$flags.jsonl" \
                        2> "../paired-layout/shape-r$round-k$k-m$m-t$t-$phase-f$flags.stderr"
                done
            done
        done
    fi
done
