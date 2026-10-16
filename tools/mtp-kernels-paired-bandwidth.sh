#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p paired-short paired-long
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
            -fsS "http://$host:$port/health" > "paired-short/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in paired-*/*.stderr; do
            if [ -s "$f" ]; then echo "=== $f"; tail -3 "$f"; fi
        done
    fi
    tar -czf paired-bandwidth.tar.gz paired-short paired-long evidence round6/binary-sha256.txt
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
sha256sum tools/bench-decode-fusion tests/test_fg_vk > ../paired-short/binary-sha256.txt
cmp ../round6/binary-sha256.txt ../paired-short/binary-sha256.txt
run(){
    local flags=$1 n=$2 prefix=$3 seed=$4 path=$5
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --seed "$seed" --batch "$n" --warmup "$((prefix/n))" \
        --iterations "$((72/n))" --windows 3 > "$path.jsonl" 2> "$path.stderr"
}
for kind in short long; do
    if [ "$kind" = short ]; then
        prefix=264
        variants=(2378 16714 262474 8522 33098 65866 524618 532810)
    else
        prefix=4096
        variants=(4426 131402 1053002 1179978)
    fi
    for round in 1 2 3 4 5; do
        seed=$((700+round))
        order=("${variants[@]}")
        ns=(2 4)
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
done
