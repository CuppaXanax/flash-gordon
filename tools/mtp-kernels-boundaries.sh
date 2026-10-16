#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p boundaries
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../boundaries/build.txt 2>&1
cd ..
active=()
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> boundaries/restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "boundaries/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf boundaries.tar.gz boundaries
    exit "$rc"
}
trap finish EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
for n in 1 2 3 4; do
    FG_ROOFLINE=0 FG_SMALL_KERNELS=0 FG_SHADER_DIR=../baseline/vulkan ../baseline/bench \
        --q4 --small-n --no-clocks --batch "$n" --warmup 3 --iterations 1 --windows 1 \
        > "../boundaries/default-old-n$n.jsonl" 2> "../boundaries/default-old-n$n.stderr"
    FG_ROOFLINE=0 FG_SMALL_KERNELS=0 ./tools/bench-decode-fusion \
        --q4 --small-n --no-clocks --batch "$n" --warmup 3 --iterations 1 --windows 1 \
        > "../boundaries/default-new-n$n.jsonl" 2> "../boundaries/default-new-n$n.stderr"
done
for prefix in 3 255 256 2047 2048 2049 2050 2051 2052; do
    FG_SMALL_KERNELS=10 ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 1 --warmup "$((prefix-1))" --iterations 1 --windows 1 \
        --small-test --allow-reassociation \
        > "../boundaries/prefix-$prefix.jsonl" 2> "../boundaries/prefix-$prefix.stderr"
done
cd ..
