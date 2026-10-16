#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p paired-profile
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
active=()
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> paired-profile/restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "paired-profile/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf paired-profile.tar.gz paired-profile
    exit "$rc"
}
trap finish EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -j8 tests/test_fg_vk > ../paired-profile/build.txt 2>&1
FG_SMALL_KERNELS=330 DS4_REMOTE_TEST_FILTER=expert_small_component \
    ./tests/test_fg_vk > ../paired-profile/component-f330.txt 2>&1
for round in 1 2 3 4 5; do
    for phase in before candidate after; do
        flags=0
        if [ "$phase" = candidate ]; then flags=330; fi
        FG_ROOFLINE=0 FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 \
            --small-n --no-clocks --batch 4 --warmup 66 --seed "$round" \
            --iterations 1 --windows 1 --profile \
            > "../paired-profile/r$round-$phase.jsonl" 2> "../paired-profile/r$round-$phase.stderr"
    done
done
cd ..
