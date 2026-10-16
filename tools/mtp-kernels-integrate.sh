#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p integrated
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
cd src
sha256sum vulkan/fg_small*.spv vulkan/fg_gdn_small_cached.spv > ../integrated/spv-before.txt
cd ..
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../integrated/build.txt 2>&1
sha256sum vulkan/fg_small*.spv vulkan/fg_gdn_small_cached.spv > ../integrated/spv-after.txt
cmp ../integrated/spv-before.txt ../integrated/spv-after.txt
cd ..
active=()
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> integrated/restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "integrated/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf integrated.tar.gz integrated
    exit "$rc"
}
trap finish EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
FG_SMALL_KERNELS=330 DS4_REMOTE_TEST_FILTER=expert_small_component \
    ./tests/test_fg_vk > ../integrated/component-f330.txt 2>&1
for flags in 0 330; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../integrated/oracles-f$flags.jsonl" 2> "../integrated/oracles-f$flags.stderr"
done
cd ..
