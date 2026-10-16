#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p src/.build .build cache evidence round3
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
if [ ! -f evidence/governor-original.toml ]; then
    cp /etc/cyan-skillfish-governor-smu/config.toml evidence/governor-original.toml
fi
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
if [ ! -f evidence/services-original.txt ]; then
    printf '%s\n' "${active[@]}" > evidence/services-original.txt
fi
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "round3/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf round3.tar.gz round3 evidence
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../round3/build.txt 2>&1
DS4_REMOTE_TEST_FILTER=qsa_attention_union ./tests/test_fg_vk > ../round3/qsa-cpu.txt 2>&1
for flags in 18698 4362 8458 30986; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../round3/oracles-f$flags.jsonl" 2> "../round3/oracles-f$flags.stderr"
done
for n in 2 3 4; do
    for flags in 266 2314 18698 8458 4362 30986; do
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
            --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
            > "../round3/profile-f$flags-n$n.jsonl" 2> "../round3/profile-f$flags-n$n.stderr"
    done
done
for flags in 266 4362; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 4 --warmup 1024 --iterations 1 --windows 1 --profile --small-test \
        --allow-reassociation > "../round3/long-f$flags.jsonl" 2> "../round3/long-f$flags.stderr"
done
RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=30986 \
    ./tools/bench-decode-fusion --q4 --small-n --no-clocks --batch 4 \
    --warmup 2 --iterations 1 --windows 1 > ../round3/isa.jsonl 2> ../round3/isa.stderr
cd ..
