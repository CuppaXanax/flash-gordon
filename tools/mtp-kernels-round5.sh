#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p src/.build .build cache evidence round5
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
            -fsS "http://$host:$port/health" > "round5/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in round5/*.txt round5/*.stderr; do echo "=== $f"; tail -5 "$f"; done
    fi
    tar -czf round5.tar.gz round5 evidence
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../round5/build.txt 2>&1
for flags in 0 131072; do
    FG_SMALL_KERNELS=$flags DS4_REMOTE_TEST_FILTER=qsa_attention_union \
        ./tests/test_fg_vk > "../round5/qsa-cpu-f$flags.txt" 2>&1
done
FG_SMALL_KERNELS=262410 DS4_REMOTE_TEST_FILTER=expert_small_component \
    ./tests/test_fg_vk > ../round5/expert-cpu.txt 2>&1
for flags in 262410 131330; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../round5/oracles-f$flags.jsonl" 2> "../round5/oracles-f$flags.stderr"
done
for n in 1 2 4; do
    for flags in 266 16650 262410; do
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
            --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
            --routing-trace "../round5/routes-f$flags-n$n.jsonl" \
            > "../round5/profile-f$flags-n$n.jsonl" 2> "../round5/profile-f$flags-n$n.stderr"
    done
done
for flags in 266 4362 131338; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 4 --warmup 1024 --iterations 1 --windows 1 --profile --small-test \
        --allow-reassociation > "../round5/long-f$flags.jsonl" 2> "../round5/long-f$flags.stderr"
done
for flags in 262410 131338; do
    RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=$flags \
        ./tools/bench-decode-fusion --q4 --small-n --no-clocks --batch 4 \
        --warmup 1024 --iterations 1 --windows 1 > "../round5/isa-f$flags.jsonl" \
        2> "../round5/isa-f$flags.stderr"
done
