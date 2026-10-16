#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p src/.build .build cache evidence round6
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
            -fsS "http://$host:$port/health" > "round6/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in round6/*.txt round6/*.stderr; do echo "=== $f"; tail -5 "$f"; done
    fi
    tar -czf round6.tar.gz round6 evidence
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../round6/build.txt 2>&1
sha256sum tools/bench-decode-fusion tests/test_fg_vk > ../round6/binary-sha256.txt
for flags in 0 1048576; do
    FG_SMALL_KERNELS=$flags DS4_REMOTE_TEST_FILTER=small_qsa_select \
        ./tests/test_fg_vk > "../round6/topk-cpu-f$flags.txt" 2>&1
done
FG_SMALL_KERNELS=330 DS4_REMOTE_TEST_FILTER=expert_small_component \
    ./tests/test_fg_vk > ../round6/expert-cpu.txt 2>&1
for flags in 524618 532810 1053002 1179978; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../round6/oracles-f$flags.jsonl" 2> "../round6/oracles-f$flags.stderr"
done
for n in 2 4; do
    for flags in 330 524618 532810; do
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
            --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
            > "../round6/profile-f$flags-n$n.jsonl" 2> "../round6/profile-f$flags-n$n.stderr"
    done
done
for flags in 330 4426 131402 1053002 1179978; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 4 --warmup 1024 --iterations 1 --windows 1 --profile --small-test \
        --allow-reassociation > "../round6/long-f$flags.jsonl" 2> "../round6/long-f$flags.stderr"
done
for flags in 524618 1053002; do
    RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=$flags \
        ./tools/bench-decode-fusion --q4 --small-n --no-clocks --batch 4 \
        --warmup 1024 --iterations 1 --windows 1 > "../round6/isa-f$flags.jsonl" \
        2> "../round6/isa-f$flags.stderr"
done
