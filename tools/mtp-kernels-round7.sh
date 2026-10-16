#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p src/.build .build cache evidence round7
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
            -fsS "http://$host:$port/health" > "round7/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in round7/*.txt round7/*.stderr; do echo "=== $f"; tail -5 "$f"; done
    fi
    tar -czf round7.tar.gz round7 evidence
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan FG_EXPERT_Q8_WIDE=1
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../round7/build.txt 2>&1
sha256sum tools/bench-decode-fusion tests/test_fg_vk > ../round7/binary-sha256.txt
FG_SMALL_KERNELS=2097482 DS4_REMOTE_TEST_FILTER=small_dense_vector_layout \
    ./tests/test_fg_vk > ../round7/vector-residency.txt 2>&1
FG_SMALL_KERNELS=8388608 DS4_REMOTE_TEST_FILTER=small_qsa_select \
    ./tests/test_fg_vk > ../round7/wave-topk-cpu.txt 2>&1
for wide in 0 1; do
    FG_EXPERT_Q8_WIDE=$wide FG_SMALL_KERNELS=4194634 DS4_REMOTE_TEST_FILTER=expert_small_component \
        ./tests/test_fg_vk > "../round7/expert-cpu-wide$wide.txt" 2>&1
    FG_EXPERT_Q8_WIDE=$wide FG_SMALL_KERNELS=4194634 DS4_REMOTE_TEST_FILTER=small_q8_down_native \
        ./tests/test_fg_vk > "../round7/exact-q8-wide$wide.txt" 2>&1
done
FG_SMALL_KERNELS=8393034 ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
    --batch 4 --warmup 1024 --iterations 1 --windows 1 --profile --small-test \
    --allow-reassociation > ../round7/long-wave.jsonl 2> ../round7/long-wave.stderr
for flags in 2097482 2105674 2621770 2130250; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../round7/oracles-f$flags.jsonl" 2> "../round7/oracles-f$flags.stderr"
done
for flags in 4194634 2097482 6291786; do
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --mixed-experts --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../round7/mixed-oracles-f$flags.jsonl" 2> "../round7/mixed-oracles-f$flags.stderr"
done
for n in 2 4; do
    for flags in 330 2097482 2105674 2621770 2130250; do
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
            --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
            > "../round7/profile-f$flags-n$n.jsonl" 2> "../round7/profile-f$flags-n$n.stderr"
    done
    for flags in 330 4194634 6291786; do
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --mixed-experts --small-n --no-clocks \
            --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
            > "../round7/mixed-profile-f$flags-n$n.jsonl" 2> "../round7/mixed-profile-f$flags-n$n.stderr"
    done
done
for flags in 2097482 4194634; do
    RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=$flags \
        ./tools/bench-decode-fusion --q4 --mixed-experts --small-n --no-clocks --batch 4 \
        --warmup 2 --iterations 1 --windows 1 > "../round7/isa-f$flags.jsonl" \
        2> "../round7/isa-f$flags.stderr"
done
