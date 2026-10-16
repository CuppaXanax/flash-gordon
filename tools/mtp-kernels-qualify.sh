#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p qualify
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../qualify/build.txt 2>&1
sha256sum tools/bench-decode-fusion tests/test_fg_vk shaders/fg_small* shaders/fg_gdn_small_cached.comp \
    src/vk.c src/qsa.c > ../qualify/hashes.txt
cd ..
active=()
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> qualify/restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "qualify/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf qualify.tar.gz qualify evidence
    exit "$rc"
}
trap finish EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
for service in "${active[@]}"; do systemctl stop "$service"; done
cat /sys/class/drm/card*/device/pp_dpm_sclk > qualify/sclk-observed.txt
cd src
for flags in 74 330; do
    FG_SMALL_KERNELS=$flags DS4_REMOTE_TEST_FILTER=expert_small_component \
        ./tests/test_fg_vk > "../qualify/component-f$flags.txt" 2>&1
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
        --session-test --rollback-test --allow-reassociation \
        > "../qualify/oracles-f$flags.jsonl" 2> "../qualify/oracles-f$flags.stderr"
done
run(){
    local round=$1 n=$2 flags=$3 phase=$4
    FG_ROOFLINE=0 FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion \
        --q4 --small-n --no-clocks --batch "$n" --seed "$round" \
        --warmup "$((264/n))" --windows 3 --iterations "$((72/n))" \
        > "../qualify/r$round-n$n-$phase-f$flags.jsonl" \
        2> "../qualify/r$round-n$n-$phase-f$flags.stderr"
}
for round in 1 2 3 4 5; do
    run "$round" 1 0 before
    for n in 2 3 4; do
        if ((round%2)); then candidates=(74 330); else candidates=(330 74); fi
        run "$round" "$n" 0 a0
        run "$round" "$n" "${candidates[0]}" b0
        run "$round" "$n" 0 a1
        run "$round" "$n" "${candidates[1]}" b1
        run "$round" "$n" 0 a2
    done
    run "$round" 1 0 after
done
for flags in 0 74 330; do
    for n in 2 3 4; do
        FG_ROOFLINE=0 FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 \
            --small-n --no-clocks --batch "$n" --warmup "$((264/n))" \
            --iterations 1 --windows 1 --profile \
            > "../qualify/profile-f$flags-n$n.jsonl" 2> "../qualify/profile-f$flags-n$n.stderr"
    done
done
cd ..
