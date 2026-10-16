#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p src/.build .build cache evidence round2
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
cp /etc/cyan-skillfish-governor-smu/config.toml evidence/governor-original.toml
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
printf '%s\n' "${active[@]}" > evidence/services-original.txt
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "round2/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf round2.tar.gz round2 evidence
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../round2/build.txt 2>&1
for flags in 0 2048; do
    FG_SMALL_KERNELS=$flags DS4_REMOTE_TEST_FILTER=expert_small_component \
        ./tests/test_fg_vk > "../round2/cpu-f$flags.txt" 2>&1
done
FG_SMALL_KERNELS=2314 ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
    --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
    --session-test --rollback-test --allow-reassociation \
    > ../round2/oracles-f2314.jsonl 2> ../round2/oracles-f2314.stderr
for n in 2 3 4; do
    for round in 1 2 3; do
        for phase in before candidate after; do
            flags=266
            if [ "$phase" = candidate ]; then flags=2314; fi
            FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
                --batch "$n" --warmup "$((264/n))" --seed "$round" \
                --iterations 1 --windows 1 --profile \
                > "../round2/n$n-r$round-$phase.jsonl" 2> "../round2/n$n-r$round-$phase.stderr"
        done
    done
done
RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=2314 \
    ./tools/bench-decode-fusion --q4 --small-n --no-clocks --batch 4 \
    --warmup 2 --iterations 1 --windows 1 > ../round2/isa.jsonl 2> ../round2/isa.stderr
cd ..
