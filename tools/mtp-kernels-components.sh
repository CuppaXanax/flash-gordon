#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p half
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
tar -xf candidate.tar -C src
cd src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../half/build.txt 2>&1
cd ..
active=()
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        systemctl is-active "$service" >> half/restored.txt || rc=1
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "half/health-$port.json" || rc=1
    done
    cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    tar -czf half.tar.gz half
    exit "$rc"
}
trap finish EXIT
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
for service in "${active[@]}"; do systemctl stop "$service"; done
cd src
for flags in 0 64 512 1024; do
    rc=0
    FG_SMALL_KERNELS=$flags DS4_REMOTE_TEST_FILTER=expert_small_component \
        ./tests/test_fg_vk > "../half/component-f$flags.txt" 2>&1 || rc=$?
    echo "component flags=$flags rc=$rc"
    if [ "$flags" != 512 ] && [ "$flags" != 1024 ] && [ "$rc" != 0 ]; then
        tail -15 "../half/component-f$flags.txt"
        exit "$rc"
    fi
    if ((flags>=512)) && [ "$rc" = 0 ]; then
        FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
            --batch 2 --warmup 1 --iterations 1 --windows 1 --small-test \
            --allow-reassociation > "../half/block-f$flags.jsonl" 2> "../half/block-f$flags.stderr"
        for n in 2 3 4; do
            for f in 0 "$flags" 0; do
                FG_SMALL_KERNELS=$f ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
                    --batch "$n" --warmup "$((264/n))" --iterations 1 --windows 1 --profile \
                    > "../half/profile-n$n-f$f-$RANDOM.jsonl" 2> ../half/profile.stderr
            done
            RADV_DEBUG=shaderstats MESA_SHADER_CACHE_DISABLE=true FG_SMALL_KERNELS=$flags \
                ./tools/bench-decode-fusion --q4 --small-n --no-clocks --batch 4 \
                --warmup 2 --iterations 1 --windows 1 > ../half/isa.jsonl 2> ../half/isa.stderr
        done
    fi
done
cd ..
