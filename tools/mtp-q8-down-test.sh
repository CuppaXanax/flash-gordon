#!/bin/bash
set -euo pipefail
cd /home/xander/hc-mtpk
mkdir -p q8src/.build q8 cache .build
export TMPDIR=$PWD/.build XDG_CACHE_HOME=$PWD/cache
cp /etc/cyan-skillfish-governor-smu/config.toml q8/governor-before.toml
active=()
for service in bc250-embed bc250-rerank; do
    if systemctl is-active --quiet "$service"; then active+=("$service"); fi
done
printf '%s\n' "${active[@]}" > q8/services-before.txt
finish(){
    local rc=$?
    cd /home/xander/hc-mtpk
    for service in "${active[@]}"; do systemctl start "$service" || rc=1; done
    for service in "${active[@]}"; do
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --retry 60 --retry-connrefused --retry-delay 2 --max-time 3 \
            -fsS "http://$host:$port/health" > "q8/health-$port.json" || rc=1
    done
    cmp q8/governor-before.toml /etc/cyan-skillfish-governor-smu/config.toml || rc=1
    printf '%s\n' "$rc" > q8/exit-code.txt
    if [ "$rc" != 0 ]; then
        shopt -s nullglob
        for f in q8/*.stderr q8/*.txt; do echo "=== $f"; tail -6 "$f"; done
    fi
    tar -czf q8.tar.gz q8
    exit "$rc"
}
trap finish EXIT
for service in "${active[@]}"; do systemctl stop "$service"; done
tar -xf q8-source.tar -C q8src
cd q8src
export TMPDIR=$PWD/.build FG_SHADER_DIR=$PWD/vulkan
make -s -B -j8 tools/bench-decode-fusion tests/test_fg_vk > ../q8/build.txt 2>&1
for wide in 0 1; do
    FG_EXPERT_Q8_WIDE=$wide DS4_REMOTE_TEST_FILTER=small_q8_down_native \
        ./tests/test_fg_vk > "../q8/down-native-w$wide.txt" 2>&1
done
for flags in 0 330; do
    FG_SMALL_KERNELS=$flags DS4_REMOTE_TEST_FILTER=expert_small_component \
        ./tests/test_fg_vk > "../q8/component-f$flags.txt" 2>&1
    extra=()
    if [ "$flags" != 0 ]; then extra+=(--allow-reassociation); fi
    FG_SMALL_KERNELS=$flags ./tools/bench-decode-fusion --q4 --mixed-experts \
        --small-n --no-clocks --batch 2 --warmup 1 --iterations 1 --windows 1 \
        --small-test --session-test --rollback-test "${extra[@]}" \
        > "../q8/mixed-f$flags.jsonl" 2> "../q8/mixed-f$flags.stderr"
done
