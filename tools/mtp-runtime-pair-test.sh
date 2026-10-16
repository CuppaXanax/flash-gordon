#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
root=/home/xander/hc-mtp
cd "$root"
mkdir -p .build
export TMPDIR="$PWD/.build" XDG_CACHE_HOME="$PWD/.build/cache"
addresses=" $(hostname -I) "
if [[ "$addresses" == *" 192.168.42.51 "* ]]; then parity=0; peer=192.168.42.52
elif [[ "$addresses" == *" 192.168.42.52 "* ]]; then parity=1; peer=192.168.42.51
else echo "pair fixture is restricted to .51/.52" >&2; exit 1; fi
rule="rule family=\"ipv4\" source address=\"$peer/32\" port port=\"29600-29675\" protocol=\"tcp\" accept"
case "${1:-build}" in
open)
    test ! -e runtime-pair-firewall.owned
    if sudo -n firewall-cmd --query-rich-rule="$rule" >/dev/null; then
        echo "refusing to borrow a pre-existing firewall rule" >&2; exit 1
    fi
    sudo -n firewall-cmd --add-rich-rule="$rule" --timeout=600
    printf '%s\n' "$rule" >runtime-pair-firewall.owned
    ;;
close)
    if [[ -f runtime-pair-firewall.owned ]]; then
        test "$(cat runtime-pair-firewall.owned)" = "$rule"
        if sudo -n firewall-cmd --query-rich-rule="$rule" >/dev/null; then
            sudo -n firewall-cmd --remove-rich-rule="$rule"
        fi
        rm runtime-pair-firewall.owned
    fi
    ;;
build)
    mkdir -p source
    tar -xf source.tar -C source
    make -C source -j8 all tests/test_mtp_ring tests/test_api test-mtp \
        tools/mtp-embedding tools/bench-decode-fusion tests/test_fg_vk \
        >runtime-pair-build.log 2>&1 || { tail -60 runtime-pair-build.log; exit 1; }
    ./source/tests/test_api >runtime-pair-api.log 2>&1
    (cd source; python3 -m unittest discover -s tests -p test_mtp_serving_qualify.py) \
        >runtime-pair-client.log 2>&1
    sha256sum source/tests/test_mtp_ring source/tests/test_api
    echo "Numerical-double runtime fixture and API gates built; no GPU serving claim."
    ;;
gpu)
    sudo -n prlimit --pid "$$" --memlock=268435456:268435456
    cd source
    export FG_SHADER_DIR="$PWD/vulkan"
    FG_SMALL_KERNELS=0 DS4_REMOTE_TEST_FILTER=expert_small_component \
        ./tests/test_fg_vk >../runtime-pair-gpu-component.log 2>&1
    FG_SMALL_KERNELS=0 ./tools/bench-decode-fusion --q4 --small-n --no-clocks \
        --batch 2 --warmup 1 --iterations 1 --windows 1 \
        --small-test --session-test --rollback-test >../runtime-pair-gpu-block.log 2>&1
    grep -E 'small_n_sessions|small_block_test|small_causal_test|rollback' ../runtime-pair-gpu-block.log
    ;;
release)
    cd source
    sha256sum flash-gordon tools/mtp-embedding vulkan/*.spv >release.sha256
    tar -czf ../serving-release.tar.gz flash-gordon tools/mtp-embedding vulkan release.sha256
    sha256sum ../serving-release.tar.gz
    ;;
run)
    for reject in 0 1 2 3; do
        base=$((29600+reject*20))
        ./source/tests/test_mtp_ring 192.168.42.51 192.168.42.52 "$base" "$reject" "$parity" \
            >"runtime-pair-reject-$reject.log" 2>&1 || {
                tail -50 "runtime-pair-reject-$reject.log"; exit 1;
            }
        echo "host-parity=$parity rejection=$reject PASS"
        grep -E 'PASS$' "runtime-pair-reject-$reject.log" || true
    done
    ;;
evidence)
    tar -czf runtime-pair-evidence.tar.gz runtime-pair-*.log
    sha256sum runtime-pair-evidence.tar.gz
    ;;
*) echo "expected build, gpu, release, open, run, close or evidence" >&2; exit 2 ;;
esac
