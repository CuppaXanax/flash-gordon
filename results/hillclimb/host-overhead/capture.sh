#!/usr/bin/env bash
set -euo pipefail
blade=$(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)
[[ $blade =~ ^5[0-3]$ ]]
release=/home/xander/flash-gordon-live/20261016-host-overhead
cd "$release"
bash -n tools/decode-instance.sh
if (ip(){ echo 'inet 192.168.42.42'; }; export -f ip; bash tools/decode-instance.sh replay "$release"); then
    echo 'Offline physical-host guard failed' >&2; exit 1
fi
mkdir -p .build-scratch
export TMPDIR="$PWD/.build-scratch"
cc -shared -fPIC -O2 results/hillclimb/host-overhead/instrumentation/vk-wall.c \
    -o vk-wall.so -ldl -pthread
LD_PRELOAD="$PWD/vk-wall.so" DS4_REMOTE_TEST_FILTER=submit_group_benchmark \
    ./tests/test_fg_vk >cpu-wall.log 2>&1 || { tail -30 cpu-wall.log; exit 1; }
grep HOST_GROUP_BENCH cpu-wall.log
sha256sum flash-gordon tests/test_fg_vk tests/test_qsa_prefill >binary-sha256.txt
uname -r >kernel.txt
gzip -c cpu-wall.log >cpu-wall.log.gz
tar -czf host-results.tar.gz cpu-wall.log.gz binary-sha256.txt kernel.txt
rmdir "$TMPDIR"
