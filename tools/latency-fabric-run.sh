#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
mkdir -p build cache
export TMPDIR=$PWD/build XDG_CACHE_HOME=$PWD/cache
pgrep -af 'flash-gordon|llama|bench|sol-|mtp' || true
host=$(hostname -I)
case "$host" in
    *192.168.42.52*) ranks="0 2 4 6";;
    *192.168.42.53*) ranks="1 3 5 7";;
    *) exit 1;;
esac
firewall-cmd --add-port=19600-19615/tcp >/dev/null
pids=()
cleanup(){
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    firewall-cmd --remove-port=19600-19615/tcp >/dev/null
}
trap cleanup EXIT
for rank in $ranks; do
    ./bench-fabric-latency "$rank" > "fabric-$rank.jsonl" 2> "fabric-$rank.log" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
pids=()
cat fabric-*.jsonl
