#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
host=$(hostname -I)
case "$host" in
    *192.168.42.52*) ranks="0 2 4 6";;
    *192.168.42.53*) ranks="1 3 5 7";;
    *) exit 1;;
esac
firewall_owned=0
if ! firewall-cmd --query-port=19800-19815/tcp >/dev/null; then
    firewall-cmd --add-port=19800-19815/tcp >/dev/null
    firewall_owned=1
fi
pids=()
cleanup(){
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    if ((firewall_owned)); then firewall-cmd --remove-port=19800-19815/tcp >/dev/null; fi
}
trap cleanup EXIT
for rank in $ranks; do
    timeout 180 source/tools/bench-bf16-hops "$rank" \
        > "hop-$rank.jsonl" 2> "hop-$rank.stderr" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
pids=()
cat hop-*.jsonl
