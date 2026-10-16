set -euo pipefail
dir=/home/xander/flash-gordon-live/20261016-abq-multisess-capacity
release=/home/xander/flash-gordon-live/20261016-abq-multisess
manifest=/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
pid=$(cat "/home/xander/flash-gordon-live/20261015-abq-operator/rank-$rank.pid")
[[ $(readlink "/proc/$pid/exe") == "$release/flash-gordon" ]]
test ! -e "$dir"
mkdir "$dir"
mv /home/xander/flash-gordon-live/20261016-abq-multisess-memory.py "$dir/memory.py"
nohup python3 -u "$dir/memory.py" --pid "$pid" --manifest "$manifest" --release "$release" \
    --seconds 3600 >"$dir/memory.jsonl" 2>"$dir/memory.err" </dev/null &
echo "$!" >"$dir/monitor.pid"
sleep 1
kill -0 "$(cat "$dir/monitor.pid")"
echo "MONITOR rank=$rank serving=$pid monitor=$(cat "$dir/monitor.pid")"
df -B1 "$manifest"
