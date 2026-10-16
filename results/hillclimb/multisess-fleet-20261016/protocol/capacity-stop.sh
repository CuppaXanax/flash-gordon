set -euo pipefail
dir=/home/xander/flash-gordon-live/20261016-abq-multisess-capacity
if [[ -f $dir/monitor.pid ]]; then
  pid=$(cat "$dir/monitor.pid")
  [[ $pid =~ ^[0-9]+$ ]]
  if [[ -e /proc/$pid/cmdline ]]; then
    args=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    if [[ $args == *"$dir/memory.py"* ]]; then
      kill -TERM "$pid"
      for ((waited=0;waited<10;waited++)); do
        [[ -e /proc/$pid/stat && $(awk '{print $3}' "/proc/$pid/stat") != Z ]] || break
        sleep 1
      done
    fi
  fi
fi
wc -l "$dir/memory.jsonl"
cat "$dir/memory.err"
df -B1 /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm
