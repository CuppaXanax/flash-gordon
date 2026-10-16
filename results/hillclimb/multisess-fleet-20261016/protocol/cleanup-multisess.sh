set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
dir=/home/xander/flash-gordon-live/20261016-abq-multisess-capacity
if [[ -d $dir ]]; then
  if [[ -f $dir/monitor.pid ]]; then
    monitor=$(cat "$dir/monitor.pid")
    [[ $monitor =~ ^[0-9]+$ ]]
    if [[ -r /proc/$monitor/cmdline ]]; then
      command=$(tr '\0' ' ' <"/proc/$monitor/cmdline")
      if [[ $command == *"$dir/memory.py"* ]]; then
        echo "Observer PID $monitor is still active; refusing cleanup" >&2
        exit 1
      fi
    fi
  fi
  rm -rf -- "$dir"
fi
test ! -e /home/xander/flash-gordon-live/20261016-abq-multisess-memory.py
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
pid=$(cat "$run/rank-$rank.pid")
[[ $(readlink "/proc/$pid/exe") == '@@RELEASE@@/flash-gordon' ]]
printf '%s  %s\n' '@@BIN@@' "/proc/$pid/exe" | sha256sum -c -
printf '%s  %s\n' '@@MFSHA@@' '@@MF@@' | sha256sum -c -
echo "CLEAN rank=$rank pid=$pid"
df -B1 '@@MF@@'
unused=/home/xander/flash-gordon-live/20261016-abq-multisess
if [[ '@@RELEASE@@' != "$unused" && -d $unused ]]; then
  for entry in /proc/[0-9]*/exe; do
    target=$(readlink "$entry" 2>/dev/null || true)
    if [[ $target == "$unused/flash-gordon" || $target == "$unused/flash-gordon (deleted)" ]]; then
      echo 'Unselected candidate is still mapped; refusing removal' >&2
      exit 1
    fi
  done
  printf '%s  %s\n' de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58 \
    "$unused/flash-gordon" | sha256sum -c -
  rm -rf -- "$unused"
  echo 'Removed unmapped, unselected candidate release'
fi
