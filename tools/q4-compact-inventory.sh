set -euo pipefail
cd /home/xander/hc-q4
find . -maxdepth 2 -type f \( -name '*.fgm' -o -name '*q4*.json' -o -name 'dense-q4.gguf' \) -printf '%p %s bytes\n'
if test -d /home/xander/fg-ring-pack-abliterated; then
  find -L /home/xander/fg-ring-pack-abliterated -maxdepth 1 -type f -name '*.fgm' -printf '%p %s bytes\n'
else
  echo 'Baseline pack directory absent on this idle blade.'
fi
df -h .
