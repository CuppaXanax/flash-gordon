set -euo pipefail
old=/home/xander/fg-ring-pack-abliterated
original=/home/xander/flash-gordon-live/20260926-abl2
df -B1 /home/xander
sha256sum "$original/flash-gordon" "$old/manifest.fgm"
du -sB1 "$old"
for proc in /proc/[0-9]*; do
  test -O "$proc" || continue
  exe=$(readlink "$proc/exe" 2>/dev/null) || continue
  case "$exe" in /home/xander/flash-gordon-live/*/flash-gordon) ;; *) continue ;; esac
  printf 'PROCESS pid=%s exe=%s args=' "${proc##*/}" "$exe"
  tr '\0' ' ' < "$proc/cmdline"
  printf '\n'
done
