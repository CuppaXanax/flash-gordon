#!/usr/bin/env bash
set -euo pipefail
release=/home/xander/flash-gordon-live/20261016-abq-multisess
operator=/home/xander/flash-gordon-live/20261015-abq-operator
date -Is
hostname
df -BG /home/xander
sha256sum "$release/flash-gordon" /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm
test -x "$operator/launch-current.sh"
if test ! -e "$operator/launch-pre-stack-20261016.sh"; then
    cp -p "$operator/launch-current.sh" "$operator/launch-pre-stack-20261016.sh"
    cp -p "$operator/current.json" "$operator/pre-stack-20261016.json"
fi
sha256sum "$operator/launch-current.sh" "$operator/launch-pre-stack-20261016.sh"
for proc in /proc/[0-9]*; do
    test -O "$proc" || continue
    exe=$(readlink "$proc/exe" 2>/dev/null) || continue
    [[ $exe == "$release/flash-gordon" ]] || continue
    printf 'PID=%s CMD=' "${proc##*/}"
    tr '\0' ' ' <"$proc/cmdline"; echo
    tr '\0' '\n' <"$proc/environ" | grep '^FG_' | sort
done
