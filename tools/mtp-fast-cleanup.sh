#!/usr/bin/env bash
set -euo pipefail
cd /home/xander
for link in /proc/[0-9]*/cwd; do
    directory="$(readlink "$link" 2>/dev/null || true)"
    case "$directory" in
        /home/xander/hc-mtp|/home/xander/hc-mtp/*)
            echo "experiment still has a process using $link" >&2
            exit 1 ;;
    esac
done
rm -rf /home/xander/hc-mtp
test ! -e /home/xander/hc-mtp
echo "hc-mtp absent: first check"
sleep 1
test ! -e /home/xander/hc-mtp
echo "hc-mtp absent: second check"
