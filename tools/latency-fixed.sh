#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-lat
tar -xf source.tar
sed -i 's/\r$//' tools/latency-run.sh
cp /etc/cyan-skillfish-governor-smu/config.toml original.toml
systemctl is-active --quiet cyan-skillfish-governor-smu
grep -q '^min = 1000$' original.toml
grep -q '^max = 2000$' original.toml
sed 's/^min = 1000$/min = 2000/' original.toml > pinned.toml
gp=
restore(){
    if [ -n "$gp" ]; then kill "$gp" 2>/dev/null || true; wait "$gp" 2>/dev/null || true; fi
    systemctl start cyan-skillfish-governor-smu
    cmp original.toml /etc/cyan-skillfish-governor-smu/config.toml
    systemctl is-active cyan-skillfish-governor-smu > restored.txt
}
trap restore EXIT
systemctl stop cyan-skillfish-governor-smu
/usr/local/libexec/cyan-skillfish-governor-smu -v "$PWD/pinned.toml" > governor.log 2>&1 &
gp=$!
sleep 2
kill -0 "$gp"
cat /sys/class/drm/card*/device/pp_dpm_sclk > sclk-start.txt
bash tools/latency-run.sh
cat /sys/class/drm/card*/device/pp_dpm_sclk > sclk-end.txt
restore
trap - EXIT
tar -cf fixed.tar gpu.jsonl original.toml pinned.toml sclk-start.txt sclk-end.txt restored.txt
