#!/usr/bin/env bash
set -euo pipefail
root=/home/xander/hc-bf16
cd "$root"
host=$(hostname -I)
if pgrep -f '(^| )((source|hardware)/tools/(bf16-offline|bench-bf16-hops|fp16-wire-oracle|mtp-real-place)|python3 (source/tools/bf16_source.py|hardware/tools/fp16_q8_restore.py))( |$)'; then
    echo 'Refusing cleanup while owned numerical/transport work remains active' >&2
    exit 1
fi
governor=$(sha256sum /etc/cyan-skillfish-governor-smu/config.toml)
[[ "${governor%% *}" == a5e83975e4fbea57d6110cbf4aa81fd58367d044c2bc602815ff0ca21ede6757 ]]
for device in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
    current=$(sha256sum "$device")
    [[ "${current%% *}" == 6d1dbb9c582de44ad1dbc0d24eca4d831d8172bef0593680ecd24f6d0ce9def5 ]]
done
case "$host" in
    *192.168.42.50*)
        test -f offline.complete
        for service in bc250-embed bc250-rerank; do
            systemctl is-active --quiet "$service"
            port=8091
            [[ "$service" != bc250-rerank ]] || port=8092
            curl --max-time 10 -fsS "http://192.168.42.50:$port/health" > "$service.health.json"
            python3 -c 'import json,sys;assert json.load(open(sys.argv[1]))["status"]=="ok"' "$service.health.json"
        done
        ;;
    *192.168.42.52*|*192.168.42.53*)
        for service in bc250-embed bc250-rerank; do
            state=$(systemctl is-active "$service" || true)
            [[ "$state" == inactive ]]
        done
        systemctl is-active --quiet firewalld
        set +e
        firewall-cmd --query-port=19800-19815/tcp >/dev/null
        firewall_status=$?
        set -e
        [[ "$firewall_status" == 1 ]]
        sockets=$(ss -H -ltn)
        if printf '%s\n' "$sockets" | awk '{print $4}' | grep -Eq ':(1980[0-9]|1981[0-5])$'; then
            echo 'Pair port still listening' >&2
            exit 1
        fi
        ;;
    *) exit 1;;
esac
if [[ "$host" == *192.168.42.53* ]]; then
    [[ $(stat -c %s /home/xander/hc-q4/ablation-source-check/dense-mixed-04.gguf) == 2071870848 ]]
fi
cd /home/xander
rm -rf -- "$root"
test ! -e "$root"
sleep 2
test ! -e "$root"
HOST="$host" python3 - <<'PY'
from datetime import datetime,timezone
import json,os
print(json.dumps(dict(host=os.environ["HOST"].split()[0],
    cleaned="/home/xander/hc-bf16",absence_checks=2,owned_processes=0,
    governor_config_sha256="a5e83975e4fbea57d6110cbf4aa81fd58367d044c2bc602815ff0ca21ede6757",
    performance_level_unchanged=True,original_service_state_verified=True,
    health_verified=".50" in os.environ["HOST"],temporary_pair_ports_closed=True,
    utc=datetime.now(timezone.utc).isoformat())))
PY
