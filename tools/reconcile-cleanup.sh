#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname -I)" == *192.168.42.50* ]]
root=/home/xander/hc-reconcile
test -f "$root/complete"
python3 - <<'PY'
from pathlib import Path
root='/home/xander/hc-reconcile'
owned=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        if str(p.joinpath('cwd').resolve()).startswith(root):
            owned.append(p.name)
    except (PermissionError,FileNotFoundError,ProcessLookupError):
        pass
assert not owned,('owned process remains',owned)
PY
sha256sum -c "$root/evidence/governor-before.sha256" >/dev/null
cmp "$root/evidence/performance-before.txt" <(cat /sys/class/drm/card*/device/power_dpm_force_performance_level)
for service in bc250-embed bc250-rerank; do
    systemctl start "$service"
    port=8091
    [[ "$service" != bc250-rerank ]] || port=8092
    healthy=false
    for i in $(seq 1 40); do
        if curl --max-time 3 -fsS "http://192.168.42.50:$port/health" >/dev/null 2>&1; then
            healthy=true; break
        fi
        sleep 2
    done
    "$healthy"
done
rm -rf -- "$root"
test ! -e "$root"
sleep 2
test ! -e "$root"
printf '{"host":"192.168.42.50","utc":"%s","owned_root_absent_twice":true,"owned_processes":0,"services_healthy":true,"governor_unchanged":true,"performance_level_unchanged":true}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
