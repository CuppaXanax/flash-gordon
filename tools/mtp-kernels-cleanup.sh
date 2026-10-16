#!/bin/bash
set -euo pipefail
root=/home/xander/hc-mtpk
cd "$root"
cmp evidence/governor-original.toml /etc/cyan-skillfish-governor-smu/config.toml
printf 'governor_config_sha256='
sha256sum /etc/cyan-skillfish-governor-smu/config.toml | cut -d' ' -f1
systemctl is-active cyan-skillfish-governor-smu.service
if [ -x /usr/local/bin/bc250-cu-live-manager ]; then
    /usr/local/bin/bc250-cu-live-manager status
fi
for service in bc250-embed bc250-rerank; do
    if grep -qx "$service" evidence/services-original.txt; then
        systemctl is-active "$service"
        port=8091
        if [ "$service" = bc250-rerank ]; then port=8092; fi
        host=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -1)
        curl --max-time 3 -fsS "http://$host:$port/health"
        printf '\n'
    else
        ! systemctl is-active --quiet "$service"
    fi
done
cd /home/xander
rm -rf "$root"
test ! -e "$root"
printf 'workspace_absent_1=true\n'
sleep 2
test ! -e "$root"
printf 'workspace_absent_2=true\n'
