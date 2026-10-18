set -euo pipefail
systemctl is-enabled flash-gordon-ring.service
systemctl is-active flash-gordon-ring.service
for unit in flash-gordon-incident-deadline.timer flash-gordon-incident-deadline-v2.timer; do
  if systemctl is-active --quiet "$unit"; then exit 1; fi
done
journalctl -u flash-gordon-ring.service --no-pager -n 12