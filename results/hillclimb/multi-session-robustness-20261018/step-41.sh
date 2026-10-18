set -euo pipefail
systemctl stop flash-gordon-robust-deadline.timer flash-gordon-robust-deadline.service || true
systemctl enable --now flash-gordon-ring.service
systemctl is-enabled flash-gordon-ring.service
systemctl is-active flash-gordon-ring.service
systemctl show flash-gordon-ring.service -p LimitMEMLOCK --value
if systemctl is-active --quiet flash-gordon-robust-deadline.timer; then exit 1; fi
journalctl -u flash-gordon-ring.service --no-pager -n 15