set -euo pipefail
test ! -e '/home/xander/flash-gordon-live/20261015-abq-operator/lease-active.json'
touch '/home/xander/flash-gordon-live/20261015-abq-operator/supervisor.pause'
systemctl stop flash-gordon-ring.service
test "$(systemctl show flash-gordon-ring.service -p LimitMEMLOCK --value)" = infinity