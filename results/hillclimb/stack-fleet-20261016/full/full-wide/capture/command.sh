set -euo pipefail
test -f '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423/owner' || exit 0
test "$(cat '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423/owner')" = 'dc239eaa04e34fdcbb9e554a46cee423'
test -O '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423' && test ! -L '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423'
test "$(readlink -f '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423')" = '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423'
rm -rf -- '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423'
