set -euo pipefail
test -f '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602/owner' || exit 0
test "$(cat '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602/owner')" = 'ad343506d21c49e8a3a112f98d2eb602'
test -O '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602' && test ! -L '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602'
test "$(readlink -f '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602')" = '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602'
rm -rf -- '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602'
