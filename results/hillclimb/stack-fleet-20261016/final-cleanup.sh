set -euo pipefail
date -Is
test ! -e '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602' && test ! -L '/home/xander/hc-abq-stack-ad343506d21c49e8a3a112f98d2eb602'
test ! -e '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423' && test ! -L '/home/xander/hc-abq-stack-dc239eaa04e34fdcbb9e554a46cee423'
d=/home/xander/hc-abq-stack-live-20261016
if test -e "$d"; then
  test -d "$d" && test -O "$d" && test ! -L "$d"
  rm -rf -- "$d"
fi
test ! -e "$d"
echo OWNED_STACK_SCRATCH_CLEAN
df -BG /home/xander
