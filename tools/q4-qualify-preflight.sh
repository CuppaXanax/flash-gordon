set -euo pipefail
root=/home/xander/hc-q4/qualify
mkdir -p "$root"
stat -f -c 'FILESYSTEM %T' "$root"
test ! -e "$root/clone-source"
test ! -e "$root/clone-target"
printf 'q4-clone-proof\n' > "$root/clone-source"
cp --reflink=always "$root/clone-source" "$root/clone-target"
cmp "$root/clone-source" "$root/clone-target"
rm -- "$root/clone-source" "$root/clone-target"
echo 'REFLINK_SUPPORTED=1'
