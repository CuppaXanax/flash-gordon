set -euo pipefail
archive=/home/xander/flash-gordon-live/20261016-abq-multisess.tar.gz
release=/home/xander/flash-gordon-live/20261016-abq-multisess
printf '%s  %s\n' 55404066fa9a902fa7474637a380d8c1ec85caf7324bfe51741f9a1204e8cdbb "$archive" | sha256sum -c -
test ! -e "$release"
mkdir "$release"
tar -xzf "$archive" -C "$release"
printf '%s  %s\n' de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58 "$release/flash-gordon" | sha256sum -c -
rm "$archive"
