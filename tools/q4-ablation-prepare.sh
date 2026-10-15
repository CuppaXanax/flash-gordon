set -euo pipefail
root=/home/xander/hc-q4/stage-ablation-control
pack=/home/xander/fg-ring-pack-abliterated-q4-control
release=/home/xander/flash-gordon-live/20261015-q4-ablation
previous=/home/xander/flash-gordon-live/20261015-q4-foreign
printf '%s  %s\n' de8e1067732baaf9bb49083fbdebc55a991c78619d4156624f6ec03ddfc2022b "$previous/flash-gordon" | sha256sum -c -
cd "$root/code"
make -j4 tools/q4-eval-batch > "$root/evaluator-build.log" 2>&1 || { tail -40 "$root/evaluator-build.log"; exit 1; }
if [[ ! -e $pack ]]; then
  test -s "$root/pack/manifest.fgm"
  mv "$root/pack" "$pack"
else
  test -s "$pack/manifest.fgm"
  test ! -e "$root/pack"
fi
if [[ ! -e $release ]]; then
  mkdir -p "$release/operator"
  cp --reflink=always "$previous/flash-gordon" "$release/flash-gordon"
  cp --reflink=always tools/q4-eval-batch "$release/q4-eval-batch"
  cp -a "$previous/vulkan" "$release/"
else
  cmp "$previous/flash-gordon" "$release/flash-gordon"
  cmp tools/q4-eval-batch "$release/q4-eval-batch"
fi
rm -f "$root/dense-q4.gguf"
sha256sum "$pack/manifest.fgm" "$release/flash-gordon" "$release/q4-eval-batch"
df -B1 /home/xander
