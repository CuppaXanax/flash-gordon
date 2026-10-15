set -euo pipefail
label=${1:?label}
pack=${2:?pack}
release=${3:?release}
[[ $label == qualify-ordinary || $label == qualify-protected ]]
[[ $pack == /home/xander/fg-ring-pack-abliterated-q4 ||
   $pack == /home/xander/fg-ring-pack-abliterated-q4-protected ]]
[[ $release =~ ^/home/xander/flash-gordon-live/[0-9]{8}-q4[-a-z0-9]*$ ]]
root="/home/xander/hc-q4/stage-$label"
if [[ ! -e $pack ]]; then
  test -s "$root/pack/manifest.fgm"
  # Mutable session state must never point into the immutable source pack.
  find "$root/pack" -maxdepth 1 -type l -name '*.state*' -delete
  mv -- "$root/pack" "$pack"
else
  test -s "$pack/manifest.fgm"
  test ! -e "$root/pack"
fi
if [[ $label == qualify-ordinary ]]; then
  if [[ ! -e $release ]]; then
    mkdir -p "$release/operator"
    cp --reflink=always "$root/code/flash-gordon" "$release/flash-gordon"
    cp --reflink=always "$root/code/tools/q4-eval-batch" "$release/q4-eval-batch"
    cp -a "$root/code/vulkan" "$release/"
  else
    cmp "$root/code/flash-gordon" "$release/flash-gordon"
    cmp "$root/code/tools/q4-eval-batch" "$release/q4-eval-batch"
  fi
fi
old=/home/xander/fg-ring-pack-abliterated
teacher=/home/xander/fg-ring-pack-abliterated-q4-teacher
if [[ ! -d $teacher ]]; then
  mkdir "$teacher"
  cp --reflink=always "$old/manifest.fgm" "$teacher/manifest.fgm"
  for file in "$old"/*; do
    name=${file##*/}
    case "$name" in manifest.fgm|*.state*) continue ;; esac
    test ! -d "$file" || continue
    ln -s "$file" "$teacher/$name"
  done
fi
if [[ ! -e $teacher/tokenizer ]]; then
  test -d "$old/tokenizer"
  ln -s "$old/tokenizer" "$teacher/tokenizer"
fi
test -s "$teacher/tokenizer/tokenizer.fgt"
if [[ -f $root/dense-q4.gguf ]]; then rm -- "$root/dense-q4.gguf"; fi
cmp "$old/manifest.fgm" "$teacher/manifest.fgm"
sha256sum "$release/flash-gordon" "$release/q4-eval-batch" "$pack/manifest.fgm" "$teacher/manifest.fgm" "$old/manifest.fgm"
du -sB1 "$pack"
du -sB1 --apparent-size "$pack"
df -B1 /home/xander
