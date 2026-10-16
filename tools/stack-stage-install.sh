#!/usr/bin/env bash
set -euo pipefail
root=/home/xander/hc-abq-stack-live-20261016
stack=/home/xander/flash-gordon-live/20261016-abq-stack-adaptive
teacher=/home/xander/flash-gordon-live/20261016-abq-quality-adaptive
printf '%s  %s\n' 8a319610c35103f94c02b7dd03f9ad12379e7ce541707ee9cd0438e25e373db1 "$root/stack.tar.gz" | sha256sum -c -
printf '%s  %s\n' 65e79056c9f85635bd29f950ba5f957ec0c55701e9fea0672da84da7a2fae88a "$root/teacher.tar.gz" | sha256sum -c -
test ! -e "$stack"
test ! -e "$teacher"
mkdir "$stack" "$teacher"
tar -xzf "$root/stack.tar.gz" -C "$stack" --no-same-owner
tar -xzf "$root/teacher.tar.gz" -C "$teacher" --no-same-owner
chmod 755 "$stack/flash-gordon" "$stack/q4-eval-batch" "$stack/tests/test_fg_vk" "$teacher/q4-eval-batch"
sha256sum "$stack/flash-gordon" "$stack/q4-eval-batch" "$teacher/q4-eval-batch"
ldd "$stack/flash-gordon"
ldd "$teacher/q4-eval-batch"
