#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-mtp
sha256sum source.tar native-q8.gguf source/flash-gordon \
    source/tools/bench-mtp-head source/tools/bench-decode-fusion \
    > serving-artifacts.sha256
tar -czf serving-evidence.tar.gz serving-artifacts.sha256 serving-build.log \
    serving-head-selftest.jsonl serving-small-sessions.jsonl \
    serving-small-sessions.stderr serving-native-sessions.jsonl \
    serving-native-sessions.stderr serving-pair-summary.json \
    serving-pair-A serving-pair-B serving-pair-A2 serving-bench-* \
    source/.build/small-profile-1.jsonl source/.build/small-profile-2.jsonl \
    source/.build/small-profile-3.jsonl source/.build/small-profile-4.jsonl
sha256sum serving-evidence.tar.gz
