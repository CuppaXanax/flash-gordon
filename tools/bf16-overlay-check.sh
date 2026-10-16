#!/usr/bin/env bash
set -euo pipefail
cd /home/xander/hc-bf16
echo 'e742dbf593f03c5c608de229329755e7cb19138e7b1c4031632eaf49f1acf4d1  overlay.gguf' | sha256sum -c -
