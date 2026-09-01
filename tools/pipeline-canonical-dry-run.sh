#!/usr/bin/env bash
set -euo pipefail

filter="${DS4_REMOTE_TEST_FILTER:-pipeline_canonical_dry_run}"
if [[ "$filter" != "pipeline_canonical_dry_run" && "$filter" != "all" ]]; then
  echo "unsupported filter for canonical pipeline dry-run: $filter" >&2
  exit 2
fi

repo="${FLASH_GORDON_REPO:-$HOME/flash-gordon}"
cd "$repo"

make -j"$(nproc)" flash-gordon
./flash-gordon pack --dry-run --profile pipeline-8stage-262k \
  --output /srv/flash-gordon/pipeline-8stage-262k \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  --source /models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
