#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for file in .build/gap/*.sh results/hillclimb/prod-gap/current/*.sh results/hillclimb/ab-current/launch-current.sh; do
    bash -n "$file"
done
printf '%s\n' 'Bash syntax checks passed.'
