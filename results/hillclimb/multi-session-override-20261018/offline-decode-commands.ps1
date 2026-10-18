$ErrorActionPreference='Stop'
Set-Location D:\looking-glass-labs\fg-wt\multi-session-int
$null=New-Item -ItemType Directory -Force .build\prodinclude
git show 31508ee:src/pipeline.c|Set-Content .build\pipeline-prod.c
git show 31508ee:include/fg_pipeline.h|Set-Content .build\prodinclude\fg_pipeline.h
git diff --exit-code 11c7e39 -- src\pipeline.c include\fg_pipeline.h
if($LASTEXITCODE-ne0){throw 'Candidate pipeline must match qualified source'}
wsl.exe -d Ubuntu-24.04 --cd /mnt/d/looking-glass-labs/fg-wt/multi-session-int bash -lc 'export TMPDIR="$PWD/.build"; gcc -std=c17 -O2 -D_XOPEN_SOURCE=700 -I.build/prodinclude -Iinclude .build/decode-overhead-bench.c .build/pipeline-prod.c -o .build/decode-overhead-prod && gcc -std=c17 -O2 -D_XOPEN_SOURCE=700 -Iinclude .build/decode-overhead-bench.c src/pipeline.c -o .build/decode-overhead-new && python3 .build/run-decode-overhead.py'
if($LASTEXITCODE-ne0){throw 'Offline microbench failed'}
