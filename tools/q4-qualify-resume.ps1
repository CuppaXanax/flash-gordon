#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidatePattern('^[a-z0-9-]+$')][string]$Label='qualify-ordinary'
)
$ErrorActionPreference='Stop'
$OutputDir=(Get-Item -LiteralPath $OutputDir).FullName
$plans=@(Get-Content (Join-Path $OutputDir 'preflight.json') -Raw | ConvertFrom-Json)
$root=Split-Path $PSScriptRoot
$archive=Join-Path $OutputDir 'repacker-fix.tar'
$command=Join-Path $OutputDir 'resume-command.sh'
& tar -cf $archive -C $root tools/q4_repack.c tests/test_q4_repack.c
if($LASTEXITCODE){throw 'Cannot archive repacker fix'}
$archiveSha=(Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$results=@()
try {
    foreach($plan in $plans){
        $rows=[regex]::Matches($plan.Stdout,'(?m)^PLAN_RANK rank=(\d+) layers=(\d+) .*?endpoint=([^\s]+)')
        $owned=@($rows | Where-Object {$_.Groups[3].Value.Split(':')[0] -eq $plan.IP})
        if($owned.Count -ne 1 -or $rows.Count -ne 8){throw 'Invalid manifest rank map'}
        $baseline=[regex]::Match($plan.Stdout,'BASELINE_SHA ([a-f0-9]{64})').Groups[1].Value
        if($baseline.Length -ne 64){throw 'Missing sealed input digest'}
        $text=@'
set -euo pipefail
root=/home/xander/hc-q4/stage-__LABEL__
printf '%s  %s\n' __PATCH_SHA__ "$root/repacker-fix.tar" | sha256sum -c -
tar -xf "$root/repacker-fix.tar" -C "$root/code"
cd "$root/code"
make -j4 tools/q4-repack tests/test_q4_repack > "$root/repacker-fix.log" 2>&1 || { tail -60 "$root/repacker-fix.log"; exit 1; }
./tests/test_q4_repack
manifest=$(cat "$root/source-manifest-path")
printf '%s  %s\n' __BASELINE_SHA__ "$manifest" | sha256sum -c -
if test ! -s "$root/pack/manifest.fgm"; then
  if test -d "$root/pack"; then rmdir -- "$root/pack"; fi
  ./tools/q4-repack "$manifest" /home/xander/fg-ring-pack-abliterated \
    "$root/dense-q4.gguf" "$root/pack" __RANK__ --keep-q8-head --cook-dense --reflink
fi
./tools/q4-repack "$root/pack/manifest.fgm" "$root/pack" "$root/dense-q4.gguf" \
  "$root/unused" --keep-q8-head --cook-dense --plan | tee "$root/sealed-resident-ledger.txt"
printf 'CANDIDATE_SHA '
sha256sum "$root/pack/manifest.fgm" | cut -d' ' -f1
'@
        $text=$text.Replace('__LABEL__',$Label).Replace('__PATCH_SHA__',$archiveSha).
          Replace('__BASELINE_SHA__',$baseline).Replace('__RANK__',$owned[0].Groups[1].Value)
        [IO.File]::WriteAllText($command,$text.Replace("`r",''))
        $records=@(& (Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1') -ScriptPath $command `
          -Targets @($plan.IP) -UploadPath $archive `
          -UploadDestination "/home/xander/hc-q4/stage-$Label/repacker-fix.tar" -PassThru)
        $results+=$records
        $results | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutputDir 'resumed.json')
        if(@($records | Where-Object ExitCode -ne 0).Count){throw "Repacker resume failed on $($plan.IP)"}
    }
    $hashes=@($results | ForEach-Object {
        $m=[regex]::Match($_.Stdout,'CANDIDATE_SHA ([a-f0-9]{64})')
        if(-not $m.Success){throw 'Missing candidate digest'}
        $m.Groups[1].Value
    } | Sort-Object -Unique)
    if($results.Count -ne 8 -or $hashes.Count -ne 1){throw 'Incomplete or inconsistent sealed owner packs'}
    $results | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutputDir 'staged.json')
    Write-Host "SEALED owners=8 manifest=$($hashes[0])"
} finally {
    Remove-Item -LiteralPath $command -ErrorAction SilentlyContinue
}
