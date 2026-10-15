#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceArchive,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidatePattern('^[a-z0-9-]+$')][string]$Label = "q4-compact",
    [ValidatePattern('^[a-z0-9-]+$')][string]$OverlayName = 'dense-q4',
    [ValidatePattern('^/home/xander/hc-q4[-a-z0-9/]*$')]
    [string]$OverlayDirectory = '/home/xander/hc-q4',
    [switch]$ParallelOwners,
    [switch]$AfterLease
)
$ErrorActionPreference = "Stop"
if (-not $AfterLease) { throw "No production access without -AfterLease. Run only after .42-.49 are released." }
$SourceArchive = (Get-Item -LiteralPath $SourceArchive).FullName
$OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$fleet = Join-Path $PSScriptRoot "Invoke-Q4Fleet.ps1"
$command = Join-Path $OutputDir "stage-command.sh"
$targets = @(42..49 | ForEach-Object { "192.168.42.$_" })
$root = "/home/xander/hc-q4/stage-$Label"
$ownerCommands = @()
function Run-Remote([string[]]$Addresses, [string]$Text, [hashtable]$Transfer = @{}) {
    [IO.File]::WriteAllText($command, $Text, [Text.UTF8Encoding]::new($false))
    $result = @(& $fleet -ScriptPath $command -Targets $Addresses -PassThru @Transfer)
    if (@($result | Where-Object ExitCode -ne 0).Count) {
        throw "Staging failed; no services were changed: $($result | ConvertTo-Json -Depth 5)"
    }
    return $result
}
try {
    $overlay = Join-Path $OutputDir "$OverlayName.gguf"
    $fetchScript = @'
set -euo pipefail
test -s /home/xander/hc-q4/dense-q4.gguf
printf 'OVERLAY_SHA '
sha256sum /home/xander/hc-q4/dense-q4.gguf | cut -d' ' -f1
'@
    $overlayRemote = "$OverlayDirectory/$OverlayName.gguf"
    $fetch = Run-Remote @('192.168.42.53') ($fetchScript.Replace('/home/xander/hc-q4/dense-q4.gguf',$overlayRemote)) @{
        DownloadSource=$overlayRemote; DownloadPath=$overlay
    }
    $match = [regex]::Match($fetch[0].Stdout, 'OVERLAY_SHA ([a-f0-9]{64})')
    $overlaySha = (Get-FileHash -LiteralPath $overlay -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not $match.Success -or $match.Groups[1].Value -ne $overlaySha) { throw "Overlay transfer SHA256 mismatch" }
    $codeSha = (Get-FileHash -LiteralPath $SourceArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    Run-Remote $targets "set -eu; mkdir -p '$root'; test ! -e '$root/pack'" | Out-Null
    Run-Remote $targets "set -eu; test -s '$root/code.tar'" @{
        UploadPath=$SourceArchive; UploadDestination="$root/code.tar"
    } | Out-Null
    $preflight = @'
set -euo pipefail
root=/home/xander/hc-q4/stage-__LABEL__
cd "$root"
printf '%s  %s\n' __CODE_SHA__ code.tar | sha256sum -c -
printf '%s  %s\n' __OVERLAY_SHA__ dense-q4.gguf | sha256sum -c -
old=/home/xander/fg-ring-pack-abliterated
test -d "$old"
if test -f "$old/manifest.fgm"; then
  manifest="$old/manifest.fgm"
else
  mapfile -t files < <(find -L "$old" -maxdepth 1 -type f -name '*.fgm')
  test "${#files[@]}" -eq 1
  manifest="${files[0]}"
fi
printf '%s\n' "$(realpath "$manifest")" > source-manifest-path
printf 'BASELINE_SHA '
sha256sum "$manifest" | cut -d' ' -f1
mkdir code
tar -xf code.tar -C code
cd code
make -j8 flash-gordon tools/q4-repack tools/q4-eval-batch tests/test_q4_repack shaders > "$root/build.log" 2>&1 || { tail -60 "$root/build.log"; exit 1; }
./tests/test_q4_repack
./tools/q4-repack "$manifest" "$old" "$root/dense-q4.gguf" "$root/unused" \
  --keep-q8-head --cook-dense --plan | tee "$root/plan.txt"
'@
    $preflight = $preflight.Replace('__LABEL__',$Label).Replace('__CODE_SHA__',$codeSha).Replace('__OVERLAY_SHA__',$overlaySha)
    $plans = Run-Remote $targets $preflight @{
        UploadPath=$overlay; UploadDestination="$root/dense-q4.gguf"
    }
    $plans | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputDir "preflight.json")
    $baselineHashes = @($plans | ForEach-Object {
        $m = [regex]::Match($_.Stdout, 'BASELINE_SHA ([a-f0-9]{64})')
        if (-not $m.Success) { throw "Missing baseline manifest SHA" }
        $m.Groups[1].Value
    } | Sort-Object -Unique)
    if ($baselineHashes.Count -ne 1) { throw "Owners disagree about the baseline manifest" }
    $rankMap = @{}
    foreach ($plan in $plans) {
        $rows = [regex]::Matches($plan.Stdout, '(?m)^PLAN_RANK rank=(\d+) layers=(\d+) .*?endpoint=([^\s]+)')
        if ($rows.Count -ne 8 -or @($rows | Where-Object { $_.Groups[2].Value -ne '6' }).Count) {
            throw "Expected the eight-rank, six-layers-per-rank layout"
        }
        $owned = @($rows | Where-Object { $_.Groups[3].Value.Split(':')[0] -eq $plan.IP })
        if ($owned.Count -ne 1) { throw "Cannot derive manifest rank for $($plan.IP); do not guess from IP arithmetic" }
        $rankMap[$plan.IP] = [int]$owned[0].Groups[1].Value
    }
    if (@($rankMap.Values | Sort-Object -Unique).Count -ne 8) { throw "Duplicate rank ownership" }
    $results = @()
    foreach ($ip in $targets) {
        $pack = @'
set -euo pipefail
root=/home/xander/hc-q4/stage-__LABEL__
manifest=$(cat "$root/source-manifest-path")
printf '%s  %s\n' __BASELINE_SHA__ "$manifest" | sha256sum -c -
cd "$root/code"
unset FG_DENSE_SOA FG_DENSE_K_COOKED
./tools/q4-repack "$manifest" /home/xander/fg-ring-pack-abliterated \
  "$root/dense-q4.gguf" "$root/pack" __RANK__ --keep-q8-head --cook-dense --reflink
./tools/q4-repack "$root/pack/manifest.fgm" "$root/pack" "$root/dense-q4.gguf" \
  "$root/unused" --keep-q8-head --cook-dense --plan | tee "$root/sealed-resident-ledger.txt"
printf 'CANDIDATE_SHA '
sha256sum "$root/pack/manifest.fgm" | cut -d' ' -f1
'@
        $pack = $pack.Replace('__LABEL__',$Label).Replace('__BASELINE_SHA__',$baselineHashes[0]).Replace('__RANK__',"$($rankMap[$ip])")
        if ($ParallelOwners) {
            $path = Join-Path $OutputDir "pack-$ip.sh"
            [IO.File]::WriteAllText($path,$pack.Replace("`r",''))
            $ownerCommands += @{IP=$ip; Path=$path}
        } else {
            $results += Run-Remote @($ip) $pack
            $results | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputDir "staged.json")
        }
    }
    if ($ParallelOwners) {
        $results = @($ownerCommands | ForEach-Object -Parallel {
            & $using:fleet -ScriptPath $_.Path -Targets @($_.IP) -PassThru
        } -ThrottleLimit 8)
        $results | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputDir "staged.json")
        if ($results.Count -ne 8 -or @($results | Where-Object ExitCode -ne 0).Count) {
            throw "Owner-local repack failed; inspect staged.json. No services were changed."
        }
    }
    $candidateHashes = @($results | ForEach-Object {
        $m = [regex]::Match($_.Stdout, 'CANDIDATE_SHA ([a-f0-9]{64})')
        if (-not $m.Success) { throw "Missing candidate manifest SHA" }
        $m.Groups[1].Value
    } | Sort-Object -Unique)
    if ($candidateHashes.Count -ne 1) { throw "Candidate manifest hashes disagree; do not start workers" }
    Write-Host "Staged eight owner-local packs: $root/pack; manifest SHA $($candidateHashes[0])"
    Write-Host "Not deployed. Baseline remains required for external-artifact symlinks. Run KLD/top-1 gates separately."
} finally {
    Remove-Item -LiteralPath $command -ErrorAction SilentlyContinue
    foreach ($item in $ownerCommands) {
        Remove-Item -LiteralPath $item.Path -ErrorAction SilentlyContinue
    }
}
