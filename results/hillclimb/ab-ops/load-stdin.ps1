$ErrorActionPreference = 'Stop'
$root = 'D:\looking-glass-labs\fg-wt\ab-harness'
$build = Join-Path $root '.build\abq'
$results = Join-Path $root 'results\hillclimb'
$runner = Join-Path $root 'tools\ab-fleet.ps1'
$targets = @('192.168.42.42','192.168.42.43','192.168.42.44','192.168.42.45','192.168.42.46','192.168.42.47','192.168.42.48','192.168.42.49')
$config = Get-Content "$build\multi-fixed-config.json" -Raw | ConvertFrom-Json -AsHashtable
function Fleet([string]$Text,[string]$Tag) {
    $path = Join-Path $build "$Tag.sh"
    [IO.File]::WriteAllText($path,$Text.Replace("`r",''))
    $r = @(& $runner -ScriptPath $path -Targets $targets -PassThru -CommandTimeout 180)
    $r | ConvertTo-Json -Depth 12 | Set-Content "$build\$Tag.json"
    if ($r.Count -ne 8 -or @($r | Where-Object ExitCode -ne 0).Count) { throw "Fleet $Tag failed." }
}
$outcome = @()
foreach ($tag in @('warm-forensic','near262k-final')) {
    $row = @{probe=$tag; started=(Get-Date -Format o); complete=$false}
    $sampling = $false
    try {
        & "$root\tools\ab-deploy.ps1" -Deployment $config.B -Arm "diagnostic-$tag"
        if (-not $?) { throw 'Diagnostic deployment failed.' }
        $text = @'
set -euo pipefail
root=/home/xander/hc-abq-20261015-memory
rank=$(($(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)-42))
pid=$(cat "/home/xander/flash-gordon-live/20261015-abq-operator/rank-$rank.pid")
cd "$root"
test ! -e __TAG__.jsonl
nohup python3 "$root/memory.py" --pid "$pid" --manifest '__MANIFEST__' --release '__RELEASE__' --seconds 14400 >__TAG__.jsonl 2>__TAG__.err </dev/null &
echo "$!" >monitor-__TAG__.pid
sleep 3
kill -0 "$(cat monitor-__TAG__.pid)"
test -s __TAG__.jsonl
'@
        $sampling=$true
        Fleet ($text.Replace('__TAG__',$tag).Replace('__MANIFEST__',$config.B.manifest).Replace('__RELEASE__',$config.B.release)) "diagnostic-$tag-start"
        $extra = if ($tag -eq 'near262k-final') {
            @('-MixedPrefill','-CapacityProbe','-AgentContextTokens','260000','-ConcurrentRepeats','1','-RequestTimeoutSeconds','10800')
        } else { @('-AgentContextTokens','4096','-ConcurrentRepeats','1','-RequestTimeoutSeconds','300') }
        & pwsh -NoProfile -File 'D:\looking-glass-labs\fg-wt\multi-session-int\tools\hillclimb.ps1' `
            -ConcurrentGeneration -ChatContextTokens 64 -ConcurrentTokens 128 -ChatDelayMilliseconds 200 `
            -ResultsDirectory "$results\ab-multi-$tag" @extra *> "$build\ab-multi-$tag.log"
        $row.exit_code=$LASTEXITCODE
        $row.complete=$LASTEXITCODE -in @(0,2)
    } catch { $row.error=$_.Exception.Message }
    finally {
        try {
            if ($sampling) {
                $text = @'
set -euo pipefail
root=/home/xander/hc-abq-20261015-memory
rank=$(($(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)-42))
cd "$root"
cp "/home/xander/flash-gordon-live/20261015-abq-operator/rank-$rank.log" "__TAG__-rank.log"
if test -f "monitor-__TAG__.pid"; then
  pid=$(cat "monitor-__TAG__.pid")
  [[ $pid =~ ^[0-9]+$ ]]
  if test -e "/proc/$pid/exe"; then
    test -O "/proc/$pid"
    tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fx "$root/memory.py" >/dev/null
    kill -TERM "$pid"
    for i in $(seq 1 10); do test -e "/proc/$pid/exe" || break; sleep 1; done
    test ! -e "/proc/$pid/exe"
  fi
fi
tar -czf "__TAG__.tgz" "__TAG__.jsonl" "__TAG__.err" "__TAG__-rank.log"
tail -n 70 "__TAG__-rank.log"
cat "__TAG__.err"
grep -E 'oom_kill|pswpin|pswpout' /proc/vmstat
grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
'@
                Fleet ($text.Replace('__TAG__',$tag)) "diagnostic-$tag-stop"
            }
        } catch { $row.capture_error=$_.Exception.Message }
        finally {
            & "$root\tools\ab-restore.ps1"
            if (-not $?) { throw 'Original restore failed after diagnostic.' }
            $row.restored_original=$true
        }
    }
    $destination=Join-Path $results "ab-memory\$tag"
    $null=New-Item -ItemType Directory -Force $destination
    'true' | Set-Content "$build\diagnostic-download.sh"
    if ($sampling) {
        foreach ($target in $targets) {
            $r=@(& $runner -ScriptPath "$build\diagnostic-download.sh" -Targets @($target) `
                -DownloadSource "/home/xander/hc-abq-20261015-memory/$tag.tgz" `
                -DownloadPath "$destination\$target.tgz" -PassThru)
            if($r.Count -ne 1 -or $r[0].ExitCode){ throw "Diagnostic download failed: $target" }
        }
    }
    $row.finished=(Get-Date -Format o)
    $outcome += $row
    $outcome | ConvertTo-Json -Depth 10 | Set-Content "$results\ab-load-final-diagnostics.json"
    Write-Host "DIAGNOSTIC $tag complete=$($row.complete) exit=$($row.exit_code) restored=$($row.restored_original)"
}
