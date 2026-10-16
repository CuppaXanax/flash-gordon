$ErrorActionPreference='Stop'
$root='D:\looking-glass-labs\fg-wt\ab-harness'
$build=Join-Path $root '.build\abq'
$results=Join-Path $root 'results\hillclimb'
$runner=Join-Path $root 'tools\ab-fleet.ps1'
$targets=@(42..49|ForEach-Object{"192.168.42.$_"})
$text=@'
set -euo pipefail
root=/home/xander/hc-abq-20261015-memory
rank=$(($(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)-42))
cd "$root"
pid=$(cat monitor-near262k-final.pid)
[[ $pid =~ ^[0-9]+$ ]]
if test -e "/proc/$pid/exe"; then
  test -O "/proc/$pid"
  tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fx "$root/memory.py" >/dev/null
  kill -TERM "$pid"
  for i in $(seq 1 10); do test -e "/proc/$pid/exe" || break; sleep 1; done
  test ! -e "/proc/$pid/exe"
fi
cp "/home/xander/flash-gordon-live/20261015-abq-operator/rank-$rank.log" near262k-final-rank.log
tar -czf near262k-final.tgz near262k-final.jsonl near262k-final.err near262k-final-rank.log
tail -n 3 near262k-final-rank.log
'@
[IO.File]::WriteAllText("$build\capture-final.sh",$text.Replace("`r",''))
try {
    $r=@(& $runner -ScriptPath "$build\capture-final.sh" -Targets $targets -PassThru -CommandTimeout 180)
    $r|ConvertTo-Json -Depth 10|Set-Content "$results\ab-ops\capture-final.json"
    if($r.Count -ne 8 -or @($r|Where-Object ExitCode -ne 0).Count){throw 'Final interrupted capture failed.'}
} finally {
    & "$root\tools\ab-restore.ps1"
    if(-not $?){throw 'Original restore failed after user cancellation.'}
}
$destination=Join-Path $results 'ab-memory\near262k-final'
$null=New-Item -ItemType Directory -Force $destination
'true'|Set-Content "$build\final-download.sh"
foreach($target in $targets){
    $r=@(& $runner -ScriptPath "$build\final-download.sh" -Targets @($target) `
      -DownloadSource '/home/xander/hc-abq-20261015-memory/near262k-final.tgz' `
      -DownloadPath "$destination\$target.tgz" -PassThru)
    if($r.Count -ne 1 -or $r[0].ExitCode){throw "Final archive download failed: $target"}
}
@{probe='near262k-final';complete=$false;cancelled_by_user=$true;restored_original=$true;finished=(Get-Date -Format o)}|
  ConvertTo-Json|Set-Content "$results\ab-ops\final-capacity-outcome.json"
Write-Host 'Final diagnostic interrupted, all eight archives secured, original restored.'
