$ErrorActionPreference='Stop'
$root='D:\looking-glass-labs\fg-wt\ab-harness'
$build=Join-Path $root '.build\abq'
$results=Join-Path $root 'results\hillclimb'
$runner=Join-Path $root 'tools\ab-fleet.ps1'
$targets=@('192.168.42.42','192.168.42.43','192.168.42.44','192.168.42.45','192.168.42.46','192.168.42.47','192.168.42.48','192.168.42.49')
foreach($probe in @('warm','near262k','warm-fixed','near262k-fixed-interrupted','warm-forensic','near262k-final')){
    if(@(Get-ChildItem "$results\ab-memory\$probe" -Filter '*.tgz').Count-ne8){
        throw "Archive all eight $probe samplers before cleanup."
    }
}
$verified=Get-Content "$results\ab-current\verify.json" -Raw|ConvertFrom-Json
if(@($verified).Count-ne8-or@($verified|Where-Object ExitCode -ne 0).Count){throw 'Verify final serving identity first.'}
$text=@'
set -euo pipefail
root=/home/xander/hc-abq-20261015-memory
if test -e "$root" || test -L "$root"; then
test -d "$root" && test -O "$root" && test ! -L "$root"
test "$(readlink -f "$root")" = "$root"
for file in "$root"/monitor*.pid; do
  test -f "$file" || continue
  pid=$(cat "$file")
  [[ $pid =~ ^[0-9]+$ ]]
  test -e "/proc/$pid/exe" || continue
  test -O "/proc/$pid"
  if tr '\0' '\n' <"/proc/$pid/cmdline" | grep -Fx "$root/memory.py" >/dev/null; then
    kill -TERM "$pid"
    for i in $(seq 1 10); do test -e "/proc/$pid/exe" || break; sleep 1; done
    test ! -e "/proc/$pid/exe"
  fi
done
for proc in /proc/[0-9]*; do
  test -O "$proc" || continue
  [[ ${proc##*/} != "$$" && ${proc##*/} != "$PPID" ]] || continue
  if tr '\0' '\n' <"$proc/cmdline" 2>/dev/null | grep -F "$root" >/dev/null; then
    echo "Live PID ${proc##*/} references scratch" >&2; exit 1
  fi
  cwd=$(readlink "$proc/cwd" 2>/dev/null) || cwd=
  [[ $cwd != "$root" && $cwd != "$root/"* ]]
  for fd in "$proc"/fd/*; do
    value=$(readlink "$fd" 2>/dev/null) || continue
    [[ $value != "$root/"* ]]
  done
  if grep -F "$root/" "$proc/maps" >/dev/null 2>&1; then
    echo "PID ${proc##*/} maps scratch" >&2; exit 1
  fi
done
find "$root" -maxdepth 1 -type f -printf 'ARCHIVED %f %s\n'
rm -rf -- "$root"
test ! -e "$root"
fi
pack=/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04
test -d "$pack" && test ! -L "$pack"
test "$(readlink -f "$pack")" = "$pack"
for file in "$pack"/qsa-owner-rank-*-s1.state; do
  test -e "$file" || continue
  test -f "$file" && test -O "$file" && test ! -L "$file"
  for proc in /proc/[0-9]*; do
    test -O "$proc" || continue
    for fd in "$proc"/fd/*; do
      value=$(readlink "$fd" 2>/dev/null) || continue
      [[ $value != "$file" ]]
    done
    if grep -F "$file" "$proc/maps" >/dev/null 2>&1; then
      echo "PID ${proc##*/} maps candidate slot 1" >&2; exit 1
    fi
  done
  echo "REMOVING task-created stopped slot-1 cache $file"
  rm -- "$file"
done
printf '%s  %s\n' '0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252' "$pack/manifest.fgm" | sha256sum -c -
if compgen -G '/home/xander/hc-abq*' >/dev/null; then
  echo 'Unexpected hc-abq scratch remains' >&2; exit 1
fi
df -B1 /home/xander
echo CLEANUP_OK
'@
[IO.File]::WriteAllText("$build\cleanup.sh",$text.Replace("`r",''))
$r=@(& $runner -ScriptPath "$build\cleanup.sh" -Targets $targets -PassThru -CommandTimeout 180)
$r|ConvertTo-Json -Depth 12|Set-Content "$results\ab-current\cleanup.json"
if($r.Count-ne8-or@($r|Where-Object ExitCode -ne 0).Count){throw 'Cleanup failed; see receipts.'}
$health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 10
$chat=Invoke-RestMethod 'http://192.168.42.42:8080/v1/chat/completions' -Method Post `
    -ContentType 'application/json' -TimeoutSec 180 -Body (@{
      model='Qwen3.8-Flash-Next-Uncensored';temperature=0;max_tokens=16
      messages=@(@{role='user';content='/no_think Output exactly OK.'})
    }|ConvertTo-Json -Depth 5)
if($health.status -ne 'ok' -or [string]::IsNullOrWhiteSpace($chat.choices[0].message.content)){throw 'Post-cleanup generation failed.'}
@{verified_at=(Get-Date -Format o);health=$health;chat=$chat}|
    ConvertTo-Json -Depth 12|Set-Content "$results\ab-current\post-cleanup-health.json"
Write-Host 'All eight scratch roots and unused slot-1 state files removed; API still generates.'
