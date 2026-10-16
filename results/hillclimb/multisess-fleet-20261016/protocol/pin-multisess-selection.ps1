param([ValidateSet('A','B')][string]$Arm='A')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $root
$output=Join-Path $root 'results\hillclimb\multisess-fleet-20261016'
$config=Get-Content "$output\configuration.json" -Raw|ConvertFrom-Json -AsHashtable
$selected=$config[$Arm]
$run='/home/xander/flash-gordon-live/20261015-abq-operator'
$all=@(42..49|ForEach-Object{"192.168.42.$_"})
$workers=@(43..49|ForEach-Object{"192.168.42.$_"})
$destination=Join-Path $output 'selected'
New-Item -ItemType Directory -Force $destination|Out-Null
if($Arm -eq 'B'){
    $analysis=Get-Content "$output\analysis.json" -Raw|ConvertFrom-Json
    $capacity=Get-Content "$output\capacity\capacity.json" -Raw|ConvertFrom-Json
    $memory=Get-Content "$output\capacity\memory-summary.json" -Raw|ConvertFrom-Json
    if(-not ($analysis.protected_pass -and $analysis.aggregate_improves -and $analysis.aggregate_clear_win -and
        $analysis.solo_hash_identity -and $analysis.concurrent_hash_identity -and
        $capacity.complete -and $capacity.phase1.correctness_pass -and
        $capacity.single_active_agent_parity -and $memory.safe)){
        throw 'Promotion gate does not pass.'
    }
}
function New-Launcher($deployment){
    $text=@'
#!/usr/bin/env bash
set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
if [[ ${1:-start} == start ]]; then
  printf '%s  %s\n' '@@BIN@@' '@@RELEASE@@/flash-gordon' | sha256sum -c -
  printf '%s  %s\n' '@@MANIFESTSHA@@' '@@MANIFEST@@' | sha256sum -c -
fi
exec bash "$run/ab-instance-multisess.sh" "${1:-start}" '@@RELEASE@@' '@@MANIFEST@@' '@@SESSIONS@@' '0'
'@
    $text.Replace('@@BIN@@',$deployment.binary_sha256).Replace('@@RELEASE@@',$deployment.release).
        Replace('@@MANIFESTSHA@@',$deployment.manifest_sha256).Replace('@@MANIFEST@@',$deployment.manifest).
        Replace('@@SESSIONS@@',[string]$deployment.sessions).Replace("`r",'')+"`n"
}
function Enc([string]$text){[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($text))}
function Fleet([string]$name,[string]$text,[string[]]$targets,[hashtable]$extra=@{}){
    $file=Join-Path $root ".build\selection-$name.sh"
    [IO.File]::WriteAllText($file,$text.Replace("`r",''))
    $receipt=@(& "$ab\tools\ab-fleet.ps1" -ScriptPath $file -Targets $targets -PassThru -CommandTimeout 180 @extra)
    $receipt|ConvertTo-Json -Depth 10|Set-Content "$destination\$name.json"
    if($receipt.Count -ne $targets.Count -or @($receipt|Where-Object ExitCode -ne 0).Count){
        throw "Persistent selection $name failed."
    }
}
try{
    $live=Get-Content "$ab\.build\abq\live.json" -Raw|ConvertFrom-Json -AsHashtable
    if($live.release -ne $selected.release -or $live.sessions -ne $selected.sessions){
        & "$ab\tools\ab-deploy.ps1" -Deployment $selected -Arm "FINAL-$Arm"
    }
    $helper=Join-Path $root '.build\ab-instance-multisess.sh'
    [IO.File]::WriteAllText($helper,([IO.File]::ReadAllText("$ab\tools\ab-instance.sh").Replace("`r",'')))
    $launcher=New-Launcher $selected
    $fallback=New-Launcher $config.A
    [IO.File]::WriteAllText("$destination\launch-current.sh",$launcher)
    [IO.File]::WriteAllText("$destination\launch-main-mix04.sh",$fallback)
    $identity=@{};foreach($key in $selected.Keys){$identity[$key]=$selected[$key]}
    $identity['source_commit']=if($Arm -eq 'A'){'a63a166'}else{'2d0f8a7'}
    $identity['selected_at']=[DateTimeOffset]::Now.ToString('o')
    $identity['qualification']='MULTI_SESSION_FLEET.md'
    $identity|ConvertTo-Json -Depth 8|Set-Content "$destination\current.json"
    $install=@'
set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
printf '%s' '@@FALLBACK@@' | base64 -d >"$run/launch-main-mix04.sh"
printf '%s' '@@FALLBACKJSON@@' | base64 -d >"$run/main-mix04.json"
printf '%s' '@@LAUNCHER@@' | base64 -d >"$run/launch-current.new"
chmod 755 "$run/launch-main-mix04.sh" "$run/launch-current.new"
mv "$run/launch-current.new" "$run/launch-current.sh"
printf '%s' '@@IDENTITY@@' | base64 -d >"$run/current.new.json"
mv "$run/current.new.json" "$run/current.json"
sha256sum "$run/launch-current.sh" "$run/current.json" "$run/ab-instance-multisess.sh"
'@
    $install=$install.Replace('@@FALLBACK@@',(Enc $fallback)).
        Replace('@@FALLBACKJSON@@',(Enc ($config.A|ConvertTo-Json -Depth 8))).
        Replace('@@LAUNCHER@@',(Enc $launcher)).
        Replace('@@IDENTITY@@',(Enc ($identity|ConvertTo-Json -Depth 8)))
    Fleet 'install' $install $all @{UploadPath=$helper;UploadDestination="$run/ab-instance-multisess.sh"}
    Fleet 'restart-stop' "bash '$run/launch-current.sh' stop`n" $all
    Fleet 'restart-workers' "bash '$run/launch-current.sh' start`n" $workers
    Fleet 'restart-coordinator' "bash '$run/launch-current.sh' start`n" @('192.168.42.42')
    $ready=$false
    for($i=0;$i -lt 180;$i++){
        try{
            if((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status -eq 'ok'){
                $ready=$true;break
            }
        }catch{}
        Start-Sleep -Seconds 2
    }
    if(-not $ready){throw 'Persistent API readiness timeout'}
    $body=@{model='Qwen3.8-Flash-Next-Uncensored';temperature=0;seed=1729;max_tokens=32
        user='multisess-final-verification';messages=@(
            @{role='system';content='Answer directly and briefly.'},
            @{role='user';content='/no_think Output exactly OK.'})}
    $reply=Invoke-RestMethod 'http://192.168.42.42:8080/v1/chat/completions' -Method Post `
        -ContentType 'application/json' -Body ($body|ConvertTo-Json -Depth 8) -TimeoutSec 120
    if(-not $reply.choices[0].message.content.Trim() -or $reply.usage.completion_tokens -lt 1){
        throw 'Persistent API did not generate visible content'
    }
    @{health=(Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 10)
        models=(Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 10)
        reply=$reply;timestamp=[DateTimeOffset]::Now.ToString('o')} |
        ConvertTo-Json -Depth 15|Set-Content "$destination\api-health.json"
    if($Arm -eq 'B'){
        & pwsh -NoProfile -File "$root\tools\hillclimb.ps1" -ConcurrentGeneration `
            -ConcurrentRepeats 1 -ConcurrentTokens 64 -ChatContextTokens 64 -AgentContextTokens 64 `
            -PromptSalt 'multisess-20261016-restart' -RequestTimeoutSeconds 120 `
            -ResultsDirectory "$destination\restart-concurrency" *> "$destination\restart-concurrency.log"
        if($LASTEXITCODE -ne 0){throw 'Restarted two-session API failed concurrent generation'}
        $restartFile=Get-ChildItem "$destination\restart-concurrency" -Filter '*.json' |
            Sort-Object LastWriteTime|Select-Object -Last 1
        $restart=Get-Content $restartFile.FullName -Raw|ConvertFrom-Json
        if(-not $restart.concurrency.complete -or -not $restart.metadata.correctness_pass){
            throw 'Restarted two-session API lacks complete correct concurrent generation'
        }
    }
    $verify=@'
set -euo pipefail
run=/home/xander/flash-gordon-live/20261015-abq-operator
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
pid=$(cat "$run/rank-$rank.pid")
[[ $(readlink "/proc/$pid/exe") == '@@RELEASE@@/flash-gordon' ]]
printf '%s  %s\n' '@@BIN@@' "/proc/$pid/exe" | sha256sum -c -
printf '%s  %s\n' '@@MFSHA@@' '@@MF@@' | sha256sum -c -
mapfile -d '' -t args <"/proc/$pid/cmdline"
[[ ${args[1]} == "$([[ $rank == 0 ]] && echo api || echo rank)" ]]
found=false
context=false
gpu_index=false
cache=false
session_flag=false
for ((i=0;i+1<${#args[@]};i++)); do
  if [[ ${args[i]} == --manifest && ${args[i+1]} == '@@MF@@' ]]; then found=true; fi
  if [[ ${args[i]} == --context-tokens && ${args[i+1]} == 262144 ]]; then context=true; fi
  if [[ ${args[i]} == --gpu-index-tokens && ${args[i+1]} == 262144 ]]; then gpu_index=true; fi
  if [[ ${args[i]} == --qsa-page-cache-mib && ${args[i+1]} == 700 ]]; then cache=true; fi
  if [[ ${args[i]} == --sessions ]]; then
    [[ ${args[i+1]} == '@@SESSIONS@@' ]]
    session_flag=true
  fi
done
$found
if [[ $rank == 0 ]]; then
  $context; $gpu_index; $cache
  if [[ '@@SESSIONS@@' == 2 ]]; then $session_flag; fi
fi
printf 'rank=%s pid=%s argv=' "$rank" "$pid"; printf '%s ' "${args[@]}"; echo
shader=false
wide=false
trace=false
while IFS= read -r -d '' entry; do
  key=${entry%%=*}
  case "$key" in
    FG_SHADER_DIR) [[ $entry == 'FG_SHADER_DIR=@@RELEASE@@/vulkan' ]]; shader=true; printf 'ENV %s\n' "$entry" ;;
    FG_EXPERT_Q8_WIDE) [[ $entry == FG_EXPERT_Q8_WIDE=0 ]]; wide=true; printf 'ENV %s\n' "$entry" ;;
    FG_RING_TRACE) [[ $rank == 0 && $entry == FG_RING_TRACE=1 ]]; trace=true; printf 'ENV %s\n' "$entry" ;;
    FG_*) echo "Unexpected flag name $key" >&2; exit 1 ;;
    LD_PRELOAD) echo 'Unexpected LD_PRELOAD' >&2; exit 1 ;;
  esac
done <"/proc/$pid/environ"
$shader; $wide
if [[ $rank == 0 ]]; then $trace; fi
sha256sum "$run/current.json" "$run/launch-current.sh" "$run/ab-instance-multisess.sh"
grep -E 'MemAvailable|SwapFree|SwapTotal' /proc/meminfo
'@
    $verify=$verify.Replace('@@RELEASE@@',$selected.release).Replace('@@BIN@@',$selected.binary_sha256).
        Replace('@@MF@@',$selected.manifest).Replace('@@MFSHA@@',$selected.manifest_sha256).
        Replace('@@SESSIONS@@',[string]$selected.sessions)
    Fleet 'verify' $verify $all
    Fleet 'coordinator-log' "sha256sum '$run/rank-0.log'`n" @('192.168.42.42') @{
        DownloadSource="$run/rank-0.log";DownloadPath="$destination\coordinator.log"}
    $identity|ConvertTo-Json -Depth 8|Set-Content "$output\selection.json"
    $selected|ConvertTo-Json -Depth 8|Set-Content "$ab\.build\abq\live.json"
    Write-Host "PERSISTENT_SELECTION_VERIFIED arm=$Arm sessions=$($selected.sessions)"
}catch{
    $_|Out-String|Set-Content "$destination\failure-$Arm.txt"
    if($Arm -eq 'B'){
        & "$ab\tools\ab-deploy.ps1" -Deployment $config.A -Arm 'PERSISTENCE-FAILURE-RESTORE'
        & pwsh -NoProfile -File $PSCommandPath -Arm A
    }
    throw
}
