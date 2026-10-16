param(
    [string[]]$Phases=@('normal','trace','wall','post'),
    [string]$Library='prod-vk-wall.so'
)
$ErrorActionPreference='Stop'
. "$PSScriptRoot\common.ps1"
$all=@(42..49|ForEach-Object{"192.168.42.$_"})
$null=New-Item -ItemType Directory -Force "$output\logs"
$source=Get-Content "$root\tools\prod-drm-snapshot.py" -Raw
$snapshot="python3 - <<'PY'`n"+$source.Replace("`r",'')+"`nPY`n"
$run='/home/xander/flash-gordon-live/20261015-abq-operator'
$prepare="prepare-$([IO.Path]::GetFileNameWithoutExtension($Library))"
$null=Fleet "set -euo pipefail`nmkdir -p '$remote'`ntest -O '$remote' && test ! -L '$remote'" $prepare
foreach($ip in $all){
    $r=@(& $runner -ScriptPath "$build\$prepare.sh" -Targets @($ip) `
      -UploadPath "$build\$Library" -UploadDestination "$remote/$Library" -PassThru)
    if($r.Count -ne 1 -or $r[0].ExitCode){throw "Interposer stage failed on $ip"}
}
$null=Fleet "sha256sum '$remote/$Library'" "$Library-hashes"
function Start-Phase([string]$Name,[bool]$Trace,[bool]$Wall){
    $null=Fleet "bash '$run/launch-current.sh' stop" "$Name-stop"
    $prefix=if($Trace){"export FG_DECODE_RING_TRACE=1 FG_FABRIC_PROFILE=1 FG_GENERATE_TRACE=1 FG_OUTPUT_SPLIT_TRACE=1`n"}else{''}
    if($Wall){$prefix+="export LD_PRELOAD='$remote/$Library'`n"}
    $text=$prefix+"bash '$run/launch-current.sh' start"
    $null=Fleet $text "$Name-workers" @(1..7)
    $null=Fleet $text "$Name-coordinator" @(0)
    $ready=$false
    for($i=0;$i-lt180;$i++){
        try{if((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status-eq'ok'){$ready=$true;break}}catch{}
        Start-Sleep -Seconds 2
    }
    if(-not $ready){throw "$Name did not become ready"}
}
try{
    foreach($phase in $Phases){
        Start-Phase $phase ($phase-in@('trace','wall','engine')) ($phase-in@('wall','engine'))
        $cases=if($phase-eq'post'){@(4096)}else{@(4096,4096,4096,32768)}
        for($i=0;$i-lt$cases.Count;$i++){
            $tag="$phase-$i"
            $dir="$output\$phase"
            $null=New-Item -ItemType Directory -Force $dir
            & python "$root\tools\prod-gap-probe.py" --output "$dir\$i-prime.json" `
              --contexts $cases[$i] --repeats 1 --salt "production-gap-fixed-$i" --prime-only
            if($LASTEXITCODE){throw "$tag prime failed"}
            $null=Fleet $snapshot "$tag-before"
            & python "$root\tools\prod-gap-probe.py" --output "$dir\$i-decode.json" --decode-from "$dir\$i-prime.json"
            if($LASTEXITCODE){throw "$tag decode failed"}
            $null=Fleet $snapshot "$tag-after"
        }
        $null=Fleet "set -euo pipefail`nrank=`$((`$(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1)-42))`ngzip -c '$run/rank-'`$rank'.log' >'$remote/$phase.log.gz'" "$phase-capture"
        foreach($ip in $all){
            $r=@(& $runner -ScriptPath "$build\$prepare.sh" -Targets @($ip) `
              -DownloadSource "$remote/$phase.log.gz" -DownloadPath "$output\logs\$phase-$ip.log.gz" -PassThru)
            if($r.Count-ne1-or$r[0].ExitCode){throw "$phase download failed on $ip"}
        }
        Write-Host "PHASE $phase complete"
    }
}finally{
    $null=Fleet "bash '$run/launch-current.sh' stop" 'final-normal-stop'
    $null=Fleet "bash '$run/launch-current.sh' start" 'final-normal-workers' @(1..7)
    $null=Fleet "bash '$run/launch-current.sh' start" 'final-normal-coordinator' @(0)
}
Write-Host 'All trace phases completed; mix-04 restarted without diagnostic environment.'
