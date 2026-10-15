#requires -Version 7.0
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^/home/xander/fg-ring-pack-abliterated-q4(-protected|-teacher|-control|-ablate-[a-z0-9-]+|-mixed)?/manifest\.fgm$')]
    [string]$Manifest,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9-]+$')][string]$Tag,
    [Parameter(Mandatory)][string]$OutputDir,
    [ValidatePattern('^/home/xander/flash-gordon-live/[0-9]{8}-q4[-a-zA-Z0-9]*$')]
    [string]$Release='/home/xander/flash-gordon-live/20261015-q4-ablation',
    [switch]$WallOnly
)
$ErrorActionPreference='Stop'
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir|Out-Null
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$helper=Join-Path $PSScriptRoot 'decode-instance.ps1'
$original='/home/xander/fg-ring-pack-abliterated/manifest.fgm'
$command=Join-Path $OutputDir "$Tag-command.sh"
function Instance([string]$Action,[string]$Path,[string]$Mode,[string]$Binary=$release){
    & $helper -Action $Action -Manifest $Path -Release $Binary -Mode $Mode `
      -CaptureTag $Tag -Ranks @(0..7) -FleetScript $fleet
}
function Ready {
    for($i=0;$i -lt 120;$i++){
        try{if((Invoke-RestMethod http://192.168.42.42:8080/health -TimeoutSec 2).status -eq 'ok'){return}}catch{}
        Start-Sleep -Seconds 3
    }
    throw 'Profile API readiness timeout'
}
function Fleet([string]$Text,[string]$IP,[string]$Download='',[string]$Source=''){
    [IO.File]::WriteAllText($command,$Text.Replace("`r",''))
    $args=@{ScriptPath=$command;Targets=@($IP);PassThru=$true}
    if($Download){$args.DownloadPath=$Download;$args.DownloadSource=$Source}
    $r=@(& $fleet @args)
    $r|ConvertTo-Json -Depth 5 -Compress|Add-Content (Join-Path $OutputDir "$Tag-fleet.jsonl")
    if(@($r|Where-Object ExitCode -ne 0).Count){throw 'Profile fleet operation failed'}
}
try{
    Instance stop $original plain
    Instance stop $Manifest plain
    Instance start $Manifest $(if($WallOnly){'plain'}else{'prefill'})
    Ready
    $model=(Invoke-RestMethod http://192.168.42.42:8080/v1/models).data[0].id
    foreach($tokens in @(4096,131072)){
        if($tokens -eq 131072 -and -not $WallOnly){
            foreach($rank in 0..7){
                $log="$release/operator/rank-$rank-prefill-$Tag.log"
                $offset="/home/xander/hc-q4/profile-$Tag-rank-$rank.offset"
                Fleet "set -e;stat -c '%s' '$log' > '$offset'" "192.168.42.$(42+$rank)"
            }
        }
        $nonce="q4-profile-fixed-20261015-$tokens"
        $content="/no_think nonce $nonce. "+(' hello'*($tokens-48))+"`nReply with one word: done."
        $body=[ordered]@{model=$model;temperature=0;top_p=1;seed=1729;max_tokens=1;stream=$false
            messages=@([ordered]@{role='system';content="Isolated benchmark $nonce. Answer briefly."},
                       [ordered]@{role='user';content=$content})}|ConvertTo-Json -Depth 6 -Compress
        $bytes=[Text.Encoding]::UTF8.GetBytes($body)
        $requestSHA=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
        $timer=[Diagnostics.Stopwatch]::StartNew()
        $response=Invoke-WebRequest http://192.168.42.42:8080/v1/chat/completions `
            -Method Post -ContentType 'application/json' -Body $bytes -TimeoutSec 1800
        $timer.Stop()
        @{target_tokens=$tokens;request_sha256=$requestSHA;kernel_profile=(-not $WallOnly)
          wall_seconds=$timer.Elapsed.TotalSeconds;headers=$response.Headers
          response=($response.Content|ConvertFrom-Json)}|ConvertTo-Json -Depth 12|
            Set-Content (Join-Path $OutputDir "$Tag-$tokens.json")
    }
    if(-not $WallOnly){foreach($rank in 0..7){
        $log="$release/operator/rank-$rank-prefill-$Tag.log"
        $remote="/home/xander/hc-q4/profile-$Tag-rank-$rank.log"
        $offset="/home/xander/hc-q4/profile-$Tag-rank-$rank.offset"
        Fleet "set -e;offset=`$(cat '$offset');tail -c +`$((offset+1)) '$log'|grep '^PREFILL_' > '$remote';test -s '$remote'" `
          "192.168.42.$(42+$rank)" (Join-Path $OutputDir "$Tag-rank-$rank.log") $remote
    }}
}finally{
    $stopped=$false
    try{
        Instance stop $Manifest plain
        $stopped=$true
    }finally{
        try{
            if($stopped){
                $pack=$Manifest.Substring(0,$Manifest.LastIndexOf('/'))
                foreach($rank in 0..7){
                    Fleet "set -e;find '$pack' -maxdepth 1 -type f -name 'qsa-*.state*' -delete" `
                      "192.168.42.$(42+$rank)"
                }
            }
        }finally{
            Instance start $original plain '/home/xander/flash-gordon-live/20260926-abl2'
            Ready
            Remove-Item $command -ErrorAction SilentlyContinue
        }
    }
}
