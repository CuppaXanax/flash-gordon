#requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('Plan','Run','Restore')][string]$Action='Plan',
    [switch]$LeaseConfirmed,
    [ValidatePattern('^[A-Za-z0-9_-]{4,80}$')][string]$LeaseId='',
    [string]$Artifacts='',
    [string]$ProductionRunner='D:\looking-glass-labs\fg-wt\ab-harness\tools\ab-fleet.ps1',
    [string]$DraftRunner='',
    [ValidatePattern('^http://(192\.168\.42\.42|127\.0\.0\.1)(:\d+)?$')]
    [string]$ApiBaseUrl='http://192.168.42.42:8080'
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(-not $DraftRunner){$DraftRunner=Join-Path $PSScriptRoot 'Invoke-MTPFleet.ps1'}
if(-not $Artifacts){$Artifacts=Join-Path $repo '.build\mtp-lease'}
if($Action -ne 'Restore'){
    $prepared=Get-Content (Join-Path $Artifacts 'prepared.json') -Raw | ConvertFrom-Json -AsHashtable
    $release=Join-Path $Artifacts 'release.tar.gz'
    if((Get-FileHash $release -Algorithm SHA256).Hash.ToLowerInvariant() -ne $prepared.files['release.tar.gz']){
        throw 'Release differs from the prebuilt/validated artifact.'
    }
    if($prepared.files['native-q8.gguf'] -ne '536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b'){
        throw 'Unqualified native overlay identity.'
    }
    if(-not (Test-Path (Join-Path $Artifacts 'corpus.tar'))){throw 'Missing frozen corpus archive.'}
    if((Get-FileHash (Join-Path $Artifacts 'corpus.tar')).Hash.ToLowerInvariant() -ne $prepared.corpus_sha256){
        throw 'Corpus archive differs from the frozen preparation.'
    }
}
if($Action -eq 'Plan'){
    [ordered]@{
        productionAccess=$false; ring='192.168.42.42-.49'; draftBlade='192.168.42.51'
        chatCases=20; codeCases=20; generatedTokens=259; acceptanceAnchors=256; k=@(1,2,3)
        greedyCasesPerCategory=4; greedyPairedRuns=24; greedyTokens=256
        verifier='causal N=1 reference, not small-N serving qualification'
        workDeadlineMinutes=35; watchdogWorkerMinute=38; watchdogCoordinatorMinute=39
        wallBudgetMinutes=45; restore='main a63a166 + sealed mix-04, sessions1 wide0'
        releaseSha256=$prepared.files['release.tar.gz']
    } | ConvertTo-Json -Depth 6
    return
}
if(-not $LeaseConfirmed -or -not $LeaseId){throw 'An explicitly confirmed exclusive lease and LeaseId are required.'}
$all=@(42..49 | ForEach-Object {"192.168.42.$_"})
$workers=@(43..49 | ForEach-Object {"192.168.42.$_"})
$run=Join-Path $Artifacts "runs\$LeaseId"
if($Action -eq 'Run' -and (Test-Path $run)){throw 'Use a fresh LeaseId; previous lease evidence is immutable.'}
$null=New-Item -ItemType Directory -Force $run
$started=[DateTimeOffset]::UtcNow
$script:workEnd=$started.AddMinutes(35)
$script:restoreMode=$Action -eq 'Restore'
$script:sequence=0
$root='/home/xander/hc-mtp'
function Remote {
    param([string]$Name,[string]$Text,[string[]]$Targets,
          [string]$Upload='',[string]$Destination='',[string]$Download='',
          [string]$Local='',[int]$Timeout=120)
    $seconds=if($script:restoreMode){300}else{[int]($script:workEnd-[DateTimeOffset]::UtcNow).TotalSeconds}
    if($seconds -le 0){throw '35-minute work deadline; restoring production.'}
    $script:sequence++
    $tag='{0:D3}-{1}' -f $script:sequence,$Name
    $scriptFile=Join-Path $run "$tag.sh"
    [IO.File]::WriteAllText($scriptFile,"set -euo pipefail`n"+($Text -replace "`r",'')+"`n")
    $args=@{ScriptPath=$scriptFile;Targets=$Targets;CommandTimeout=[Math]::Min($Timeout,$seconds);PassThru=$true}
    if($Upload){$args.UploadPath=$Upload;$args.UploadDestination=$Destination}
    if($Download){$args.DownloadSource=$Download;$args.DownloadPath=$Local}
    $runner=if($Targets.Count -eq 1 -and $Targets[0] -eq '192.168.42.51'){
        $DraftRunner
    }else{$ProductionRunner}
    $results=@(& $runner @args)
    $results | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $run "$tag.json")
    if($results.Count -ne $Targets.Count -or @($results | Where-Object ExitCode -ne 0).Count){
        throw "Fleet step $tag failed; receipts retained."
    }
    return $results
}
function Guard {
    param([string]$Name,[string]$Command,[string[]]$Targets)
    Remote $Name "test `"`$(cat '$root/lease.id')`" = '$LeaseId'; bash '$root/lease.sh' $Command" $Targets
}
function ApiCheck {
    $health=Invoke-RestMethod "$ApiBaseUrl/health" -TimeoutSec 3
    $models=Invoke-RestMethod "$ApiBaseUrl/v1/models" -TimeoutSec 5
    if($health.status -ne 'ok' -or $models.data[0].id -ne 'Qwen3.8-Flash-Next-Uncensored'){
        throw 'Production health/model identity mismatch.'
    }
    $body=@{model='Qwen3.8-Flash-Next-Uncensored';temperature=0;max_tokens=16
        messages=@(@{role='user';content='/no_think Output exactly OK.'})} | ConvertTo-Json -Depth 6
    $chat=Invoke-RestMethod "$ApiBaseUrl/v1/chat/completions" -Method Post `
        -ContentType 'application/json' -Body $body -TimeoutSec 60
    $text=$chat.choices[0].message.content
    if([string]::IsNullOrWhiteSpace($text)){throw 'Production greedy chat was empty.'}
    return @{health=$health;models=$models;greedy=$chat;text=$text}
}
function RestoreProduction {
    $script:restoreMode=$true
    $failures=@()
    try {$null=Guard 'stop-candidates' 'stop-candidate' $all}catch{$failures+=$_.Exception.Message}
    try {$null=Guard 'restore-workers' 'restore' $workers}catch{$failures+=$_.Exception.Message}
    try {$null=Guard 'restore-coordinator' 'restore' @('192.168.42.42')}catch{$failures+=$_.Exception.Message}
    $ready=$false;$restored=$null
    $end=[DateTimeOffset]::UtcNow.AddMinutes(6)
    do{
        try {$restored=ApiCheck;$ready=$true}catch{Start-Sleep -Seconds 5}
    }while(-not $ready -and [DateTimeOffset]::UtcNow -lt $end)
    if(-not $ready){throw "Production restoration API check failed; watchdog remains armed. $($failures -join '; ') "}
    $null=Guard 'verify-production' 'verify' $all
    $restored | ConvertTo-Json -Depth 30 | Set-Content -Encoding utf8 (Join-Path $run 'restored-api.json')
    $null=Guard 'cancel-watchdogs' 'cancel' $all
    return $restored
}
if($Action -eq 'Restore'){
    $null=RestoreProduction
    Write-Output 'Pinned a63a166 + mix-04 restored and API/greedy checked.'
    return
}
$armed=$false;$mutated=$false;$complete=$false;$failure=$null;$before=$null;$created=$false
try {
    $null=Remote 'preflight-draft-release' "cd '$root'; sha256sum -c native-q8.gguf.sha256; cd source; sha256sum -c release.sha256" @('192.168.42.51')
    $created=$true
    $null=Remote 'create-owned-scratch' "test ! -e '$root'; mkdir -p '$root'; printf '%s\n' '$LeaseId' > '$root/lease.id'" $all
    $guardFile=Join-Path $run 'lease.sh'
    [IO.File]::WriteAllText($guardFile,
        ([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'mtp-lease-guard.sh')) -replace "`r",''))
    $null=Remote 'upload-guard' ':' $all -Upload $guardFile -Destination "$root/lease.sh"
    $checks=@(Guard 'preflight-production' 'preflight' $all)
    foreach($check in $checks){
        $epochs=@($check.Stdout -split "`r?`n" | Where-Object {$_ -match '^[0-9]{10}$'})
        if($epochs.Count -ne 1 -or [Math]::Abs([long]$epochs[0]-[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) -gt 30){
            throw 'Owner clock differs from the restore-watchdog clock by over thirty seconds.'
        }
    }
    $before=ApiCheck
    $before | ConvertTo-Json -Depth 30 | Set-Content -Encoding utf8 (Join-Path $run 'before-api.json')
    $null=Remote 'stage-release' ':' $all -Upload $release -Destination "$root/release.tar.gz"
    $null=Remote 'stage-corpus' ':' $all -Upload (Join-Path $Artifacts 'corpus.tar') -Destination "$root/corpus.tar"
    $null=Guard 'stage-immutable-pack-links' 'stage' $all
    $workerDeadline=$started.AddMinutes(38).ToUnixTimeSeconds()
    $coordinatorDeadline=$started.AddMinutes(39).ToUnixTimeSeconds()
    $armed=$true
    $null=Guard 'arm-worker-watchdogs' "arm $workerDeadline" $workers
    $null=Guard 'arm-coordinator-watchdog' "arm $coordinatorDeadline" @('192.168.42.42')
    $mutated=$true
    $null=Guard 'stop-production' 'stop-prod' $all
    $null=Guard 'start-candidate-workers' 'start' $workers
    $budget=[Math]::Max(60,[int]($script:workEnd-[DateTimeOffset]::UtcNow).TotalSeconds-30)
    $null=Guard 'start-candidate-coordinator' "start $budget" @('192.168.42.42')
    do{
        Start-Sleep -Seconds 10
        $r=@(Guard 'baseline-status' 'status' @('192.168.42.42'))
    }while($r[0].Stdout -notmatch 'baseline-ready')
    $index=Join-Path $run 'cases.tsv'
    $null=Remote 'read-capture-index' ':' @('192.168.42.42') -Download "$root/capture/cases.tsv" -Local $index
    $lines=@(Get-Content $index)
    if($lines.Count -ne 40){throw 'Baseline must include exactly forty eligible cases.'}
    $lastRanks=@($lines | ForEach-Object {($_ -split "`t")[5]} | Select-Object -Unique)
    if($lastRanks.Count -ne 1 -or $lastRanks[0] -notmatch '^[1-7]$'){throw 'Invalid final owner.'}
    $finalRank=[int]$lastRanks[0];$finalIp="192.168.42.$(42+$finalRank)"
    $null=Guard 'pack-baseline-coordinator' 'pack-baseline' @('192.168.42.42')
    $null=Guard 'pack-final-owner-HC' 'pack-baseline' @($finalIp)
    foreach($rank in @(0,$finalRank)){
        $file=Join-Path $run "baseline-$rank.tar"
        $null=Remote "download-baseline-$rank" ':' @("192.168.42.$(42+$rank)") `
            -Download "$root/baseline-$rank.tar" -Local $file -Timeout 240
        $null=Remote "upload-offline-$rank" ':' @('192.168.42.51') `
            -Upload $file -Destination "$root/baseline-$rank.tar" -Timeout 240
    }
    $null=Remote 'run-real-HC-native-drafts' "bash '$root/source/tools/mtp-capture-offline.sh'" `
        @('192.168.42.51') -Download "$root/proposals.tar" -Local (Join-Path $run 'proposals.tar') -Timeout 900
    $null=Remote 'upload-measured-proposals' ':' @('192.168.42.42') `
        -Upload (Join-Path $run 'proposals.tar') -Destination "$root/proposals.tar"
    $null=Guard 'start-paired-greedy-gates' 'proposals' @('192.168.42.42')
    do{
        Start-Sleep -Seconds 10
        $r=@(Guard 'greedy-status' 'status' @('192.168.42.42'))
    }while($r[0].Stdout -notmatch 'qualification-passed')
    $complete=$true
} catch {
    $failure=$_
} finally {
    if($mutated -or $armed){
        try{
            $restored=RestoreProduction
            if($before -and $restored.text -ne $before.text){throw 'Restored greedy sanity differs from pre-lease output.'}
            foreach($ip in $all){
                $null=Guard "evidence-$ip" 'evidence' @($ip)
                $null=Remote "download-evidence-$ip" ':' @($ip) -Download "$root/final-evidence.tar.gz" `
                    -Local (Join-Path $run "evidence-$ip.tar.gz")
            }
            $capture=Join-Path $run 'analysis'
            if($complete){
                $null=New-Item -ItemType Directory -Force $capture
                & tar -xf (Join-Path $run 'proposals.tar') -C $capture
                if($LASTEXITCODE -ne 0){throw 'Proposal archive extraction failed.'}
                $null=Remote 'download-greedy-receipts' ':' @('192.168.42.42') -Download "$root/capture/greedy.jsonl" `
                    -Local (Join-Path $capture 'greedy.jsonl')
                & python (Join-Path $PSScriptRoot 'mtp_capture.py') summarize $capture --require-greedy
                if($LASTEXITCODE -ne 0){throw 'Measured acceptance/greedy analysis failed.'}
            }
            $null=Guard 'remove-owned-ring-scratch' 'cleanup' $all
            $null=Remote 'remove-owned-draft-scratch' "bash '$root/source/tools/mtp-fast-cleanup.sh'" @('192.168.42.51')
        }catch{
            if(-not $failure){$failure=$_}
        }
    }elseif($created){
        try{
            $null=Remote 'remove-unstarted-scratch' "test `"`$(cat '$root/lease.id')`" = '$LeaseId'; test ! -e '$root/candidate.pid'; test ! -e '$root/watchdog.pid'; cd /home/xander; rm -rf '$root'" $all
        }catch{if(-not $failure){$failure=$_}}
    }
    [ordered]@{lease=$LeaseId;complete=$complete;failure=[string]$failure;
        elapsedSeconds=([DateTimeOffset]::UtcNow-$started).TotalSeconds;
        restoredApiReceipt=(Join-Path $run 'restored-api.json')} |
        ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $run 'outcome.json')
}
if($failure){throw $failure}
if(-not $complete){throw 'Qualification incomplete; do not claim acceptance/greedy success.'}
Write-Output "Capture and paired greedy gates completed; production restored. Evidence: $run"
