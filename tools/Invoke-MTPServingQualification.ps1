#requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('Plan','Run','Restore')][string]$Action='Plan',
    [switch]$LeaseConfirmed,
    [ValidatePattern('^[A-Za-z0-9_-]{4,80}$')][string]$LeaseId='',
    [long]$LeaseStartedUnixSeconds=0,
    [ValidateRange(1,35)][int]$WorkMinutes=35,
    [string]$Artifacts='',
    [string]$NativeOverlay='',
    [string]$ProductionRunner='D:\looking-glass-labs\fg-wt\ab-harness\tools\ab-fleet.ps1',
    [ValidatePattern('^http://(192\.168\.42\.42|127\.0\.0\.1)(:\d+)?$')]
    [string]$ApiBaseUrl='http://192.168.42.42:8080'
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(-not $Artifacts){$Artifacts=Join-Path $repo '.build\mtp-serving-lease'}
if(-not $NativeOverlay){$NativeOverlay=Join-Path $repo '.build\mtp-lease\native-q8.gguf'}
$phases=@(@('A0',0,0),@('B0',0,330),
    @('K1',1,0),@('S1',1,330),@('A1',0,0),
    @('K2',2,0),@('S2',2,330),@('A2',0,0),
    @('K3',3,0),@('S3',3,330),@('A3',0,0))
if($Action -eq 'Plan'){
    [ordered]@{
        productionAccess=$false; ring='192.168.42.42-.49'; phases=$phases; smallKernelMasks=@(0,330)
        greedy='eight prompts, same binary/pack/kernels, exact token-ID SHA256/count and completion identity'
        perf='three repeats of two chat/two code prompts, 128 output tokens, bracketed paired ratios'
        coexist='three two-session chat/code overlaps per phase; SSE/stop/cancel on K3/S3'
        workDeadlineMinutes=$WorkMinutes; watchdogWorkerMinute=38; watchdogCoordinatorMinute=39
        wallBudgetMinutes=45
        restore='31508ee / 20261016-abq-stack-adaptive + sealed mix-04, sessions2 wide1; exact pinned launcher/env'
        exclusions='no clocks/governors/weights/launch-current changes; both masks require greedy identity; no promotion or long-context claim'
    } | ConvertTo-Json -Depth 8
    return
}
if(-not $LeaseConfirmed -or -not $LeaseId){
    throw 'Explicit exclusive ring lease confirmation and a fresh LeaseId are required.'
}
$release=Join-Path $Artifacts 'release.tar.gz'
if($Action -eq 'Run'){
    $prepared=Get-Content (Join-Path $Artifacts 'prepared.json') -Raw | ConvertFrom-Json
    if((Get-FileHash $release -Algorithm SHA256).Hash.ToLowerInvariant() -ne $prepared.releaseSha256){
        throw 'Candidate release differs from the prevalidated artifact.'
    }
    if((Get-FileHash $NativeOverlay -Algorithm SHA256).Hash.ToLowerInvariant() -ne
       '536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b'){
        throw 'Native Q8 overlay identity differs from the qualified artifact.'
    }
    $auditPath=Join-Path $Artifacts 'format-audit.json'
    if(-not $prepared.formatAuditSha256 -or -not (Test-Path $auditPath) -or
       (Get-FileHash $auditPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $prepared.formatAuditSha256){
        throw 'A sealed exhaustive real-pack format audit is required before accessing production.'
    }
    $formatAudit=Get-Content $auditPath -Raw | ConvertFrom-Json
    if(-not $formatAudit.supported -or -not $formatAudit.real_pack_gate_passed -or
       $formatAudit.candidate_release_sha256 -ne $prepared.releaseSha256 -or
       $formatAudit.manifest_sha256 -ne '0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252' -or
       $formatAudit.native_overlay_sha256 -ne '536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b'){
        throw 'Format/real-Q8 gates do not qualify this candidate release and sealed production pack.'
    }
}
$all=@(42..49 | ForEach-Object {"192.168.42.$_"})
$workers=@(43..49 | ForEach-Object {"192.168.42.$_"})
$root='/home/xander/hc-mtp'
$run=Join-Path $Artifacts "runs\$LeaseId"
if($Action -eq 'Run' -and (Test-Path $run)){throw 'Lease evidence is immutable; use a fresh LeaseId.'}
$null=New-Item -ItemType Directory -Force $run
$started=if($LeaseStartedUnixSeconds){[DateTimeOffset]::FromUnixTimeSeconds($LeaseStartedUnixSeconds)}else{[DateTimeOffset]::UtcNow}
$script:workEnd=$started.AddMinutes($WorkMinutes)
$script:restoreMode=$Action -eq 'Restore'
$script:sequence=0
function Remote {
    param([string]$Name,[string]$Text,[string[]]$Targets,
          [string]$Upload='',[string]$Destination='',[string]$Download='',
          [string]$Local='',[int]$Timeout=180)
    $remaining=if($script:restoreMode){300}else{[int]($script:workEnd-[DateTimeOffset]::UtcNow).TotalSeconds}
    if($remaining -le 0){throw 'Serving qualification deadline reached; restore now.'}
    $script:sequence++
    $tag='{0:D3}-{1}' -f $script:sequence,$Name
    $file=Join-Path $run "$tag.sh"
    [IO.File]::WriteAllText($file,"set -euo pipefail`n"+($Text -replace "`r",'')+"`n")
    $arguments=@{ScriptPath=$file;Targets=$Targets;PassThru=$true;CommandTimeout=[Math]::Max(60,[Math]::Min($Timeout,$remaining))}
    if($Upload){$arguments.UploadPath=$Upload;$arguments.UploadDestination=$Destination}
    if($Download){$arguments.DownloadSource=$Download;$arguments.DownloadPath=$Local}
    $results=@(& $ProductionRunner @arguments)
    $results | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $run "$tag.json")
    if($results.Count -ne $Targets.Count -or @($results | Where-Object ExitCode -ne 0).Count){
        throw "Fleet step $tag failed."
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
        throw 'API health/model mismatch.'
    }
    return @{health=$health;models=$models}
}
function ApiGreedy {
    $body=@{model='Qwen3.8-Flash-Next-Uncensored';temperature=0;max_tokens=16
        user='lease-health';messages=@(@{role='user';content='/no_think Output exactly OK.'})} |
        ConvertTo-Json -Depth 6
    $result=Invoke-RestMethod "$ApiBaseUrl/v1/chat/completions" -Method Post `
        -ContentType application/json -Body $body -TimeoutSec 60
    if([string]::IsNullOrWhiteSpace($result.choices[0].message.content)){throw 'Empty greedy health response.'}
    return $result
}
function WaitApi {
    $end=[DateTimeOffset]::UtcNow.AddSeconds(150)
    if(-not $script:restoreMode -and $end -gt $script:workEnd){$end=$script:workEnd}
    do {
        try {return ApiCheck}catch{Start-Sleep -Seconds 2}
    } while([DateTimeOffset]::UtcNow -lt $end)
    throw 'API failed readiness deadline.'
}
function Client {
    param([string[]]$Arguments)
    if([DateTimeOffset]::UtcNow -ge $script:workEnd){throw 'Client work deadline reached.'}
    $seconds=[Math]::Min(300,[int]($script:workEnd-[DateTimeOffset]::UtcNow).TotalSeconds)
    $out=Join-Path $run "client-$([guid]::NewGuid().ToString('N')).stdout"
    $err=Join-Path $run "$([IO.Path]::GetFileNameWithoutExtension($out)).stderr"
    $start=[Diagnostics.ProcessStartInfo]::new('python')
    $start.UseShellExecute=$false
    $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    $start.ArgumentList.Add((Join-Path $PSScriptRoot 'mtp_serving_qualify.py'))
    foreach($argument in $Arguments){$start.ArgumentList.Add($argument)}
    $process=[Diagnostics.Process]::Start($start)
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
    if(-not $process.WaitForExit([Math]::Max(1,$seconds)*1000)){
        $process.Kill($true);$process.WaitForExit()
        [IO.File]::WriteAllText($out,$stdout.GetAwaiter().GetResult())
        [IO.File]::WriteAllText($err,$stderr.GetAwaiter().GetResult())
        throw 'HTTP client timed out; restore production.'
    }
    [IO.File]::WriteAllText($out,$stdout.GetAwaiter().GetResult())
    [IO.File]::WriteAllText($err,$stderr.GetAwaiter().GetResult())
    if($process.ExitCode){throw "HTTP client failed; see $err"}
}
function RestoreProduction {
    $script:restoreMode=$true
    $restoreErrors=@()
    try {$null=Guard 'stop-candidates' 'stop-candidate' $all}catch{$restoreErrors+=$_.Exception.Message}
    try {$null=Guard 'restore-workers' 'restore' $workers}catch{$restoreErrors+=$_.Exception.Message}
    try {$null=Guard 'restore-coordinator' 'restore' @('192.168.42.42')}catch{$restoreErrors+=$_.Exception.Message}
    $api=WaitApi
    $greedy=ApiGreedy
    $null=Guard 'verify-pinned-production' 'verify' $all
    @{api=$api;greedy=$greedy} | ConvertTo-Json -Depth 30 |
        Set-Content -Encoding utf8 (Join-Path $run 'restored-api.json')
    $null=Guard 'cancel-watchdogs' 'cancel' $all
    if($restoreErrors.Count){Write-Warning ($restoreErrors -join '; ')}
    return $greedy
}
if($Action -eq 'Restore'){
    $null=RestoreProduction
    Write-Output 'Current stack-1/mix-04 sessions2 wide1 production restored and verified.'
    return
}
$created=$false;$armed=$false;$mutated=$false;$complete=$false;$failure=$null;$before=$null
$productionVerified=$false;$scratchRemoved=$false;$finalizationErrors=@()
try {
    $created=$true
    $null=Remote 'create-owned-scratch' "test ! -e '$root'; mkdir -p '$root'; printf '%s\n' '$LeaseId' > '$root/lease.id'" $all
    $guardFile=Join-Path $run 'lease.sh'
    [IO.File]::WriteAllText($guardFile,([IO.File]::ReadAllText((Join-Path $PSScriptRoot 'mtp-lease-guard.sh')) -replace "`r",''))
    $null=Remote 'upload-guard' ':' $all -Upload $guardFile -Destination "$root/lease.sh"
    $checks=@(Guard 'preflight-production' 'preflight' $all)
    foreach($check in $checks){
        $epochs=@($check.Stdout -split "`r?`n" | Where-Object {$_ -match '^[0-9]{10}$'})
        if($epochs.Count -ne 1 -or [Math]::Abs([long]$epochs[0]-[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) -gt 30){
            throw 'Owner/watchdog clock skew exceeds thirty seconds.'
        }
    }
    $before=ApiGreedy
    $before | ConvertTo-Json -Depth 30 | Set-Content -Encoding utf8 (Join-Path $run 'before-api.json')
    $null=Remote 'stage-release' ':' $all -Upload $release -Destination "$root/release.tar.gz"
    $null=Guard 'stage-serving' 'stage-serving' $all
    $null=Remote 'stage-native-final-owner' "echo '536bab54b0a1bdea23b65871c4cdc38deee5803771501a0c770a0510bcd3283b  $root/native-q8.gguf' | sha256sum -c -" `
        @('192.168.42.49') -Upload $NativeOverlay -Destination "$root/native-q8.gguf" -Timeout 240
    $embedding=Join-Path $run 'embedding.fgme'
    $null=Remote 'fetch-exact-embedding' ':' @('192.168.42.42') -Download "$root/embedding.fgme" -Local $embedding
    $null=Remote 'stage-exact-embedding-final-owner' ':' @('192.168.42.49') -Upload $embedding -Destination "$root/embedding.fgme"
    $armed=$true
    $null=Guard 'arm-worker-watchdogs' "arm $($started.AddMinutes(38).ToUnixTimeSeconds())" $workers
    $null=Guard 'arm-coordinator-watchdog' "arm $($started.AddMinutes(39).ToUnixTimeSeconds())" @('192.168.42.42')
    foreach($phase in $phases){
        $name=[string]$phase[0];$cap=[int]$phase[1];$small=[int]$phase[2]
        if(-not $mutated){$mutated=$true;$null=Guard 'stop-production' 'stop-prod' $all}
        else {$null=Guard "stop-before-$name" 'stop-candidate' $all}
        $null=Guard "start-workers-$name" "start-serving $name $cap $small" $workers
        $null=Guard "start-coordinator-$name" "start-serving $name $cap $small" @('192.168.42.42')
        $null=WaitApi
        $greedyFile=Join-Path $run "$name-greedy.json"
        Client @('run','--phase',$name,'--suite','greedy','--base',$ApiBaseUrl,'--output',$greedyFile)
        $trace=Join-Path $run "$name-rank-0.log"
        $null=Remote "token-receipts-$name" ':' @('192.168.42.42') -Download "$root/serving-$name-rank-0.log" -Local $trace
        Client @('verify','--candidate',$greedyFile,'--baseline',(Join-Path $run 'A0-greedy.json'),
            '--trace',$trace,'--baseline-trace',(Join-Path $run 'A0-rank-0.log'))
        Client @('run','--phase',$name,'--suite','perf','--base',$ApiBaseUrl,
            '--output',(Join-Path $run "$name-perf.json"))
        if($name -in @('K3','S3')){
            Client @('run','--phase',$name,'--suite','stream','--base',$ApiBaseUrl,
                '--output',(Join-Path $run "$name-stream.json"))
        }
        $null=Remote "complete-round-traces-$name" ':' @('192.168.42.42') `
            -Download "$root/serving-$name-rank-0.log" -Local $trace
    }
    Client @('summarize','--directory',$run)
    $complete=$true
} catch {$failure=$_}
finally {
    if($mutated -or $armed){
        try {
            $restored=RestoreProduction
            $productionVerified=$true
            if($before.choices[0].message.content -ne $restored.choices[0].message.content){
                throw 'Restored production greedy sanity differs from pre-lease.'
            }
        } catch {
            $finalizationErrors+=$_.Exception.Message
            if(-not $failure){$failure=$_}
        }
        if($productionVerified){
            foreach($ip in $all){
                try {
                    $null=Guard "evidence-$ip" 'evidence' @($ip)
                    $null=Remote "download-evidence-$ip" ':' @($ip) -Download "$root/final-evidence.tar.gz" `
                        -Local (Join-Path $run "evidence-$ip.tar.gz")
                } catch {
                    $finalizationErrors+=$_.Exception.Message
                    if(-not $failure){$failure=$_}
                }
            }
            try {
                $null=Guard 'remove-owned-scratch' 'cleanup' $all
                $scratchRemoved=$true
            } catch {
                $finalizationErrors+=$_.Exception.Message
                if(-not $failure){$failure=$_}
            }
        }
    } elseif($created){
        try {
            $null=Remote 'cleanup-unstarted' "if test -f '$root/lease.id' && test `"`$(cat '$root/lease.id')`" = '$LeaseId'; then test ! -e '$root/candidate.pid'; test ! -e '$root/watchdog.pid'; cd /home/xander; rm -rf '$root'; fi" $all
            $scratchRemoved=$true
        } catch {
            $finalizationErrors+=$_.Exception.Message
            if(-not $failure){$failure=$_}
        }
    }
    @{lease=$LeaseId;complete=$complete;failure=[string]$failure;
        productionVerified=$productionVerified;scratchRemoved=$scratchRemoved;
        finalizationErrors=$finalizationErrors;
        elapsedSeconds=([DateTimeOffset]::UtcNow-$started).TotalSeconds} |
        ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $run 'outcome.json')
}
if($failure){throw $failure}
if(-not $complete){throw 'Serving qualification incomplete; no performance or exactness claim.'}
Write-Output "MTP serving gates complete; pinned production restored. Evidence: $run"
