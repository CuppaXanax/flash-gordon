$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $root
$output=Join-Path $root 'results\hillclimb\multisess-fleet-20261016\capacity'
New-Item -ItemType Directory -Force $output | Out-Null
$all=@(42..49|ForEach-Object{"192.168.42.$_"})
$A=Get-Content "$ab\results\hillclimb\prod-gap\current\current.json" -Raw|ConvertFrom-Json -AsHashtable
$B=@{release='/home/xander/flash-gordon-live/20261016-abq-multisess'
    manifest=$A.manifest;manifest_sha256=$A.manifest_sha256;sessions=2;wide=0
    binary_sha256='de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58'}
$BaseUrl='http://192.168.42.42:8080/v1';$invariant=[Globalization.CultureInfo]::InvariantCulture
$RequestTimeoutSeconds=2700;$PromptSalt='multisess-20261016-dual-near-limit';$script:promptNonce=0
$records=@{metadata=@{model='Qwen3.8-Flash-Next-Uncensored'}}
$handler=[Net.Http.SocketsHttpHandler]::new();$handler.MaxConnectionsPerServer=8
$client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(2700)
$t=$null;$e=$null
$ast=[Management.Automation.Language.Parser]::ParseFile("$root\tools\hillclimb.ps1",[ref]$t,[ref]$e)
foreach($s in $ast.EndBlock.Statements){
    if($s -is [Management.Automation.Language.FunctionDefinitionAst]){Invoke-Expression $s.Extent.Text}
}
$result=[ordered]@{started=[DateTimeOffset]::Now.ToString('o');complete=$false
    phase1=$null;phase2=$null;dual_decode=$null;failure=$null;observation_failure=$null}
$monitorStarted=$false
function Save-Capacity{
    $result|ConvertTo-Json -Depth 50|Set-Content "$output\capacity.json"
}
try{
    # Previous paired arm is B. Snapshot verifies its identity without a new restart.
    & "$ab\tools\ab-deploy.ps1" -Deployment $B -SnapshotOnly
    # Match a cold mixed-agent input, but run it alone through its retained slot.
    $reference=Get-Content (Join-Path (Split-Path $output -Parent) 'pair-0-A\battery.json') -Raw |
        ConvertFrom-Json
    $referenceAgent=$reference.mixed.concurrency.trials[0].load.streams[1]
    $PromptSalt='multisess-20261016-pair-0/mixed';$script:promptNonce=2
    $purpose='Print a long numbered list of distinct practical software tasks, one per line. Continue until the token limit.'
    $isolatedAgent=New-Prompt 32768 $purpose -Agent
    Write-Host "SINGLE_ACTIVE_AGENT_PARITY start $([DateTimeOffset]::Now.ToString('o'))"
    $isolatedReply=Invoke-NonStream $isolatedAgent.Messages 256 -User 'hillclimb-agent'
    $isolatedHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
        [Text.Encoding]::UTF8.GetBytes($isolatedReply.Content))).ToLowerInvariant()
    $parity=$isolatedHash -eq $referenceAgent.sha256 -and
        $isolatedReply.CompletionTokens -eq $referenceAgent.completion_tokens -and
        $isolatedReply.PromptTokens -eq $referenceAgent.prompt_tokens -and
        (Get-HeaderNumber $isolatedReply.Headers 'X-Flash-Gordon-Reused-Tokens') -eq 0
    @{equal=$parity;reference_sha256=$referenceAgent.sha256;isolated_sha256=$isolatedHash
        reply=$isolatedReply;reference='pair-0-A/mixed/concurrency.trials[0].load.streams[1]'} |
        ConvertTo-Json -Depth 15|Set-Content "$output\single-active-agent-parity.json"
    $result['single_active_agent_parity']=$parity
    Write-Host "SINGLE_ACTIVE_AGENT_PARITY equal=$parity"
    $PromptSalt='multisess-20261016-dual-near-limit';$script:promptNonce=0
    $result.started=[DateTimeOffset]::Now.ToString('o')
    $receipts=@(& "$ab\tools\ab-fleet.ps1" -ScriptPath "$root\.build\capacity-start.sh" `
        -Targets $all -UploadPath "$ab\tools\ab-memory.py" `
        -UploadDestination /home/xander/flash-gordon-live/20261016-abq-multisess-memory.py `
        -PassThru -CommandTimeout 90)
    $monitorStarted=$true
    $receipts|ConvertTo-Json -Depth 8|Set-Content "$output\monitor-start.json"
    if($receipts.Count -ne 8 -or @($receipts|Where-Object ExitCode -ne 0).Count){throw 'Memory monitors failed'}
    Write-Host "CAPACITY agent-near262K/chat64 start $([DateTimeOffset]::Now.ToString('o'))"
    & pwsh -NoProfile -File "$root\tools\hillclimb.ps1" -ConcurrentGeneration `
        -MixedPrefill -CapacityProbe -ConcurrentRepeats 1 -ConcurrentTokens 64 `
        -AgentContextTokens 261800 -ChatContextTokens 64 -PromptSalt $PromptSalt `
        -RequestTimeoutSeconds 2700 -ResultsDirectory "$output\agent-near-limit" `
        *> "$output\agent-near-limit.log"
    if($LASTEXITCODE -notin @(0,2)){throw 'Near-limit agent generation failed'}
    $file=Get-ChildItem "$output\agent-near-limit" -Filter '*.json'|Select-Object -Last 1
    $phase1=Get-Content $file.FullName -Raw|ConvertFrom-Json
    if(-not $phase1.concurrency.complete){throw 'Agent phase incomplete'}
    $agentRow=$phase1.concurrency.trials[0].load.streams[1]
    if($agentRow.prompt_tokens -lt 261700){throw 'Agent did not reach near-limit context'}
    $result.phase1=@{prompt_tokens=$agentRow.prompt_tokens;completion_tokens=$agentRow.completion_tokens
        wall_seconds=$agentRow.wall_seconds;chat=$phase1.concurrency.trials[0].load.streams[0]
        correctness_pass=$phase1.metadata.correctness_pass}
    Save-Capacity
    $qualification=Get-Content (Join-Path (Split-Path $output -Parent) 'analysis.json') -Raw|ConvertFrom-Json
    $remaining=([DateTimeOffset]::Parse('2026-10-16T02:10:00-07:00')-[DateTimeOffset]::Now).TotalSeconds
    $extend=$qualification.protected_pass -and $qualification.aggregate_improves -and
        $qualification.solo_hash_identity -and $qualification.concurrent_hash_identity -and $parity -and
        $remaining -gt ($agentRow.wall_seconds*1.25+480)
    $chatTarget=if($extend){261800}else{32768}
    $result['chat_capacity_target']=$chatTarget
    $result['both_near_limit']=$extend
    Write-Host "CAPACITY agent complete; advancing chat target=$chatTarget $([DateTimeOffset]::Now.ToString('o'))"
    $purpose='Print a long numbered list of distinct practical software tasks, one per line. Continue until the token limit.'
    $script:promptNonce=2
    $agent=New-Prompt 261800 $purpose -Agent
    $script:promptNonce=4
    $chat=New-Prompt $chatTarget $purpose
    $chatReply=Invoke-NonStream $chat.Messages 64 -User 'hillclimb-chat'
    if($chatReply.PromptTokens -lt ($chatTarget-100) -or $chatReply.CompletionTokens -lt 32){
        throw 'Near-limit chat did not complete sustained generation'
    }
    $result.phase2=$chatReply
    Save-Capacity
    $agentFollow=@($agent.Messages)+@(@{role='assistant';content=$agentRow.visible_text},
        @{role='user';content="/no_think $purpose"})
    $chatFollow=@($chat.Messages)+@(@{role='assistant';content=$chatReply.Content},
        @{role='user';content="/no_think $purpose"})
    Write-Host "CAPACITY both resident; concurrent decode $([DateTimeOffset]::Now.ToString('o'))"
    $dual=Invoke-Concurrency $chatFollow $agentFollow 64
    foreach($row in $dual.streams){
        $minimum=if($row.name -eq 'agent-style'){261700}else{$chatTarget-100}
        if($row.completion_tokens -lt 32 -or $row.prompt_tokens -lt $minimum -or
            $row.prefix.reused_tokens -lt $minimum){
            throw "Near-limit resident continuation failed for $($row.name)"
        }
    }
    $result.dual_decode=$dual;$result.complete=$true
}catch{
    $result.failure=($_|Out-String)
    Write-Host "CAPACITY FAILURE $($result.failure)"
}finally{
    if($monitorStarted){
        try{
            $receipts=@(& "$ab\tools\ab-fleet.ps1" -ScriptPath "$root\.build\capacity-stop.sh" `
                -Targets $all -PassThru -CommandTimeout 60)
            $receipts|ConvertTo-Json -Depth 8|Set-Content "$output\monitor-stop.json"
            foreach($ip in $all){
                $receipt=@(& "$ab\tools\ab-fleet.ps1" -ScriptPath "$root\.build\multisess-noop.sh" `
                    -Targets @($ip) -DownloadSource /home/xander/flash-gordon-live/20261016-abq-multisess-capacity/memory.jsonl `
                    -DownloadPath "$output\$ip-memory.jsonl" -PassThru -CommandTimeout 60)
                if($receipt.Count -ne 1 -or $receipt[0].ExitCode -ne 0){throw "Download failed $ip"}
            }
            $receipt=@(& "$ab\tools\ab-fleet.ps1" -ScriptPath "$root\.build\multisess-noop.sh" `
                -Targets @('192.168.42.42') `
                -DownloadSource /home/xander/flash-gordon-live/20261015-abq-operator/rank-0.log `
                -DownloadPath "$output\coordinator.log" -PassThru -CommandTimeout 60)
            if($receipt.Count -ne 1 -or $receipt[0].ExitCode -ne 0){throw 'Coordinator log download failed'}
        }catch{
            $result.observation_failure=($_|Out-String);$result.complete=$false
        }
    }
    $result['finished']=[DateTimeOffset]::Now.ToString('o');Save-Capacity
    $client.Dispose();$handler.Dispose()
    if(-not $result.complete -or -not $result.single_active_agent_parity){
        & "$ab\tools\ab-deploy.ps1" -Deployment $A -Arm 'CAPACITY-FAILURE-RESTORE'
    }
}
if(-not $result.complete){exit 1}
if(-not $result.single_active_agent_parity){
    Write-Host 'SINGLE_ACTIVE_AGENT_PARITY_REJECTED; main/mix-04 restored'
    exit 1
}
Write-Host 'DUAL_NEAR_LIMIT_CAPACITY_COMPLETE'
