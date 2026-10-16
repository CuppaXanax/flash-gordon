param([switch]$CapacityOnly,[int]$StartPair=0)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
$output=Join-Path $root 'results\hillclimb\multisess-fleet-20261016'
Set-Location $root
$A=Get-Content "$ab\results\hillclimb\prod-gap\current\current.json" -Raw | ConvertFrom-Json -AsHashtable
$B=@{release='/home/xander/flash-gordon-live/20261016-abq-multisess'
    manifest=$A.manifest;manifest_sha256=$A.manifest_sha256;sessions=2;wide=0
    binary_sha256='de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58'}
if(-not (Test-Path "$output\configuration.json")){
@{A=$A;B=$B;source='2d0f8a7';main_ancestor='a63a166';pairs=5;
    order=@('AB','BA','AB','BA','AB');started=[DateTimeOffset]::Now.ToString('o')
    gate='Solo higher-is-better paired median ratio >=0.97 and CI not wholly below 0.97; chat TTFT reversed. Aggregate paired median improves. Capacity must complete without OOM/swap/reserve violation.'} |
    ConvertTo-Json -Depth 8 | Set-Content "$output\configuration.json"
}
$BaseUrl='http://192.168.42.42:8080/v1'
$invariant=[Globalization.CultureInfo]::InvariantCulture
$handler=[Net.Http.SocketsHttpHandler]::new();$handler.MaxConnectionsPerServer=8
$client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(900)
$records=@{metadata=@{model='Qwen3.8-Flash-Next-Uncensored'}}
$tokens=$null;$parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile("$root\tools\hillclimb.ps1",[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw ($parseErrors|Out-String)}
foreach($statement in $ast.EndBlock.Statements){
    if($statement -is [Management.Automation.Language.FunctionDefinitionAst]){
        Invoke-Expression $statement.Extent.Text
    }
}
. "$ab\tools\hillclimb-ab.ps1"
$PromptSalt=''
function Deploy($config,$name){
    Write-Host "DEPLOY $name $([DateTimeOffset]::Now.ToString('o'))"
    & "$ab\tools\ab-deploy.ps1" -Deployment $config -Arm $name
}
function Reconcile-Logs($run,$battery,[string]$directory,[string]$name){
    $noop=Join-Path $root '.build\multisess-noop.sh'
    [IO.File]::WriteAllText($noop,"true`n")
    $log=Join-Path $directory "$name-server.log"
    $receipt=@(& "$ab\tools\ab-fleet.ps1" -ScriptPath $noop -Targets @('192.168.42.42') `
        -DownloadSource '/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.log' `
        -DownloadPath $log -PassThru -CommandTimeout 60)
    if($receipt.Count -ne 1 -or $receipt[0].ExitCode -ne 0){throw 'Log usage download failed'}
    $usage=@{}
    foreach($line in Get-Content $log){
        if($line -match '^request (?<id>chatcmpl-[^:]+): .*reused (?<reused>\d+), .*prefill (?<prefilled>\d+)/(?<prompt>\d+) tokens .*generation (?<count>\d+) tokens'){
            if($usage.ContainsKey($Matches.id)){throw "Ambiguous request id $($Matches.id)"}
            $usage[$Matches.id]=@{completion_tokens=[int]$Matches.count
                prompt_tokens=[int]$Matches.prompt;reused_tokens=[int]$Matches.reused}
        }
    }
    foreach($trial in $run.concurrency.trials){
        foreach($row in @($trial.isolated)+@($trial.load.streams)+@($battery.chat)){
            if($null -eq $row){continue}
            if(-not $row.request_id -or -not $usage.ContainsKey($row.request_id)){
                if($null -ne $row.completion_tokens -and
                    $row.accounting_source -eq 'Exact coordinator completion log matched by SSE request ID'){
                    continue
                }
                throw "Missing exact log usage for $($row.request_id)"
            }
            $actual=$usage[$row.request_id]
            if($actual.completion_tokens -lt 32){throw "Unsustained generation $($row.request_id)"}
            if($null -ne $row.completion_tokens -and $row.completion_tokens -ne $actual.completion_tokens){
                throw "SSE/log usage mismatch $($row.request_id)"
            }
            $row.completion_tokens=$actual.completion_tokens;$row.prompt_tokens=$actual.prompt_tokens
            $row.prefix=@{reused_tokens=$actual.reused_tokens}
            $row.client_decode_tps=$actual.completion_tokens/($row.wall_seconds-$row.ttft_ms/1000.0)
            $row.client_end_to_end_tps=$actual.completion_tokens/$row.wall_seconds
            $row|Add-Member accounting_source 'Exact coordinator completion log matched by SSE request ID' -Force
        }
        $trial.load.aggregate_client_decode_tps=(
            $trial.load.streams.completion_tokens|Measure-Object -Sum).Sum/$trial.load.aggregate_window_seconds
        $trial.load.aggregate_wall_tps=(
            $trial.load.streams.completion_tokens|Measure-Object -Sum).Sum/$trial.load.batch_wall_seconds
        $trial.load.token_count_source='Exact coordinator completion logs, matched by each SSE request ID'
    }
    $run.concurrency|Add-Member server_log_reconciliation @{
        file=(Split-Path $log -Leaf);sha256=(Get-FileHash $log -Algorithm SHA256).Hash.ToLowerInvariant()
        note='Raw harness JSON remains incomplete; this reconciled battery uses actual counts of these exact generations.'}
    $run.concurrency.complete=$true
}
function Battery([int]$pair,[string]$arm){
    $directory=Join-Path $output "pair-$pair-$arm"
    New-Item -ItemType Directory -Force $directory | Out-Null
    $path=Join-Path $directory 'battery.json'
    $prior=$null
    if(Test-Path $path){
        $prior=Get-Content $path -Raw|ConvertFrom-Json -AsHashtable
        if($prior.complete){Write-Host "SKIP completed $pair $arm";return}
    }
    $config=if($arm -eq 'A'){$A}else{$B}
    Deploy $config "$pair-$arm"
    $salt="multisess-20261016-pair-$pair"
    $battery=[ordered]@{pair=$pair;arm=$arm;salt=$salt;started=[DateTimeOffset]::Now.ToString('o')
        complete=$false;solo=@();chat=$null;concurrency=$null;mixed=$null;metrics=[ordered]@{}}
    if($prior){
        $battery=$prior
        $battery['resumed']=[DateTimeOffset]::Now.ToString('o')
        Write-Host "RESUME keeping every valid prior metric for $pair $arm"
    }
    $battery|ConvertTo-Json -Depth 30|Set-Content $path
    $warm=New-ABPrompt 64 "$salt/warmup" 'Output exactly OK.'
    $null=Invoke-NonStream $warm 16 -User 'solo'
    foreach($context in @(4096,32768)){
        if(@($battery.solo|Where-Object target -eq $context).Count){continue}
        $messages=New-ABPrompt $context "$salt/solo/$context" 'Output exactly Ready.'
        $cold=Invoke-NonStream $messages 1 -User 'solo'
        if((Get-HeaderNumber $cold.Headers 'X-Flash-Gordon-Reused-Tokens') -ne 0){throw 'Cold sample reused tokens'}
        $follow=@($messages)+@(@{role='assistant';content=$cold.Content},
            @{role='user';content='/no_think Write a long numbered list of distinct practical software tasks. Continue until the token limit.'})
        $decode=Invoke-NonStream $follow 128 -User 'solo'
        if($decode.CompletionTokens -lt 32){throw "Insufficient solo decode $context"}
        if((Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Reused-Tokens') -lt ($cold.PromptTokens-8)){throw 'Lost solo prefix'}
        $decodeWall=$decode.WallSeconds-(Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Prefill-Seconds')
        Add-ABMetric $battery "solo-$context.prefill_engine_tps" (Get-HeaderNumber $cold.Headers 'X-Flash-Gordon-Prefill-TPS')
        Add-ABMetric $battery "solo-$context.prefill_wall_tps" ($cold.PromptTokens/$cold.WallSeconds)
        Add-ABMetric $battery "solo-$context.decode_engine_tps" (Get-HeaderNumber $decode.Headers 'X-Flash-Gordon-Decode-TPS')
        Add-ABMetric $battery "solo-$context.decode_client_tps" ($decode.CompletionTokens/$decodeWall)
        $battery.solo+=@{target=$context;cold=$cold;decode=$decode}
        $battery|ConvertTo-Json -Depth 30|Set-Content $path
        Write-Host "SOLO $pair $arm $context prefill=$($battery.metrics["solo-$context.prefill_engine_tps"].value) decode=$($battery.metrics["solo-$context.decode_engine_tps"].value)"
    }
    if($null -eq $battery.chat){
        $chat=New-ABPrompt 0 "$salt/chat" 'Write a long numbered list of practical software tasks. Continue until the token limit.'
        $battery.chat=Get-GenerationStats (Finish-Sse (Start-Sse $chat 128 -User 'solo')) 'solo-chat'
    }
    if($arm -ne 'A' -and $battery.chat.completion_tokens -lt 32){throw 'Insufficient chat decode'}
    Add-ABMetric $battery 'chat.ttft_ms' $battery.chat.ttft_ms $false
    foreach($mixed in @($false,$true)){
        $name=if($mixed){'mixed'}else{'concurrency'}
        $runDirectory=Join-Path $directory $name
        $arguments=@('-NoProfile','-File',"$root\tools\hillclimb.ps1",'-ConcurrentGeneration',
            '-ConcurrentRepeats','1','-ConcurrentTokens','256','-ChatContextTokens','64',
            '-AgentContextTokens',$(if($mixed){'32768'}else{'64'}),
            '-PromptSalt',"$salt/$name",'-ResultsDirectory',$runDirectory,
            '-RequestTimeoutSeconds','900')
        if($mixed){$arguments+=@('-MixedPrefill','-CapacityProbe')}
        if($arm -eq 'A'){$arguments+='-AllowSerializedBaseline'}
        if($null -ne $battery[$name] -and $battery[$name].concurrency.complete){
            $run=$battery[$name];$exit=0
        }else{
            & pwsh @arguments *> "$directory\$name.log"
            $exit=$LASTEXITCODE
            $file=Get-ChildItem $runDirectory -Filter '*.json'|Sort-Object LastWriteTime|Select-Object -Last 1
            if(-not $file){throw "No $name evidence"}
            $run=Get-Content $file.FullName -Raw|ConvertFrom-Json
            if($arm -eq 'A' -and $run.concurrency.requires_server_log_accounting){
                Reconcile-Logs $run $battery $directory $name
            }
        }
        $battery[$name]=$run
        $battery|ConvertTo-Json -Depth 50|Set-Content $path
        if($exit -notin @(0,2) -or -not $run.concurrency.complete){throw "$name run failed; inspect $directory\$name.log"}
        $trial=$run.concurrency.trials[0]
        foreach($index in @(0,1)){
            $stream=$trial.load.streams[$index]
            Add-ABMetric $battery "$name.$($stream.name).tps" $stream.client_decode_tps
            Add-ABMetric $battery "$name.$($stream.name).end_to_end_tps" $stream.client_end_to_end_tps
            Add-ABMetric $battery "$name.$($stream.name).ttft_ms" $stream.ttft_ms $false
        }
        Add-ABMetric $battery "$name.aggregate_tps" $trial.load.aggregate_client_decode_tps
        Add-ABMetric $battery "$name.aggregate_wall_tps" $trial.load.aggregate_wall_tps
        Add-ABMetric $battery "$name.chat.gap_p95_ms" $trial.load.streams[0].inter_event_p95_ms $false
        $battery|ConvertTo-Json -Depth 50|Set-Content $path
        Write-Host "MODE $pair $arm $name aggregate=$($trial.load.aggregate_client_decode_tps) chatTTFT=$($trial.load.streams[0].ttft_ms) gap95=$($trial.load.streams[0].inter_event_p95_ms)"
    }
    $battery.complete=$true;$battery.finished=[DateTimeOffset]::Now.ToString('o')
    $battery|ConvertTo-Json -Depth 50|Set-Content $path
}
try{
    if(-not $CapacityOnly){
        for($pair=$StartPair;$pair -lt 5;$pair++){
            $order=if($pair%2){@('B','A')}else{@('A','B')}
            foreach($arm in $order){Battery $pair $arm}
        }
    }
    Write-Host 'PAIRED_BATTERY_COMPLETE'
} catch{
    $_|Out-String|Set-Content "$output\failure.txt"
    Write-Host "FAILURE $_"
    Deploy $A 'FAILURE-RESTORE'
    throw
} finally{
    $client.Dispose();$handler.Dispose()
}
