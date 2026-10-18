$ErrorActionPreference='Stop'
$work='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $work
$out=Join-Path $work '.build\incident-validation'
$null=New-Item -ItemType Directory -Force $out
$operator='/home/xander/flash-gordon-live/20261015-abq-operator'
$candidate='/home/xander/flash-gordon-live/20261018-abq-session-recovery'
$baseline=Get-Content "$ab\.build\abq\live.json" -Raw|ConvertFrom-Json -AsHashtable
$fixed=$baseline.Clone()
$fixed.release=$candidate
$fixed.binary_sha256=(Get-FileHash "$work\flash-gordon").Hash.ToLowerInvariant()
$fixed.Remove('capture_program');$fixed.Remove('capture_sha256')
$all=@(42..49|ForEach-Object{"192.168.42.$_"})
$step=0
$started=[DateTimeOffset]::UtcNow
$deadline=$started.AddMinutes(43)
$hard=$started.AddMinutes(45)
$selection='A'
$complete=$false
$supervised=$false
$result=[ordered]@{started=$started.ToString('o');deadline=$deadline.ToString('o');
  hard_deadline=$hard.ToString('o');A=$baseline;B=$fixed;pairs=@();complete=$false;selected='A'}
function Enc([string]$text){[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($text.Replace("`r",'')))}
function Save { $result|ConvertTo-Json -Depth 55|Set-Content "$out\result.json" }
function Fleet([string]$text,[int[]]$ranks=@(0..7),[switch]$Root,[switch]$Upload){
  $script=Join-Path $out "step-$script:step.sh"
  [IO.File]::WriteAllText($script,$text.Replace("`r",''))
  $args=@{ScriptPath=$script;Targets=@($ranks|ForEach-Object{"192.168.42.$(42+$_)"});PassThru=$true;CommandTimeout=120}
  if($Root){$args.Sudo=$true}
  if($Upload){$args.UploadPath="$work\.build\incident-release.tar.gz";$args.UploadDestination="$operator/incident-release.tar.gz"}
  $records=@(& "$work\tools\fg-fleet.ps1" @args)
  $records|ConvertTo-Json -Depth 12|Set-Content "$out\step-$script:step.json"
  $script:step++
  if($records.Count-ne$ranks.Count-or@($records|Where-Object ExitCode -ne 0).Count){
    $records|ForEach-Object{Write-Host $_.IP $_.Stdout $_.Stderr}
    throw "Fleet step failed: $script"
  }
  return $records
}
function Ready {
  for($i=0;$i-lt100;$i++){
    try {if((Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2).status-eq'ok'){return}}catch{}
    Start-Sleep -Seconds 1
  }
  throw 'Ring readiness timed out'
}
function Deploy([string]$arm){
  if([DateTimeOffset]::UtcNow-ge$deadline.AddMinutes(-2)){throw 'Qualification deadline near; stop new experiments'}
  if($script:currentArm-eq$arm){return}
  Write-Host "INCIDENT_DEPLOY arm=$arm elapsed=$(([DateTimeOffset]::UtcNow-$started).TotalMinutes)"
  $null=Fleet "set -euo pipefail`ntouch '$operator/supervisor.pause'`nbash '$operator/launch-current.sh' stop"
  $name=if($arm-eq'B'){'incident'}else{'lease-baseline'}
  $json=if($arm-eq'B'){'incident.json'}else{'lease-baseline.json'}
  $null=Fleet "set -euo pipefail`ncp '$operator/launch-$name.sh' '$operator/launch-current.sh'`ncp '$operator/$json' '$operator/current.json'"
  $null=Fleet "bash '$operator/launch-current.sh' start" @(1..7)
  $null=Fleet "bash '$operator/launch-current.sh' start" @(0)
  Ready
  $script:currentArm=$arm
}
Save
try {
  $files=@('fg_ring_control.py','fg_ring_supervisor.py','fg_lease_deadline.py')
  $writes=@($files|ForEach-Object{"printf '%s' '$(Enc (Get-Content "$work\tools\$_" -Raw))' | base64 -d >'$operator/$_'"})-join"`n"
  $pre=Fleet @"
set -euo pipefail
op='$operator'
test -f "`$op/launch-current.sh"
test ! -e "`$op/lease-active.json"
printf '%s  %s\n' '$($baseline.binary_sha256)' '$($baseline.release)/flash-gordon' | sha256sum -c - >&2
cp "`$op/launch-current.sh" "`$op/launch-lease-baseline.sh"
cp "`$op/current.json" "`$op/lease-baseline.json"
touch "`$op/supervisor.pause"
$writes
ip=`$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.42\.4[2-9]$' | head -1)
rank=`$((`${ip##*.}-42))
printf '%s\n' "`$rank" >"`$op/supervisor-rank"
python3 "`$op/fg_ring_control.py" --operator "`$op" --action status >&2
python3 -c 'import json;from pathlib import Path;print(json.dumps({"host_key":Path("/etc/ssh/ssh_host_ed25519_key.pub").read_text().strip()}))'
"@
  $keys=@{}
  foreach($item in $pre){$keys[$item.IP]=($item.Stdout.Trim()|ConvertFrom-Json).host_key}
  $public=Fleet @"
set -euo pipefail
test ! -e '$operator/supervisor-id'
ssh-keygen -q -t ed25519 -N '' -f '$operator/supervisor-id'
cat '$operator/supervisor-id.pub'
"@ @(0)
  $pub=$public[0].Stdout.Trim()
  if($pub-notmatch'^ssh-ed25519 [A-Za-z0-9+/=]+ '){throw 'Invalid supervisor public key'}
  $known=(@(1..7|ForEach-Object{"192.168.42.$(42+$_) $($keys["192.168.42.$(42+$_)"])"})-join"`n")+"`n"
  $config=@{operator=$operator;nodes=@(0..7|ForEach-Object{@{rank=$_;host="192.168.42.$(42+$_)"}});user='xander';
    identity="$operator/supervisor-id";known_hosts="$operator/supervisor-known-hosts";health='http://127.0.0.1:8080/health';startup_timeout=120;interval=5}
  $auth="from=`"192.168.42.42`",restrict,command=`"python3 $operator/fg_ring_control.py --operator $operator`" $pub"
  $null=Fleet @"
set -euo pipefail
mkdir -p /home/xander/.ssh
chmod 700 /home/xander/.ssh
printf '%s' '$(Enc ($auth+"`n"))' | base64 -d >>/home/xander/.ssh/authorized_keys
chmod 600 /home/xander/.ssh/authorized_keys
"@ @(1..7)
  $null=Fleet @"
set -euo pipefail
printf '%s' '$(Enc $known)' | base64 -d >'$operator/supervisor-known-hosts'
printf '%s' '$(Enc ($config|ConvertTo-Json -Depth 10))' | base64 -d >'$operator/supervisor.json'
python3 '$operator/fg_ring_supervisor.py' --config '$operator/supervisor.json' --check
"@ @(0)
  $active=@{started=$started.ToString('o');deadline=$deadline.ToString('o')}|ConvertTo-Json
  $null=Fleet "printf '%s' '$(Enc $active)' | base64 -d >'$operator/lease-active.json'"
  $unit=Get-Content "$work\tools\flash-gordon-ring.service" -Raw
  $null=Fleet @"
set -euo pipefail
printf '%s' '$(Enc $unit)' | base64 -d >/etc/systemd/system/flash-gordon-ring.service
systemctl daemon-reload
systemctl enable --now flash-gordon-ring.service
remaining=`$(($($deadline.ToUnixTimeSeconds())-`$(date +%s)))
test "`$remaining" -gt 120
systemd-run --unit=flash-gordon-incident-deadline --on-active="`${remaining}s" --uid=xander --property=UMask=0077 --property=KillMode=process --property=LimitMEMLOCK=infinity /usr/bin/python3 '$operator/fg_lease_deadline.py' --config '$operator/supervisor.json'
systemctl is-enabled flash-gordon-ring.service
systemctl is-active flash-gordon-ring.service
"@ @(0) -Root
  $supervised=$true
  $source=(Get-Content "$ab\tools\ab-instance.sh" -Raw).Replace("`r",'')
  $flags=@($fixed.flags.Keys|Where-Object{$_-ne'FG_EXPERT_Q8_WIDE'}|ForEach-Object{"$_='$($fixed.flags[$_])'"})-join' '
  $source=$source.Replace('# AB_STACK_FLAGS',"flags+=($flags)")
  $check=@"
if [[ `$action == start ]]; then
printf '%s  %s\n' '$($fixed.binary_sha256)' '$candidate/flash-gordon' | sha256sum -c -
printf '%s  %s\n' '$($fixed.manifest_sha256)' '$($fixed.manifest)' | sha256sum -c -
hash=`$(cd '$candidate'; find vulkan -type f -name '*.spv' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d ' ' -f 1)
test "`$hash" = '$($fixed.shader_tree_sha256)'
fi
"@
  $source=$source.Replace('case "$action" in',"$check`ncase `"`$action`" in").Replace("`r",'')
  $sourceHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($source))).ToLowerInvariant()
  $launcher=@"
#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\n' '$sourceHash' '$operator/ab-instance-incident.sh' | sha256sum -c -
bash '$operator/ab-instance-incident.sh' "`${1:?action}" '$candidate' '$($fixed.manifest)' 2 1
"@
  $null=Fleet @"
set -euo pipefail
test ! -e '$candidate'
mkdir '$candidate'
tar -xzf '$operator/incident-release.tar.gz' -C '$candidate' --no-same-owner flash-gordon
cp -a '$($baseline.release)/vulkan' '$candidate/vulkan'
chmod 755 '$candidate/flash-gordon'
printf '%s' '$(Enc $source)' | base64 -d >'$operator/ab-instance-incident.sh'
printf '%s' '$(Enc $launcher)' | base64 -d >'$operator/launch-incident.sh'
printf '%s' '$(Enc ($fixed|ConvertTo-Json -Depth 15))' | base64 -d >'$operator/incident.json'
printf '%s  %s\n' '$($fixed.binary_sha256)' '$candidate/flash-gordon' | sha256sum -c -
"@ -Upload
  # Load the existing benchmark functions without running their CLI entry point.
  foreach($path in @("$ab\tools\hillclimb.ps1","$ab\tools\hillclimb-ab.ps1")){
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
    if($errors.Count){throw 'Benchmark parse error'}
    $functions=@($ast.EndBlock.Statements|Where-Object{$_-is[Management.Automation.Language.FunctionDefinitionAst]}|ForEach-Object{$_.Extent.Text})-join"`n"
    . ([scriptblock]::Create($functions))
  }
  $BaseUrl='http://192.168.42.42:8080/v1'
  $invariant=[Globalization.CultureInfo]::InvariantCulture
  $handler=[Net.Http.SocketsHttpHandler]::new();$handler.MaxConnectionsPerServer=8
  $client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(240)
  $records=@{metadata=@{model=Get-Model}}
  $referencePath="$out\absent-reference.json"
  $ABConfig='incident';$GreedyOutput='';$WriteReference=$false
  $script:prefillPrompts=@{};$script:promptNonce=0
  Deploy 'B'
  $long=New-ABPrompt 46080 'incident-long' 'Output exactly Ready.'
  $side=New-ABPrompt 1820 'incident-side' 'Print a long numbered list of practical software tasks. Continue until the token limit.'
  $pending=@()
  foreach($spec in @(@{messages=$long;limit=1},@{messages=$side;limit=128})){
    $body=New-BodyJson $spec.messages $spec.limit $false|ConvertFrom-Json -AsHashtable
    $body.Remove('user')
    $request=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post,"$BaseUrl/chat/completions")
    $request.Content=[Net.Http.StringContent]::new(($body|ConvertTo-Json -Depth 30 -Compress),[Text.Encoding]::UTF8,'application/json')
    $pending+=@{request=$request;task=$client.SendAsync($request)}
  }
  $dual=@()
  foreach($item in $pending){
    $response=$item.task.GetAwaiter().GetResult()
    $text=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if(-not $response.IsSuccessStatusCode){throw "Dual prefill failed: $text"}
    $dual+=@{headers=Get-HeaderMap $response;payload=$text|ConvertFrom-Json}
    $response.Dispose();$item.request.Dispose()
  }
  $result.dual_prefill=$dual
  $null=Invoke-NonStream @(@{role='system';content='Answer directly.'},@{role='user';content='/no_think What is going on in this thread owo'}) 8
  $follow=@($long)+@(@{role='assistant';content=$dual[0].payload.choices[0].message.content},@{role='user';content='/no_think Output exactly Reused.'})
  $reuse=Invoke-NonStream $follow 8
  $result.side_request_continuation=@{headers=$reuse.Headers;prompt_tokens=$dual[0].payload.usage.prompt_tokens;content=$reuse.Content}
  if((Get-HeaderNumber $reuse.Headers 'X-Flash-Gordon-Reused-Tokens')-lt($dual[0].payload.usage.prompt_tokens-8)){throw 'Side request evicted long agent conversation'}
  Save
  $before=Fleet "cat '$operator/rank-0.pid'" @(0)
  $null=Fleet "rm -f '$operator/supervisor.pause'" @(0)
  $faultBody=New-BodyJson (New-ABPrompt 64 'incident-fault' 'Print a long numbered list. Continue until the token limit.') 512 $false
  $faultRequest=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post,"$BaseUrl/chat/completions")
  $faultRequest.Content=[Net.Http.StringContent]::new($faultBody,[Text.Encoding]::UTF8,'application/json')
  $faultTask=$client.SendAsync($faultRequest)
  Start-Sleep -Milliseconds 700
  $null=Fleet "bash '$operator/launch-current.sh' stop" @(3)
  $recovered=$false
  for($i=0;$i-lt100;$i++){
    try {
      $health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2
      if($health.status-eq'ok'-and$health.recovery_epoch-ge1){$recovered=$true;break}
    }catch{}
    Start-Sleep -Seconds 1
  }
  if(-not $recovered){throw 'Live API recovery did not complete'}
  $after=Fleet "cat '$operator/rank-0.pid'" @(0)
  $faultResponse=$faultTask.GetAwaiter().GetResult()
  $result.recovery=@{before_pid=$before[0].Stdout.Trim();after_pid=$after[0].Stdout.Trim();
    failed_request_status=[int]$faultResponse.StatusCode;error=$faultResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();health=$health}
  if($before[0].Stdout.Trim()-ne$after[0].Stdout.Trim()-or$faultResponse.IsSuccessStatusCode){throw 'Fault test killed API or did not fail affected request'}
  $faultResponse.Dispose();$faultRequest.Dispose()
  $null=Invoke-NonStream (New-ABPrompt 64 'incident-recovered' 'Say recovered briefly.') 16
  Save
  for($round=0;$round-lt5;$round++){
    $pair=[ordered]@{round=$round;salt=[Guid]::NewGuid().ToString('N');A=$null;B=$null}
    $order=if($round%2){@('B','A')}else{@('A','B')}
    foreach($arm in $order){
      Deploy $arm
      $pair[$arm]=Invoke-ABBattery $pair.salt @(4096,32768) 64
      $pair[$arm]|ConvertTo-Json -Depth 45|Set-Content "$out\round-$round-$arm.json"
    }
    $pair.hash_match=Test-ABCanonicalHashes $pair.A.hashes $pair.B.hashes
    $result.pairs+=,$pair
    Save
  }
  $result.metrics=@(Get-ABSummary $result.pairs)
  $result.hash_identity=@($result.pairs|Where-Object{-not $_.hash_match}).Count-eq0
  $guards=@($result.metrics|Where-Object{$_.metric-like'decode-*'-or$_.metric-like'prefill-*'-or$_.metric-eq'chat.ttft_ms'-or$_.metric-eq'concurrent.normalized_aggregate_tps'})
  $result.performance_pass=$result.hash_identity-and@($guards|Where-Object{-not $_.passes_primary_guard}).Count-eq0
  $result.complete=$true
  Save
  # These real SSE concurrency runs measure visible TTFT and loaded gaps.
  foreach($arm in @('B')){
    Deploy $arm
    & "$work\tools\hillclimb.ps1" -ConcurrentGeneration -MixedPrefill -CapacityProbe -ConcurrentRepeats 1 -ConcurrentTokens 128 -AgentContextTokens 32768 -PromptSalt incident-mixed-paired -ResultsDirectory "$out\sse-$arm" -RequestTimeoutSeconds 240
    if(-not $?){throw "Loaded SSE probe failed: $arm"}
  }
  if(-not $result.performance_pass){throw 'Protected paired gate failed; restore baseline'}
  $result.awaiting_incident_tests=$true
  Save
  Write-Host 'PAIRED_GATE_COMPLETE: B active; run bounded incident/recovery probes before final selection'
  $complete=$true
} catch {
  $result.error=$_.Exception.Message
  Write-Host "INCIDENT_ERROR $($result.error)"
} finally {
  if(-not $complete){
    try {
      $null=Fleet "set -euo pipefail`ntouch '$operator/supervisor.pause'`npython3 '$operator/fg_ring_control.py' --operator '$operator' --action restore" @(0)
      $null=Fleet "python3 '$operator/fg_ring_control.py' --operator '$operator' --action restore" @(1..7)
      $null=Fleet "bash '$operator/launch-current.sh' start" @(1..7)
      $null=Fleet "bash '$operator/launch-current.sh' start" @(0)
      Ready
      $null=Fleet "rm -f '$operator/lease-active.json' '$operator/supervisor.pause'"
      if($supervised){$null=Fleet 'systemctl stop flash-gordon-incident-deadline.timer' @(0) -Root}
      $result.restored_baseline=$true
    } catch {$result.restore_error=$_.Exception.Message;Write-Host "RESTORE_ERROR $($result.restore_error)"}
  }
  $result.finished=[DateTimeOffset]::UtcNow.ToString('o')
  Save
}
