$ErrorActionPreference='Stop'
$work='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $work
$out=Join-Path $work '.build\robust-override'
if(Test-Path "$out\result.json"){throw 'This lease driver must not start a second window'}
$null=New-Item -ItemType Directory -Force $out
$operator='/home/xander/flash-gordon-live/20261015-abq-operator'
$previous=Get-Content .build\incident-validation\result.json -Raw|ConvertFrom-Json -AsHashtable
$baseline=$previous.A
$fixed=$baseline.Clone()
$fixed.release='/home/xander/flash-gordon-live/20261018-abq-session-recovery-r2'
$fixed.binary_sha256='3405c8976eca580ca2e8690bb336b0ca13e69ab761dbd7bffee590bb280cad17'
$fixed.source='11c7e39'
$fixed.Remove('capture_program');$fixed.Remove('capture_sha256')
if((Get-FileHash flash-gordon).Hash.ToLowerInvariant()-ne$fixed.binary_sha256){throw 'Candidate binary changed'}
$step=0;$currentArm='';$promoted=$false;$armed=$false;$entered=$false
$guards=@('decode-short.engine_tps','decode-short.client_tps','decode-32768.engine_tps',
  'decode-32768.client_tps','chat.ttft_ms','cached-prefix.ttft_ms','cached-32768.ttft_ms')
$result=[ordered]@{policy='explicit-lease-authority-override-v1';source='11c7e39';A=$baseline;B=$fixed;
  primary_metrics=$guards;pairs=@();selected='A';promoted=$false;complete=$false}
function Enc([string]$s){[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($s.Replace("`r",'')))}
function Save {$result|ConvertTo-Json -Depth 60|Set-Content "$out\result.json"}
function Budget {
  if([DateTimeOffset]::UtcNow-ge$deadline.AddMinutes(-3)){throw 'Stop experiments with restoration reserve'}
}
function Fleet([string]$text,[int[]]$ranks=@(0..7),[switch]$Root){
  $path=Join-Path $out "step-$script:step.sh"
  [IO.File]::WriteAllText($path,$text.Replace("`r",''))
  $args=@{ScriptPath=$path;Targets=@($ranks|ForEach-Object{"192.168.42.$(42+$_)"});
    PassThru=$true;CommandTimeout=150}
  if($Root){$args.Sudo=$true}
  $records=@(& "$work\tools\fg-fleet.ps1" @args)
  $records|ConvertTo-Json -Depth 14|Set-Content "$out\step-$script:step.json"
  $script:step++
  if($records.Count-ne$ranks.Count-or@($records|Where-Object ExitCode -ne 0).Count){
    $records|ForEach-Object{Write-Host $_.IP $_.Stdout $_.Stderr}
    throw "Fleet step failed: $path"
  }
  return $records
}
function Ready([int]$epoch=-1){
  $until=[DateTimeOffset]::UtcNow.AddSeconds(150)
  while([DateTimeOffset]::UtcNow-lt$until){
    try {
      $h=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 2
      if($h.status-eq'ok'-and($epoch-lt0-or$h.recovery_epoch-ge$epoch)){return $h}
    }catch{}
    Start-Sleep -Seconds 1
  }
  throw 'Ring readiness timed out'
}
function Deploy([string]$arm,[switch]$Force,[switch]$Restoring){
  if(-not$Restoring){Budget}
  if(-not$Force-and$script:currentArm-eq$arm){return}
  Write-Host "ROBUST_DEPLOY arm=$arm elapsed=$(([DateTimeOffset]::UtcNow-$started).TotalMinutes)"
  $null=Fleet "touch '$operator/supervisor.pause'; systemctl stop flash-gordon-ring.service" @(0) -Root
  $null=Fleet "touch '$operator/supervisor.pause'; bash '$operator/launch-current.sh' stop"
  $name=if($arm-eq'B'){'incident-r2'}else{'lease-baseline'}
  $json=if($arm-eq'B'){'robust-candidate.json'}else{'lease-baseline.json'}
  $null=Fleet "set -euo pipefail`ncp '$operator/launch-$name.sh' '$operator/launch-current.sh'`ncp '$operator/$json' '$operator/current.json'"
  $null=Fleet "bash '$operator/launch-current.sh' start" @(1..7)
  $null=Fleet "bash '$operator/launch-current.sh' start" @(0)
  $null=Ready
  $script:currentArm=$arm
}
function Verify($recipe){
  $verify=Get-Content "$ab\tools\stack-verify-selected.sh" -Raw
  $null=Fleet ("set -- '$(Enc ($recipe|ConvertTo-Json -Depth 18))'`n"+$verify)
}
function LoadBench {
  foreach($path in @("$ab\tools\hillclimb.ps1","$ab\tools\hillclimb-ab.ps1")){
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
    if($errors.Count){throw 'Benchmark parse failure'}
    $functions=@($ast.EndBlock.Statements|Where-Object{$_-is[Management.Automation.Language.FunctionDefinitionAst]}|ForEach-Object{$_.Extent.Text})-join"`n"
    . ([scriptblock]::Create($functions))
  }
  $battery=($ast.EndBlock.Statements|Where-Object{
    $_-is[Management.Automation.Language.FunctionDefinitionAst]-and$_.Name-eq'Invoke-ABBattery'}).Extent.Text
  $anchor='        Write-Host "AB context='
  if(-not$battery.Contains($anchor)){throw '32K TTFT injection anchor changed'}
  $extra=@'
        if($context-eq32768){
            $sseFollow=@($follow)+@(
                @{role='assistant';content=$decode.Content},
                @{role='user';content='/no_think State the next action briefly.'})
            $ttft32=Invoke-Ttft 'cached32k' $sseFollow 16
            if(-not$ttft32.finish_received-or[string]::IsNullOrEmpty($ttft32.visible_text)){
                throw 'Missing 32K SSE completion'}
            $battery.cached32k=$ttft32
            Add-ABMetric $battery 'cached-32768.ttft_ms' $ttft32.ttft_ms $false
        }
'@
  . ([scriptblock]::Create($battery.Replace($anchor,$extra+"`n"+$anchor)))
  # Export the loaded functions from this helper's local scope.
  Get-ChildItem function:|Where-Object{
    $_.Name-in @($ast.EndBlock.Statements|Where-Object{
      $_-is[Management.Automation.Language.FunctionDefinitionAst]}|ForEach-Object Name)
  }|ForEach-Object{Set-Item -Path "function:script:$($_.Name)" -Value $_.ScriptBlock}
}
# Loading in the driver scope also exports the first file's HTTP/SSE functions.
foreach($path in @("$ab\tools\hillclimb.ps1","$ab\tools\hillclimb-ab.ps1")){
  $tokens=$null;$errors=$null
  $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
  if($errors.Count){throw 'Benchmark parse failure'}
  . ([scriptblock]::Create((@($ast.EndBlock.Statements|Where-Object{
    $_-is[Management.Automation.Language.FunctionDefinitionAst]}|ForEach-Object{$_.Extent.Text})-join"`n")))
}
LoadBench
$BaseUrl='http://192.168.42.42:8080/v1'
$invariant=[Globalization.CultureInfo]::InvariantCulture
$handler=[Net.Http.SocketsHttpHandler]::new();$handler.MaxConnectionsPerServer=8
$client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(240)
$ABConfig='robustness';$GreedyOutput='';$WriteReference=$false
$referencePath="$out\absent-reference.json"
$script:prefillPrompts=@{};$script:promptNonce=0
function NoUser([object[]]$messages,[int]$limit){
  $body=New-BodyJson $messages $limit $false|ConvertFrom-Json -AsHashtable
  $body.Remove('user')
  $request=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post,"$BaseUrl/chat/completions")
  $request.Content=[Net.Http.StringContent]::new(($body|ConvertTo-Json -Depth 30 -Compress),[Text.Encoding]::UTF8,'application/json')
  try{
    $response=$client.SendAsync($request).GetAwaiter().GetResult()
    $text=$response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if(-not$response.IsSuccessStatusCode){throw "Replay HTTP $([int]$response.StatusCode): $text"}
    return @{payload=$text|ConvertFrom-Json;headers=Get-HeaderMap $response}
  }finally{if($response){$response.Dispose()};$request.Dispose()}
}
function Qos([string]$arm){
  Budget
  try{
    & "$work\tools\hillclimb.ps1" -ConcurrentGeneration -MixedPrefill -CapacityProbe -ConcurrentRepeats 1 -ConcurrentTokens 128 -AgentContextTokens 32768 -PromptSalt robustness-qos-comparison -ResultsDirectory "$out\qos-$arm" -RequestTimeoutSeconds 240
    $artifact=Get-ChildItem "$out\qos-$arm" -Filter '*.json'|Sort-Object LastWriteTime|Select-Object -Last 1
    if(-not$artifact){throw 'No mixed-prefill artifact'}
    $data=Get-Content $artifact.FullName -Raw|ConvertFrom-Json -AsHashtable
    $result["qos_$arm"]=@{path=$artifact.Name;concurrency=$data.concurrency;errors=$data.errors}
  }catch{
    $result["qos_$arm"]=@{error=$_.Exception.Message}
    Deploy $arm -Force
  }
  Save
}
$started=[DateTimeOffset]::UtcNow
$deadline=$started.AddMinutes(40)
$result.started=$started.ToString('o');$result.deadline=$deadline.ToString('o')
$result.hard_deadline=$started.AddMinutes(45).ToString('o')
$previousGate=Get-Content "$work\.build\robust-promotion\result.json" -Raw|ConvertFrom-Json -AsHashtable
$result.override='2026-10-18 lease authority explicitly accepts short-decode paired .965996 with CI crossing 1; promote unchanged candidate if functional replay, transport recovery and SIGKILL watchdog checks pass.'
$result.prior_qualification=@{commit='54ee55b';source='11c7e39';pairs=5;
  short_engine_benefit=($previousGate.metrics|Where-Object metric -eq 'decode-short.engine_tps').benefit_ratio;performance_pass=$false;
  reference_hashes=$previousGate.pairs[0].A.hashes}
Save
try{
  $entered=$true
  $null=Fleet "set -euo pipefail`ntest ! -e '$operator/lease-active.json'`ntouch '$operator/supervisor.pause'`nsystemctl stop flash-gordon-ring.service`ntest `"`$(systemctl show flash-gordon-ring.service -p LimitMEMLOCK --value)`" = infinity" @(0) -Root
  Verify $baseline
  $active=@{started=$result.started;deadline=$result.deadline;task='robustness-promotion'}|ConvertTo-Json
  $writes=@('fg_ring_control.py','fg_ring_supervisor.py','fg_lease_deadline.py')|ForEach-Object{
    "printf '%s' '$(Enc (Get-Content "$work\tools\$_" -Raw))' | base64 -d >'$operator/$_'"
  }
  $null=Fleet @"
set -euo pipefail
test ! -e '$operator/lease-active.json'
cp '$operator/launch-current.sh' '$operator/launch-lease-baseline.sh'
cp '$operator/current.json' '$operator/lease-baseline.json'
printf '%s' '$(Enc $active)' | base64 -d >'$operator/lease-active.json'
printf '%s' '$(Enc ($fixed|ConvertTo-Json -Depth 18))' | base64 -d >'$operator/robust-candidate.json'
$($writes-join"`n")
printf '%s  %s\n' '$($fixed.binary_sha256)' '$($fixed.release)/flash-gordon' | sha256sum -c -
bash -n '$operator/launch-incident-r2.sh'
"@
  $timerCommand="/usr/bin/setpriv --reuid=xander --regid=xander --init-groups /usr/bin/env HOME=/home/xander USER=xander /usr/bin/python3 '$operator/fg_lease_deadline.py' --config '$operator/supervisor.json'; status=`$?; /usr/bin/systemctl start flash-gordon-ring.service; exit `$status"
  $null=Fleet @"
set -euo pipefail
command -v setpriv
remaining=`$(($($deadline.ToUnixTimeSeconds())-`$(date +%s)))
test "`$remaining" -gt 300
systemd-run --unit=flash-gordon-override-deadline --on-active="`${remaining}s" --property=UMask=0077 --property=KillMode=process --property=LimitMEMLOCK=infinity /usr/bin/bash -c "`$(printf '%s' '$(Enc $timerCommand)' | base64 -d)"
systemctl is-active flash-gordon-override-deadline.timer
"@ @(0) -Root
  $armed=$true
  $records=@{metadata=@{model=Get-Model}}
  Deploy 'B'
  Budget
  $base=New-ABPrompt 44000 'robust-agent-44-messages' 'Output exactly Ready.'
  $content=[string]$base[1].content;$width=[int][Math]::Ceiling($content.Length/21)
  $agent=@($base[0])
  for($i=0;$i-lt21;$i++){
    $offset=$i*$width;$length=[Math]::Min($width,$content.Length-$offset)
    $agent+=@{role='user';content=$content.Substring($offset,$length)}
    $agent+=@{role='assistant';content="Acknowledged section $i."}
  }
  $agent+=@{role='user';content='/no_think Output exactly Ready.'}
  if($agent.Count-ne44){throw 'Replay fixture must contain 44 messages'}
  $prime=NoUser $agent 1
  $title=NoUser @(@{role='system';content='Give this thread a short title.'},
    @{role='user';content='What is going on in this thread owo'}) 8
  $follow=@($agent)+@(@{role='assistant';content=$prime.payload.choices[0].message.content},
    @{role='user';content='/no_think Output exactly Reused.'})
  $replay=NoUser $follow 8
  $reused=Get-HeaderNumber $replay.headers 'X-Flash-Gordon-Reused-Tokens'
  $result.replay=@{messages=$agent.Count;prompt_tokens=$prime.payload.usage.prompt_tokens;
    prime=$prime;title=$title;continuation=$replay;reused_tokens=$reused;
    pass=$prime.payload.usage.prompt_tokens-ge40000-and$reused-ge($prime.payload.usage.prompt_tokens-8)}
  Save
  if(-not$result.replay.pass){throw '44K agent prefix lost after two-message title'}
  $null=Fleet "rm -f '$operator/supervisor.pause'"
  $null=Fleet 'systemctl start flash-gordon-ring.service; systemctl is-active flash-gordon-ring.service' @(0) -Root
  $oldHealth=Ready
  $before=Fleet "cat '$operator/rank-0.pid'" @(0)
  $faultRequest=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post,"$BaseUrl/chat/completions")
  $faultRequest.Content=[Net.Http.StringContent]::new(
    (New-BodyJson (New-ABPrompt 64 'robust-fault' 'Print a long numbered list. Continue until the token limit.') 512 $false),
    [Text.Encoding]::UTF8,'application/json')
  $faultTask=$client.SendAsync($faultRequest)
  Start-Sleep -Milliseconds 700
  $null=Fleet "bash '$operator/launch-current.sh' stop" @(3)
  $health=Ready ($oldHealth.recovery_epoch+1)
  $after=Fleet "cat '$operator/rank-0.pid'" @(0)
  $faultResponse=$faultTask.GetAwaiter().GetResult()
  $next=Invoke-NonStream (New-ABPrompt 64 'robust-recovered' 'Say recovered briefly.') 16
  $result.transport=@{before_pid=$before[0].Stdout.Trim();after_pid=$after[0].Stdout.Trim();
    request_status=[int]$faultResponse.StatusCode;error=$faultResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult();
    health=$health;next_text=$next.Content;pass=($before[0].Stdout.Trim()-eq$after[0].Stdout.Trim()-and
      -not$faultResponse.IsSuccessStatusCode-and-not[string]::IsNullOrWhiteSpace($next.Content))}
  $faultResponse.Dispose();$faultRequest.Dispose();Save
  if(-not$result.transport.pass){throw 'Transport request failure/recovery gate failed'}
  Budget
  $kill=Fleet @"
set -euo pipefail
python3 - <<'PY'
import os,signal
from pathlib import Path
op=Path('$operator')
pid=int((op/'rank-0.pid').read_text())
assert os.readlink(f'/proc/{pid}/exe')=='$($fixed.release)/flash-gordon'
assert b'$($fixed.manifest)' in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
os.kill(pid,signal.SIGKILL)
print(pid)
PY
"@ @(0)
  $health=Ready
  $after=Fleet "cat '$operator/rank-0.pid'" @(0)
  $next=Invoke-NonStream (New-ABPrompt 64 'robust-watchdog-restarted' 'Say restarted briefly.') 16
  $result.watchdog=@{killed_pid=$kill[0].Stdout.Trim();after_pid=$after[0].Stdout.Trim();
    health=$health;next_text=$next.Content;pass=($kill[0].Stdout.Trim()-ne$after[0].Stdout.Trim()-and
      -not[string]::IsNullOrWhiteSpace($next.Content))}
  Save
  if(-not$result.watchdog.pass){throw 'Coordinator SIGKILL watchdog gate failed'}
  Verify $fixed
  $greedy=@(Invoke-Correctness)
  if(@($greedy|Where-Object{[string]::IsNullOrWhiteSpace($_.text)}).Count-or
    -not(Test-ABCanonicalHashes $previousGate.pairs[0].A.hashes $greedy)){throw 'Final candidate API hashes failed'}
  $result.final_greedy=$greedy
  $result.selected='B';$result.promoted=$true;$promoted=$true
}catch{
  $result.error=$_.Exception.Message
  Write-Host "ROBUST_ERROR $($result.error)"
}finally{
  if($entered){
    try{
      if(-not$promoted){
        Deploy 'A' -Force -Restoring
        Verify $baseline
        $restored=@(Invoke-Correctness)
        if(@($restored|Where-Object{[string]::IsNullOrWhiteSpace($_.text)}).Count-or
          -not(Test-ABCanonicalHashes $previousGate.pairs[0].A.hashes $restored)){
          throw 'Restored baseline API hashes failed'}
        $result.final_greedy=$restored
      }
      $null=Fleet "rm -f '$operator/lease-active.json' '$operator/supervisor.pause'"
      $receipt=Fleet @"
set -euo pipefail
systemctl stop flash-gordon-override-deadline.timer flash-gordon-override-deadline.service || true
systemctl enable --now flash-gordon-ring.service
systemctl is-enabled flash-gordon-ring.service
systemctl is-active flash-gordon-ring.service
systemctl show flash-gordon-ring.service -p LimitMEMLOCK --value
if systemctl is-active --quiet flash-gordon-override-deadline.timer; then exit 1; fi
journalctl -u flash-gordon-ring.service --no-pager -n 15
"@ @(0) -Root
      $result.supervisor_receipt=$receipt[0].Stdout
      $result.final_health=Ready
      $result.final_models=Invoke-RestMethod "$BaseUrl/models" -TimeoutSec 3
      $result.final_verified=[DateTimeOffset]::UtcNow.ToString('o')
      $result.lease_elapsed_minutes=([DateTimeOffset]::UtcNow-$started).TotalMinutes
      $result.lease_released=$true;$result.complete=$true
      Write-Host "ROBUST_COMPLETE selected=$($result.selected) promoted=$promoted elapsed=$($result.lease_elapsed_minutes) lease_released=true"
    }catch{
      $result.finalize_error=$_.Exception.Message
      Write-Host "ROBUST_FINALIZE_ERROR $($result.finalize_error)"
    }
  }
  Save
}
