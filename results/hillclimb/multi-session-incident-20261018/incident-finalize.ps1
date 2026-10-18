$ErrorActionPreference='Stop'
Set-Location 'D:\looking-glass-labs\fg-wt\multi-session-int'
$src=(Get-Content .build\incident-qualify.ps1 -Raw).Replace("`r",'')
. ([scriptblock]::Create($src.Substring(0,$src.IndexOf("`nSave`ntry {"))))
$result=Get-Content "$out\result.json" -Raw|ConvertFrom-Json -AsHashtable
$started=[DateTimeOffset]::Parse($result.started)
$deadline=[DateTimeOffset]::Parse($result.deadline)
$hard=[DateTimeOffset]::Parse($result.hard_deadline)
$baseline=$result.A
$step=700
$r2=$result.B.Clone()
$r2.release+="-r2"
$r2.binary_sha256=(Get-FileHash "$work\flash-gordon").Hash.ToLowerInvariant()
$result.final_routing_source='182d47e'
$result.final_routing_deployment=$r2
$result.selected='A'
$result.selection_reason='The final replay/in-flight routing follow-up was not included in all five paired samples. No promotion without a complete gate on the final source.'
$pause="touch '$operator/supervisor.pause'"
try {
  if([DateTimeOffset]::UtcNow-lt$deadline.AddMinutes(-3)){
    $oldHash=$result.B.binary_sha256
    $oldRelease=$result.B.release
    $text=@"
set -euo pipefail
op='$operator'
test ! -e '$($r2.release)'
mkdir '$($r2.release)'
tar -xzf "`$op/incident-release-r2.tar.gz" -C '$($r2.release)' --no-same-owner
cp -a '$($baseline.release)/vulkan' '$($r2.release)/vulkan'
chmod 755 '$($r2.release)/flash-gordon'
sed 's|$oldRelease|$($r2.release)|g;s|$oldHash|$($r2.binary_sha256)|g' "`$op/ab-instance-incident.sh" >"`$op/ab-instance-incident-r2.sh"
hash=`$(sha256sum "`$op/ab-instance-incident-r2.sh" | cut -d ' ' -f 1)
printf '#!/usr/bin/env bash\nset -euo pipefail\nprintf \"%%s  %%s\\n\" \"%s\" \"%s\" | sha256sum -c -\nbash \"%s\" \"`${1:?action}\" \"%s\" \"%s\" 2 1\n' "`$hash" "`$op/ab-instance-incident-r2.sh" "`$op/ab-instance-incident-r2.sh" '$($r2.release)' '$($r2.manifest)' >"`$op/launch-incident-r2.sh"
printf '%s' '$(Enc ($r2|ConvertTo-Json -Depth 15))' | base64 -d >"`$op/incident-r2.json"
bash -n "`$op/launch-incident-r2.sh"
printf '%s  %s\n' '$($r2.binary_sha256)' '$($r2.release)/flash-gordon' | sha256sum -c -
"@
    $stage=Join-Path $out 'stage-r2.sh'
    [IO.File]::WriteAllText($stage,$text.Replace("`r",''))
    $records=@(& "$work\tools\fg-fleet.ps1" -ScriptPath $stage -Targets $all -PassThru -UploadPath "$work\.build\incident-release-r2.tar.gz" -UploadDestination "$operator/incident-release-r2.tar.gz")
    $records|ConvertTo-Json -Depth 12|Set-Content "$out\r2-stage.json"
    if(@($records|Where-Object ExitCode -ne 0).Count){throw 'Final routing stage failed'}
    $null=Fleet "$pause`nbash '$operator/launch-current.sh' stop"
    $null=Fleet "cp '$operator/launch-incident-r2.sh' '$operator/launch-current.sh'`ncp '$operator/incident-r2.json' '$operator/current.json'"
    $null=Fleet "bash '$operator/launch-current.sh' start" @(1..7)
    $null=Fleet "bash '$operator/launch-current.sh' start" @(0)
    Ready
    & "$work\tools\hillclimb.ps1" -ConcurrentGeneration -ChatContextTokens 64 -AgentContextTokens 4096 -ConcurrentTokens 64 -ConcurrentRepeats 1 -PromptSalt final-routing-replay -ResultsDirectory "$out\r2-warm" -RequestTimeoutSeconds 120
    if(-not $?){throw 'Final routing warm SSE probe failed'}
    $result.final_routing_warm_pass=$true
  } else {
    $result.final_routing_warm_skipped='Insufficient time before original rollback deadline.'
  }
} catch {
  $result.final_routing_probe_error=$_.Exception.Message
} finally {
  # The final source has not completed the paired gate: always restore A.
  $null=Fleet "$pause`nbash '$operator/launch-current.sh' stop" @(0)
  $null=Fleet "bash '$operator/launch-current.sh' stop" @(1..7)
  $null=Fleet "cp '$operator/launch-lease-baseline.sh' '$operator/launch-current.sh'`ncp '$operator/lease-baseline.json' '$operator/current.json'"
  $null=Fleet 'for unit in flash-gordon-incident-deadline.timer flash-gordon-incident-deadline-v2.timer flash-gordon-incident-deadline.service flash-gordon-incident-deadline-v2.service; do systemctl stop "$unit" || true; done' @(0) -Root
  # Test the enabled watchdog's cold-boot path, not just manual rank launch.
  $null=Fleet "rm -f '$operator/supervisor.pause'" @(0)
  Ready
  $verify=Get-Content "$ab\tools\stack-verify-selected.sh" -Raw
  $null=Fleet ("set -- '$(Enc ($baseline|ConvertTo-Json -Depth 15))'`n"+$verify)
  foreach($path in @("$ab\tools\hillclimb.ps1","$ab\tools\hillclimb-ab.ps1")){
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors)
    $functions=@($ast.EndBlock.Statements|Where-Object{$_-is[Management.Automation.Language.FunctionDefinitionAst]}|ForEach-Object{$_.Extent.Text})-join"`n"
    . ([scriptblock]::Create($functions))
  }
  $BaseUrl='http://192.168.42.42:8080/v1';$invariant=[Globalization.CultureInfo]::InvariantCulture
  $handler=[Net.Http.SocketsHttpHandler]::new()
  $client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(60)
  $records=@{metadata=@{model=Get-Model}};$ABConfig='incident';$GreedyOutput='';$WriteReference=$false
  $referencePath="$out\absent-reference.json"
  $greedy=@(Invoke-Correctness)
  $result.final_greedy=$greedy
  if(@($greedy|Where-Object{[string]::IsNullOrWhiteSpace($_.text)}).Count){throw 'Restored API has empty greedy text'}
  if(-not(Test-ABCanonicalHashes $result.pairs[0].A.hashes $greedy)){throw 'Restored stack greedy hashes changed'}
  $result.restored_baseline=$true
  $result.final_health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 3
  $result.final_models=Invoke-RestMethod 'http://192.168.42.42:8080/v1/models' -TimeoutSec 3
  $null=Fleet "rm -f '$operator/lease-active.json' '$operator/supervisor.pause' '$operator/incident-release.tar.gz' '$operator/incident-release-r2.tar.gz'"
  $receipt=Fleet @"
set -euo pipefail
systemctl is-enabled flash-gordon-ring.service
systemctl is-active flash-gordon-ring.service
for unit in flash-gordon-incident-deadline.timer flash-gordon-incident-deadline-v2.timer; do
  if systemctl is-active --quiet "`$unit"; then exit 1; fi
done
journalctl -u flash-gordon-ring.service --no-pager -n 12
"@ @(0) -Root
  $result.supervisor_receipt=$receipt[0].Stdout
  $result.final_verified=[DateTimeOffset]::UtcNow.ToString('o')
  $result.lease_elapsed_minutes=([DateTimeOffset]::UtcNow-$started).TotalMinutes
  $result.lease_released=$true
  Save
  Write-Host "RESTORED_STACK supervised=true restart_tested=true hashes=true elapsed=$($result.lease_elapsed_minutes) lease_released=true"
}
