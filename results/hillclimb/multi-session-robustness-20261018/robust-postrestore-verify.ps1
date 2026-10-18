$ErrorActionPreference='Stop'
$work='D:\looking-glass-labs\fg-wt\multi-session-int'
$ab='D:\looking-glass-labs\fg-wt\ab-harness'
Set-Location $work
$path="$work\.build\robust-promotion\result.json"
$r=Get-Content $path -Raw|ConvertFrom-Json -AsHashtable
if($r.promoted-or-not$r.complete){throw 'Only finalize the completed baseline restore'}
if([DateTimeOffset]::UtcNow-ge[DateTimeOffset]$r.hard_deadline){throw 'Original lease has expired'}
foreach($script in @("$ab\tools\hillclimb.ps1","$ab\tools\hillclimb-ab.ps1")){
  $tokens=$null;$errors=$null
  $ast=[Management.Automation.Language.Parser]::ParseFile($script,[ref]$tokens,[ref]$errors)
  if($errors.Count){throw 'Benchmark parse error'}
  . ([scriptblock]::Create((@($ast.EndBlock.Statements|Where-Object{
    $_-is[Management.Automation.Language.FunctionDefinitionAst]}|ForEach-Object{$_.Extent.Text})-join"`n")))
}
$BaseUrl='http://192.168.42.42:8080/v1'
$invariant=[Globalization.CultureInfo]::InvariantCulture
$handler=[Net.Http.SocketsHttpHandler]::new()
$client=[Net.Http.HttpClient]::new($handler);$client.Timeout=[TimeSpan]::FromSeconds(60)
$records=@{metadata=@{model=Get-Model}}
$ABConfig='robustness';$GreedyOutput='';$WriteReference=$false
$referencePath="$work\.build\robust-promotion\absent-reference.json"
$greedy=@(Invoke-Correctness)
if(@($greedy|Where-Object{[string]::IsNullOrWhiteSpace($_.text)}).Count-or
  -not(Test-ABCanonicalHashes $r.pairs[0].A.hashes $greedy)){throw 'Restored baseline greedy API failed'}
$r.final_greedy=$greedy
$r.cleanup_completed_at=$r.final_verified
$r.final_health=Invoke-RestMethod 'http://192.168.42.42:8080/health' -TimeoutSec 3
$r.final_models=Invoke-RestMethod "$BaseUrl/models" -TimeoutSec 3
$r.final_verified=[DateTimeOffset]::UtcNow.ToString('o')
$r.lease_elapsed_minutes=([DateTimeOffset]::UtcNow-[DateTimeOffset]$r.started).TotalMinutes
$r.release_finalization_note='Completed mandatory real-generation verification omitted by the fail-fast restore path, within the original 60-minute grant; no new experiments or lease clock.'
$r|ConvertTo-Json -Depth 60|Set-Content $path
$client.Dispose()
Write-Host "BASELINE_FINAL_VERIFIED hashes=5 health=$($r.final_health.status) elapsed=$($r.lease_elapsed_minutes) fleet_released=true"
