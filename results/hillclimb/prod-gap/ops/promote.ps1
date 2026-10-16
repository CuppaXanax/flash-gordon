$ErrorActionPreference='Stop'
$root='D:\looking-glass-labs\fg-wt\ab-harness'
$build=Join-Path $root '.build\gap'
$output=Join-Path $root 'results\hillclimb\prod-gap'
$null=New-Item -ItemType Directory -Force $output
$targets=@(42..49|ForEach-Object{"192.168.42.$_"})
$run='/home/xander/flash-gordon-live/20261015-abq-operator'
$text="set -euo pipefail`ncd '$run'`n"
foreach($item in @(
    @{source='launch-current.sh';name='launch-original.sh'},
    @{source='current.json';name='original.json'})){
    $data=[IO.File]::ReadAllBytes("$root\results\hillclimb\ab-current\$($item.source)")
    $b64=[Convert]::ToBase64String($data)
    $text+="printf '%s' '$b64' | base64 -d >'$($item.name)'`nchmod 700 '$($item.name)'`n"
}
$text+="bash -n launch-original.sh`nsha256sum launch-original.sh original.json`n"
[IO.File]::WriteAllText("$build\retain-original.sh",$text.Replace("`r",''))
$r=@(& "$root\tools\ab-fleet.ps1" -ScriptPath "$build\retain-original.sh" -Targets $targets -PassThru)
$r|ConvertTo-Json -Depth 8|Set-Content "$output\retain-original.json"
if($r.Count -ne 8 -or @($r|Where-Object ExitCode -ne 0).Count){throw 'Retaining original rollback failed.'}
$config=(Get-Content "$root\results\hillclimb\ab-ops\configs\mix-config.json" -Raw|ConvertFrom-Json -AsHashtable).B
& "$root\tools\ab-deploy.ps1" -Deployment $config -Arm 'user-authorized-mix04-promotion'
$config|ConvertTo-Json -Depth 8|Set-Content "$build\mix-current.json"
$installer=Get-Content "$root\results\hillclimb\ab-ops\install-current.ps1" -Raw
$installer=$installer.Replace("'results\hillclimb\ab-current'","'results\hillclimb\prod-gap\current'")
& ([scriptblock]::Create($installer)) -ConfigPath "$build\mix-current.json"
Write-Host 'Mix-04 promoted, pinned and restart-tested.'
