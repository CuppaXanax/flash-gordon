$root='D:\looking-glass-labs\fg-wt\ab-harness'
$build=Join-Path $root '.build\gap'
$output=Join-Path $root 'results\hillclimb\prod-gap'
$runner=Join-Path $root 'tools\ab-fleet.ps1'
$remote='/home/xander/hc-abq-gap'
function Fleet([string]$Text,[string]$Tag,[int[]]$Ranks=@(0..7)){
    $path=Join-Path $build "$Tag.sh"
    [IO.File]::WriteAllText($path,$Text.Replace("`r",''))
    $targets=@($Ranks|ForEach-Object{"192.168.42.$(42+$_)"})
    $r=@(& $runner -ScriptPath $path -Targets $targets -PassThru -CommandTimeout 180)
    $r|ConvertTo-Json -Depth 12|Set-Content "$output\$Tag.json"
    if($r.Count -ne $targets.Count -or @($r|Where-Object ExitCode -ne 0).Count){throw "Fleet step $Tag failed."}
    return $r
}
