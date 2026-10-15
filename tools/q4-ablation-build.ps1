#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][string]$RankPreflight,
    [ValidateSet('all','gdn-qkv','gdn-out','gdn-other','qsa','shared-gate','shared-up','shared-down','hc','other')]
    [string[]]$Families=@('all','gdn-qkv','gdn-out','gdn-other','qsa','shared-gate','shared-up','shared-down','hc','other')
)
$ErrorActionPreference='Stop'
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir|Out-Null
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$command=Join-Path $OutputDir 'build-command.sh'
$root='/home/xander/hc-q4/ablation-families'
$code='/home/xander/hc-q4/stage-ablation-aligned/code'
$source='/home/xander/hc-q4/ablation-source-check/dense-aligned-q4.gguf'
$targets=@(42..49|ForEach-Object{"192.168.42.$_"})
$rankMap=@{}
foreach($plan in (Get-Content $RankPreflight -Raw|ConvertFrom-Json)){
    $rows=[regex]::Matches($plan.Stdout,'(?m)^PLAN_RANK rank=(\d+) layers=(\d+) .*?endpoint=([^\s]+)')
    $owned=@($rows|Where-Object{$_.Groups[3].Value.Split(':')[0] -eq $plan.IP})
    if($owned.Count -ne 1 -or $owned[0].Groups[2].Value -ne '6'){throw 'Invalid manifest-derived owner map'}
    $rankMap[$plan.IP]=[int]$owned[0].Groups[1].Value
}
if($rankMap.Count -ne 8 -or @($rankMap.Values|Sort-Object -Unique).Count -ne 8){throw 'Incomplete rank map'}
function Fleet([string]$Text,[string[]]$Hosts,[hashtable]$Transfer=@{}){
    [IO.File]::WriteAllText($command,$Text.Replace("`r",''))
    $r=@(& $fleet -ScriptPath $command -Targets $Hosts -PassThru @Transfer)
    if(@($r|Where-Object ExitCode -ne 0).Count){throw ($r|ConvertTo-Json -Depth 5)}
    return $r
}
$paths=@()
try{
    $overlay=Join-Path $OutputDir 'dense-aligned-q4.gguf'
    $r=Fleet "set -e;sha256sum '$source'" @('192.168.42.53') @{
        DownloadSource=$source;DownloadPath=$overlay}
    $sha=(Get-FileHash $overlay -Algorithm SHA256).Hash.ToLower()
    if($r[0].Stdout.Split(' ')[0] -ne $sha){throw 'Q4 overlay transfer hash mismatch'}
    Fleet "set -eu;mkdir -p '$root';test -x '$code/tools/q4-repack'" $targets|Out-Null
    Fleet "set -eu;printf '%s  %s\n' '$sha' '$root/dense-aligned-q4.gguf'|sha256sum -c -" $targets @{
        UploadPath=$overlay;UploadDestination="$root/dense-aligned-q4.gguf"}|Out-Null
    foreach($ip in $targets){
        $text=@'
set -euo pipefail
root=/home/xander/hc-q4/ablation-families
code=/home/xander/hc-q4/stage-ablation-aligned/code
old=/home/xander/fg-ring-pack-abliterated
printf '%s  %s\n' 35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb "$old/manifest.fgm"|sha256sum -c -
cd "$code"
for family in __FAMILIES__;do
  stage="$root/pack-$family"
  pack="/home/xander/fg-ring-pack-abliterated-q4-ablate-$family"
  test ! -e "$stage"
  test ! -e "$pack"
  flags=()
  if [[ $family != all ]];then flags+=(--family "$family");fi
  ./tools/q4-repack "$old/manifest.fgm" "$old" "$root/dense-aligned-q4.gguf" "$stage" __RANK__ --keep-q8-head --keep-original-routers --cook-dense --reflink "${flags[@]}" > "$root/repack-$family.log" 2>&1 || { tail -30 "$root/repack-$family.log";exit 1; }
  ./tools/q4-repack "$stage/manifest.fgm" "$stage" "$root/dense-aligned-q4.gguf" "$root/unused" __RANK__ --keep-q8-head --keep-original-routers --cook-dense --plan "${flags[@]}" > "$root/resident-$family.txt"
  mv "$stage" "$pack"
  printf 'FAMILY name=%s sha=%s\n' "$family" "$(sha256sum "$pack/manifest.fgm"|cut -d' ' -f1)"
done
df -B1 /home/xander
'@
        $text=$text.Replace('__FAMILIES__',($Families -join ' ')).Replace('__RANK__',"$($rankMap[$ip])")
        $path=Join-Path $OutputDir "build-$ip.sh"
        [IO.File]::WriteAllText($path,$text.Replace("`r",''))
        $paths+=@{IP=$ip;Path=$path}
    }
    $records=@($paths|ForEach-Object -Parallel {
        & $using:fleet -Targets @($_.IP) -ScriptPath $_.Path -PassThru
    } -ThrottleLimit 8)
    $records|ConvertTo-Json -Depth 7|Set-Content (Join-Path $OutputDir 'family-packs.json')
    if($records.Count -ne 8 -or @($records|Where-Object ExitCode -ne 0).Count){throw 'Family repack failed'}
    foreach($family in $Families){
        $hashes=@($records|ForEach-Object{
            $m=[regex]::Match($_.Stdout,"FAMILY name=$family sha=([a-f0-9]{64})")
            if(-not $m.Success){throw "Missing seal for $family"}
            $m.Groups[1].Value
        }|Sort-Object -Unique)
        if($hashes.Count -ne 1){throw "Owner seals differ: $family"}
        Write-Host "$family $($hashes[0])"
    }
}finally{
    Remove-Item $command -ErrorAction SilentlyContinue
    foreach($item in $paths){Remove-Item $item.Path -ErrorAction SilentlyContinue}
}
