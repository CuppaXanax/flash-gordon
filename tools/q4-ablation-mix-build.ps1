#requires -Version 7.0
param(
    [Parameter(Mandatory)][ValidatePattern('^mix-[a-z0-9-]+$')][string]$Label,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9-]+$')][string]$OverlayName,
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][string]$RankPreflight
)
$ErrorActionPreference='Stop'
$OutputDir=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
New-Item -ItemType Directory -Force $OutputDir|Out-Null
$fleet=Join-Path $PSScriptRoot 'Invoke-Q4Fleet.ps1'
$command=Join-Path $OutputDir 'mix-command.sh'
$source="/home/xander/hc-q4/ablation-source-check/$OverlayName"
$root="/home/xander/hc-q4/ablation-$Label"
$pack="/home/xander/fg-ring-pack-abliterated-q4-ablate-$Label"
$targets=@(42..49|ForEach-Object{"192.168.42.$_"})
$rankMap=@{}
foreach($record in (Get-Content $RankPreflight -Raw|ConvertFrom-Json)){
    $rows=[regex]::Matches($record.Stdout,'(?m)^PLAN_RANK rank=(\d+) layers=(\d+) .*?endpoint=([^\s]+)')
    $owned=@($rows|Where-Object{$_.Groups[3].Value.Split(':')[0] -eq $record.IP})
    if($owned.Count -ne 1 -or $owned[0].Groups[2].Value -ne '6'){throw 'Invalid owner map'}
    $rankMap[$record.IP]=[int]$owned[0].Groups[1].Value
}
if($rankMap.Count -ne 8 -or @($rankMap.Values|Sort-Object -Unique).Count -ne 8){throw 'Incomplete owner map'}
function Fleet([string]$Text,[string[]]$Hosts,[hashtable]$Transfer=@{}){
    [IO.File]::WriteAllText($command,$Text.Replace("`r",''))
    $r=@(& $fleet -ScriptPath $command -Targets $Hosts -PassThru @Transfer)
    if(@($r|Where-Object ExitCode -ne 0).Count){throw ($r|ConvertTo-Json -Depth 4)}
    return $r
}
$paths=@()
try{
    $inventory=Join-Path $OutputDir "$OverlayName.json"
    Fleet "set -e;test -s '$source.json'" @('192.168.42.53') @{
        DownloadSource="$source.json";DownloadPath=$inventory}|Out-Null
    $info=Get-Content $inventory -Raw|ConvertFrom-Json
    $families=@($info.changed_families)
    $routerFlags=if($info.bf16_router_count){''}else{'--keep-original-routers'}
    $allowed=@('gdn-qkv','gdn-out','gdn-other','qsa','shared-gate','shared-up','shared-down','hc','other')
    if(-not $families.Count -or @($families|Where-Object{$_ -notin $allowed}).Count){throw 'Invalid changed families'}
    $overlay=Join-Path $OutputDir "$OverlayName.gguf"
    $r=Fleet "set -e;sha256sum '$source.gguf'" @('192.168.42.53') @{
        DownloadSource="$source.gguf";DownloadPath=$overlay}
    $sha=(Get-FileHash $overlay -Algorithm SHA256).Hash.ToLowerInvariant()
    if($sha -ne $info.sha256 -or $r[0].Stdout.Split(' ')[0] -ne $sha){throw 'Mixed overlay hash mismatch'}
    $minimum=(Get-Item $overlay).Length+3GB
    Fleet "set -e;test ! -e '$pack';mkdir -p '$root';test `"`$(df --output=avail -B1 /home/xander|tail -1)`" -gt $minimum" $targets|Out-Null
    Fleet "set -e;printf '%s  %s\n' '$sha' '$root/dense.gguf'|sha256sum -c -" $targets @{
        UploadPath=$overlay;UploadDestination="$root/dense.gguf"}|Out-Null
    foreach($ip in $targets){
        $text=@'
set -euo pipefail
root=__ROOT__
pack=__PACK__
old=/home/xander/fg-ring-pack-abliterated
code=/home/xander/hc-q4/stage-ablation-aligned/code
printf '%s  %s\n' 35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb "$old/manifest.fgm"|sha256sum -c -
cd "$code"
./tools/q4-repack "$old/manifest.fgm" "$old" "$root/dense.gguf" "$root/pack" __RANK__ --keep-q8-head __ROUTERS__ --cook-dense --reflink __FAMILIES__ > "$root/repack.log" 2>&1 || { tail -30 "$root/repack.log";exit 1; }
./tools/q4-repack "$root/pack/manifest.fgm" "$root/pack" "$root/dense.gguf" "$root/unused" __RANK__ --keep-q8-head __ROUTERS__ --cook-dense --plan __FAMILIES__ > "$root/resident.txt"
mv "$root/pack" "$pack"
printf 'MIX_SHA %s\n' "$(sha256sum "$pack/manifest.fgm"|cut -d' ' -f1)"
grep -E '^PLAN_(RANK|COORDINATOR)' "$root/resident.txt"
rm "$root/dense.gguf"
df -B1 /home/xander
'@
        $flags=($families|ForEach-Object{"--family $_"}) -join ' '
        $text=$text.Replace('__ROOT__',$root).Replace('__PACK__',$pack).
            Replace('__RANK__',"$($rankMap[$ip])").Replace('__FAMILIES__',$flags).
            Replace('__ROUTERS__',$routerFlags)
        $path=Join-Path $OutputDir "mix-$ip.sh"
        [IO.File]::WriteAllText($path,$text.Replace("`r",''))
        $paths+=@{IP=$ip;Path=$path}
    }
    $records=@($paths|ForEach-Object -Parallel{
        & $using:fleet -ScriptPath $_.Path -Targets @($_.IP) -PassThru
    } -ThrottleLimit 8)
    $records|ConvertTo-Json -Depth 6|Set-Content (Join-Path $OutputDir 'mixed-packs.json')
    if($records.Count -ne 8 -or @($records|Where-Object ExitCode -ne 0).Count){throw 'Mixed repack failed'}
    $seals=@($records|ForEach-Object{
        $m=[regex]::Match($_.Stdout,'MIX_SHA ([a-f0-9]{64})')
        if(-not $m.Success){throw 'Missing mixed seal'}
        $m.Groups[1].Value
    }|Sort-Object -Unique)
    if($seals.Count -ne 1){throw 'Owner mixed seals differ'}
    Write-Host "$pack $($seals[0])"
}finally{
    Remove-Item $command -ErrorAction SilentlyContinue
    foreach($item in $paths){Remove-Item $item.Path -ErrorAction SilentlyContinue}
}
