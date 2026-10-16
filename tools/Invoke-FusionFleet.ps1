#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScriptPath,
    [string[]]$Targets = @('192.168.42.51'),
    [int]$CommandTimeout = 3600,
    [string]$UploadPath = "",
    [string]$UploadDestination = "",
    [string]$DownloadSource = "",
    [string]$DownloadPath = "",
    [switch]$Sudo,
    [switch]$PassThru
)
$ErrorActionPreference = 'Stop'
foreach ($target in $Targets) {
    if ($target -notin @('192.168.42.50','192.168.42.51')) { throw "Fusion target denied: $target" }
}
$runner = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
$text = [IO.File]::ReadAllText($runner)
$old = 'tmp=`$(mktemp /tmp/bc250-fleet.XXXXXX.sh);'
$new = 'mkdir -p /home/xander/hc-fusion; tmp=/home/xander/hc-fusion/.fleet-$([guid]::NewGuid().ToString(''N'')).sh;'
if (-not $text.Contains($old)) { throw 'Fleet staging changed; review before use.' }
$text = $text.Replace($old, $new)
$text = $text.Replace('$PSScriptRoot', "'$([IO.Path]::GetDirectoryName($runner))'")
& ([scriptblock]::Create($text)) @PSBoundParameters
