#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScriptPath,
    [Parameter(Mandatory)][string[]]$Targets,
    [int]$CommandTimeout = 3600,
    [string]$UploadPath = "",
    [string]$UploadDestination = "",
    [string]$DownloadSource = "",
    [string]$DownloadPath = "",
    [switch]$PassThru
)
$ErrorActionPreference = "Stop"
$runner = "D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1"
$text = [IO.File]::ReadAllText($runner)
$old = 'tmp=`$(mktemp /tmp/bc250-fleet.XXXXXX.sh);'
$new = 'mkdir -p /home/xander/hc-q4; tmp=/home/xander/hc-q4/.fleet-$([guid]::NewGuid().ToString(''N'')).sh;'
if ($text.Contains($old)) { $text = $text.Replace($old, $new) }
elseif ($text.Contains('mktemp') -or -not $text.Contains('bash -c')) {
    throw "Fleet runner staging changed; review before use."
}
$text = $text.Replace('$PSScriptRoot', "'$([IO.Path]::GetDirectoryName($runner))'")
& ([scriptblock]::Create($text)) @PSBoundParameters
