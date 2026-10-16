#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScriptPath,
    [string[]]$Targets = @("192.168.42.51"),
    [int]$CommandTimeout = 3600,
    [string]$UploadPath = "",
    [string]$UploadDestination = "",
    [string]$DownloadSource = "",
    [string]$DownloadPath = "",
    [switch]$PassThru
)
$ErrorActionPreference = "Stop"
if (-not $Targets.Count) { throw "An explicit nonempty MTP target set is required." }
foreach ($target in $Targets) {
    if ($target -notin @("192.168.42.51", "192.168.42.52", "192.168.42.53")) {
        throw "MTP experiments may use only .51, .52 and .53."
    }
    foreach ($remotePath in @($UploadDestination, $DownloadSource)) {
        if ($remotePath -and -not $remotePath.StartsWith("/home/xander/hc-mtp/")) {
            throw "MTP transfers must stay under /home/xander/hc-mtp/."
        }
    }
}
$runner = "D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1"
$text = [IO.File]::ReadAllText($runner)
$old = 'tmp=`$(mktemp /tmp/bc250-fleet.XXXXXX.sh);'
$new = 'mkdir -p /home/xander/hc-mtp; tmp=/home/xander/hc-mtp/.fleet-$([guid]::NewGuid().ToString(''N'')).sh;'
if ($text.Contains($old)) { $text = $text.Replace($old, $new) }
elseif ($text.Contains('mktemp') -or -not $text.Contains('bash -c')) {
    throw "Fleet runner staging changed; review before use."
}
$text = $text.Replace('$PSScriptRoot', "'$([IO.Path]::GetDirectoryName($runner))'")
$PSBoundParameters["Targets"] = $Targets
& ([scriptblock]::Create($text)) @PSBoundParameters
