#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScriptPath,
    [string[]]$Targets = @('192.168.42.50'),
    [int]$CommandTimeout = 7200,
    [string]$UploadPath = "",
    [string]$UploadDestination = "",
    [string]$DownloadSource = "",
    [string]$DownloadPath = "",
    [switch]$Sudo,
    [switch]$PassThru
)
$ErrorActionPreference = 'Stop'
foreach ($name in @('ScriptPath','UploadPath','DownloadPath')) {
    if ($PSBoundParameters.ContainsKey($name) -and $PSBoundParameters[$name]) {
        $PSBoundParameters[$name]=[IO.Path]::GetFullPath($PSBoundParameters[$name],(Get-Location).Path)
    }
}
foreach ($target in $Targets) {
    if ($target -notin @('192.168.42.50','192.168.42.53')) { throw "Kernel target denied: $target" }
}
$runner = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
$text = [IO.File]::ReadAllText($runner)
$old = 'tmp=`$(mktemp /tmp/bc250-fleet.XXXXXX.sh);'
$new = 'mkdir -p /home/xander/hc-mtpk; tmp=/home/xander/hc-mtpk/.fleet-$([guid]::NewGuid().ToString(''N'')).sh;'
if (-not $text.Contains($old)) { throw 'Fleet staging changed; review before use.' }
$text = $text.Replace($old, $new)
$text = $text.Replace('$PSScriptRoot', "'$([IO.Path]::GetDirectoryName($runner))'")
& ([scriptblock]::Create($text)) @PSBoundParameters
