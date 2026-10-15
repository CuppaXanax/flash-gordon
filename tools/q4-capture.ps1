#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$Manifest,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$Coordinator = "192.168.42.42",
    [string]$OutputOwner = "192.168.42.46"
)
$ErrorActionPreference = "Stop"
$fleet = Join-Path $PSScriptRoot "Invoke-Q4Fleet.ps1"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$contexts = Join-Path $OutputDir "contexts.jsonl"
& python (Join-Path $PSScriptRoot "q4_quality.py") --contexts $contexts
if ($LASTEXITCODE) { throw "Context generation failed" }
$script = Join-Path $OutputDir "capture-command.sh"
function Run-Fleet([string]$Target, [string]$Text, [string]$Download = "") {
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($script), $Text, [Text.UTF8Encoding]::new($false))
    $parameters = @{ScriptPath=$script; Targets=@($Target); PassThru=$true}
    if ($Download) {
        $parameters.DownloadSource = "/home/xander/hc-q4/eval-latest.fgl"
        $parameters.DownloadPath = $Download
    }
    $result = & $fleet @parameters
    if (@($result | Where-Object ExitCode -ne 0).Count) {
        throw "Capture command failed on ${Target}: $($result.Stderr) $($result.Stdout)"
    }
}
function Base64([string]$Value) { [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Value)) }
$build64 = Base64 $BuildDir
$manifest64 = Base64 $Manifest
try {
    foreach ($line in Get-Content -LiteralPath $contexts -Encoding utf8) {
        $item = $line | ConvertFrom-Json
        $destination = Join-Path $OutputDir "$($item.id).fgl"
        if (Test-Path -LiteralPath $destination) { throw "Refuse stale capture: $destination" }
        Run-Fleet $OutputOwner "set -e; rm -f /home/xander/hc-q4/eval-latest.fgl"
        $prompt64 = Base64 $item.prompt
        $command = @'
set -euo pipefail
build=$(printf '%s' '__BUILD__' | base64 -d)
manifest=$(printf '%s' '__MANIFEST__' | base64 -d)
prompt=$(printf '%s' '__PROMPT__' | base64 -d; printf '.')
prompt="${prompt%.}"
cd "$build"
./flash-gordon eval --manifest "$manifest" --prompt "$prompt" --generate 0
'@
        $command = $command.Replace("__BUILD__", $build64).Replace("__MANIFEST__", $manifest64).Replace("__PROMPT__", $prompt64)
        Run-Fleet $Coordinator $command
        Run-Fleet $OutputOwner "test -s /home/xander/hc-q4/eval-latest.fgl" $destination
        Write-Host "Captured $($item.id)"
    }
} finally {
    Remove-Item -LiteralPath $script -ErrorAction SilentlyContinue
}
