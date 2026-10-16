#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScriptPath,
    [Parameter(Mandatory)][ValidateNotNullOrEmpty()]
    [ValidatePattern('^192\.168\.42\.4[2-9]$')][string[]]$Targets,
    [string]$UploadPath = '',
    [string]$UploadDestination = '',
    [string]$DownloadSource = '',
    [string]$DownloadPath = '',
    [int]$CommandTimeout = 3600,
    [switch]$PassThru,
    [string]$SourceFleetScript = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
)
$ErrorActionPreference = 'Stop'
$source = Get-Content -LiteralPath $SourceFleetScript -Raw
$envLine = '$envPath = Join-Path $PSScriptRoot ".env"'
if (-not $source.Contains($envLine)) { throw 'Unsupported fleet credential-path layout.' }
$envPath = Join-Path (Split-Path (Resolve-Path $SourceFleetScript).Path -Parent) '.env'
$source = $source.Replace($envLine, ('$envPath = ' + "'" + $envPath.Replace("'","''") + "'"))
$pattern = '(?m)^            \$remote = "tmp=.*\r?$'
if ([regex]::Matches($source,$pattern).Count -ne 2) { throw 'Unsupported fleet command layout.' }
$source = [regex]::Replace($source,$pattern,{
    param($match)
    if ($match.Value.Contains('sudoPwB64')) {
        return '            throw "The A/B stdin runner does not support sudo."'
    }
    return '            $remote = "printf ''%s'' ''$scriptB64'' | base64 -d | tr -d ''\015'' | env$remoteEnv bash -s"'
})
if ($source -match '/tmp|/var/tmp|\bmktemp\b') { throw 'Fleet source still contains a temporary-file path.' }
$arguments = @{}
foreach ($key in $PSBoundParameters.Keys) {
    if ($key -ne 'SourceFleetScript') { $arguments[$key] = $PSBoundParameters[$key] }
}
# Keep the user's runner/credentials/transfers, but execute its payload on stdin.
& ([scriptblock]::Create($source)) @arguments
