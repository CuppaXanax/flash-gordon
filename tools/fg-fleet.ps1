#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScriptPath,
    [Parameter(Mandatory)][ValidatePattern('^192\.168\.42\.4[2-9]$')][string[]]$Targets,
    [string]$UploadPath = '',
    [string]$UploadDestination = '',
    [int]$CommandTimeout = 180,
    [switch]$PassThru,
    [switch]$Sudo,
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
        return '            $remote = "(printf ''%s'' ''$sudoPwB64'' | base64 -d; printf ''\n'') | sudo -S -p '''' env$remoteEnv bash -c ''printf %s $scriptB64 | base64 -d | tr -d `"\015`" | bash -s''"'
    }
    return '            $remote = "printf ''%s'' ''$scriptB64'' | base64 -d | tr -d ''\015'' | env$remoteEnv bash -s"'
})
if ($source -match '/tmp|/var/tmp|\bmktemp\b') { throw 'Fleet source still writes temporary files.' }
$arguments = @{}
foreach ($key in $PSBoundParameters.Keys) {
    if ($key -ne 'SourceFleetScript') { $arguments[$key] = $PSBoundParameters[$key] }
}
& ([scriptblock]::Create($source)) @arguments
