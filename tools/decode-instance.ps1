param(
    [Parameter(Mandatory)][ValidateSet('stage','stop','start','logs','status')][string]$Action,
    [Parameter(Mandatory)][ValidatePattern('^/home/xander/flash-gordon-live/([0-9]{8}-decode[-a-zA-Z0-9]*|20260926-abl2)$')][string]$Release,
    [Parameter(Mandatory)][string]$FleetScript,
    [ValidateSet('plain','trace','profile','control')][string]$Mode = 'plain',
    [ValidateRange(0,7)][int[]]$Ranks = @(0..7)
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$output = Join-Path $root 'results\hillclimb'
$null = New-Item -ItemType Directory -Force $output
$script = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'decode-instance.sh'))
$script = $script -replace 'action=\$\{1:\?action\}', "action='$Action'"
$script = $script -replace 'release=\$\{2:\?release\}', "release='$Release'"
$script = $script -replace 'mode=\$\{3:-plain\}', "mode='$Mode'"
$path = Join-Path $output "decode-operator-$PID.sh"
[IO.File]::WriteAllText($path, $script.Replace("`r",''))
try {
    $targets = @($Ranks | Sort-Object -Unique | ForEach-Object { "192.168.42.$(42+$_)" })
    if (-not $targets.Count) { throw 'At least one rank is required' }
    $records = @(& $FleetScript -ScriptPath $path -Targets $targets -CommandTimeout 180 -PassThru)
    $records | ConvertTo-Json -Depth 12 |
        Set-Content (Join-Path $output "decode-fleet-$Action-$Mode.json")
    foreach ($record in $records) {
        Write-Host "[$($record.IP)] $($record.ExitCode)"
        if ($Action -ne 'logs') { Write-Host $record.Stdout $record.Stderr }
    }
    if ($records.Count -ne $targets.Count -or
        @($records | Where-Object ExitCode -ne 0).Count) {
        throw 'Fleet action failed'
    }
} finally {
    Remove-Item -LiteralPath $path -Force
}
