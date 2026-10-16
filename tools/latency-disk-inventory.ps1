#requires -Version 7.0
param([switch]$PassThru)
$ErrorActionPreference = 'Stop'
$runner = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
$text = [IO.File]::ReadAllText($runner)
$pattern = '(?m)^            \$remote = "tmp=.*$'
if ([regex]::Matches($text, $pattern).Count -ne 2) { throw 'Fleet command layout changed.' }
# No script staging or disk writes on the qualification fleet.
$replacement = '            $remote = "df -B1 /home/xander; lsblk -o NAME,TYPE,SIZE,ROTA,MODEL"'
$text = [regex]::Replace($text, $pattern, [Text.RegularExpressions.MatchEvaluator]{ param($match) $replacement })
$text = $text.Replace('$PSScriptRoot', "'$([IO.Path]::GetDirectoryName($runner))'")
& ([scriptblock]::Create($text)) -ScriptPath "$PSScriptRoot\latency-probe.sh" `
    -Targets @(42..49 | ForEach-Object { "192.168.42.$_" }) -PassThru:$PassThru
