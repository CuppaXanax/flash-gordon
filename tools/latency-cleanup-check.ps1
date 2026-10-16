#requires -Version 7.0
$ErrorActionPreference = 'Stop'
$runner = 'D:\looking-glass-labs\bc-250-dbg\Invoke-BC250Fleet.ps1'
$text = [IO.File]::ReadAllText($runner)
$pattern = '(?m)^            \$remote = "tmp=.*$'
if ([regex]::Matches($text, $pattern).Count -ne 2) { throw 'Fleet command layout changed.' }
# Read-only verification must not recreate the deleted staging directory.
$command = 'set -eu; test ! -e /home/xander/hc-lat; echo hc-lat_absent; systemctl is-active cyan-skillfish-governor-smu; sudo -n firewall-cmd --list-ports; if hostname -I | grep -q 192.168.42.53; then test -d /home/xander/hc-q4; echo hc-q4_preserved; fi'
$replacement = '            $remote = ''' + $command.Replace("'", "''") + ''''
$text = [regex]::Replace($text, $pattern, [Text.RegularExpressions.MatchEvaluator]{ param($match) $replacement })
$text = $text.Replace('$PSScriptRoot', "'$([IO.Path]::GetDirectoryName($runner))'")
& ([scriptblock]::Create($text)) -ScriptPath "$PSScriptRoot\latency-cleanup.sh" `
    -Targets @('192.168.42.52','192.168.42.53') -PassThru
