$src=(Get-Content 'D:\looking-glass-labs\fg-wt\multi-session-int\.build\incident-qualify.ps1' -Raw).Replace("`r",'')
. ([scriptblock]::Create($src.Substring(0,$src.IndexOf("`nSave`ntry {"))))
$result=Get-Content "$out\result.json" -Raw|ConvertFrom-Json -AsHashtable
$started=[DateTimeOffset]::Parse($result.started)
$baseline=$result.A;$step=750
$result.selected='A';$result.eligible=$false
$result.final_source_paired_gate=$false
$result.final_routing_source='182d47e'
$result.final_routing_binary_sha256='3405c8976eca580ca2e8690bb336b0ca13e69ab761dbd7bffee590bb280cad17'
$result.final_routing_warm_pass=$true
$result.selection_reason='Final replay/busy-slot routing was not included in five prototype pairs; loaded-prefill QoS remains weak. No promotion.'
$result.restore_resource_fault='Systemd inherited default memlock, causing coordinator fixed-buffer registration ENOMEM. Set LimitMEMLOCK=infinity and restart-tested the watchdog; baseline is healthy.'
$result.lease_overrun_reason='Emergency restoration after cold-start memlock failure extended the original 45-minute window. No further performance experiments were run.'
$close=(Get-Content "$work\.build\incident-finalize.ps1" -Raw).Replace("`r",'')
$tail=$close.Substring($close.IndexOf('  $verify=Get-Content'))
$tail=$tail.Substring(0,$tail.LastIndexOf("`n}"))
. ([scriptblock]::Create($tail))
