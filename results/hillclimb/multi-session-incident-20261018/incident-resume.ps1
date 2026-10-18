$src=(Get-Content 'D:\looking-glass-labs\fg-wt\multi-session-int\.build\incident-qualify.ps1' -Raw).Replace("`r",'')
$prefix=$src.Substring(0,$src.IndexOf("`nSave`ntry {"))
. ([scriptblock]::Create($prefix))
$result=Get-Content "$out\result.json" -Raw|ConvertFrom-Json -AsHashtable
$result.deployment_launcher_issue='CRLF was removed after hashing the instance script; recalculated launcher pin before starting. Candidate is healthy; original lease/deadline retained.'
$result.Remove('error');$result.Remove('restore_error')
$started=[DateTimeOffset]::Parse($result.started)
$deadline=[DateTimeOffset]::Parse($result.deadline)
$hard=[DateTimeOffset]::Parse($result.hard_deadline)
$step=110;$supervised=$true;$complete=$false;$script:currentArm='B'
$bodyStart=$src.IndexOf('  # Load the existing benchmark')
$bodyEnd=$src.IndexOf("`n} catch {`n  `$result.error")
if($bodyStart-lt0-or$bodyEnd-lt$bodyStart){throw 'Cannot identify qualification body'}
$body=$src.Substring($bodyStart,$bodyEnd-$bodyStart)
$tail=$src.Substring($bodyEnd)
. ([scriptblock]::Create("try {`n"+$body+$tail))
