$ErrorActionPreference='Stop'
$work='D:\looking-glass-labs\fg-wt\multi-session-int'
$src=(Get-Content "$work\.build\robust-promote.ps1" -Raw).Replace("`r",'')
$src=$src.Replace("'.build\robust-promotion'","'.build\robust-override'")
$src=$src.Replace("policy='robustness-nonregression-v1'","policy='explicit-lease-authority-override-v1'")
$src=$src.Replace('$deadline=$started.AddMinutes(55)','$deadline=$started.AddMinutes(40)')
$src=$src.Replace('$started.AddMinutes(60)','$started.AddMinutes(45)')
$src=$src.Replace('flash-gordon-robust-deadline','flash-gordon-override-deadline')
$src=$src.Replace("`nSave`ntry{",@'

$previousGate=Get-Content "$work\.build\robust-promotion\result.json" -Raw|ConvertFrom-Json -AsHashtable
$result.override='2026-10-18 lease authority explicitly accepts short-decode paired .965996 with CI crossing 1; promote unchanged candidate if functional replay, transport recovery and SIGKILL watchdog checks pass.'
$result.prior_qualification=@{commit='54ee55b';source='11c7e39';pairs=5;
  short_engine_benefit=($previousGate.metrics|Where-Object metric -eq 'decode-short.engine_tps').benefit_ratio;performance_pass=$false;
  reference_hashes=$previousGate.pairs[0].A.hashes}
Save
try{
'@)
$begin=$src.IndexOf("  Deploy 'A'`n  Qos 'A'")
$end=$src.IndexOf('  $base=New-ABPrompt 44000')
if($begin-lt0-or$end-le$begin){throw 'Qualification extraction anchors changed'}
$src=$src.Substring(0,$begin)+"  Deploy 'B'`n  Budget`n"+$src.Substring($end)
$src=$src.Replace("  Qos 'B'`n",'')
$src=$src.Replace('$result.pairs[0].A.hashes','$previousGate.pairs[0].A.hashes')
$anchor='        Verify $baseline'
$src=$src.Replace($anchor,@'
        Verify $baseline
        $restored=@(Invoke-Correctness)
        if(@($restored|Where-Object{[string]::IsNullOrWhiteSpace($_.text)}).Count-or
          -not(Test-ABCanonicalHashes $previousGate.pairs[0].A.hashes $restored)){
          throw 'Restored baseline API hashes failed'}
        $result.final_greedy=$restored
'@)
$tokens=$null;$errors=$null
$null=[Management.Automation.Language.Parser]::ParseInput($src,[ref]$tokens,[ref]$errors)
if($errors.Count){$errors|Format-List;throw 'Override driver syntax failure'}
if($src.Contains("  Qos 'A'")-or$src.Contains("  Qos 'B'")-or$src.Contains('for($round=')){
  throw 'Override must not rerun or silently pool performance samples'}
[IO.File]::WriteAllText("$work\.build\override-promote.ps1",$src,[Text.UTF8Encoding]::new($false))
'Override driver prepared; no fleet access'
