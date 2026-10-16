# Current production deployment

**Current selection: main `a63a166` plus sealed mix-04, sessions 1, no new
arithmetic flags.** The user explicitly authorized this promotion after the
original paired study: mix-04 passed KL and had zero significant regressions.
Its six inconclusive metrics remain inconclusive; this is a revised operator
decision, **not a claim that the original noninferiority gate passed**.
The historical decision/evidence in `results\hillclimb\AB_QUALIFY.md` is retained
unchanged. Promotion, sanity and gap diagnostics are in
`results\hillclimb\PROD_GAP.md` and `prod-gap`.

| Item | Current value |
|---|---|
| Owners | 192.168.42.42–192.168.42.49; ranks 0–7 |
| API | `http://192.168.42.42:8080/v1` |
| Model | `Qwen3.8-Flash-Next-Uncensored` |
| Release | `/home/xander/flash-gordon-live/20261015-abq-main` |
| Binary SHA256 | `1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789` |
| Manifest | `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm` |
| Manifest SHA256 | `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252` |
| Sessions / expert wide | Single session; `FG_EXPERT_Q8_WIDE=0` |
| Retained operator | `/home/xander/flash-gordon-live/20261015-abq-operator` |

The binary, weights and manifest of the original release are unchanged.
Operator scripts, logs and PID records are outside that release. The same
`nohup` launch semantics as the original are retained: this survives SSH/client
logout, **not host reboot automatically**. No boot-start systemd service is
claimed. After reboot, run the persistent launcher on workers, then coordinator.

## Persistent launcher and exact commands

Each owner retains `launch-current.sh`, `ab-instance.sh` and `current.json` in
the operator directory. The launcher pins the exact release/manifest, checks
both SHA256 values before start, and derives its rank only from .42–.49.
`ab-instance.sh` refuses duplicate serving processes. Stop enumerates owned
numeric PIDs and requires the exact executable and `--manifest` argument before
TERM/wait; there is no name kill, forced kill, or floating "latest" symlink.
Archived copies and restart/identity/health receipts are in
`results\hillclimb\prod-gap\current`; the earlier original-launch receipts remain
in `results\hillclimb\ab-current`.

Run as `xander`. Coordinator command, from the immutable release directory:

```bash
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20261015-abq-main/vulkan \
  FG_EXPERT_Q8_WIDE=0 FG_RING_TRACE=1 \
  /home/xander/flash-gordon-live/20261015-abq-main/flash-gordon api \
  --manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm \
  --model-name Qwen3.8-Flash-Next-Uncensored \
  --host 0.0.0.0 --port 8080 \
  --context-tokens 262144 --gpu-index-tokens 262144 \
  --qsa-hot-tokens 0 --qsa-page-cache-mib 700 \
  >/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.log \
  2>&1 </dev/null &
echo "$!" >/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.pid
```

Workers use the following command with `N=1` on .43 through `N=7` on .49:

```bash
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20261015-abq-main/vulkan \
  FG_EXPERT_Q8_WIDE=0 \
  /home/xander/flash-gordon-live/20261015-abq-main/flash-gordon rank \
  --manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm --rank "$N" \
  >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.log" \
  2>&1 </dev/null &
echo "$!" >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.pid"
```

Do not enable `FG_DECODE_PROFILE`; it disables static replay and changes the
measurement. No fusion/MTP or other experimental switches are part of this
deployment. The executable does not receive `--sessions`.

## One-command rollback

From the `cuppaxanax/ab-harness` worktree root, with the exclusive fleet lease:

```powershell
pwsh -NoProfile -File .\tools\ab-rollback-original.ps1
```

This was actually exercised before re-promoting mix-04. It stops the exact
current executable/manifest, restores retained `launch-original.sh` and
`original.json` as the pinned current selection on all eight, starts workers
before coordinator, and checks health/model/real greedy chat. It does not rely
on a name kill or on an ephemeral local record to identify the running process.
Original release and pack remain immutable:
`20260926-abl2/flash-gordon`, SHA256
`e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2`;
original manifest SHA256
`35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb`.

## Restart the complete ring

Do not run this while another fleet operator holds the production lease.
Stop all eight, start the seven workers, then start .42. Use the retained
launcher rather than the historical name-kill scripts:

```powershell
$runner = Join-Path $PWD 'tools\ab-fleet.ps1'
$all = @('192.168.42.42','192.168.42.43','192.168.42.44','192.168.42.45',
         '192.168.42.46','192.168.42.47','192.168.42.48','192.168.42.49')
$workers = @('192.168.42.43','192.168.42.44','192.168.42.45','192.168.42.46',
             '192.168.42.47','192.168.42.48','192.168.42.49')
$null = New-Item -ItemType Directory -Force .build
$script = Join-Path $PWD '.build\restart-current.sh'
function Invoke-Current($action, $targets) {
    [IO.File]::WriteAllText($script,
        "bash '/home/xander/flash-gordon-live/20261015-abq-operator/launch-current.sh' $action`n")
    $r = @(& $runner -ScriptPath $script -Targets $targets -PassThru)
    if ($r.Count -ne $targets.Count -or @($r | Where-Object ExitCode -ne 0).Count) {
        throw "Current deployment $action failed; inspect owned-PID status."
    }
}
Invoke-Current stop $all
Invoke-Current start $workers
Invoke-Current start @('192.168.42.42')
```

Allow loading to finish, then verify `/health` reports `status=ok`,
`/v1/models` lists the exact model above, and a real greedy chat returns visible
content. `launch-current.sh status` reports process arguments and file hashes;
the final archived verification additionally checks all eight live executable
inodes, exact argv/environment and the immutable original identities.

Final verification at **2026-10-15 23:50 UTC-07** passed after a fresh complete
ring restart and experiment cleanup. All eight had only the pinned `FG_`
environment above; `LD_PRELOAD` and every diagnostic trace/profile switch were
absent. `/health` was `ok`, the model ID matched, and greedy chat returned `OK`.
Receipts: `prod-gap\current\verify.json`, `prod-gap\final-cleanup.json`, and
`prod-gap\final-api-health.json`. `/home/xander/hc-abq-gap` was removed on all
eight only after its libraries were unmapped and all evidence downloaded.
