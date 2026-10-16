# Current production deployment

The paired qualification in `results\hillclimb\AB_QUALIFY.md` selected **the
original deployment**, not an experimental binary. No candidate satisfied all
quality, noninferiority and improvement requirements.

| Item | Current value |
|---|---|
| Owners | 192.168.42.42–192.168.42.49; ranks 0–7 |
| API | `http://192.168.42.42:8080/v1` |
| Model | `Qwen3.8-Flash-Next-Uncensored` |
| Release | `/home/xander/flash-gordon-live/20260926-abl2` |
| Binary SHA256 | `e4b9e9e494f6650c8c2e9e2ae80be5ae21214442a53525c30ab579767788fbb2` |
| Manifest | `/home/xander/fg-ring-pack-abliterated/manifest.fgm` |
| Manifest SHA256 | `35678c379786df939d5b75ac393fa90b3924bed0942ebebe565cea7d517f4bcb` |
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
`results\hillclimb\ab-current`.

Run as `xander`. Coordinator command, from the immutable release directory:

```bash
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20260926-abl2/vulkan \
  FG_EXPERT_Q8_WIDE=0 FG_RING_TRACE=1 \
  /home/xander/flash-gordon-live/20260926-abl2/flash-gordon api \
  --manifest /home/xander/fg-ring-pack-abliterated/manifest.fgm \
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
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20260926-abl2/vulkan \
  FG_EXPERT_Q8_WIDE=0 \
  /home/xander/flash-gordon-live/20260926-abl2/flash-gordon rank \
  --manifest /home/xander/fg-ring-pack-abliterated/manifest.fgm --rank "$N" \
  >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.log" \
  2>&1 </dev/null &
echo "$!" >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.pid"
```

Do not enable `FG_DECODE_PROFILE`; it disables static replay and changes the
measurement. No fusion/MTP or other experimental switches are part of this
deployment. The original executable does not receive `--sessions`.

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
