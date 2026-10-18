# Current production deployment

**Latest robustness requalification, 2026-10-18: no runtime promotion.**
Five new pairs pass 32K decode, TTFT and all 25 canonical greedy comparisons,
but short decode's paired-median benefit is **0.965996**, below the explicit
0.97 floor (CI [0.879970, 1.016834]: not a demonstrated statistical regression).
The exact stack baseline below was restored and all eight identities, health,
models and five real greedy API generations verified at
**2026-10-18 22:05:49 UTC**. The new 60-minute lease was released after
**36.23 minutes**. Main was not advanced. Production's cold-32K loaded chat
already falls to **8.37 TPS**, with **200.07 ms p95 event gaps**; QoS was not
this run's veto. Replay/transport/SIGKILL rechecks were skipped on fail-fast
restoration after the solo guard failed.
[Full new results on the incident branch](https://github.com/CuppaXanax/flash-gordon/blob/cuppaxanax/multi-session-int/results/hillclimb/MULTI_SESSION_ROBUSTNESS.md).

**Current selection: stack-1/adaptive integration `31508ee` plus sealed
mix-04, sessions 2, with the explicit full flag recipe below.** Five paired
rounds against previous production `2d0f8a7` passed all six primary guards:
paired median gains include short decode 12.0%, 32K decode 11.1%, 32K prefill
5.4%, and concurrent aggregate 15.3%. Full-vocabulary KL/top-1 and exact
same-kernel submission-group hashes passed.
**Tradeoff: secondary cached-prefix TTFT is 4.35% slower by paired median
latency ratio.** It is descriptive, not a veto under the user's new policy.
[Full decision, spreads and marginal CIs](../results/hillclimb/STACK_FLEET.md).
The underlying adaptive scheduler's earlier loaded-QoS/mixed-prefill
limitations are not requalified here. No near-262K memory forensic was run.
The sealed pack, historical `AB_QUALIFY.md` and `PROD_GAP.md` remain unchanged.

**2026-10-18 incident follow-up: runtime selection remains this exact stack
baseline; incident runtime fixes are NOT promoted.** An ordered ring watchdog
is now installed, boot-enabled and restart-tested on rank 0. Incident evidence
and the final routing follow-up are on `cuppaxanax/multi-session-int`, in
`results/hillclimb/MULTI_SESSION_INCIDENT.md`. The old baseline still has its
prefix-routing limitation; supervision does not turn it into the new runtime.
Restoration/API identity was verified at 21:18:28 UTC. The single validation
lease overran to 47.17 minutes while correcting cold-start memlock inheritance;
the fleet is released, with no runtime promotion.

| Item | Current value |
|---|---|
| Owners | 192.168.42.42–192.168.42.49; ranks 0–7 |
| API | `http://192.168.42.42:8080/v1` |
| Model | `Qwen3.8-Flash-Next-Uncensored` |
| Release | `/home/xander/flash-gordon-live/20261016-abq-stack-adaptive` |
| Binary SHA256 | `02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0` |
| Source | Runtime build `31508ee`; branch tip `7381f93` changes only build-product ignores |
| Shader-tree SHA256 | `5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633` |
| Manifest | `/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm` |
| Manifest SHA256 | `0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252` |
| Sessions / expert wide | 2 sessions; `FG_EXPERT_Q8_WIDE=1` |
| Retained operator | `/home/xander/flash-gordon-live/20261015-abq-operator` |

The binary, weights and manifest of the original release are unchanged.
Operator scripts, logs and PID records are outside that release. The same
`nohup` launch semantics as the original are retained. The enabled
`flash-gordon-ring.service` on .42 now provides ordered failure/boot recovery:
stop coordinator, stop workers, start workers, start coordinator. It uses the
same pinned `launch-current.sh`/`current.json`, dedicated restricted SSH and
strict host keys. The old baseline needs a full coordinator restart on a
transport fault; listener-preserving recovery belongs to the unpromoted code.
The unit must retain `LimitMEMLOCK=infinity`: the default systemd allowance
caused `register fixed buffer: Cannot allocate memory` during the cold restart.
After correcting it, ordered startup reached `ring_ready` at 21:17:37 UTC.
This was a cold process restart, not a host reboot.

## Required supervision pause for every future fleet lease

Before any deployment, rollback or manual restart, run on .42:

```bash
op=/home/xander/flash-gordon-live/20261015-abq-operator
touch "$op/supervisor.pause"
sudo systemctl stop flash-gordon-ring.service
sudo systemctl is-active flash-gordon-ring.service  # must report inactive
```

Stopping this watchdog deliberately leaves serving processes alone
(`KillMode=process`). After restoring/pinning and verifying the selected ring:

```bash
rm -f "$op/supervisor.pause"
sudo systemctl start flash-gordon-ring.service
sudo systemctl is-enabled flash-gordon-ring.service
sudo systemctl is-active flash-gordon-ring.service
```

Do not leave a qualification deadline timer armed when releasing a fleet lease.
Restarting this unit alone does not restart an already healthy serving ring.

## Persistent launcher and exact commands

Each owner retains `launch-current.sh`, `launch-stack-3bac55ceb9c2.sh`,
`ab-instance-stack-3bac55ceb9c2.sh` and `current.json` in
the operator directory. The launcher pins the exact release/manifest, checks
the launcher body, binary, manifest and shader-tree hashes before start, and
derives its rank only from .42–.49.
The pinned instance script refuses duplicate serving processes. Stop enumerates owned
numeric PIDs and requires the exact executable and `--manifest` argument before
TERM/wait; there is no name kill, forced kill, or floating "latest" symlink.
Current launcher and complete restart/identity/health receipts are in the
[qualification archive](../results/hillclimb/stack-fleet-20261016/selected).
The earlier `prod-gap\current` and `ab-current` receipts are historical.

Run as `xander`. Coordinator command, from the immutable release directory:

```bash
ulimit -c 0
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20261016-abq-stack-adaptive/vulkan \
  FG_EXPERT_Q8_WIDE=1 FG_RING_TRACE=1 \
  FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024 \
  FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 \
  FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2 \
  /home/xander/flash-gordon-live/20261016-abq-stack-adaptive/flash-gordon api \
  --manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm \
  --model-name Qwen3.8-Flash-Next-Uncensored \
  --host 0.0.0.0 --port 8080 \
  --context-tokens 262144 --gpu-index-tokens 262144 \
  --qsa-hot-tokens 0 --qsa-page-cache-mib 700 --sessions 2 \
  >/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.log \
  2>&1 </dev/null &
echo "$!" >/home/xander/flash-gordon-live/20261015-abq-operator/rank-0.pid
```

Workers use the following command with `N=1` on .43 through `N=7` on .49:

```bash
ulimit -c 0
nohup env \
  FG_SHADER_DIR=/home/xander/flash-gordon-live/20261016-abq-stack-adaptive/vulkan \
  FG_EXPERT_Q8_WIDE=1 \
  FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024 \
  FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 \
  FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2 \
  /home/xander/flash-gordon-live/20261016-abq-stack-adaptive/flash-gordon rank \
  --manifest /home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm --rank "$N" \
  >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.log" \
  2>&1 </dev/null &
echo "$!" >"/home/xander/flash-gordon-live/20261015-abq-operator/rank-$N.pid"
```

Do not enable `FG_DECODE_PROFILE`; it disables static replay and changes the
measurement. No fusion/MTP or additional unlisted switches are part of this
deployment. The pinned script scrubs profiling/other experiment flags before
supplying the explicit recipe above. Only the coordinator receives `--sessions 2`. The QSA cache budget
is per session. Clients need stable, distinct top-level OpenAI `user` values
(for example `chat` and `agent`) for resident prefix affinity; omitting `user`
does not provide durable conversation affinity.

## Roll back to pre-stack adaptive production (sessions 2)

With the exclusive fleet lease:

```powershell
pwsh -NoProfile -File .\tools\ab-rollback-pre-stack.ps1
```

This stops the exact pinned current instance, restores retained
`launch-pre-stack-20261016.sh`/`pre-stack-20261016.json`, starts workers before
coordinator, and verifies all eight live executable/manifest/shader hashes,
argv, FG environment, model, health and real greedy generation.
It restores `2d0f8a7`, release `20261016-abq-multisess`, binary
`de6117dec8302daab1666fa9a990ec3c97f817acc1671160cad06f005324ef58`,
same mix-04, sessions 2 and wide 0. This actual stack-to-adaptive rollback
was exercised before the final full-stack re-promotion/restart.

## Older main/mix-04 fallback (sessions 1)

From this worktree, with the exclusive fleet lease:

```powershell
pwsh -NoProfile -File .\tools\ab-rollback-main-mix04.ps1
```

This uses retained `launch-main-mix04.sh`/`main-mix04.json`, stops owned numeric
PIDs through the current hash-pinned launcher, starts workers before coordinator,
updates local live state, and verifies health/models/real generation. It restores
`a63a166`, binary SHA256
`1ae81229f037c60bad39693292ec73ff70e7c5aeaef02bc6e61be22acdce8789`,
with the same sealed mix-04 manifest and no `--sessions` argument. The new
wrapper is syntax-checked; its A configuration was exercised repeatedly in the
paired study. The wrapper was not invoked after the final selected restart.

## Historical original-Q8 rollback (not the main/mix-04 fallback)

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
inodes, exact argv/environment and the pinned current identities.

Final verification on **2026-10-16 UTC-07**, after the complete selected ring
restart and observer cleanup, passed. All eight live executable/manifest hashes,
argv and allowed FG environment were checked. Health/models and real greedy
generation passed; a fresh genuine two-stream generation also passed after
restart. Only the flags above are present; no `LD_PRELOAD` or decode profiler.
[Final timestamp and API receipt](../results/hillclimb/stack-fleet-20261016/final-api-health.json);
the sibling `final-cleanup.json` verifies all-eight experiment-scratch cleanup.
Retained launchers survive logout; automatic host-boot/systemd persistence is
not claimed.
