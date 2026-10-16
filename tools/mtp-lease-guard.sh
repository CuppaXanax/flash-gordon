#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
root=/home/xander/hc-mtp
operator=/home/xander/flash-gordon-live/20261015-abq-operator
prod=/home/xander/flash-gordon-live/20261016-abq-stack-adaptive/flash-gordon
pack=/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04
binary_sha=02930a164e38c9c5ee50ae11bf0a746601cc587fdff4e31205da6861e850a6e0
shader_sha=5b7551d49ae10081e29e87aae59d289b1eff7479931d0986425803258d0dc633
manifest_sha=0898fe90f71deb6e9498451bd5fb827a128079e9881f204787b94a933f8e5252
ip="$(hostname -I)"
rank=-1
for last in {42..49}; do
    if [[ " $ip " == *" 192.168.42.$last "* ]]; then rank=$((last-42)); fi
done
[[ "$rank" -ge 0 ]] || { echo "not a production ring owner" >&2; exit 1; }

identity() {
    local pid="$1" executable="$2" manifest="$3"
    [[ "$pid" =~ ^[1-9][0-9]*$ && -d "/proc/$pid" ]] || return 1
    [[ "$(readlink "/proc/$pid/exe")" == "$executable" ]] || return 1
    python3 - "$pid" "$manifest" <<'PY'
import pathlib,sys
args=pathlib.Path('/proc/'+sys.argv[1]+'/cmdline').read_bytes().split(b'\0')
expected=sys.argv[2].encode()
assert any(args[i]==b'--manifest' and args[i+1]==expected for i in range(len(args)-1))
PY
}
stop_candidate() {
    [[ -f "$root/candidate.pid" ]] || return 0
    local pid
    pid="$(cat "$root/candidate.pid")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    if [[ ! -d "/proc/$pid" ]]; then rm "$root/candidate.pid"; return 0; fi
    identity "$pid" "$root/release/flash-gordon" "$root/pack/manifest.fgm" || {
        echo "refusing non-owned PID $pid" >&2; return 1;
    }
    kill -TERM "$pid"
    for _ in {1..100}; do
        if [[ ! -e "/proc/$pid/exe" ]]; then rm "$root/candidate.pid"; return 0; fi
        sleep 0.2
    done
    echo "owned candidate did not stop; no forced kill" >&2
    return 1
}
check_hashes() {
    printf '%s  %s\n' "$binary_sha" "$prod" "$manifest_sha" "$pack/manifest.fgm" | sha256sum -c -
    local tree
    tree="$(cd "$(dirname "$prod")"; find vulkan -type f -name '*.spv' -print0 |
        LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d ' ' -f 1)"
    [[ "$tree" == "$shader_sha" ]]
}
prod_identity() {
    local pid
    pid="$(cat "$operator/rank-$rank.pid")"
    identity "$pid" "$prod" "$pack/manifest.fgm"
    python3 - "$pid" "$rank" <<'PY'
import pathlib,sys
env=dict(x.split(b'=',1) for x in pathlib.Path('/proc/'+sys.argv[1]+'/environ').read_bytes().split(b'\0') if b'=' in x)
args=pathlib.Path('/proc/'+sys.argv[1]+'/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
rank=int(sys.argv[2])
options={b'--manifest':b'/home/xander/fg-ring-pack-abliterated-q4-ablate-mix-04/manifest.fgm'}
if rank==0:
    options.update({b'--model-name':b'Qwen3.8-Flash-Next-Uncensored',b'--host':b'0.0.0.0',
        b'--port':b'8080',b'--context-tokens':b'262144',b'--gpu-index-tokens':b'262144',
        b'--qsa-hot-tokens':b'0',b'--qsa-page-cache-mib':b'700',b'--sessions':b'2'})
else:
    options[b'--rank']=str(rank).encode()
assert args[1]==(b'api' if rank==0 else b'rank')
assert len(args)==2+2*len(options) and dict(zip(args[2::2],args[3::2]))==options
expected={b'FG_SHADER_DIR':b'/home/xander/flash-gordon-live/20261016-abq-stack-adaptive/vulkan',
    b'FG_EXPERT_Q8_WIDE':b'1',b'FG_ROOFLINE':b'2047',b'FG_ROOF_DENSE':b'19',
    b'FG_ROOF_QSA':b'3',b'FG_ROOF_EXPERT':b'33024',b'FG_ROOF_HC':b'96',
    b'FG_ROOF_RECURRENT':b'3',b'FG_ROOF_GDN':b'1',b'FG_NGRAM_LOCAL_CACHE':b'1',
    b'FG_DECODE_SUBMIT_GROUP':b'2'}
if sys.argv[2]=='0': expected[b'FG_RING_TRACE']=b'1'
assert {k:v for k,v in env.items() if k.startswith(b'FG_')}==expected
assert not env.get(b'LD_PRELOAD')
PY
}
case "${1:-}" in
preflight)
    check_hashes
    [[ -f "$operator/launch-current.sh" ]]
    prod_identity
    bash "$operator/launch-current.sh" status
    date +%s
    ;;
stage)
    check_hashes
    [[ ! -e "$root/candidate.pid" && ! -e "$root/watchdog.pid" ]]
    mkdir -p "$root/release" "$root/pack" "$root/capture" "$root/corpus" "$root/.cache"
    tar -xzf "$root/release.tar.gz" -C "$root/release"
    (cd "$root/release"; sha256sum -c release.sha256)
    cp "$pack/manifest.fgm" "$root/pack/manifest.fgm"
    for source in "$pack"/rank-??.fgw "$pack"/ngram-rank-??.iq4nl \
        "$pack"/ngram.iq4nl "$pack"/tokenizer; do
        [[ -e "$source" ]] && ln -s "$source" "$root/pack/$(basename "$source")"
    done
    tar -xf "$root/corpus.tar" -C "$root/corpus"
    ;;
stage-serving)
    check_hashes
    [[ ! -e "$root/candidate.pid" && ! -e "$root/watchdog.pid" ]]
    mkdir -p "$root/release" "$root/pack" "$root/capture" "$root/.cache"
    tar -xzf "$root/release.tar.gz" -C "$root/release"
    (cd "$root/release"; sha256sum -c release.sha256)
    cp "$pack/manifest.fgm" "$root/pack/manifest.fgm"
    for source in "$pack"/rank-??.fgw "$pack"/ngram-rank-??.iq4nl \
        "$pack"/ngram.iq4nl "$pack"/tokenizer; do
        [[ -e "$source" ]] && ln -s "$source" "$root/pack/$(basename "$source")"
    done
    if [[ "$rank" == 0 ]]; then
        "$root/release/tools/mtp-embedding" "$root/pack/manifest.fgm" "$root/embedding.fgme"
        sha256sum "$root/embedding.fgme"
    fi
    ;;
arm)
    deadline="${2:?absolute Unix restore deadline}"
    [[ "$deadline" =~ ^[0-9]+$ && "$deadline" -gt "$(date +%s)" ]]
    printf '%s\n' "$deadline" > "$root/lease.deadline"
    nohup bash "$root/lease.sh" watchdog "$deadline" >"$root/watchdog.log" 2>&1 </dev/null &
    echo "$!" > "$root/watchdog.pid"
    sleep 0.1
    kill -0 "$(cat "$root/watchdog.pid")"
    ;;
watchdog)
    deadline="${2:?}"
    while [[ "$(date +%s)" -lt "$deadline" ]]; do
        [[ ! -e "$root/watchdog.cancel" ]] || exit 0
        sleep 1
    done
    [[ ! -e "$root/watchdog.cancel" ]] || exit 0
    echo "restore watchdog expired $(date -Is)"
    stop_candidate
    check_hashes
    if ! prod_identity >/dev/null 2>&1; then bash "$operator/launch-current.sh" start; fi
    prod_identity
    bash "$operator/launch-current.sh" status
    ;;
stop-prod)
    check_hashes
    bash "$operator/launch-current.sh" stop
    ;;
start)
    [[ ! -e "$root/candidate.pid" ]]
    command=(rank --manifest "$root/pack/manifest.fgm" --rank "$rank"
        --mtp-capture-dir "$root/capture")
    if [[ "$rank" == 0 ]]; then
        command=(mtp-qualify --manifest "$root/pack/manifest.fgm"
            --cases "$root/corpus/cases.tsv" --directory "$root/capture"
            --budget-seconds "${2:-1800}")
    fi
    nohup env -i HOME=/home/xander USER=xander PATH="$PATH" LANG=C.UTF-8 \
        TMPDIR="$root/.cache" XDG_CACHE_HOME="$root/.cache" \
        FG_SHADER_DIR="$root/release/vulkan" FG_EXPERT_Q8_WIDE=0 \
        "$root/release/flash-gordon" "${command[@]}" >"$root/rank-$rank.log" 2>&1 </dev/null &
    echo "$!" > "$root/candidate.pid"
    sleep 0.2
    identity "$(cat "$root/candidate.pid")" "$root/release/flash-gordon" "$root/pack/manifest.fgm"
    ;;
start-serving)
    phase="${2:?phase name}"; cap="${3:?maximum draft (0=off)}"
    small="${4:-0}"
    sessions="${5:-2}"; fixed="${6:-0}"; local_head="${7:-0}"
    qsa_cache="${8:-700}"
    [[ "$phase" =~ ^[A-Za-z0-9_-]{1,32}$ && "$cap" =~ ^[0-3]$ ]]
    [[ "$small" == 0 || "$small" == 330 ]]
    [[ "$sessions" =~ ^[12]$ && "$fixed" =~ ^[0-3]$ && "$fixed" -le "$cap" ]]
    [[ "$local_head" =~ ^[01]$ ]]
    [[ "$qsa_cache" == 440 || "$qsa_cache" == 700 ]]
    [[ ! -e "$root/candidate.pid" && ! -e "$root/watchdog.cancel" ]]
    find "$root/pack" -maxdepth 1 -type f -name 'qsa*.state' -delete
    [[ "$(( $(date +%s)+30 ))" -lt "$(cat "$root/lease.deadline")" ]]
    command=(rank --manifest "$root/pack/manifest.fgm" --rank "$rank")
    trace=()
    if [[ "$rank" == 0 ]]; then
        command=(api --manifest "$root/pack/manifest.fgm"
            --model-name Qwen3.8-Flash-Next-Uncensored --host 0.0.0.0 --port 8080
            --context-tokens 262144 --gpu-index-tokens 262144
            --qsa-hot-tokens 0 --qsa-page-cache-mib "$qsa_cache" --sessions "$sessions")
        trace=(FG_RING_TRACE=1)
        if [[ "$cap" != 0 ]]; then
            command+=(--experimental-mtp --mtp-overlay "$root/native-q8.gguf"
                --mtp-embedding "$root/embedding.fgme" --mtp-window 8192 --mtp-max-draft "$cap")
            if [[ "$fixed" != 0 ]]; then command+=(--mtp-fixed-draft "$fixed"); fi
            if [[ "$local_head" == 1 ]]; then command+=(--mtp-local-head "$root/local-head.q8"); fi
        fi
    fi
    (
        cd "$root/release"
        exec nohup env -i HOME=/home/xander USER=xander PATH="$PATH" LANG=C.UTF-8 \
            TMPDIR="$root/.cache" XDG_CACHE_HOME="$root/.cache" \
            FG_SHADER_DIR="$root/release/vulkan" FG_EXPERT_Q8_WIDE=1 \
            FG_ROOFLINE=2047 FG_ROOF_DENSE=19 FG_ROOF_QSA=3 FG_ROOF_EXPERT=33024 \
            FG_ROOF_HC=96 FG_ROOF_RECURRENT=3 FG_ROOF_GDN=1 \
            FG_NGRAM_LOCAL_CACHE=1 FG_DECODE_SUBMIT_GROUP=2 "${trace[@]}" \
            FG_SMALL_KERNELS="$small" FG_DECODE_MS=1 \
            FG_QSA_WORKER_CACHE_MIB="$qsa_cache" \
            "$root/release/flash-gordon" "${command[@]}"
    ) >"$root/serving-$phase-rank-$rank.log" 2>&1 </dev/null &
    echo "$!" > "$root/candidate.pid"
    sleep 0.2
    identity "$(cat "$root/candidate.pid")" "$root/release/flash-gordon" "$root/pack/manifest.fgm"
    ;;
status)
    if [[ -f "$root/capture/qualification.failed" ]]; then
        cat "$root/capture/qualification.failed"; exit 1
    fi
    if [[ -f "$root/capture/qualification.passed" ]]; then echo qualification-passed
    else
        identity "$(cat "$root/candidate.pid")" "$root/release/flash-gordon" "$root/pack/manifest.fgm"
        if [[ -f "$root/capture/baseline.ready" ]]; then echo baseline-ready
        else echo baseline-running; fi
    fi
    ;;
pack-baseline)
    cd "$root/capture"
    if [[ "$rank" == 0 ]]; then
        [[ -f baseline.ready ]]
        cp "$root/pack/manifest.fgm" manifest.fgm
        tar -cf "$root/baseline-0.tar" cases.tsv *.tokens.u32 embedding.q8 head.q8 manifest.fgm
    else
        tar -cf "$root/baseline-$rank.tar" hc-*.f32 hc-*.coverage
    fi
    sha256sum "$root/baseline-$rank.tar"
    ;;
proposals)
    [[ "$rank" == 0 && -f "$root/capture/baseline.ready" ]]
    tar -xf "$root/proposals.tar" -C "$root/capture"
    [[ -f "$root/capture/acceptance-summary.json" ]]
    touch "$root/capture/proposals.ready"
    ;;
stop-candidate) stop_candidate ;;
restore)
    stop_candidate
    check_hashes
    if ! prod_identity >/dev/null 2>&1; then bash "$operator/launch-current.sh" start; fi
    prod_identity
    ;;
verify)
    check_hashes
    prod_identity
    bash "$operator/launch-current.sh" status
    ;;
cancel)
    touch "$root/watchdog.cancel"
    ;;
evidence)
    cd "$root"
    shopt -s nullglob
    hashfiles=(capture/hc-* capture/routes-*)
    : > capture-hashes.sha256
    if [[ "${#hashfiles[@]}" -gt 0 ]]; then sha256sum "${hashfiles[@]}" > capture-hashes.sha256; fi
    files=(capture/*.json capture/*.jsonl capture/*.u32 capture/routes-*.coverage
        capture/qualification.*
        rank-*.log serving-*.log serving-*.json watchdog*.log lease.sh
        capture-hashes.sha256 pack/manifest.fgm release/release.sha256)
    if [[ -f capture/cases.tsv ]]; then files+=(capture/cases.tsv); fi
    if [[ -f transport.jsonl ]]; then files+=(transport.jsonl transport/bench); fi
    tar -czf final-evidence.tar.gz "${files[@]}"
    sha256sum final-evidence.tar.gz
    ;;
cleanup)
    [[ -f "$root/watchdog.cancel" ]]
    sleep 2
    if [[ -f "$root/watchdog.pid" ]]; then
        pid="$(cat "$root/watchdog.pid")"
        if [[ -e "/proc/$pid/cmdline" ]]&&tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$root/lease.sh"; then
            echo "watchdog has not exited" >&2; exit 1
        fi
    fi
    cd /home/xander
    rm -rf "$root"
    [[ ! -e "$root" ]]
    echo "hc-mtp removed"
    ;;
*) echo "unknown lease action" >&2; exit 2 ;;
esac
