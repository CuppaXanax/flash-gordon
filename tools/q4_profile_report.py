#!/usr/bin/env python3
"""Summarize matched long-prefill logs without mistaking partial GPU coverage for totals."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re


def milliseconds(fields, key):
    value=float(fields[key])
    if not math.isfinite(value) or value<0:raise ValueError("invalid profile timestamp")
    return value


def collect(directory, prefix):
    routes = defaultdict(lambda: dict(calls=0,pairs=0,tiles=0,capacity=0,distinct=0))
    kernels = defaultdict(lambda: dict(calls=0,gpu_ms=0.0))
    layers = defaultdict(lambda: dict(calls=0,wall_ms=0.0,gpu_ms=0.0,kernel_ms=0.0))
    workers = defaultdict(lambda: dict(calls=0,decode_ms=0.0,forward_ms=0.0,
                                      compute_ms=0.0,send_ms=0.0,total_ms=0.0))
    for rank in range(8):
        path = directory/f"{prefix}-rank-{rank}.log"
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = dict(re.findall(r"([a-z_]+)=([^\s]+)",line))
            if line.startswith("PREFILL_ROUTE "):
                key=f"{rank}:{fields['layer']}:N{fields['tokens']}:W{fields['width']}"
                entry=routes[key]
                entry["calls"]+=1
                for name in ("pairs","tiles","distinct"):entry[name]+=int(fields[name])
                entry["capacity"]+=int(fields["tiles"])*int(fields["width"])
            elif line.startswith(("PREFILL_LAYER_KERNEL ","PREFILL_PROFILE_KERNEL ")):
                # Rank 0 captures only one chunk; worker captures are owner-layer windows.
                key=f"{rank}:{fields.get('scope','')}:{fields['kernel']}"
                kernels[key]["calls"]+=int(fields["calls"])
                kernels[key]["gpu_ms"]+=milliseconds(fields,"gpu_ms")
            elif line.startswith("PREFILL_BLOCK_LAYER "):
                entry=layers[f"{rank}:{fields['layer']}"]
                entry["calls"]+=1;entry["wall_ms"]+=milliseconds(fields,"ms")
            elif line.startswith("PREFILL_LAYER_PROFILE "):
                entry=layers[f"{rank}:{fields['layer']}"]
                entry["gpu_ms"]+=milliseconds(fields,"gpu_ms")
                entry["kernel_ms"]+=milliseconds(fields,"kernel_ms")
            elif line.startswith("PREFILL_WORKER "):
                entry=workers[f"{rank}:{fields['layer']}:N{fields['tokens']}"]
                entry["calls"]+=1
                for key in ("decode_ms","forward_ms","compute_ms","send_ms","total_ms"):
                    entry[key]+=milliseconds(fields,key)
    if not routes or not kernels:raise ValueError("incomplete route/kernel profile")
    totals={key:sum(r[key] for r in routes.values()) for key in
            ("calls","pairs","tiles","capacity","distinct")}
    if totals["pairs"]<=0 or any(r["capacity"]<r["pairs"] for r in routes.values()):
        raise ValueError("invalid route tile geometry")
    totals["tile_occupancy"]=totals["pairs"]/totals["capacity"]
    return dict(route_totals=totals,routes=dict(routes),kernels=dict(kernels),
                owner_layer_windows=dict(layers),
                worker_rpc_timings=dict(workers),
                coverage="Routes cover instrumented grouped expert prefill schedules. "
                         "Rank 0 GPU queries cover one chunk; ranks 1–7 queries cover "
                         "owner-layer windows, not all expert GPU work. Worker RPC durations "
                         "overlap across ranks and are not whole-model wall time.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--baseline",required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    requests=[json.loads((args.directory/f"{prefix}-131072.json").read_text(encoding="utf-8-sig"))
              for prefix in (args.baseline,args.candidate)]
    request_sha=requests[0].get("request_sha256")
    if not isinstance(request_sha,str) or not re.fullmatch("[a-f0-9]{64}",request_sha) or \
       request_sha != requests[1].get("request_sha256"):
        raise ValueError("different/missing frozen request body hashes")
    base=collect(args.directory,args.baseline)
    candidate=collect(args.directory,args.candidate)
    if base["route_totals"]["pairs"] != candidate["route_totals"]["pairs"]:
        raise ValueError("different routed-pair workloads; check exact prompt and log offsets")
    result=dict(baseline_name=args.baseline,candidate_name=args.candidate,
                request_sha256=request_sha,baseline=base,candidate=candidate)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({name:value["route_totals"] for name,value in
                     (("baseline",base),("candidate",candidate))},indent=2))


if __name__=="__main__":
    main()
