import json
import statistics
import subprocess
from pathlib import Path

root=Path(".build")
records=[]
for mode,count in [("receive",200000),("budget",10000000)]:
    for round in range(9):
        order=("prod","new") if round%2==0 else ("new","prod")
        for arm in order:
            command=[str(root/("decode-overhead-"+arm)),mode,str(count)]
            row=json.loads(subprocess.check_output(command,text=True))
            row.update(arm=arm,round=round)
            records.append(row)
summary=[]
for mode in ("receive","budget"):
    old=[r["ns_per_op"] for r in records if r["mode"]==mode and r["arm"]=="prod"]
    new=[r["ns_per_op"] for r in records if r["mode"]==mode and r["arm"]=="new"]
    summary.append(dict(mode=mode,prod_median_ns=statistics.median(old),
                        new_median_ns=statistics.median(new),
                        paired_delta_median_ns=statistics.median([b-a for a,b in zip(old,new)]),
                        prod_range_ns=[min(old),max(old)],new_range_ns=[min(new),max(new)]))
result=dict(method="Nine alternating local WSL CPU-only paired processes; actual matching pipeline.c/header from production31508ee and candidate11c7e39, fake immediately ready fabric; no GPU, network or model",
            summary=summary,samples=records)
(root/"decode-overhead.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(summary,indent=2))
