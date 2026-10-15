set -euo pipefail
manifest=${1:?manifest}
python3 - "$manifest" <<'PY'
import glob,json,os,re,subprocess,sys
manifest=sys.argv[1]
result={"manifest":manifest,"processes":[]}
for path in glob.glob("/proc/[0-9]*"):
    try:
        if os.stat(path).st_uid!=os.getuid(): continue
        exe=os.readlink(path+"/exe")
        if not re.fullmatch(r"/home/xander/flash-gordon-live/(?:[0-9]{8}-q4[-a-zA-Z0-9]*|20260926-abl2)/flash-gordon",exe): continue
        args=open(path+"/cmdline","rb").read().split(b"\0")
        if not any(args[i]==b"--manifest" and args[i+1].decode()==manifest for i in range(len(args)-1)): continue
        status=open(path+"/status").read()
        rss=re.search(r"^VmRSS:\s*(\d+) kB",status,re.M)
        clients={}
        for info in glob.glob(path+"/fdinfo/*"):
            try:
                fields=dict(line.split(":",1) for line in open(info) if ":" in line)
                if "drm-client-id" not in fields: continue
                identity=(fields.get("drm-pdev","").strip(),fields["drm-client-id"].strip())
                memory={}
                for key,value in fields.items():
                    if not key.startswith(("drm-memory-","drm-resident-","drm-total-")): continue
                    match=re.fullmatch(r"\s*(\d+)\s+(KiB|kB|MiB|B)\s*",value)
                    if match: memory[key]=int(match[1])*{"KiB":1024,"kB":1024,"MiB":1048576,"B":1}[match[2]]
                clients[identity]=memory
            except (OSError,ValueError): pass
        totals={}
        for memory in clients.values():
            for key,value in memory.items(): totals[key]=totals.get(key,0)+value
        result["processes"].append({"pid":int(path.rsplit("/",1)[1]),"exe":exe,
            "rss_bytes":int(rss[1])*1024 if rss else None,"drm_clients":len(clients),
            "drm_bytes":totals})
    except (OSError,UnicodeError): pass
directory=manifest.rsplit("/",1)[0]
for key,flags in (("allocated_extent_bytes",["-sB1"]),("apparent_bytes",["-sB1","--apparent-size"])):
    result[key]=int(subprocess.check_output(["du",*flags,directory],text=True).split()[0])
stat=os.statvfs(directory)
result["available_disk_bytes"]=stat.f_bavail*stat.f_frsize
print(json.dumps(result,sort_keys=True))
PY
