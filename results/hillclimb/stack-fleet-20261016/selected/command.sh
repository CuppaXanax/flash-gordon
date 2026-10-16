set -- 'ew0KICAicmVsZWFzZSI6ICIvaG9tZS94YW5kZXIvZmxhc2gtZ29yZG9uLWxpdmUvMjAyNjEwMTYtYWJxLXN0YWNrLWFkYXB0aXZlIiwNCiAgInNlc3Npb25zIjogMiwNCiAgIm1hbmlmZXN0IjogIi9ob21lL3hhbmRlci9mZy1yaW5nLXBhY2stYWJsaXRlcmF0ZWQtcTQtYWJsYXRlLW1peC0wNC9tYW5pZmVzdC5mZ20iLA0KICAiZmxhZ3MiOiB7DQogICAgIkZHX05HUkFNX0xPQ0FMX0NBQ0hFIjogMSwNCiAgICAiRkdfUk9PRl9ERU5TRSI6IDE5LA0KICAgICJGR19ERUNPREVfU1VCTUlUX0dST1VQIjogMiwNCiAgICAiRkdfUk9PRkxJTkUiOiAyMDQ3LA0KICAgICJGR19ST09GX0hDIjogOTYsDQogICAgIkZHX1JPT0ZfR0ROIjogMSwNCiAgICAiRkdfUk9PRl9FWFBFUlQiOiAzMzAyNCwNCiAgICAiRkdfUk9PRl9SRUNVUlJFTlQiOiAzLA0KICAgICJGR19FWFBFUlRfUThfV0lERSI6IDEsDQogICAgIkZHX1JPT0ZfUVNBIjogMw0KICB9LA0KICAiYmluYXJ5X3NoYTI1NiI6ICIwMjkzMGExNjRlMzhjOWM1ZWU1MGFlMTFiZjBhNzQ2NjAxY2M1ODdmZGZmNGUzMTIwNWRhNjg2MWU4NTBhNmUwIiwNCiAgInNoYWRlcl90cmVlX3NoYTI1NiI6ICI1Yjc1NTFkNDlhZTEwMDgxZTI5ZTg3YWFlNTlkMjg5YjFlZmY3NDc5OTMxZDA5ODY0MjU4MDMyNThkMGRjNjMzIiwNCiAgImNhcHR1cmVfc2hhMjU2IjogImM1OTcyZmZkMjJiYjI2MTM4Mzk4NjgyOTc3MzE3M2ZkZDJlN2E2ODhjMzFiZWQxODUzZTgxYWY1MmFmMmVjZWQiLA0KICAibWFuaWZlc3Rfc2hhMjU2IjogIjA4OThmZTkwZjcxZGViNmU5NDk4NDUxYmQ1ZmI4MjdhMTI4MDc5ZTk4ODFmMjA0Nzg3Yjk0YTkzM2Y4ZTUyNTIiLA0KICAiY2FwdHVyZV9wcm9ncmFtIjogIi9ob21lL3hhbmRlci9mbGFzaC1nb3Jkb24tbGl2ZS8yMDI2MTAxNi1hYnEtc3RhY2stYWRhcHRpdmUvcTQtZXZhbC1iYXRjaCIsDQogICJ3aWRlIjogMQ0KfQ=='
#!/usr/bin/env bash
set -euo pipefail
encoded=${1:?base64 deployment JSON}
rank=$(( $(ip -4 addr show | grep -oP '192\.168\.42\.\K[0-9]+' | head -1) - 42 ))
((rank >= 0 && rank < 8))
python3 - "$encoded" "$rank" <<'PY'
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

d = json.loads(base64.b64decode(sys.argv[1], validate=True))
rank = int(sys.argv[2])
run = Path("/home/xander/flash-gordon-live/20261015-abq-operator")
pid = int((run / f"rank-{rank}.pid").read_text().strip())
proc = Path(f"/proc/{pid}")
assert proc.stat().st_uid == os.getuid(), "foreign PID owner"
assert os.readlink(proc / "exe") == d["release"] + "/flash-gordon", "wrong executable"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert digest(proc / "exe") == d["binary_sha256"], "live executable hash"
assert digest(d["manifest"]) == d["manifest_sha256"], "manifest hash"
argv = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
expected = [d["release"] + "/flash-gordon"]
if rank:
    expected += ["rank", "--manifest", d["manifest"], "--rank", str(rank)]
else:
    expected += ["api", "--manifest", d["manifest"], "--model-name",
                 "Qwen3.8-Flash-Next-Uncensored", "--host", "0.0.0.0",
                 "--port", "8080", "--context-tokens", "262144",
                 "--gpu-index-tokens", "262144", "--qsa-hot-tokens", "0",
                 "--qsa-page-cache-mib", "700"]
    if d["sessions"] == 2:
        expected += ["--sessions", "2"]
assert argv == expected, ("argv", argv)
environment = dict(item.decode().split("=", 1) for item in
                   (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
flags = {"FG_SHADER_DIR": d["release"] + "/vulkan",
         "FG_EXPERT_Q8_WIDE": str(d["wide"])}
flags.update({k: str(v) for k, v in d.get("flags", {}).items()})
if not rank:
    flags["FG_RING_TRACE"] = "1"
actual = {k: v for k, v in environment.items() if k.startswith("FG_")}
assert actual == flags, ("FG environment", actual, flags)
assert "LD_PRELOAD" not in environment, "observer still injected"
current = json.loads((run / "current.json").read_text())
for key in ("release", "binary_sha256", "manifest", "manifest_sha256", "sessions", "wide"):
    assert current[key] == d[key], ("persistent identity", key)
assert {k: str(v) for k, v in current.get("flags", {}).items()} == {
    k: str(v) for k, v in d.get("flags", {}).items()}, "persistent flags"
shader_root = Path(d["release"]) / "vulkan"
entries = list(shader_root.rglob("*"))
assert not shader_root.is_symlink() and not any(p.is_symlink() for p in entries)
shader_files = sorted((p for p in entries if p.is_file() and p.suffix == ".spv"),
                      key=lambda p: p.relative_to(shader_root.parent).as_posix())
assert shader_files, "empty shader tree"
tree = hashlib.sha256("".join(
    f"{digest(p)}  {p.relative_to(shader_root.parent).as_posix()}\n"
    for p in shader_files).encode()).hexdigest()
assert tree == d["shader_tree_sha256"], "shader tree changed"
print(json.dumps({"rank": rank, "pid": pid, "argv": argv, "flags": actual,
                  "binary_sha256": digest(proc / "exe"), "shader_tree_sha256": tree,
                  "manifest_sha256": digest(d["manifest"]),
                  "launcher_sha256": digest(run / "launch-current.sh"),
                  "persistent_identity_verified": True}))
PY
