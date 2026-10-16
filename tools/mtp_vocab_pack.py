"""Package a draft-only ID subset with its full authenticated Q8 source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

VOCAB_ROWS = 248320
RAW_BYTES = VOCAB_ROWS * (2560 // 32 * 34)


def pack(source, output, ids, parent_sha256):
    if (not ids or len(ids) > VOCAB_ROWS or
        any(type(i) is not int or not 0 <= i < VOCAB_ROWS for i in ids) or
        len(set(ids)) != len(ids)):
        raise ValueError("IDs must be unique full-vocabulary integers")
    parent = bytes.fromhex(parent_sha256)
    if len(parent) != 32:
        raise ValueError("parent must be sealed output.weight cooked SHA256")
    if source.stat().st_size != RAW_BYTES:
        raise ValueError("source must be the complete raw 248320x2560 Q8_0 head")
    ids = sorted(ids)
    with source.open("rb") as src, output.open("xb") as dst:
        dst.write(b"FGDVOC1\0" + struct.pack("<I", len(ids)) + parent)
        dst.write(struct.pack(f"<{len(ids)}I", *ids))
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(rows=len(ids), full_rows=VOCAB_ROWS, file_bytes=output.stat().st_size,
                sha256=digest.hexdigest(), parent_cooked_sha256=parent.hex(),
                scope="Draft-only resident subset; full source retained for runtime authentication")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ids", type=Path, required=True, help="JSON list of global token IDs")
    parser.add_argument("--parent-sha256", required=True, help="Sealed output.weight cooked SHA256, not manifest/raw-head SHA")
    args = parser.parse_args()
    print(json.dumps(pack(args.source,args.output,
        json.loads(args.ids.read_text()),args.parent_sha256),indent=2))


if __name__ == "__main__":
    main()
