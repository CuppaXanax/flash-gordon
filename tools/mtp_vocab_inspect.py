"""Validate a raw Q8 or FGDVOC1 draft head before fleet staging."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

VOCAB_ROWS = 248320
RAW_BYTES = VOCAB_ROWS * (2560 // 32 * 34)


def inspect(path, raw_sha256, parent_sha256):
    size = path.stat().st_size
    whole, source = hashlib.sha256(), hashlib.sha256()
    rows, kind = VOCAB_ROWS, "raw-q8"
    with path.open("rb") as file:
        if size != RAW_BYTES:
            header = file.read(44)
            if len(header) != 44 or header[:8] != b"FGDVOC1\0":
                raise ValueError("invalid draft-head container")
            rows = struct.unpack("<I", header[8:12])[0]
            if not 0 < rows <= VOCAB_ROWS or size != 44 + 4*rows + RAW_BYTES:
                raise ValueError("invalid draft-head dimensions")
            if header[12:].hex() != parent_sha256:
                raise ValueError("draft-head cooked parent mismatch")
            ids = file.read(4*rows)
            if len(ids) != 4*rows:
                raise ValueError("truncated draft IDs")
            values = struct.unpack(f"<{rows}I", ids)
            if values[-1] >= VOCAB_ROWS or any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError("draft IDs must be sorted, unique and in range")
            whole.update(header)
            whole.update(ids)
            kind = "FGDVOC1"
        for block in iter(lambda: file.read(1024*1024), b""):
            whole.update(block)
            source.update(block)
    if source.hexdigest() != raw_sha256:
        raise ValueError("draft-head complete raw source mismatch")
    return dict(format=kind, rows=rows, file_bytes=size,
                sha256=whole.hexdigest(), raw_source_sha256=source.hexdigest(),
                parent_cooked_sha256=parent_sha256,
                full_source_authenticated=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--parent-sha256", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.path, args.raw_sha256, args.parent_sha256)))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
