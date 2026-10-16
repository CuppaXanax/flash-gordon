"""Retain compact qualification receipts; index, but never publish, raw logits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def collect(source, destination):
    if destination.exists():
        raise ValueError("Evidence destination must be fresh")
    destination.mkdir(parents=True)
    metadata, raw = [], []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in qualification output: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        is_raw = path.suffix in {".fgl", ".f32"} or path.name.endswith(".tar.gz")
        is_metadata = path.suffix in {".json", ".jsonl", ".log", ".ps1", ".py", ".sh", ".sha256", ".fgev"}
        if not (is_raw or is_metadata):
            continue
        record = {"path": relative.as_posix(), "bytes": path.stat().st_size,
                  "sha256": digest(path)}
        if is_raw:
            raw.append(record)
        else:
            if record["bytes"] > 8 * 1024 * 1024:
                raise ValueError(f"Oversized metadata: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            metadata.append(record)
    index = {"local_source": str(source.resolve()), "metadata": metadata,
             "raw_retained_locally_only": raw}
    (destination / "evidence-index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (destination / ".gitattributes").write_text("* -text -whitespace\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    index = collect(args.source, args.destination)
    print(f"Retained {len(index['metadata'])} receipts; indexed "
          f"{len(index['raw_retained_locally_only'])} local-only raw files")
