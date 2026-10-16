"""Reconstruct only the sealed ngram tensor using bounded pinned HTTP ranges."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil

from mtp_format_audit import load_manifest
from mtp_source import Source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_map", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.source_map.read_text())
    assert load_manifest(args.manifest)["sha256"] == metadata["manifest_sha256"]
    records = [r for r in metadata["records"] if r["target"]["kind"] == 3]
    assert len(records) == 1 and not records[0]["overlay"]
    record = records[0]
    target, span = record["target"], record["source"]
    assert span["bytes"] == target["bytes"] and span["type"] == target["type"] == 20
    source = Source(metadata["repo"], metadata["revision"])
    assert source.revision == metadata["revision"]
    args.directory.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(args.directory).free > target["bytes"] + (2 << 30)
    path = args.directory / "ngram.iq4nl.partial"
    digest = hashlib.sha256()
    chunk = 16 << 20
    with path.open("wb") as output, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(0, span["bytes"], 4*chunk):
            requests = [(span["file_offset"]+i, min(chunk, span["bytes"]-i))
                        for i in range(begin, min(begin+4*chunk, span["bytes"]), chunk)]
            for data in pool.map(lambda pair: source.read_range(span["shard"], *pair), requests):
                output.write(data)
                digest.update(data)
            if begin % (1 << 30) == 0:
                print(json.dumps(dict(downloaded=min(begin+4*chunk, span["bytes"]),
                                      total=span["bytes"])), flush=True)
        assert digest.hexdigest() == target["sha256"], "sealed ngram SHA differs"
        output.truncate((target["bytes"]+4095) & ~4095)
    path.replace(args.directory / "ngram.iq4nl")
    receipt = dict(bytes=target["bytes"], sha256=digest.hexdigest(),
                   repo=metadata["repo"], revision=source.revision,
                   shard=span["shard"], file_offset=span["file_offset"])
    (args.directory / "ngram-source.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
