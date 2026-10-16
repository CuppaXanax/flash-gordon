"""Extract only complete, regular full-vocabulary capture records."""
import argparse
import shutil
import tarfile
from pathlib import Path


def extract(archive, output, count):
    expected = {f"{i:04d}.fgl" for i in range(count)}
    seen = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = member.name.removeprefix("./")
            if member.isdir() and name in ("", "."):
                continue
            if (name not in expected or name in seen or not member.isfile()
                    or member.size != 12 + 248320 * 4):
                raise ValueError(f"Invalid capture archive member: {member.name}")
            seen.add(name)
            with source.extractfile(member) as incoming, (Path(output) / name).open("xb") as target:
                shutil.copyfileobj(incoming, target)
    if seen != expected:
        raise ValueError("Incomplete capture archive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("output")
    parser.add_argument("count", type=int)
    args = parser.parse_args()
    if not 400 <= args.count <= 10000:
        parser.error("Expected 400-10000 full-vocabulary frontiers")
    extract(args.archive, args.output, args.count)
