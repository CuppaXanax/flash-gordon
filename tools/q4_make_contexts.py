#!/usr/bin/env python3
"""Freeze distinct, UTF-8-safe next-token frontiers from Q8 teacher continuations."""
import argparse
import json
from pathlib import Path
import struct
from q4_quality import PROMPTS


def base_contexts():
    return [(f"{i:02}", f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            for i, prompt in enumerate(PROMPTS)]


def expand(raw, per_prompt):
    if raw[:8] != b"FGCONT01":
        raise ValueError("not a teacher continuation capture")
    prefixes = [bytearray() for _ in PROMPTS]
    expected = [0 for _ in PROMPTS]
    counts = [0 for _ in PROMPTS]
    seen = [set() for _ in PROMPTS]
    contexts, positions, offset = [], [], 8
    base = base_contexts()
    while offset < len(raw):
        if offset+16 > len(raw):
            raise ValueError("truncated continuation record")
        prompt, step, token, nbytes = struct.unpack_from("<IIII", raw, offset)
        offset += 16
        if prompt >= len(PROMPTS) or step != expected[prompt] or offset+nbytes > len(raw):
            raise ValueError("invalid continuation order/length")
        expected[prompt] += 1
        prefix = bytes(prefixes[prompt])
        if counts[prompt] < per_prompt and prefix not in seen[prompt] and b"\0" not in prefix:
            try:
                text = prefix.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                pass
            else:
                key = f"{prompt:02}-{step:03}"
                contexts.append((key, base[prompt][1]+text))
                positions.append(dict(id=key, prompt=prompt, continuation_step=step,
                                      teacher_next_token=token, prefix_utf8_bytes=len(prefix)))
                seen[prompt].add(prefix)
                counts[prompt] += 1
        prefixes[prompt].extend(raw[offset:offset+nbytes])
        offset += nbytes
    if any(n != per_prompt for n in counts):
        raise ValueError(f"insufficient distinct complete UTF-8 prefixes: {counts}")
    return contexts, positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-frames", type=Path)
    p.add_argument("--continuations", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--per-prompt", type=int, default=24)
    args = p.parse_args()
    if args.base_frames:
        with args.base_frames.open("xb") as file:
            file.write(b"FGEVAL01")
            for _, text in base_contexts():
                raw = text.encode("utf-8")
                file.write(struct.pack("<I", len(raw))+raw)
        return
    if not args.continuations or not args.output or args.per_prompt < 1:
        p.error("--continuations, --output and a positive --per-prompt are required")
    contexts, positions = expand(args.continuations.read_bytes(), args.per_prompt)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(contexts, file, ensure_ascii=False, indent=2)
        file.write("\n")
    args.output.with_suffix(".positions.json").write_text(json.dumps(positions, indent=2)+"\n")
    print(f"Frozen {len(contexts)} distinct next-token frontiers; no candidate feedback.")


if __name__ == "__main__":
    main()
