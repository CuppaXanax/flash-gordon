#!/usr/bin/env python3
"""Compare real cooked Q4/Q5/IQ4 dequant values against independent ggml routines."""
import argparse
import ctypes
import json
from pathlib import Path
import struct
from q4_source import header

LUT = (-127, -104, -83, -65, -49, -35, -22, -10,
       1, 13, 25, 38, 53, 69, 89, 113)


def value(data, width, row, col, kind, compact):
    groups, g, j = width//32, col//32, col % 32
    qbytes = 20 if kind == 13 else 16
    if kind == 20:
        d, = struct.unpack_from("<e", data, (g*8+row)*2)
        quant = groups*16+(row*groups+g)*16
        q = (data[quant+j % 16] >> (j//16*4)) & 15
        return d*LUT[q]
    if compact:
        d, dm = struct.unpack_from("<ee", data, (g//8*8+row)*4)
        meta = groups*4+(g*8+row)*2
        d *= data[meta]
        dm *= data[meta+1]
        quant = groups*20+(row*groups+g)*qbytes
    else:
        d, = struct.unpack_from("<f", data, (g*16+row)*4)
        dm, = struct.unpack_from("<f", data, (g*16+row)*4+32)
        quant = groups*64+(row*groups+g)*qbytes
    if kind == 12:
        q = (data[quant+j % 16] >> (j//16*4)) & 15
    else:
        offset = quant+(j % 16//4)*5
        bits = int.from_bytes(data[offset:offset+5], "little")
        q = (bits >> ((j % 4+j//16*4)*5)) & 31
    return d*q-dm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, action="append", required=True)
    p.add_argument("--cook-library", type=Path, required=True)
    p.add_argument("--ggml-library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cook = ctypes.CDLL(str(args.cook_library.resolve()))
    ggml = ctypes.CDLL(str(args.ggml_library.resolve()))
    reports = []
    for path in args.gguf:
        _, tensors, base, _ = header(str(path))
        with path.open("rb") as file:
            for t in tensors:
                kind = t["type"]
                if kind not in (12, 13, 20) or len(t["shape"]) != 2:
                    continue
                width, rows = t["shape"]
                rowbytes = width//(32 if kind == 20 else 256)*{12:144, 13:176, 20:18}[kind]
                dequant = getattr(ggml, "dequantize_row_"+{12:"q4_K", 13:"q5_K", 20:"iq4_nl"}[kind])
                dequant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
                dequant.restype = None
                checked = 0
                for row in sorted({0, rows//2, rows-1}):
                    first = row//8*8
                    count = min(8, rows-first)
                    file.seek(base+t["offset"]+first*rowbytes)
                    raw = file.read(count*rowbytes)
                    if len(raw) != count*rowbytes:
                        raise ValueError("short quantized tensor")
                    reference = (ctypes.c_float*width)()
                    dequant(ctypes.create_string_buffer(raw[(row-first)*rowbytes:(row-first+1)*rowbytes]), reference, width)
                    for compact in (False, True):
                        suffix = "compact" if compact else "soa"
                        size = getattr(cook, f"fg_dense_{suffix}_bytes")
                        size.argtypes = [ctypes.c_uint32]*3
                        size.restype = ctypes.c_uint64
                        nbytes = size(width, count, kind)
                        out = ctypes.create_string_buffer(nbytes)
                        convert = getattr(cook, f"fg_cook_dense_{suffix}")
                        convert.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]+[ctypes.c_uint32]*3
                        convert.restype = ctypes.c_bool
                        if not convert(ctypes.create_string_buffer(raw), out, nbytes, width, count, kind):
                            raise ValueError("cook failed")
                        data = out.raw
                        for col in range(width):
                            actual = ctypes.c_float(value(data, width, row-first, col, kind, compact)).value
                            if actual != reference[col]:
                                raise ValueError(f"{path.name}:{t['name']} compact={compact} row={row} col={col}: {actual} != ggml {reference[col]}")
                            checked += 1
                reports.append(dict(file=path.name, name=t["name"], type=kind,
                                    checked_values=checked, passed=True))
    result = dict(passed=bool(reports), tensors=len(reports),
                  checked_values=sum(r["checked_values"] for r in reports), rows=reports)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
