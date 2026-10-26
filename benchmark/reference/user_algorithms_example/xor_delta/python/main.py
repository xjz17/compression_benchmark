#!/usr/bin/env python3
"""Reference user codec: per-column delta encoding (lossless). Entry: main.py."""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path

_MAGIC = 0x31435557  # WCUC1 little-endian


def _read_columns(csv_path: Path, data_type: str) -> list[list[float]]:
    columns: list[list[float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return columns
        for _ in header:
            columns.append([])
        for row in reader:
            for col_idx, cell in enumerate(row):
                if col_idx >= len(columns):
                    break
                text = (cell or "").strip()
                if not text:
                    continue
                try:
                    val = float(text)
                except ValueError:
                    continue
                if data_type == "int" and val != int(val):
                    continue
                columns[col_idx].append(val)
    return [c for c in columns if c]


def _encode_columns(columns: list[list[float]], data_type: str) -> bytes:
    chunks: list[bytes] = [struct.pack("<III", _MAGIC, 1 if data_type == "int" else 2, len(columns))]
    for col in columns:
        chunks.append(struct.pack("<I", len(col)))
        if not col:
            continue
        if data_type == "int":
            prev = int(col[0])
            chunks.append(struct.pack("<q", prev))
            for v in col[1:]:
                cur = int(v)
                chunks.append(struct.pack("<q", cur - prev))
                prev = cur
        else:
            prev = float(col[0])
            chunks.append(struct.pack("<d", prev))
            for v in col[1:]:
                cur = float(v)
                chunks.append(struct.pack("<d", cur - prev))
                prev = cur
    return b"".join(chunks)


def _decode_columns(blob: bytes, data_type: str) -> list[list[float]]:
    offset = 0
    magic, tag, ncol = struct.unpack_from("<III", blob, offset)
    offset += 12
    if magic != _MAGIC:
        raise ValueError("bad magic")
    columns: list[list[float]] = []
    for _ in range(ncol):
        n, = struct.unpack_from("<I", blob, offset)
        offset += 4
        col: list[float] = []
        if n == 0:
            columns.append(col)
            continue
        if tag == 1:
            v, = struct.unpack_from("<q", blob, offset)
            offset += 8
            col.append(float(v))
            for _i in range(1, n):
                delta, = struct.unpack_from("<q", blob, offset)
                offset += 8
                v += delta
                col.append(float(v))
        else:
            v, = struct.unpack_from("<d", blob, offset)
            offset += 8
            col.append(v)
            for _i in range(1, n):
                delta, = struct.unpack_from("<d", blob, offset)
                offset += 8
                v += delta
                col.append(v)
        columns.append(col)
    return columns


def cmd_compress(inp: Path, out: Path, data_type: str) -> int:
    columns = _read_columns(inp, data_type)
    if not columns:
        print("no numeric columns", file=sys.stderr)
        return 2
    out.write_bytes(_encode_columns(columns, data_type))
    return 0


def cmd_decompress(inp: Path, out: Path, data_type: str) -> int:
    columns = _decode_columns(inp.read_bytes(), data_type)
    nrows = max((len(c) for c in columns), default=0)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([f"col{i}" for i in range(len(columns))])
        for r in range(nrows):
            row = []
            for i in range(len(columns)):
                if r < len(columns[i]):
                    v = columns[i][r]
                    row.append(int(v) if data_type == "int" else v)
                else:
                    row.append("")
            writer.writerow(row)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="reference user codec (main.py)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--data-type", choices=["int", "float"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["compress", "decompress"], default="compress")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    if args.mode == "compress":
        return cmd_compress(inp, out, args.data_type)
    return cmd_decompress(inp, out, args.data_type)


if __name__ == "__main__":
    raise SystemExit(main())
