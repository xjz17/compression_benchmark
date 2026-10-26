"""Append-only writers for benchmark compression_result CSV reports."""

from __future__ import annotations

import csv
import os

import pandas as pd

COMPRESSION_REPORT_FIELDS = (
    'algorithm',
    'originalSize',
    'compressedSize',
    'compressionTime',
    'decompressionTime',
    'isLossy',
    'lossRate',
)


def compression_report_needs_header(path: str) -> bool:
    return not os.path.isfile(path) or os.path.getsize(path) == 0


def existing_algorithms_from_report(path: str) -> set:
    names: set = set()
    if not os.path.isfile(path):
        return names
    try:
        df = pd.read_csv(path)
        if 'algorithm' not in df.columns:
            return names
        for algo in df['algorithm'].dropna().astype(str):
            key = algo.strip()
            if key and key not in ('chimp_int', 'chimp32'):
                names.add(key)
    except Exception:
        pass
    return names


def append_compression_report_row(path: str, record: dict) -> None:
    """Append one row; write header only when the report file is new or empty."""
    row = {k: record.get(k, 0 if k == 'isLossy' else '0') for k in COMPRESSION_REPORT_FIELDS}
    write_header = compression_report_needs_header(path)
    with open(path, 'a', encoding='utf-8', newline='\n') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=COMPRESSION_REPORT_FIELDS,
            lineterminator='\n',
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
