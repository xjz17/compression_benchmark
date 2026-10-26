#!/usr/bin/env python3
"""
Insert or replace ``buff_float_kernel_native`` in ``backend/dataset/*/*_float_*_compression_result.csv``.

Metrics come from the same Python bench as ``process_csv_simple`` (``bench_buff_float_kernel_native_columns``).
The row is written to **Java, Python, and C++** float CSVs with **identical** numbers so the ranking table
shows BUFF under any Implement Language; timings reflect the Python in-process kernel (ratio matches the codec).

Run after pulling the repo or when datasets change::

    python3 scripts/append_buff_float_kernel_native_staticdata.py

Env: ``WEB_COMPRESSION_BUFF_KERNEL_NATIVE=1`` (default). Override ``WEB_COMPRESSION_BUFF_INT_SCALE`` /
``WEB_COMPRESSION_BUFF_PRED`` to match other BUFF benches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from dataset_csv_source import resolve_benchmark_dataset_csv

sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
os.environ.setdefault("WEB_COMPRESSION_BUFF_KERNEL_NATIVE", "1")

from myapp.native_research_cli_python import (  # noqa: E402
    bench_buff_float_kernel_native_columns,
)

DATASETS = [
    ("AI4I-Maintenance", "AI4I-Maintenance.csv"),
    ("Air-Quality", "Air-Quality.csv"),
    ("Metro-Traffic", "Metro-Traffic.csv"),
]

LANG_SUFFIXES = ("python", "java", "cpp")


def _float_columns(path: Path) -> list[list[float]]:
    raw = pd.read_csv(path)
    sub = raw.select_dtypes(include=np.number).dropna(axis=1, how="all")
    out: list[list[float]] = []
    for col in sub.columns:
        out.append([float(x) for x in sub[col].tolist()])
    return out


def main() -> None:
    for slug, fname in DATASETS:
        src = resolve_benchmark_dataset_csv(BACKEND, slug, fname)
        if src is None:
            print(f"Skip {slug}: missing dataset/{slug}/{fname}")
            continue
        print(f"Bench buff_float_kernel_native for {slug} (large catalogs can take several minutes)…")
        columns = _float_columns(src)
        if not columns:
            print(f"Skip {slug}: no float columns")
            continue
        o, c, te, td = bench_buff_float_kernel_native_columns(columns)
        if o <= 0:
            print(f"Skip {slug}: bench returned empty")
            continue
        row = {
            "algorithm": "buff_float_kernel_native",
            "originalSize": int(o),
            "compressedSize": int(c),
            "compressionTime": int(te),
            "decompressionTime": int(td),
        }
        out_root = BACKEND / "dataset" / slug
        for lang in LANG_SUFFIXES:
            report = out_root / f"{slug}_float_{lang}_compression_result.csv"
            if not report.is_file():
                print(f"Missing {report.name}, skip")
                continue
            df = pd.read_csv(report)
            df = df[df["algorithm"].astype(str) != "buff_float_kernel_native"]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(report, index=False, lineterminator="\n")
            print(f"Updated {report.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
