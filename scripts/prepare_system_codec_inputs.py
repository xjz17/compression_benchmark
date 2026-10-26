#!/usr/bin/env python3
"""Select the smallest CSV datasets and convert numeric columns to SCDBIN01."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from myapp.experiment_data_policy import is_nan_numeric
from project_dataset_scope import CORE_CSV_DATASETS, PROJECT_DATASETS

MATRIX_INPUT_POLICY = "drop_nan_common_numeric_row_mask_v1"

MAGIC = b"SCDBIN01"
# ``PAPER_DATASETS`` remains the benchmark-ready CSV subset for this converter.
# ``PROJECT_DATASETS`` is the wider 179-dataset research scope and includes
# BenchmarkHost TsFile inputs that require their own loader.
PAPER_DATASETS = list(CORE_CSV_DATASETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--require-paper-19", action="store_true")
    return parser.parse_args()


def csv_files(root: Path) -> list[Path]:
    files = [path for path in root.glob("*.csv") if path.is_file()]
    if not files:
        raise RuntimeError(f"no CSV datasets found in {root}")
    return sorted(files, key=lambda path: (path.stat().st_size, path.name))


def paper_csv_files(root: Path) -> list[Path]:
    files = [root / f"{dataset}.csv" for dataset in PAPER_DATASETS]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"canonical 19-dataset pool is incomplete in {root}: missing {', '.join(missing)}"
        )
    return sorted(files, key=lambda path: (path.stat().st_size, path.name))


def load_numeric_columns(path: Path, *, audit: dict | None = None) -> tuple[list[str], list[list[float]], int]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise RuntimeError(f"empty CSV: {path}")
        raw_rows = [row for row in reader if row]

    column_count = len(header)
    parsed: list[list[float | None]] = [[] for _ in range(column_count)]
    numeric = [True] * column_count
    observed = [False] * column_count
    for row in raw_rows:
        for column in range(column_count):
            text = row[column].strip() if column < len(row) else ""
            if is_nan_numeric(text):
                parsed[column].append(None)
                continue
            try:
                value = float(text)
            except ValueError:
                numeric[column] = False
                parsed[column].append(None)
            else:
                parsed[column].append(None if math.isnan(value) else value)
                if not math.isnan(value):
                    observed[column] = True

    selected = [column for column in range(column_count) if numeric[column] and observed[column]]
    if not selected:
        raise RuntimeError(f"no numeric columns with observed non-NaN values in {path}")
    # SCDBIN01 is a rectangular matrix: use one mask, never independently shift columns.
    retained_rows = [row for row in range(len(raw_rows))
                     if all(parsed[column][row] is not None for column in selected)]
    names = [header[column] or f"column_{column}" for column in selected]
    values = [[parsed[column][row] for row in retained_rows] for column in selected]
    if audit is not None:
        audit.update(input_policy=MATRIX_INPUT_POLICY, source_rows=len(raw_rows),
                     rows=len(retained_rows), rows_removed=len(raw_rows) - len(retained_rows),
                     retained_columns=names)
    if not retained_rows:
        raise RuntimeError(f"no rows without NaN/missing values across selected numeric columns in {path}")
    return names, values, len(retained_rows)


def write_scdbin(path: Path, names: list[str], columns: list[list[float]], rows: int) -> None:
    if len(names) != len(columns) or any(len(column) != rows for column in columns):
        raise ValueError("SCDBIN columns must share the declared row count")
    if any(math.isnan(value) for column in columns for value in column):
        raise ValueError("SCDBIN experiment input contains NaN; apply the common row mask first")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack(">Iq", len(columns), rows))
        for name in names:
            encoded = name.encode("utf-8")
            if len(encoded) > 65535:
                raise RuntimeError(f"column name is too long: {name!r}")
            handle.write(struct.pack(">H", len(encoded)))
            handle.write(encoded)
        for column in columns:
            for value in column:
                handle.write(struct.pack(">d", value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    candidates = (
        paper_csv_files(args.dataset_dir) if args.require_paper_19 else csv_files(args.dataset_dir)
    )
    selected = candidates[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "dataset\tsource_path\tsource_bytes\tsource_sha256\trows\tnumeric_columns\tscdbin_path\tscdbin_sha256\tinput_policy\tsource_rows\trows_removed\tretained_columns"
    ]
    for source in selected:
        audit: dict = {}
        names, columns, rows = load_numeric_columns(source, audit=audit)
        output = args.output_dir / f"{source.stem}.scdbin"
        write_scdbin(output, names, columns, rows)
        manifest_lines.append(
            "\t".join(
                [
                    source.stem,
                    str(source.resolve()),
                    str(source.stat().st_size),
                    sha256_file(source),
                    str(rows),
                    str(len(columns)),
                    str(output.resolve()),
                    sha256_file(output),
                    audit["input_policy"],
                    str(audit["source_rows"]),
                    str(audit["rows_removed"]),
                    "|".join(audit["retained_columns"]),
                ]
            )
        )
    args.manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
