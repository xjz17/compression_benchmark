#!/usr/bin/env python3
"""Normalize the benchmark source CSV schemas used by CompressionBench.

The migration removes catalogued empty columns and materializes semantic
timestamps as signed Unix-epoch milliseconds.  Naive timestamps are interpreted
as UTC; timestamps carrying an offset are converted to UTC.  Air-Quality's
separate Date and Time fields are combined into one ``timestamp`` field.

Files are streamed to a sibling temporary file, validated, and atomically
replaced only after the complete output has been written successfully.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import os


EMPTY_MARKERS = {"", "null", "nan", "none", "na", "n/a"}

SPECS = {
    "Air-Quality.csv": {"drop": ["Unnamed: 15", "Unnamed: 16"], "combine": ("Date", "Time")},
    "Arade.csv": {"drop": ["column_5", "column_6"]},
    "Bird-migration.csv": {"drop": ["manually-marked-outlier"], "timestamps": ["timestamp"]},
    "Comments.csv": {"timestamps": ["date"]},
    "medium_posts.csv": {"timestamps": ["published"]},
    "meteoblue.csv": {"timestamps": ["timestamp"]},
    "Metro-Traffic.csv": {"timestamps": ["date_time"]},
    "NYC_Yellow_Taxi_Trip.csv": {
        "timestamps": ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    },
    "rating.csv": {"timestamps": ["timestamp"]},
    "Uberlandia.csv": {"drop": ["column_65"]},
}


def epoch_milliseconds(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y%m%dT%H%M", "%Y/%m/%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"unsupported timestamp {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return str(int(parsed.timestamp() * 1000))


def migrate(path: Path, spec: dict[str, object]) -> tuple[int, int]:
    temporary = path.with_name(f".{path.name}.migration.tmp")
    rows = 0
    timestamps = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            drop_names = list(spec.get("drop", []))
            combine = spec.get("combine")
            timestamp_names = list(spec.get("timestamps", []))

            # Empty trailing headers in Air-Quality have no stable csv.DictReader key.
            if path.name == "Air-Quality.csv":
                header = [name if name else f"Unnamed: {i}" for i, name in enumerate(header)]

            missing = [name for name in drop_names + timestamp_names if name not in header]
            if combine:
                missing.extend(name for name in combine if name not in header)
            if missing:
                raise ValueError(f"{path}: missing expected columns {missing}")

            drop_indices = {header.index(name) for name in drop_names}
            timestamp_indices = {header.index(name) for name in timestamp_names}
            combined_time_index = None
            if combine:
                date_index, time_index = (header.index(name) for name in combine)
                combined_time_index = time_index
                drop_indices.add(time_index)
                timestamp_indices.add(date_index)
                header[date_index] = "timestamp"

            output_header = [name for index, name in enumerate(header) if index not in drop_indices]
            with temporary.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(output_header)
                for row_number, row in enumerate(reader, start=2):
                    if len(row) < len(header):
                        row.extend([""] * (len(header) - len(row)))
                    elif len(row) > len(header):
                        raise ValueError(f"{path}:{row_number}: too many fields")
                    for index in drop_indices:
                        if index == combined_time_index:
                            continue
                        if row[index].strip().lower() not in EMPTY_MARKERS:
                            raise ValueError(
                                f"{path}:{row_number}: expected empty {header[index]!r}, got {row[index]!r}"
                            )
                    if combine:
                        row[date_index] = epoch_milliseconds(f"{row[date_index]} {row[time_index]}")
                        timestamps += 1
                    for index in timestamp_indices:
                        if combine and index == date_index:
                            continue
                        row[index] = epoch_milliseconds(row[index])
                        timestamps += 1
                    writer.writerow([value for index, value in enumerate(row) if index not in drop_indices])
                    rows += 1
                target.flush()
                os.fsync(target.fileno())
        os.replace(temporary, path)
        return rows, timestamps
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing the original CSV files")
    parser.add_argument("--dataset", action="append", help="CSV stem to migrate (repeatable)")
    args = parser.parse_args()
    selected = set(args.dataset or [])
    missing = []
    for filename, spec in SPECS.items():
        if selected and Path(filename).stem not in selected:
            continue
        path = args.root / filename
        if not path.is_file():
            missing.append(filename)
            print(f"MISSING {path}")
            continue
        rows, timestamps = migrate(path, spec)
        print(f"OK {filename}: rows={rows}, converted_timestamps={timestamps}")
    if missing:
        print("Missing source files: " + ", ".join(missing))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
