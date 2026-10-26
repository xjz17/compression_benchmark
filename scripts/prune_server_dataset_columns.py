#!/usr/bin/env python3
"""Prune non-numeric benchmark columns and materialize timestamps as integers.

Each CSV is profiled in one streaming pass. Columns containing any non-null,
non-numeric value are treated as strings unless their catalog metadata or name
marks them as timestamps. Fully null columns are removed. Semantic timestamp
columns are retained and written as UTC Unix-epoch microseconds; missing or
unparseable timestamp cells use their zero-based row position, matching the
benchmark materialization path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Iterable


NULL_MARKERS = frozenset({"", "null", "nan", "none", "na", "n/a"})
TIMESTAMP_LOGICAL_TYPES = frozenset({"timestamp", "datetime", "date", "time"})
TIMESTAMP_NAME_TOKENS = frozenset({"timestamp", "datetime", "date", "time", "published"})
INTEGER_LOGICAL_TYPES = frozenset({"integer", "int", "long"})
FLOAT_LOGICAL_TYPES = frozenset({"float", "double"})
TOKEN_RE = re.compile(r"[^a-z0-9]+")
csv.field_size_limit(sys.maxsize)


def normalized(value: object) -> str:
    return str(value or "").strip()


def is_null(value: object) -> bool:
    return normalized(value).lower() in NULL_MARKERS


def is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def is_timestamp_column(name: str, logical_type: str) -> bool:
    if logical_type.strip().lower() in TIMESTAMP_LOGICAL_TYPES:
        return True
    low = name.strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", low)
    if "timestamp" in compact or "datetime" in compact:
        return True
    tokens = {token for token in TOKEN_RE.split(low) if token}
    return bool(tokens & TIMESTAMP_NAME_TOKENS)


def timestamp_microseconds(value: str, row_index: int) -> str:
    text = normalized(value).strip('"')
    if is_null(text):
        return str(row_index)
    try:
        numeric = float(text)
    except ValueError:
        numeric = math.nan
    if math.isfinite(numeric):
        if abs(numeric) < 100_000_000_000:
            return str(int(round(numeric * 1_000_000)))
        return str(int(round(numeric)))
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        parsed = None
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H.%M.%S",
            "%d/%m/%Y",
            "%H.%M.%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y%m%dT%H%M",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return str(row_index)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return str(int(round(parsed.timestamp() * 1_000_000)))


def read_column_info(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def profile_csv(path: Path, info_rows: list[dict[str, str]]) -> tuple[list[str], list[bool], list[bool], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        width = len(header)
        has_value = [False] * width
        has_non_numeric = [False] * width
        rows = 0
        for rows, row in enumerate(reader, start=1):
            if len(row) < width:
                row.extend([""] * (width - len(row)))
            elif len(row) > width:
                raise ValueError(f"{path}:{rows + 1}: expected {width} fields, got {len(row)}")
            for index, value in enumerate(row):
                if is_null(value):
                    continue
                has_value[index] = True
                if not is_numeric(normalized(value)):
                    has_non_numeric[index] = True
        return header, has_value, has_non_numeric, rows


def classify_columns(
    header: list[str], info_rows: list[dict[str, str]], has_value: list[bool], has_non_numeric: list[bool]
) -> tuple[list[int], list[int], list[dict[str, object]]]:
    info_by_index = {}
    for row in info_rows:
        try:
            info_by_index[int(row.get("column_index", ""))] = row
        except ValueError:
            continue
    keep: list[int] = []
    timestamp_indexes: list[int] = []
    decisions: list[dict[str, object]] = []
    for index, name in enumerate(header):
        info = info_by_index.get(index, {})
        logical = normalized(info.get("logical_type")).lower()
        timestamp = is_timestamp_column(name, logical)
        reason = "keep_numeric"
        if timestamp:
            keep.append(index)
            timestamp_indexes.append(index)
            reason = "keep_timestamp"
        elif not has_value[index]:
            reason = "drop_all_null"
        elif logical == "string" or has_non_numeric[index]:
            reason = "drop_string"
        else:
            keep.append(index)
        decisions.append(
            {
                "old_index": index,
                "column_name": name,
                "logical_type": logical,
                "has_value": has_value[index],
                "has_non_numeric": has_non_numeric[index],
                "decision": reason,
            }
        )
    return keep, timestamp_indexes, decisions


def rewrite_csv(path: Path, header: list[str], keep: list[int], timestamps: set[int]) -> int:
    temporary = path.with_name(f".{path.name}.prune.tmp")
    rows = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source)
            next(reader)
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow([header[index] for index in keep])
            for row_index, row in enumerate(reader):
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                output = []
                for index in keep:
                    value = row[index]
                    if index in timestamps:
                        value = timestamp_microseconds(value, row_index)
                    output.append(value)
                writer.writerow(output)
                rows += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        return rows
    finally:
        temporary.unlink(missing_ok=True)


def rewrite_column_info(
    path: Path, info_rows: list[dict[str, str]], header: list[str], keep: list[int], timestamps: set[int]
) -> None:
    by_index = {}
    for row in info_rows:
        try:
            by_index[int(row.get("column_index", ""))] = row
        except ValueError:
            continue
    fieldnames = list(info_rows[0]) if info_rows else [
        "column_index", "column_name", "logical_type", "max_point_number", "column_csv_size_bytes"
    ]
    temporary = path.with_name(f".{path.name}.prune.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for new_index, old_index in enumerate(keep):
                row = dict(by_index.get(old_index, {}))
                row["column_index"] = str(new_index)
                row["column_name"] = header[old_index]
                if old_index in timestamps:
                    row["logical_type"] = "timestamp"
                    row["max_point_number"] = "0"
                writer.writerow({field: row.get(field, "") for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_datasets(csv_root: Path, catalog_root: Path, selected: set[str]) -> Iterable[tuple[str, Path, Path]]:
    for bundle in sorted(catalog_root.iterdir()):
        if not bundle.is_dir() or (selected and bundle.name not in selected):
            continue
        csv_path = csv_root / f"{bundle.name}.csv"
        info_path = bundle / f"{bundle.name}_column_info.csv"
        if csv_path.is_file() and info_path.is_file():
            yield bundle.name, csv_path, info_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    selected = set(args.dataset)
    audit: dict[str, object] = {"datasets": {}}
    processed = 0
    for slug, csv_path, info_path in iter_datasets(args.csv_root, args.catalog_root, selected):
        print(f"PROFILE {slug}", flush=True)
        info_rows = read_column_info(info_path)
        header, has_value, has_non_numeric, profiled_rows = profile_csv(csv_path, info_rows)
        keep, timestamp_indexes, decisions = classify_columns(header, info_rows, has_value, has_non_numeric)
        timestamps = set(timestamp_indexes)
        dropped = [item for item in decisions if str(item["decision"]).startswith("drop_")]
        print(
            f"REWRITE {slug}: rows={profiled_rows} columns={len(header)} -> {len(keep)} "
            f"dropped={len(dropped)} timestamps={len(timestamps)}",
            flush=True,
        )
        written_rows = rewrite_csv(csv_path, header, keep, timestamps)
        if written_rows != profiled_rows:
            raise RuntimeError(f"{slug}: profiled {profiled_rows} rows but wrote {written_rows}")
        rewrite_column_info(info_path, info_rows, header, keep, timestamps)
        audit["datasets"][slug] = {
            "rows": written_rows,
            "old_columns": len(header),
            "new_columns": len(keep),
            "timestamp_old_indexes": timestamp_indexes,
            "decisions": decisions,
        }
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        processed += 1
        print(f"OK {slug}", flush=True)
    if selected and processed != len(selected):
        found = set(audit["datasets"])
        raise SystemExit(f"missing selected datasets: {sorted(selected - found)}")
    print(f"DONE datasets={processed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
