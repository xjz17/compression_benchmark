#!/usr/bin/env python3
"""Stream the paper-19 CSV pool into audited little-endian numeric columns.

CSV parsing/preparation is outside codec timing. No source file is changed.
Known numeric columns are strict; missing metadata is inferred from every row,
then materialized in a second streaming pass. Peak memory is bounded by one CSV
record and small per-column write buffers, not by dataset row count.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

from project_dataset_scope import CORE_CSV_DATASETS

VERSION = "exhaustive-graph-column-input-v1"
INPUT_POLICY = "drop_nan_per_numeric_column_v1"
PAPER_DATASETS = CORE_CSV_DATASETS
MISSING = frozenset({"", "nan", "+nan", "-nan", "na", "n/a", "null", "none", "<na>"})
INT_TYPES = frozenset({"integer", "int", "int64", "long"})
FLOAT_TYPES = frozenset({"float", "float64", "double"})
INT_TOKEN = re.compile(r"^[+-]?[0-9]+$")
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_stat(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _missing(cell: str) -> bool:
    return cell.strip().lower() in MISSING


def _parse_float(cell: str) -> float | None:
    value = float(cell)
    return None if math.isnan(value) else value


def _decimal_places(cell: str) -> int:
    text = cell.lower().lstrip("+-")
    if text in {"inf", "infinity", "nan"}:
        return 0
    mantissa, _, exponent = text.partition("e")
    fraction = mantissa.partition(".")[2]
    # Match metadata's decimal scale convention, ignoring trailing zeros.
    return max(0, len(fraction.rstrip("0")) - (int(exponent) if exponent else 0))


def resolve_source(source_repo: Path, dataset: str, dataset_root: Path | None) -> Path | None:
    roots = [dataset_root] if dataset_root is not None else []
    roots += [source_repo / "backend/dataset_csv", source_repo / "orignial_datasets"]
    for root in roots:
        path = root / f"{dataset}.csv"
        if path.is_file():
            return path.resolve()
    return None


def _metadata(source_repo: Path, dataset: str) -> tuple[Path, list[dict[str, str]]]:
    path = source_repo / "backend/dataset" / dataset / f"{dataset}_column_info.csv"
    if not path.is_file():
        return path, []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row.get("column_name", "") for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate metadata column names: {path}")
    return path, rows


def _column(dataset: str, index: int | None, name: str, info: dict[str, str] | None) -> dict:
    logical = (info or {}).get("logical_type", "").strip().lower()
    timestamp = name in {"timestamp", "date_time"} or logical == "timestamp"
    kind = "int64" if timestamp or logical in INT_TYPES else "float64" if logical in FLOAT_TYPES else None
    # Source migration can materialize former string/date columns as numbers.
    # Their old catalog label must not silently remove numeric source columns.
    inferred = kind is None
    return {
        "key": f"{dataset}__col{index}" if index is not None else f"{dataset}__catalog_{(info or {}).get('column_index', name)}",
        "dataset": dataset, "column_name": name, "source_index": index,
        "catalog_index": (info or {}).get("column_index"), "catalog_logical_type": logical or None,
        "dtype": kind, "kind": kind, "byte_order": "little",
        "max_point": int((info or {}).get("max_point_number", 0) or 0),
        "type_source": "numeric_timestamp" if timestamp else "full_source_inference" if inferred else "catalog_by_name",
        "status": "pending" if kind or inferred else "nonnumeric",
        "input_policy": INPUT_POLICY, "source_rows": 0, "retained_rows": 0,
        "removed_nan_count": 0, "raw_path": None, "raw_bytes": 0, "raw_sha256": None,
        "timestamp": timestamp, "infer": inferred,
    }


def _read_header(source: Path) -> list[str]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header or len(header) != len(set(header)):
        raise ValueError(f"empty or duplicate source header: {source}")
    return header


def _rows(source: Path, header: list[str]):
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != header:
            raise ValueError("source header changed during preparation")
        for row_index, row in enumerate(reader, 1):
            if len(row) > len(header):
                raise ValueError(f"CSV record {row_index} exceeds header width")
            yield row_index, row


def _infer_columns(source: Path, header: list[str], columns: list[dict], progress) -> None:
    unknown = [column for column in columns if column["infer"]]
    if not unknown:
        return
    states = {column["key"]: {"integer": True, "numeric": True, "observed": False,
                                "integer_range": False, "float_precision_loss": False, "max_point": 0}
              for column in unknown}
    for row_index, row in _rows(source, header):
        for column in unknown:
            state = states[column["key"]]
            if not state["numeric"]:
                continue
            index = column["source_index"]
            cell = row[index].strip() if index < len(row) else ""
            if _missing(cell):
                continue
            if INT_TOKEN.fullmatch(cell):
                integer = int(cell)
                state["observed"] = True
                state["integer_range"] |= not I64_MIN <= integer <= I64_MAX
                try:
                    state["float_precision_loss"] |= int(float(integer)) != integer
                except (ValueError, OverflowError):
                    state["float_precision_loss"] = True
            else:
                try:
                    value = _parse_float(cell)
                except (ValueError, OverflowError):
                    state["numeric"] = False
                    continue
                if value is None:
                    continue
                state["observed"] = True
                state["integer"] = False
                state["max_point"] = max(state["max_point"], _decimal_places(cell))
        if row_index % 100000 == 0:
            progress({"phase": "infer", "dataset": columns[0]["dataset"], "source_rows": row_index})
    for column in unknown:
        state = states[column["key"]]
        if not state["numeric"]:
            column["status"] = "nonnumeric"
        elif not state["observed"]:
            column["status"] = "empty_untyped"
        elif state["integer"] and state["integer_range"]:
            column["status"] = "unsupported_integer_range"
        elif not state["integer"] and state["float_precision_loss"]:
            column["status"] = "unsupported_mixed_numeric_precision"
        else:
            column["kind"] = column["dtype"] = "int64" if state["integer"] else "float64"
            column["max_point"] = state["max_point"]


def _cache_valid(manifest: dict, identity: dict) -> bool:
    if manifest.get("identity") != identity or manifest.get("status") not in {"complete", "complete_with_issues"}:
        return False
    for column in manifest.get("columns", []):
        if column["status"] not in {"ready", "empty"}:
            continue
        raw = Path(column["raw_path"])
        if not raw.is_file() or raw.stat().st_size != column["raw_bytes"] or sha256_file(raw) != column["raw_sha256"]:
            return False
    return True


def prepare_dataset(source_repo: Path, output: Path, dataset: str,
                    dataset_root: Path | None = None, progress=lambda value: None) -> dict:
    dataset_dir = output / "datasets" / dataset
    manifest_path = dataset_dir / "manifest.json"
    source = resolve_source(source_repo, dataset, dataset_root)
    metadata_path, metadata_rows = _metadata(source_repo, dataset)
    metadata_sha = sha256_file(metadata_path) if metadata_path.is_file() else None
    if source is None:
        columns = [_column(dataset, None, row["column_name"], row) for row in metadata_rows]
        for column in columns:
            column["status"] = "missing_source"
        result = {"version": VERSION, "dataset": dataset, "status": "missing_source", "columns": columns,
                  "metadata_path": str(metadata_path), "metadata_sha256": metadata_sha}
        atomic_json(manifest_path, result)
        return result
    before = source_stat(source)
    source_sha = sha256_file(source)
    identity = {"version": VERSION, "input_policy": INPUT_POLICY, "source_path": str(source),
                "source_bytes": before[0], "source_sha256": source_sha,
                "metadata_path": str(metadata_path), "metadata_sha256": metadata_sha,
                "preparer_sha256": sha256_file(Path(__file__))}
    if source_stat(source) != before:
        raise ValueError(f"source changed while hashing: {source}")
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _cache_valid(cached, identity):
                progress({"phase": "verified_cache", "dataset": dataset})
                return cached
        except (OSError, ValueError, KeyError, TypeError):
            pass
    header = _read_header(source)
    by_name = {row["column_name"]: row for row in metadata_rows}
    columns = [_column(dataset, index, name, by_name.get(name)) for index, name in enumerate(header)]
    absent = [_column(dataset, None, row["column_name"], row) for row in metadata_rows if row["column_name"] not in header]
    for column in absent:
        column["status"] = "catalog_column_absent_from_source"
    _infer_columns(source, header, columns, progress)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    temporary_paths = {}
    digests = {}
    packers = {}
    source_rows = 0
    try:
        for column in columns:
            if column["status"] != "pending":
                continue
            key = column["key"]
            raw_path = dataset_dir / f"column_{column['source_index']}.{column['kind']}.bin"
            temporary = raw_path.with_name(raw_path.name + f".tmp.{os.getpid()}")
            column["raw_path"] = str(raw_path.resolve())
            writers[key] = temporary.open("wb", buffering=65536)
            temporary_paths[key] = temporary
            digests[key] = hashlib.sha256()
            packers[key] = struct.Struct("<q" if column["kind"] == "int64" else "<d")
        for row_index, row in _rows(source, header):
            source_rows = row_index
            for column in columns:
                if column["status"] != "pending":
                    continue
                index = column["source_index"]
                cell = row[index].strip() if index < len(row) else ""
                if _missing(cell):
                    column["removed_nan_count"] += 1
                    continue
                try:
                    if column["kind"] == "int64":
                        value = int(cell)
                        if not I64_MIN <= value <= I64_MAX:
                            raise OverflowError("integer outside signed int64 range")
                    else:
                        value = _parse_float(cell)
                        if value is None:
                            column["removed_nan_count"] += 1
                            continue
                    payload = packers[column["key"]].pack(value)
                except (ValueError, OverflowError, struct.error) as exc:
                    column["status"] = "unsupported_timestamp_text" if column["timestamp"] and not INT_TOKEN.fullmatch(cell) else "invalid_numeric_value"
                    column["error"] = {"source_row": row_index, "error_type": type(exc).__name__, "message": str(exc)[:160]}
                    continue
                writers[column["key"]].write(payload)
                digests[column["key"]].update(payload)
                column["retained_rows"] += 1
            if row_index % 100000 == 0:
                progress({"phase": "materialize", "dataset": dataset, "source_rows": row_index})
        if source_stat(source) != before:
            raise ValueError(f"source changed during preparation: {source}")
        if metadata_sha is not None and sha256_file(metadata_path) != metadata_sha:
            raise ValueError(f"metadata changed during preparation: {metadata_path}")
        for column in columns:
            column["source_rows"] = source_rows
            key = column["key"]
            if key not in writers:
                continue
            handle = writers[key]
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if column["status"] == "pending":
                raw_path = Path(column["raw_path"])
                os.replace(temporary_paths[key], raw_path)
                column["raw_bytes"] = column["retained_rows"] * 8
                column["raw_sha256"] = digests[key].hexdigest()
                column["status"] = "ready" if column["retained_rows"] else "empty"
                if raw_path.stat().st_size != column["raw_bytes"]:
                    raise ValueError(f"unexpected numeric output size: {raw_path}")
            else:
                column["partial_retained_rows_before_error"] = column["retained_rows"]
                column["removed_nan_count_scope"] = "through_first_error"
                column["retained_rows"] = 0
                column["raw_path"] = None
                temporary_paths[key].unlink(missing_ok=True)
        for column in columns + absent:
            column.pop("infer", None)
            column.pop("timestamp", None)
            column.update(source_path=str(source), source_sha256=source_sha,
                          metadata_path=str(metadata_path), metadata_sha256=metadata_sha)
        issues = [column for column in columns if column["status"] not in {"ready", "empty", "empty_untyped", "nonnumeric"}]
        issues += [column for column in absent if column["kind"] is not None]
        result = {"version": VERSION, "dataset": dataset, "identity": identity,
                  "status": "complete_with_issues" if issues else "complete", "source_rows": source_rows,
                  "source_column_count": len(header), "columns": columns + absent,
                  "prepared_unix_seconds": time.time()}
        atomic_json(manifest_path, result)
        return result
    finally:
        for handle in writers.values():
            handle.close()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)


def prepare_columns(source_repo: Path | str, output: Path | str,
                    datasets=PAPER_DATASETS, dataset_root: Path | str | None = None,
                    progress=lambda value: None) -> dict:
    source_repo, output = Path(source_repo).resolve(), Path(output).resolve()
    dataset_root = Path(dataset_root).resolve() if dataset_root is not None else None
    datasets = list(datasets)
    if len(datasets) != len(set(datasets)) or any(dataset not in PAPER_DATASETS for dataset in datasets):
        raise ValueError("datasets must be a unique subset of the canonical paper-19 list")
    # CSV fields such as long comments are allowed without retaining whole files.
    csv.field_size_limit(sys.maxsize)
    sources = {dataset: resolve_source(source_repo, dataset, dataset_root) for dataset in datasets}
    order = sorted(datasets, key=lambda dataset: (sources[dataset].stat().st_size if sources[dataset] else float("inf"), dataset))
    manifest = {"version": VERSION, "input_policy": INPUT_POLICY, "source_repo": str(source_repo),
                "requested_datasets": datasets, "processing_order": order, "status": "running",
                "metric_denominator": "raw_bytes = retained_rows * original_numeric_width_bytes (8)",
                "datasets": [], "columns": []}
    atomic_json(output / "manifest.json", manifest)
    for dataset in order:
        progress({"phase": "start", "dataset": dataset, "source_path": str(sources[dataset])})
        try:
            result = prepare_dataset(source_repo, output, dataset, dataset_root, progress)
        except Exception as exc:
            # Retain known source columns even when a malformed CSV record aborts
            # the dataset, so its difficult columns do not vanish from coverage.
            failed_columns = []
            try:
                _, metadata_rows = _metadata(source_repo, dataset)
                by_name = {row["column_name"]: row for row in metadata_rows}
                if sources[dataset] is not None:
                    header = _read_header(sources[dataset])
                    failed_columns = [_column(dataset, index, name, by_name.get(name))
                                      for index, name in enumerate(header)]
                else:
                    failed_columns = [_column(dataset, None, row["column_name"], row) for row in metadata_rows]
                for column in failed_columns:
                    column["status"] = "preparation_failed"
                    column.pop("infer", None)
                    column.pop("timestamp", None)
            except (OSError, ValueError, KeyError):
                pass
            result = {"version": VERSION, "dataset": dataset, "status": "preparation_failed", "columns": failed_columns,
                      "error_type": type(exc).__name__, "error": str(exc)}
            atomic_json(output / "datasets" / dataset / "manifest.json", result)
        manifest["datasets"].append({key: value for key, value in result.items() if key != "columns"})
        manifest["columns"].extend(result["columns"])
        atomic_json(output / "manifest.json", manifest)
        progress({"phase": "dataset_complete", "dataset": dataset, "status": result["status"],
                  "ready_columns": sum(column["status"] == "ready" for column in result["columns"])})
    manifest["status"] = "complete" if all(result["status"] == "complete" for result in manifest["datasets"]) else "complete_with_issues"
    manifest["ready_columns"] = sum(column["status"] == "ready" for column in manifest["columns"])
    manifest["prepared_raw_bytes"] = sum(column["raw_bytes"] for column in manifest["columns"] if column["status"] in {"ready", "empty"})
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--datasets", nargs="+", help="Canonical names, separated by spaces or commas; default all 20")
    args = parser.parse_args()
    datasets = [name for item in args.datasets for name in item.split(",")] if args.datasets else PAPER_DATASETS
    manifest = prepare_columns(args.source_repo, args.output, datasets, args.dataset_root,
                               progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    print(json.dumps({"status": manifest["status"], "ready_columns": manifest["ready_columns"],
                      "manifest": str((args.output / "manifest.json").resolve())}), flush=True)
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
