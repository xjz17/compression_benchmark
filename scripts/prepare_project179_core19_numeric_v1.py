#!/usr/bin/env python3
"""Materialize the Core19 numeric/UTC-us derived view without changing sources.

Inputs are the current Core19 v2 dataset manifests, the typed numeric manifests
they freeze, and the preserved removed/string-column CSV backups. Timestamp
columns are handled by an explicit policy before all remaining strings are
dropped. The resulting numeric candidates are restored to original source
order and the longest whole-column prefix at or below the byte limit is copied
to a new, receipt-backed output tree.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, TextIO


VERSION = "project179-core19-numeric-utc-v2"
DEFAULT_LIMIT_BYTES = 1 << 30
DEFAULT_CHUNK_BYTES = 64 << 20
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
MISSING = frozenset({"", "nan", "+nan", "-nan", "na", "n/a", "null", "none", "<na>", "''"})
ALLOWED_DTYPES = frozenset({"int64", "float64"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def timestamp_formats(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return one or more strict, ordered timestamp formats from policy."""
    has_format = "format" in entry
    has_formats = "formats" in entry
    if has_format == has_formats:
        raise ValueError("timestamp policy must define exactly one of format or formats")
    if has_format:
        value = entry.get("format")
        if not isinstance(value, str) or not value:
            raise ValueError("timestamp format must be a nonempty string")
        return (value,)
    values = entry.get("formats")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("timestamp formats must be a nonempty unique string list")
    return tuple(values)


def timestamp_format_receipt(entry: Mapping[str, Any]) -> dict[str, Any]:
    formats = timestamp_formats(entry)
    if "format" in entry:
        return {"format": formats[0]}
    return {"formats": list(formats)}


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema") != "project179-core19-utc-policy-v2":
        raise ValueError("unsupported Core19 UTC policy schema")
    target = policy.get("target_timestamp", {})
    if target.get("dtype") != "int64" or target.get("unit") != "unix_epoch_microseconds":
        raise ValueError("policy target must be signed int64 Unix epoch microseconds")
    datasets = policy.get("datasets")
    if not isinstance(datasets, dict) or len(datasets) != 19:
        raise ValueError("policy must contain exactly 19 Core datasets")
    entries = []
    for dataset, dataset_policy in datasets.items():
        timestamps = dataset_policy.get("timestamps")
        if not isinstance(timestamps, list):
            raise ValueError(f"timestamp policy must be a list: {dataset}")
        names, indices = set(), set()
        for entry in timestamps:
            name, index = str(entry.get("output_name", "")), int(entry["output_source_index"])
            if not name or name in names or index in indices:
                raise ValueError(f"duplicate/empty timestamp output in {dataset}: {name}@{index}")
            names.add(name)
            indices.add(index)
            source, rule = entry.get("source"), entry.get("timezone_rule")
            if source == "backup_string":
                if not entry.get("source_columns"):
                    raise ValueError(f"backup timestamp lacks columns/format: {dataset}:{name}")
                try:
                    timestamp_formats(entry)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid backup timestamp format policy: {dataset}:{name}: {exc}"
                    ) from exc
                if rule not in {"exact_offset", "utc_z", "assume_utc_coordinate", "utc_midnight_coordinate", "drop_time_only"}:
                    raise ValueError(f"unsupported timezone rule: {dataset}:{name}:{rule}")
            elif source == "prepared_numeric":
                if not entry.get("source_column") or entry.get("source_unit") not in {
                    "unix_epoch_seconds", "unix_epoch_milliseconds", "unix_epoch_microseconds"
                } or rule != "unix_epoch":
                    raise ValueError(f"invalid numeric timestamp policy: {dataset}:{name}")
            else:
                raise ValueError(f"unsupported timestamp source: {dataset}:{name}:{source}")
            instant = bool(entry.get("event_utc_instant_claim"))
            if instant and rule not in {"exact_offset", "utc_z", "unix_epoch"}:
                raise ValueError(f"unproved UTC-instant claim: {dataset}:{name}")
            if rule in {"assume_utc_coordinate", "utc_midnight_coordinate", "drop_time_only"} and instant:
                raise ValueError(f"coordinate-only rule claims an instant: {dataset}:{name}")
            if entry.get("action") == "drop" and rule != "drop_time_only":
                raise ValueError(f"only time-only policies may be dropped: {dataset}:{name}")
            entries.append(entry)
    counts = policy.get("policy_counts", {})
    expected = {
        "timestamp_entries": len(entries),
        "convertible_if_sources_validate": sum(entry.get("action") != "drop" for entry in entries),
        "exact_utc_instant_claim_true": sum(bool(entry.get("event_utc_instant_claim")) for entry in entries),
        "coordinate_only_instant_claim_false": sum(
            entry.get("action") != "drop" and not bool(entry.get("event_utc_instant_claim"))
            for entry in entries
        ),
        "time_only_dropped": sum(entry.get("action") == "drop" for entry in entries),
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError(f"policy_counts drift: expected {expected}, got {counts}")


def _safe_output(path: Path) -> Path:
    result = path.resolve()
    if result in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"unsafe output directory: {result}")
    return result


def _missing(value: str) -> bool:
    return value.strip().lower() in MISSING


def _epoch_us(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("timestamp is not timezone-coordinate aware")
    utc = value.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc - epoch
    result = ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds
    if not I64_MIN <= result <= I64_MAX:
        raise OverflowError("UTC microseconds outside signed int64")
    return result


def parse_timestamp_us(entry: Mapping[str, Any], cells: list[str]) -> int:
    if len(cells) != len(entry["source_columns"]):
        raise ValueError("timestamp source-column arity mismatch")
    text = str(entry.get("join_with", "")).join(cell.strip() for cell in cells)
    errors: list[str] = []
    for timestamp_format in timestamp_formats(entry):
        try:
            value = datetime.strptime(text, timestamp_format)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        raise ValueError(
            f"time data {text!r} does not match any policy format: {errors}"
        )
    rule = str(entry["timezone_rule"])
    if rule == "exact_offset":
        if value.tzinfo is None:
            raise ValueError("exact_offset parser produced an offset-naive value")
    elif rule in {"utc_z", "assume_utc_coordinate", "utc_midnight_coordinate"}:
        if value.tzinfo is not None:
            raise ValueError(f"{rule} requires an offset-naive parsed value")
        value = value.replace(tzinfo=timezone.utc)
    else:
        raise ValueError(f"unsupported string timestamp timezone rule: {rule}")
    return _epoch_us(value)


@contextmanager
def open_backup_csv(path: Path) -> Iterator[TextIO]:
    """Open either an uncompressed test fixture or a production .csv.zst."""
    if path.suffix != ".zst":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle
        return
    try:
        process = subprocess.Popen(
            ["zstdcat", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise RuntimeError("zstdcat is required to read preserved .csv.zst backups") from exc
    assert process.stdout is not None
    wrapper = io.TextIOWrapper(process.stdout, encoding="utf-8-sig", newline="")
    try:
        yield wrapper
        wrapper.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        code = process.wait()
        if code:
            raise RuntimeError(f"zstdcat failed for {path}: {stderr.strip()}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def _backup_path(backup_root: Path, dataset: str) -> Path | None:
    for name in (
        f"{dataset}_removed_or_timestamp_columns.csv.zst",
        f"{dataset}_removed_or_timestamp_columns.csv",
    ):
        path = backup_root / name
        if path.is_file():
            return path.resolve()
    return None


def _metadata_path(
    backup_root: Path, dataset: str, active_manifest: Mapping[str, Any]
) -> Path | None:
    local = backup_root / f"{dataset}_column_info.csv"
    if local.is_file():
        return local.resolve()
    upstream = active_manifest.get("identity", {}).get("upstream_metadata_path")
    if upstream and Path(str(upstream)).is_file():
        return Path(str(upstream)).resolve()
    return None


def _read_metadata(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["column_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate metadata source indices: {path}")
    return rows


def _source_index(name: str, metadata: list[dict[str, str]], fallback: int) -> int:
    # Headerless CSV inputs retain their original source position in the
    # canonical ``column_N`` name.  Some numeric-preparation manifests and
    # metadata files renumber columns densely after string columns are removed
    # (Telco is the production example), so their ``column_index`` is only a
    # prepared-view position.  Prefer the canonical name before consulting
    # metadata or we can collide with restored timestamp columns.
    generic = re.fullmatch(r"column_(\d+)", name)
    if generic:
        return int(generic.group(1))
    matches = [int(row["column_index"]) for row in metadata if row.get("column_name") == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous original source index for {name}")
    return fallback


def _prepared_manifest(
    active_path: Path, active: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], str]:
    identity = active.get("identity", {})
    value = identity.get("prepared_dataset_manifest_path")
    expected = str(identity.get("prepared_dataset_manifest_sha256", ""))
    if not value or not expected:
        raise ValueError(f"active manifest lacks prepared-manifest identity: {active_path}")
    path = Path(str(value))
    if not path.is_absolute():
        path = (active_path.parent / path).resolve()
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"prepared manifest hash mismatch: {path}: {actual} != {expected}")
    return path, _json(path), actual


def _prepared_ready_columns(
    active: Mapping[str, Any], prepared: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_rows = active.get("identity", {}).get("ready_raw_columns", [])
    expected = {str(row["key"]): row for row in expected_rows}
    ready = [dict(row) for row in prepared.get("columns", []) if row.get("status") == "ready"]
    if not ready:
        raise ValueError("prepared manifest has no ready numeric columns")
    for row in ready:
        key = str(row["key"])
        if key not in expected:
            raise ValueError(f"prepared ready column absent from active identity: {key}")
        frozen = expected[key]
        if int(row["raw_bytes"]) != int(frozen["bytes"]):
            raise ValueError(f"frozen byte mismatch for {key}")
        if str(row["raw_sha256"]) != str(frozen["sha256"]):
            raise ValueError(f"frozen hash mismatch for {key}")
        if Path(str(row["raw_path"])).resolve() != Path(str(frozen["path"])).resolve():
            raise ValueError(f"frozen path mismatch for {key}")
    if set(expected) != {str(row["key"]) for row in ready}:
        raise ValueError("active identity and prepared ready-column inventories differ")
    return ready


def _verify_raw(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"missing or wrong-size raw column: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"raw column hash mismatch: {path}: {actual} != {expected_sha256}")


def _copy_raw(source: Path, destination: Path, size: int, digest: str) -> None:
    _verify_raw(source, size, digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, 8 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        _verify_raw(temporary, size, digest)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _split_row_counts(rows: int) -> dict[str, int]:
    train = rows // 10
    validation = rows // 10
    return {"train": train, "validation": validation, "test": rows - train - validation}


def _chunk_row_counts(rows: int, chunk_bytes: int, *, minimum_chunks: int = 1) -> list[int]:
    if rows == 0:
        return []
    maximum_rows = max(1, chunk_bytes // 8)
    count = max(minimum_chunks, (rows + maximum_rows - 1) // maximum_rows)
    count = min(rows, count)
    quotient, remainder = divmod(rows, count)
    return [quotient + (index < remainder) for index in range(count)]


def _materialize_chunks(raw: Path, column_dir: Path, rows: int, chunk_bytes: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    split_rows = _split_row_counts(rows)
    chunks: list[dict[str, Any]] = []
    column_dir.mkdir(parents=True, exist_ok=True)
    source_start = 0
    with raw.open("rb") as reader:
        for split in ("train", "validation", "test"):
            minimum = 3 if split == "train" and split_rows[split] >= 3 else 1
            for index, chunk_rows in enumerate(
                _chunk_row_counts(split_rows[split], chunk_bytes, minimum_chunks=minimum)
            ):
                destination = column_dir / f"{split}-{index:05d}.bin"
                temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
                digest = hashlib.sha256()
                remaining = chunk_rows * 8
                try:
                    with temporary.open("wb") as writer:
                        while remaining:
                            block = reader.read(min(8 << 20, remaining))
                            if not block:
                                raise ValueError(f"short raw column while chunking: {raw}")
                            writer.write(block)
                            digest.update(block)
                            remaining -= len(block)
                        writer.flush()
                        os.fsync(writer.fileno())
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                chunks.append({
                    "split": split, "index": index, "rows": chunk_rows,
                    "bytes": chunk_rows * 8, "path": str(destination.resolve()),
                    "sha256": digest.hexdigest(), "source_start_row": source_start,
                    "source_end_row_exclusive": source_start + chunk_rows,
                })
                source_start += chunk_rows
        if reader.read(1):
            raise ValueError(f"raw column has trailing bytes beyond retained_rows: {raw}")
    if source_start != rows:
        raise ValueError(f"chunk row accounting mismatch: {raw}")
    return split_rows, chunks


def _materialize_numeric_timestamp(
    dataset: str,
    entry: Mapping[str, Any],
    prepared_by_name: Mapping[str, dict[str, Any]],
    staging: Path,
) -> dict[str, Any]:
    name = str(entry["source_column"])
    source = prepared_by_name.get(name)
    base = {
        "dataset": dataset,
        "column_name": str(entry["output_name"]),
        "source_index": int(entry["output_source_index"]),
        "dtype": "int64",
        "timestamp_unit": "unix_epoch_microseconds",
        "timezone_rule": str(entry["timezone_rule"]),
        "event_utc_instant_claim": bool(entry["event_utc_instant_claim"]),
        "source_kind": "prepared_numeric",
        "source_column": name,
        "source_unit": str(entry["source_unit"]),
    }
    if source is None:
        return base | {"status": "blocked_timestamp_source_missing"}
    source_path = Path(str(source["raw_path"]))
    source_bytes, source_sha = int(source["raw_bytes"]), str(source["raw_sha256"])
    _verify_raw(source_path, source_bytes, source_sha)
    dtype = str(source["dtype"])
    if dtype not in ALLOWED_DTYPES or source_bytes % 8:
        return base | {"status": "blocked_invalid_numeric_timestamp_source", "source_dtype": dtype}
    multiplier = {
        "unix_epoch_seconds": 1_000_000,
        "unix_epoch_milliseconds": 1_000,
        "unix_epoch_microseconds": 1,
    }.get(str(entry["source_unit"]))
    if multiplier is None:
        raise ValueError(f"unsupported numeric timestamp unit: {entry['source_unit']}")
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"timestamp_{int(entry['output_source_index']):04d}.int64.bin.tmp.{os.getpid()}"
    digest = hashlib.sha256()
    rows = 0
    pack = struct.Struct("<q")
    unpack = struct.Struct("<q" if dtype == "int64" else "<d")
    try:
        with source_path.open("rb") as reader, temporary.open("wb") as writer:
            while True:
                block = reader.read(8)
                if not block:
                    break
                if len(block) != 8:
                    raise ValueError(f"partial numeric value in {source_path}")
                value = unpack.unpack(block)[0]
                if dtype == "float64":
                    if not math.isfinite(value):
                        raise ValueError(f"nonfinite numeric timestamp at retained row {rows + 1}")
                    numerator, denominator = value.as_integer_ratio()
                    scaled = numerator * multiplier
                    if scaled % denominator:
                        raise ValueError(f"timestamp not exactly representable as microseconds at retained row {rows + 1}")
                    result = scaled // denominator
                else:
                    result = int(value) * multiplier
                if not I64_MIN <= result <= I64_MAX:
                    raise OverflowError(f"timestamp microseconds overflow at retained row {rows + 1}")
                payload = pack.pack(result)
                writer.write(payload)
                digest.update(payload)
                rows += 1
            writer.flush()
            os.fsync(writer.fileno())
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return base | {
            "status": "blocked_invalid_numeric_timestamp_source",
            "error_type": type(exc).__name__, "error": str(exc),
            "source_path": str(source_path), "source_bytes": source_bytes,
            "source_sha256": source_sha,
        }
    return base | {
        "status": "candidate_ready", "retained_rows": rows, "removed_missing_count": 0,
        "raw_path": str(temporary), "raw_bytes": rows * 8, "raw_sha256": digest.hexdigest(),
        "source_path": str(source_path), "source_bytes": source_bytes, "source_sha256": source_sha,
    }


def _materialize_backup_timestamps(
    dataset: str,
    entries: list[dict[str, Any]],
    backup: Path | None,
    staging: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    convertible: list[dict[str, Any]] = []
    for entry in entries:
        base = {
            "dataset": dataset, "column_name": str(entry["output_name"]),
            "source_index": int(entry["output_source_index"]), "dtype": "int64",
            "timestamp_unit": "unix_epoch_microseconds",
            "timezone_rule": str(entry["timezone_rule"]),
            "event_utc_instant_claim": bool(entry["event_utc_instant_claim"]),
            "source_kind": "backup_string", "source_columns": list(entry["source_columns"]),
        } | timestamp_format_receipt(entry)
        if entry.get("action") == "drop":
            results.append(base | {
                "status": "dropped_time_only", "blocker_reason": str(entry["blocker_reason"])
            })
        elif backup is None:
            results.append(base | {
                "status": "blocked_timestamp_source_missing",
                "blocker_reason": "preserved removed/string-column backup is absent",
            })
        else:
            convertible.append({"policy": entry, "result": base})
    if not convertible:
        return results
    assert backup is not None
    backup_bytes, backup_sha = backup.stat().st_size, sha256_file(backup)
    staging.mkdir(parents=True, exist_ok=True)
    writers: dict[str, Any] = {}
    temporary: dict[str, Path] = {}
    digests: dict[str, Any] = {}
    pack = struct.Struct("<q")
    try:
        with open_backup_csv(backup) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"empty backup header: {backup}")
            required = {
                column for item in convertible for column in item["policy"]["source_columns"]
            }
            duplicate_required_columns = {
                column for column in required if reader.fieldnames.count(column) > 1
            }
            if duplicate_required_columns:
                raise ValueError(
                    f"duplicate required timestamp columns in {backup}: "
                    f"{sorted(duplicate_required_columns)}"
                )
            missing_columns = {
                column
                for item in convertible for column in item["policy"]["source_columns"]
                if column not in reader.fieldnames
            }
            if missing_columns:
                for item in convertible:
                    item["result"].update(
                        status="blocked_timestamp_columns_missing",
                        blocker_reason=f"backup lacks columns: {sorted(missing_columns)}",
                    )
                return results + [item["result"] for item in convertible]
            for item in convertible:
                result = item["result"]
                key = result["column_name"]
                path = staging / f"timestamp_{result['source_index']:04d}.int64.bin.tmp.{os.getpid()}"
                temporary[key] = path
                writers[key] = path.open("wb", buffering=65536)
                digests[key] = hashlib.sha256()
                result.update(retained_rows=0, removed_missing_count=0, source_rows=0)
            for row_number, row in enumerate(reader, 1):
                for item in convertible:
                    result, policy = item["result"], item["policy"]
                    result["source_rows"] = row_number
                    if result.get("status", "candidate_pending") != "candidate_pending":
                        continue
                    cells = [(row.get(name) or "").strip() for name in policy["source_columns"]]
                    if any(_missing(cell) for cell in cells):
                        result["removed_missing_count"] += 1
                        continue
                    try:
                        value = parse_timestamp_us(policy, cells)
                        payload = pack.pack(value)
                    except Exception as exc:
                        result.update(
                            status="blocked_invalid_timestamp_value", source_row=row_number,
                            source_value=cells, error_type=type(exc).__name__, error=str(exc),
                        )
                        continue
                    writers[result["column_name"]].write(payload)
                    digests[result["column_name"]].update(payload)
                    result["retained_rows"] += 1
        for item in convertible:
            result = item["result"]
            handle = writers[result["column_name"]]
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if result.get("status", "candidate_pending") == "candidate_pending":
                result.update(
                    status="candidate_ready", raw_path=str(temporary[result["column_name"]]),
                    raw_bytes=result["retained_rows"] * 8,
                    raw_sha256=digests[result["column_name"]].hexdigest(),
                )
            else:
                temporary[result["column_name"]].unlink(missing_ok=True)
            result.update(
                backup_path=str(backup), backup_bytes=backup_bytes, backup_sha256=backup_sha
            )
    finally:
        for handle in writers.values():
            if not handle.closed:
                handle.close()
    return results + [item["result"] for item in convertible]


def _manifest_valid(manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    if manifest.get("identity") != identity or manifest.get("status") not in {
        "complete", "complete_with_blocked_timestamps"
    }:
        return False
    try:
        for column in manifest.get("columns", []):
            _verify_raw(
                Path(str(column["raw_path"])), int(column["raw_bytes"]), str(column["raw_sha256"])
            )
            for chunk in column.get("chunks", []):
                _verify_raw(
                    Path(str(chunk["path"])), int(chunk["bytes"]), str(chunk["sha256"])
                )
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True


def _candidate_from_numeric(
    dataset: str, row: Mapping[str, Any], metadata: list[dict[str, str]]
) -> dict[str, Any]:
    dtype = str(row["dtype"])
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"unsupported numeric dtype {dtype} in {dataset}:{row['column_name']}")
    path = Path(str(row["raw_path"]))
    return {
        "dataset": dataset, "column_name": str(row["column_name"]),
        "source_index": _source_index(str(row["column_name"]), metadata, int(row["source_index"])),
        "prepared_source_index": int(row["source_index"]), "dtype": dtype,
        "source_kind": "prepared_numeric", "status": "candidate_ready",
        "raw_path": str(path), "raw_bytes": int(row["raw_bytes"]),
        "raw_sha256": str(row["raw_sha256"]), "retained_rows": int(row["retained_rows"]),
        "removed_missing_count": int(row.get("removed_nan_count", 0)),
    }


def prepare_dataset(
    dataset: str,
    active_root: Path,
    backup_root: Path,
    output: Path,
    dataset_policy: Mapping[str, Any],
    policy_sha256: str,
    *,
    limit_bytes: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    resume: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    active_path = active_root / "datasets" / dataset / "manifest.json"
    active_sha = sha256_file(active_path)
    active = _json(active_path)
    prepared_path, prepared, prepared_sha = _prepared_manifest(active_path, active)
    prepared_rows = _prepared_ready_columns(active, prepared)
    metadata_path = _metadata_path(backup_root, dataset, active)
    metadata = _read_metadata(metadata_path)
    backup = _backup_path(backup_root, dataset)
    backup_identity = None if backup is None else {
        "path": str(backup), "bytes": backup.stat().st_size, "sha256": sha256_file(backup)
    }
    metadata_identity = None if metadata_path is None else {
        "path": str(metadata_path), "bytes": metadata_path.stat().st_size,
        "sha256": sha256_file(metadata_path),
    }
    identity = {
        "version": VERSION, "dataset": dataset, "limit_bytes": limit_bytes,
        "chunk_bytes": chunk_bytes,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "split_spec": {
            "train_fraction": "0.1", "validation_fraction": "0.1", "test_fraction": "0.8",
            "rounding": "floor train and validation counts; test receives remainder",
            "train_chunking": "at least 3 near-equal chunks when train rows >= 3",
            "validation_test_chunking": "minimum number of near-equal max-bounded chunks",
            "max_chunk_bytes": chunk_bytes, "value_width_bytes": 8,
        },
        "policy_sha256": policy_sha256, "dataset_policy": dataset_policy,
        "active_manifest_path": str(active_path.resolve()), "active_manifest_sha256": active_sha,
        "prepared_manifest_path": str(prepared_path), "prepared_manifest_sha256": prepared_sha,
        "backup": backup_identity, "metadata": metadata_identity,
    }
    dataset_dir = output / "datasets" / dataset
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        prior = _json(manifest_path)
        if _manifest_valid(prior, identity):
            return prior | {"resume_status": "verified_existing"}
        if verify_only:
            raise RuntimeError(f"invalid cached dataset receipt: {dataset}")
        if not resume:
            raise FileExistsError(f"dataset output exists; use --resume: {manifest_path}")
    elif verify_only:
        raise RuntimeError(f"missing dataset receipt: {dataset}")

    timestamps = [dict(row) for row in dataset_policy.get("timestamps", [])]
    replaced = {
        str(name) for row in timestamps for name in row.get("replaces_prepared_columns", [])
    }
    prepared_by_name = {str(row["column_name"]): row for row in prepared_rows}
    if len(prepared_by_name) != len(prepared_rows):
        raise ValueError(f"duplicate prepared column names: {dataset}")
    numeric = [
        _candidate_from_numeric(dataset, row, metadata)
        for row in prepared_rows if str(row["column_name"]) not in replaced
    ]
    staging = dataset_dir / ".staging"
    string_entries = [row for row in timestamps if row["source"] == "backup_string"]
    timestamp_results = _materialize_backup_timestamps(dataset, string_entries, backup, staging)
    for row in timestamps:
        if row["source"] == "prepared_numeric":
            timestamp_results.append(
                _materialize_numeric_timestamp(dataset, row, prepared_by_name, staging)
            )

    candidates = numeric + [row for row in timestamp_results if row["status"] == "candidate_ready"]
    candidates.sort(key=lambda row: (int(row["source_index"]), str(row["column_name"])))
    indices = [int(row["source_index"]) for row in candidates]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate output source indices after timestamp replacement: {dataset}")
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total = 0
    capped = False
    for row in candidates:
        size = int(row["raw_bytes"])
        if capped or total + size > limit_bytes:
            capped = True
            excluded.append(row | {"status": "excluded_by_1g_whole_column_prefix"})
        else:
            selected.append(row)
            total += size

    dataset_dir.mkdir(parents=True, exist_ok=True)
    columns: list[dict[str, Any]] = []
    for active_index, row in enumerate(selected):
        destination = dataset_dir / f"column_{int(row['source_index']):04d}.{row['dtype']}.bin"
        source = Path(str(row["raw_path"]))
        if str(row["source_kind"]) == "prepared_numeric" and row not in timestamp_results:
            _copy_raw(source, destination, int(row["raw_bytes"]), str(row["raw_sha256"]))
        else:
            _verify_raw(source, int(row["raw_bytes"]), str(row["raw_sha256"]))
            os.replace(source, destination)
        columns.append({
            key: value for key, value in (row | {
                "status": "ready", "active_index": active_index,
                "raw_path": str(destination.resolve()),
            }).items() if key not in {"prepared_source_index"}
        })
        split_rows, chunks = _materialize_chunks(
            destination,
            dataset_dir / "chunks" / f"column_{int(row['source_index']):04d}",
            int(row["retained_rows"]), chunk_bytes,
        )
        columns[-1]["split_rows"] = split_rows
        columns[-1]["chunks"] = chunks
    for row in excluded:
        if row.get("source_kind") == "backup_string" or row in timestamp_results:
            Path(str(row["raw_path"])).unlink(missing_ok=True)

    consumed = {
        name for row in timestamps for name in row.get("source_columns", [])
    }
    string_receipts = []
    for row in metadata:
        if str(row.get("logical_type", "")).lower() not in {"string", "timestamp", "datetime", "date", "time"}:
            continue
        name = str(row["column_name"])
        if name in consumed:
            continue
        string_receipts.append({
            "column_name": name, "source_index": int(row["column_index"]),
            "status": "dropped_remaining_string", "logical_type": row.get("logical_type"),
        })
    blocked = [row for row in timestamp_results if row["status"].startswith("blocked_")]
    intentional_drops = [row for row in timestamp_results if row["status"] == "dropped_time_only"]
    status = "complete_with_blocked_timestamps" if blocked else "complete"
    result = {
        "schema": "project179-core19-numeric-utc-dataset-v1", "version": VERSION,
        "status": status, "dataset": dataset, "identity": identity,
        "selection": {
            "kind": "longest_whole_numeric_column_prefix_after_timestamp_conversion",
            "limit_bytes": limit_bytes, "selected_bytes": total,
            "candidate_columns": len(candidates), "selected_columns": len(columns),
            "selected_source_indices": [int(row["source_index"]) for row in columns],
            "excluded_by_cap_count": len(excluded),
        },
        "columns": columns,
        "excluded_numeric_columns": [
            {key: value for key, value in row.items() if key != "raw_path"} for row in excluded
        ],
        "timestamp_receipts": [
            {key: value for key, value in row.items() if key != "raw_path"}
            for row in timestamp_results
        ],
        "string_receipts": string_receipts,
        "counts": {
            "timestamp_admitted": sum(row["status"] == "candidate_ready" for row in timestamp_results),
            "timestamp_selected": sum(column.get("timestamp_unit") == "unix_epoch_microseconds" for column in columns),
            "timestamp_blocked": len(blocked), "time_only_dropped": len(intentional_drops),
            "remaining_strings_dropped": len(string_receipts),
            "numeric_candidates": len(candidates), "numeric_selected": len(columns),
            "chunks": sum(len(column["chunks"]) for column in columns),
            "chunk_bytes": sum(
                int(chunk["bytes"]) for column in columns for chunk in column["chunks"]
            ),
        },
        "prepared_unix_seconds": time.time(),
    }
    atomic_json(manifest_path, result)
    return result | {"resume_status": "materialized_and_verified"}


def prepare_scope(
    active_root: Path | str,
    backup_root: Path | str,
    output: Path | str,
    policy_path: Path | str,
    *,
    datasets: list[str] | None = None,
    limit_bytes: int = DEFAULT_LIMIT_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    resume: bool = False,
    verify_only: bool = False,
    progress=lambda value: None,
) -> dict[str, Any]:
    active_root = Path(active_root).resolve()
    backup_root = Path(backup_root).resolve()
    output = _safe_output(Path(output))
    policy_path = Path(policy_path).resolve()
    if limit_bytes <= 0:
        raise ValueError("limit_bytes must be positive")
    if chunk_bytes < 8 or chunk_bytes % 8:
        raise ValueError("chunk_bytes must be a positive multiple of 8")
    policy = _json(policy_path)
    validate_policy(policy)
    all_policies = policy.get("datasets")
    assert isinstance(all_policies, dict)
    requested = list(all_policies) if datasets is None else list(datasets)
    if len(requested) != len(set(requested)) or any(name not in all_policies for name in requested):
        raise ValueError("datasets must be a distinct subset of the policy")
    policy_sha = sha256_file(policy_path)
    csv.field_size_limit(sys.maxsize)
    results = []
    for dataset in requested:
        result = prepare_dataset(
            dataset, active_root, backup_root, output, all_policies[dataset], policy_sha,
            limit_bytes=limit_bytes, chunk_bytes=chunk_bytes,
            resume=resume, verify_only=verify_only,
        )
        results.append(result)
        progress({
            "dataset": dataset, "status": result["status"],
            "resume_status": result["resume_status"], "counts": result["counts"],
        })
    totals = {
        key: sum(int(result["counts"][key]) for result in results)
        for key in (
            "timestamp_admitted", "timestamp_selected", "timestamp_blocked",
            "time_only_dropped", "remaining_strings_dropped", "numeric_candidates",
            "numeric_selected", "chunks", "chunk_bytes",
        )
    }
    status = (
        "complete_with_blocked_timestamps"
        if any(result["status"] == "complete_with_blocked_timestamps" for result in results)
        else "complete"
    )
    top = {
        "schema": "project179-core19-numeric-utc-scope-v1", "version": VERSION,
        "status": status, "dataset_count": len(results), "datasets": requested,
        "policy_path": str(policy_path), "policy_sha256": policy_sha,
        "active_root": str(active_root), "backup_root": str(backup_root),
        "limit_bytes": limit_bytes, "chunk_bytes": chunk_bytes, "totals": totals,
        "dataset_manifests": [
            {
                "dataset": result["dataset"],
                "path": str((output / "datasets" / result["dataset"] / "manifest.json").resolve()),
                "sha256": sha256_file(output / "datasets" / result["dataset"] / "manifest.json"),
                "status": result["status"], "counts": result["counts"],
            }
            for result in results
        ],
        "producer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "completed_unix_seconds": time.time(),
    }
    top_path = output / "manifest.json"
    if verify_only:
        if not top_path.is_file():
            raise RuntimeError(f"missing top-level receipt: {top_path}")
        prior = _json(top_path)
        stable_keys = ("schema", "version", "status", "dataset_count", "datasets", "policy_sha256", "limit_bytes", "chunk_bytes", "totals", "dataset_manifests")
        if any(prior.get(key) != top.get(key) for key in stable_keys):
            raise RuntimeError("top-level receipt does not match verified dataset receipts")
        return prior | {"verification_status": "verified_existing"}
    atomic_json(top_path, top)
    return top


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmark/reference/datasets/project179_core19_utc_policy_20260928.json",
    )
    parser.add_argument("--datasets", nargs="+", help="Optional policy dataset names, separated by spaces or commas")
    parser.add_argument("--limit-bytes", type=int, default=DEFAULT_LIMIT_BYTES)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    datasets = None
    if args.datasets:
        datasets = [name for item in args.datasets for name in item.split(",")]
    result = prepare_scope(
        args.active_root, args.backup_root, args.output, args.policy,
        datasets=datasets, limit_bytes=args.limit_bytes,
        chunk_bytes=args.chunk_bytes,
        resume=args.resume, verify_only=args.verify_only,
        progress=lambda value: print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True),
    )
    print(json.dumps({
        "status": result["status"], "dataset_count": result["dataset_count"],
        "totals": result["totals"], "manifest": str((args.output / "manifest.json").resolve()),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
