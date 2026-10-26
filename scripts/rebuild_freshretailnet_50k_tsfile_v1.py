#!/usr/bin/env python3
"""Rebuild FreshRetailNet-50K into width-preserving Apache TsFiles.

The frozen BenchmarkHost conversion narrows ``hours_sale`` from the documented
``Sequence(float64)`` to FLOAT and represents ``hours_stock_status`` as FLOAT.
This one-time, fail-closed rebuild starts from the exact direct-upstream
Parquet snapshot audited by :mod:`audit_rebuild_freshretailnet_50k_v1` and
publishes two derived TsFiles with a shared schema:

* every ``hours_sale[0..23]`` value is DOUBLE;
* every ``hours_stock_status[0..23]`` value is INT32 after an exhaustive
  0/1 and safe-narrowing check;
* the three documented scalar INT32 fields remain INT32;
* the six scalar float64 fields remain DOUBLE;
* the seven source INT64 entity identifiers use a canonical decimal STRING
  tag representation that is exactly reversible to INT64; and
* the date-only ``dt`` label is encoded as Unix milliseconds at UTC midnight,
  explicitly as a reversible coordinate convention rather than an inferred
  event timezone.

The upstream train/eval names are retained only as two source fragments.  The
generated formal collection spec requires a global ``(entity, Time)`` merge
before the experiment's own per-entity chronological 80/20 split.  The tool
never overwrites a prior rebuild, verifies every output value through an
independent cold TsFile read, and emits exact source/output/spec fingerprints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from audit_rebuild_freshretailnet_50k_v1 import (
    CREATED_DATE,
    ENTITY_COLUMNS,
    SCHEMA as SOURCE_AUDIT_SCHEMA,
    BENCHMARKHOST_REVISION,
    UPSTREAM_REVISION,
    FreshRetailAuditError,
    _date_days,
    _physical_schema,
    audit_and_rebuild,
    sha256_file,
)


REBUILD_SCHEMA = "freshretailnet-50k-width-preserving-tsfile-rebuild-v1"
TABLE_NAME = "freshretailnet_50k_rebuilt"
DERIVED_SUBDIRECTORY = "rebuilt_width_preserving_v1"
SOURCE_AUDIT_REPOSITORY_PATH = (
    "analysis_outputs/weight_conditioned_grpo_20260917_v1/data_audit/"
    "freshretailnet_50k_parquet_rebuild_audit_v1.json"
)
SOURCE_AUDIT_SHA256 = (
    "b7db7dc11a43bbe987b71fdf20b88f8bdc251f456e8e16eab35e91cf7feb9985"
)
BENCHMARKHOST_LOCATOR = "https://huggingface.co/datasets/BenchmarkHost/freshretailnet_50k"
DIRECT_UPSTREAM_LOCATOR = (
    "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K"
)
MILLISECONDS_PER_DAY = 86_400_000
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("sale_amount", "DOUBLE"),
    ("stock_hour6_22_cnt", "INT32"),
    ("discount", "DOUBLE"),
    ("holiday_flag", "INT32"),
    ("activity_flag", "INT32"),
    ("precpt", "DOUBLE"),
    ("avg_temperature", "DOUBLE"),
    ("avg_humidity", "DOUBLE"),
    ("avg_wind_level", "DOUBLE"),
)
HOURLY_SALE_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (f"hours_sale_{hour:02d}", "DOUBLE") for hour in range(24)
)
HOURLY_STATUS_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (f"hours_stock_status_{hour:02d}", "INT32") for hour in range(24)
)
VALUE_FIELDS = (*SCALAR_FIELDS, *HOURLY_SALE_FIELDS, *HOURLY_STATUS_FIELDS)
NUMERIC_SCHEMA_SHA256 = (
    "d371ccfbf737bf6b459d535d3943eb480d0ea93e64e79159736c441a9ce73aeb"
)


class FreshRetailRebuildError(RuntimeError):
    """The frozen rebuild contract or a reconstructed output was violated."""


def _progress(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)


@dataclass(frozen=True)
class SplitBuildResult:
    source_split: str
    output_name: str
    rows: int
    entities: int
    output_bytes: int
    output_sha256: str
    semantic_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_split": self.source_split,
            "output_name": self.output_name,
            "rows": self.rows,
            "entities": self.entities,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "semantic_sha256": self.semantic_sha256,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshRetailRebuildError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise FreshRetailRebuildError(f"{label} must be a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _without_normalized_artifacts(report: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(report))
    for split in value.get("splits", []):
        if isinstance(split, dict):
            for key in ("normalized_path", "normalized_bytes", "normalized_sha256"):
                split.pop(key, None)
    return value


def validate_frozen_source_audit(
    audit_path: Path, observed_report: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Bind a live exhaustive audit to the frozen, reviewed audit receipt."""
    if sha256_file(audit_path) != SOURCE_AUDIT_SHA256:
        raise FreshRetailRebuildError(
            "frozen FreshRetailNet audit fingerprint changed; review a new audit version "
            "instead of silently accepting it"
        )
    frozen = _load_json(audit_path, "FreshRetailNet frozen source audit")
    required = {
        "schema": SOURCE_AUDIT_SCHEMA,
        "created_date": CREATED_DATE,
        "dataset_id": "freshretailnet_50k",
        "benchmarkhost_revision": BENCHMARKHOST_REVISION,
        "direct_upstream_revision": UPSTREAM_REVISION,
        "entities": 50_000,
        "total_rows": 4_850_000,
        "entity_time_duplicates_within_or_across_files": 0,
        "cross_file_date_gaps": 0,
        "status": "source_audit_and_width_normalization_passed",
    }
    for key, expected in required.items():
        if frozen.get(key) != expected:
            raise FreshRetailRebuildError(
                f"frozen source audit field changed: {key}: {frozen.get(key)!r}"
            )
    if _without_normalized_artifacts(frozen) != _without_normalized_artifacts(observed_report):
        raise FreshRetailRebuildError(
            "live exhaustive Parquet audit does not match the frozen reviewed audit"
        )
    return frozen


def _tsfile_module():
    try:
        import tsfile
    except Exception as error:  # pragma: no cover - depends on runtime installation
        raise FreshRetailRebuildError(
            f"TSFILE_DEPENDENCY_BLOCKER: Apache tsfile Python package unavailable: {error}"
        ) from error
    return tsfile


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    numeric = {
        "1": "INT32",
        "2": "INT64",
        "3": "FLOAT",
        "4": "DOUBLE",
        "11": "STRING",
        "0": "TAG",
    }
    return numeric.get(text, text.upper())


_SEMANTIC_ROW_DTYPE = np.dtype(
    [
        *((f"entity_{index}", "<i8") for index, _name in enumerate(ENTITY_COLUMNS)),
        ("timestamp_unix_ms", "<i8"),
        *(
            (name, "<i4" if kind == "INT32" else "<f8")
            for name, kind in VALUE_FIELDS
        ),
    ],
    align=False,
)
SEMANTIC_ROW_BYTES = 412
SEMANTIC_DIGEST_DEFINITION = (
    "For each canonical seven-INT64 entity, SHA-256 hashes chronological packed "
    "rows containing all entity INT64 values, the INT64 Unix-ms timestamp, 30 "
    "little-endian FLOAT64 fields, and 27 little-endian INT32 fields. The split "
    "digest hashes the canonical-decimal entity key, row count, and entity digest "
    "in lexicographic entity-key order. Thus physical inter-entity scan order is "
    "irrelevant while every reconstructed tag, timestamp, value bit, and within-"
    "entity chronological association remains covered."
)

if _SEMANTIC_ROW_DTYPE.itemsize != SEMANTIC_ROW_BYTES:
    raise AssertionError(
        "FreshRetailNet packed semantic row width changed without a schema version"
    )


class _EntitySemanticAccumulator:
    """Order-independent across entities, exact and chronological within each entity."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, ...], tuple[int, bytes]] = {}
        self._current_key: tuple[str, ...] | None = None
        self._current_hasher: Any | None = None
        self._current_rows = 0
        self._last_timestamp: int | None = None
        self.rows = 0

    def _finish_current(self) -> None:
        if self._current_key is None:
            return
        if self._current_key in self._records:
            raise FreshRetailRebuildError(
                "entity rows are not contiguous in the physical source or cold scan: "
                f"{self._current_key}"
            )
        assert self._current_hasher is not None
        self._records[self._current_key] = (
            self._current_rows,
            self._current_hasher.digest(),
        )
        self._current_key = None
        self._current_hasher = None
        self._current_rows = 0
        self._last_timestamp = None

    def update(self, native: Mapping[str, np.ndarray]) -> None:
        required = {
            "timestamp_unix_ms",
            *(f"entity:{name}" for name in ENTITY_COLUMNS),
            *(name for name, _kind in VALUE_FIELDS),
        }
        if set(native) != required:
            raise FreshRetailRebuildError(
                "semantic row fields changed: "
                f"missing={sorted(required - set(native))}, "
                f"unexpected={sorted(set(native) - required)}"
            )
        lengths = {len(np.asarray(values)) for values in native.values()}
        if len(lengths) != 1:
            raise FreshRetailRebuildError(
                f"semantic row columns have inconsistent lengths: {sorted(lengths)}"
            )
        count = lengths.pop()
        if count == 0:
            return

        rows = np.empty(count, dtype=_SEMANTIC_ROW_DTYPE)
        entity_arrays: list[np.ndarray] = []
        for index, name in enumerate(ENTITY_COLUMNS):
            values = np.ascontiguousarray(native[f"entity:{name}"], dtype="<i8")
            if values.ndim != 1:
                raise FreshRetailRebuildError(f"entity column is not one-dimensional: {name}")
            rows[f"entity_{index}"] = values
            entity_arrays.append(values)
        timestamps = np.ascontiguousarray(native["timestamp_unix_ms"], dtype="<i8")
        if timestamps.ndim != 1:
            raise FreshRetailRebuildError("timestamp column is not one-dimensional")
        rows["timestamp_unix_ms"] = timestamps
        for name, kind in VALUE_FIELDS:
            values = np.ascontiguousarray(
                native[name], dtype="<i4" if kind == "INT32" else "<f8"
            )
            if values.ndim != 1:
                raise FreshRetailRebuildError(f"value column is not one-dimensional: {name}")
            rows[name] = values

        changed = np.zeros(count - 1, dtype=np.bool_)
        for values in entity_arrays:
            changed |= values[1:] != values[:-1]
        boundaries = [0, *(np.flatnonzero(changed) + 1).tolist(), count]
        for start, end in zip(boundaries, boundaries[1:]):
            key = tuple(str(int(values[start])) for values in entity_arrays)
            if key != self._current_key:
                self._finish_current()
                if key in self._records:
                    raise FreshRetailRebuildError(
                        "entity rows reappear after another entity: " f"{key}"
                    )
                self._current_key = key
                self._current_hasher = hashlib.sha256()
                self._current_rows = 0
                self._last_timestamp = None
            segment_timestamps = timestamps[start:end]
            if self._last_timestamp is not None and int(segment_timestamps[0]) <= self._last_timestamp:
                raise FreshRetailRebuildError(
                    f"entity time is not strictly increasing across batches: {key}"
                )
            if len(segment_timestamps) > 1 and np.any(
                segment_timestamps[1:] <= segment_timestamps[:-1]
            ):
                raise FreshRetailRebuildError(
                    f"entity time is not strictly increasing within a batch: {key}"
                )
            assert self._current_hasher is not None
            self._current_hasher.update(rows[start:end].tobytes(order="C"))
            self._current_rows += end - start
            self.rows += end - start
            self._last_timestamp = int(segment_timestamps[-1])

    def finish(self) -> tuple[dict[tuple[str, ...], tuple[int, bytes]], str]:
        self._finish_current()
        overall = hashlib.sha256()
        overall.update(struct.pack("<Q", self.rows))
        overall.update(struct.pack("<Q", len(self._records)))
        for key in sorted(self._records):
            count, digest = self._records[key]
            key_bytes = _canonical_json(list(key))
            overall.update(struct.pack("<I", len(key_bytes)))
            overall.update(key_bytes)
            overall.update(struct.pack("<Q", count))
            overall.update(digest)
        return dict(self._records), overall.hexdigest()


def _require_no_nulls(batch: pa.RecordBatch) -> None:
    nulls = {
        name: column.null_count
        for name, column in zip(batch.schema.names, batch.columns)
        if column.null_count
    }
    if nulls:
        raise FreshRetailRebuildError(f"source row group contains nulls: {nulls}")


def _batch_columns(batch: pa.RecordBatch) -> tuple[list[int], list[list[Any]], dict[str, np.ndarray]]:
    """Return timestamps, writer columns, and native arrays for hashing."""
    _require_no_nulls(batch)
    days = _date_days(batch.column(batch.schema.get_field_index("dt")))
    timestamps_array = np.asarray(days, dtype=np.int64) * MILLISECONDS_PER_DAY
    if np.any(timestamps_array < INT64_MIN) or np.any(timestamps_array > INT64_MAX):
        raise FreshRetailRebuildError("date-only coordinate is outside signed INT64 milliseconds")
    timestamps = timestamps_array.tolist()

    writer_columns: list[list[Any]] = []
    native: dict[str, np.ndarray] = {"timestamp_unix_ms": timestamps_array}
    for name in ENTITY_COLUMNS:
        array = batch.column(batch.schema.get_field_index(name)).to_numpy(
            zero_copy_only=False
        )
        array = np.asarray(array, dtype=np.int64)
        native[f"entity:{name}"] = array
        writer_columns.append([str(int(value)) for value in array.tolist()])

    for name, kind in SCALAR_FIELDS:
        array = batch.column(batch.schema.get_field_index(name)).to_numpy(
            zero_copy_only=False
        )
        dtype = np.int32 if kind == "INT32" else np.float64
        array = np.asarray(array, dtype=dtype)
        native[name] = array
        writer_columns.append(array.tolist())

    sale = batch.column(batch.schema.get_field_index("hours_sale"))
    status = batch.column(batch.schema.get_field_index("hours_stock_status"))
    sale_lengths = pc.list_value_length(sale).to_numpy(zero_copy_only=False)
    status_lengths = pc.list_value_length(status).to_numpy(zero_copy_only=False)
    if np.any(sale_lengths != 24) or np.any(status_lengths != 24):
        raise FreshRetailRebuildError("hourly sequences must have exactly 24 values")
    sale_flat = pc.list_flatten(sale)
    status_flat = pc.list_flatten(status)
    if sale_flat.null_count or status_flat.null_count:
        raise FreshRetailRebuildError("hourly sequences contain null elements")
    sale_values = np.asarray(sale_flat.to_numpy(zero_copy_only=False), dtype=np.float64)
    status64 = np.asarray(status_flat.to_numpy(zero_copy_only=False), dtype=np.int64)
    if np.any((status64 != 0) & (status64 != 1)):
        observed = np.unique(status64[(status64 != 0) & (status64 != 1)])[:8].tolist()
        raise FreshRetailRebuildError(
            "hours_stock_status violates the audited 0/1 INT32 narrowing contract: "
            f"sample={observed}"
        )
    status_values = status64.astype(np.int32, copy=False)
    expected = batch.num_rows * 24
    if sale_values.size != expected or status_values.size != expected:
        raise FreshRetailRebuildError("hourly sequence flattening changed the element count")
    for hour in range(24):
        name = f"hours_sale_{hour:02d}"
        array = sale_values[hour::24]
        native[name] = array
        writer_columns.append(array.tolist())
    for hour in range(24):
        name = f"hours_stock_status_{hour:02d}"
        array = status_values[hour::24]
        native[name] = array
        writer_columns.append(array.tolist())
    if len(writer_columns) != len(ENTITY_COLUMNS) + len(VALUE_FIELDS):
        raise AssertionError("FreshRetailNet writer schema and extracted columns diverged")
    return timestamps, writer_columns, native


def _tsfile_schema():
    tsfile = _tsfile_module()
    fields = [
        tsfile.ColumnSchema(name, tsfile.TSDataType.STRING, tsfile.ColumnCategory.TAG)
        for name in ENTITY_COLUMNS
    ]
    for name, kind in VALUE_FIELDS:
        data_type = tsfile.TSDataType.INT32 if kind == "INT32" else tsfile.TSDataType.DOUBLE
        fields.append(tsfile.ColumnSchema(name, data_type, tsfile.ColumnCategory.FIELD))
    return tsfile.TableSchema(TABLE_NAME, fields)


def _verify_output_schema(reader: Any) -> None:
    schemas = reader.get_all_table_schemas()
    if set(schemas) != {TABLE_NAME}:
        raise FreshRetailRebuildError(
            f"rebuilt TsFile table set changed: {sorted(schemas)}"
        )
    columns = schemas[TABLE_NAME].get_columns()
    expected = [
        *( (name, "STRING", "TAG") for name in ENTITY_COLUMNS ),
        *( (name, kind, "FIELD") for name, kind in VALUE_FIELDS ),
    ]
    observed = [
        (
            str(column.get_column_name()),
            _enum_name(column.get_data_type()),
            _enum_name(column.get_category()),
        )
        for column in columns
    ]
    if observed != expected:
        raise FreshRetailRebuildError(
            f"rebuilt TsFile schema mismatch: expected={expected!r}, observed={observed!r}"
        )


def _native_row_bytes(native: Mapping[str, np.ndarray], index: int) -> bytes:
    fields = [*(f"entity:{name}" for name in ENTITY_COLUMNS), "timestamp_unix_ms"]
    fields.extend(name for name, _kind in VALUE_FIELDS)
    return b"".join(np.asarray(native[name])[index:index + 1].tobytes() for name in fields)


def _trim_reader_boundary(
    native: dict[str, np.ndarray],
    previous_signature: tuple[tuple[str, ...], int] | None,
    previous_row_bytes: bytes | None,
) -> dict[str, np.ndarray]:
    signature = (
        tuple(str(int(native[f"entity:{name}"][0])) for name in ENTITY_COLUMNS),
        int(native["timestamp_unix_ms"][0]),
    )
    if previous_signature == signature:
        if _native_row_bytes(native, 0) != previous_row_bytes:
            raise FreshRetailRebuildError(
                "reader repeated a boundary key/time with different value bits"
            )
        return {name: values[1:] for name, values in native.items()}
    return native


def _verify_rebuilt_tsfile(
    path: Path,
    *,
    expected_rows: int,
    expected_entities: Mapping[tuple[str, ...], tuple[int, bytes]],
    expected_semantic_sha256: str,
) -> tuple[int, str]:
    """Bulk cold-read all cells and compare exact entity/time/value associations."""
    tsfile = _tsfile_module()
    reader = tsfile.TsFileReader(str(path))
    accumulator = _EntitySemanticAccumulator()
    started = time.monotonic()
    next_progress = 100_000
    _progress("cold_verify_start", file=path.name, expected_rows=expected_rows)
    try:
        _verify_output_schema(reader)
        query_names = [*ENTITY_COLUMNS, *(name for name, _kind in VALUE_FIELDS)]
        result = reader.query_table(TABLE_NAME, query_names, INT64_MIN, INT64_MAX)
        try:
            metadata = result.get_metadata()
            names = [str(name) for name in metadata.get_column_list()]
            expected_names = ["time", *query_names]
            if [name.casefold() for name in names] != [
                name.casefold() for name in expected_names
            ]:
                raise FreshRetailRebuildError(
                    f"cold query column order changed: {names!r}"
                )
            previous_signature: tuple[tuple[str, ...], int] | None = None
            previous_row_bytes: bytes | None = None
            while True:
                frame = result.read_data_frame(8192)
                if frame is None or frame.empty:
                    break
                if list(frame.columns) != expected_names:
                    raise FreshRetailRebuildError(
                        f"cold data-frame column order changed: {list(frame.columns)!r}"
                    )

                timestamp_values = np.asarray(frame["time"].to_numpy())
                if timestamp_values.dtype.kind != "i" or timestamp_values.dtype.itemsize != 8:
                    raise FreshRetailRebuildError("cold timestamp representation is not INT64")
                native: dict[str, np.ndarray] = {
                    "timestamp_unix_ms": np.asarray(
                        timestamp_values, dtype=np.int64
                    )
                }
                for name in ENTITY_COLUMNS:
                    restored: list[int] = []
                    for raw in frame[name].tolist():
                        if raw is None:
                            raise FreshRetailRebuildError(
                                f"cold output tag is null: {name}"
                            )
                        text = str(raw)
                        if re.fullmatch(r"-?(0|[1-9][0-9]*)", text) is None:
                            raise FreshRetailRebuildError(
                                "entity tag is not canonical decimal INT64: "
                                f"{name}={text!r}"
                            )
                        value = int(text)
                        if not INT64_MIN <= value <= INT64_MAX or str(value) != text:
                            raise FreshRetailRebuildError(
                                "entity tag cannot be exactly restored to INT64: "
                                f"{name}={text!r}"
                            )
                        restored.append(value)
                    values = np.asarray(restored, dtype=np.int64)
                    native[f"entity:{name}"] = values

                for name, kind in VALUE_FIELDS:
                    raw = np.asarray(frame[name].to_numpy())
                    if kind == "INT32":
                        if raw.dtype.kind not in "iu":
                            raise FreshRetailRebuildError(
                                f"cold output INT32 column changed representation: {name}"
                            )
                        wide = raw.astype(np.int64, copy=False)
                        if np.any(wide < -(1 << 31)) or np.any(wide > (1 << 31) - 1):
                            raise FreshRetailRebuildError(
                                f"cold output INT32 overflow: {name}"
                            )
                        native[name] = wide.astype(np.int32, copy=False)
                    else:
                        if raw.dtype.kind != "f":
                            raise FreshRetailRebuildError(
                                f"cold output DOUBLE column changed representation: {name}"
                            )
                        native[name] = raw.astype(np.float64, copy=False)

                # tsfile-python 2.0.2 repeats the last row at a chunk boundary
                # and at EOF. Check all value bits before removing that row.
                native = _trim_reader_boundary(native, previous_signature, previous_row_bytes)
                if len(native["timestamp_unix_ms"]) == 0:
                    break
                accumulator.update(native)
                last = len(native["timestamp_unix_ms"]) - 1
                previous_signature = (
                    tuple(str(int(native[f"entity:{name}"][last])) for name in ENTITY_COLUMNS),
                    int(native["timestamp_unix_ms"][last]),
                )
                previous_row_bytes = _native_row_bytes(native, last)
                if accumulator.rows >= next_progress:
                    _progress("cold_verify_progress", file=path.name, rows=accumulator.rows,
                              expected_rows=expected_rows, elapsed_seconds=time.monotonic() - started)
                    next_progress = accumulator.rows + 100_000
        finally:
            result.close()
    finally:
        reader.close()
    observed_entities, semantic = accumulator.finish()
    if accumulator.rows != expected_rows:
        raise FreshRetailRebuildError(
            "cold output row count mismatch: "
            f"expected={expected_rows}, observed={accumulator.rows}"
        )
    if observed_entities != dict(expected_entities) or semantic != expected_semantic_sha256:
        mismatched = sorted(
            key
            for key in set(observed_entities) | set(expected_entities)
            if observed_entities.get(key) != expected_entities.get(key)
        )
        raise FreshRetailRebuildError(
            "cold TsFile reconstruction differs from the exact source-width semantics; "
            f"mismatched_entities={mismatched[:8]}, total={len(mismatched)}"
        )
    _progress("cold_verify_complete", file=path.name, rows=accumulator.rows,
              entities=len(observed_entities), elapsed_seconds=time.monotonic() - started)
    return len(observed_entities), semantic


def write_rebuilt_split(
    source_parquet: Path,
    output_tsfile: Path,
    *,
    source_split: str,
    expected_rows: int,
) -> SplitBuildResult:
    """Write and fully cold-verify one already-audited Parquet fragment."""
    if output_tsfile.exists():
        raise FreshRetailRebuildError(f"refusing to overwrite output: {output_tsfile}")
    parquet = pq.ParquetFile(source_parquet)
    if parquet.schema_arrow.remove_metadata() != _physical_schema():
        raise FreshRetailRebuildError(
            f"source physical schema changed: {source_parquet}: {parquet.schema_arrow}"
        )
    if parquet.metadata.num_rows != expected_rows:
        raise FreshRetailRebuildError(
            f"source row count changed: expected={expected_rows}, "
            f"observed={parquet.metadata.num_rows}"
        )
    output_tsfile.parent.mkdir(parents=True, exist_ok=True)
    tsfile = _tsfile_module()
    writer = tsfile.TsFileTableWriter(str(output_tsfile), _tsfile_schema())
    source_semantics = _EntitySemanticAccumulator()
    rows = 0
    started = time.monotonic()
    next_progress = 100_000
    _progress("write_start", split=source_split, expected_rows=expected_rows)
    try:
        names = [*ENTITY_COLUMNS, *(name for name, _kind in VALUE_FIELDS)]
        types = [
            *(tsfile.TSDataType.STRING for _name in ENTITY_COLUMNS),
            *(
                tsfile.TSDataType.INT32 if kind == "INT32" else tsfile.TSDataType.DOUBLE
                for _name, kind in VALUE_FIELDS
            ),
        ]
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group)
            batches = table.combine_chunks().to_batches()
            if len(batches) != 1:
                raise FreshRetailRebuildError("Parquet row group did not combine to one batch")
            batch = batches[0]
            timestamps, columns, native = _batch_columns(batch)
            tablet = tsfile.Tablet(names, types, batch.num_rows)
            tablet.set_table_name(TABLE_NAME)
            tablet.set_timestamp_list(timestamps)
            # Direct column assignment is the documented Tablet storage model and
            # avoids hundreds of millions of redundant Python range checks.  Every
            # value was type/range audited above and is checked again by cold read.
            tablet.data_list = columns
            writer.write_table(tablet)
            source_semantics.update(native)
            rows += batch.num_rows
            if rows >= next_progress:
                _progress("write_progress", split=source_split, rows=rows,
                          expected_rows=expected_rows, elapsed_seconds=time.monotonic() - started)
                next_progress = rows + 100_000
        writer.close()
        _progress("write_complete", split=source_split, rows=rows,
                  elapsed_seconds=time.monotonic() - started)
    except BaseException:
        try:
            writer.close()
        finally:
            output_tsfile.unlink(missing_ok=True)
        raise
    if rows != expected_rows:
        output_tsfile.unlink(missing_ok=True)
        raise FreshRetailRebuildError(
            f"written row count changed: expected={expected_rows}, observed={rows}"
        )
    expected_entities, semantic = source_semantics.finish()
    if source_semantics.rows != rows:
        output_tsfile.unlink(missing_ok=True)
        raise FreshRetailRebuildError(
            "source semantic row count diverged from the written row count"
        )
    try:
        observed_entity_count, observed_semantic = _verify_rebuilt_tsfile(
            output_tsfile,
            expected_rows=rows,
            expected_entities=expected_entities,
            expected_semantic_sha256=semantic,
        )
    except BaseException:
        output_tsfile.unlink(missing_ok=True)
        raise
    return SplitBuildResult(
        source_split=source_split,
        output_name=output_tsfile.name,
        rows=rows,
        entities=observed_entity_count,
        output_bytes=output_tsfile.stat().st_size,
        output_sha256=sha256_file(output_tsfile),
        semantic_sha256=observed_semantic,
    )


def value_column_spec() -> list[dict[str, str]]:
    return [
        {"source_name": name, "output_name": name, "tsfile_type": kind}
        for name, kind in VALUE_FIELDS
    ]


def numeric_schema_sha256() -> str:
    digest = _sha256_bytes(_canonical_json(value_column_spec()))
    if digest != NUMERIC_SCHEMA_SHA256:
        raise FreshRetailRebuildError(
            "FreshRetailNet native numeric schema changed without a version update"
        )
    return digest


def build_formal_spec(
    campaign_source_root: Path,
    split_results: Sequence[SplitBuildResult],
    *,
    source_audit_sha256: str = SOURCE_AUDIT_SHA256,
) -> dict[str, Any]:
    if [result.source_split for result in split_results] != ["train", "eval"]:
        raise FreshRetailRebuildError(
            "derived fragments must be declared in upstream train then eval order"
        )
    files = []
    for ordinal, result in enumerate(split_results, start=1):
        expected_name = f"freshretailnet_50k_rebuilt_series_{ordinal}.tsfile"
        if result.output_name != expected_name:
            raise FreshRetailRebuildError(
                f"derived fragment name changed: {result.output_name!r}"
            )
        repo_path = f"{DERIVED_SUBDIRECTORY}/{expected_name}"
        files.append({
            "repo_path": repo_path,
            "path": str((campaign_source_root / "freshretailnet_50k" / repo_path).resolve()),
            "expected_sha256": result.output_sha256,
            "derived_from_upstream_split": result.source_split,
        })
    schema_digest = numeric_schema_sha256()
    return {
        "format_version": "weighted-grpo-tsfile-collection-preparation-v1",
        "spec_id": "freshretailnet-50k-width-preserving-tsfile-collection-v1",
        "created_date": CREATED_DATE,
        "purpose": "exact-revision width-preserving train-validation materialization",
        "dataset_id": "freshretailnet_50k",
        "domain": "business",
        "source_group": "hf:Dingdong-Inc/FreshRetailNet-50K",
        "model": "table",
        "dataset_byte_cap": 268_435_456,
        "block_byte_cap": 1_048_576,
        "source": {
            "format": "tsfile_collection",
            "upstream_revision": BENCHMARKHOST_REVISION,
            "upstream_locator": BENCHMARKHOST_LOCATOR,
            "direct_upstream_revision": UPSTREAM_REVISION,
            "direct_upstream_locator": DIRECT_UPSTREAM_LOCATOR,
            "derivation": REBUILD_SCHEMA,
            "schema_policy": "identical_across_all_files",
            "files": files,
        },
        "evidence": {
            "hf_readme": f"{BENCHMARKHOST_LOCATOR}/blob/{BENCHMARKHOST_REVISION}/README.md",
            "direct_upstream_readme": (
                f"{DIRECT_UPSTREAM_LOCATOR}/blob/{UPSTREAM_REVISION}/README.md"
            ),
            "audit_path": SOURCE_AUDIT_REPOSITORY_PATH,
            "audit_sha256": source_audit_sha256,
            "timestamp_claim": (
                "Every entity has 97 unique consecutive Gregorian date labels after "
                "merging the two derived fragments; dates are encoded at UTC midnight"
            ),
            "timezone_boundary": (
                "UTC is the reversible coordinate convention for date-only labels, "
                "not an inferred event timezone"
            ),
            "entity_claim": (
                "Seven source INT64 identifiers are encoded as canonical decimal STRING "
                "tags and are exactly reversible"
            ),
            "schema_claim": (
                "24 hours_sale fields are DOUBLE, 24 hours_stock_status fields are "
                "INT32 after exhaustive 0/1 validation, six scalar float64 fields are "
                "DOUBLE, and three scalar int32 fields are INT32"
            ),
            "numeric_schema_sha256": schema_digest,
        },
        "query_range": {
            "start_inclusive": INT64_MIN,
            "end_inclusive": INT64_MAX,
        },
        "timestamp": {
            "source_unit": "ms",
            "output_unit": "ms",
            "timezone": "UTC",
            "semantic_kind": "calendar_date_utc_midnight_encoding",
            "source_event_timezone_inferred": False,
            "output_name": "timestamp_unix_ms",
            "duplicate_policy": "reject",
        },
        "row_contract": {
            "ordering_key": "Time",
            "emit_ordering_key_as_numeric_column": True,
            "ordering_key_output_dtype": "INT64",
            "sort_scope": "global_entity_timestamp_ascending",
            "duplicate_comparison_scope": "all_listed_source_files",
            "cross_file_entity_merge": True,
            "source_file_order": "numeric_series_suffix_ascending",
            "split_after_global_merge": "per_entity_chronological_80_20",
        },
        "table": {
            "table_name": TABLE_NAME,
            "entity_columns": list(ENTITY_COLUMNS),
            "value_columns": value_column_spec(),
        },
        "reconstruction": {
            "hourly_field_mapping": {
                "hours_sale_00..23": "hours_sale list indexes 0..23",
                "hours_stock_status_00..23": "hours_stock_status list indexes 0..23",
            },
            "entity_tag_inverse": "parse each canonical decimal tag as signed INT64",
            "date_inverse": "Unix milliseconds / 86400000 -> Gregorian date-only dt label",
            "upstream_fragment_labels": ["train", "eval"],
            "fragment_labels_are_experiment_split": False,
        },
    }


def rebuild_frozen_source(
    source_root: Path,
    campaign_source_root: Path,
    formal_spec_output: Path,
    *,
    source_audit_path: Path,
    formal_source_root: Path | None = None,
    receipt_output: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    campaign_source_root = campaign_source_root.resolve()
    source_audit_path = source_audit_path.resolve()
    formal_spec_output = formal_spec_output.resolve()
    formal_source_root = (
        formal_source_root.resolve()
        if formal_source_root is not None
        else campaign_source_root
    )
    if formal_source_root.name != "training_raw":
        raise FreshRetailRebuildError(
            "formal source root must end in /training_raw to satisfy the frozen "
            "source layout; use --campaign-source-root .../training_raw"
        )
    if formal_spec_output.exists():
        raise FreshRetailRebuildError(
            f"refusing to overwrite formal spec: {formal_spec_output}"
        )
    if receipt_output is not None and receipt_output.resolve().exists():
        raise FreshRetailRebuildError(
            f"refusing to overwrite receipt: {receipt_output.resolve()}"
        )
    final_root = campaign_source_root / "freshretailnet_50k" / DERIVED_SUBDIRECTORY
    if final_root.exists():
        raise FreshRetailRebuildError(f"refusing to overwrite rebuild: {final_root}")

    try:
        _progress("source_audit_start", source_root=str(source_root))
        observed_audit = audit_and_rebuild(source_root)
    except FreshRetailAuditError as error:
        raise FreshRetailRebuildError(f"FROZEN_SOURCE_AUDIT_BLOCKER: {error}") from error
    frozen_audit = validate_frozen_source_audit(source_audit_path, observed_audit)
    _progress("source_audit_complete", source_audit_sha256=SOURCE_AUDIT_SHA256)

    parent = final_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{DERIVED_SUBDIRECTORY}.", dir=parent))
    split_results: list[SplitBuildResult] = []
    stage = "write_and_cold_verify_fragments"
    receipt: dict[str, Any] | None = None
    try:
        for ordinal, (split, rows) in enumerate(
            (("train", 4_500_000), ("eval", 350_000)), start=1
        ):
            output = temporary / f"freshretailnet_50k_rebuilt_series_{ordinal}.tsfile"
            split_results.append(write_rebuilt_split(
                source_root / "data" / f"{split}.parquet",
                output,
                source_split=split,
                expected_rows=rows,
            ))

        local_spec = build_formal_spec(campaign_source_root, split_results)
        formal_spec = build_formal_spec(formal_source_root, split_results)
        from prepare_weighted_grpo_timeseries_v1 import (
            DEFAULT_COHORT_MANIFEST,
            load_cohort_manifest,
        )
        from validate_weighted_grpo_formal_dataset_specs_v1 import validate_formal_spec

        stage = "static_formal_spec_validation"
        _progress(stage)
        cohort = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
        cohort_entry = next(
            item for item in cohort["datasets"] if item["id"] == "freshretailnet_50k"
        )
        static_formal_audit = validate_formal_spec(
            formal_spec,
            cohort_entry=cohort_entry,
            revision=BENCHMARKHOST_REVISION,
            expected_numeric_schema_sha256=NUMERIC_SCHEMA_SHA256,
        )
        source_files = {
            item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]}
            for item in frozen_audit["upstream_repository_inventory"]
        }
        receipt = {
            "schema": REBUILD_SCHEMA,
            "created_date": CREATED_DATE,
            "dataset_id": "freshretailnet_50k",
            "status": "full_cold_reconstruction_verified_pending_admission",
            "formal_training_eligible": False,
            "source": {
                "benchmarkhost_revision": BENCHMARKHOST_REVISION,
                "direct_upstream_revision": UPSTREAM_REVISION,
                "audit_path": SOURCE_AUDIT_REPOSITORY_PATH,
                "audit_sha256": SOURCE_AUDIT_SHA256,
                "files": source_files,
            },
            "output": {
                "table_name": TABLE_NAME,
                "derived_subdirectory": DERIVED_SUBDIRECTORY,
                "fragments": [result.as_dict() for result in split_results],
                "rows": sum(result.rows for result in split_results),
                "entities": 50_000,
                "value_columns": len(VALUE_FIELDS),
                "native_type_counts": {"DOUBLE": 30, "INT32": 27},
                "numeric_schema_sha256": numeric_schema_sha256(),
                "cold_decode_scope": "all rows, tags, timestamps, and value cells",
                "semantic_digest_definition": SEMANTIC_DIGEST_DEFINITION,
                "semantic_packed_row_bytes": SEMANTIC_ROW_BYTES,
            },
            "local_spec_canonical_sha256": _sha256_bytes(_canonical_json(local_spec)),
            "formal_spec_canonical_sha256": _sha256_bytes(_canonical_json(formal_spec)),
            "formal_contract": {
                "merge_before_split": True,
                "split": "per-entity chronological first 80% train, last 20% validation",
                "timestamp": "calendar date encoded at UTC midnight in Unix milliseconds",
                "source_event_timezone_inferred": False,
                "identifier_reconstruction": "canonical decimal STRING tag to INT64",
                "hourly_reconstruction": "field suffix 00..23 maps to list position 0..23",
            },
        }
        _write_json_atomic(temporary / "rebuild_receipt.json", receipt)
        _write_json_atomic(temporary / "preparation_spec.json", local_spec)
        os.replace(temporary, final_root)
        temporary = Path()

        # Exercise the production loader and a bounded schema/data audit after the
        # final paths exist.  Full value equality was already checked above.
        from prepare_weighted_grpo_tsfile_v1 import (
            audit_tsfile_spec,
            load_tsfile_spec,
        )

        internal_spec = final_root / "preparation_spec.json"
        loaded = load_tsfile_spec(internal_spec)
        stage = "production_collection_loader_audit"
        _progress(stage, source_scan="all rows merged before selecting 32 audit rows")
        bounded_audit = audit_tsfile_spec(
            loaded,
            cohort_manifest=load_cohort_manifest(DEFAULT_COHORT_MANIFEST),
            row_limit=32,
        )
        if bounded_audit.get("status") != "schema_and_bounded_rows_verified":
            raise FreshRetailRebuildError("production TsFile preparation audit did not pass")
        _progress("production_collection_loader_audit_complete")
        stage = "publish_verified_receipts"
        receipt["status"] = "rebuilt_and_full_cold_reconstruction_verified"
        receipt["formal_training_eligible"] = False
        receipt["admission_boundary"] = "source reconstruction only; prepared-data and cohort gates remain required"
        receipt["production_preparer_audit"] = bounded_audit
        receipt["formal_spec_static_audit"] = static_formal_audit
        receipt["local_spec_file_sha256"] = sha256_file(internal_spec)
        _write_json_atomic(final_root / "rebuild_receipt.json", receipt)
        _write_json_atomic(formal_spec_output, formal_spec)
        receipt["formal_spec_file_sha256"] = sha256_file(formal_spec_output)
        _write_json_atomic(final_root / "rebuild_receipt.json", receipt)
        if receipt_output is not None:
            _write_json_atomic(receipt_output.resolve(), receipt)
        return receipt
    except BaseException as error:
        preserved = final_root if final_root.exists() else temporary
        if preserved.exists() and preserved.parent == parent:
            failure = {
                "schema": "freshretailnet-50k-rebuild-failure-v1",
                "status": "failed_preserved_for_recovery",
                "formal_training_eligible": False,
                "failed_stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "preserved_directory": str(preserved),
                "full_cold_verified_fragments": [result.as_dict() for result in split_results],
                "formal_spec_published": formal_spec_output.exists(),
                "recovery": (
                    "Retain and independently rehash these artifacts. Fix the reported gate "
                    "before publishing a formal spec; an existing directory is never overwritten."
                ),
            }
            try:
                if receipt is not None:
                    receipt["status"] = "full_cold_verified_but_admission_failed"
                    receipt["formal_training_eligible"] = False
                    receipt["failure"] = failure
                    _write_json_atomic(preserved / "rebuild_receipt.json", receipt)
                _write_json_atomic(preserved / "rebuild_failure_receipt.json", failure)
                _progress("rebuild_failed_artifacts_preserved", directory=str(preserved),
                          failed_stage=stage, error=str(error))
            except OSError as receipt_error:
                print(f"FAILED_TO_WRITE_RECOVERY_RECEIPT: {receipt_error}", file=sys.stderr)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--campaign-source-root", type=Path, required=True)
    parser.add_argument(
        "--formal-source-root",
        type=Path,
        help=(
            "optional installed campaign root written into the external formal spec; "
            "the local campaign root remains the build and cold-verification location"
        ),
    )
    parser.add_argument("--formal-spec-output", type=Path, required=True)
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=repository_root / SOURCE_AUDIT_REPOSITORY_PATH,
    )
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = rebuild_frozen_source(
            args.source_root,
            args.campaign_source_root,
            args.formal_spec_output,
            source_audit_path=args.source_audit,
            formal_source_root=args.formal_source_root,
            receipt_output=args.receipt_output,
        )
    except FreshRetailRebuildError as error:
        print(f"FRESHRETAILNET_REBUILD_BLOCKER: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
