#!/usr/bin/env python3
"""Export the frozen project179 THULab TsFiles to capped numeric CSVs.

The exporter deliberately uses the existing scope/revision-bound timestamp
policy parser.  It does not infer timestamp units or promote arbitrary strings
to timestamps.  Each dataset is written as::

    <dataset>/data.csv
    <dataset>/series_map.csv
    <dataset>/columns.csv
    <dataset>/dataset_manifest.json

``data.csv`` contains ``timestamp_unix_us``, an integer ``series_id``, and a
wide prefix of the retained numeric/approved-temporal value columns.  Original
TAG values are present only in ``series_map.csv``.  The column map and receipt
are inside the dataset directory; all four published files count against the
decimal 1,000,000,000-byte dataset cap.

Apache TsFile Python 2.1.5 has two relevant behaviours which are easy to get
wrong:

* ``read_data_frame(N)`` must be guarded by ``while result.next()``.  Calling
  it repeatedly without the guard repeats the final row at EOF.
* its pandas dtype conversion can fail after consuming a batch when an integer
  column contains nulls.  A source-file SQLite savepoint lets this exporter
  roll back the partial batch scan, reopen the query, and use the exact
  row/cell API (including ``is_null_by_index``) without duplication.

All schema TAG, FIELD, and ATTRIBUTE columns are normally projected, even
though ordinary strings are not emitted.  This is the conservative coverage
anchor for sparse table rows: a query of only one sparse retained FIELD, or
TAGs alone, can omit rows in the 2.1.5 SDK.  The only exception omits nine
unqueryable ordinary STRING fields from two hash-bound Africa source objects
and requires their frozen decoded-row counts.  Values are staged in SQLite
and inserted in batches.  Final rows are globally ordered by converted
timestamp, frozen source order, and frozen source row order.

The exact 46+12 positive-infinity ``z_gpp`` cells in those same frozen Africa
objects are serialized as lowercase ``inf`` and counted in the receipt.
True IEEE FLOAT/DOUBLE NaNs are serialized as lowercase ``nan`` and counted
separately from SDK nulls.  Every other positive infinity, every negative
infinity, and any NaN in a non-floating column fails closed.

The two approved temporal STRING fields in the exact frozen Forbush source
also contain non-NULL empty strings.  Those cells are represented by an empty
CSV field, just like a source NULL, but complete zero-based source-row ordinal
lists in the receipt preserve which representation was present upstream.  No
other source or temporal field receives this exception.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ProcessPoolExecutor,
    wait,
)
import ctypes
import errno
import csv
from dataclasses import dataclass
import faulthandler
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence

import prepare_project179_thulab_numeric_v1 as policy_core
from prepare_weighted_grpo_timeseries_v1 import PreparationError
from prepare_weighted_grpo_tsfile_v1 import _typed_entity_component


FORMAT_VERSION = "project179-thulab-numeric-wide-csv-v1"
RECEIPT_VERSION = "project179-thulab-numeric-wide-csv-receipt-v1"
COLLECTION_VERSION = "project179-thulab-numeric-wide-csv-collection-v1"
FAILURE_VERSION = "project179-thulab-numeric-wide-csv-failure-v1"
SCHEMA_PREFLIGHT_VERSION = "project179-thulab-numeric-wide-csv-schema-preflight-v1"
WORKER_DIAGNOSTIC_VERSION = "project179-thulab-csv-worker-diagnostic-v1"
WORKER_DIAGNOSTIC_DIRECTORY = ".worker_diagnostics"
WORKER_DIAGNOSTIC_HEARTBEAT_SECONDS = 30.0
READ_MODE_NOT_MATERIALIZED = "not_materialized"
READ_MODE_DATAFRAME = "dataframe"
READ_MODE_DATAFRAME_FALLBACK_ROWWISE = "dataframe_fallback_rowwise"
READ_MODE_ROWWISE_DIRECT = "rowwise_direct_source_bound"
REQUIRED_TSFILE_VERSION = "2.1.5"
DEFAULT_DATASET_BYTE_CAP = 1_000_000_000
DEFAULT_BATCH_ROWS = 8192
DEFAULT_INSERT_BATCH_ROWS = 4096
DEFAULT_VALUE_CACHE_SIZE = 65536
DEFAULT_ENTITY_INSERT_CACHE_SIZE = 262144
DEFAULT_WORKERS = 1
MAX_WORKERS = 10
EXPECTED_THULAB_DATASET_COUNT = 160
EXPECTED_POLICY_DECISION_COUNTS = {
    "admit_dataset_adapter": 3,
    "admit_exact_utc": 34,
    "admit_stored_epoch_coordinate": 123,
    "block": 0,
}
ROW_EXCEPTION_SCHEMA = "project179-thulab-csv-row-exceptions-v1"
MACROECON_ROW_EXCEPTION_ID = "macroecon-eight-all-null-int64-min-first-rows-v1"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

# TsFile 2.1.5 rejects these nine schema columns with ColumnNotExistError 50,
# despite returning them from get_all_table_schemas().  This is not a general
# permission to omit strings from the row-universe query.  The exception is
# restricted to the two immutable source objects below and is closed by exact
# decoded row counts after the query completes.
AFRICA_QUERY_EXCEPTION_ID = "africa9-ordinary-string-sdk-column-not-found-v1"
AFRICA_OMITTED_QUERY_COLUMNS = (
    "ndvi_categorías",
    "sipi_categorías",
    "arvi_categorías",
    "evi_categorías",
    "spi_1_categorías",
    "spi_3_categorías",
    "spi_6_categorías",
    "spi_9_categorías",
    "spi_12_categorías",
)
AFRICA_QUERY_EXCEPTIONS: Mapping[tuple[str, str], int] = {
    (
        "train/train.tsfile",
        "7d4c84fed1e3ad52af66fe0b2d3e76e34bc17adca5cf4446e39f5433e087abca",
    ): 21_660,
    (
        "test/test.tsfile",
        "8a42c37fb5c9ea9359e31b5655908cbb6e6c8ef74d0e1c8799564632389bdd3c",
    ): 5_415,
}
AFRICA_QUERY_EXCEPTION_TABLES: Mapping[tuple[str, str], str] = {
    (
        "train/train.tsfile",
        "7d4c84fed1e3ad52af66fe0b2d3e76e34bc17adca5cf4446e39f5433e087abca",
    ): "africa_senegal_mauritania_crops_spi_train",
    (
        "test/test.tsfile",
        "8a42c37fb5c9ea9359e31b5655908cbb6e6c8ef74d0e1c8799564632389bdd3c",
    ): "africa_senegal_mauritania_crops_spi_test",
}
AFRICA_POSITIVE_INFINITY_COUNTS: Mapping[
    tuple[str, str], Mapping[str, int]
] = {
    (
        "train/train.tsfile",
        "7d4c84fed1e3ad52af66fe0b2d3e76e34bc17adca5cf4446e39f5433e087abca",
    ): {"z_gpp": 46},
    (
        "test/test.tsfile",
        "8a42c37fb5c9ea9359e31b5655908cbb6e6c8ef74d0e1c8799564632389bdd3c",
    ): {"z_gpp": 12},
}
POSITIVE_INFINITY_SPOOL_TOKEN = "__project179_positive_infinity_v1__"
NAN_SPOOL_TOKEN = "__project179_nan_v1__"

# TsFile Python 2.1.5 reproducibly segfaults when this source is reopened and
# advanced rowwise after its dataframe iterator raised a nullable-value
# conversion error.  A fresh process reading the same full projection directly
# through the exact row/cell API completes.  Keep this workaround content-,
# schema-, table-, and path-bound: it is not permission to infer read modes for
# another source with similar values or a same-named table.
FORBUSH_DIRECT_ROWWISE_SOURCE = (
    "forbush_decreases.tsfile",
    "35f8df03af8e5fb1c46a6a69f39baad746cfb9575b613a2c91497a8e47b14f46",
    "8c32538e20c5df65737767e6c8695fbd9a3c880f092c792c77c2752ee198d6f1",
    "forbush_decreases",
)
FORBUSH_DATASET_ID = "forbush_decreases"
FORBUSH_SOURCE_ROWS = 8_726
FORBUSH_TEMPORAL_EMPTY_STRING_REPRESENTATION = (
    "csv_empty_source_non_null_empty_distinct_from_null"
)
FORBUSH_TEMPORAL_EMPTY_STRING_EXPECTATIONS: Mapping[str, Mapping[str, int]] = {
    "cme_datetime_utc": {
        "source_non_null_empty_count": 1_629,
        "source_null_count": 6_441,
        "source_valid_count": 656,
    },
    "solar_source_datetime_utc": {
        "source_non_null_empty_count": 1_284,
        "source_null_count": 6_441,
        "source_valid_count": 1_001,
    },
}
FORBUSH_TEMPORAL_EMPTY_STRING_OUTPUT_COLUMNS: Mapping[str, str] = {
    "cme_datetime_utc": "c0021",
    "solar_source_datetime_utc": "c0022",
}
FORBUSH_TEMPORAL_CONVERSION_ANNOTATIONS: tuple[Mapping[str, str], ...] = (
    {
        "name": "cme_datetime_utc",
        "tsfile_type": "STRING",
        "category": "FIELD",
        "decision": "parse_as_int64_unix_us_value_field",
    },
    {
        "name": "solar_source_datetime_utc",
        "tsfile_type": "STRING",
        "category": "FIELD",
        "decision": "parse_as_int64_unix_us_value_field",
    },
)


class CsvExportError(PreparationError):
    """Raised when the CSV export cannot satisfy its frozen contract."""


class _DataFrameFallback(RuntimeError):
    """Internal signal to restart one source through the exact row API."""


class _TerminationRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"termination requested by signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class WorkerDiagnostics:
    """Durable worker identity and native-crash evidence paths."""

    worker_pid: int
    dataset_id: str
    events_path: Path
    faulthandler_path: Path


class _BoundedCache:
    def __init__(self, limit: int = DEFAULT_VALUE_CACHE_SIZE) -> None:
        self.limit = limit
        self.values: OrderedDict[Any, Any] = OrderedDict()

    def get_or_create(self, key: Any, factory: Any) -> Any:
        if key in self.values:
            value = self.values.pop(key)
            self.values[key] = value
            return value
        value = factory()
        self.values[key] = value
        if len(self.values) > self.limit:
            self.values.popitem(last=False)
        return value


class _FreeDiskGuard:
    def __init__(self, path: Path, reserve_bytes: int) -> None:
        self.path = path
        self.reserve_bytes = reserve_bytes

    def check(self, context: str) -> None:
        free = shutil.disk_usage(self.path).free
        if free < self.reserve_bytes:
            raise CsvExportError(
                f"free-disk reserve breached during {context}: "
                f"free={free}, reserve={self.reserve_bytes}"
            )


@dataclass(frozen=True)
class OutputColumn:
    header: str
    logical_key: str
    table_name: str
    source_name: str
    source_type: str
    conversion: str
    dtype: str


@dataclass(frozen=True)
class ScanStats:
    source_rows: int
    eligible_rows: int
    filtered_rows: int
    dataframe_attempted: bool
    rowwise_fallback: bool
    null_tag_rows: int
    null_tag_cells: int
    positive_infinity_counts: tuple[tuple[str, int], ...]
    read_mode: str = READ_MODE_DATAFRAME
    nan_counts: tuple[tuple[str, int], ...] = ()
    temporal_empty_string_audits: tuple[Mapping[str, Any], ...] = ()
    timestamp_policy_filtered_rows: int = 0
    row_exception_filtered_rows: int = 0
    row_exception: Mapping[str, Any] | None = None


@dataclass
class _TemporalEmptyStringTracker:
    """Exact source ordinals for one source-bound empty-string exception."""

    field_names: tuple[str, ...]
    source_non_null_empty_row_ordinals: dict[str, list[int]]
    source_null_row_ordinals: dict[str, list[int]]


@dataclass(frozen=True)
class RowExceptionSource:
    source_path: str
    table_name: str
    source_sha256: str
    expected_source_rows: int
    expected_eligible_rows: int
    expected_dropped_rows: int


@dataclass(frozen=True)
class RowExceptionPolicy:
    file_sha256: str
    exception_id: str
    scope_sha256: str
    timestamp_policy_file_sha256: str
    dataset_id: str
    source_receipt_sha256: str
    expected_source_file_count: int
    expected_candidate_value_column_count: int
    expected_source_rows: int
    expected_eligible_rows: int
    expected_dropped_rows: int
    physical_time: int
    source_row_ordinal: int
    sources: tuple[RowExceptionSource, ...]

    @property
    def sources_by_path(self) -> Mapping[str, RowExceptionSource]:
        return {item.source_path: item for item in self.sources}


@dataclass
class _RowExceptionTracker:
    dropped_source_row_ordinals: list[int]
    physical_time: int | None = None
    all_value_fields_null: bool = False


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CsvExportError(
            f"{label} keys differ from the frozen schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise CsvExportError(f"{label} must be one lowercase SHA-256 hex digest")
    return text


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CsvExportError(f"{label} must be an integer >= {minimum}")
    return value


def _load_row_exception_policy(
    path: Path | None,
    *,
    scope_sha256: str,
    timestamp_policy_file_sha256: str,
) -> RowExceptionPolicy | None:
    """Strictly parse the one user-approved, immutable row exception file."""

    if path is None:
        return None
    if path.is_symlink():
        raise CsvExportError("row-exception policy must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise CsvExportError("row-exception policy must be a regular file")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise CsvExportError("row-exception policy root must be an object")
    _require_exact_keys(
        raw,
        {
            "schema",
            "exception_id",
            "authorization",
            "scope_sha256",
            "timestamp_policy_file_sha256",
            "dataset_id",
            "source_receipt_sha256",
            "expected_source_file_count",
            "expected_candidate_value_column_count",
            "expected_source_rows",
            "expected_eligible_rows",
            "expected_dropped_rows",
            "predicate",
            "sources",
        },
        "row-exception policy",
    )
    if raw["schema"] != ROW_EXCEPTION_SCHEMA:
        raise CsvExportError("unsupported row-exception policy schema")
    if raw["exception_id"] != MACROECON_ROW_EXCEPTION_ID:
        raise CsvExportError("unsupported row-exception policy identity")
    if raw["dataset_id"] != "macroecon":
        raise CsvExportError("row-exception policy may bind only dataset 'macroecon'")
    if raw["scope_sha256"] != scope_sha256:
        raise CsvExportError("row-exception policy scope SHA-256 mismatch")
    if raw["timestamp_policy_file_sha256"] != timestamp_policy_file_sha256:
        raise CsvExportError("row-exception policy timestamp-policy SHA-256 mismatch")
    authorization = raw["authorization"]
    if (
        not isinstance(authorization, str)
        or "dropping only these eight" not in authorization
        or "2026-09-30" not in authorization
    ):
        raise CsvExportError("row-exception policy authorization statement drift")

    predicate = raw["predicate"]
    if not isinstance(predicate, Mapping):
        raise CsvExportError("row-exception predicate must be an object")
    _require_exact_keys(
        predicate,
        {
            "physical_time",
            "source_row_ordinal",
            "schema_column_category",
            "schema_column_type",
            "require_all_value_fields_null",
            "require_no_tag_or_attribute_columns",
        },
        "row-exception predicate",
    )
    if dict(predicate) != {
        "physical_time": INT64_MIN,
        "source_row_ordinal": 0,
        "schema_column_category": "FIELD",
        "schema_column_type": "FLOAT",
        "require_all_value_fields_null": True,
        "require_no_tag_or_attribute_columns": True,
    }:
        raise CsvExportError("row-exception predicate differs from the approved rule")

    source_values = raw["sources"]
    if not isinstance(source_values, list) or len(source_values) != 8:
        raise CsvExportError("row-exception policy must contain exactly eight sources")
    sources: list[RowExceptionSource] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(source_values):
        label = f"row-exception source[{index}]"
        if not isinstance(item, Mapping):
            raise CsvExportError(f"{label} must be an object")
        _require_exact_keys(
            item,
            {
                "path",
                "table_name",
                "sha256",
                "expected_source_rows",
                "expected_eligible_rows",
                "expected_dropped_rows",
            },
            label,
        )
        source_path = item["path"]
        table_name = item["table_name"]
        if not isinstance(source_path, str) or not source_path.endswith(".tsfile"):
            raise CsvExportError(f"{label}.path must be a relative .tsfile path")
        path_parts = Path(source_path)
        if path_parts.is_absolute() or ".." in path_parts.parts or path_parts.as_posix() != source_path:
            raise CsvExportError(f"{label}.path is not a canonical relative path")
        if not isinstance(table_name, str) or table_name != Path(source_path).stem:
            raise CsvExportError(f"{label}.table_name must equal the frozen file stem")
        if source_path in seen_paths:
            raise CsvExportError(f"duplicate row-exception source path: {source_path}")
        seen_paths.add(source_path)
        source_rows = _require_int(item["expected_source_rows"], f"{label}.expected_source_rows", minimum=1)
        eligible_rows = _require_int(item["expected_eligible_rows"], f"{label}.expected_eligible_rows")
        dropped_rows = _require_int(item["expected_dropped_rows"], f"{label}.expected_dropped_rows")
        if dropped_rows != 1 or eligible_rows + dropped_rows != source_rows:
            raise CsvExportError(f"{label} row counts do not describe one dropped row")
        sources.append(
            RowExceptionSource(
                source_path=source_path,
                table_name=table_name,
                source_sha256=_require_sha256(item["sha256"], f"{label}.sha256"),
                expected_source_rows=source_rows,
                expected_eligible_rows=eligible_rows,
                expected_dropped_rows=dropped_rows,
            )
        )

    expected_source_rows = _require_int(raw["expected_source_rows"], "expected_source_rows", minimum=1)
    expected_eligible_rows = _require_int(raw["expected_eligible_rows"], "expected_eligible_rows", minimum=1)
    expected_dropped_rows = _require_int(raw["expected_dropped_rows"], "expected_dropped_rows", minimum=1)
    if (
        expected_dropped_rows != 8
        or expected_eligible_rows + expected_dropped_rows != expected_source_rows
        or sum(item.expected_dropped_rows for item in sources) != expected_dropped_rows
    ):
        raise CsvExportError("row-exception policy aggregate row counts are inconsistent")
    return RowExceptionPolicy(
        file_sha256=_sha256(resolved),
        exception_id=str(raw["exception_id"]),
        scope_sha256=str(raw["scope_sha256"]),
        timestamp_policy_file_sha256=str(raw["timestamp_policy_file_sha256"]),
        dataset_id=str(raw["dataset_id"]),
        source_receipt_sha256=_require_sha256(raw["source_receipt_sha256"], "source_receipt_sha256"),
        expected_source_file_count=_require_int(raw["expected_source_file_count"], "expected_source_file_count", minimum=1),
        expected_candidate_value_column_count=_require_int(raw["expected_candidate_value_column_count"], "expected_candidate_value_column_count", minimum=1),
        expected_source_rows=expected_source_rows,
        expected_eligible_rows=expected_eligible_rows,
        expected_dropped_rows=expected_dropped_rows,
        physical_time=INT64_MIN,
        source_row_ordinal=0,
        sources=tuple(sources),
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CsvExportError(f"refusing to overwrite existing path: {path}")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _emit_progress(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "schema": "project179-thulab-csv-progress-v1",
                "event": event,
                **fields,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def _worker_diagnostic_directory(output_root: Path) -> Path:
    """Return the stage-local diagnostic directory, never following a link."""

    directory = output_root / WORKER_DIAGNOSTIC_DIRECTORY
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise CsvExportError(f"worker diagnostic path is not a real directory: {directory}")
    return directory


def _append_worker_diagnostic(
    diagnostics: WorkerDiagnostics,
    event: str,
    **fields: Any,
) -> None:
    """Append and fsync one compact JSONL worker event.

    A worker process is the sole writer of its PID-named file.  O_APPEND keeps
    repeated dataset assignments to the same ProcessPool worker ordered, while
    O_NOFOLLOW prevents a diagnostic file from being redirected through a
    symlink if a supposedly fresh collection stage was tampered with.
    """

    payload = (
        json.dumps(
            {
                "schema": WORKER_DIAGNOSTIC_VERSION,
                "event": event,
                "recorded_at_unix_ns": time.time_ns(),
                "worker_pid": diagnostics.worker_pid,
                "dataset_id": diagnostics.dataset_id,
                "events_path": str(
                    Path(WORKER_DIAGNOSTIC_DIRECTORY) / diagnostics.events_path.name
                ),
                "faulthandler_path": str(
                    Path(WORKER_DIAGNOSTIC_DIRECTORY)
                    / diagnostics.faulthandler_path.name
                ),
                **fields,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(diagnostics.events_path, flags, 0o600)
    with os.fdopen(descriptor, "ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def _begin_worker_diagnostics(
    output_root: Path,
    dataset_id: str,
) -> tuple[WorkerDiagnostics, Any]:
    """Durably bind a worker PID to its dataset before loading native code."""

    worker_pid = os.getpid()
    directory = _worker_diagnostic_directory(output_root)
    diagnostics = WorkerDiagnostics(
        worker_pid=worker_pid,
        dataset_id=dataset_id,
        events_path=directory / f"worker-{worker_pid}.events.jsonl",
        faulthandler_path=directory / f"worker-{worker_pid}.faulthandler.log",
    )
    _append_worker_diagnostic(
        diagnostics,
        "worker_task_start",
        process_name=multiprocessing.current_process().name,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(diagnostics.faulthandler_path, flags, 0o600)
    handle = os.fdopen(descriptor, "ab", buffering=0)
    enabled = False
    try:
        header = (
            json.dumps(
                {
                    "schema": WORKER_DIAGNOSTIC_VERSION,
                    "event": "faulthandler_enabled",
                    "recorded_at_unix_ns": time.time_ns(),
                    "worker_pid": worker_pid,
                    "dataset_id": dataset_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        handle.write(header)
        os.fsync(handle.fileno())
        faulthandler.enable(file=handle, all_threads=True)
        enabled = True
        _append_worker_diagnostic(diagnostics, "faulthandler_enabled")
    except BaseException:
        if enabled:
            faulthandler.disable()
        handle.close()
        raise
    return diagnostics, handle


def _finish_worker_diagnostics(handle: Any) -> None:
    """Stop faulthandler and durably flush a normally exiting worker task."""

    try:
        faulthandler.disable()
    finally:
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()


def _source_category_counts(plan: policy_core.SourcePlan) -> Mapping[str, int]:
    """Summarize the frozen schema categories without querying the source."""

    counts: Counter[str] = Counter()
    for field in plan.fields:
        temporal = field.temporal
        counts[
            str(temporal.source_category) if temporal is not None else "FIELD"
        ] += 1
    for dropped in plan.dropped_columns:
        counts[str(dropped.get("category", "UNKNOWN"))] += 1
    return {name: counts[name] for name in sorted(counts)}


def _record_worker_source_event(
    diagnostics: WorkerDiagnostics,
    event: str,
    plan: policy_core.SourcePlan,
    *,
    source_ordinal: int,
    **fields: Any,
) -> Mapping[str, Any]:
    """Fsync an exact PID/dataset/source binding and return its log fields."""

    metadata = {
        "worker_pid": diagnostics.worker_pid,
        "source_ordinal": source_ordinal,
        "source_path": plan.source_path,
        "source_sha256": plan.expected_sha256,
        "source_schema_sha256": plan.schema_sha256,
        "table_name": plan.table_name,
        "source_action": plan.source_action,
        "source_category_counts": _source_category_counts(plan),
        **fields,
    }
    _append_worker_diagnostic(diagnostics, event, **metadata)
    return metadata


class _WorkerSourceHeartbeat:
    """Rate-limit durable in-source progress without changing query batches."""

    def __init__(
        self,
        diagnostics: WorkerDiagnostics | None,
        plan: policy_core.SourcePlan,
        source_ordinal: int,
    ) -> None:
        self.diagnostics = diagnostics
        self.plan = plan
        self.source_ordinal = source_ordinal
        self.last_fsync = time.monotonic()

    def record(
        self,
        phase: str,
        *,
        force: bool = False,
        source_rows: int | None = None,
        eligible_rows: int | None = None,
        read_mode: str | None = None,
    ) -> None:
        if self.diagnostics is None:
            return
        now = time.monotonic()
        if not force and now - self.last_fsync < WORKER_DIAGNOSTIC_HEARTBEAT_SECONDS:
            return
        fields: dict[str, Any] = {"phase": phase}
        if source_rows is not None:
            fields["source_rows"] = source_rows
        if eligible_rows is not None:
            fields["eligible_rows"] = eligible_rows
        if read_mode is not None:
            fields["read_mode"] = read_mode
        _record_worker_source_event(
            self.diagnostics,
            "source_phase",
            self.plan,
            source_ordinal=self.source_ordinal,
            **fields,
        )
        self.last_fsync = now


def _remove_success_worker_diagnostics(stage: Path) -> None:
    """Keep the published CSV whitelist unchanged after a successful run."""

    directory = stage / WORKER_DIAGNOSTIC_DIRECTORY
    if not directory.exists() and not directory.is_symlink():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise CsvExportError("worker diagnostic directory changed before publication")
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file():
            raise CsvExportError(f"unexpected worker diagnostic object: {child}")
        child.unlink()
    directory.rmdir()


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacement on Linux/macOS."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    system = platform.system()
    if system == "Linux" and hasattr(libc, "renameat2"):
        # AT_FDCWD=-100, RENAME_NOREPLACE=1.
        result = libc.renameat2(
            ctypes.c_int(-100),
            ctypes.c_char_p(source_bytes),
            ctypes.c_int(-100),
            ctypes.c_char_p(destination_bytes),
            ctypes.c_uint(1),
        )
    elif system == "Darwin" and hasattr(libc, "renamex_np"):
        # RENAME_EXCL=0x4.
        result = libc.renamex_np(
            ctypes.c_char_p(source_bytes),
            ctypes.c_char_p(destination_bytes),
            ctypes.c_uint(0x4),
        )
    else:
        raise CsvExportError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CsvExportError(f"refusing to overwrite existing output root: {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _load_tsfile_module() -> Any:
    try:
        installed = package_version("tsfile")
    except PackageNotFoundError as error:
        raise CsvExportError("the pinned tsfile package is not installed") from error
    if installed != REQUIRED_TSFILE_VERSION:
        raise CsvExportError(
            f"tsfile version drift: required {REQUIRED_TSFILE_VERSION}, found {installed}"
        )
    try:
        import tsfile  # type: ignore
    except Exception as error:  # pragma: no cover - import environment dependent
        raise CsvExportError(f"cannot import tsfile {installed}: {error}") from error
    return tsfile


def _africa_query_exception(
    plan: policy_core.SourcePlan,
    *,
    verified_source_rows: int | None = None,
) -> Mapping[str, Any] | None:
    """Return the one hash-bound projection exception, or no exception.

    The frozen source path alone is intentionally insufficient: many datasets
    use generic ``train/train.tsfile`` names.  The content hash selects the
    exception, the frozen schema metadata proves that only ordinary dropped
    STRING FIELDs are omitted, and the completed query must reproduce the
    frozen decoded-row count.
    """

    key = (
        str(getattr(plan, "source_path", "")),
        str(getattr(plan, "expected_sha256", "")).lower(),
    )
    expected_rows = AFRICA_QUERY_EXCEPTIONS.get(key)
    if expected_rows is None:
        return None
    expected_table = AFRICA_QUERY_EXCEPTION_TABLES[key]
    if plan.table_name != expected_table:
        raise CsvExportError(
            f"{plan.source_path}: frozen Africa projection exception table drift: "
            f"actual={plan.table_name!r}, expected={expected_table!r}"
        )

    retained_names = {field.name for field in plan.fields}
    tag_names = set(plan.tag_names)
    dropped_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for item in plan.dropped_columns:
        name = item.get("name")
        if isinstance(name, str):
            dropped_by_name.setdefault(name, []).append(item)
    schema_rows: list[dict[str, str]] = []
    for name in AFRICA_OMITTED_QUERY_COLUMNS:
        definitions = dropped_by_name.get(name, [])
        if len(definitions) != 1 or name in retained_names or name in tag_names:
            raise CsvExportError(
                f"{plan.source_path}: frozen Africa projection exception schema drift "
                f"for {name!r}"
            )
        item = definitions[0]
        dtype = str(item.get("tsfile_type", "")).upper()
        category = str(item.get("category", "")).upper()
        decision = str(item.get("decision", ""))
        if (dtype, category, decision) != (
            "STRING",
            "FIELD",
            "drop_non_numeric_field",
        ):
            raise CsvExportError(
                f"{plan.source_path}: frozen Africa projection exception may omit only "
                f"ordinary STRING FIELDs, got {name!r}=({dtype!r}, {category!r}, "
                f"{decision!r})"
            )
        schema_rows.append(
            {
                "name": name,
                "tsfile_type": dtype,
                "category": category,
                "decision": decision,
            }
        )
    if verified_source_rows is not None and verified_source_rows != expected_rows:
        raise CsvExportError(
            f"{plan.source_path}: frozen Africa row-coverage count mismatch: "
            f"actual={verified_source_rows}, expected={expected_rows}"
        )
    return {
        "exception_id": AFRICA_QUERY_EXCEPTION_ID,
        "source_path": plan.source_path,
        "source_sha256": plan.expected_sha256,
        "expected_source_rows": expected_rows,
        "verified_source_rows": verified_source_rows,
        "omitted_query_columns": list(AFRICA_OMITTED_QUERY_COLUMNS),
        "omitted_query_column_schema": schema_rows,
        "omitted_column_contract": "ordinary_STRING_FIELD_drop_non_numeric_field",
        "row_coverage_proof": "frozen_source_sha256_and_exact_source_row_count",
    }


def _all_query_columns(plan: policy_core.SourcePlan) -> list[str]:
    """Project every schema TAG/FIELD/ATTRIBUTE so sparse rows cannot disappear.

    ``SourcePlan.fields`` contains emitted numeric/temporal values, while
    ``dropped_columns`` supplies the remaining schema columns.  Physical Time
    is implicit and must not be listed in ``query_table``.
    """

    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name != "Time" and name not in seen:
            seen.add(name)
            ordered.append(name)

    for name in plan.tag_names:
        add(name)
    for field in plan.fields:
        add(field.name)
    for item in plan.dropped_columns:
        name = item.get("name")
        category = str(item.get("category", "")).upper()
        if category not in {"TAG", "FIELD", "ATTRIBUTE"}:
            raise CsvExportError(
                f"{plan.source_path}: unsupported schema category for row-coverage "
                f"projection: {category!r}"
            )
        if isinstance(name, str):
            add(name)
    if not ordered:
        raise CsvExportError(
            f"{plan.source_path}: table query has no schema columns to anchor row coverage"
        )
    exception = _africa_query_exception(plan)
    if exception is not None:
        omitted = set(AFRICA_OMITTED_QUERY_COLUMNS)
        ordered = [name for name in ordered if name not in omitted]
        if not ordered:
            raise CsvExportError(
                f"{plan.source_path}: Africa projection exception removed every query column"
            )
    return ordered


def _temporal_conversion_label(temporal: Any) -> str:
    source_bound_type = getattr(policy_core, "SourceBoundTemporalField", ())
    if source_bound_type and isinstance(temporal, source_bound_type):
        return "policy_source_bound_integer_epoch_to_int64_us"
    return "policy_temporal_string_to_int64_us"


def _output_columns(plans: Sequence[policy_core.SourcePlan]) -> tuple[OutputColumn, ...]:
    logical: dict[str, tuple[str, str, str, str, str]] = {}
    for plan in plans:
        if plan.source_action != "materialize":
            continue
        for field in plan.fields:
            dtype = field.dtype
            conversion = (
                _temporal_conversion_label(field.temporal)
                if field.temporal is not None
                else "identity"
            )
            key = _canonical_json(
                [plan.table_name, field.name, field.tsfile_type, conversion]
            )
            logical[key] = (
                plan.table_name,
                field.name,
                field.tsfile_type,
                conversion,
                dtype,
            )
    if not logical:
        raise CsvExportError("dataset has no retained numeric or approved temporal value column")
    ordered = sorted(
        logical.items(),
        key=lambda item: (1 if item[1][3] != "identity" else 0, item[0]),
    )
    return tuple(
        OutputColumn(
            header=f"c{index:04d}",
            logical_key=key,
            table_name=value[0],
            source_name=value[1],
            source_type=value[2],
            conversion=value[3],
            dtype=value[4],
        )
        for index, (key, value) in enumerate(ordered)
    )


def _column_by_logical_key(columns: Sequence[OutputColumn]) -> dict[str, OutputColumn]:
    return {column.logical_key: column for column in columns}


def _field_logical_key(plan: policy_core.SourcePlan, field: policy_core.FieldSpec) -> str:
    conversion = (
        _temporal_conversion_label(field.temporal)
        if field.temporal is not None
        else "identity"
    )
    return _canonical_json([plan.table_name, field.name, field.tsfile_type, conversion])


def _open_spool(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE rows (
            timestamp_unix_us INTEGER NOT NULL,
            source_ordinal INTEGER NOT NULL,
            source_row_ordinal INTEGER NOT NULL,
            entity_key TEXT NOT NULL,
            values_json TEXT NOT NULL,
            PRIMARY KEY (source_ordinal, source_row_ordinal)
        );
        CREATE TABLE entities (
            entity_key TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            tag_names_json TEXT NOT NULL,
            typed_tag_values_json TEXT NOT NULL
        );
        CREATE TABLE entity_sources (
            entity_key TEXT NOT NULL,
            source_path TEXT NOT NULL,
            PRIMARY KEY (entity_key, source_path)
        );
        CREATE TABLE series_ids (
            entity_key TEXT PRIMARY KEY,
            series_id INTEGER NOT NULL UNIQUE
        );
        """
    )
    return connection


def _metadata_layout(result: Any) -> tuple[list[str], list[str], dict[str, int]]:
    metadata = result.get_metadata()
    names = [str(value) for value in metadata.get_column_list()]
    types = [
        str(getattr(value, "name", value)).upper()
        for value in metadata.get_data_type_list()
    ]
    if not names or names[0].casefold() != "time" or types[0] != "INT64":
        raise CsvExportError("query result must begin with the implicit INT64 Time column")
    if len(names) != len(set(names)):
        raise CsvExportError(f"query result contains duplicate column names: {names!r}")
    return names, types, {name: index for index, name in enumerate(names)}


def _is_dataframe_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        result = pd.isna(value)
        if getattr(result, "shape", ()) == ():
            return bool(result)
        return False
    except Exception:
        return False


def _is_ieee_nan(value: Any) -> bool:
    """Recognize a numeric NaN without treating the literal string ``nan`` as one."""

    if isinstance(value, (str, bytes, bytearray, bool)):
        return False
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _normalize_numeric(value: Any, dtype: str, column: str) -> int | float | str:
    try:
        if dtype == "INT32":
            integer = int(value)
            if integer < -(1 << 31) or integer > (1 << 31) - 1:
                raise OverflowError
            return integer
        if dtype == "INT64":
            integer = int(value)
            if integer < INT64_MIN or integer > INT64_MAX:
                raise OverflowError
            return integer
        if dtype in {"FLOAT", "DOUBLE"}:
            numeric = float(value)
            if math.isnan(numeric):
                if not _is_ieee_nan(value):
                    raise CsvExportError(
                        f"non-numeric NaN spelling for {column!r} is not approved"
                    )
                # JSON itself has no portable NaN representation.  Keep an
                # exact, otherwise-invalid string sentinel in the spool and
                # emit the approved lowercase CSV token only at formatting.
                return NAN_SPOOL_TOKEN
            if numeric == -math.inf:
                raise CsvExportError(
                    f"negative infinity for {column!r} is not approved by the CSV contract"
                )
            if numeric == math.inf:
                # The source/path/field/count authorization is checked after
                # the complete source scan.  A JSON-safe token prevents
                # Python's non-standard Infinity literal entering the spool.
                return POSITIVE_INFINITY_SPOOL_TOKEN
            return numeric
    except (TypeError, ValueError, OverflowError) as error:
        raise CsvExportError(
            f"value for {column!r} cannot be represented as {dtype}: {value!r}"
        ) from error
    raise CsvExportError(f"unsupported retained value type for {column!r}: {dtype}")


def _check_positive_infinity_contract(
    plan: policy_core.SourcePlan,
    counts: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    """Admit exactly the user-approved 58 Africa ``z_gpp`` +Inf cells."""

    key = (
        str(getattr(plan, "source_path", "")),
        str(getattr(plan, "expected_sha256", "")).lower(),
    )
    expected = dict(AFRICA_POSITIVE_INFINITY_COUNTS.get(key, {}))
    actual = {
        str(name): int(count)
        for name, count in counts.items()
        if int(count) != 0
    }
    if actual != expected:
        raise CsvExportError(
            f"{plan.source_path}: positive-infinity contract mismatch: "
            f"actual={actual}, expected={expected}; only the exact frozen Africa "
            "z_gpp cells are approved"
        )
    return tuple(sorted(actual.items()))


def _row_filter_keeps(
    plan: policy_core.SourcePlan,
    raw_by_name: Mapping[str, Any],
    null_by_name: Mapping[str, bool],
) -> bool:
    if plan.row_filter is None:
        return True
    for condition in plan.row_filter.conditions:
        name = condition.source_name
        if null_by_name[name]:
            return False
        if condition.operator == "not_null":
            continue
        if condition.operator == "int64_gte":
            assert condition.value is not None
            if int(raw_by_name[name]) < condition.value:
                return False
            continue
        raise CsvExportError(f"unsupported frozen row filter operator: {condition.operator}")
    return True


def _row_exception_source(
    policy: RowExceptionPolicy | None,
    binding: policy_core.DatasetBinding,
    plan: policy_core.SourcePlan,
) -> RowExceptionSource | None:
    if policy is None:
        return None
    if binding.row.dataset_id != policy.dataset_id:
        return None
    source = policy.sources_by_path.get(plan.source_path)
    if source is None:
        return None
    if (
        source.source_sha256 != plan.expected_sha256
        or source.table_name != plan.table_name
    ):
        raise CsvExportError(
            f"{plan.source_path}: row-exception source hash/table binding mismatch"
        )
    return source


def _row_exception_drops(
    plan: policy_core.SourcePlan,
    exception: RowExceptionSource | None,
    tracker: _RowExceptionTracker | None,
    *,
    source_row_ordinal: int,
    names: Sequence[str],
    types: Sequence[str],
    raw_values: Sequence[Any],
    null_flags: Sequence[bool],
) -> bool:
    """Match and prove the approved invalid-time row before time conversion."""

    if exception is None:
        return False
    if tracker is None:
        raise AssertionError("row-exception source requires a tracker")
    physical_time = raw_values[0]
    if source_row_ordinal != 0 and not null_flags[0] and int(physical_time) == INT64_MIN:
        raise CsvExportError(
            f"{plan.source_path}: unexpected INT64_MIN Time at source row "
            f"{source_row_ordinal}"
        )
    if source_row_ordinal != 0 or null_flags[0] or int(physical_time) != INT64_MIN:
        return False
    if names[0].casefold() != "time" or types[0] != "INT64":
        raise CsvExportError(f"{plan.source_path}: row exception requires physical INT64 Time")
    if plan.tag_names or plan.dropped_columns:
        raise CsvExportError(
            f"{plan.source_path}: row exception requires no TAG, ATTRIBUTE, or dropped fields"
        )
    expected_names = [field.name for field in plan.fields]
    if list(names[1:]) != expected_names or list(types[1:]) != ["FLOAT"] * len(expected_names):
        raise CsvExportError(
            f"{plan.source_path}: row-exception full-schema FLOAT FIELD projection drift"
        )
    if not all(null_flags[index] for index in range(1, len(names))):
        populated = [
            names[index]
            for index in range(1, len(names))
            if not null_flags[index]
        ]
        raise CsvExportError(
            f"{plan.source_path}: approved invalid-time row contains payload in "
            f"{populated[:8]}"
        )
    if tracker.dropped_source_row_ordinals:
        raise CsvExportError(f"{plan.source_path}: row exception matched more than once")
    tracker.dropped_source_row_ordinals.append(source_row_ordinal)
    tracker.physical_time = int(physical_time)
    tracker.all_value_fields_null = True
    return True


def _prepare_spool_row(
    plan: policy_core.SourcePlan,
    *,
    source_ordinal: int,
    source_row_ordinal: int,
    source_path: str,
    names: Sequence[str],
    types: Sequence[str],
    raw_values: Sequence[Any],
    null_flags: Sequence[bool],
    column_by_key: Mapping[str, OutputColumn],
    temporal_cache: _BoundedCache,
    entity_cache: _BoundedCache,
    positive_infinity_counts: Counter[str] | None = None,
    nan_counts: Counter[str] | None = None,
    row_exception: RowExceptionSource | None = None,
    row_exception_tracker: _RowExceptionTracker | None = None,
    temporal_empty_string_tracker: _TemporalEmptyStringTracker | None = None,
    filtered_counts: Counter[str] | None = None,
) -> tuple[
    tuple[int, int, int, str, str] | None,
    tuple[str, str, str, str] | None,
    tuple[str, str] | None,
]:
    if len(raw_values) != len(names) or len(null_flags) != len(names):
        raise CsvExportError("query row width differs from query metadata")
    raw = dict(zip(names, raw_values))
    nulls = dict(zip(names, null_flags))
    # The SDK commonly reports the implicit coordinate as lowercase ``time``
    # while frozen adapter policies name it ``Time``.
    raw["Time"] = raw_values[0]
    nulls["Time"] = null_flags[0]
    _observe_temporal_empty_strings(
        temporal_empty_string_tracker,
        source_row_ordinal=source_row_ordinal,
        names=names,
        types=types,
        raw_values=raw_values,
        null_flags=null_flags,
    )
    # The exact row API is the authority for NULL versus IEEE NaN.  NaN is
    # meaningful only in true FLOAT/DOUBLE fields; it must never silently
    # become a timestamp, integer, TAG identity, or ordinary string value.
    for index, (name, dtype) in enumerate(zip(names, types)):
        if null_flags[index] or not _is_ieee_nan(raw_values[index]):
            continue
        if (
            index == 0
            or name in plan.tag_names
            or dtype not in {"FLOAT", "DOUBLE"}
        ):
            raise CsvExportError(
                f"NaN value for non-floating payload column {name!r} "
                f"({dtype}) is not approved"
            )
    if _row_exception_drops(
        plan,
        row_exception,
        row_exception_tracker,
        source_row_ordinal=source_row_ordinal,
        names=names,
        types=types,
        raw_values=raw_values,
        null_flags=null_flags,
    ):
        if filtered_counts is not None:
            filtered_counts["row_exception"] += 1
        return None, None, None
    if not _row_filter_keeps(plan, raw, nulls):
        if filtered_counts is not None:
            filtered_counts["timestamp_policy"] += 1
        return None, None, None

    if plan.primary_temporal is None:
        if nulls[names[0]]:
            raise CsvExportError("physical Time cannot be null")
        assert plan.timestamp_source_unit is not None
        assert plan.timestamp_output_unit is not None
        timestamp = policy_core._convert_timestamp(
            raw[names[0]], plan.timestamp_source_unit, plan.timestamp_output_unit
        )
    else:
        source_name = plan.primary_temporal.source_name
        if nulls.get(source_name, True):
            raise CsvExportError(f"primary temporal field {source_name!r} is null")
        temporal = plan.primary_temporal
        cache_key = (_field_logical_key(plan, next(
            field for field in plan.fields if field.name == temporal.source_name
        )), str(raw[source_name]))
        timestamp = temporal_cache.get_or_create(
            cache_key,
            lambda: policy_core._convert_temporal_value(raw[source_name], temporal),
        )

    tag_cache_parts: list[tuple[str, bool, Any]] = []
    for tag_name in plan.tag_names:
        tag_type = types[names.index(tag_name)]
        is_null = nulls.get(tag_name, True)
        if is_null:
            # ``TYPE:null`` is an explicit typed null identity, distinct from
            # both an absent tag and the literal string "null" (STRING:"null").
            tag_cache_parts.append((tag_type, True, None))
            continue
        tag_value = raw[tag_name]
        try:
            hash(tag_value)
            hashable_value = tag_value
        except TypeError:
            hashable_value = repr(tag_value)
        tag_cache_parts.append((tag_type, False, hashable_value))
    entity_cache_key = (plan.table_name, tuple(tag_cache_parts))

    def build_entity() -> tuple[str, list[str]]:
        typed = [
            (
                f"{tag_type}:null"
                if is_null
                else _typed_entity_component(tag_type, raw[tag_name])
            )
            for tag_name, (tag_type, is_null, _value) in zip(
                plan.tag_names, tag_cache_parts
            )
        ]
        if not typed:
            typed = ["SINGLE:__single_entity__"]
        return _canonical_json([plan.table_name, typed]), typed

    entity_key, typed_tags = entity_cache.get_or_create(entity_cache_key, build_entity)

    # Sparse JSON is deliberate: an absent output key means CSV null/empty.
    # Repeating thousands of null cNNNN keys per row would make wide sparse
    # datasets (for example macroeconomic panels) needlessly explode on disk.
    values: dict[str, int | float | str] = {}
    for field in plan.fields:
        output_name = column_by_key[_field_logical_key(plan, field)].header
        if nulls.get(field.name, True):
            continue
        elif field.temporal is not None:
            if _is_approved_temporal_empty_string(
                temporal_empty_string_tracker,
                field.name,
                raw[field.name],
            ):
                # The exact source-row ordinal was recorded above.  Omitting
                # this sparse JSON key intentionally emits a CSV empty field;
                # the receipt keeps it distinguishable from source NULL.
                continue
            temporal_key = (_field_logical_key(plan, field), str(raw[field.name]))
            values[output_name] = temporal_cache.get_or_create(
                temporal_key,
                lambda: policy_core._convert_temporal_value(
                    raw[field.name], field.temporal
                ),
            )
        else:
            normalized = _normalize_numeric(
                raw[field.name], field.tsfile_type, field.name
            )
            values[output_name] = normalized
            if normalized == POSITIVE_INFINITY_SPOOL_TOKEN:
                if positive_infinity_counts is None:
                    raise CsvExportError(
                        "positive infinity encountered without a source-bound counter"
                    )
                positive_infinity_counts[field.name] += 1
            elif normalized == NAN_SPOOL_TOKEN:
                if nan_counts is None:
                    raise CsvExportError("NaN encountered without a source-bound counter")
                nan_counts[field.name] += 1
    values_json = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    row = (timestamp, source_ordinal, source_row_ordinal, entity_key, values_json)
    entity = (
        entity_key,
        plan.table_name,
        _canonical_json(list(plan.tag_names)),
        _canonical_json(typed_tags),
    )
    return row, entity, (entity_key, source_path)


class _BufferedSpoolWriter:
    def __init__(
        self,
        connection: sqlite3.Connection,
        batch_rows: int,
        disk_guard: _FreeDiskGuard | None = None,
    ) -> None:
        if batch_rows <= 0:
            raise CsvExportError("insert batch size must be positive")
        self.connection = connection
        self.batch_rows = batch_rows
        self.disk_guard = disk_guard
        self.rows: list[tuple[int, int, int, str, str]] = []
        self.entities: list[tuple[str, str, str, str]] = []
        self.sources: list[tuple[str, str]] = []
        self.seen_entities: OrderedDict[str, None] = OrderedDict()
        self.seen_entity_sources: OrderedDict[tuple[str, str], None] = OrderedDict()

    @staticmethod
    def _remember(cache: OrderedDict[Any, None], key: Any) -> bool:
        if key in cache:
            cache.move_to_end(key)
            return False
        cache[key] = None
        if len(cache) > DEFAULT_ENTITY_INSERT_CACHE_SIZE:
            cache.popitem(last=False)
        return True

    def add(
        self,
        row: tuple[int, int, int, str, str],
        entity: tuple[str, str, str, str],
        source: tuple[str, str],
    ) -> None:
        self.rows.append(row)
        if self._remember(self.seen_entities, entity[0]):
            self.entities.append(entity)
        if self._remember(self.seen_entity_sources, source):
            self.sources.append(source)
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.connection.executemany(
            "INSERT INTO rows VALUES (?, ?, ?, ?, ?)", self.rows
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)", self.entities
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO entity_sources VALUES (?, ?)", self.sources
        )
        self.rows.clear()
        self.entities.clear()
        self.sources.clear()
        if self.disk_guard is not None:
            self.disk_guard.check("SQLite spool batch")


def _query_result(reader: Any, plan: policy_core.SourcePlan) -> Any:
    return reader.query_table(
        plan.table_name,
        _all_query_columns(plan),
        INT64_MIN,
        INT64_MAX,
    )


def _source_read_mode(plan: policy_core.SourcePlan) -> str:
    binding = (
        plan.source_path,
        plan.expected_sha256,
        getattr(plan, "schema_sha256", None),
        plan.table_name,
    )
    if binding == FORBUSH_DIRECT_ROWWISE_SOURCE:
        return READ_MODE_ROWWISE_DIRECT
    guarded_path, _guarded_sha, _guarded_schema, guarded_table = (
        FORBUSH_DIRECT_ROWWISE_SOURCE
    )
    if plan.source_path == guarded_path or plan.table_name == guarded_table:
        names = ("source_path", "source_sha256", "source_schema_sha256", "table_name")
        mismatches = [
            name
            for name, actual, expected in zip(
                names, binding, FORBUSH_DIRECT_ROWWISE_SOURCE
            )
            if actual != expected
        ]
        raise CsvExportError(
            "guarded Forbush direct-rowwise source binding mismatch: "
            + ", ".join(mismatches)
        )
    return READ_MODE_DATAFRAME


def _forbush_temporal_empty_string_tracker(
    binding: policy_core.DatasetBinding,
    plan: policy_core.SourcePlan,
    column_by_key: Mapping[str, OutputColumn],
    read_mode: str,
) -> _TemporalEmptyStringTracker | None:
    """Bind the sole approved temporal empty-string exception exactly.

    Merely sharing a source/table name is intentionally insufficient.  The
    native-crash workaround has already selected the hash- and schema-bound
    direct-rowwise mode; this additional guard pins the dataset and complete
    retained schema before an empty value can bypass the ordinary parser.
    """

    if read_mode != READ_MODE_ROWWISE_DIRECT:
        return None
    dataset_id = getattr(getattr(binding, "row", None), "dataset_id", None)
    if dataset_id != FORBUSH_DATASET_ID:
        raise CsvExportError(
            "guarded Forbush temporal-empty source has an unexpected dataset binding"
        )
    if (
        plan.primary_temporal is not None
        or plan.timestamp_source_unit != "ms"
        or plan.timestamp_output_unit != "us"
        or plan.row_filter is not None
        or plan.tag_names
        or len(plan.fields) != 23
    ):
        raise CsvExportError(
            f"{plan.source_path}: guarded Forbush temporal-empty plan shape drift"
        )
    # The core historically names this provenance collection
    # ``dropped_columns`` even when a STRING field is retained through an
    # approved temporal conversion.  Accept only the two exact frozen
    # conversion annotations; neither an omitted annotation nor a real drop
    # receives the empty-string exception.
    annotations = tuple(dict(item) for item in plan.dropped_columns)
    if annotations != FORBUSH_TEMPORAL_CONVERSION_ANNOTATIONS:
        raise CsvExportError(
            f"{plan.source_path}: guarded Forbush temporal conversion annotations drift"
        )

    fields_by_name = {field.name: field for field in plan.fields}
    expected_names = tuple(sorted(FORBUSH_TEMPORAL_EMPTY_STRING_EXPECTATIONS))
    actual_temporal_names = tuple(
        sorted(field.name for field in plan.fields if field.temporal is not None)
    )
    if actual_temporal_names != expected_names:
        raise CsvExportError(
            f"{plan.source_path}: guarded Forbush temporal field set mismatch: "
            f"actual={actual_temporal_names!r}, expected={expected_names!r}"
        )
    for source_name in expected_names:
        field = fields_by_name.get(source_name)
        if field is None:
            raise CsvExportError(
                f"{plan.source_path}: missing guarded temporal field {source_name!r}"
            )
        temporal = field.temporal
        if (
            field.tsfile_type != "STRING"
            or not isinstance(temporal, policy_core.TemporalStringField)
            or temporal.source_name != source_name
            or temporal.source_type != "STRING"
            or temporal.source_category != "FIELD"
            or temporal.parse_kind != "strptime"
            or temporal.datetime_format != "%Y-%m-%d %H:%M:%S"
            or temporal.timezone_name != "UTC"
            or _temporal_conversion_label(temporal)
            != "policy_temporal_string_to_int64_us"
        ):
            raise CsvExportError(
                f"{plan.source_path}: guarded temporal rule drift for {source_name!r}"
            )
        output_column = column_by_key.get(_field_logical_key(plan, field))
        expected_output = FORBUSH_TEMPORAL_EMPTY_STRING_OUTPUT_COLUMNS[source_name]
        if output_column is None or output_column.header != expected_output:
            raise CsvExportError(
                f"{plan.source_path}: guarded temporal output column mismatch for "
                f"{source_name!r}: actual="
                f"{None if output_column is None else output_column.header!r}, "
                f"expected={expected_output!r}"
            )
    return _TemporalEmptyStringTracker(
        field_names=expected_names,
        source_non_null_empty_row_ordinals={name: [] for name in expected_names},
        source_null_row_ordinals={name: [] for name in expected_names},
    )


def _observe_temporal_empty_strings(
    tracker: _TemporalEmptyStringTracker | None,
    *,
    source_row_ordinal: int,
    names: Sequence[str],
    types: Sequence[str],
    raw_values: Sequence[Any],
    null_flags: Sequence[bool],
) -> None:
    if tracker is None:
        return
    for source_name in tracker.field_names:
        try:
            index = names.index(source_name)
        except ValueError as error:
            raise CsvExportError(
                f"guarded Forbush query omitted temporal field {source_name!r}"
            ) from error
        if types[index] != "STRING":
            raise CsvExportError(
                f"guarded Forbush temporal field {source_name!r} changed type"
            )
        if null_flags[index]:
            tracker.source_null_row_ordinals[source_name].append(source_row_ordinal)
        elif raw_values[index] == "":
            tracker.source_non_null_empty_row_ordinals[source_name].append(
                source_row_ordinal
            )


def _is_approved_temporal_empty_string(
    tracker: _TemporalEmptyStringTracker | None,
    source_name: str,
    raw_value: Any,
) -> bool:
    return (
        tracker is not None
        and source_name in tracker.field_names
        and isinstance(raw_value, str)
        and raw_value == ""
    )


def _finalize_forbush_temporal_empty_string_audits(
    plan: policy_core.SourcePlan,
    tracker: _TemporalEmptyStringTracker | None,
    column_by_key: Mapping[str, OutputColumn],
    *,
    source_rows: int,
    eligible_rows: int,
) -> tuple[Mapping[str, Any], ...]:
    if tracker is None:
        return ()
    if source_rows != FORBUSH_SOURCE_ROWS or eligible_rows != FORBUSH_SOURCE_ROWS:
        raise CsvExportError(
            f"{plan.source_path}: guarded temporal-empty row count mismatch: "
            f"source={source_rows}, eligible={eligible_rows}, "
            f"expected={FORBUSH_SOURCE_ROWS}"
        )
    fields_by_name = {field.name: field for field in plan.fields}
    audits: list[Mapping[str, Any]] = []
    for source_name in tracker.field_names:
        empty_ordinals = tracker.source_non_null_empty_row_ordinals[source_name]
        null_ordinals = tracker.source_null_row_ordinals[source_name]
        for label, ordinals in (
            ("source_non_null_empty", empty_ordinals),
            ("source_null", null_ordinals),
        ):
            if (
                any(isinstance(value, bool) or not isinstance(value, int) for value in ordinals)
                or ordinals != sorted(ordinals)
                or len(ordinals) != len(set(ordinals))
                or any(value < 0 or value >= source_rows for value in ordinals)
            ):
                raise CsvExportError(
                    f"{plan.source_path}: invalid {label} source-row ordinals "
                    f"for {source_name!r}"
                )
        if set(empty_ordinals).intersection(null_ordinals):
            raise CsvExportError(
                f"{plan.source_path}: empty and NULL source-row ordinals overlap "
                f"for {source_name!r}"
            )
        valid_count = source_rows - len(empty_ordinals) - len(null_ordinals)
        actual_counts = {
            "source_non_null_empty_count": len(empty_ordinals),
            "source_null_count": len(null_ordinals),
            "source_valid_count": valid_count,
        }
        expected_counts = dict(
            FORBUSH_TEMPORAL_EMPTY_STRING_EXPECTATIONS[source_name]
        )
        if actual_counts != expected_counts:
            raise CsvExportError(
                f"{plan.source_path}: guarded temporal-empty counts mismatch for "
                f"{source_name!r}: actual={actual_counts}, expected={expected_counts}"
            )
        field = fields_by_name[source_name]
        output_column = column_by_key[_field_logical_key(plan, field)].header
        audits.append(
            {
                "source_name": source_name,
                "source_type": "STRING",
                "source_category": "FIELD",
                "output_column": output_column,
                "conversion": "policy_temporal_string_to_int64_us",
                "representation": FORBUSH_TEMPORAL_EMPTY_STRING_REPRESENTATION,
                "source_non_null_empty_count": len(empty_ordinals),
                "source_non_null_empty_row_ordinals": list(empty_ordinals),
                "source_null_count": len(null_ordinals),
                "source_null_row_ordinals": list(null_ordinals),
                "source_valid_count": valid_count,
            }
        )
    return tuple(audits)


def _scan_dataframe(
    result: Any,
    plan: policy_core.SourcePlan,
    *,
    source_ordinal: int,
    source_path: str,
    batch_rows: int,
    writer: _BufferedSpoolWriter,
    column_by_key: Mapping[str, OutputColumn],
    temporal_cache: _BoundedCache,
    entity_cache: _BoundedCache,
    positive_infinity_counts: Counter[str],
    nan_counts: Counter[str],
    row_exception: RowExceptionSource | None,
    row_exception_tracker: _RowExceptionTracker | None,
    filtered_counts: Counter[str],
    diagnostic_heartbeat: _WorkerSourceHeartbeat | None = None,
) -> tuple[int, int, int, int]:
    names, types, _indexes = _metadata_layout(result)
    source_rows = 0
    eligible_rows = 0
    # The outer next() is mandatory for tsfile 2.1.5.  read_data_frame includes
    # the current row and advances at most batch_rows-1 more rows internally.
    while result.next():
        try:
            frame = result.read_data_frame(batch_rows)
        except Exception as error:
            raise _DataFrameFallback(
                f"read_data_frame could not preserve this source exactly: {error}"
            ) from error
        if frame is None or len(frame) == 0:
            raise CsvExportError("read_data_frame returned empty after next() succeeded")
        frame_names = [str(value) for value in frame.columns]
        if frame_names != names:
            raise CsvExportError(
                f"data-frame columns differ from result metadata: {frame_names!r} != {names!r}"
            )
        tuples = list(frame.itertuples(index=False, name=None))
        # Pandas loses the distinction between a null DOUBLE/FLOAT and an
        # actual IEEE NaN.  Restart through is_null_by_index whenever a frame
        # contains a missing marker.  This also covers nullable integer casts
        # in SDK builds that return a frame rather than raising.
        if any(_is_dataframe_missing(value) for row in tuples for value in row):
            raise _DataFrameFallback("data frame contains null/NaN markers requiring exact API")
        for values in tuples:
            ordinal = source_rows
            source_rows += 1
            prepared = _prepare_spool_row(
                plan,
                source_ordinal=source_ordinal,
                source_row_ordinal=ordinal,
                source_path=source_path,
                names=names,
                types=types,
                raw_values=values,
                null_flags=[False] * len(values),
                column_by_key=column_by_key,
                temporal_cache=temporal_cache,
                entity_cache=entity_cache,
                positive_infinity_counts=positive_infinity_counts,
                nan_counts=nan_counts,
                row_exception=row_exception,
                row_exception_tracker=row_exception_tracker,
                filtered_counts=filtered_counts,
            )
            row, entity, source = prepared
            if row is not None:
                assert entity is not None and source is not None
                writer.add(row, entity, source)
                eligible_rows += 1
        if diagnostic_heartbeat is not None:
            diagnostic_heartbeat.record(
                "dataframe_scan",
                source_rows=source_rows,
                eligible_rows=eligible_rows,
            )
    writer.flush()
    # Any null marker forces the whole source through the exact row API, so a
    # successful dataframe-only scan necessarily observed no null TAG cells.
    return source_rows, eligible_rows, 0, 0


def _scan_rowwise(
    result: Any,
    plan: policy_core.SourcePlan,
    *,
    source_ordinal: int,
    source_path: str,
    writer: _BufferedSpoolWriter,
    column_by_key: Mapping[str, OutputColumn],
    temporal_cache: _BoundedCache,
    entity_cache: _BoundedCache,
    positive_infinity_counts: Counter[str],
    nan_counts: Counter[str],
    row_exception: RowExceptionSource | None,
    row_exception_tracker: _RowExceptionTracker | None,
    temporal_empty_string_tracker: _TemporalEmptyStringTracker | None,
    filtered_counts: Counter[str],
    diagnostic_heartbeat: _WorkerSourceHeartbeat | None = None,
) -> tuple[int, int, int, int]:
    names, types, _indexes = _metadata_layout(result)
    source_rows = 0
    eligible_rows = 0
    null_tag_rows = 0
    null_tag_cells = 0
    while result.next():
        nulls = [bool(result.is_null_by_index(index + 1)) for index in range(len(names))]
        values = [
            None if nulls[index] else result.get_value_by_index(index + 1)
            for index in range(len(names))
        ]
        tag_null_count = sum(
            1 for tag_name in plan.tag_names if nulls[names.index(tag_name)]
        )
        if tag_null_count:
            null_tag_rows += 1
            null_tag_cells += tag_null_count
        prepared = _prepare_spool_row(
            plan,
            source_ordinal=source_ordinal,
            source_row_ordinal=source_rows,
            source_path=source_path,
            names=names,
            types=types,
            raw_values=values,
            null_flags=nulls,
            column_by_key=column_by_key,
            temporal_cache=temporal_cache,
            entity_cache=entity_cache,
            positive_infinity_counts=positive_infinity_counts,
            nan_counts=nan_counts,
            row_exception=row_exception,
            row_exception_tracker=row_exception_tracker,
            temporal_empty_string_tracker=temporal_empty_string_tracker,
            filtered_counts=filtered_counts,
        )
        source_rows += 1
        row, entity, source = prepared
        if row is not None:
            assert entity is not None and source is not None
            writer.add(row, entity, source)
            eligible_rows += 1
        if (
            diagnostic_heartbeat is not None
            and source_rows % DEFAULT_BATCH_ROWS == 0
        ):
            diagnostic_heartbeat.record(
                "rowwise_scan",
                source_rows=source_rows,
                eligible_rows=eligible_rows,
            )
    writer.flush()
    return source_rows, eligible_rows, null_tag_rows, null_tag_cells


def _finalize_row_exception_source(
    plan: policy_core.SourcePlan,
    exception: RowExceptionSource | None,
    tracker: _RowExceptionTracker | None,
    *,
    source_rows: int,
    eligible_rows: int,
    timestamp_policy_filtered_rows: int,
    row_exception_filtered_rows: int,
) -> Mapping[str, Any] | None:
    if exception is None:
        if row_exception_filtered_rows:
            raise CsvExportError(
                f"{plan.source_path}: row-exception count appeared without a policy"
            )
        return None
    if tracker is None:
        raise AssertionError("row-exception source requires a tracker")
    observed_dropped_rows = len(tracker.dropped_source_row_ordinals)
    expected = (
        exception.expected_source_rows,
        exception.expected_eligible_rows,
        exception.expected_dropped_rows,
    )
    observed = (source_rows, eligible_rows, row_exception_filtered_rows)
    if observed != expected:
        raise CsvExportError(
            f"{plan.source_path}: row-exception source counts mismatch: "
            f"actual={observed}, expected={expected}"
        )
    if timestamp_policy_filtered_rows != 0:
        raise CsvExportError(
            f"{plan.source_path}: approved macroecon source unexpectedly used a "
            "timestamp policy row filter"
        )
    if (
        observed_dropped_rows != exception.expected_dropped_rows
        or tracker.dropped_source_row_ordinals != [0]
        or tracker.physical_time != INT64_MIN
        or not tracker.all_value_fields_null
    ):
        raise CsvExportError(
            f"{plan.source_path}: row-exception predicate proof is incomplete"
        )
    return {
        "exception_id": MACROECON_ROW_EXCEPTION_ID,
        "source_path": exception.source_path,
        "source_sha256": exception.source_sha256,
        "table_name": exception.table_name,
        "expected_source_rows": exception.expected_source_rows,
        "observed_source_rows": source_rows,
        "expected_eligible_rows": exception.expected_eligible_rows,
        "observed_eligible_rows": eligible_rows,
        "expected_dropped_rows": exception.expected_dropped_rows,
        "observed_dropped_rows": row_exception_filtered_rows,
        "dropped_source_row_ordinals": list(tracker.dropped_source_row_ordinals),
        "physical_time": tracker.physical_time,
        "all_value_fields_null": tracker.all_value_fields_null,
    }


def _scan_source(
    connection: sqlite3.Connection,
    binding: policy_core.DatasetBinding,
    plan: policy_core.SourcePlan,
    *,
    source_ordinal: int,
    batch_rows: int,
    insert_batch_rows: int,
    tsfile_module: Any,
    column_by_key: Mapping[str, OutputColumn],
    disk_guard: _FreeDiskGuard | None = None,
    row_exception_policy: RowExceptionPolicy | None = None,
    worker_diagnostics: WorkerDiagnostics | None = None,
) -> ScanStats:
    heartbeat = _WorkerSourceHeartbeat(worker_diagnostics, plan, source_ordinal)
    read_mode = _source_read_mode(plan)
    if plan.source_action != "materialize":
        if read_mode == READ_MODE_ROWWISE_DIRECT:
            raise CsvExportError(
                "guarded Forbush source must remain a materialized full-row scan"
            )
        heartbeat.record("source_excluded", force=True)
        return ScanStats(
            0,
            0,
            0,
            False,
            False,
            0,
            0,
            (),
            read_mode=READ_MODE_NOT_MATERIALIZED,
        )
    heartbeat.record("read_mode_selected", force=True, read_mode=read_mode)
    heartbeat.record("source_hash_start", force=True)
    if _sha256(plan.path) != plan.expected_sha256:
        raise CsvExportError(f"source changed before query: {plan.source_path}")
    heartbeat.record("source_hash_verified", force=True)
    savepoint = f"source_{source_ordinal}"
    connection.execute(f"SAVEPOINT {savepoint}")
    reader = None
    result = None
    dataframe_attempted = read_mode == READ_MODE_DATAFRAME
    temporal_cache = _BoundedCache()
    entity_cache = _BoundedCache(DEFAULT_ENTITY_INSERT_CACHE_SIZE)
    positive_infinity_counts: Counter[str] = Counter()
    nan_counts: Counter[str] = Counter()
    exception = _row_exception_source(row_exception_policy, binding, plan)
    filtered_counts: Counter[str] = Counter()
    row_exception_tracker = (
        _RowExceptionTracker([]) if exception is not None else None
    )
    temporal_empty_string_tracker = _forbush_temporal_empty_string_tracker(
        binding,
        plan,
        column_by_key,
        read_mode,
    )
    if read_mode == READ_MODE_ROWWISE_DIRECT:
        try:
            heartbeat.record("rowwise_direct_reader_open_start", force=True)
            reader = tsfile_module.TsFileReader(str(plan.path))
            heartbeat.record("rowwise_direct_reader_open_complete", force=True)
            heartbeat.record("rowwise_direct_query_start", force=True)
            result = _query_result(reader, plan)
            heartbeat.record("rowwise_direct_query_open", force=True)
            writer = _BufferedSpoolWriter(connection, insert_batch_rows, disk_guard)
            heartbeat.record("rowwise_direct_scan_start", force=True)
            (
                source_rows,
                eligible_rows,
                null_tag_rows,
                null_tag_cells,
            ) = _scan_rowwise(
                result,
                plan,
                source_ordinal=source_ordinal,
                source_path=plan.source_path,
                writer=writer,
                column_by_key=column_by_key,
                temporal_cache=temporal_cache,
                entity_cache=entity_cache,
                positive_infinity_counts=positive_infinity_counts,
                nan_counts=nan_counts,
                row_exception=exception,
                row_exception_tracker=row_exception_tracker,
                temporal_empty_string_tracker=temporal_empty_string_tracker,
                filtered_counts=filtered_counts,
                diagnostic_heartbeat=heartbeat,
            )
            heartbeat.record(
                "rowwise_direct_scan_complete",
                force=True,
                source_rows=source_rows,
                eligible_rows=eligible_rows,
            )
            _africa_query_exception(plan, verified_source_rows=source_rows)
            checked_positive_infinity_counts = _check_positive_infinity_contract(
                plan, positive_infinity_counts
            )
            temporal_empty_string_audits = (
                _finalize_forbush_temporal_empty_string_audits(
                    plan,
                    temporal_empty_string_tracker,
                    column_by_key,
                    source_rows=source_rows,
                    eligible_rows=eligible_rows,
                )
            )
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        finally:
            if result is not None:
                heartbeat.record("rowwise_direct_result_close_start", force=True)
                result.close()
                heartbeat.record("rowwise_direct_result_close_complete", force=True)
            if reader is not None:
                heartbeat.record("rowwise_direct_reader_close_start", force=True)
                reader.close()
                heartbeat.record("rowwise_direct_reader_close_complete", force=True)
        heartbeat.record("source_postquery_hash_start", force=True)
        if _sha256(plan.path) != plan.expected_sha256:
            raise CsvExportError(f"source changed during query: {plan.source_path}")
        heartbeat.record("source_postquery_hash_verified", force=True)
        row_exception_receipt = _finalize_row_exception_source(
            plan,
            exception,
            row_exception_tracker,
            source_rows=source_rows,
            eligible_rows=eligible_rows,
            timestamp_policy_filtered_rows=filtered_counts["timestamp_policy"],
            row_exception_filtered_rows=filtered_counts["row_exception"],
        )
        return ScanStats(
            source_rows=source_rows,
            eligible_rows=eligible_rows,
            filtered_rows=source_rows - eligible_rows,
            dataframe_attempted=False,
            rowwise_fallback=False,
            null_tag_rows=null_tag_rows,
            null_tag_cells=null_tag_cells,
            positive_infinity_counts=checked_positive_infinity_counts,
            read_mode=READ_MODE_ROWWISE_DIRECT,
            nan_counts=tuple(
                sorted(
                    (name, int(count))
                    for name, count in nan_counts.items()
                    if count
                )
            ),
            temporal_empty_string_audits=temporal_empty_string_audits,
            timestamp_policy_filtered_rows=filtered_counts["timestamp_policy"],
            row_exception_filtered_rows=filtered_counts["row_exception"],
            row_exception=row_exception_receipt,
        )
    try:
        heartbeat.record("dataframe_reader_open_start", force=True)
        reader = tsfile_module.TsFileReader(str(plan.path))
        heartbeat.record("dataframe_reader_open_complete", force=True)
        heartbeat.record("dataframe_query_start", force=True)
        result = _query_result(reader, plan)
        heartbeat.record("dataframe_query_open", force=True)
        writer = _BufferedSpoolWriter(connection, insert_batch_rows, disk_guard)
        heartbeat.record("dataframe_scan_start", force=True)
        source_rows, eligible_rows, null_tag_rows, null_tag_cells = _scan_dataframe(
            result,
            plan,
            source_ordinal=source_ordinal,
            source_path=plan.source_path,
            batch_rows=batch_rows,
            writer=writer,
            column_by_key=column_by_key,
            temporal_cache=temporal_cache,
            entity_cache=entity_cache,
            positive_infinity_counts=positive_infinity_counts,
            nan_counts=nan_counts,
            row_exception=exception,
            row_exception_tracker=row_exception_tracker,
            filtered_counts=filtered_counts,
            diagnostic_heartbeat=heartbeat,
        )
        heartbeat.record(
            "dataframe_scan_complete",
            force=True,
            source_rows=source_rows,
            eligible_rows=eligible_rows,
        )
        _africa_query_exception(plan, verified_source_rows=source_rows)
        checked_positive_infinity_counts = _check_positive_infinity_contract(
            plan, positive_infinity_counts
        )
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        fallback = False
    except _DataFrameFallback:
        heartbeat.record(
            "dataframe_fallback",
            force=True,
        )
        if result is not None:
            heartbeat.record("dataframe_result_close_start", force=True)
            result.close()
            heartbeat.record("dataframe_result_close_complete", force=True)
            result = None
        if reader is not None:
            heartbeat.record("dataframe_reader_close_start", force=True)
            reader.close()
            heartbeat.record("dataframe_reader_close_complete", force=True)
            reader = None
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        connection.execute(f"SAVEPOINT {savepoint}")
        positive_infinity_counts = Counter()
        nan_counts = Counter()
        filtered_counts = Counter()
        row_exception_tracker = (
            _RowExceptionTracker([]) if exception is not None else None
        )
        try:
            heartbeat.record("rowwise_reader_open_start", force=True)
            reader = tsfile_module.TsFileReader(str(plan.path))
            heartbeat.record("rowwise_reader_open_complete", force=True)
            heartbeat.record("rowwise_query_start", force=True)
            result = _query_result(reader, plan)
            heartbeat.record("rowwise_query_open", force=True)
            writer = _BufferedSpoolWriter(connection, insert_batch_rows, disk_guard)
            heartbeat.record("rowwise_scan_start", force=True)
            (
                source_rows,
                eligible_rows,
                null_tag_rows,
                null_tag_cells,
            ) = _scan_rowwise(
                result,
                plan,
                source_ordinal=source_ordinal,
                source_path=plan.source_path,
                writer=writer,
                column_by_key=column_by_key,
                temporal_cache=temporal_cache,
                entity_cache=entity_cache,
                positive_infinity_counts=positive_infinity_counts,
                nan_counts=nan_counts,
                row_exception=exception,
                row_exception_tracker=row_exception_tracker,
                temporal_empty_string_tracker=None,
                filtered_counts=filtered_counts,
                diagnostic_heartbeat=heartbeat,
            )
            heartbeat.record(
                "rowwise_scan_complete",
                force=True,
                source_rows=source_rows,
                eligible_rows=eligible_rows,
            )
            _africa_query_exception(plan, verified_source_rows=source_rows)
            checked_positive_infinity_counts = _check_positive_infinity_contract(
                plan, positive_infinity_counts
            )
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            fallback = True
        except Exception:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    finally:
        if result is not None:
            heartbeat.record("result_close_start", force=True)
            result.close()
            heartbeat.record("result_close_complete", force=True)
        if reader is not None:
            heartbeat.record("reader_close_start", force=True)
            reader.close()
            heartbeat.record("reader_close_complete", force=True)
    heartbeat.record("source_postquery_hash_start", force=True)
    if _sha256(plan.path) != plan.expected_sha256:
        raise CsvExportError(f"source changed during query: {plan.source_path}")
    heartbeat.record("source_postquery_hash_verified", force=True)
    row_exception_receipt = _finalize_row_exception_source(
        plan,
        exception,
        row_exception_tracker,
        source_rows=source_rows,
        eligible_rows=eligible_rows,
        timestamp_policy_filtered_rows=filtered_counts["timestamp_policy"],
        row_exception_filtered_rows=filtered_counts["row_exception"],
    )
    return ScanStats(
        source_rows=source_rows,
        eligible_rows=eligible_rows,
        filtered_rows=source_rows - eligible_rows,
        dataframe_attempted=dataframe_attempted,
        rowwise_fallback=fallback,
        null_tag_rows=null_tag_rows,
        null_tag_cells=null_tag_cells,
        positive_infinity_counts=checked_positive_infinity_counts,
        read_mode=(
            READ_MODE_DATAFRAME_FALLBACK_ROWWISE
            if fallback
            else READ_MODE_DATAFRAME
        ),
        nan_counts=tuple(
            sorted(
                (name, int(count))
                for name, count in nan_counts.items()
                if count
            )
        ),
        timestamp_policy_filtered_rows=filtered_counts["timestamp_policy"],
        row_exception_filtered_rows=filtered_counts["row_exception"],
        row_exception=row_exception_receipt,
    )


def _assign_series_ids(connection: sqlite3.Connection) -> int:
    pending: list[tuple[str, int]] = []
    count = 0
    for count, (entity_key,) in enumerate(
        connection.execute("SELECT entity_key FROM entities ORDER BY entity_key"),
        start=1,
    ):
        pending.append((entity_key, count - 1))
        if len(pending) >= DEFAULT_INSERT_BATCH_ROWS:
            connection.executemany(
                "INSERT INTO series_ids(entity_key, series_id) VALUES (?, ?)", pending
            )
            pending.clear()
    if pending:
        connection.executemany(
            "INSERT INTO series_ids(entity_key, series_id) VALUES (?, ?)", pending
        )
    connection.commit()
    return count


def _create_output_order_index(
    connection: sqlite3.Connection,
    disk_guard: _FreeDiskGuard | None = None,
) -> None:
    """Persist the final row order so sizing and emission do not re-sort."""

    if disk_guard is not None:
        disk_guard.check("output-order index start")
    connection.execute(
        "CREATE INDEX rows_output_order_idx ON rows("
        "timestamp_unix_us, source_ordinal, source_row_ordinal)"
    )
    connection.commit()
    if disk_guard is not None:
        disk_guard.check("output-order index completion")


def _csv_line(values: Sequence[Any]) -> bytes:
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerow(values)
    return stream.getvalue().encode("utf-8")


SERIES_MAP_HEADER = _csv_line(
    [
        "series_id",
        "table_name",
        "tag_names_json",
        "typed_tag_values_json",
        "source_paths_json",
    ]
)


def _series_map_line(connection: sqlite3.Connection, series_id: int) -> bytes:
    row = connection.execute(
        """
        SELECT e.entity_key, e.table_name, e.tag_names_json,
               e.typed_tag_values_json
        FROM series_ids AS s
        JOIN entities AS e ON e.entity_key = s.entity_key
        WHERE s.series_id=?
        """,
        (series_id,),
    ).fetchone()
    if row is None:
        raise AssertionError(f"unknown series id in spooled row: {series_id}")
    entity_key, table_name, tag_names, typed_tags = row
    source_paths = [
        item[0]
        for item in connection.execute(
            "SELECT source_path FROM entity_sources WHERE entity_key=? ORDER BY source_path",
            (entity_key,),
        )
    ]
    return _csv_line(
        [
            series_id,
            table_name,
            tag_names,
            typed_tags,
            _canonical_json(source_paths),
        ]
    )


def _write_series_map(
    connection: sqlite3.Connection,
    path: Path,
    series_ids: set[int] | None = None,
    *,
    stop_after_bytes: int | None = None,
    disk_guard: _FreeDiskGuard | None = None,
) -> tuple[int, str, bool]:
    if path.exists() or path.is_symlink():
        raise CsvExportError(f"refusing to overwrite existing path: {path}")
    with path.open("xb") as handle:
        handle.write(SERIES_MAP_HEADER)
        exceeded = stop_after_bytes is not None and len(SERIES_MAP_HEADER) > stop_after_bytes
        selected = (
            [item[0] for item in connection.execute("SELECT series_id FROM series_ids ORDER BY series_id")]
            if series_ids is None
            else sorted(series_ids)
        )
        for position, series_id in enumerate(selected, start=1):
            if exceeded:
                break
            handle.write(_series_map_line(connection, int(series_id)))
            if stop_after_bytes is not None and handle.tell() > stop_after_bytes:
                exceeded = True
            if disk_guard is not None and position % 10_000 == 0:
                disk_guard.check("series map write")
        handle.flush()
        os.fsync(handle.fileno())
    return path.stat().st_size, _sha256(path), exceeded


def _format_csv_value(value: Any, dtype: str) -> str:
    if value is None:
        return ""
    if value == POSITIVE_INFINITY_SPOOL_TOKEN:
        if dtype not in {"float32", "float64"}:
            raise CsvExportError("positive-infinity token appeared in a non-float column")
        return "inf"
    if value == NAN_SPOOL_TOKEN:
        if dtype not in {"float32", "float64"}:
            raise CsvExportError("NaN token appeared in a non-float column")
        return "nan"
    if dtype in {"int32", "int64"}:
        return str(int(value))
    numeric = float(value)
    if math.isnan(numeric) or numeric == -math.inf:
        raise CsvExportError(
            "raw NaN or negative-infinity spooled value is not approved"
        )
    if numeric == math.inf:
        # Accepting a raw in-memory +Inf keeps this primitive total for tests;
        # production spools use the JSON-safe sentinel above.
        return "inf"
    return repr(numeric)


def _ordered_spool_rows(connection: sqlite3.Connection) -> Iterable[tuple[int, int, str]]:
    query = """
        SELECT r.timestamp_unix_us, s.series_id, r.values_json
        FROM rows AS r
        JOIN series_ids AS s ON s.entity_key = r.entity_key
        ORDER BY r.timestamp_unix_us, r.source_ordinal, r.source_row_ordinal
    """
    return connection.execute(query)


def _write_data_csv(
    connection: sqlite3.Connection,
    path: Path,
    columns: Sequence[OutputColumn],
    *,
    stop_after_bytes: int | None = None,
    disk_guard: _FreeDiskGuard | None = None,
) -> tuple[int, int, str, bool]:
    if path.exists() or path.is_symlink():
        raise CsvExportError(f"refusing to overwrite existing path: {path}")
    header = _csv_line(
        ["timestamp_unix_us", "series_id", *(item.header for item in columns)]
    )
    emitted = 0
    exceeded = stop_after_bytes is not None and len(header) > stop_after_bytes
    with path.open("xb") as handle:
        handle.write(header)
        written = len(header)
        for timestamp, series_id, values_json in _ordered_spool_rows(connection):
            if exceeded:
                break
            values = json.loads(values_json)
            line = _csv_line(
                [
                    timestamp,
                    series_id,
                    *(
                        _format_csv_value(values.get(column.header), column.dtype)
                        for column in columns
                    ),
                ]
            )
            handle.write(line)
            written += len(line)
            emitted += 1
            if stop_after_bytes is not None and written > stop_after_bytes:
                exceeded = True
            if disk_guard is not None and emitted % 100_000 == 0:
                disk_guard.check("data CSV write")
        handle.flush()
        os.fsync(handle.fileno())
    return path.stat().st_size, emitted, _sha256(path), exceeded


def _columns_header() -> bytes:
    return _csv_line(
        [
            "column",
            "logical_key_json",
            "table_name",
            "source_name",
            "source_type",
            "output_dtype",
            "conversion",
        ]
    )


def _column_line(column: OutputColumn) -> bytes:
    return _csv_line(
        [
            column.header,
            column.logical_key,
            column.table_name,
            column.source_name,
            column.source_type,
            column.dtype,
            column.conversion,
        ]
    )


def _columns_csv_prefix_bytes(columns: Sequence[OutputColumn]) -> list[int]:
    """Return exact columns.csv sizes for every retained leading prefix."""

    sizes = [len(_columns_header())]
    for column in columns:
        sizes.append(sizes[-1] + len(_column_line(column)))
    return sizes


def _write_columns_csv(
    path: Path,
    columns: Sequence[OutputColumn],
) -> tuple[int, str]:
    if path.exists() or path.is_symlink():
        raise CsvExportError(f"refusing to overwrite existing path: {path}")
    with path.open("xb") as handle:
        handle.write(_columns_header())
        for column in columns:
            handle.write(_column_line(column))
        handle.flush()
        os.fsync(handle.fileno())
    return path.stat().st_size, _sha256(path)


def _fixed_count(value: int) -> str:
    if value < 0 or value >= 10**20:
        raise CsvExportError(f"manifest counter is outside fixed uint20 range: {value}")
    return f"{value:020d}"


def _emitted_nan_receipt_rows(
    base: Mapping[str, Any],
    selected: Sequence[OutputColumn],
    emitted_nan_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Describe NaNs that actually survive both column and row capping."""

    source_rows = base.get("source_nan_by_column", [])
    if not isinstance(source_rows, list):
        raise CsvExportError("source_nan_by_column must be a list")
    source_counts: dict[str, int] = {}
    for item in source_rows:
        if not isinstance(item, Mapping):
            raise CsvExportError("source_nan_by_column entries must be objects")
        header = str(item.get("output_column", ""))
        count = item.get("count")
        if not header or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CsvExportError("invalid source NaN receipt entry")
        if header in source_counts:
            raise CsvExportError("duplicate source NaN output column")
        source_counts[header] = count

    selected_by_header = {column.header: column for column in selected}
    if emitted_nan_counts is None:
        counts = {
            header: source_counts.get(header, 0) for header in selected_by_header
        }
    else:
        counts = {
            str(header): int(count) for header, count in emitted_nan_counts.items()
        }
        unknown = set(counts) - set(selected_by_header)
        if unknown:
            raise CsvExportError(
                f"emitted NaN counts contain unselected columns: {sorted(unknown)}"
            )
    rows: list[dict[str, Any]] = []
    for header, column in selected_by_header.items():
        count = int(counts.get(header, 0))
        if count < 0 or count > source_counts.get(header, 0):
            raise CsvExportError(
                f"emitted NaN count for {header} is outside its source count"
            )
        if count:
            rows.append(
                {
                    "output_column": header,
                    "output_dtype": column.dtype,
                    "count": count,
                }
            )
    return rows


def _manifest_bytes(
    base: Mapping[str, Any],
    *,
    cap: int,
    total_rows: int,
    emitted_rows: int,
    selected: Sequence[OutputColumn],
    dropped: Sequence[str],
    data_meta: tuple[int, str],
    map_meta: tuple[int, str],
    columns_meta: tuple[int, str],
    next_row_bytes: int,
    next_row_would_total: int,
    next_trailing_column_would_total: int,
    emitted_nan_counts: Mapping[str, int] | None = None,
) -> bytes:
    manifest_size = 0
    other_bytes = data_meta[0] + map_meta[0] + columns_meta[0]
    for _attempt in range(4):
        emitted_nan_by_column = _emitted_nan_receipt_rows(
            base, selected, emitted_nan_counts
        )
        payload = {
            **base,
            "emitted_nan_cells": sum(
                int(item["count"]) for item in emitted_nan_by_column
            ),
            "emitted_nan_by_column": emitted_nan_by_column,
            "cap_accounting": "all four files in this dataset directory",
            "cap_bytes_decimal_20": _fixed_count(cap),
            "actual_total_bytes_decimal_20": _fixed_count(other_bytes + manifest_size),
            "dataset_manifest_bytes_decimal_20": _fixed_count(manifest_size),
            "eligible_rows_decimal_20": _fixed_count(total_rows),
            "emitted_rows_decimal_20": _fixed_count(emitted_rows),
            "lost_rows_due_to_byte_cap_decimal_20": _fixed_count(
                total_rows - emitted_rows
            ),
            "next_row_bytes_decimal_20": _fixed_count(next_row_bytes),
            "next_row_would_total_bytes_decimal_20": _fixed_count(
                next_row_would_total
            ),
            "next_trailing_column_would_total_bytes_decimal_20": _fixed_count(
                next_trailing_column_would_total
            ),
            "candidate_value_column_count_decimal_20": _fixed_count(
                len(selected) + len(dropped)
            ),
            "retained_value_column_count_decimal_20": _fixed_count(len(selected)),
            "dropped_trailing_value_column_count_decimal_20": _fixed_count(
                len(dropped)
            ),
            "candidate_value_columns": [
                *(column.header for column in selected),
                *dropped,
            ],
            "retained_value_columns": [column.header for column in selected],
            "dropped_trailing_value_columns": list(dropped),
            "last_retained_value_column": selected[-1].header,
            "first_dropped_trailing_value_column": (
                dropped[0] if dropped else None
            ),
            "files": [
                {
                    "path": "data.csv",
                    "bytes_decimal_20": _fixed_count(data_meta[0]),
                    "sha256": data_meta[1],
                },
                {
                    "path": "series_map.csv",
                    "bytes_decimal_20": _fixed_count(map_meta[0]),
                    "sha256": map_meta[1],
                },
                {
                    "path": "columns.csv",
                    "bytes_decimal_20": _fixed_count(columns_meta[0]),
                    "sha256": columns_meta[1],
                },
                {
                    "path": "dataset_manifest.json",
                    "bytes_decimal_20": _fixed_count(manifest_size),
                    "sha256": None,
                },
            ],
        }
        rendered = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        if len(rendered) == manifest_size:
            return rendered
        manifest_size = len(rendered)
    raise AssertionError("dataset manifest size did not reach a fixed point")


def _manifest_total_for_candidate(
    base: Mapping[str, Any],
    *,
    cap: int,
    total_rows: int,
    emitted_rows: int,
    selected: Sequence[OutputColumn],
    dropped: Sequence[str],
    data_bytes: int,
    map_bytes: int,
    columns_meta: tuple[int, str],
    next_row_bytes: int = 0,
    next_row_would_total: int = 0,
    next_trailing_column_would_total: int = 0,
    emitted_nan_counts: Mapping[str, int] | None = None,
) -> tuple[int, bytes]:
    manifest = _manifest_bytes(
        base,
        cap=cap,
        total_rows=total_rows,
        emitted_rows=emitted_rows,
        selected=selected,
        dropped=dropped,
        data_meta=(data_bytes, "0" * 64),
        map_meta=(map_bytes, "0" * 64),
        columns_meta=columns_meta,
        next_row_bytes=next_row_bytes,
        next_row_would_total=next_row_would_total,
        next_trailing_column_would_total=next_trailing_column_would_total,
        emitted_nan_counts=emitted_nan_counts,
    )
    return data_bytes + map_bytes + columns_meta[0] + len(manifest), manifest


def _write_exact_prefix(
    connection: sqlite3.Connection,
    data_path: Path,
    map_path: Path,
    columns: Sequence[OutputColumn],
    *,
    cap: int,
    total_rows: int,
    dropped: Sequence[str],
    columns_meta: tuple[int, str],
    manifest_base: Mapping[str, Any],
    next_trailing_column_would_total: int,
    disk_guard: _FreeDiskGuard | None = None,
) -> tuple[
    tuple[int, str],
    tuple[int, str],
    int,
    int,
    int,
    set[int],
    dict[str, int],
]:
    header = _csv_line(
        ["timestamp_unix_us", "series_id", *(item.header for item in columns)]
    )
    data_size = len(header)
    map_size = len(SERIES_MAP_HEADER)
    used: set[int] = set()
    map_lines: dict[int, bytes] = {}
    emitted = 0
    next_row_bytes = 0
    next_row_would_total = 0
    emitted_nan_counts: Counter[str] = Counter()
    with data_path.open("xb") as handle:
        handle.write(header)
        for timestamp, series_id, values_json in _ordered_spool_rows(connection):
            values = json.loads(values_json)
            line = _csv_line(
                [
                    timestamp,
                    series_id,
                    *(
                        _format_csv_value(values.get(column.header), column.dtype)
                        for column in columns
                    ),
                ]
            )
            series_id = int(series_id)
            row_nan_counts = Counter(
                column.header
                for column in columns
                if values.get(column.header) == NAN_SPOOL_TOKEN
            )
            proposed_nan_counts = emitted_nan_counts + row_nan_counts
            added_map_bytes = 0
            if series_id not in used:
                map_lines.setdefault(series_id, _series_map_line(connection, series_id))
                added_map_bytes = len(map_lines[series_id])
            proposed_total, _manifest = _manifest_total_for_candidate(
                manifest_base,
                cap=cap,
                total_rows=total_rows,
                emitted_rows=emitted + 1,
                selected=columns,
                dropped=dropped,
                data_bytes=data_size + len(line),
                map_bytes=map_size + added_map_bytes,
                columns_meta=columns_meta,
                next_trailing_column_would_total=(
                    next_trailing_column_would_total
                ),
                emitted_nan_counts=proposed_nan_counts,
            )
            if proposed_total > cap:
                next_row_bytes = len(line) + added_map_bytes
                next_row_would_total = proposed_total
                break
            handle.write(line)
            data_size += len(line)
            map_size += added_map_bytes
            used.add(series_id)
            emitted_nan_counts.update(row_nan_counts)
            emitted += 1
            if disk_guard is not None and emitted % 100_000 == 0:
                disk_guard.check("capped chronological prefix write")
        handle.flush()
        os.fsync(handle.fileno())
    map_bytes, map_sha, map_exceeded = _write_series_map(
        connection, map_path, used, disk_guard=disk_guard
    )
    if map_exceeded:
        raise AssertionError("bounded prefix series map unexpectedly exceeded no limit")
    if map_bytes != map_size:
        raise AssertionError("series-map byte accounting differs from emitted series set")
    return (
        (data_path.stat().st_size, _sha256(data_path)),
        (map_bytes, map_sha),
        emitted,
        next_row_bytes,
        next_row_would_total,
        used,
        dict(emitted_nan_counts),
    )


def _verify_series_mapping(
    connection: sqlite3.Connection, data_path: Path, map_path: Path
) -> int:
    connection.execute("DROP TABLE IF EXISTS temp.verify_data_series")
    connection.execute("DROP TABLE IF EXISTS temp.verify_map_series")
    connection.execute("CREATE TEMP TABLE verify_data_series(series_id INTEGER PRIMARY KEY)")
    connection.execute("CREATE TEMP TABLE verify_map_series(series_id INTEGER PRIMARY KEY)")
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or header[:2] != ["timestamp_unix_us", "series_id"]:
            raise AssertionError("published data.csv header changed")
        batch: list[tuple[int]] = []
        for row in reader:
            batch.append((int(row[1]),))
            if len(batch) >= DEFAULT_INSERT_BATCH_ROWS:
                connection.executemany(
                    "INSERT OR IGNORE INTO verify_data_series VALUES (?)", batch
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO verify_data_series VALUES (?)", batch
            )
    with map_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or not header or header[0] != "series_id":
            raise AssertionError("published series_map.csv header changed")
        connection.executemany(
            "INSERT INTO verify_map_series VALUES (?)",
            ((int(row[0]),) for row in reader),
        )
    if connection.execute(
        "SELECT series_id FROM verify_data_series EXCEPT "
        "SELECT series_id FROM verify_map_series LIMIT 1"
    ).fetchone() is not None:
        raise AssertionError("series map is missing a data.csv series ID")
    if connection.execute(
        "SELECT series_id FROM verify_map_series EXCEPT "
        "SELECT series_id FROM verify_data_series LIMIT 1"
    ).fetchone() is not None:
        raise AssertionError("series map contains an unused series ID")
    count = int(connection.execute("SELECT COUNT(*) FROM verify_map_series").fetchone()[0])
    connection.execute("DROP TABLE temp.verify_data_series")
    connection.execute("DROP TABLE temp.verify_map_series")
    return count


def _measure_data_prefix_bytes(
    connection: sqlite3.Connection,
    columns: Sequence[OutputColumn],
    disk_guard: _FreeDiskGuard | None = None,
) -> tuple[list[int], int]:
    """Return exact data.csv byte sizes for every leading column prefix.

    All emitted cells are numeric tokens (or empty), so CSV quoting never
    changes a token.  Each added column contributes one comma per row plus its
    non-null token lengths.  Sparse JSON lets this run in O(rows + present
    cells + columns), rather than materializing every null in a wide union.
    """

    by_header = {column.header: index for index, column in enumerate(columns)}
    token_bytes = [0] * len(columns)
    base = len(_csv_line(["timestamp_unix_us", "series_id"]))
    row_count = 0
    for timestamp, series_id, values_json in _ordered_spool_rows(connection):
        base += len(str(int(timestamp))) + 1 + len(str(int(series_id))) + 1
        values = json.loads(values_json)
        for header, value in values.items():
            index = by_header.get(header)
            if index is None:
                raise CsvExportError(
                    f"spool contains an unknown output column: {header!r}"
                )
            if value is None:
                continue
            token_bytes[index] += len(
                _format_csv_value(value, columns[index].dtype).encode("utf-8")
            )
        row_count += 1
        if disk_guard is not None and row_count % 100_000 == 0:
            disk_guard.check("ordered spool size scan")
    sizes = [base]
    running = base
    for column, tokens in zip(columns, token_bytes):
        running += 1 + len(column.header.encode("utf-8")) + row_count + tokens
        sizes.append(running)
    return sizes, row_count


def _candidate_total_for_prefix(
    manifest_base: Mapping[str, Any],
    columns: Sequence[OutputColumn],
    data_prefix_bytes: Sequence[int],
    *,
    retained_count: int,
    cap: int,
    total_rows: int,
    map_bytes: int,
    columns_prefix_bytes: Sequence[int],
    next_trailing_column_would_total: int = 0,
) -> int:
    selected = columns[:retained_count]
    dropped = [column.header for column in columns[retained_count:]]
    total, _manifest = _manifest_total_for_candidate(
        manifest_base,
        cap=cap,
        total_rows=total_rows,
        emitted_rows=total_rows,
        selected=selected,
        dropped=dropped,
        data_bytes=data_prefix_bytes[retained_count],
        map_bytes=map_bytes,
        columns_meta=(columns_prefix_bytes[retained_count], "0" * 64),
        next_trailing_column_would_total=next_trailing_column_would_total,
    )
    return total


def _widest_full_row_prefix(
    manifest_base: Mapping[str, Any],
    columns: Sequence[OutputColumn],
    data_prefix_bytes: Sequence[int],
    *,
    cap: int,
    total_rows: int,
    map_bytes: int,
    columns_prefix_bytes: Sequence[int],
) -> tuple[int, int]:
    """Find maximum full-row leading-column count, plus exact K+1 proof."""

    low, high = 1, len(columns)
    if _candidate_total_for_prefix(
        manifest_base,
        columns,
        data_prefix_bytes,
        retained_count=1,
        cap=cap,
        total_rows=total_rows,
        map_bytes=map_bytes,
        columns_prefix_bytes=columns_prefix_bytes,
    ) > cap:
        proof = (
            _candidate_total_for_prefix(
                manifest_base,
                columns,
                data_prefix_bytes,
                retained_count=2,
                cap=cap,
                total_rows=total_rows,
                map_bytes=map_bytes,
                columns_prefix_bytes=columns_prefix_bytes,
            )
            if len(columns) > 1
            else 0
        )
        return 0, proof
    while low < high:
        middle = (low + high + 1) // 2
        total = _candidate_total_for_prefix(
            manifest_base,
            columns,
            data_prefix_bytes,
            retained_count=middle,
            cap=cap,
            total_rows=total_rows,
            map_bytes=map_bytes,
            columns_prefix_bytes=columns_prefix_bytes,
        )
        if total <= cap:
            low = middle
        else:
            high = middle - 1
    proof = 0
    if low < len(columns):
        proof = _candidate_total_for_prefix(
            manifest_base,
            columns,
            data_prefix_bytes,
            retained_count=low + 1,
            cap=cap,
            total_rows=total_rows,
            map_bytes=map_bytes,
            columns_prefix_bytes=columns_prefix_bytes,
        )
    return low, proof


def _materialize_capped_csvs(
    connection: sqlite3.Connection,
    dataset_dir: Path,
    columns: Sequence[OutputColumn],
    *,
    dataset_byte_cap: int,
    manifest_base: Mapping[str, Any],
    disk_guard: _FreeDiskGuard | None = None,
) -> Mapping[str, Any]:
    if not 0 < dataset_byte_cap <= DEFAULT_DATASET_BYTE_CAP:
        raise CsvExportError(
            f"dataset byte cap must be in 1..{DEFAULT_DATASET_BYTE_CAP}"
        )
    dataset_dir.mkdir()
    total_rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    token = uuid.uuid4().hex
    data_candidate = dataset_dir / f".data-{token}.csv"
    map_candidate = dataset_dir / f".map-{token}.csv"
    columns_candidate = dataset_dir / f".columns-{token}.csv"
    # One bounded map pass and one sparse spool scan determine the widest
    # full-row column prefix.  This replaces the old one-column-at-a-time CSV
    # rewrites, which are infeasible for 5,000+ column datasets.
    map_bytes, map_sha, map_exceeded = _write_series_map(
        connection,
        map_candidate,
        stop_after_bytes=dataset_byte_cap,
        disk_guard=disk_guard,
    )
    data_prefix_bytes, measured_rows = _measure_data_prefix_bytes(
        connection, columns, disk_guard
    )
    if measured_rows != total_rows:
        raise AssertionError("ordered byte scan row count differs from spool")
    columns_prefix_bytes = _columns_csv_prefix_bytes(columns)
    retained_count, next_trailing_column_would_total = _widest_full_row_prefix(
        manifest_base,
        columns,
        data_prefix_bytes,
        cap=dataset_byte_cap,
        total_rows=total_rows,
        map_bytes=map_bytes,
        columns_prefix_bytes=columns_prefix_bytes,
    )
    if map_exceeded:
        retained_count = 0
    selected_count = max(1, retained_count)
    selected = list(columns[:selected_count])
    dropped = [column.header for column in columns[selected_count:]]
    columns_meta = _write_columns_csv(columns_candidate, selected)
    if columns_meta[0] != columns_prefix_bytes[selected_count]:
        raise AssertionError("columns.csv byte accounting differs from retained prefix")

    if retained_count >= 1 and not map_exceeded:
        data_bytes, emitted, data_sha, data_exceeded = _write_data_csv(
            connection, data_candidate, selected, disk_guard=disk_guard
        )
        if data_exceeded or emitted != total_rows:
            raise AssertionError("selected full-row data write ended prematurely")
        if data_bytes != data_prefix_bytes[retained_count]:
            raise AssertionError("measured and written data.csv bytes differ")
        data_meta = (data_bytes, data_sha)
        map_meta = (map_bytes, map_sha)
        next_row_bytes = 0
        next_row_would_total = 0
        emitted_nan_counts = None
    else:
        # Even one value column is irreducible.  Keep the exact maximum
        # chronological row prefix.  Map rows are added only when their series
        # first appears, so unused mappings never consume the cap.
        map_candidate.unlink()
        (
            data_meta,
            map_meta,
            emitted,
            next_row_bytes,
            next_row_would_total,
            used,
            emitted_nan_counts,
        ) = _write_exact_prefix(
            connection,
            data_candidate,
            map_candidate,
            selected,
            cap=dataset_byte_cap,
            total_rows=total_rows,
            dropped=dropped,
            columns_meta=columns_meta,
            manifest_base=manifest_base,
            next_trailing_column_would_total=(
                next_trailing_column_would_total
            ),
            disk_guard=disk_guard,
        )

    if disk_guard is not None:
        disk_guard.check("final CSV materialization")

    manifest = _manifest_bytes(
        manifest_base,
        cap=dataset_byte_cap,
        total_rows=total_rows,
        emitted_rows=emitted,
        selected=selected,
        dropped=dropped,
        data_meta=data_meta,
        map_meta=map_meta,
        columns_meta=columns_meta,
        next_row_bytes=next_row_bytes,
        next_row_would_total=next_row_would_total,
        next_trailing_column_would_total=(
            next_trailing_column_would_total
        ),
        emitted_nan_counts=emitted_nan_counts,
    )
    data_path = dataset_dir / "data.csv"
    map_path = dataset_dir / "series_map.csv"
    columns_path = dataset_dir / "columns.csv"
    manifest_path = dataset_dir / "dataset_manifest.json"
    os.rename(data_candidate, data_path)
    os.rename(map_candidate, map_path)
    os.rename(columns_candidate, columns_path)
    with manifest_path.open("xb") as handle:
        handle.write(manifest)
        handle.flush()
        os.fsync(handle.fileno())
    published = [data_path, map_path, columns_path, manifest_path]
    actual_total = sum(path.stat().st_size for path in published)
    if actual_total > dataset_byte_cap:
        raise AssertionError("published dataset files exceed the exact byte cap")
    mapped_count = _verify_series_mapping(connection, data_path, map_path)
    if emitted < total_rows and next_row_would_total <= dataset_byte_cap:
        raise AssertionError("row-prefix truncation did not prove the next row exceeds cap")
    if dropped and next_trailing_column_would_total <= dataset_byte_cap:
        raise AssertionError("column-prefix truncation did not prove the next column exceeds cap")
    emitted_nan_by_column = _emitted_nan_receipt_rows(
        manifest_base, selected, emitted_nan_counts
    )
    return {
        "actual_capped_bytes": actual_total,
        "cap_bytes": dataset_byte_cap,
        "retained_value_columns": [column.header for column in selected],
        "dropped_trailing_value_columns": dropped,
        "eligible_rows": total_rows,
        "emitted_rows": emitted,
        "lost_rows_due_to_byte_cap": total_rows - emitted,
        "next_row_bytes": next_row_bytes,
        "next_row_would_total_bytes": next_row_would_total,
        "next_trailing_column_would_total_bytes": (
            next_trailing_column_would_total
        ),
        "series_count_in_output": mapped_count,
        "emitted_nan_cells": sum(
            int(item["count"]) for item in emitted_nan_by_column
        ),
        "emitted_nan_by_column": emitted_nan_by_column,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        },
    }


def _check_filter_contract(
    policy: policy_core.TimestampPolicy,
    plans: Sequence[policy_core.SourcePlan],
    stats: Sequence[ScanStats],
) -> None:
    by_rule: dict[str, list[int]] = {}
    for plan, stat in zip(plans, stats):
        if plan.adapter_rule_id is None or plan.row_filter is None:
            continue
        counts = by_rule.setdefault(plan.adapter_rule_id, [0, 0, 0])
        counts[0] += stat.source_rows
        counts[1] += stat.source_rows - stat.timestamp_policy_filtered_rows
        counts[2] += stat.timestamp_policy_filtered_rows
    for rule in policy.source_rules:
        if rule.row_filter is None:
            continue
        actual = by_rule.get(rule.rule_id, [0, 0, 0])
        expected = [
            rule.row_filter.expected_input_rows,
            rule.row_filter.expected_output_rows,
            rule.row_filter.expected_drop_rows,
        ]
        if actual != expected:
            raise CsvExportError(
                f"adapter row-filter count mismatch for {rule.rule_id}: "
                f"actual={actual}, expected={expected}"
            )


def _positive_infinity_receipt_rows(
    plan: policy_core.SourcePlan,
    stat: ScanStats,
    column_by_key: Mapping[str, OutputColumn],
) -> list[dict[str, Any]]:
    fields = {field.name: field for field in plan.fields}
    rows: list[dict[str, Any]] = []
    for source_name, count in stat.positive_infinity_counts:
        field = fields.get(source_name)
        if field is None or field.tsfile_type not in {"FLOAT", "DOUBLE"}:
            raise AssertionError("positive-infinity count lacks a retained float field")
        output = column_by_key[_field_logical_key(plan, field)]
        rows.append(
            {
                "source_name": source_name,
                "output_column": output.header,
                "source_type": field.tsfile_type,
                "count": count,
            }
        )
    return sorted(rows, key=lambda item: str(item["output_column"]))


def _nan_receipt_rows(
    plan: policy_core.SourcePlan,
    stat: ScanStats,
    column_by_key: Mapping[str, OutputColumn],
) -> list[dict[str, Any]]:
    """Bind each eligible source NaN count to its source and output columns."""

    fields = {field.name: field for field in plan.fields}
    rows: list[dict[str, Any]] = []
    for source_name, count in stat.nan_counts:
        field = fields.get(source_name)
        if field is None or field.tsfile_type not in {"FLOAT", "DOUBLE"}:
            raise AssertionError("NaN count lacks a retained float field")
        output = column_by_key[_field_logical_key(plan, field)]
        rows.append(
            {
                "source_name": source_name,
                "source_type": field.tsfile_type,
                "output_column": output.header,
                "count": count,
            }
        )
    return sorted(
        rows,
        key=lambda item: (str(item["output_column"]), str(item["source_name"])),
    )


def _dataset_source_nan_receipt_rows(
    plans: Sequence[policy_core.SourcePlan],
    stats: Sequence[ScanStats],
    column_by_key: Mapping[str, OutputColumn],
) -> list[dict[str, Any]]:
    """Aggregate eligible (pre-cap) NaNs by stable output column."""

    counts: Counter[tuple[str, str]] = Counter()
    for plan, stat in zip(plans, stats):
        if plan.source_action != "materialize":
            continue
        for row in _nan_receipt_rows(plan, stat, column_by_key):
            output = column_by_key[
                _field_logical_key(
                    plan,
                    next(field for field in plan.fields if field.name == row["source_name"]),
                )
            ]
            counts[(str(row["output_column"]), output.dtype)] += int(row["count"])
    return [
        {"output_column": header, "output_dtype": dtype, "count": count}
        for (header, dtype), count in sorted(counts.items())
        if count
    ]


def _check_frozen_policy_closure(
    policy_path: Path,
    policies: Mapping[str, policy_core.TimestampPolicy],
) -> None:
    """Fail closed unless the supplied policy is the complete frozen THULab160 set."""

    raw = json.loads(policy_path.resolve().read_text(encoding="utf-8"))
    values = raw.get("datasets")
    if not isinstance(values, list):
        raise CsvExportError("timestamp policy datasets must be a list")
    if raw.get("scope_timestamp_dataset_count") != EXPECTED_THULAB_DATASET_COUNT:
        raise CsvExportError("timestamp policy does not declare the frozen THULab160 scope")
    if len(values) != EXPECTED_THULAB_DATASET_COUNT or len(policies) != EXPECTED_THULAB_DATASET_COUNT:
        raise CsvExportError(
            "timestamp policy closure must contain exactly "
            f"{EXPECTED_THULAB_DATASET_COUNT} datasets"
        )
    actual = Counter(policy.decision for policy in policies.values())
    actual_counts = {
        name: int(actual.get(name, 0)) for name in EXPECTED_POLICY_DECISION_COUNTS
    }
    if actual_counts != EXPECTED_POLICY_DECISION_COUNTS:
        raise CsvExportError(
            "timestamp policy decision counts differ from frozen 34/123/3 closure: "
            f"{actual_counts}"
        )
    if raw.get("decision_counts") != EXPECTED_POLICY_DECISION_COUNTS:
        raise CsvExportError(
            "timestamp policy declared decision_counts differ from computed frozen closure"
        )


def _bind_row_exception_to_bindings(
    policy: RowExceptionPolicy | None,
    bindings: Sequence[policy_core.DatasetBinding],
) -> None:
    if policy is None:
        return
    if policy.expected_source_file_count != 64:
        raise CsvExportError("row-exception policy must bind exactly 64 macroecon sources")
    if policy.expected_candidate_value_column_count != 5548:
        raise CsvExportError("row-exception policy must bind exactly 5548 FLOAT fields")
    matches = [item for item in bindings if item.row.dataset_id == policy.dataset_id]
    if len(matches) != 1:
        raise CsvExportError("row-exception dataset is absent or duplicated in scope bindings")
    binding = matches[0]
    if binding.scope_sha256 != policy.scope_sha256:
        raise CsvExportError("row-exception binding scope SHA-256 mismatch")
    if binding.policy_file_sha256 != policy.timestamp_policy_file_sha256:
        raise CsvExportError("row-exception binding timestamp-policy SHA-256 mismatch")
    if binding.source_receipt_sha256 != policy.source_receipt_sha256:
        raise CsvExportError("row-exception binding source-receipt SHA-256 mismatch")
    if len(binding.source_files) != policy.expected_source_file_count:
        raise CsvExportError(
            "row-exception macroecon source-file count mismatch: "
            f"actual={len(binding.source_files)}, "
            f"expected={policy.expected_source_file_count}"
        )
    bound_sources = {
        relative: digest for relative, _path, _size, digest in binding.source_files
    }
    for source in policy.sources:
        if bound_sources.get(source.source_path) != source.source_sha256:
            raise CsvExportError(
                f"row-exception source binding mismatch for {source.source_path}"
            )


def _validate_row_exception_dataset_schema(
    binding: policy_core.DatasetBinding,
    plans: Sequence[policy_core.SourcePlan],
    policy: RowExceptionPolicy | None,
) -> None:
    if policy is None or binding.row.dataset_id != policy.dataset_id:
        return
    if len(plans) != policy.expected_source_file_count:
        raise CsvExportError("row-exception macroecon plan count mismatch")
    binding_paths = {relative for relative, _path, _size, _digest in binding.source_files}
    if {plan.source_path for plan in plans} != binding_paths:
        raise CsvExportError("row-exception macroecon plans differ from frozen source files")
    source_rules = policy.sources_by_path
    field_count = 0
    for plan in plans:
        if (
            plan.source_action != "materialize"
            or plan.primary_temporal is not None
            or plan.timestamp_source_column != "Time"
            or plan.timestamp_source_unit != "ms"
            or plan.timestamp_output_unit != "us"
            or plan.tag_names
            or plan.dropped_columns
        ):
            raise CsvExportError(
                f"{plan.source_path}: row-exception macroecon schema/timestamp contract drift"
            )
        if not plan.fields or any(
            field.tsfile_type != "FLOAT" or field.temporal is not None
            for field in plan.fields
        ):
            raise CsvExportError(
                f"{plan.source_path}: row-exception requires only FLOAT FIELD values"
            )
        field_count += len(plan.fields)
        source = source_rules.get(plan.source_path)
        if source is not None and (
            source.table_name != plan.table_name
            or source.source_sha256 != plan.expected_sha256
        ):
            raise CsvExportError(
                f"{plan.source_path}: row-exception table/hash plan binding mismatch"
            )
    candidate_count = len(_output_columns(plans))
    if (
        field_count != policy.expected_candidate_value_column_count
        or candidate_count != policy.expected_candidate_value_column_count
    ):
        raise CsvExportError(
            "row-exception macroecon candidate FLOAT field count mismatch: "
            f"schema={field_count}, candidates={candidate_count}, "
            f"expected={policy.expected_candidate_value_column_count}"
        )


def _check_row_exception_dataset_counts(
    binding: policy_core.DatasetBinding,
    stats: Sequence[ScanStats],
    policy: RowExceptionPolicy | None,
) -> None:
    observed_drops = sum(item.row_exception_filtered_rows for item in stats)
    if policy is None or binding.row.dataset_id != policy.dataset_id:
        if observed_drops:
            raise CsvExportError("row-exception rows appeared outside the bound dataset")
        return
    actual = (
        sum(item.source_rows for item in stats),
        sum(item.eligible_rows for item in stats),
        observed_drops,
    )
    expected = (
        policy.expected_source_rows,
        policy.expected_eligible_rows,
        policy.expected_dropped_rows,
    )
    if actual != expected:
        raise CsvExportError(
            f"row-exception macroecon aggregate counts mismatch: "
            f"actual={actual}, expected={expected}"
        )
    if sum(item.timestamp_policy_filtered_rows for item in stats) != 0:
        raise CsvExportError(
            "row-exception macroecon dataset unexpectedly used timestamp-policy filtering"
        )
    receipts = [item.row_exception for item in stats if item.row_exception is not None]
    if len(receipts) != len(policy.sources):
        raise CsvExportError("row-exception per-source receipt count mismatch")


def export_dataset(
    binding: policy_core.DatasetBinding,
    plans: Sequence[policy_core.SourcePlan],
    output_root: Path,
    *,
    dataset_byte_cap: int,
    batch_rows: int,
    insert_batch_rows: int,
    tsfile_module: Any,
    disk_reserve_bytes: int,
    row_exception_policy: RowExceptionPolicy | None = None,
    worker_diagnostics: WorkerDiagnostics | None = None,
) -> Mapping[str, Any]:
    if binding.policy is None or binding.policy.decision == "block":
        raise CsvExportError(f"dataset has no admitted timestamp policy: {binding.row.dataset_id}")
    materialized_plans = [plan for plan in plans if plan.source_action == "materialize"]
    _validate_row_exception_dataset_schema(binding, plans, row_exception_policy)
    columns = _output_columns(materialized_plans)
    column_by_key = _column_by_logical_key(columns)
    dataset_dir = output_root / binding.row.dataset_id
    if dataset_dir.exists() or dataset_dir.is_symlink():
        raise CsvExportError(f"refusing to overwrite existing dataset: {dataset_dir}")
    spool_parent = output_root / ".spool"
    spool_parent.mkdir(exist_ok=True)
    disk_guard = _FreeDiskGuard(output_root, disk_reserve_bytes)
    disk_guard.check("dataset start")
    dataset_started = time.monotonic()
    _emit_progress(
        "dataset_start",
        dataset_id=binding.row.dataset_id,
        source_count=len(plans),
        candidate_value_column_count=len(columns),
    )
    with tempfile.TemporaryDirectory(
        prefix=f"{binding.row.dataset_id}-", dir=spool_parent
    ) as temp_text:
        connection = _open_spool(Path(temp_text) / "rows.sqlite3")
        try:
            stats: list[ScanStats] = []
            for source_ordinal, plan in enumerate(plans):
                source_started = time.monotonic()
                source_log_fields: Mapping[str, Any] = {}
                if worker_diagnostics is not None:
                    source_log_fields = _record_worker_source_event(
                        worker_diagnostics,
                        "source_start",
                        plan,
                        source_ordinal=source_ordinal,
                    )
                _emit_progress(
                    "source_start",
                    dataset_id=binding.row.dataset_id,
                    **source_log_fields,
                )
                stat = _scan_source(
                    connection,
                    binding,
                    plan,
                    source_ordinal=source_ordinal,
                    batch_rows=batch_rows,
                    insert_batch_rows=insert_batch_rows,
                    tsfile_module=tsfile_module,
                    column_by_key=column_by_key,
                    disk_guard=disk_guard,
                    row_exception_policy=row_exception_policy,
                    worker_diagnostics=worker_diagnostics,
                )
                stats.append(stat)
                completion_fields = {
                    "source_rows": stat.source_rows,
                    "eligible_rows": stat.eligible_rows,
                    "filtered_rows": stat.filtered_rows,
                    "read_mode": stat.read_mode,
                    "elapsed_seconds": round(
                        time.monotonic() - source_started, 6
                    ),
                }
                if worker_diagnostics is not None:
                    source_log_fields = _record_worker_source_event(
                        worker_diagnostics,
                        "source_complete",
                        plan,
                        source_ordinal=source_ordinal,
                        **completion_fields,
                    )
                else:
                    source_log_fields = {
                        "source_path": plan.source_path,
                        **completion_fields,
                    }
                _emit_progress(
                    "source_complete",
                    dataset_id=binding.row.dataset_id,
                    **source_log_fields,
                )
            connection.commit()
            _check_filter_contract(binding.policy, plans, stats)
            _check_row_exception_dataset_counts(binding, stats, row_exception_policy)
            series_count = _assign_series_ids(connection)
            _create_output_order_index(connection, disk_guard)
            source_rows = sum(item.source_rows for item in stats)
            eligible_rows = sum(item.eligible_rows for item in stats)
            source_nan_by_column = _dataset_source_nan_receipt_rows(
                plans, stats, column_by_key
            )
            manifest_base = {
                "schema": RECEIPT_VERSION,
                "status": "complete",
                "dataset_id": binding.row.dataset_id,
                "repo_id": binding.row.repo_id,
                "revision": binding.row.revision,
                "scope_id": binding.scope_id,
                "scope_sha256": binding.scope_sha256,
                "timestamp_policy_file_sha256": binding.policy_file_sha256,
                "timestamp_policy_sha256": binding.policy.policy_sha256,
                "timestamp_policy_decision": binding.policy.decision,
                "row_exception_policy_file_sha256": (
                    row_exception_policy.file_sha256
                    if row_exception_policy is not None
                    else None
                ),
                "row_exception_id": (
                    row_exception_policy.exception_id
                    if row_exception_policy is not None
                    and binding.row.dataset_id == row_exception_policy.dataset_id
                    else None
                ),
                "source_receipt_sha256": binding.source_receipt_sha256,
                "source_tsfiles": [
                    {"path": relative, "bytes": size, "sha256": digest}
                    for relative, _path, size, digest in binding.source_files
                ],
                "exporter_sha256": binding.producer_sha256,
                "tsfile_version": REQUIRED_TSFILE_VERSION,
                "row_coverage_query": (
                    "all_schema_TAG_FIELD_ATTRIBUTE_columns_physical_Time_implicit_"
                    "except_frozen_Africa9_string_fields"
                ),
                "null_tag_encoding": "TYPE:null",
                "null_tag_values_preserved": True,
                "null_tag_rows": sum(item.null_tag_rows for item in stats),
                "null_tag_cells": sum(item.null_tag_cells for item in stats),
                "positive_infinity_encoding": "inf",
                "positive_infinity_values_preserved": True,
                "positive_infinity_cells": sum(
                    count
                    for item in stats
                    for _source_name, count in item.positive_infinity_counts
                ),
                "nan_encoding": "nan",
                "nan_values_preserved": True,
                "source_nan_cells": sum(
                    int(item["count"]) for item in source_nan_by_column
                ),
                "source_nan_by_column": source_nan_by_column,
                "dataframe_batch_rows": batch_rows,
                "sqlite_insert_batch_rows": insert_batch_rows,
                "source_rows": source_rows,
                "policy_filtered_rows": source_rows - eligible_rows,
                "timestamp_policy_filtered_rows": sum(
                    item.timestamp_policy_filtered_rows for item in stats
                ),
                "row_exception_filtered_rows": sum(
                    item.row_exception_filtered_rows for item in stats
                ),
                "series_count_discovered": series_count,
                "sqlite_output_order_index": (
                    "timestamp_unix_us,source_ordinal,source_row_ordinal"
                ),
                "source_scans": [
                    {
                        "source_path": plan.source_path,
                        "source_sha256": plan.expected_sha256,
                        "source_schema_sha256": plan.schema_sha256,
                        "table_name": plan.table_name,
                        "source_action": plan.source_action,
                        "source_rows": stat.source_rows,
                        "eligible_rows": stat.eligible_rows,
                        "filtered_rows": stat.filtered_rows,
                        "timestamp_policy_filtered_rows": (
                            stat.timestamp_policy_filtered_rows
                        ),
                        "row_exception_filtered_rows": (
                            stat.row_exception_filtered_rows
                        ),
                        "row_exception": stat.row_exception,
                        "read_mode": stat.read_mode,
                        "dataframe_attempted": stat.dataframe_attempted,
                        "rowwise_fallback": stat.rowwise_fallback,
                        "null_tag_rows": stat.null_tag_rows,
                        "null_tag_cells": stat.null_tag_cells,
                        "positive_infinity_cells": sum(
                            count for _source_name, count in stat.positive_infinity_counts
                        ),
                        "positive_infinity_by_column": (
                            _positive_infinity_receipt_rows(
                                plan, stat, column_by_key
                            )
                            if plan.source_action == "materialize"
                            else []
                        ),
                        "nan_cells": sum(count for _name, count in stat.nan_counts),
                        "nan_by_column": (
                            _nan_receipt_rows(plan, stat, column_by_key)
                            if plan.source_action == "materialize"
                            else []
                        ),
                        "temporal_empty_string_audits": list(
                            stat.temporal_empty_string_audits
                        ),
                        "query_columns": (
                            _all_query_columns(plan)
                            if plan.source_action == "materialize"
                            else []
                        ),
                        "row_coverage_exception": (
                            _africa_query_exception(
                                plan,
                                verified_source_rows=stat.source_rows,
                            )
                            if plan.source_action == "materialize"
                            else None
                        ),
                    }
                    for plan, stat in zip(plans, stats)
                ],
                "ordinary_strings_emitted_to_data_csv": False,
                "series_mapping_is_payload": False,
            }
            materialized = _materialize_capped_csvs(
                connection,
                dataset_dir,
                columns,
                dataset_byte_cap=dataset_byte_cap,
                manifest_base=manifest_base,
                disk_guard=disk_guard,
            )
        finally:
            connection.close()
    if eligible_rows != int(materialized["eligible_rows"]):
        raise AssertionError("spool row count changed during materialization")
    _emit_progress(
        "dataset_complete",
        dataset_id=binding.row.dataset_id,
        candidate_value_column_count=len(columns),
        retained_value_column_count=len(materialized["retained_value_columns"]),
        eligible_rows=eligible_rows,
        emitted_rows=int(materialized["emitted_rows"]),
        actual_capped_bytes=int(materialized["actual_capped_bytes"]),
        elapsed_seconds=round(time.monotonic() - dataset_started, 6),
    )
    return {**manifest_base, **materialized}


def _disk_gate(
    path: Path,
    binding: policy_core.DatasetBinding,
    *,
    dataset_byte_cap: int,
    min_free_temp_bytes: int,
    temp_space_factor: float,
    active_reserved_bytes: int = 0,
) -> int:
    source_bytes = sum(size for _relative, _path, size, _digest in binding.source_files)
    reservation = max(
        dataset_byte_cap * 2,
        int(math.ceil(source_bytes * temp_space_factor)),
    )
    # ``min_free_temp_bytes`` is a collection-wide reserve.  Counting it once
    # here (rather than once per worker) prevents a 10-worker run from
    # spuriously demanding 10 * 50 GB while still reserving the full projected
    # temporary footprint of every active worker.
    collection_reserve = max(dataset_byte_cap, min_free_temp_bytes)
    required = collection_reserve + active_reserved_bytes + reservation
    free = shutil.disk_usage(path).free
    if free < required:
        raise CsvExportError(
            f"insufficient free temporary space for {binding.row.dataset_id}: "
            f"free={free}, required={required}, source_bytes={source_bytes}, "
            f"active_reserved_bytes={active_reserved_bytes}, "
            f"candidate_reserved_bytes={reservation}, "
            f"collection_reserve_bytes={collection_reserve}"
        )
    return reservation


def _schema_preflight_dataset_record(
    binding: policy_core.DatasetBinding,
    plans: Sequence[policy_core.SourcePlan],
) -> Mapping[str, Any]:
    expected_paths = [relative for relative, _path, _size, _digest in binding.source_files]
    actual_paths = [plan.source_path for plan in plans]
    if actual_paths != expected_paths:
        raise CsvExportError(
            f"{binding.row.dataset_id}: schema plans do not preserve frozen source order"
        )
    sources = [
        {
            "source_path": plan.source_path,
            "source_sha256": plan.expected_sha256,
            "table_name": plan.table_name,
            "schema_sha256": plan.schema_sha256,
            "source_action": plan.source_action,
        }
        for plan in plans
    ]
    schema_sha = hashlib.sha256(_canonical_json(sources).encode("utf-8")).hexdigest()
    return {
        "dataset_id": binding.row.dataset_id,
        "repo_id": binding.row.repo_id,
        "revision": binding.row.revision,
        "status": "passed",
        "binding_sha256": binding.binding_sha256,
        "source_file_count": len(binding.source_files),
        "schema_sha256": schema_sha,
        "sources": sources,
    }


def _preflight_bindings(
    bindings: Sequence[policy_core.DatasetBinding],
    row_exception_policy: RowExceptionPolicy | None,
) -> tuple[
    tuple[tuple[policy_core.DatasetBinding, tuple[policy_core.SourcePlan, ...]], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Plan every selected source before any collection stage is created.

    All dataset errors are collected so a long all-160 preflight reports every
    incompatible schema in one pass instead of discovering them one run at a
    time.  Failed datasets never enter the returned executable plan.
    """

    planned: list[
        tuple[policy_core.DatasetBinding, tuple[policy_core.SourcePlan, ...]]
    ] = []
    records: list[Mapping[str, Any]] = []
    errors: list[Mapping[str, Any]] = []
    for binding in bindings:
        try:
            plans = tuple(policy_core._schema_for_binding(binding))
            _validate_row_exception_dataset_schema(
                binding, plans, row_exception_policy
            )
            # Exercise the exact output-column contract during preflight too;
            # this catches admitted schemas that retain no usable value.
            _output_columns(
                [plan for plan in plans if plan.source_action == "materialize"]
            )
            record = _schema_preflight_dataset_record(binding, plans)
            planned.append((binding, plans))
            records.append(record)
        except Exception as error:
            failure = {
                "dataset_id": binding.row.dataset_id,
                "repo_id": binding.row.repo_id,
                "revision": binding.row.revision,
                "status": "failed",
                "binding_sha256": binding.binding_sha256,
                "source_file_count": len(binding.source_files),
                "schema_sha256": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            records.append(failure)
            errors.append(failure)
    return tuple(planned), tuple(records), tuple(errors)


def _schema_preflight_report(
    *,
    scope_id: str,
    scope_sha256: str,
    policy_sha256: str,
    exporter_sha256: str,
    bindings: Sequence[policy_core.DatasetBinding],
    records: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA_PREFLIGHT_VERSION,
        "status": "failed" if errors else "passed",
        "scope_id": scope_id,
        "scope_sha256": scope_sha256,
        "timestamp_policy_file_sha256": policy_sha256,
        "exporter_sha256": exporter_sha256,
        "tsfile_version": REQUIRED_TSFILE_VERSION,
        "dataset_count": len(bindings),
        "source_count": sum(len(binding.source_files) for binding in bindings),
        "dataset_ids": [binding.row.dataset_id for binding in bindings],
        "datasets": list(records),
        "error_count": len(errors),
        "errors": list(errors),
    }


def _export_dataset_worker(
    binding: policy_core.DatasetBinding,
    plans: Sequence[policy_core.SourcePlan],
    output_root: Path,
    dataset_byte_cap: int,
    batch_rows: int,
    insert_batch_rows: int,
    disk_reserve_bytes: int,
    row_exception_policy: RowExceptionPolicy | None,
) -> Mapping[str, Any]:
    """Spawn-safe dataset worker; every worker owns its TsFile SDK objects."""

    diagnostics, faulthandler_log = _begin_worker_diagnostics(
        output_root, binding.row.dataset_id
    )
    try:
        result = export_dataset(
            binding,
            plans,
            output_root,
            dataset_byte_cap=dataset_byte_cap,
            batch_rows=batch_rows,
            insert_batch_rows=insert_batch_rows,
            tsfile_module=_load_tsfile_module(),
            disk_reserve_bytes=disk_reserve_bytes,
            row_exception_policy=row_exception_policy,
            worker_diagnostics=diagnostics,
        )
    except BaseException as error:
        _append_worker_diagnostic(
            diagnostics,
            "worker_task_error",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
    else:
        _append_worker_diagnostic(diagnostics, "worker_task_complete")
        return result
    finally:
        _finish_worker_diagnostics(faulthandler_log)


def _validate_workers(workers: int) -> int:
    if not 1 <= workers <= MAX_WORKERS:
        raise CsvExportError(f"workers must be in 1..{MAX_WORKERS}")
    return workers


def _run_planned_exports(
    planned: Sequence[
        tuple[policy_core.DatasetBinding, Sequence[policy_core.SourcePlan]]
    ],
    stage: Path,
    receipts: list[Mapping[str, Any]],
    *,
    workers: int,
    dataset_byte_cap: int,
    batch_rows: int,
    insert_batch_rows: int,
    min_free_temp_bytes: int,
    temp_space_factor: float,
    row_exception_policy: RowExceptionPolicy | None,
    executor_factory: Any = None,
) -> None:
    """Run isolated dataset workers while one coordinator owns publication.

    At most ``workers`` jobs are submitted.  Each admitted dataset owns a new
    one-task spawn executor so no native TsFile state can cross dataset
    boundaries, including when running with ``workers=1``.  On the first
    failure no new job is admitted, pending futures are cancelled, active
    workers are allowed to settle, and successful receipts are returned in
    frozen binding order.  Therefore the caller can move exactly one
    quiescent stage to its non-reusable failure location.
    """

    workers = _validate_workers(workers)
    disk_reserve_bytes = max(dataset_byte_cap, min_free_temp_bytes)

    def new_executor() -> Any:
        if executor_factory is not None:
            return executor_factory(1)
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )

    active: dict[Any, tuple[int, int, Any]] = {}
    completed: dict[int, Mapping[str, Any]] = {}
    failures: list[tuple[int, BaseException]] = []
    next_index = 0

    def collect(done: Iterable[Any]) -> None:
        for future in sorted(done, key=lambda item: active[item][0]):
            index, _reservation, executor = active.pop(future)
            try:
                if future.cancelled():
                    continue
                receipt = future.result()
                expected_id = planned[index][0].row.dataset_id
                if receipt.get("dataset_id") != expected_id:
                    raise CsvExportError(
                        f"worker receipt dataset mismatch: expected {expected_id!r}, "
                        f"found {receipt.get('dataset_id')!r}"
                    )
                completed[index] = receipt
            except CancelledError:
                continue
            except BaseException as error:
                failures.append((index, error))
            finally:
                try:
                    # Every executor receives exactly one task and is never
                    # recycled for a later dataset.
                    executor.shutdown(wait=True, cancel_futures=True)
                except BaseException as error:
                    failures.append((index, error))

    coordinator_error: BaseException | None = None
    try:
        while next_index < len(planned) or active:
            while (
                not failures
                and next_index < len(planned)
                and len(active) < workers
            ):
                binding, plans = planned[next_index]
                active_reserved = sum(item[1] for item in active.values())
                try:
                    reservation = _disk_gate(
                        stage,
                        binding,
                        dataset_byte_cap=dataset_byte_cap,
                        min_free_temp_bytes=min_free_temp_bytes,
                        temp_space_factor=temp_space_factor,
                        active_reserved_bytes=active_reserved,
                    )
                except CsvExportError as error:
                    if active:
                        # A running job may shortly release its spool.  Wait
                        # for progress and retry this same binding in order.
                        break
                    failures.append((next_index, error))
                    break
                executor = new_executor()
                try:
                    future = executor.submit(
                        _export_dataset_worker,
                        binding,
                        plans,
                        stage,
                        dataset_byte_cap,
                        batch_rows,
                        insert_batch_rows,
                        disk_reserve_bytes,
                        row_exception_policy,
                    )
                except BaseException as error:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    finally:
                        failures.append((next_index, error))
                    break
                active[future] = (next_index, reservation, executor)
                next_index += 1

            if failures:
                for future in active:
                    future.cancel()
                if active:
                    done, _pending = wait(tuple(active))
                    collect(done)
                break
            if not active:
                break
            done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
            collect(done)
    except BaseException as error:
        coordinator_error = error
        for future in active:
            future.cancel()
    finally:
        if active:
            # ``wait`` plus each executor's ``shutdown(wait=True)`` in collect
            # are essential: the caller may rename the stage immediately after
            # return.  No child may retain an open spool or write into the old
            # stage, including after SIGINT/KeyboardInterrupt.
            done, _pending = wait(tuple(active))
            collect(done)

    receipts.extend(completed[index] for index in sorted(completed))
    if coordinator_error is not None:
        raise coordinator_error
    if failures:
        _index, error = min(failures, key=lambda item: item[0])
        raise error


def _worker_diagnostic_receipts(stage: Path) -> list[Mapping[str, Any]]:
    directory = stage / WORKER_DIAGNOSTIC_DIRECTORY
    if not directory.exists() and not directory.is_symlink():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise CsvExportError("failed stage worker diagnostics are not a real directory")
    receipts: list[Mapping[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise CsvExportError(f"unexpected worker diagnostic object: {path}")
        receipts.append(
            {
                "path": f"{WORKER_DIAGNOSTIC_DIRECTORY}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return receipts


def _preserve_failed_stage(
    stage: Path,
    output: Path,
    receipts: Sequence[Mapping[str, Any]],
    error: BaseException,
) -> Path:
    """Keep failed-run evidence hidden and structurally impossible to pass.

    A failed stage is never reused as a checkpoint and never published under
    the requested output name.  Completed per-dataset directories remain
    available for diagnosis, while the exact collection-completion filename
    is removed if a late publication failure happened after it was written.
    """

    if not stage.is_dir() or stage.is_symlink():
        raise CsvExportError(f"failed export stage is unavailable: {stage}")
    completion = stage / "CSV_EXPORT_MANIFEST.json"
    if completion.exists() or completion.is_symlink():
        unpublished = stage / "CSV_EXPORT_UNPUBLISHED_MANIFEST.json"
        if unpublished.exists() or unpublished.is_symlink():
            raise CsvExportError("failed stage already contains an unpublished manifest")
        os.rename(completion, unpublished)

    failed_stage = stage.parent / f".{output.name}.failed-{uuid.uuid4().hex}"
    _rename_directory_noreplace(stage, failed_stage)
    completed: list[dict[str, Any]] = []
    for item in receipts:
        dataset_id = str(item.get("dataset_id", ""))
        receipt = failed_stage / dataset_id / "dataset_manifest.json"
        if not dataset_id or not receipt.is_file() or receipt.is_symlink():
            raise CsvExportError(
                "completed dataset receipt is missing while preserving failure evidence"
            )
        completed.append(
            {
                "dataset_id": dataset_id,
                "receipt": f"{dataset_id}/dataset_manifest.json",
                "receipt_sha256": _sha256(receipt),
            }
        )
    worker_diagnostics = _worker_diagnostic_receipts(failed_stage)
    dataset_directories = sorted(
        item.name
        for item in failed_stage.iterdir()
        if item.is_dir()
        and not item.is_symlink()
        and item.name not in {".spool", WORKER_DIAGNOSTIC_DIRECTORY}
    )
    failure = {
        "schema": FAILURE_VERSION,
        "status": "failed",
        "collection_complete": False,
        "published": False,
        "reusable_as_checkpoint": False,
        "intended_output": str(output.resolve()),
        "failure_stage": str(failed_stage.resolve()),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "completed_dataset_count": len(completed),
        "completed_datasets": completed,
        "dataset_directories_present": dataset_directories,
        "worker_diagnostics_present": bool(worker_diagnostics),
        "worker_diagnostics": worker_diagnostics,
        "completion_manifest_present": False,
        "unpublished_manifest_present": (
            failed_stage / "CSV_EXPORT_UNPUBLISHED_MANIFEST.json"
        ).is_file(),
    }
    _write_new_json(failed_stage / "CSV_EXPORT_FAILURE.json", failure)
    _emit_progress(
        "collection_failed",
        failure_stage=str(failed_stage.resolve()),
        intended_output=str(output.resolve()),
        completed_dataset_count=len(completed),
        error_type=type(error).__name__,
        error_message=str(error),
    )
    return failed_stage


def run_export(
    *,
    scope_path: Path,
    mirror: Path,
    policy_path: Path,
    row_exceptions_path: Path | None = None,
    output: Path,
    dataset_ids: Sequence[str] = (),
    dataset_byte_cap: int = DEFAULT_DATASET_BYTE_CAP,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    insert_batch_rows: int = DEFAULT_INSERT_BATCH_ROWS,
    min_free_temp_bytes: int = 0,
    temp_space_factor: float = 12.0,
    workers: int = DEFAULT_WORKERS,
    schema_preflight_only: bool = False,
) -> Mapping[str, Any]:
    if output.exists() or output.is_symlink():
        raise CsvExportError(f"refusing to overwrite existing output path: {output}")
    if not 0 < dataset_byte_cap <= DEFAULT_DATASET_BYTE_CAP:
        raise CsvExportError(
            f"dataset byte cap must be in 1..{DEFAULT_DATASET_BYTE_CAP}"
        )
    if batch_rows <= 0 or insert_batch_rows <= 0:
        raise CsvExportError("batch sizes must be positive")
    if temp_space_factor <= 0:
        raise CsvExportError("temp space factor must be positive")
    workers = _validate_workers(workers)
    _load_tsfile_module()
    scope, rows, scope_sha = policy_core.load_scope(scope_path)
    policies, policy_sha = policy_core.load_policies(
        policy_path,
        scope_id=str(scope["scope_id"]),
        scope_sha256=scope_sha,
        scope_rows=rows,
    )
    _check_frozen_policy_closure(policy_path, policies)
    row_exception_policy = _load_row_exception_policy(
        row_exceptions_path,
        scope_sha256=scope_sha,
        timestamp_policy_file_sha256=policy_sha,
    )
    producer = Path(__file__).resolve()
    exporter_sha = _sha256(producer)
    # Single-dataset attempts still validate the global frozen manifest, but
    # must not reread the entire 10GB mirror 160 times.  Bind/hash the selected
    # sources and the complete macro exception source set before filtering.
    requested = list(dataset_ids)
    bind_dataset_ids = None
    if requested:
        if len(requested) != len(set(requested)):
            raise CsvExportError("--dataset values must be unique")
        unknown = sorted(set(requested) - {row.dataset_id for row in rows})
        if unknown:
            raise CsvExportError(f"unknown requested datasets: {unknown}")
        bind_dataset_ids = list(requested)
        if row_exception_policy is not None and row_exception_policy.dataset_id not in bind_dataset_ids:
            bind_dataset_ids.append(row_exception_policy.dataset_id)
    bindings = policy_core.bind_mirror(
        mirror,
        scope=scope,
        rows=rows,
        scope_sha256=scope_sha,
        policies=policies,
        policy_file_sha256=policy_sha,
        dataset_byte_cap=dataset_byte_cap,
        chunk_byte_cap=1,
        metadata_compressed_byte_cap=1,
        storage_layout="legacy_v1",
        producer_sha256=exporter_sha,
        helper_timeseries_sha256=_sha256(policy_core.HELPER_TIMESERIES_PATH),
        helper_tsfile_sha256=_sha256(policy_core.HELPER_TSFILE_PATH),
        dataset_ids=bind_dataset_ids,
    )
    _bind_row_exception_to_bindings(row_exception_policy, bindings)
    if requested:
        if len(requested) != len(set(requested)):
            raise CsvExportError("--dataset values must be unique")
        by_id = {item.row.dataset_id: item for item in bindings}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise CsvExportError(f"unknown requested datasets: {unknown}")
        bindings = tuple(by_id[item] for item in requested)
    parent = output.resolve().parent
    if not parent.is_dir():
        raise CsvExportError(
            f"output parent must already exist as a trusted directory: {parent}"
        )
    planned, preflight_records, preflight_errors = _preflight_bindings(
        bindings, row_exception_policy
    )
    preflight_report = _schema_preflight_report(
        scope_id=str(scope["scope_id"]),
        scope_sha256=scope_sha,
        policy_sha256=policy_sha,
        exporter_sha256=exporter_sha,
        bindings=bindings,
        records=preflight_records,
        errors=preflight_errors,
    )
    if schema_preflight_only:
        exporter_after = _sha256(producer)
        if exporter_after != exporter_sha:
            raise CsvExportError("exporter source changed during schema preflight")
        _write_new_json(output, preflight_report)
        if preflight_errors:
            failed_ids = [str(item["dataset_id"]) for item in preflight_errors]
            raise CsvExportError(
                f"schema preflight failed for {len(failed_ids)} datasets: {failed_ids}"
            )
        return preflight_report
    if preflight_errors:
        failed_ids = [str(item["dataset_id"]) for item in preflight_errors]
        raise CsvExportError(
            f"schema preflight failed for {len(failed_ids)} datasets: {failed_ids}"
        )

    # Check every dataset's standalone temporary-space requirement before
    # creating the collection stage.  Aggregate active-worker reservations
    # are enforced again dynamically by the scheduler.
    for binding, _plans in planned:
        _disk_gate(
            parent,
            binding,
            dataset_byte_cap=dataset_byte_cap,
            min_free_temp_bytes=min_free_temp_bytes,
            temp_space_factor=temp_space_factor,
        )
    stage = parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir()
    receipts: list[Mapping[str, Any]] = []
    try:
        _run_planned_exports(
            planned,
            stage,
            receipts,
            workers=workers,
            dataset_byte_cap=dataset_byte_cap,
            batch_rows=batch_rows,
            insert_batch_rows=insert_batch_rows,
            min_free_temp_bytes=min_free_temp_bytes,
            temp_space_factor=temp_space_factor,
            row_exception_policy=row_exception_policy,
        )
        spool = stage / ".spool"
        if spool.exists():
            spool.rmdir()
        _remove_success_worker_diagnostics(stage)
        manifest = {
            "schema": COLLECTION_VERSION,
            "status": "complete",
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_sha,
            "timestamp_policy_file_sha256": policy_sha,
            "row_exception_policy_file_sha256": (
                row_exception_policy.file_sha256
                if row_exception_policy is not None
                else None
            ),
            "tsfile_version": REQUIRED_TSFILE_VERSION,
            "schema_preflight_version": SCHEMA_PREFLIGHT_VERSION,
            "schema_preflight_dataset_count": len(preflight_records),
            "schema_preflight_source_count": preflight_report["source_count"],
            "workers": workers,
            "dataset_byte_cap": dataset_byte_cap,
            "dataset_count": len(receipts),
            "datasets": [
                {
                    "dataset_id": item["dataset_id"],
                    "receipt": f"{item['dataset_id']}/dataset_manifest.json",
                    "actual_capped_bytes": item["actual_capped_bytes"],
                    "emitted_rows": item["emitted_rows"],
                    "lost_rows_due_to_byte_cap": item["lost_rows_due_to_byte_cap"],
                }
                for item in receipts
            ],
        }
        exporter_after = _sha256(producer)
        if exporter_after != exporter_sha:
            raise CsvExportError("exporter source changed during the run")
        _write_new_json(stage / "CSV_EXPORT_MANIFEST.json", manifest)
        _rename_directory_noreplace(stage, output)
        return manifest
    except BaseException as error:
        if stage.exists():
            try:
                _preserve_failed_stage(stage, output, receipts, error)
            except Exception as preservation_error:
                print(
                    "ERROR: failed to write complete failure-stage receipt; "
                    f"stage remains at {stage.resolve()}: {preservation_error}",
                    file=sys.stderr,
                    flush=True,
                )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--timestamp-policies", type=Path, required=True)
    parser.add_argument("--row-exceptions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema-preflight-only",
        action="store_true",
        help=(
            "validate every selected frozen source schema and write the fresh "
            "JSON report to --output without creating an export stage"
        ),
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dataset-byte-cap", type=int, default=DEFAULT_DATASET_BYTE_CAP)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument(
        "--insert-batch-rows", type=int, default=DEFAULT_INSERT_BATCH_ROWS
    )
    parser.add_argument("--min-free-temp-bytes", type=int, default=0)
    parser.add_argument("--temp-space-factor", type=float, default=12.0)
    args = parser.parse_args(argv)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum: int, _frame: Any) -> None:
        raise _TerminationRequested(signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            result = run_export(
                scope_path=args.scope,
                mirror=args.mirror,
                policy_path=args.timestamp_policies,
                row_exceptions_path=args.row_exceptions,
                output=args.output,
                dataset_ids=args.dataset,
                dataset_byte_cap=args.dataset_byte_cap,
                batch_rows=args.batch_rows,
                insert_batch_rows=args.insert_batch_rows,
                min_free_temp_bytes=args.min_free_temp_bytes,
                temp_space_factor=args.temp_space_factor,
                workers=args.workers,
                schema_preflight_only=args.schema_preflight_only,
            )
        except _TerminationRequested as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 128 + error.signum
        except (
            OSError,
            ValueError,
            sqlite3.Error,
            PreparationError,
            json.JSONDecodeError,
        ) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
