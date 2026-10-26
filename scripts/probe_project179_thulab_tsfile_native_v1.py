#!/usr/bin/env python3
"""Read-only, crash-isolated probe for the pinned native TsFile query paths.

This diagnostic deliberately does not contain the exporter's DataFrame-to-row
fallback.  ``dataframe`` mode exercises the exact native fast path, while
``rowwise`` mode exercises ``is_null_by_index``/``get_value_by_index`` over the
same projection.  Run one source per fresh OS process so a native crash is
attributable to a single immutable source object.

The source is never opened for writing.  A new JSONL progress log is fsynced
after every event, and Python's faulthandler writes to a separate new file.
Both diagnostic paths must be absent before the probe starts.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import signal
import stat
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence, TextIO


REQUIRED_TSFILE_VERSION = "2.1.5"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
DEFAULT_BATCH_ROWS = 8192
ALLOWED_CATEGORIES = frozenset({"TAG", "FIELD", "ATTRIBUTE"})


class ProbeError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _type_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _category_name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _schema_columns(schema: Any) -> list[dict[str, str]]:
    columns = (
        schema.get_columns()
        if hasattr(schema, "get_columns")
        else getattr(schema, "columns", ())
    )
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, column in enumerate(columns):
        name = str(
            column.get_column_name()
            if hasattr(column, "get_column_name")
            else getattr(column, "column_name", "")
        )
        dtype = _type_name(
            column.get_data_type()
            if hasattr(column, "get_data_type")
            else getattr(column, "data_type", "")
        )
        category = _category_name(
            column.get_category()
            if hasattr(column, "get_category")
            else getattr(column, "category", "")
        )
        if not name or name in seen or category not in ALLOWED_CATEGORIES:
            raise ProbeError(
                f"invalid schema column {index}: name={name!r}, "
                f"type={dtype!r}, category={category!r}"
            )
        seen.add(name)
        result.append({"name": name, "tsfile_type": dtype, "category": category})
    if not result:
        raise ProbeError("selected table has no TAG/FIELD/ATTRIBUTE schema columns")
    return result


def _new_log(path: Path, role: str) -> TextIO:
    parent = path.resolve().parent
    if not parent.is_dir() or parent.is_symlink():
        raise ProbeError(f"{role} parent must be an existing real directory: {parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ProbeError(f"cannot create new {role}: {path}: {error}") from error
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


class _DurableJsonl:
    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.sequence = 0

    def emit(self, stage: str, **payload: Any) -> None:
        row = {
            "schema": "project179-thulab-tsfile-native-probe-event-v1",
            "sequence": self.sequence,
            "pid": os.getpid(),
            "stage": stage,
            **payload,
        }
        self.sequence += 1
        self.handle.write(_canonical_json(row) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def _regular_source(path: Path) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ProbeError(f"source is unavailable: {path}: {error}") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ProbeError(f"source must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise ProbeError(f"source path must not traverse symlinks: {path} -> {resolved}")
    return resolved


def _select_table(schemas: Any, requested: str | None) -> tuple[str, Any]:
    if not isinstance(schemas, Mapping) or not schemas:
        raise ProbeError("TsFile reader returned no table schemas")
    if requested is None:
        if len(schemas) != 1:
            raise ProbeError(
                "--table is required for a multi-table source; available="
                f"{sorted(map(str, schemas))}"
            )
        key = next(iter(schemas))
        return str(key), schemas[key]
    if requested not in schemas:
        raise ProbeError(
            f"table {requested!r} is absent; available={sorted(map(str, schemas))}"
        )
    return requested, schemas[requested]


def _select_columns(
    schema_rows: Sequence[Mapping[str, str]], requested: Sequence[str]
) -> list[str]:
    available = {str(row["name"]) for row in schema_rows}
    if requested:
        if len(requested) != len(set(requested)):
            raise ProbeError("--column values must be unique")
        missing = [name for name in requested if name not in available]
        if missing:
            raise ProbeError(f"requested columns are absent from schema: {missing}")
        selected = list(requested)
    else:
        # Match the exporter: physical Time is implicit in query results.
        selected = [str(row["name"]) for row in schema_rows if row["name"] != "Time"]
    if not selected:
        raise ProbeError("query projection is empty")
    return selected


def _metadata(result: Any, columns: Sequence[str]) -> tuple[list[str], list[str]]:
    metadata = result.get_metadata()
    names = [str(value) for value in metadata.get_column_list()]
    types = [_type_name(value) for value in metadata.get_data_type_list()]
    if (
        not names
        or names[0].casefold() != "time"
        or types[0] != "INT64"
        or names[1:] != list(columns)
        or len(types) != len(names)
    ):
        raise ProbeError(
            f"query metadata mismatch: names={names!r}, types={types!r}, "
            f"requested={list(columns)!r}"
        )
    return names, types


def _scan_dataframe(
    result: Any,
    names: Sequence[str],
    logger: _DurableJsonl,
    *,
    batch_rows: int,
    max_rows: int,
) -> int:
    rows = 0
    batches = 0
    while max_rows == 0 or rows < max_rows:
        if not result.next():
            break
        request_rows = batch_rows if max_rows == 0 else min(batch_rows, max_rows - rows)
        frame = result.read_data_frame(request_rows)
        if frame is None or len(frame) == 0:
            raise ProbeError("read_data_frame returned empty after next() succeeded")
        frame_names = [str(value) for value in frame.columns]
        if frame_names != list(names):
            raise ProbeError(
                f"data-frame columns differ: {frame_names!r} != {list(names)!r}"
            )
        batch_count = len(frame)
        rows += batch_count
        batches += 1
        logger.emit(
            "dataframe_batch_complete",
            batch_index=batches - 1,
            batch_rows=batch_count,
            rows_complete=rows,
        )
    return rows


def _scan_rowwise(
    result: Any,
    names: Sequence[str],
    logger: _DurableJsonl,
    *,
    batch_rows: int,
    max_rows: int,
) -> int:
    rows = 0
    null_cells = 0
    while (max_rows == 0 or rows < max_rows) and result.next():
        for index in range(1, len(names) + 1):
            is_null = bool(result.is_null_by_index(index))
            if is_null:
                null_cells += 1
            else:
                # Fetch every projected value to exercise native conversion.
                result.get_value_by_index(index)
        rows += 1
        if rows % batch_rows == 0:
            logger.emit(
                "rowwise_batch_complete",
                batch_index=(rows // batch_rows) - 1,
                batch_rows=batch_rows,
                rows_complete=rows,
                null_cells=null_cells,
            )
    if rows % batch_rows:
        logger.emit(
            "rowwise_batch_complete",
            batch_index=rows // batch_rows,
            batch_rows=rows % batch_rows,
            rows_complete=rows,
            null_cells=null_cells,
        )
    return rows


def run_probe(
    *,
    dataset: str,
    source: Path,
    expected_sha256: str,
    table: str | None,
    columns: Sequence[str],
    mode: str,
    batch_rows: int,
    max_rows: int,
    logger: _DurableJsonl,
) -> None:
    source = _regular_source(source)
    if batch_rows <= 0:
        raise ProbeError("--batch-rows must be positive")
    if max_rows < 0:
        raise ProbeError("--max-rows must be non-negative (zero means full scan)")
    try:
        installed = package_version("tsfile")
    except PackageNotFoundError as error:
        raise ProbeError("the pinned tsfile package is not installed") from error
    if installed != REQUIRED_TSFILE_VERSION:
        raise ProbeError(
            f"tsfile version drift: required {REQUIRED_TSFILE_VERSION}, found {installed}"
        )
    try:
        import tsfile  # type: ignore
    except Exception as error:
        raise ProbeError(f"cannot import tsfile {installed}: {error}") from error

    before_sha = _sha256(source)
    if before_sha != expected_sha256:
        raise ProbeError(
            f"source SHA-256 mismatch: expected={expected_sha256}, actual={before_sha}"
        )
    logger.emit(
        "probe_start",
        dataset=dataset,
        mode=mode,
        source=str(source),
        source_bytes=source.stat().st_size,
        source_sha256=before_sha,
        tsfile_version=installed,
        batch_rows=batch_rows,
        max_rows=max_rows,
    )
    reader = result = None
    try:
        logger.emit("reader_open_start")
        reader = tsfile.TsFileReader(str(source))
        logger.emit("reader_open_complete")
        logger.emit("schema_read_start")
        table_name, schema = _select_table(reader.get_all_table_schemas(), table)
        schema_rows = _schema_columns(schema)
        schema_sha = hashlib.sha256(_canonical_json(schema_rows).encode("utf-8")).hexdigest()
        query_columns = _select_columns(schema_rows, columns)
        logger.emit(
            "schema_read_complete",
            table_name=table_name,
            schema_sha256=schema_sha,
            schema_columns=schema_rows,
            query_columns=query_columns,
        )
        logger.emit("query_open_start", table_name=table_name, query_columns=query_columns)
        result = reader.query_table(table_name, query_columns, INT64_MIN, INT64_MAX)
        logger.emit("query_open_complete")
        names, types = _metadata(result, query_columns)
        logger.emit("query_metadata_complete", result_columns=names, result_types=types)
        logger.emit("scan_start", mode=mode)
        if mode == "dataframe":
            rows = _scan_dataframe(
                result,
                names,
                logger,
                batch_rows=batch_rows,
                max_rows=max_rows,
            )
        elif mode == "rowwise":
            rows = _scan_rowwise(
                result,
                names,
                logger,
                batch_rows=batch_rows,
                max_rows=max_rows,
            )
        else:  # argparse prevents this path; retain fail-closed library behavior.
            raise ProbeError(f"unsupported mode: {mode}")
        logger.emit("scan_complete", mode=mode, rows_complete=rows)
        logger.emit("result_close_start")
        closing_result = result
        result = None
        closing_result.close()
        logger.emit("result_close_complete")
        logger.emit("reader_close_start")
        closing_reader = reader
        reader = None
        closing_reader.close()
        logger.emit("reader_close_complete")
    finally:
        if result is not None:
            closing_result = result
            result = None
            logger.emit("result_close_finally_start")
            closing_result.close()
            logger.emit("result_close_finally_complete")
        if reader is not None:
            closing_reader = reader
            reader = None
            logger.emit("reader_close_finally_start")
            closing_reader.close()
            logger.emit("reader_close_finally_complete")
    after_sha = _sha256(source)
    if after_sha != before_sha:
        raise ProbeError(
            f"source changed during read-only probe: before={before_sha}, after={after_sha}"
        )
    logger.emit("probe_complete", source_sha256=after_sha)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--expected-sha",
        required=True,
        help="lowercase SHA-256 of the immutable source",
    )
    parser.add_argument("--table")
    parser.add_argument(
        "--column",
        dest="columns",
        action="append",
        default=[],
        help="query one exact schema column; repeat to control projection/order; omit for all",
    )
    parser.add_argument("--mode", choices=("dataframe", "rowwise"), required=True)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="stop after this many rows; zero performs a full scan",
    )
    parser.add_argument("--jsonl-log", type=Path, required=True)
    parser.add_argument("--faulthandler-log", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    json_handle: TextIO | None = None
    fault_handle: TextIO | None = None
    logger: _DurableJsonl | None = None
    try:
        expected_sha256 = args.expected_sha.strip().lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ProbeError("--expected-sha must be exactly 64 hexadecimal characters")
        if not args.dataset.strip():
            raise ProbeError("--dataset must not be empty")
        source_absolute = args.source.absolute()
        if source_absolute in {
            args.jsonl_log.absolute(), args.faulthandler_log.absolute()
        }:
            raise ProbeError("diagnostic logs must not name the raw source")
        if args.jsonl_log.absolute() == args.faulthandler_log.absolute():
            raise ProbeError("JSONL and faulthandler logs must be different files")
        json_handle = _new_log(args.jsonl_log, "JSONL log")
        logger = _DurableJsonl(json_handle)
        fault_handle = _new_log(args.faulthandler_log, "faulthandler log")
        faulthandler.enable(file=fault_handle, all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=fault_handle, all_threads=True)
        run_probe(
            dataset=args.dataset,
            source=args.source,
            expected_sha256=expected_sha256,
            table=args.table,
            columns=args.columns,
            mode=args.mode,
            batch_rows=args.batch_rows,
            max_rows=args.max_rows,
            logger=logger,
        )
        return 0
    except BaseException as error:
        if logger is not None:
            try:
                logger.emit(
                    "python_exception",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback=traceback.format_exc(),
                )
            except Exception:
                pass
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        if hasattr(signal, "SIGUSR1"):
            try:
                faulthandler.unregister(signal.SIGUSR1)
            except Exception:
                pass
        try:
            faulthandler.disable()
        except Exception:
            pass
        if fault_handle is not None:
            fault_handle.flush()
            os.fsync(fault_handle.fileno())
            fault_handle.close()
        if json_handle is not None:
            json_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
