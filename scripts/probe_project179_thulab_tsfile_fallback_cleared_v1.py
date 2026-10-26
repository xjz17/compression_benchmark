#!/usr/bin/env python3
"""Cleared-lifetime control for the TsFile DataFrame-to-rowwise fallback.

Each cycle uses one process and one immutable source.  It opens a DataFrame
query and runs until the exact exporter fallback condition.  Unlike the
original reproducer, it completely exits the exception handler, clears the
exception/traceback chain (including DataFrame-frame locals), and forces
garbage collection before closing the result and reader.  It then reopens the
same source and performs the identical full rowwise scan.  No converted output
is written.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import signal
import stat
import sys
import traceback
from typing import Any, Mapping, Sequence, TextIO


REQUIRED_TSFILE_VERSION = "2.1.5"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
ALLOWED_CATEGORIES = frozenset({"TAG", "FIELD", "ATTRIBUTE"})


class ProbeError(RuntimeError):
    pass


class _DataFrameFallback(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_file(path: Path, role: str) -> TextIO:
    absolute = path.absolute()
    parent = absolute.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent:
        raise ProbeError(f"{role} parent must be an existing real directory: {parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags, 0o600)
    except OSError as error:
        raise ProbeError(f"cannot create new {role} {absolute}: {error}") from error
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


class _Log:
    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.sequence = 0

    def emit(self, stage: str, **values: Any) -> None:
        row = {
            "schema": "project179-thulab-tsfile-fallback-cleared-probe-event-v1",
            "sequence": self.sequence,
            "pid": os.getpid(),
            "stage": stage,
            **values,
        }
        self.sequence += 1
        self.handle.write(_json(row) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def _source(path: Path) -> Path:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ProbeError(f"source unavailable: {path}: {error}") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ProbeError(f"source must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise ProbeError(f"source path traverses a symlink: {path} -> {resolved}")
    return resolved


def _name(value: Any) -> str:
    return str(getattr(value, "name", value)).upper()


def _schema_rows(schema: Any) -> list[dict[str, str]]:
    columns = (
        schema.get_columns()
        if hasattr(schema, "get_columns")
        else getattr(schema, "columns", ())
    )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for column in columns:
        name = str(
            column.get_column_name()
            if hasattr(column, "get_column_name")
            else getattr(column, "column_name", "")
        )
        dtype = _name(
            column.get_data_type()
            if hasattr(column, "get_data_type")
            else getattr(column, "data_type", "")
        )
        category = _name(
            column.get_category()
            if hasattr(column, "get_category")
            else getattr(column, "category", "")
        )
        if not name or name in seen or category not in ALLOWED_CATEGORIES:
            raise ProbeError(
                f"invalid schema column: name={name!r}, type={dtype!r}, "
                f"category={category!r}"
            )
        seen.add(name)
        rows.append({"name": name, "tsfile_type": dtype, "category": category})
    if not rows:
        raise ProbeError("table has no TAG/FIELD/ATTRIBUTE columns")
    return rows


def _plan(reader: Any, table: str | None, columns: Sequence[str]) -> tuple[str, list[dict[str, str]], list[str]]:
    schemas = reader.get_all_table_schemas()
    if not isinstance(schemas, Mapping) or not schemas:
        raise ProbeError("reader returned no table schemas")
    if table is None:
        if len(schemas) != 1:
            raise ProbeError(f"--table required; available={sorted(map(str, schemas))}")
        table_key = next(iter(schemas))
        table_name = str(table_key)
        schema = schemas[table_key]
    else:
        if table not in schemas:
            raise ProbeError(f"table {table!r} absent; available={sorted(map(str, schemas))}")
        table_name, schema = table, schemas[table]
    rows = _schema_rows(schema)
    available = {row["name"] for row in rows}
    if columns:
        if len(columns) != len(set(columns)):
            raise ProbeError("--column values must be unique")
        missing = [name for name in columns if name not in available]
        if missing:
            raise ProbeError(f"columns absent from schema: {missing}")
        query_columns = list(columns)
    else:
        query_columns = [row["name"] for row in rows if row["name"] != "Time"]
    if not query_columns:
        raise ProbeError("query projection is empty")
    return table_name, rows, query_columns


def _metadata(result: Any, columns: Sequence[str]) -> list[str]:
    metadata = result.get_metadata()
    names = [str(value) for value in metadata.get_column_list()]
    types = [_name(value) for value in metadata.get_data_type_list()]
    if (
        not names
        or names[0].casefold() != "time"
        or types[0] != "INT64"
        or names[1:] != list(columns)
        or len(types) != len(names)
    ):
        raise ProbeError(f"query metadata mismatch: names={names!r}, types={types!r}")
    return names


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        result = pd.isna(value)
        return bool(result) if getattr(result, "shape", ()) == () else False
    except Exception:
        return False


def _dataframe_attempt(
    result: Any,
    names: Sequence[str],
    log: _Log,
    cycle: int,
    batch_rows: int,
    max_rows: int,
) -> int:
    rows = 0
    batches = 0
    while max_rows == 0 or rows < max_rows:
        log.emit("dataframe_next_start", cycle=cycle, rows_complete=rows)
        if not result.next():
            break
        request = batch_rows if max_rows == 0 else min(batch_rows, max_rows - rows)
        log.emit("dataframe_read_start", cycle=cycle, rows_complete=rows, request_rows=request)
        try:
            frame = result.read_data_frame(request)
        except Exception as error:
            raise _DataFrameFallback(
                f"read_data_frame failed: {type(error).__name__}: {error}"
            ) from error
        if frame is None or len(frame) == 0:
            raise ProbeError("read_data_frame returned empty after next()")
        if [str(value) for value in frame.columns] != list(names):
            raise ProbeError("DataFrame columns differ from query metadata")
        tuples = list(frame.itertuples(index=False, name=None))
        rows += len(tuples)
        batches += 1
        log.emit(
            "dataframe_batch_materialized",
            cycle=cycle,
            batch_index=batches - 1,
            batch_rows=len(tuples),
            rows_complete=rows,
        )
        if any(_missing(value) for row in tuples for value in row):
            raise _DataFrameFallback("data frame contains a null/NaN marker")
    return rows


def _rowwise_scan(
    result: Any,
    names: Sequence[str],
    log: _Log,
    cycle: int,
    batch_rows: int,
    max_rows: int,
) -> tuple[int, int]:
    rows = 0
    null_cells = 0
    log.emit("rowwise_next_start", cycle=cycle, rows_complete=0)
    while (max_rows == 0 or rows < max_rows) and result.next():
        if rows == 0:
            log.emit("rowwise_current_row_access_start", cycle=cycle)
        for index in range(1, len(names) + 1):
            if result.is_null_by_index(index):
                null_cells += 1
            else:
                result.get_value_by_index(index)
        rows += 1
        if rows % batch_rows == 0:
            log.emit(
                "rowwise_batch_complete",
                cycle=cycle,
                rows_complete=rows,
                null_cells=null_cells,
            )
            log.emit("rowwise_next_start", cycle=cycle, rows_complete=rows)
    if rows % batch_rows:
        log.emit(
            "rowwise_batch_complete",
            cycle=cycle,
            rows_complete=rows,
            null_cells=null_cells,
        )
    return rows, null_cells


def _clear_exception_chain(error: BaseException) -> tuple[int, int]:
    """Sever traceback chains and clear completed frames before native close."""
    exception_count = 0
    frame_count = 0
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        exception_count += 1
        next_error = current.__cause__ or current.__context__
        tb = current.__traceback__
        cursor = tb
        while cursor is not None:
            frame_count += 1
            cursor = cursor.tb_next
        if tb is not None:
            traceback.clear_frames(tb)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        current = next_error
    return exception_count, frame_count


def _close(result: Any, reader: Any, log: _Log, cycle: int, prefix: str) -> None:
    if result is not None:
        log.emit(f"{prefix}_result_close_start", cycle=cycle)
        result.close()
        log.emit(f"{prefix}_result_close_complete", cycle=cycle)
    if reader is not None:
        log.emit(f"{prefix}_reader_close_start", cycle=cycle)
        reader.close()
        log.emit(f"{prefix}_reader_close_complete", cycle=cycle)


def _rowwise_reopen(
    tsfile: Any,
    source: Path,
    table: str,
    columns: Sequence[str],
    expected_names: Sequence[str],
    log: _Log,
    cycle: int,
    batch_rows: int,
    max_rows: int,
) -> tuple[int, int]:
    reader = result = None
    try:
        log.emit("rowwise_reader_open_start", cycle=cycle)
        reader = tsfile.TsFileReader(str(source))
        log.emit("rowwise_reader_open_complete", cycle=cycle)
        log.emit("rowwise_query_open_start", cycle=cycle)
        result = reader.query_table(table, list(columns), INT64_MIN, INT64_MAX)
        log.emit("rowwise_query_open_complete", cycle=cycle)
        names = _metadata(result, columns)
        if names != list(expected_names):
            raise ProbeError("rowwise metadata drifted from DataFrame query")
        return _rowwise_scan(result, names, log, cycle, batch_rows, max_rows)
    finally:
        closing_result, closing_reader = result, reader
        result = reader = None
        _close(closing_result, closing_reader, log, cycle, "rowwise")


def run(args: argparse.Namespace, log: _Log) -> None:
    source = _source(args.source)
    expected_sha = args.expected_sha.strip().lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise ProbeError("--expected-sha must be exactly 64 hexadecimal characters")
    before_sha = _sha256(source)
    if before_sha != expected_sha:
        raise ProbeError(f"source SHA mismatch: expected={expected_sha}, actual={before_sha}")
    try:
        installed = package_version("tsfile")
    except PackageNotFoundError as error:
        raise ProbeError("tsfile package is absent") from error
    if installed != REQUIRED_TSFILE_VERSION:
        raise ProbeError(f"tsfile version drift: expected 2.1.5, found {installed}")
    import tsfile  # type: ignore

    log.emit(
        "probe_start",
        dataset=args.dataset,
        source=str(source),
        source_sha256=before_sha,
        source_bytes=source.stat().st_size,
        cycles=args.cycles,
        batch_rows=args.batch_rows,
        max_rows=args.max_rows,
        tsfile_version=installed,
        variant="cleared_exception_lifetime",
    )
    for cycle in range(args.cycles):
        reader = result = None
        log.emit("cycle_start", cycle=cycle)
        try:
            log.emit("dataframe_reader_open_start", cycle=cycle)
            reader = tsfile.TsFileReader(str(source))
            log.emit("dataframe_reader_open_complete", cycle=cycle)
            table, schema_rows, columns = _plan(reader, args.table, args.columns)
            schema_sha = hashlib.sha256(_json(schema_rows).encode("utf-8")).hexdigest()
            log.emit(
                "schema_complete",
                cycle=cycle,
                table_name=table,
                schema_sha256=schema_sha,
                schema_columns=schema_rows,
                query_columns=columns,
            )
            log.emit("dataframe_query_open_start", cycle=cycle)
            result = reader.query_table(table, columns, INT64_MIN, INT64_MAX)
            log.emit("dataframe_query_open_complete", cycle=cycle)
            names = _metadata(result, columns)
            log.emit("dataframe_scan_start", cycle=cycle, result_columns=names)
            fallback_details: dict[str, str] | None = None
            fallback_error: _DataFrameFallback | None = None
            try:
                dataframe_rows = _dataframe_attempt(
                    result, names, log, cycle, args.batch_rows, args.max_rows
                )
            except _DataFrameFallback as fallback:
                # Retain only plain-text evidence after this handler exits.  The
                # exception object is cleared below, outside the active except.
                fallback_details = {
                    "error_type": type(fallback).__name__,
                    "error_message": str(fallback),
                    "traceback": traceback.format_exc(),
                }
                fallback_error = fallback

            if fallback_details is not None:
                if fallback_error is None:
                    raise ProbeError("fallback details exist without exception object")
                cleared_exceptions, cleared_frames = _clear_exception_chain(
                    fallback_error
                )
                fallback_error = None
                collected_objects = gc.collect()
                exception_state_clear = all(value is None for value in sys.exc_info())
                log.emit(
                    "dataframe_fallback",
                    cycle=cycle,
                    **fallback_details,
                )
                log.emit(
                    "dataframe_fallback_lifetime_cleared",
                    cycle=cycle,
                    variant="cleared_exception_lifetime",
                    cleared_exceptions=cleared_exceptions,
                    cleared_traceback_frames=cleared_frames,
                    gc_collected_objects=collected_objects,
                    exception_state_clear=exception_state_clear,
                )
                if not exception_state_clear:
                    raise ProbeError("Python exception state remained active after handler")
                closing_result, closing_reader = result, reader
                result = reader = None
                _close(closing_result, closing_reader, log, cycle, "dataframe_fallback")
                rowwise_rows, null_cells = _rowwise_reopen(
                    tsfile,
                    source,
                    table,
                    columns,
                    names,
                    log,
                    cycle,
                    args.batch_rows,
                    args.max_rows,
                )
                log.emit(
                    "cycle_complete",
                    cycle=cycle,
                    dataframe_outcome="fallback",
                    dataframe_rows=None,
                    rowwise_rows=rowwise_rows,
                    rowwise_null_cells=null_cells,
                )
            else:
                log.emit("dataframe_scan_complete", cycle=cycle, rows=dataframe_rows)
                closing_result, closing_reader = result, reader
                result = reader = None
                _close(closing_result, closing_reader, log, cycle, "dataframe_success")
                rowwise_rows, null_cells = _rowwise_reopen(
                    tsfile,
                    source,
                    table,
                    columns,
                    names,
                    log,
                    cycle,
                    args.batch_rows,
                    args.max_rows,
                )
                log.emit(
                    "cycle_complete",
                    cycle=cycle,
                    dataframe_outcome="complete",
                    dataframe_rows=dataframe_rows,
                    rowwise_rows=rowwise_rows,
                    rowwise_null_cells=null_cells,
                )
        finally:
            closing_result, closing_reader = result, reader
            result = reader = None
            _close(closing_result, closing_reader, log, cycle, "cycle_finally")
        if _sha256(source) != before_sha:
            raise ProbeError(f"source changed after cycle {cycle}")
        log.emit("cycle_source_hash_verified", cycle=cycle, source_sha256=before_sha)
    log.emit("probe_complete", cycles_complete=args.cycles, source_sha256=before_sha)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--table")
    parser.add_argument("--column", dest="columns", action="append", default=[])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--fault-log", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.dataset.strip():
        parser.error("--dataset must not be empty")
    if args.cycles <= 0 or args.batch_rows <= 0 or args.max_rows < 0:
        parser.error("cycles/batch-rows must be positive and max-rows non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    events = fault = None
    log = None
    try:
        if args.source.absolute() in {args.events.absolute(), args.fault_log.absolute()}:
            raise ProbeError("logs must not name the raw source")
        if args.events.absolute() == args.fault_log.absolute():
            raise ProbeError("events and fault log paths must differ")
        events = _new_file(args.events, "events log")
        log = _Log(events)
        fault = _new_file(args.fault_log, "faulthandler log")
        faulthandler.enable(file=fault, all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=fault, all_threads=True)
        run(args, log)
        return 0
    except BaseException as error:
        if log is not None:
            try:
                log.emit(
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
        if fault is not None:
            fault.flush()
            os.fsync(fault.fileno())
            fault.close()
        if events is not None:
            events.close()


if __name__ == "__main__":
    raise SystemExit(main())
