#!/usr/bin/env python3
"""Read-only same-process native-reader reuse probe for finance1k sources.

The frozen schema preflight supplies source order, action, hashes, table names,
and schema hashes.  Exactly the 892 materialized finance1k sources are scanned
in that order in one OS process by the sibling fallback lifecycle probe.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, TextIO

import probe_project179_thulab_tsfile_fallback_v1 as fallback


SCHEMA_PREFLIGHT = "project179-thulab-numeric-wide-csv-schema-preflight-v1"
EXPECTED_DATASET = "finance1k"
EXPECTED_SOURCE_COUNT = 893
EXPECTED_MATERIALIZE_COUNT = 892


class ReuseProbeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, role: str) -> Path:
    absolute = path.absolute()
    if not absolute.is_file() or absolute.is_symlink() or absolute.resolve() != absolute:
        raise ReuseProbeError(f"{role} must be a regular real file: {absolute}")
    return absolute


def _real_directory(path: Path, role: str) -> Path:
    absolute = path.absolute()
    if not absolute.is_dir() or absolute.is_symlink() or absolute.resolve() != absolute:
        raise ReuseProbeError(f"{role} must be an existing real directory: {absolute}")
    return absolute


def _required_text(item: Mapping[str, Any], key: str, context: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ReuseProbeError(f"{context}.{key} must be non-empty text")
    return value


def _sha_text(item: Mapping[str, Any], key: str, context: str) -> str:
    value = _required_text(item, key, context)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReuseProbeError(f"{context}.{key} must be lowercase SHA-256")
    return value


def _load_sources(report_path: Path, dataset: str) -> tuple[str, list[dict[str, str]]]:
    path = _real_file(report_path, "schema preflight")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReuseProbeError(f"cannot parse schema preflight: {error}") from error
    if not isinstance(report, Mapping):
        raise ReuseProbeError("schema preflight root must be an object")
    if report.get("schema") != SCHEMA_PREFLIGHT or report.get("status") != "passed":
        raise ReuseProbeError("schema preflight is not the passed frozen v1 report")
    if dataset != EXPECTED_DATASET:
        raise ReuseProbeError(f"this probe is frozen to {EXPECTED_DATASET!r}")
    datasets = report.get("datasets")
    if not isinstance(datasets, list):
        raise ReuseProbeError("schema preflight datasets must be a list")
    matches = [item for item in datasets if isinstance(item, Mapping) and item.get("dataset_id") == dataset]
    if len(matches) != 1 or matches[0].get("status") != "passed":
        raise ReuseProbeError(f"expected exactly one passed report for {dataset}")
    record = matches[0]
    raw_sources = record.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != EXPECTED_SOURCE_COUNT:
        raise ReuseProbeError(
            f"{dataset} must contain exactly {EXPECTED_SOURCE_COUNT} frozen sources"
        )
    selected: list[dict[str, str]] = []
    excluded = 0
    for ordinal, value in enumerate(raw_sources):
        context = f"datasets[{dataset}].sources[{ordinal}]"
        if not isinstance(value, Mapping):
            raise ReuseProbeError(f"{context} must be an object")
        action = _required_text(value, "source_action", context)
        if action == "exclude_entire_source_file":
            excluded += 1
            continue
        if action != "materialize":
            raise ReuseProbeError(f"{context} has unsupported source_action {action!r}")
        relative = _required_text(value, "source_path", context)
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise ReuseProbeError(f"{context}.source_path is not canonical relative POSIX")
        selected.append(
            {
                "source_path": relative,
                "source_sha256": _sha_text(value, "source_sha256", context),
                "table_name": _required_text(value, "table_name", context),
                "schema_sha256": _sha_text(value, "schema_sha256", context),
            }
        )
    if len(selected) != EXPECTED_MATERIALIZE_COUNT or excluded != 1:
        raise ReuseProbeError(
            "finance1k action closure mismatch: "
            f"materialize={len(selected)}, excluded={excluded}"
        )
    return _sha256(path), selected


def _producer_columns(
    tsfile: Any,
    raw: Path,
    table: str,
    expected_schema_sha: str,
    log: fallback._Log,
    source_ordinal: int,
) -> list[str]:
    reader = None
    try:
        log.emit("source_schema_reader_open_start", source_ordinal=source_ordinal)
        reader = tsfile.TsFileReader(str(raw))
        log.emit("source_schema_reader_open_complete", source_ordinal=source_ordinal)
        table_name, rows, _schema_order = fallback._plan(reader, table, ())
        if table_name != table:
            raise ReuseProbeError(f"table identity drift: {table_name!r} != {table!r}")
        actual_schema_sha = hashlib.sha256(fallback._json(rows).encode("utf-8")).hexdigest()
        if actual_schema_sha != expected_schema_sha:
            raise ReuseProbeError(
                f"source schema SHA mismatch: expected={expected_schema_sha}, "
                f"actual={actual_schema_sha}"
            )
        # Finance plans contain one TAG plus numeric FIELDs.  This matches the
        # producer's tag_names-first query construction, preserving schema
        # order within each category.
        tags = [row["name"] for row in rows if row["category"] == "TAG" and row["name"] != "Time"]
        values = [row["name"] for row in rows if row["category"] != "TAG" and row["name"] != "Time"]
        columns = tags + values
        if not columns or len(columns) != len(set(columns)):
            raise ReuseProbeError("producer query projection is empty or duplicated")
        log.emit(
            "source_schema_verified",
            source_ordinal=source_ordinal,
            schema_sha256=actual_schema_sha,
            schema_columns=rows,
            query_columns=columns,
        )
        return columns
    finally:
        if reader is not None:
            log.emit("source_schema_reader_close_start", source_ordinal=source_ordinal)
            reader.close()
            log.emit("source_schema_reader_close_complete", source_ordinal=source_ordinal)


def run(args: argparse.Namespace, log: fallback._Log) -> None:
    mirror = _real_directory(args.mirror, "raw mirror")
    schema_report_sha, sources = _load_sources(args.schema_report, args.dataset)
    try:
        installed = fallback.package_version("tsfile")
    except fallback.PackageNotFoundError as error:
        raise ReuseProbeError("tsfile package is absent") from error
    if installed != fallback.REQUIRED_TSFILE_VERSION:
        raise ReuseProbeError(
            f"tsfile version drift: expected {fallback.REQUIRED_TSFILE_VERSION}, found {installed}"
        )
    import tsfile  # type: ignore

    dataset_root = _real_directory(mirror / args.dataset, "dataset mirror")
    log.emit(
        "reuse_probe_start",
        dataset=args.dataset,
        mirror=str(mirror),
        schema_report=str(args.schema_report.absolute()),
        schema_report_sha256=schema_report_sha,
        materialize_source_count=len(sources),
        max_rows=args.max_rows,
        tsfile_version=installed,
    )
    for source_ordinal, source in enumerate(sources):
        raw = (dataset_root / source["source_path"]).absolute()
        try:
            raw.relative_to(dataset_root)
        except ValueError as error:
            raise ReuseProbeError(f"source escapes dataset mirror: {raw}") from error
        raw = fallback._source(raw)
        actual_sha = _sha256(raw)
        if actual_sha != source["source_sha256"]:
            raise ReuseProbeError(
                f"source SHA mismatch for {source['source_path']}: "
                f"expected={source['source_sha256']}, actual={actual_sha}"
            )
        log.emit(
            "source_start",
            source_ordinal=source_ordinal,
            source_count=len(sources),
            source_path=source["source_path"],
            source_sha256=source["source_sha256"],
            table_name=source["table_name"],
            expected_schema_sha256=source["schema_sha256"],
        )
        columns = _producer_columns(
            tsfile,
            raw,
            source["table_name"],
            source["schema_sha256"],
            log,
            source_ordinal,
        )
        child_args = SimpleNamespace(
            dataset=args.dataset,
            source=raw,
            expected_sha=source["source_sha256"],
            table=source["table_name"],
            columns=columns,
            cycles=1,
            batch_rows=8192,
            max_rows=args.max_rows,
        )
        fallback.run(child_args, log)
        after_sha = _sha256(raw)
        if after_sha != source["source_sha256"]:
            raise ReuseProbeError(f"source changed after probe: {source['source_path']}")
        log.emit(
            "source_end",
            source_ordinal=source_ordinal,
            source_count=len(sources),
            source_path=source["source_path"],
            source_sha256=after_sha,
            expected_schema_sha256=source["schema_sha256"],
        )
    log.emit("reuse_probe_complete", sources_complete=len(sources))


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--schema-report", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--fault-log", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.max_rows < 0:
        parser.error("--max-rows must be non-negative (zero means full scan)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    events: TextIO | None = None
    fault: TextIO | None = None
    log: fallback._Log | None = None
    try:
        if args.events.absolute() == args.fault_log.absolute():
            raise ReuseProbeError("events and fault log paths must differ")
        events = fallback._new_file(args.events, "events log")
        log = fallback._Log(events)
        fault = fallback._new_file(args.fault_log, "faulthandler log")
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
