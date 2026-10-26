#!/usr/bin/env python3
"""Resume-safe Python staticdata backfill for columns that exceed batch timeouts.

The task TSV must contain:
``dataset,dtype,column_index,column_name,algorithm`` (tab separated).  Each
physical column is benchmarked in row segments.  A completed segment is
committed to SQLite before the next one starts; timed-out segments are split on
the configured patch boundary.  Only after all intervals are complete and
lossless are their raw sizes/times summed into a session-specific report.

This runner deliberately writes only suffixed reports.  It never edits the
project's final ``*_python_compression_result.csv`` files.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BACKEND = (ROOT / "backend").resolve()
REPORT_FIELDS = (
    "column_index",
    "algorithm",
    "originalSize",
    "compressedSize",
    "compressionTime",
    "decompressionTime",
    "isLossy",
    "lossRate",
)
RAW_FIELDS = ("originalSize", "compressedSize", "compressionTime", "decompressionTime")


@dataclass(frozen=True)
class Task:
    dataset: str
    dtype: str
    column_index: int
    column_name: str
    algorithm: str


@dataclass(frozen=True)
class Mapping:
    source_csv: Path
    column_info: Path
    source_index: int
    info_index: int
    source_name: str
    info_row: dict[str, str]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _clean_name(value: object) -> str:
    return str(value or "").strip().removeprefix("\ufeff")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_tasks(path: Path) -> list[Task]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"dataset", "dtype", "column_index", "column_name", "algorithm"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"task TSV missing fields: {', '.join(sorted(missing))}")
        tasks: list[Task] = []
        seen: set[tuple[str, str, int, str, str]] = set()
        for line_no, row in enumerate(reader, start=2):
            dataset = str(row.get("dataset") or "").strip()
            dtype = str(row.get("dtype") or "").strip().lower()
            column_name = _clean_name(row.get("column_name"))
            algorithm = str(row.get("algorithm") or "").strip()
            try:
                column_index = int(str(row.get("column_index") or "").strip())
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid column_index") from exc
            if not dataset or not column_name or not algorithm:
                raise ValueError(f"{path}:{line_no}: empty dataset/column_name/algorithm")
            if dtype not in {"int", "float"}:
                raise ValueError(f"{path}:{line_no}: dtype must be int or float")
            if algorithm.lower() == "reger":
                raise ValueError(f"{path}:{line_no}: reger is multi-column and cannot run here")
            key = (dataset, dtype, column_index, column_name, algorithm)
            if key not in seen:
                seen.add(key)
                tasks.append(Task(*key))
    return tasks


def _resolve_source(dataset_root: Path, dataset: str) -> Path:
    candidates = (dataset_root / f"{dataset}.csv", dataset_root / dataset / f"{dataset}.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    exact_ci = [
        path for path in dataset_root.glob("*.csv") if path.stem.casefold() == dataset.casefold()
    ]
    if len(exact_ci) == 1:
        return exact_ci[0].resolve()
    raise FileNotFoundError(f"cleaned source CSV not found for {dataset!r} under {dataset_root}")


def _resolve_column_info(column_info_root: Path, dataset: str) -> Path:
    candidates = (
        column_info_root / dataset / f"{dataset}_column_info.csv",
        column_info_root / f"{dataset}_column_info.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"column_info not found for {dataset!r} under {column_info_root}")


def _dtype_matches(logical: object, dtype: str) -> bool:
    value = str(logical or "").strip().lower()
    if dtype == "int":
        return value in {"integer", "int", "long", "timestamp"}
    return value in {"float", "double"}


def _map_task(task: Task, dataset_root: Path, column_info_root: Path) -> Mapping:
    source = _resolve_source(dataset_root, task.dataset)
    info = _resolve_column_info(column_info_root, task.dataset)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    with info.open("r", encoding="utf-8-sig", newline="") as handle:
        info_rows = list(csv.DictReader(handle))

    target = _clean_name(task.column_name)
    source_indexes = [idx for idx, name in enumerate(header) if _clean_name(name) == target]
    info_matches: list[tuple[int, dict[str, str]]] = []
    for row in info_rows:
        if _clean_name(row.get("column_name")) != target or not _dtype_matches(row.get("logical_type"), task.dtype):
            continue
        try:
            info_matches.append((int(str(row.get("column_index") or "").strip()), row))
        except ValueError:
            continue
    if not source_indexes:
        raise ValueError(f"{task.dataset}: column {task.column_name!r} absent from cleaned source header")
    if not info_matches:
        raise ValueError(
            f"{task.dataset}: column {task.column_name!r} absent/mismatched in cleaned column_info for {task.dtype}"
        )

    # Match by name first.  Index is only a duplicate-name tie-breaker because
    # null-column removal can shift every subsequent remote index.
    common = [(idx, row) for idx, row in info_matches if idx in source_indexes]
    if len(common) == 1:
        source_index = info_index = common[0][0]
        info_row = common[0][1]
    elif len(source_indexes) == 1 and len(info_matches) == 1:
        source_index = source_indexes[0]
        info_index, info_row = info_matches[0]
    elif task.column_index in source_indexes:
        source_index = task.column_index
        same = [(idx, row) for idx, row in info_matches if idx == source_index]
        if len(same) != 1:
            raise ValueError(f"{task.dataset}: ambiguous duplicate column name {task.column_name!r}")
        info_index, info_row = same[0]
    else:
        raise ValueError(f"{task.dataset}: ambiguous duplicate column name {task.column_name!r}")

    return Mapping(
        source_csv=source,
        column_info=info,
        source_index=int(source_index),
        info_index=int(info_index),
        source_name=_clean_name(header[source_index]),
        info_row={str(k): str(v or "") for k, v in info_row.items()},
    )


def _source_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _task_id(task: Task, mapping: Mapping, suffix: str, patch_rows: int) -> str:
    payload = {
        "task": asdict(task),
        "source": _source_signature(mapping.source_csv),
        "source_index": mapping.source_index,
        "info_index": mapping.info_index,
        "report_suffix": suffix,
        "patch_rows": patch_rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _unmapped_task_id(task: Task, suffix: str) -> str:
    payload = {"task": asdict(task), "report_suffix": suffix, "unmapped": True}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _import_regen():
    scripts_dir = str((ROOT / "scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    previous = Path.cwd()
    try:
        import regenerate_staticdata_python as regen
    finally:
        os.chdir(previous)
    return regen


def _materialize_many(
    source_csv: Path,
    dataset: str,
    dtype: str,
    columns: list[tuple[int, str]],
    cache_dir: Path,
) -> tuple[dict[int, Path], int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["COMPRESSION_BENCHMARK_COLUMN_CSV_CACHE_DIR"] = str(cache_dir.resolve())
    regen = _import_regen()
    paths, row_count, cleanup = regen._materialize_column_csvs(
        source_csv,
        dataset,
        dtype,
        columns,
    )
    if cleanup:
        raise RuntimeError("durable column cache was not used")
    return {idx: path.resolve() for idx, path in paths.items()}, int(row_count)


def _count_valid_numeric(path: Path, dtype: str) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            text = str(row[0] if row else "").strip()
            if not text:
                continue
            try:
                if dtype == "int":
                    int(text)
                else:
                    value = float(text)
                    if not math.isfinite(value):
                        continue
            except ValueError:
                continue
            count += 1
    return count


def _mapped_info_path(state_dir: Path, task_id: str, mapping: Mapping) -> Path:
    path = state_dir / "mapped_column_info" / f"{task_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(mapping.info_row)
    row["column_index"] = str(mapping.source_index)
    row["column_name"] = mapping.source_name
    fieldnames = list(row)
    for field in ("column_index", "column_name", "logical_type", "max_point_number"):
        if field not in fieldnames:
            fieldnames.append(field)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(tmp, path)
    return path.resolve()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks (
          task_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, dtype TEXT NOT NULL,
          output_index INTEGER NOT NULL, column_name TEXT NOT NULL,
          algorithm TEXT NOT NULL, source_csv TEXT NOT NULL,
          source_index INTEGER NOT NULL, total_rows INTEGER NOT NULL,
          report_path TEXT NOT NULL, status TEXT NOT NULL,
          error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS segments (
          task_id TEXT NOT NULL, start_row INTEGER NOT NULL, end_row INTEGER NOT NULL,
          status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          original_size INTEGER, compressed_size INTEGER,
          compression_time INTEGER, decompression_time INTEGER,
          is_lossy INTEGER, loss_rate REAL, error TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL,
          PRIMARY KEY (task_id, start_row, end_row)
        );
        """
    )
    db.commit()
    return db


def _normalise_segment_rows(initial: int, patch_rows: int) -> int:
    if patch_rows <= 0:
        raise ValueError("patch_rows must be positive")
    if initial < patch_rows:
        return patch_rows
    return (initial // patch_rows) * patch_rows


def _ensure_task(
    db: sqlite3.Connection,
    task_id: str,
    task: Task,
    mapping: Mapping,
    total_rows: int,
    report_path: Path,
    initial_segment_rows: int,
) -> None:
    now = _now()
    db.execute(
        """INSERT OR IGNORE INTO tasks
        (task_id,dataset,dtype,output_index,column_name,algorithm,source_csv,
         source_index,total_rows,report_path,status,error,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            task_id, task.dataset, task.dtype, task.column_index, task.column_name,
            task.algorithm, str(mapping.source_csv), mapping.source_index, total_rows,
            str(report_path), "pending", "", now, now,
        ),
    )
    stored = db.execute("SELECT total_rows, source_csv FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if stored is None or int(stored["total_rows"]) != total_rows or stored["source_csv"] != str(mapping.source_csv):
        raise RuntimeError(f"state mismatch for task {task_id}")
    count = int(db.execute("SELECT count(*) FROM segments WHERE task_id=?", (task_id,)).fetchone()[0])
    if count == 0:
        if total_rows == 0:
            raise ValueError(f"{task.dataset}/{task.column_name}: cleaned column has no rows")
        for start in range(0, total_rows, initial_segment_rows):
            end = min(total_rows, start + initial_segment_rows)
            db.execute(
                "INSERT INTO segments(task_id,start_row,end_row,status,updated_at) VALUES (?,?,?,?,?)",
                (task_id, start, end, "pending", now),
            )
    db.execute(
        "UPDATE segments SET status='pending', error='interrupted before completion', updated_at=? "
        "WHERE task_id=? AND status='running'",
        (now, task_id),
    )
    db.execute(
        "UPDATE segments SET status='pending', error='retrying previous failure', updated_at=? "
        "WHERE task_id=? AND status='failed'",
        (now, task_id),
    )
    db.commit()


def _record_skipped(
    db: sqlite3.Connection,
    *,
    task_id: str,
    task: Task,
    mapping: Mapping | None,
    report_path: Path,
    reason: str,
) -> None:
    now = _now()
    db.execute(
        """INSERT INTO tasks
        (task_id,dataset,dtype,output_index,column_name,algorithm,source_csv,
         source_index,total_rows,report_path,status,error,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(task_id) DO UPDATE SET
          status=excluded.status,error=excluded.error,updated_at=excluded.updated_at""",
        (
            task_id,
            task.dataset,
            task.dtype,
            task.column_index,
            task.column_name,
            task.algorithm,
            str(mapping.source_csv) if mapping else "",
            mapping.source_index if mapping else -1,
            0,
            str(report_path),
            "skipped_unmapped_or_empty",
            reason,
            now,
            now,
        ),
    )
    db.commit()


def _worker_request(
    *, task: Task, task_id: str, column_csv: Path, mapped_info: Path,
    source_index: int, start: int, end: int, patch_rows: int,
) -> dict[str, object]:
    return {
        "task": asdict(task), "task_id": task_id, "column_csv": str(column_csv),
        "mapped_info": str(mapped_info), "source_index": source_index,
        "start": start, "end": end, "patch_rows": patch_rows,
    }


def _write_segment_csv(
    source: Path, target: Path, start: int, end: int, dtype: str
) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    written = 0
    with source.open("r", encoding="utf-8", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        header = next(reader, [])
        if not header:
            raise ValueError(f"empty materialized column CSV: {source}")
        writer.writerow([header[0]])
        for row in itertools.islice(reader, start, end):
            cell = row[0] if row else ""
            writer.writerow([cell])
            written += 1
            text = str(cell).strip()
            if text:
                try:
                    if dtype == "int":
                        int(text)
                    else:
                        number = float(text)
                        if not math.isfinite(number):
                            continue
                    valid += 1
                except ValueError:
                    pass
    return written, valid


def _select_worker_row(report: Path, algorithm: str) -> dict[str, str] | None:
    if not report.is_file():
        return None
    with report.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    exact = [row for row in rows if str(row.get("algorithm") or "").strip() == algorithm]
    if len(exact) == 1:
        return exact[0]
    from myapp.lzbench_codec_catalog import canonical_report_algorithm_key

    wanted = canonical_report_algorithm_key(algorithm) or algorithm
    canonical = [
        row
        for row in rows
        if (canonical_report_algorithm_key(str(row.get("algorithm") or "").strip()) or "")
        == wanted
    ]
    if len(canonical) == 1:
        return canonical[0]
    return None


def _worker_main(request_path: Path, result_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    task = Task(**request["task"])
    start, end = int(request["start"]), int(request["end"])
    patch_rows = int(request["patch_rows"])
    source_index = int(request["source_index"])
    column_csv = Path(request["column_csv"]).expanduser().resolve()
    mapped_info = Path(request["mapped_info"]).expanduser().resolve()
    work_dir = result_path.parent / "worker_tmp"
    segment_csv = work_dir / f"{request['task_id']}_{start}_{end}_{os.getpid()}.csv"
    segment_report = segment_csv.with_suffix(".report.csv")
    try:
        written, valid = _write_segment_csv(column_csv, segment_csv, start, end, task.dtype)
        if written != end - start:
            raise ValueError(f"segment short read: expected={end-start}, got={written}")
        if valid == 0:
            row = {field: 0 for field in RAW_FIELDS}
            row.update({"isLossy": 0, "lossRate": 0.0})
        else:
            os.environ["COMPRESSION_BENCHMARK_CSV_PATCH_ROWS"] = str(patch_rows)
            regen = _import_regen()
            from staticdata_regen_common import configure_allowlist

            configure_allowlist([task.algorithm])
            regen.process_csv_single_column_patch(
                str(segment_csv), task.dtype, str(segment_report), str(mapped_info), source_index
            )
            selected = _select_worker_row(segment_report, task.algorithm)
            if selected is None:
                raise RuntimeError(f"algorithm {task.algorithm!r} produced no unique report row")
            row = {
                field: int(float(str(selected.get(field) or "0"))) for field in RAW_FIELDS
            }
            row["isLossy"] = int(float(str(selected.get("isLossy") or "0")))
            row["lossRate"] = float(str(selected.get("lossRate") or "0"))
        payload = {
            "task_id": request["task_id"], "start": start, "end": end,
            "algorithm": task.algorithm, "valid_values": valid, **row,
        }
        _atomic_json(result_path, payload)
        return 0
    finally:
        segment_csv.unlink(missing_ok=True)
        segment_report.unlink(missing_ok=True)


def _terminate_group(process: subprocess.Popen[bytes], grace_sec: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_segment(
    db: sqlite3.Connection, *, task: Task, task_id: str, column_csv: Path,
    mapped_info: Path, source_index: int, segment: sqlite3.Row, patch_rows: int,
    timeout_sec: int, state_dir: Path, log,
) -> str:
    start, end = int(segment["start_row"]), int(segment["end_row"])
    now = _now()
    db.execute(
        "UPDATE segments SET status='running', attempts=attempts+1, error='', updated_at=? "
        "WHERE task_id=? AND start_row=? AND end_row=?",
        (now, task_id, start, end),
    )
    db.commit()
    req_dir = state_dir / "requests"
    result_dir = state_dir / "results"
    req_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    request_path = req_dir / f"{task_id}_{start}_{end}.json"
    result_path = result_dir / f"{task_id}_{start}_{end}.json"
    result_path.unlink(missing_ok=True)
    _atomic_json(
        request_path,
        _worker_request(
            task=task, task_id=task_id, column_csv=column_csv, mapped_info=mapped_info,
            source_index=source_index, start=start, end=end, patch_rows=patch_rows,
        ),
    )
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request_path),
        "--worker-result", str(result_path),
    ]
    print(f"SEGMENT start={start} end={end} rows={end-start} algorithm={task.algorithm}", file=log, flush=True)
    process = subprocess.Popen(
        command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        try:
            returncode = process.wait(timeout=timeout_sec if timeout_sec > 0 else None)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            return "timeout"
        except BaseException:
            _terminate_group(process)
            raise
    finally:
        for stale in (result_dir / "worker_tmp").glob(f"{task_id}_{start}_{end}_*.csv*"):
            stale.unlink(missing_ok=True)
    if returncode != 0 or not result_path.is_file():
        return f"worker exit {returncode}"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("task_id") != task_id or int(result.get("start", -1)) != start or int(result.get("end", -1)) != end:
        return "worker result identity mismatch"
    if int(result.get("isLossy", 0)) != 0 or float(result.get("lossRate", 0)) != 0:
        return f"lossy segment isLossy={result.get('isLossy')} lossRate={result.get('lossRate')}"
    db.execute(
        """UPDATE segments SET status='complete', original_size=?, compressed_size=?,
        compression_time=?, decompression_time=?, is_lossy=?, loss_rate=?, error='', updated_at=?
        WHERE task_id=? AND start_row=? AND end_row=?""",
        (
            int(result["originalSize"]), int(result["compressedSize"]),
            int(result["compressionTime"]), int(result["decompressionTime"]),
            int(result["isLossy"]), float(result["lossRate"]), _now(), task_id, start, end,
        ),
    )
    db.commit()
    print(
        f"SEGMENT complete start={start} end={end} original={result['originalSize']} "
        f"compressed={result['compressedSize']}",
        file=log, flush=True,
    )
    return "complete"


def _split_or_fail(
    db: sqlite3.Connection, task_id: str, segment: sqlite3.Row, patch_rows: int, reason: str
) -> None:
    start, end = int(segment["start_row"]), int(segment["end_row"])
    span = end - start
    now = _now()
    units = math.ceil(span / patch_rows)
    if reason == "timeout" and units > 1:
        split = start + (units // 2) * patch_rows
        if start < split < end:
            db.execute(
                "UPDATE segments SET status='split', error=?, updated_at=? "
                "WHERE task_id=? AND start_row=? AND end_row=?",
                (reason, now, task_id, start, end),
            )
            for left, right in ((start, split), (split, end)):
                db.execute(
                    "INSERT OR IGNORE INTO segments(task_id,start_row,end_row,status,updated_at) VALUES (?,?,?,?,?)",
                    (task_id, left, right, "pending", now),
                )
            db.commit()
            return
    db.execute(
        "UPDATE segments SET status='failed', error=?, updated_at=? "
        "WHERE task_id=? AND start_row=? AND end_row=?",
        (reason, now, task_id, start, end),
    )
    db.commit()


def _complete_leaf_segments(db: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    return list(
        db.execute(
            "SELECT * FROM segments WHERE task_id=? AND status='complete' ORDER BY start_row,end_row",
            (task_id,),
        )
    )


def _aggregate_if_complete(
    db: sqlite3.Connection, task_id: str, task: Task, total_rows: int
) -> dict[str, object] | None:
    leaves = _complete_leaf_segments(db, task_id)
    cursor = 0
    for row in leaves:
        if int(row["start_row"]) != cursor or int(row["end_row"]) <= cursor:
            return None
        cursor = int(row["end_row"])
    if cursor != total_rows or not leaves:
        return None
    if any(int(row["is_lossy"] or 0) != 0 or float(row["loss_rate"] or 0) != 0 for row in leaves):
        raise RuntimeError(f"{task.dataset}/{task.column_name}/{task.algorithm}: lossy segment detected")
    out: dict[str, object] = {"column_index": task.column_index, "algorithm": task.algorithm}
    db_names = {
        "originalSize": "original_size", "compressedSize": "compressed_size",
        "compressionTime": "compression_time", "decompressionTime": "decompression_time",
    }
    for field, db_name in db_names.items():
        out[field] = sum(int(row[db_name] or 0) for row in leaves)
    out.update({"isLossy": 0, "lossRate": 0})
    return out


def _read_report(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{field: str(row.get(field, "") or "") for field in REPORT_FIELDS} for row in csv.DictReader(handle)]


def _row_key(row: dict[str, object]) -> tuple[int, str] | None:
    try:
        return int(float(str(row.get("column_index") or ""))), str(row.get("algorithm") or "").strip()
    except ValueError:
        return None


def _upsert_report(path: Path, row: dict[str, object]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows: dict[tuple[int, str], dict[str, object]] = {}
        for existing in _read_report(path):
            key = _row_key(existing)
            if key is not None:
                rows[key] = existing
        rows[(int(row["column_index"]), str(row["algorithm"]))] = row
        ordered = [rows[key] for key in sorted(rows)]
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: item.get(field, "") for field in REPORT_FIELDS} for item in ordered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def _write_status(db: sqlite3.Connection, path: Path, current: str | None = None) -> None:
    task_rows = list(db.execute("SELECT * FROM tasks ORDER BY dataset,dtype,output_index,algorithm"))
    tasks: list[dict[str, object]] = []
    summary: dict[str, int] = {}
    for task in task_rows:
        counts = {
            str(row["status"]): int(row["n"])
            for row in db.execute(
                "SELECT status,count(*) n FROM segments WHERE task_id=? GROUP BY status", (task["task_id"],)
            )
        }
        status = str(task["status"])
        summary[status] = summary.get(status, 0) + 1
        tasks.append(
            {
                "task_id": task["task_id"], "dataset": task["dataset"], "dtype": task["dtype"],
                "column_index": task["output_index"], "column_name": task["column_name"],
                "algorithm": task["algorithm"], "status": status, "segments": counts,
                "total_rows": task["total_rows"], "error": task["error"],
            }
        )
    _atomic_json(path, {"updated_at": _now(), "current_task_id": current, "summary": summary, "tasks": tasks})


def _report_path(report_root: Path, task: Task, suffix: str) -> Path:
    return (
        report_root / task.dataset / f"{task.dataset}_{task.dtype}_python_compression_result.csv{suffix}"
    ).resolve()


def _run_task(
    db: sqlite3.Connection, *, task: Task, mapping: Mapping, task_id: str,
    report_path: Path, state_dir: Path, column_csv: Path, total_rows: int,
    patch_rows: int, initial_segment_rows: int,
    timeout_sec: int, status_path: Path,
) -> bool:
    mapped_info = _mapped_info_path(state_dir, task_id, mapping)
    _ensure_task(db, task_id, task, mapping, total_rows, report_path, initial_segment_rows)
    db.execute("UPDATE tasks SET status='running',error='',updated_at=? WHERE task_id=?", (_now(), task_id))
    db.commit()
    log_path = state_dir / "logs" / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        print(
            f"TASK {_now()} dataset={task.dataset} dtype={task.dtype} output_index={task.column_index} "
            f"column={task.column_name!r} source_index={mapping.source_index} algorithm={task.algorithm} rows={total_rows}",
            file=log, flush=True,
        )
        while True:
            segment = db.execute(
                "SELECT * FROM segments WHERE task_id=? AND status='pending' ORDER BY start_row,end_row LIMIT 1",
                (task_id,),
            ).fetchone()
            if segment is None:
                break
            _write_status(db, status_path, task_id)
            result = _run_segment(
                db, task=task, task_id=task_id, column_csv=column_csv, mapped_info=mapped_info,
                source_index=mapping.source_index, segment=segment, patch_rows=patch_rows,
                timeout_sec=timeout_sec, state_dir=state_dir, log=log,
            )
            if result != "complete":
                _split_or_fail(db, task_id, segment, patch_rows, result)
                print(
                    f"SEGMENT {result} start={segment['start_row']} end={segment['end_row']}",
                    file=log, flush=True,
                )
            _write_status(db, status_path, task_id)

    try:
        row = _aggregate_if_complete(db, task_id, task, total_rows)
    except RuntimeError as exc:
        db.execute(
            "UPDATE tasks SET status='failed',error=?,updated_at=? WHERE task_id=?",
            (str(exc), _now(), task_id),
        )
        db.commit()
        _write_status(db, status_path)
        return False
    if row is None:
        counts = {
            str(item["status"]): int(item["n"])
            for item in db.execute(
                "SELECT status,count(*) n FROM segments WHERE task_id=? GROUP BY status", (task_id,)
            )
        }
        error = "incomplete segment coverage: " + json.dumps(counts, sort_keys=True)
        db.execute(
            "UPDATE tasks SET status='failed',error=?,updated_at=? WHERE task_id=?",
            (error, _now(), task_id),
        )
        db.commit()
        _write_status(db, status_path)
        return False
    _upsert_report(report_path, row)
    db.execute("UPDATE tasks SET status='complete',error='',updated_at=? WHERE task_id=?", (_now(), task_id))
    db.commit()
    _write_status(db, status_path)
    return True


def _suffix(raw: str) -> str:
    value = str(raw or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("--report-suffix must be a non-empty filename suffix without path separators")
    return value if value.startswith(".") else "." + value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, help="Tab-separated task file")
    parser.add_argument("--dataset-root", type=Path, default=BACKEND / "dataset_csv")
    parser.add_argument("--column-info-root", type=Path, default=BACKEND / "dataset")
    parser.add_argument("--report-root", type=Path, default=BACKEND / "dataset")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "analysis_outputs" / "backfill_python_chunked")
    parser.add_argument("--report-suffix", default="")
    parser.add_argument("--patch-rows", type=int, default=10_000)
    parser.add_argument("--initial-segment-rows", type=int, default=500_000)
    parser.add_argument("--timeout-sec", type=int, default=7_200)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker_request:
        if not args.worker_result:
            raise SystemExit("--worker-result is required with --worker-request")
        return _worker_main(args.worker_request.expanduser().resolve(), args.worker_result.expanduser().resolve())
    if not args.tasks:
        raise SystemExit("--tasks is required")
    suffix = _suffix(args.report_suffix)
    tasks_path = args.tasks.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    info_root = args.column_info_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()
    state_dir = args.state_dir.expanduser().resolve()
    if not tasks_path.is_file():
        raise SystemExit(f"task TSV not found: {tasks_path}")
    if not dataset_root.is_dir() or not info_root.is_dir():
        raise SystemExit(f"dataset/column-info root missing: {dataset_root} / {info_root}")
    try:
        initial_rows = _normalise_segment_rows(args.initial_segment_rows, args.patch_rows)
        tasks = _load_tasks(tasks_path)
        if args.max_tasks > 0:
            tasks = tasks[: args.max_tasks]
        mapped: list[tuple[Task, Mapping]] = []
        unmapped: list[tuple[Task, str]] = []
        for task in tasks:
            try:
                mapped.append((task, _map_task(task, dataset_root, info_root)))
            except (OSError, ValueError) as exc:
                unmapped.append((task, str(exc)))
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"tasks={len(mapped)} patch_rows={args.patch_rows} initial_segment_rows={initial_rows} "
        f"timeout_sec={args.timeout_sec} suffix={suffix}",
        flush=True,
    )
    for task, mapping in mapped:
        print(
            f"MAP {task.dataset} {task.dtype} local_index={task.column_index} "
            f"column={task.column_name!r} -> cleaned_source_index={mapping.source_index} "
            f"cleaned_info_index={mapping.info_index} algorithm={task.algorithm}",
            flush=True,
        )
    for task, reason in unmapped:
        print(
            f"SKIP {task.dataset} {task.dtype} local_index={task.column_index} "
            f"column={task.column_name!r} algorithm={task.algorithm}: {reason}",
            flush=True,
        )
    if args.dry_run:
        return 0

    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "status.json"
    db = _connect(state_dir / "state.sqlite3")
    completed = failed = 0
    try:
        for task, reason in unmapped:
            report_path = _report_path(report_root, task, suffix)
            _record_skipped(
                db,
                task_id=_unmapped_task_id(task, suffix),
                task=task,
                mapping=None,
                report_path=report_path,
                reason=reason,
            )

        # Materialize and scan once for all algorithms targeting the same
        # canonical cleaned column.  All-null columns are skipped as a group.
        prepared: dict[tuple[str, str, int, str], tuple[Path, int, int]] = {}
        groups: dict[tuple[Path, str, str], dict[int, str]] = {}
        for task, mapping in mapped:
            groups.setdefault((mapping.source_csv, task.dataset, task.dtype), {})[
                mapping.source_index
            ] = mapping.source_name
        for (source_csv, dataset, dtype), columns_by_index in groups.items():
            paths, total_rows = _materialize_many(
                source_csv,
                dataset,
                dtype,
                sorted(columns_by_index.items()),
                state_dir / "column_cache",
            )
            for source_index, column_name in columns_by_index.items():
                key = (str(source_csv), dtype, source_index, column_name)
                column_csv = paths[source_index]
                prepared[key] = (
                    column_csv,
                    total_rows,
                    _count_valid_numeric(column_csv, dtype),
                )

        for task, mapping in mapped:
            report_path = _report_path(report_root, task, suffix)
            task_id = _task_id(task, mapping, suffix, args.patch_rows)
            key = (str(mapping.source_csv), task.dtype, mapping.source_index, task.column_name)
            column_csv, total_rows, valid_count = prepared[key]
            if valid_count == 0:
                reason = (
                    f"cleaned column {task.column_name!r} has no valid {task.dtype} values "
                    f"across {total_rows} rows"
                )
                _record_skipped(
                    db,
                    task_id=task_id,
                    task=task,
                    mapping=mapping,
                    report_path=report_path,
                    reason=reason,
                )
                print(f"SKIP task_id={task_id}: {reason}", flush=True)
                continue
            print(f"RUN task_id={task_id} {task.dataset}/{task.dtype}/{task.column_name}/{task.algorithm}", flush=True)
            try:
                ok = _run_task(
                    db, task=task, mapping=mapping, task_id=task_id, report_path=report_path,
                    state_dir=state_dir, column_csv=column_csv, total_rows=total_rows,
                    patch_rows=args.patch_rows,
                    initial_segment_rows=initial_rows, timeout_sec=args.timeout_sec,
                    status_path=status_path,
                )
            except BaseException as exc:
                db.execute(
                    "UPDATE tasks SET status='failed',error=?,updated_at=? WHERE task_id=?",
                    (f"{type(exc).__name__}: {exc}", _now(), task_id),
                )
                db.commit()
                _write_status(db, status_path)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"FAILED task_id={task_id}: {type(exc).__name__}: {exc}", flush=True)
                ok = False
            completed += int(ok)
            failed += int(not ok)
    finally:
        _write_status(db, status_path)
        db.close()
    print(f"DONE completed={completed} failed={failed} status={status_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
