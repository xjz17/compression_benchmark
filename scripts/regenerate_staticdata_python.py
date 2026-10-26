#!/usr/bin/env python3
"""Rebuild *_python_compression_result.csv under backend/dataset/<slug>/ from master CSVs.

By default **appends** missing ``algorithm`` rows. Set ``WEB_COMPRESSION_STATICDATA_REGEN=1`` to
delete existing reports and rewrite them (same as regenerate_staticdata_java.py / C++).

Source: ``backend/dataset_csv/<slug>.csv`` (discovered automatically).

Reads master CSV, slices one column at a time into temporary CSVs, then runs
``process_csv_single_column_patch`` (still patch-based by
``WEB_COMPRESSION_CSV_PATCH_ROWS``, default 10000). ``{slug}_column_info.csv`` drives
column selection and per-column ``max_point_number``.

Requires: backend venv with ``pip install -r requirements.txt`` (includes ``zfpy`` for ZFP rows).
FPCompress rows (``fpcompress_*``) are emitted by the shared tri-language
``native_lz_family`` implementation; the old external binary path is not used.
For MSDZip / MSDLC rows: ``pip install torch``, submodules under ``external/compression/``, then
``export WEB_COMPRESSION_MSDZIP_BENCH=1 WEB_COMPRESSION_MSDLC_BENCH=1`` (optional
``WEB_COMPRESSION_MSD_MAX_COLUMNS`` for shorter runs).

**SPDP** (``spdp_level_0`` … ``spdp_level_9``): in-process Python in
``backend/myapp/research_extended_codecs.py`` (reference
``benchmark/reference/upstream/spdp/standalone/SPDP_11.c``). All ten levels are always benchmarked
(like ``lzma_level_*``); use ``--algorithms spdp_level_N`` for partial refresh.

**FPC** (``fpc``): in-process Python in ``backend/myapp/research_extended_codecs.py`` (reference
``benchmark/reference/upstream/fpc/fpc.c``). Predictor table size lg2 defaults to 16;
override with ``WEB_COMPRESSION_FPC_PRED_SIZE_LG2``.

**pFPC** (``pfpc``): in-process Python in ``research_extended_codecs.py`` (reference
``benchmark/reference/upstream/pfpc/pFPC.c``). Defaults: pred lg2=20, threads=8,
chunksize=2048; override with ``WEB_COMPRESSION_PFPC_PRED_SIZE_LG2``,
``WEB_COMPRESSION_PFPC_THREADS``, ``WEB_COMPRESSION_PFPC_CHUNKSIZE``.

**ZFP** (``ext_zfp_zfpy``, ``ext_zfp_tol_1e-*``): ``external_compression_python.py`` via pip
``zfpy`` (listed in ``requirements.txt``). Skip with ``WEB_COMPRESSION_ZFP_BENCH=0``.
**Blosc / Blosc2** (``blosc_blosclz``, ``blosc2_blosclz``): emitted by
``native_lz_family``; old package/CLI replacement paths are not used for staticdata rows.

**Partial algorithm refresh:** ``--algorithms TS_2DIFF,ts_2diff_bos,brieflz_1`` runs only the
listed algorithms and merges them into the existing report (other rows are kept). Use with
``--dataset`` / ``--dtype`` to limit scope.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_csv_source import describe_dataset_csv_lookup, iter_catalog_benchmark_datasets
from staticdata_regen_common import (
    configure_allowlist,
    load_report_rows,
    merge_partial_algorithm_rows,
    parse_algorithms,
    write_report_rows,
)
import pandas as pd

sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

_csv_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_field_limit)
        break
    except OverflowError:
        _csv_field_limit //= 10

# In-process BUFF kernel row in float/int CSVs (no comp_profiler).
os.environ.setdefault("WEB_COMPRESSION_BUFF_KERNEL_NATIVE", "1")
os.environ.setdefault("WEB_COMPRESSION_CMIX_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_CMIX_NO_PREPROCESS", "1")
os.environ.setdefault("WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES", str(64 * 1024))
os.environ.setdefault("WEB_COMPRESSION_CMIX_TIMEOUT_SEC", "180")
os.environ.setdefault("WEB_COMPRESSION_MSDZIP_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_MSDLC_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_SZ3_BENCH", "1")
os.environ.setdefault("WEB_COMPRESSION_FPZIP_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_FPCOMPRESS_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_BLOSC_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_BLOSC2_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_LZBENCH_BENCH", "0")
os.environ.setdefault(
    "WEB_COMPRESSION_FPCOMPRESS_BIN_ROOT",
    str((ROOT / "external" / "runtime" / "fpcompress").resolve()),
)
os.environ.setdefault("WEB_COMPRESSION_SERF_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_SZ_BENCH", "1")
os.environ.setdefault("WEB_COMPRESSION_BUFF_BENCH", "0")
os.environ.setdefault("WEB_COMPRESSION_CSV_PATCH_ROWS", "10000")
os.environ.setdefault("WEB_COMPRESSION_REPO_ROOT", str(ROOT.resolve()))

from myapp.benchmark_loss import format_loss_rate_csv  # noqa: E402
from myapp.processor_patch_bench import process_csv_simple_patch, process_csv_single_column_patch  # noqa: E402

REGER_TIME_COLUMN = "__reger_time__"


def _normalize_logical_type(raw: object) -> str:
    return str(raw or "").strip().lower()


def _iter_column_info_rows(column_info_path: Path) -> list[dict]:
    with column_info_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _columns_for_dtype(column_info_rows: list[dict], data_type: str) -> list[tuple[int, str]]:
    if data_type == "int":
        wanted = {"integer", "int", "long", "timestamp"}
    elif data_type == "float":
        wanted = {"float", "double"}
    elif data_type == "string":
        wanted = {"string"}
    else:
        raise ValueError(data_type)
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for row in column_info_rows:
        logical = _normalize_logical_type(row.get("logical_type"))
        if logical not in wanted:
            continue
        try:
            idx = int(row.get("column_index"))
        except (TypeError, ValueError):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        out.append((idx, str(row.get("column_name", "")).strip()))
    return out


def _read_source_header(src: Path) -> list[str]:
    with src.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _time_header_score(name: str) -> int | None:
    low = str(name or "").strip().removeprefix("\ufeff").lower()
    compact = "".join(ch for ch in low if ch.isalnum())
    if compact in {
        "rowtime",
        "time",
        "timestamp",
        "datetime",
        "date",
        "ts",
        "published",
    }:
        return 0
    if any(token in compact for token in ("timestamp", "datetime", "rowtime")):
        return 1
    if "time" in compact or "date" in compact:
        return 2
    return None


def _detect_time_column_indexes(source_header: list[str]) -> tuple[int, ...]:
    compact_by_idx = [
        "".join(ch for ch in str(name or "").strip().removeprefix("\ufeff").lower() if ch.isalnum())
        for name in source_header
    ]
    date_idx = next((idx for idx, name in enumerate(compact_by_idx) if name == "date"), None)
    time_idx = next((idx for idx, name in enumerate(compact_by_idx) if name == "time"), None)
    if date_idx is not None and time_idx is not None:
        return (date_idx, time_idx)
    ranked: list[tuple[int, int]] = []
    for idx, name in enumerate(source_header):
        score = _time_header_score(name)
        if score is not None:
            ranked.append((score, idx))
    if not ranked:
        return ()
    return (min(ranked)[1],)


def _parse_time_cell(cell: object, row_idx: int) -> int:
    text = str(cell or "").strip().strip('"')
    if not text:
        return row_idx
    try:
        numeric = float(text)
        if numeric == numeric and numeric not in (float("inf"), float("-inf")):
            if abs(numeric) < 100_000_000_000:
                return int(round(numeric * 1_000_000))
            return int(round(numeric))
    except ValueError:
        pass
    iso = text.replace("Z", "+00:00")
    try:
        dt = _dt.datetime.fromisoformat(iso)
    except ValueError:
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ):
            try:
                dt = _dt.datetime.strptime(text, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            return row_idx
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1_000_000)


@contextmanager
def _temporary_algorithm_allowlist(algorithm: str):
    keys = (
        "WEB_COMPRESSION_RUN_ALGORITHMS",
        "WEB_COMPRESSION_LZBENCH_BENCH",
        "WEB_COMPRESSION_SZ_BENCH",
        "WEB_COMPRESSION_SZ3_BENCH",
        "WEB_COMPRESSION_FPZIP_BENCH",
        "WEB_COMPRESSION_FPCOMPRESS_BENCH",
        "WEB_COMPRESSION_BLOSC_BENCH",
        "WEB_COMPRESSION_BLOSC2_BENCH",
        "WEB_COMPRESSION_ZFP_BENCH",
    )
    old = {key: os.environ.get(key) for key in keys}
    os.environ["WEB_COMPRESSION_RUN_ALGORITHMS"] = algorithm
    for key in keys[1:]:
        os.environ[key] = "0"
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _temporary_reger_allowlist():
    return _temporary_algorithm_allowlist("reger")


def _materialize_column_csvs(
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str]],
) -> tuple[dict[int, Path], int, set[Path]]:
    cache_root_raw = os.environ.get("WEB_COMPRESSION_COLUMN_CSV_CACHE_DIR", "").strip()
    cache_root = Path(cache_root_raw).expanduser() if cache_root_raw else None
    source_stat = src.stat()
    source_sig = {
        "source": str(src.resolve()),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "slug": slug,
        "data_type": data_type,
    }

    def cache_paths(col_idx: int) -> tuple[Path, Path] | None:
        if cache_root is None:
            return None
        payload = json.dumps({**source_sig, "column_index": int(col_idx)}, sort_keys=True)
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        directory = cache_root / slug / data_type
        return directory / f"col{int(col_idx)}_{key}.csv", directory / f"col{int(col_idx)}_{key}.json"

    def read_cached(col_idx: int) -> tuple[Path, int] | None:
        paths_pair = cache_paths(col_idx)
        if paths_pair is None:
            return None
        csv_path, meta_path = paths_pair
        if not csv_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_sig") != source_sig or int(meta.get("column_index")) != int(col_idx):
                return None
            return csv_path, int(meta.get("row_count", 0))
        except Exception:
            return None

    paths: dict[int, Path] = {}
    cleanup_paths: set[Path] = set()
    cached_row_count: int | None = None
    missing_columns: list[tuple[int, str]] = []
    for col_idx, col_name in columns:
        cached = read_cached(int(col_idx))
        if cached is None:
            missing_columns.append((col_idx, col_name))
            continue
        paths[int(col_idx)] = cached[0]
        cached_row_count = cached[1] if cached_row_count is None else cached_row_count

    if not missing_columns:
        return paths, int(cached_row_count or 0), cleanup_paths

    handles = []
    writers = []
    tmp_to_final: dict[int, tuple[Path, Path, Path]] = {}
    try:
        with src.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.reader(input_handle)
            source_header = next(reader, [])
            for col_idx, _col_name in missing_columns:
                paths_pair = cache_paths(int(col_idx))
                if paths_pair is None:
                    tmp_csv = tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=f"_{slug}_{data_type}_col{col_idx}.csv",
                        delete=False,
                        encoding="utf-8",
                        newline="",
                    )
                    final_path = Path(tmp_csv.name)
                    cleanup_paths.add(final_path)
                else:
                    final_path, meta_path = paths_pair
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_csv = tempfile.NamedTemporaryFile(
                        mode="w",
                        prefix=f".{final_path.stem}.",
                        suffix=".tmp",
                        dir=final_path.parent,
                        delete=False,
                        encoding="utf-8",
                        newline="",
                    )
                    tmp_to_final[int(col_idx)] = (Path(tmp_csv.name), final_path, meta_path)
                writer = csv.writer(tmp_csv, lineterminator="\n")
                header = source_header[col_idx] if 0 <= col_idx < len(source_header) else f"column_{col_idx}"
                writer.writerow([header])
                handles.append(tmp_csv)
                paths[int(col_idx)] = final_path
                writers.append((int(col_idx), writer))

            row_count = 0
            for row in reader:
                row_count += 1
                for col_idx, writer in writers:
                    writer.writerow([row[col_idx] if 0 <= col_idx < len(row) else ""])
    except Exception:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        for path in cleanup_paths:
            path.unlink(missing_ok=True)
        for tmp_path, _final_path, _meta_path in tmp_to_final.values():
            tmp_path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
    for col_idx, (tmp_path, final_path, meta_path) in tmp_to_final.items():
        os.replace(tmp_path, final_path)
        meta_tmp = meta_path.with_suffix(meta_path.suffix + f".{os.getpid()}.tmp")
        meta_tmp.write_text(
            json.dumps(
                {
                    "source_sig": source_sig,
                    "column_index": int(col_idx),
                    "row_count": int(row_count),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(meta_tmp, meta_path)
    return paths, row_count, cleanup_paths


def _materialize_multi_column_csv(
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str]],
) -> tuple[Path, int, list[str], list[tuple[int, str]], int]:
    tmp_csv = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{slug}_{data_type}_reger_multi.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )
    path = Path(tmp_csv.name)
    try:
        with src.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.reader(input_handle)
            source_header = next(reader, [])
            time_indexes = _detect_time_column_indexes(source_header)
            time_index_set = set(time_indexes)
            value_columns = list(columns)
            non_time_value_count = sum(1 for idx, _name in value_columns if int(idx) not in time_index_set)
            if not time_index_set and len(value_columns) == 1 and "timestamp" in slug.lower():
                non_time_value_count = 0
            headers: list[str] = [REGER_TIME_COLUMN]
            seen: set[str] = set()
            for col_idx, col_name in value_columns:
                raw = str(col_name).strip()
                if not raw and 0 <= col_idx < len(source_header):
                    raw = str(source_header[col_idx]).strip()
                if not raw:
                    raw = f"column_{col_idx}"
                header = raw
                if header in seen:
                    header = f"{raw}_{col_idx}"
                seen.add(header)
                headers.append(header)
            writer = csv.writer(tmp_csv, lineterminator="\n")
            writer.writerow(headers)
            row_count = 0
            for row in reader:
                if time_indexes and all(0 <= idx < len(row) for idx in time_indexes):
                    time_text = " ".join(str(row[idx]).strip() for idx in time_indexes)
                    time_value = _parse_time_cell(time_text, row_count)
                else:
                    time_value = row_count
                row_count += 1
                writer.writerow(
                    [time_value]
                    + [row[idx] if 0 <= idx < len(row) else "" for idx, _name in value_columns]
                )
    except Exception:
        try:
            tmp_csv.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)
        raise
    finally:
        try:
            tmp_csv.close()
        except Exception:
            pass
    return path, row_count, headers, value_columns, non_time_value_count


def _write_temp_column_info(
    column_info_rows: list[dict],
    columns: list[tuple[int, str]],
    headers: list[str],
) -> Path:
    fieldnames = list(column_info_rows[0].keys()) if column_info_rows else []
    for field in ("column_index", "column_name", "logical_type", "max_point_number"):
        if field not in fieldnames:
            fieldnames.append(field)
    rows_by_idx: dict[int, dict] = {}
    for row in column_info_rows:
        try:
            rows_by_idx[int(row.get("column_index"))] = row
        except (TypeError, ValueError):
            continue
    tmp_info = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_reger_column_info.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )
    path = Path(tmp_info.name)
    try:
        writer = csv.DictWriter(tmp_info, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        value_headers = list(headers)
        start_idx = 0
        if value_headers and value_headers[0] == REGER_TIME_COLUMN:
            time_row = {
                "column_index": 0,
                "column_name": REGER_TIME_COLUMN,
                "logical_type": "integer",
                "max_point_number": 0,
            }
            writer.writerow({field: time_row.get(field, "") for field in fieldnames})
            value_headers = value_headers[1:]
            start_idx = 1
        for new_idx, ((old_idx, _name), header) in enumerate(
            zip(columns, value_headers),
            start=start_idx,
        ):
            row = dict(rows_by_idx.get(int(old_idx), {}))
            row["column_index"] = int(new_idx)
            row["column_name"] = header
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    finally:
        tmp_info.close()
    return path


def _runs_reger(selected_algorithms: list[str] | None) -> bool:
    if selected_algorithms is None:
        return True
    return any(str(name).strip().lower() == "reger" for name in selected_algorithms)


def _only_reger(selected_algorithms: list[str] | None) -> bool:
    return bool(selected_algorithms) and all(
        str(name).strip().lower() == "reger" for name in selected_algorithms
    )


def _use_native_reger_backend() -> bool:
    raw = os.environ.get("WEB_COMPRESSION_PYTHON_REGER_NATIVE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="SLUG", help="Only regenerate this catalog key.")
    parser.add_argument(
        "--dtype",
        choices=("int", "float"),
        help="Only regenerate reports for this logical dtype.",
    )
    parser.add_argument(
        "--algorithms",
        metavar="NAMES",
        help=(
            "Comma-separated algorithm row names to benchmark (e.g. TS_2DIFF,ts_2diff_bos,brieflz_1). "
            "When set, only these algorithms run and existing rows for them are replaced; "
            "other algorithm rows in the report CSV are kept."
        ),
    )
    args = parser.parse_args()
    selected_algorithms = parse_algorithms(args.algorithms)
    if selected_algorithms:
        configure_allowlist(selected_algorithms)
        print(f"Allowlist algorithms ({len(selected_algorithms)}): {', '.join(selected_algorithms)}")

    pairs = iter_catalog_benchmark_datasets(BACKEND)
    if args.dataset:
        want = args.dataset.strip()
        pairs = [(s, p) for s, p in pairs if s == want]
        if not pairs:
            raise SystemExit(f"No catalog dataset matches --dataset {want!r}")
    if not pairs:
        print(
            f"No datasets found under {describe_dataset_csv_lookup(BACKEND)} "
            "with column_info in backend/dataset/"
        )
        return

    patch = os.environ.get("WEB_COMPRESSION_CSV_PATCH_ROWS", "10000")
    for slug, src in pairs:
        source_header = _read_source_header(src)
        out_root = BACKEND / "dataset" / slug
        out_root.mkdir(parents=True, exist_ok=True)
        col_info = out_root / f"{slug}_column_info.csv"
        if not col_info.is_file():
            raise SystemExit(
                f"Missing {col_info.name}; run regenerate_dataset_info.py first."
            )
        column_info_rows = _iter_column_info_rows(col_info)
        dtypes = ("int", "float")
        if args.dtype:
            dtypes = (args.dtype,)
        for dt in dtypes:
            report_path = out_root / f"{slug}_{dt}_python_compression_result.csv"
            columns = _columns_for_dtype(column_info_rows, dt)
            if not columns:
                print(f"Skip {slug} {dt}: no columns from column_info")
                write_report_rows(report_path, [], format_loss_rate=format_loss_rate_csv)
                continue
            valid_columns = [
                (idx, name)
                for idx, name in sorted(columns, key=lambda item: item[0])
                if 0 <= idx < len(source_header)
            ]
            skipped_indices = sorted({idx for idx, _name in columns} - {idx for idx, _name in valid_columns})
            for idx in skipped_indices:
                print(f"  -> Skip {slug} {dt} column_index={idx}: out of range")
            if not valid_columns:
                print(f"Skip {slug} {dt}: no in-range source columns")
                continue
            column_csv_paths: dict[int, Path] = {}
            cleanup_column_csv_paths: set[Path] = set()
            if not _only_reger(selected_algorithms):
                column_csv_paths, row_count, cleanup_column_csv_paths = _materialize_column_csvs(src, slug, dt, valid_columns)
            else:
                row_count = "streaming"
            print(
                f"Regenerate {report_path.name} "
                f"({len(columns)} columns, {row_count} rows, patch={patch}, column_info={col_info.name})"
            )
            existing_rows = (
                load_report_rows(report_path)
                if selected_algorithms and report_path.is_file()
                else []
            )
            rows: list[dict] = []
            try:
                if not _only_reger(selected_algorithms):
                    for col_idx, _col_name in columns:
                        tmp_csv_path = column_csv_paths.get(int(col_idx))
                        if tmp_csv_path is None:
                            continue
                        tmp_report_path = tmp_csv_path.with_name(f"{tmp_csv_path.stem}_report.csv")
                        tmp_report_path.unlink(missing_ok=True)
                        try:
                            print(f"  -> Python bench {slug} {dt} column_index={col_idx}")
                            process_csv_single_column_patch(
                                str(tmp_csv_path),
                                dt,
                                str(tmp_report_path),
                                str(col_info),
                                int(col_idx),
                            )
                            with tmp_report_path.open("r", encoding="utf-8", newline="") as handle:
                                for rec in csv.DictReader(handle):
                                    algo = str(rec.get("algorithm", "")).strip()
                                    if not algo or algo == "reger":
                                        continue
                                    try:
                                        row_col_idx = int(rec.get("column_index"))
                                    except (TypeError, ValueError):
                                        row_col_idx = int(col_idx)
                                    rows.append(
                                        {
                                            "column_index": row_col_idx,
                                            "algorithm": algo,
                                            "originalSize": int(float(rec.get("originalSize", 0) or 0)),
                                            "compressedSize": int(float(rec.get("compressedSize", 0) or 0)),
                                            "compressionTime": int(float(rec.get("compressionTime", 0) or 0)),
                                            "decompressionTime": int(float(rec.get("decompressionTime", 0) or 0)),
                                            "isLossy": int(float(rec.get("isLossy", 0) or 0)),
                                            "lossRate": float(rec.get("lossRate", 0) or 0),
                                        }
                                    )
                        finally:
                            tmp_report_path.unlink(missing_ok=True)
                if dt in ("int", "float") and _runs_reger(selected_algorithms):
                    multi_csv_path, _multi_rows, headers, multi_columns, non_time_value_count = _materialize_multi_column_csv(
                        src, slug, dt, valid_columns
                    )
                    if not multi_columns:
                        print(f"  -> Skip Python REGER multi-column for {slug} {dt}: no value columns")
                        multi_csv_path.unlink(missing_ok=True)
                    elif non_time_value_count <= 0:
                        print(
                            f"  -> Skip Python REGER multi-column for {slug} {dt}: "
                            "no non-time value columns"
                        )
                        multi_csv_path.unlink(missing_ok=True)
                    else:
                        multi_info_path = _write_temp_column_info(column_info_rows, multi_columns, headers)
                        multi_report_path = multi_csv_path.with_name(f"{multi_csv_path.stem}_report.csv")
                        try:
                            print(f"  -> Python bench {slug} {dt} REGER multi-column ({len(multi_columns)} value columns)")
                            if _use_native_reger_backend():
                                from regenerate_staticdata_cpp import run_cpp_benchmark

                                run_cpp_benchmark(
                                    multi_csv_path,
                                    dt,
                                    multi_report_path,
                                    column_info=multi_info_path,
                                    algorithms=["reger"],
                                )
                            else:
                                with _temporary_reger_allowlist():
                                    process_csv_simple_patch(
                                        str(multi_csv_path),
                                        dt,
                                        str(multi_report_path),
                                        str(multi_info_path),
                                    )
                            with multi_report_path.open("r", encoding="utf-8", newline="") as handle:
                                for rec in csv.DictReader(handle):
                                    algo = str(rec.get("algorithm", "")).strip()
                                    if algo != "reger":
                                        continue
                                    rows.append(
                                        {
                                            "column_index": -1,
                                            "algorithm": algo,
                                            "originalSize": int(float(rec.get("originalSize", 0) or 0)),
                                            "compressedSize": int(float(rec.get("compressedSize", 0) or 0)),
                                            "compressionTime": int(float(rec.get("compressionTime", 0) or 0)),
                                            "decompressionTime": int(float(rec.get("decompressionTime", 0) or 0)),
                                            "isLossy": int(float(rec.get("isLossy", 0) or 0)),
                                            "lossRate": float(rec.get("lossRate", 0) or 0),
                                        }
                                    )
                        finally:
                            multi_csv_path.unlink(missing_ok=True)
                            multi_info_path.unlink(missing_ok=True)
                            multi_report_path.unlink(missing_ok=True)
            finally:
                for tmp_csv_path in cleanup_column_csv_paths:
                    tmp_csv_path.unlink(missing_ok=True)
            if selected_algorithms and existing_rows:
                rows = merge_partial_algorithm_rows(
                    existing_rows,
                    rows,
                    set(selected_algorithms),
                )
            rows.sort(key=lambda r: (int(r["column_index"]), str(r["algorithm"])))
            write_report_rows(report_path, rows, format_loss_rate=format_loss_rate_csv)
            print(
                f"  -> wrote {report_path.name}: {len(rows)} rows "
                f"(column_index + algorithm metrics)"
            )
    print("Done.")


if __name__ == "__main__":
    main()
