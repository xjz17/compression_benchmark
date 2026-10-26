#!/usr/bin/env python3
"""Rebuild *_java_compression_result.csv under backend/dataset/<slug>/ from master CSVs.

Sources: ``backend/dataset_csv/<slug>.csv`` (see ``iter_catalog_benchmark_datasets``).

cmix is **CPU-only** (no GPU). It is very slow; **by default this script does not run cmix**
(``COMPRESSION_BENCHMARK_CMIX_BENCH`` defaults to ``0``). To include ``ext_cmix`` in static CSVs, run with
``export COMPRESSION_BENCHMARK_CMIX_BENCH=1`` (optionally set ``COMPRESSION_BENCHMARK_CMIX_MAX_COLUMN_BYTES``,
``COMPRESSION_BENCHMARK_CMIX_TIMEOUT_SEC``, ``COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS``).

**GPU:** MSDZip/MSDLC need CUDA. This script sets ``COMPRESSION_BENCHMARK_MSDZIP_BENCH`` / ``MSDLC`` to ``0``
by default. If your shell still has ``=1`` from elsewhere, run
``unset COMPRESSION_BENCHMARK_MSDZIP_BENCH COMPRESSION_BENCHMARK_MSDLC_BENCH`` or use a fresh shell.

**Research float rows (Sim-Piece, Elf, Camel, SZ, SZ3, Serf):** Precomputed CSVs only contain rows that
successfully ran when the script was executed. Build or install the native tools the JVM benches expect:

- **Sim-Piece:** in-repo ``SimPieceFloatNative`` (``java/target/benchmark-java-*.jar`` only; no upstream JAR).
- **SZ:** executable at ``external/compression/SZ/example/sz`` or CMake ``external/compression/SZ/build/bin/sz``, plus ``external/compression/SZ/example/sz.config``.
- **SZ3:** ``external/compression/SZ3/build/tools/sz3/sz3`` (or ``COMPRESSION_BENCHMARK_SZ3``).
- **SZ3:** on by default when ``sz3`` is installed (``COMPRESSION_BENCHMARK_SZ3_BENCH=1``; set ``0`` to skip).
- **BUFF kernel native:** on by default (``COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE=1``; in-process, no ``comp_profiler``).
- **SZ:** on by default when ``sz`` is installed (``COMPRESSION_BENCHMARK_SZ_BENCH=1``; set ``0`` to skip).
- **Serf / BUFF (Rust ``comp_profiler``):** off by default (``COMPRESSION_BENCHMARK_{SERF,BUFF}_BENCH=0``).
- **fpzip / FPCompress:** ``fpzip``, ``fpcompress_dpratio`` and ``fpcompress_dpspeed`` are
  emitted by the shared tri-language ``NativeLzFamilyBench`` implementation. The old CLI paths are not used.
- **SPDP** (``spdp_level_0`` … ``spdp_level_9``): in-process Java in ``SpdpFloat.java`` (reference
  ``benchmark/reference/upstream/spdp/standalone/SPDP_11.c``). All ten levels are always benchmarked
  (like ``lzma_level_*``); use ``--algorithms spdp_level_N`` for partial refresh.
- **FPC** (``fpc``): in-process Java in ``FpcDouble.java`` (reference
  ``benchmark/reference/upstream/fpc/fpc.c``). Predictor table size lg2 defaults to 16;
  override with ``COMPRESSION_BENCHMARK_FPC_PRED_SIZE_LG2``.
- **pFPC** (``pfpc``): in-process Java in ``PfpcDouble.java`` (reference
  ``benchmark/reference/upstream/pfpc/pFPC.c``). Defaults: pred lg2=20, threads=8,
  chunksize=2048; override with ``COMPRESSION_BENCHMARK_PFPC_PRED_SIZE_LG2``,
  ``COMPRESSION_BENCHMARK_PFPC_THREADS``, ``COMPRESSION_BENCHMARK_PFPC_CHUNKSIZE``.
- **ZFP** (``ext_zfp_zfpy``, ``ext_zfp_tol_1e-*``): ``ResearchZfpCliBench`` calls the
  ``zfp`` CLI on float64 column bytes ({@code -d}). Build standalone in
  ``external/compression/zfp`` (see README Research codecs). Override with
  ``COMPRESSION_BENCHMARK_ZFP``; skip with ``COMPRESSION_BENCHMARK_ZFP_BENCH=0``.
- **Blosc / Blosc2:** ``blosc_blosclz`` and ``blosc2_blosclz`` are emitted by
  ``NativeLzFamilyBench``. The old column CLI paths are not used.

This script **clears** any ``COMPRESSION_BENCHMARK_CPP_NATIVE_*`` variables from the environment so JVM
``ExternalCompressionBench`` rows are not skipped (those skips are for ``cpp_java_companion`` when C++
already emitted the same algorithms).

**Heap:** ``java_bench_common`` defaults to ``-Xmx8g`` when ``COMPRESSION_BENCHMARK_JAVA_OPTS`` is unset.
Override with ``export COMPRESSION_BENCHMARK_JAVA_OPTS='-Xmx16g'`` or disable with ``export COMPRESSION_BENCHMARK_JAVA_OPTS=''``.

**Report write mode:** By default, results are upserted by ``(column_index, algorithm)`` into
existing ``*_java_compression_result.csv`` files. Other rows are retained. For a full rewrite,
run with ``export COMPRESSION_BENCHMARK_STATICDATA_REGEN=1``.

**Partial algorithm refresh:** ``--algorithms TS_2DIFF,ts_2diff_bos,brieflz_1`` runs only the
listed encoders (including lzbench row names); other rows in each dtype CSV are preserved.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# cmix: cap per-column input + per-invocation timeout so regeneration finishes in reasonable time.
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS", "1")
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_MAX_COLUMN_BYTES", str(64 * 1024))
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_TIMEOUT_SEC", "180")
os.environ.setdefault("COMPRESSION_BENCHMARK_MSDZIP_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_MSDLC_BENCH", "0")
# Slow native-CLI rows: Serf/BUFF off; SZ/SZ3 on when binary exists (set *_BENCH=0 to skip).
os.environ.setdefault("COMPRESSION_BENCHMARK_SERF_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_SZ_BENCH", "1")
os.environ.setdefault("COMPRESSION_BENCHMARK_SZ3_BENCH", "1")
os.environ.setdefault("COMPRESSION_BENCHMARK_BUFF_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE", "1")
os.environ.setdefault("COMPRESSION_BENCHMARK_FPZIP_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_BLOSC_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_BLOSC2_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_LZBENCH_BENCH", "0")
os.environ.setdefault(
    "COMPRESSION_BENCHMARK_FPCOMPRESS_BIN_ROOT",
    str((ROOT / "external" / "runtime" / "fpcompress").resolve()),
)
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_csv_source import describe_dataset_csv_lookup, iter_catalog_benchmark_datasets
from staticdata_regen_common import (  # noqa: E402
    REPORT_FIELDS,
    configure_allowlist,
    load_report_rows,
    merge_partial_algorithm_rows,
    parse_algorithms,
    upsert_report_rows,
    write_report_rows,
)

import pandas as pd

from java_bench_common import JAR, run_java_benchmark

_csv_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_field_limit)
        break
    except OverflowError:
        _csv_field_limit //= 10

BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from myapp.benchmark_loss import format_loss_rate_csv as _format_loss_rate  # noqa: E402

# Datasets: backend/dataset_csv/<slug>.csv + backend/dataset/<slug>/column_info.
REGER_TIME_COLUMN = "__reger_time__"

_JAVA_EXECUTION_ALGORITHM_ALIASES = {
    "CHIMP128": "CHIMP",
}


def _java_execution_algorithms(algorithms: list[str] | None) -> list[str] | None:
    if not algorithms:
        return algorithms
    out: list[str] = []
    seen: set[str] = set()
    for algorithm in algorithms:
        name = str(algorithm or "").strip()
        if not name:
            continue
        mapped = _JAVA_EXECUTION_ALGORITHM_ALIASES.get(name, name)
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out or None


def _java_report_algorithm(
    algorithm: str,
    requested_algorithms: list[str] | None,
) -> str:
    if (
        algorithm == "CHIMP"
        and requested_algorithms
        and "CHIMP128" in requested_algorithms
        and "CHIMP" not in requested_algorithms
    ):
        return "CHIMP128"
    return algorithm


def _normalize_logical_type(raw: object) -> str:
    return str(raw or "").strip().lower()


def _is_int_logical(logical: str) -> bool:
    return logical in ("integer", "int", "long", "timestamp")


def _is_float_logical(logical: str) -> bool:
    return logical in ("float", "double")


def _is_string_logical(logical: str) -> bool:
    return logical == "string"


def _iter_column_info_rows(column_info_path: Path) -> list[dict]:
    with column_info_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _staticdata_report_out_root(slug: str) -> Path:
    override = os.environ.get("COMPRESSION_BENCHMARK_STATICDATA_REPORT_ROOT", "").strip()
    if not override:
        return BACKEND / "dataset" / slug
    out_root = Path(os.path.expanduser(override)).resolve() / slug
    out_root.mkdir(parents=True, exist_ok=True)
    src_info = BACKEND / "dataset" / slug / f"{slug}_column_info.csv"
    dst_info = out_root / src_info.name
    if src_info.is_file() and not dst_info.is_file():
        shutil.copy2(src_info, dst_info)
    return out_root


def _columns_for_dtype(
    column_info_rows: list[dict],
    data_type: str,
) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for row in column_info_rows:
        logical = _normalize_logical_type(row.get("logical_type"))
        if data_type == "int" and not _is_int_logical(logical):
            continue
        if data_type == "float" and not _is_float_logical(logical):
            continue
        if data_type == "string" and not _is_string_logical(logical):
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


def _load_report_rows(report_path: Path) -> list[dict]:
    return load_report_rows(report_path)


def _parse_algorithms(raw: str | None) -> list[str] | None:
    return parse_algorithms(raw)


def _merge_partial_algorithm_rows(
    existing_rows: list[dict],
    new_rows: list[dict],
    algorithms: set[str],
) -> list[dict]:
    return merge_partial_algorithm_rows(existing_rows, new_rows, algorithms)


def _write_report_rows(report_path: Path, rows: list[dict]) -> None:
    write_report_rows(report_path, rows, format_loss_rate=_format_loss_rate)


def _persist_report_rows(report_path: Path, rows: list[dict]) -> list[dict]:
    return upsert_report_rows(
        report_path,
        rows,
        format_loss_rate=_format_loss_rate,
        full_rewrite=os.environ.get("COMPRESSION_BENCHMARK_STATICDATA_REGEN", "").strip() == "1",
    )


def _run_materialized_column_java_bench(
    *,
    tmp_csv_path: Path,
    slug: str,
    data_type: str,
    column_index: int,
    column_info_path: Path,
    algorithms: list[str] | None = None,
) -> list[dict]:
    tmp_report_path = tmp_csv_path.with_name(
        f"{tmp_csv_path.stem}_report.csv"
    )
    tmp_report_path.unlink(missing_ok=True)
    try:
        run_java_benchmark(
            tmp_csv_path,
            data_type,
            tmp_report_path,
            column_info=column_info_path,
            column_index=column_index,
            algorithms=_java_execution_algorithms(algorithms),
        )
        try:
            bench_df = pd.read_csv(tmp_report_path)
        except Exception as exc:
            print(
                f"Skip {slug} {data_type} col={column_index}: failed to parse tmp Java report: {exc}"
            )
            return []
        rows: list[dict] = []
        for _, row in bench_df.iterrows():
            algo = str(row.get("algorithm", "")).strip()
            if not algo:
                continue
            algo = _java_report_algorithm(algo, algorithms)
            try:
                row_col_idx = int(row.get("column_index"))
            except (TypeError, ValueError):
                row_col_idx = int(column_index)
            rows.append(
                {
                    "column_index": row_col_idx,
                    "algorithm": algo,
                    "originalSize": int(float(row.get("originalSize", 0) or 0)),
                    "compressedSize": int(float(row.get("compressedSize", 0) or 0)),
                    "compressionTime": int(float(row.get("compressionTime", 0) or 0)),
                    "decompressionTime": int(float(row.get("decompressionTime", 0) or 0)),
                    "isLossy": int(float(row.get("isLossy", 0) or 0)),
                    "lossRate": float(row.get("lossRate", 0) or 0),
                }
            )
        return rows
    finally:
        tmp_report_path.unlink(missing_ok=True)


def _run_single_column_java_bench(
    *,
    raw_df: pd.DataFrame,
    slug: str,
    data_type: str,
    column_index: int,
    column_info_path: Path,
    algorithms: list[str] | None = None,
) -> list[dict]:
    if column_index < 0 or column_index >= raw_df.shape[1]:
        print(
            f"Skip {slug} {data_type} col={column_index}: out of range for master CSV with {raw_df.shape[1]} columns"
        )
        return []
    one_col_df = raw_df.iloc[:, [column_index]]
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{slug}_{data_type}_col{column_index}.csv",
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as tmp_csv:
        one_col_df.to_csv(tmp_csv.name, index=False, lineterminator="\n")
        tmp_csv_path = Path(tmp_csv.name)
    try:
        return _run_materialized_column_java_bench(
            tmp_csv_path=tmp_csv_path,
            slug=slug,
            data_type=data_type,
            column_index=column_index,
            column_info_path=column_info_path,
            algorithms=algorithms,
        )
    finally:
        tmp_csv_path.unlink(missing_ok=True)


def _read_source_header(src: Path) -> list[str]:
    with src.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _clean_header_name(name: object) -> str:
    return str(name or "").strip().removeprefix("\ufeff")


def _source_index_for_column(
    source_header: list[str],
    column_index: int,
    column_name: str,
) -> int | None:
    target = _clean_header_name(column_name)
    if 0 <= column_index < len(source_header):
        if not target or _clean_header_name(source_header[column_index]) == target:
            return column_index
    if target:
        for idx, name in enumerate(source_header):
            if _clean_header_name(name) == target:
                return idx
    return None


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


def _materialize_column_csvs(
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str]],
) -> tuple[dict[int, Path], int]:
    column_info_path = BACKEND / "dataset" / slug / f"{slug}_column_info.csv"
    timestamp_indices = {
        int(row["column_index"])
        for row in _iter_column_info_rows(column_info_path)
        if _normalize_logical_type(row.get("logical_type")) == "timestamp"
    }
    paths: dict[int, Path] = {}
    handles = []
    writers = []
    try:
        with src.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.reader(input_handle)
            source_header = next(reader, [])
            for col_idx, _col_name in columns:
                source_idx = _source_index_for_column(source_header, int(col_idx), _col_name)
                if source_idx is None:
                    continue
                tmp_csv = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=f"_{slug}_{data_type}_col{col_idx}.csv",
                    delete=False,
                    encoding="utf-8",
                    newline="",
                )
                writer = csv.writer(tmp_csv, lineterminator="\n")
                header = _col_name or source_header[source_idx] or f"column_{col_idx}"
                writer.writerow([header])
                handles.append(tmp_csv)
                paths[int(col_idx)] = Path(tmp_csv.name)
                writers.append((int(col_idx), int(source_idx), writer))

            row_count = 0
            for row in reader:
                row_count += 1
                for col_idx, source_idx, writer in writers:
                    cell = row[source_idx] if 0 <= source_idx < len(row) else ""
                    if data_type == "int" and col_idx in timestamp_indices:
                        cell = _parse_time_cell(cell, row_count - 1)
                    writer.writerow([cell])
    except Exception:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
    return paths, row_count


def _column_csvs_for_run(
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str]],
) -> tuple[dict[int, Path], int | str, bool]:
    cache_root_raw = os.environ.get("COMPRESSION_BENCHMARK_COLUMN_CACHE_DIR", "").strip()
    if not cache_root_raw:
        paths, row_count = _materialize_column_csvs(src, slug, data_type, columns)
        return paths, row_count, False
    cache_dir = Path(cache_root_raw).expanduser().resolve() / slug / data_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    missing: list[tuple[int, str]] = []
    for col_idx, col_name in columns:
        cached = cache_dir / f"column_{int(col_idx)}.csv"
        if cached.is_file() and cached.stat().st_size > 0:
            paths[int(col_idx)] = cached
        else:
            missing.append((int(col_idx), col_name))
    row_count: int | str = "cached"
    if missing:
        generated, row_count = _materialize_column_csvs(src, slug, data_type, missing)
        for col_idx, generated_path in generated.items():
            cached = cache_dir / f"column_{int(col_idx)}.csv"
            # /tmp and the project volume can be different filesystems on the
            # server. Copy through a cache-local temporary file, then atomically
            # publish it so an interrupted scan is never treated as valid.
            cached_tmp = cached.with_suffix(cached.suffix + ".tmp")
            shutil.copyfile(generated_path, cached_tmp)
            os.replace(cached_tmp, cached)
            generated_path.unlink(missing_ok=True)
            paths[int(col_idx)] = cached
    return paths, row_count, True


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
            value_columns = [
                (int(col_idx), col_name, source_idx)
                for col_idx, col_name in columns
                for source_idx in [_source_index_for_column(source_header, int(col_idx), col_name)]
                if source_idx is not None
            ]
            non_time_value_count = sum(1 for _idx, _name, source_idx in value_columns if int(source_idx) not in time_index_set)
            if not time_index_set and len(value_columns) == 1 and "timestamp" in slug.lower():
                non_time_value_count = 0
            headers: list[str] = [REGER_TIME_COLUMN]
            seen: set[str] = set()
            for col_idx, col_name, source_idx in value_columns:
                raw = str(col_name).strip()
                if not raw and 0 <= source_idx < len(source_header):
                    raw = str(source_header[source_idx]).strip()
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
                    + [
                        row[source_idx] if 0 <= source_idx < len(row) else ""
                        for _idx, _name, source_idx in value_columns
                    ]
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
    return (
        path,
        row_count,
        headers,
        [(idx, name) for idx, name, _source_idx in value_columns],
        non_time_value_count,
    )


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


def _count_data_rows(src: Path) -> int:
    with src.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


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
            "When set, only these algorithms run. Results are upserted by physical column and "
            "algorithm; every other existing row is kept."
        ),
    )
    parser.add_argument("--column-indexes", metavar="INDEXES", help="Comma-separated physical column indexes to benchmark.")
    parser.add_argument("--prepare-column-cache", action="store_true", help="Materialize selected columns into COMPRESSION_BENCHMARK_COLUMN_CACHE_DIR and exit.")
    args = parser.parse_args()
    selected_column_indexes = {
        int(part.strip()) for part in (args.column_indexes or "").split(",") if part.strip()
    }
    selected_algorithms = _parse_algorithms(args.algorithms)
    if selected_algorithms:
        selected_execution_algorithms = _java_execution_algorithms(selected_algorithms)
        configure_allowlist(selected_execution_algorithms)
        print(f"Allowlist algorithms ({len(selected_algorithms)}): {', '.join(selected_algorithms)}")
        if selected_execution_algorithms != selected_algorithms:
            print(f"Java execution algorithms: {', '.join(selected_execution_algorithms or [])}")

    # Standalone staticdata must include JVM ExternalCompressionBench rows; do not inherit
    # cpp_java_companion "skip duplicate" flags from the caller's shell.
    for _k in list(os.environ.keys()):
        if _k.startswith("COMPRESSION_BENCHMARK_CPP_NATIVE"):
            del os.environ[_k]
    if not JAR.is_file():
        raise SystemExit(
            f"Missing benchmark JAR ({JAR}). Build: cd java && mvn -DskipTests package\n"
            "  (uses java/target/benchmark-java-1.0-SNAPSHOT.jar automatically; "
            "optional: cp target/*.jar ../backend/lib/)"
        )
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
    os.environ.setdefault("COMPRESSION_BENCHMARK_CSV_PATCH_ROWS", "10000")
    os.environ.setdefault("COMPRESSION_BENCHMARK_BOS_ENCODE_ROWS", "2048")
    for slug, src in pairs:
        source_header = _read_source_header(src)
        raw = None
        out_root = _staticdata_report_out_root(slug)
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
            report_path = out_root / f"{slug}_{dt}_java_compression_result.csv"
            columns = _columns_for_dtype(column_info_rows, dt)
            if selected_column_indexes:
                columns = [item for item in columns if int(item[0]) in selected_column_indexes]
            if not columns:
                print(f"Skip {slug} {dt}: no columns from column_info")
                _persist_report_rows(report_path, [])
                continue
            report_rows: list[dict] = []
            valid_columns = [
                (idx, name)
                for idx, name in sorted(columns, key=lambda item: item[0])
                if _source_index_for_column(source_header, int(idx), name) is not None
            ]
            if args.prepare_column_cache:
                cached_paths, row_count, persistent = _column_csvs_for_run(
                    src, slug, dt, valid_columns
                )
                if not persistent:
                    for cached_path in cached_paths.values():
                        cached_path.unlink(missing_ok=True)
                    raise SystemExit("--prepare-column-cache requires COMPRESSION_BENCHMARK_COLUMN_CACHE_DIR")
                print(f"Prepared column cache for {slug} {dt}: {len(cached_paths)} columns, {row_count} rows")
                continue
            skipped_indices = sorted(
                {idx for idx, _name in columns} - {idx for idx, _name in valid_columns}
            )
            for idx in skipped_indices:
                name = next((n for i, n in columns if i == idx), "")
                print(
                    f"  -> Skip {slug} {dt} column_index={idx}: "
                    f"not found in source CSV header by index/name ({name!r})"
                )
            if _only_reger(selected_algorithms):
                row_count = "streaming"
                print(
                    f"Regenerate {report_path.name} "
                    f"({len(columns)} columns, {row_count} rows, patch={os.environ.get('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')})"
                )
                if dt in ("int", "float") and _runs_reger(selected_algorithms) and valid_columns:
                    multi_csv_path, _multi_rows, headers, multi_columns, non_time_value_count = _materialize_multi_column_csv(
                        src, slug, dt, valid_columns
                    )
                    if not multi_columns:
                        print(f"  -> Skip Java REGER multi-column for {slug} {dt}: no value columns")
                        multi_csv_path.unlink(missing_ok=True)
                    elif non_time_value_count <= 0:
                        print(
                            f"  -> Skip Java REGER multi-column for {slug} {dt}: "
                            "no non-time value columns"
                        )
                        multi_csv_path.unlink(missing_ok=True)
                    else:
                        multi_info_path = _write_temp_column_info(column_info_rows, multi_columns, headers)
                        multi_report_path = multi_csv_path.with_name(f"{multi_csv_path.stem}_report.csv")
                        try:
                            print(f"  -> Java bench {slug} {dt} REGER multi-column ({len(multi_columns)} value columns)")
                            run_java_benchmark(
                                multi_csv_path,
                                dt,
                                multi_report_path,
                                column_info=multi_info_path,
                                algorithms=["reger"],
                            )
                            try:
                                bench_df = pd.read_csv(multi_report_path)
                            except Exception as exc:
                                print(f"  -> Skip Java REGER multi report parse: {exc}")
                                bench_df = pd.DataFrame()
                            for _, row in bench_df.iterrows():
                                algo = str(row.get("algorithm", "")).strip()
                                if algo != "reger":
                                    continue
                                report_rows.append(
                                    {
                                        "column_index": -1,
                                        "algorithm": algo,
                                        "originalSize": int(float(row.get("originalSize", 0) or 0)),
                                        "compressedSize": int(float(row.get("compressedSize", 0) or 0)),
                                        "compressionTime": int(float(row.get("compressionTime", 0) or 0)),
                                        "decompressionTime": int(float(row.get("decompressionTime", 0) or 0)),
                                        "isLossy": int(float(row.get("isLossy", 0) or 0)),
                                        "lossRate": float(row.get("lossRate", 0) or 0),
                                    }
                                )
                        finally:
                            multi_csv_path.unlink(missing_ok=True)
                            multi_info_path.unlink(missing_ok=True)
                            multi_report_path.unlink(missing_ok=True)
                report_rows = _persist_report_rows(report_path, report_rows)
                print(
                    f"  -> wrote {report_path.name}: {len(report_rows)} rows "
                    f"(column_index + algorithm metrics)"
                )
                continue
            column_csv_paths, row_count, persistent_column_cache = _column_csvs_for_run(
                src, slug, dt, valid_columns
            )
            print(
                f"Regenerate {report_path.name} "
                f"({len(columns)} columns, {row_count} rows, patch={os.environ.get('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')})"
            )
            try:
                for col_idx, _col_name in columns:
                    tmp_csv_path = column_csv_paths.get(int(col_idx))
                    if tmp_csv_path is None:
                        continue
                    print(
                        f"  -> Java bench {slug} {dt} column_index={col_idx}"
                    )
                    report_rows.extend(
                        _run_materialized_column_java_bench(
                            tmp_csv_path=tmp_csv_path,
                            slug=slug,
                            data_type=dt,
                            column_index=col_idx,
                            column_info_path=col_info,
                            algorithms=selected_algorithms,
                        )
                    )
            finally:
                if not persistent_column_cache:
                    for tmp_csv_path in column_csv_paths.values():
                        tmp_csv_path.unlink(missing_ok=True)
            report_rows = _persist_report_rows(report_path, report_rows)
            print(
                f"  -> wrote {report_path.name}: {len(report_rows)} rows "
                f"(column_index + algorithm metrics)"
            )
    print("Done.")


if __name__ == "__main__":
    main()
