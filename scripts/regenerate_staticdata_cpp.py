#!/usr/bin/env python3
"""
Rebuild ``*_cpp_compression_result.csv`` under ``backend/dataset/<slug>/``.

Uses the **native** ``benchmark_cpp`` binary (``cpp/build/benchmark_cpp``). That driver runs
C++ for byte compressors, column encodings, TSFile encodings, FastPFOR/Chimp/LeCo/ALP/ELF/Camel/Sim-Piece,
and SZ/SZ3/Serf/BUFF (SZ/SZ3/Serf: external CLIs or in-process; BUFF: ``comp_profiler`` when built).
It may invoke ``scripts/cpp_java_companion.py`` for reserved JAR-only rows (with current env, migrated codecs are skipped in Java).

Requires:

- Built ``benchmark_cpp`` (``cd cpp && cmake -B build -S . && cmake --build build``), or set
  ``COMPRESSION_BENCHMARK_BENCHMARK_CPP`` to the executable path.
- Catalog ``{slug}_column_info.csv`` (from ``regenerate_dataset_info.py``); passed as ``--column-info``.
- Java JAR + research classpath (same as Java staticdata) for companion phases.
- This script sets ``COMPRESSION_BENCHMARK_PYTHON`` to the current interpreter so ``benchmark_cpp`` can run
  ``cpp_java_companion.py`` without requiring a specific Python minor version on PATH. For a manual
  ``benchmark_cpp`` run, set ``COMPRESSION_BENCHMARK_PYTHON`` to your venv's ``python`` / ``python3`` if needed.
- For this script: ``pip install numpy pandas`` in the same environment (read CSV / filter columns).
- If you changed ``java/`` (e.g. ``Ts2DiffBosImprove``), rebuild the JAR:
  ``cd java && mvn -DskipTests package && cp target/benchmark-java-1.0-SNAPSHOT.jar ../backend/lib/``
- Serf: if ``sfz`` hangs during compress/decompress, set ``COMPRESSION_BENCHMARK_SERF_BENCH=0``, or
  ``COMPRESSION_BENCHMARK_SERF_SHELL_TIMEOUT_SEC=120`` (seconds per call; ``0`` = no timeout), or
  ``COMPRESSION_BENCHMARK_CPP_SKIP_SZ_SERF=1`` to skip SZ/SZ3/Serf entirely.
- BUFF (``buff_scalar``, ``buff_simd``): build ``comp_profiler`` under ``external/compression/buff/database``
  (Rust nightly). Rows emit when ``COMPRESSION_BENCHMARK_BUFF_BENCH=1`` or the binary exists (same as Java).
- FPCompress (``fpcompress_dpratio``, ``fpcompress_dpspeed``): emitted by the shared
  tri-language ``native_lz_family`` implementation. The old external CPU binary path is not used.
- **SPDP** (``spdp_level_0`` … ``spdp_level_9``): native C++ in ``cpp/src/spdp.c`` (reference
  ``benchmark/reference/upstream/spdp/standalone/SPDP_11.c``). All ten levels are always benchmarked
  (like ``lzma_level_*``); use ``--algorithms spdp_level_N`` for partial refresh.
- **FPC** (``fpc``): native C++ in ``cpp/src/fpc.c`` (reference
  ``benchmark/reference/upstream/fpc/fpc.c``). Predictor table size lg2 defaults to 16;
  override with ``COMPRESSION_BENCHMARK_FPC_PRED_SIZE_LG2``.
- **pFPC** (``pfpc``): native C++ in ``cpp/src/pfpc.c`` (reference
  ``benchmark/reference/upstream/pfpc/pFPC.c``). Defaults: pred lg2=20, threads=8,
  chunksize=2048; override with ``COMPRESSION_BENCHMARK_PFPC_PRED_SIZE_LG2``,
  ``COMPRESSION_BENCHMARK_PFPC_THREADS``, ``COMPRESSION_BENCHMARK_PFPC_CHUNKSIZE``.
- **ZFP** (``ext_zfp_zfpy``, ``ext_zfp_tol_1e-*``): in-process via ``cpp/src/zfp_codec.cpp``
  (submodule ``external/compression/zfp``, linked as ``zfp::zfp`` when configuring ``cpp/``).
  Java uses standalone ``zfp`` CLI (see README). Skip with ``COMPRESSION_BENCHMARK_ZFP_BENCH=0``.
- **Blosc / Blosc2** (``blosc_blosclz``, ``blosc2_blosclz``): emitted by the shared
  tri-language ``native_lz_family`` implementation. The old column CLI paths are not used.

**Patch rows:** ``benchmark_cpp`` reads the typed CSV in patches (default 10000 rows via
``COMPRESSION_BENCHMARK_CSV_PATCH_ROWS``), accumulates metrics, and uses ``column_info`` ``max_point_number``
for float scaling (no full-file decimal scan).

**Partial algorithm refresh:** ``--algorithms TS_2DIFF,ts_2diff_bos,brieflz_1`` runs only the
listed algorithms and merges them into the existing report (other rows are kept). Use with
``--dataset`` / ``--dtype`` to limit scope.

**Report write mode:** By default **appends** missing ``algorithm`` rows. For a full rewrite::

    export COMPRESSION_BENCHMARK_STATICDATA_REGEN=1
    python3 scripts/regenerate_staticdata_cpp.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Same cmix defaults as regenerate_staticdata_java.py (cmix off by default; MSD* need GPU).
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS", "1")
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_MAX_COLUMN_BYTES", str(64 * 1024))
os.environ.setdefault("COMPRESSION_BENCHMARK_CMIX_TIMEOUT_SEC", "180")
os.environ.setdefault("COMPRESSION_BENCHMARK_MSDZIP_BENCH", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_MSDLC_BENCH", "0")
# Serf (sfz) is slow per column; skip unless explicitly enabled (export COMPRESSION_BENCHMARK_SERF_BENCH=1).
os.environ.setdefault("COMPRESSION_BENCHMARK_SERF_BENCH", "0")

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
os.environ.setdefault("COMPRESSION_BENCHMARK_REPO_ROOT", str(ROOT.resolve()))

_csv_field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_field_limit)
        break
    except OverflowError:
        _csv_field_limit //= 10

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_csv_source import describe_dataset_csv_lookup, iter_catalog_benchmark_datasets
from staticdata_regen_common import (  # noqa: E402
    REPORT_FIELDS,
    apply_partial_algorithm_external_benches,
    configure_allowlist,
    load_report_rows,
    merge_partial_algorithm_rows,
    parse_algorithms,
    write_report_rows,
)
from java_bench_common import (  # noqa: E402
    JAR,
    _buff_env,
    _serf_env,
    benchmark_cpp_executable_path,
    require_benchmark_cpp_executable,
)

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from myapp.benchmark_loss import format_loss_rate_csv as _format_loss_rate  # noqa: E402

# Datasets are discovered from backend/dataset_csv/<slug>.csv + backend/dataset/<slug>/column_info.
REGER_TIME_COLUMN = "__reger_time__"
FLOAT_PRECISION_FILTER_FIXED = "fixed"
FLOAT_PRECISION_FILTER_NON_FIXED = "non_fixed"
FLOAT_PRECISION_FIXED_LABEL = "Fixed precision"
FLOAT_PRECISION_NON_FIXED_LABEL = "Non-fixed precision"
FIXED_PRECISION_DOMINANT_FRACTION_THRESHOLD = 0.85
REGER_FLOAT_PRECISION_REPORT_SUFFIX = "_float_cpp_reger_precision_result.csv"
REGER_FLOAT_PRECISION_REPORT_FIELDS = (
    "precision_scope",
    "column_index",
    "algorithm",
    "originalSize",
    "compressedSize",
    "compressionTime",
    "decompressionTime",
    "isLossy",
    "lossRate",
)


def _cpp_exe() -> Path:
    exe = benchmark_cpp_executable_path()
    if exe is not None:
        return exe
    try:
        return require_benchmark_cpp_executable()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


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


def _bench_env(algorithms: list[str] | None = None) -> dict:
    env = _buff_env(_serf_env(dict(os.environ)))
    env["COMPRESSION_BENCHMARK_REPO_ROOT"] = str(ROOT.resolve())
    env.setdefault(
        "COMPRESSION_BENCHMARK_FPCOMPRESS_BIN_ROOT",
        str((ROOT / "external" / "runtime" / "fpcompress").resolve()),
    )
    env.setdefault("COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH", "0")
    env.setdefault("COMPRESSION_BENCHMARK_BLOSC_BENCH", "0")
    env.setdefault("COMPRESSION_BENCHMARK_BLOSC2_BENCH", "0")
    env.setdefault("COMPRESSION_BENCHMARK_LZBENCH_BENCH", "0")
    # benchmark_cpp invokes cpp_java_companion.py via std::system; pin to this interpreter when unset.
    env.setdefault("COMPRESSION_BENCHMARK_PYTHON", sys.executable)
    if algorithms:
        algo_csv = ",".join(a.strip() for a in algorithms if a and str(a).strip())
        if algo_csv:
            env["COMPRESSION_BENCHMARK_RUN_ALGORITHMS"] = algo_csv
            env = apply_partial_algorithm_external_benches(env, algorithms)
    return env


def run_cpp_benchmark(
    input_csv: Path,
    data_type: str,
    output_csv: Path,
    *,
    column_info: Path,
    column_index: int | None = None,
    algorithms: list[str] | None = None,
) -> None:
    cmd = [
        str(_cpp_exe().resolve()),
        "-i",
        str(input_csv.resolve()),
        "--data-type",
        data_type,
        "--output-report",
        str(output_csv.resolve()),
        "--column-info",
        str(column_info.resolve()),
    ]
    if column_index is not None:
        cmd.extend(["--column-index", str(int(column_index))])
    if algorithms:
        algo_csv = ",".join(a.strip() for a in algorithms if a and str(a).strip())
        if algo_csv:
            cmd.extend(["--algorithms", algo_csv])
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT.resolve()),
        env=_bench_env(algorithms),
    )

def _normalize_logical_type(raw: object) -> str:
    return str(raw or "").strip().lower()


def _iter_column_info_rows(column_info_path: Path) -> list[dict]:
    with column_info_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_column_precision_by_index(
    column_info_rows: list[dict],
    bundle_dir: Path,
) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in column_info_rows:
        if _normalize_logical_type(row.get("logical_type")) not in {"float", "double"}:
            continue
        try:
            column_index = int(row.get("column_index"))
        except (TypeError, ValueError):
            continue
        decimal_path = bundle_dir / "float_decimal_places" / f"column_{column_index}.csv"
        best_fraction = -1.0
        if decimal_path.is_file():
            try:
                with decimal_path.open("r", encoding="utf-8", newline="") as handle:
                    for decimal_row in csv.DictReader(handle):
                        try:
                            best_fraction = max(
                                best_fraction,
                                float(decimal_row.get("fraction", 0) or 0),
                            )
                        except (TypeError, ValueError):
                            continue
            except OSError:
                best_fraction = -1.0
        out[column_index] = (
            FLOAT_PRECISION_FIXED_LABEL
            if best_fraction >= FIXED_PRECISION_DOMINANT_FRACTION_THRESHOLD
            else FLOAT_PRECISION_NON_FIXED_LABEL
        )
    return out


def _columns_for_dtype(
    column_info_rows: list[dict],
    data_type: str,
) -> list[tuple[int, str]]:
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
    tmp_dir: Path | None = None,
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
                    dir=tmp_dir,
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
        tmp_dir = cache_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        generated, row_count = _materialize_column_csvs(
            src, slug, data_type, missing, tmp_dir=tmp_dir
        )
        for col_idx, generated_path in generated.items():
            cached = cache_dir / f"column_{int(col_idx)}.csv"
            os.replace(generated_path, cached)
            paths[int(col_idx)] = cached
    return paths, row_count, True


def _materialize_multi_column_csv(
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str]],
) -> tuple[Path, int, list[str], list[tuple[int, str]]]:
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
            value_columns = [
                (int(col_idx), col_name, source_idx)
                for col_idx, col_name in columns
                for source_idx in [_source_index_for_column(source_header, int(col_idx), col_name)]
                if source_idx is not None
            ]
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
    return path, row_count, headers, [(idx, name) for idx, name, _source_idx in value_columns]


def _run_reger_multi_scope(
    *,
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str]],
    column_info_rows: list[dict],
    column_info_path: Path,
) -> dict | None:
    multi_csv_path, _row_count, headers, multi_columns = (
        _materialize_multi_column_csv(src, slug, data_type, columns)
    )
    if not multi_columns:
        multi_csv_path.unlink(missing_ok=True)
        return None
    multi_info_path = _write_temp_column_info(column_info_rows, multi_columns, headers)
    multi_report_path = multi_csv_path.with_name(f"{multi_csv_path.stem}_report.csv")
    try:
        run_cpp_benchmark(
            multi_csv_path,
            data_type,
            multi_report_path,
            column_info=multi_info_path,
            algorithms=["reger"],
        )
        try:
            bench_df = pd.read_csv(multi_report_path)
        except Exception as exc:
            print(f"  -> Skip REGER multi report parse: {exc}")
            return None
        for _, row in bench_df.iterrows():
            if str(row.get("algorithm", "")).strip() != "reger":
                continue
            return {
                "column_index": -1,
                "algorithm": "reger",
                "originalSize": int(float(row.get("originalSize", 0) or 0)),
                "compressedSize": int(float(row.get("compressedSize", 0) or 0)),
                "compressionTime": int(float(row.get("compressionTime", 0) or 0)),
                "decompressionTime": int(float(row.get("decompressionTime", 0) or 0)),
                "isLossy": int(float(row.get("isLossy", 0) or 0)),
                "lossRate": float(row.get("lossRate", 0) or 0),
            }
    finally:
        multi_csv_path.unlink(missing_ok=True)
        multi_info_path.unlink(missing_ok=True)
        multi_report_path.unlink(missing_ok=True)
    return None


def _write_reger_float_precision_rows(bundle_dir: Path, rows: list[dict]) -> None:
    report_path = bundle_dir / f"{bundle_dir.name}{REGER_FLOAT_PRECISION_REPORT_SUFFIX}"
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=REGER_FLOAT_PRECISION_REPORT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in REGER_FLOAT_PRECISION_REPORT_FIELDS}
            out["lossRate"] = _format_loss_rate(float(out.get("lossRate", 0) or 0))
            writer.writerow(out)
    tmp_path.replace(report_path)


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
    parser.add_argument(
        "--column-indexes",
        metavar="INDEXES",
        help="Comma-separated physical column indexes to benchmark.",
    )
    parser.add_argument(
        "--prepare-column-cache",
        action="store_true",
        help="Materialize selected typed columns into COMPRESSION_BENCHMARK_COLUMN_CACHE_DIR and exit.",
    )
    args = parser.parse_args()
    selected_column_indexes = {
        int(part.strip())
        for part in (args.column_indexes or "").split(",")
        if part.strip()
    }
    selected_algorithms = parse_algorithms(args.algorithms)
    if selected_algorithms:
        configure_allowlist(selected_algorithms)
        print(f"Allowlist algorithms ({len(selected_algorithms)}): {', '.join(selected_algorithms)}")

    exe = _cpp_exe()
    print(f"Using benchmark_cpp: {exe}")

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
    for slug, src in pairs:
        source_header = _read_source_header(src)
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
            report_path = out_root / f"{slug}_{dt}_cpp_compression_result.csv"
            columns = _columns_for_dtype(column_info_rows, dt)
            if selected_column_indexes:
                columns = [
                    item for item in columns if int(item[0]) in selected_column_indexes
                ]
            if not columns:
                print(f"Skip {slug} {dt}: no columns from column_info")
                write_report_rows(report_path, [], format_loss_rate=_format_loss_rate)
                continue
            valid_columns = [
                (idx, name)
                for idx, name in sorted(columns, key=lambda item: item[0])
                if _source_index_for_column(source_header, int(idx), name) is not None
            ]
            skipped_indices = sorted({idx for idx, _name in columns} - {idx for idx, _name in valid_columns})
            for idx in skipped_indices:
                name = next((n for i, n in columns if i == idx), "")
                print(
                    f"  -> Skip {slug} {dt} column_index={idx}: "
                    f"not found in source CSV header by index/name ({name!r})"
                )
            if not valid_columns:
                print(f"Skip {slug} {dt}: no in-range source columns")
                continue
            if args.prepare_column_cache:
                cached_paths, row_count, persistent = _column_csvs_for_run(
                    src, slug, dt, valid_columns
                )
                if not persistent:
                    for cached_path in cached_paths.values():
                        cached_path.unlink(missing_ok=True)
                    raise SystemExit(
                        "--prepare-column-cache requires COMPRESSION_BENCHMARK_COLUMN_CACHE_DIR"
                    )
                print(
                    f"Prepared column cache for {slug} {dt}: "
                    f"{len(cached_paths)} columns, {row_count} rows"
                )
                continue
            column_csv_paths: dict[int, Path] = {}
            persistent_column_cache = False
            if not _only_reger(selected_algorithms):
                column_csv_paths, row_count, persistent_column_cache = _column_csvs_for_run(
                    src, slug, dt, valid_columns
                )
            else:
                row_count = "streaming"
            print(
                f"Regenerate {report_path.name} "
                f"({len(columns)} columns, {row_count} rows, patch={os.environ.get('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')})"
            )
            existing_rows = (
                load_report_rows(report_path)
                if selected_algorithms and report_path.is_file()
                else []
            )
            report_rows: list[dict] = []
            try:
                if not _only_reger(selected_algorithms):
                    for col_idx, _col_name in columns:
                        tmp_csv_path = column_csv_paths.get(int(col_idx))
                        if tmp_csv_path is None:
                            continue
                        tmp_report_path = tmp_csv_path.with_name(
                            f"{tmp_csv_path.stem}_report.csv"
                        )
                        tmp_report_path.unlink(missing_ok=True)
                        try:
                            print(
                                f"  -> C++ bench {slug} {dt} column_index={col_idx}"
                            )
                            run_cpp_benchmark(
                                tmp_csv_path,
                                dt,
                                tmp_report_path,
                                column_info=col_info,
                                column_index=col_idx,
                                algorithms=selected_algorithms,
                            )
                            try:
                                bench_df = pd.read_csv(tmp_report_path)
                            except Exception as exc:
                                print(
                                    f"  -> Skip report parse for column_index={col_idx}: {exc}"
                                )
                                continue
                            for _, row in bench_df.iterrows():
                                algo = str(row.get("algorithm", "")).strip()
                                if not algo or algo == "reger":
                                    continue
                                try:
                                    row_col_idx = int(row.get("column_index"))
                                except (TypeError, ValueError):
                                    row_col_idx = int(col_idx)
                                report_rows.append(
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
                        finally:
                            if not persistent_column_cache:
                                tmp_csv_path.unlink(missing_ok=True)
                            tmp_report_path.unlink(missing_ok=True)
                if dt in ("int", "float") and _runs_reger(selected_algorithms):
                    print(
                        f"  -> C++ bench {slug} {dt} REGER all columns "
                        f"({len(valid_columns)} value columns)"
                    )
                    reger_all = _run_reger_multi_scope(
                        src=src,
                        slug=slug,
                        data_type=dt,
                        columns=valid_columns,
                        column_info_rows=column_info_rows,
                        column_info_path=col_info,
                    )
                    if reger_all is not None:
                        report_rows.append(reger_all)

                    if dt == "float":
                        precision_by_index = _float_column_precision_by_index(
                            column_info_rows, out_root
                        )
                        precision_rows: list[dict] = []
                        for scope, label in (
                            (FLOAT_PRECISION_FILTER_FIXED, FLOAT_PRECISION_FIXED_LABEL),
                            (FLOAT_PRECISION_FILTER_NON_FIXED, FLOAT_PRECISION_NON_FIXED_LABEL),
                        ):
                            scope_columns = [
                                item
                                for item in valid_columns
                                if precision_by_index.get(int(item[0])) == label
                            ]
                            if not scope_columns:
                                continue
                            if len(scope_columns) == len(valid_columns) and reger_all is not None:
                                print(
                                    f"  -> Reuse {slug} float REGER all-columns result for {scope}"
                                )
                                scope_row = dict(reger_all)
                            else:
                                print(
                                    f"  -> C++ bench {slug} float REGER {scope} "
                                    f"({len(scope_columns)} value columns)"
                                )
                                scope_row = _run_reger_multi_scope(
                                    src=src,
                                    slug=slug,
                                    data_type=dt,
                                    columns=scope_columns,
                                    column_info_rows=column_info_rows,
                                    column_info_path=col_info,
                                )
                            if scope_row is not None:
                                precision_rows.append(
                                    {"precision_scope": scope, **scope_row}
                                )
                        _write_reger_float_precision_rows(out_root, precision_rows)
            finally:
                if not persistent_column_cache:
                    for tmp_csv_path in column_csv_paths.values():
                        tmp_csv_path.unlink(missing_ok=True)
            if selected_algorithms and existing_rows:
                report_rows = merge_partial_algorithm_rows(
                    existing_rows,
                    report_rows,
                    set(selected_algorithms),
                )
            report_rows.sort(key=lambda r: (int(r["column_index"]), str(r["algorithm"])))
            write_report_rows(report_path, report_rows, format_loss_rate=_format_loss_rate)
            print(
                f"  -> wrote {report_path.name}: {len(report_rows)} rows "
                f"(column_index + algorithm metrics)"
            )
    print("Done.")


if __name__ == "__main__":
    main()
