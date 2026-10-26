#!/usr/bin/env python3
"""Export full dataset catalog as comma-separated CSV.

Only datasets in ``DATASET_KEYS`` are exported and analyzed; anything else under
``backend/dataset/`` is ignored.
Column 原始大小（B） and tsfile大小（B） are written as byte counts (integer strings).
Per-dataset 时间戳（等间隔/非等间隔/无） is inferred from sampled timestamp spacing.

Output: ``backend/temp/dataset_catalog_export.csv``

From repository root::

    python scripts/export_dataset_catalog_csv.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tqdm is required. Install in backend venv: .venv/Scripts/pip install tqdm"
    ) from exc

_REPO = Path(__file__).resolve().parent.parent
_BACKEND = _REPO / "backend"
DATASET_ROOT = _BACKEND / "dataset"
METADATA_JSON = _REPO / "benchmark" / "reference" / "datasets" / "dataset_metadata.json"
OUTPUT_CSV = _BACKEND / "temp" / "dataset_catalog_export.csv"
_DEFAULT_EXTERNAL_CSV_ROOT = Path(r"E:/xjz/compression_benchmark/backend_dataset_csv")

# Fixed catalog order; CSV rows are written in this sequence.
DATASET_KEYS: tuple[str, ...] = (
    "1year",
    "acs_wht_f32",
    "adult",
    "agaricus-lepiota.data",
    "AI4I-Maintenance",
    "Air-pressure-1min",
    "Air-pressure-30min",
    "Air-Quality",
    "Air-sensor",
    "Arade",
    "astro_mhd_f64",
    "astro_pt_f64",
    "bank",
    "Bar-Crawl",
    "Bimbo",
    "Bird-migration",
    "Bitcoin-price",
    "Blockchain-tr",
    "books",
    "City-temp",
    "CityMaxCapita",
    "Climate_Weather",
    "CMSprovider",
    "Comments",
    "CommonGovernment",
    "Corporations",
    "Cpu-usage",
    "credit_scoring_sample",
    "credit_scoring_test",
    "credit_scoring_train",
    "Dewpoint-temp-1min",
    "Dewpoint-temp-30min",
    "Disk-usage",
    "DSL-StrongPasswordData",
    "ehresp_2014",
    "Eixo",
    "Euro2016",
    "fb",
    "flight_delays_test",
    "flight_delays_train",
    "Food",
    "Food-prices",
    "g24_78_usb2_f32",
    "Generico",
    "h3d_temp_f32",
    "HashTags",
    "Hatred",
    "hdr_night_f32",
    "hdr_palermo_f32",
    "hmeq",
    "HOUSE",
    "hst_wfc3_ir_f32",
    "hst_wfc3_uvis_f32",
    "IGlocations1",
    "IGlocations2",
    "IR-bio-temp-1minute",
    "IR-bio-temp-30minute",
    "jane_street_f64",
    "jw_mirimage_f32",
    "linear",
    "Medicare1",
    "Medicare2",
    "Medicare3",
    "medium_posts",
    "MedPayment1",
    "MedPayment2",
    "Mem-usage",
    "meteoblue",
    "Metro-Traffic",
    "miranda3d_f32",
    "MLB",
    "mlbootcamp5_train",
    "monash_australian_electricity",
    "Motos",
    "MulheresMil",
    "normal",
    "num_brain_f64",
    "num_control_f64",
    "NYC",
    "NYC_Yellow_Taxi_Trip",
    "PanCreactomy1",
    "PanCreactomy2",
    "phone_gyro_f64",
    "Physicians",
    "PM10-dust-30minutes",
    "PM10-dust-60minutes",
    "poi",
    "prolongation_service_train",
    "Provider",
    "Rail_Insurance_Claims",
    "rating",
    "RealEstate1",
    "RealEstate2",
    "realtor",
    "Redfin1",
    "Redfin2",
    "Redfin3",
    "Redfin4",
    "Rentabilidad",
    "Romance",
    "rsim_f32",
    "SalariesFrance",
    "scores",
    "solar_wind",
    "spain_gas_price_f64",
    "spam",
    "spitzer_irac_f32",
    "spooky_authors",
    "spooky_writer_sample_submission",
    "spooky_writer_test",
    "spooky_writer_train",
    "SSD-bench",
    "Stocks-DE",
    "Stocks-Germany",
    "Stocks-UK",
    "Stocks-USA",
    "TableroSistemaPenal",
    "Taxpayer",
    "Telco",
    "tpcds_catalog_f32",
    "tpcds_store_f32",
    "tpcds_web_f32",
    "tpch_lineitem_f32",
    "tpch_order_f64",
    "tpcxbb_store_f64",
    "tpcxbb_web_f64",
    "TrainsUK1",
    "TrainsUK2",
    "ts_gas_f32",
    "turbulence_f32",
    "Uberlandia",
    "USCensus",
    "vehicle-charge",
    "vgsales",
    "video_games_sales",
    "wave_f32",
    "weights_heights",
    "wesad_chest_f64",
    "wiki_ts",
    "Wind-speed",
    "winequality-white",
    "Wins",
    "World-City",
    "YaleLanguages",
)
_CATALOG_DATASET_KEYS = frozenset(DATASET_KEYS)


def is_listed_dataset(key: str) -> bool:
    return key in _CATALOG_DATASET_KEYS

HEADERS = [
    "数据集名称",
    "数据集来源",
    "服务器中路径",
    "原始数据格式",
    "时间戳（等间隔/非等间隔/无）",
    "原始大小（B）",
    "文件数",
    "列数",
    "行数",
    "总点数",
    "tsfile大小（B）",
]

MAX_TIMESTAMP_SAMPLES = 2000
TIME_COLUMN_RE = re.compile(
    r"^(time|timestamp|date|datetime|published|ts|时间|日期|时刻|采集时间|记录时间)$",
    re.I,
)
TIME_VALUE_FMTS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%H:%M:%S",
)


def progress_write(msg: str) -> None:
    tqdm.write(msg)


def progress_label(name: str) -> str:
    if len(name) > 48:
        return name[:45] + "..."
    return name


def resolve_master_csv(key: str) -> Path | None:
    roots: list[Path] = []
    env_root = os.environ.get("COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT", "").strip()
    if env_root:
        ext = Path(os.path.expanduser(env_root))
        if ext.is_dir():
            roots.append(ext)
    elif _DEFAULT_EXTERNAL_CSV_ROOT.is_dir():
        roots.append(_DEFAULT_EXTERNAL_CSV_ROOT)
    builtin = _BACKEND / "dataset_csv"
    if builtin.is_dir() and builtin not in roots:
        roots.append(builtin)
    for root in roots:
        path = root / f"{key}.csv"
        if path.is_file():
            return path
    return None


def read_dataset_info(key: str) -> tuple[int, int]:
    """Return (row_count, csv_size_bytes). Missing file -> (0, 0)."""
    path = DATASET_ROOT / key / f"{key}_dataset_info.csv"
    if not path.is_file():
        return 0, 0
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        return 0, 0
    rows = int(float(row.get("data_point_count", 0) or 0))
    size = int(float(row.get("csv_size_bytes", 0) or 0))
    return rows, size


def read_column_info_rows(key: str) -> list[dict[str, str]]:
    path = DATASET_ROOT / key / f"{key}_column_info.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_column_count(key: str) -> int:
    return len(read_column_info_rows(key))


def read_tsfile_size_bytes(key: str) -> int | None:
    path = DATASET_ROOT / key / f"{key}_catalog_tsfile_info.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row:
        return None
    raw = row.get("file_size_bytes")
    if raw in (None, ""):
        return None
    return int(float(raw))


def load_source_urls_by_key() -> dict[str, str]:
    if not METADATA_JSON.is_file():
        return {}
    meta = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
    return {
        item["key"]: str(item.get("source_url") or "")
        for item in meta.get("datasets", [])
    }


def load_catalog_entries() -> list[tuple[str, str]]:
    """Return only datasets in ``DATASET_KEYS``, preserving list order."""
    source_urls = load_source_urls_by_key()
    return [(key, source_urls.get(key, "")) for key in DATASET_KEYS]


def guess_time_column_from_column_info(column_rows: list[dict[str, str]]) -> int | None:
    for row in column_rows:
        name = str(row.get("column_name", "") or "").strip()
        if not TIME_COLUMN_RE.match(name):
            continue
        try:
            col_idx = int(float(row.get("column_index", -1)))
        except (TypeError, ValueError):
            continue
        if col_idx >= 0:
            return col_idx
    return None


def headers_from_column_info(column_rows: list[dict[str, str]]) -> list[str]:
    indexed: dict[int, str] = {}
    for row in column_rows:
        try:
            col_idx = int(float(row.get("column_index", -1)))
        except (TypeError, ValueError):
            continue
        if col_idx < 0:
            continue
        indexed[col_idx] = str(row.get("column_name", "") or f"col_{col_idx}")
    if not indexed:
        return []
    max_idx = max(indexed)
    return [indexed.get(i, f"col_{i}") for i in range(max_idx + 1)]


def parse_datetime_text(text: str) -> float | None:
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, OSError, OverflowError):
        pass
    for fmt in TIME_VALUE_FMTS:
        try:
            return datetime.strptime(text, fmt).timestamp()
        except (ValueError, OSError, OverflowError):
            continue
    return None


def is_epoch_timestamp(value: float) -> bool:
    abs_value = abs(value)
    if 1_000_000_000 <= abs_value <= 4_102_444_800:
        return True
    if 1_000_000_000_000 <= abs_value <= 4_102_444_800_000:
        return True
    return False


def parse_inferred_time_value(raw: str) -> float | None:
    """Strict parser for columns inferred without an explicit time header."""
    text = str(raw or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    parsed = parse_datetime_text(text)
    if parsed is not None:
        return parsed
    try:
        value = float(text)
    except ValueError:
        return None
    if is_epoch_timestamp(value):
        return value
    return None


def parse_named_time_value(raw: str) -> float | None:
    """Parser for columns whose header explicitly marks them as time."""
    text = str(raw or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    parsed = parse_inferred_time_value(text)
    if parsed is not None:
        return parsed
    try:
        return float(text)
    except ValueError:
        return None


def collect_time_samples(
    rows: list[list[str]],
    time_col: int,
    parser: Callable[[str], float | None],
    *,
    min_parse_ratio: float = 0.0,
) -> list[float]:
    samples: list[float] = []
    checked = 0
    for row in rows[:MAX_TIMESTAMP_SAMPLES]:
        if time_col >= len(row):
            continue
        cell = row[time_col].strip()
        if not cell:
            continue
        checked += 1
        parsed = parser(cell)
        if parsed is not None:
            samples.append(parsed)
    if checked == 0:
        return []
    if min_parse_ratio > 0 and len(samples) / checked < min_parse_ratio:
        return []
    return samples


def values_evenly_spaced(values: list[float], rel_tol: float = 1e-5) -> bool:
    if len(values) < 2:
        return True
    ordered = values
    if not all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1)):
        ordered = sorted(values)
    diffs = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    positive = [d for d in diffs if d > 0]
    if not positive:
        return len(set(diffs)) <= 1
    ref = positive[0]
    for diff in positive[1:]:
        if abs(diff - ref) > max(1e-9, abs(ref) * rel_tol):
            return False
    return True


def detect_even_from_samples(samples: list[float]) -> bool | None:
    if len(samples) < 2:
        return None
    return values_evenly_spaced(samples)


def guess_time_column_index(headers: list[str]) -> int | None:
    for idx, header in enumerate(headers):
        if TIME_COLUMN_RE.match(header.strip()):
            return idx
    return None


def detect_time_column_from_rows(rows: list[list[str]]) -> int | None:
    if not rows:
        return None
    max_cols = max(len(row) for row in rows)
    best_col: int | None = None
    best_score = 0
    for col in range(min(max_cols, 8)):
        samples = collect_time_samples(
            rows,
            col,
            parse_inferred_time_value,
            min_parse_ratio=0.8,
        )
        if len(samples) >= 2 and len(samples) > best_score:
            best_score = len(samples)
            best_col = col
    return best_col


def detect_csv_timestamp_from_rows(
    rows: list[list[str]],
    headers: list[str] | None = None,
) -> bool | None:
    if not rows:
        return None
    time_col: int | None = None
    parser = parse_inferred_time_value
    min_parse_ratio = 0.8
    if headers:
        time_col = guess_time_column_index(headers)
        if time_col is not None:
            parser = parse_named_time_value
            min_parse_ratio = 0.0
    if time_col is None:
        time_col = detect_time_column_from_rows(rows)
        if time_col is None:
            return None
    samples = collect_time_samples(
        rows,
        time_col,
        parser,
        min_parse_ratio=min_parse_ratio,
    )
    return detect_even_from_samples(samples)


def max_sample_columns(headers: list[str] | None) -> int:
    """Match reference script: timestamp checks use at most the first 8 columns."""
    limit = 8
    if headers:
        named = guess_time_column_index(headers)
        if named is not None:
            limit = max(limit, named + 1)
    return limit


def read_csv_sample(
    csv_path: Path,
    max_rows: int,
    headers_hint: list[str] | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Sample CSV rows for timestamp checks without parsing ultra-wide rows."""
    col_limit = max_sample_columns(headers_hint)
    headers: list[str] = []
    rows: list[list[str]] = []
    with csv_path.open(encoding="utf-8", errors="replace") as handle:
        for line_idx, raw_line in enumerate(handle):
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            cells = [cell.strip() for cell in line.split(",", col_limit)[:col_limit]]
            if line_idx == 0:
                headers = cells
                if guess_time_column_index(headers) is None and not any(
                    parse_inferred_time_value(cell) is not None for cell in cells
                ):
                    rows.append(cells)
                continue
            rows.append(cells)
            if len(rows) >= max_rows:
                break
    return headers, rows


def classify_timestamp_regularity(key: str) -> str:
    if not is_listed_dataset(key):
        return "无"
    csv_path = resolve_master_csv(key)
    if csv_path is None:
        return "无"

    column_rows = read_column_info_rows(key)
    named_col = guess_time_column_from_column_info(column_rows)
    meta_headers: list[str] = []
    if named_col is not None:
        meta_headers = headers_from_column_info(column_rows)
    headers, rows = read_csv_sample(
        csv_path,
        MAX_TIMESTAMP_SAMPLES,
        headers_hint=meta_headers or None,
    )
    if not rows:
        return "无"

    if meta_headers:
        headers = meta_headers

    result = detect_csv_timestamp_from_rows(rows, headers)
    if result is None:
        return "无"
    if result:
        return "等间隔"
    return "非等间隔"


def format_bytes_cell(size_bytes: int | float | str | None) -> str:
    if size_bytes is None or size_bytes == "":
        return ""
    return str(int(float(size_bytes)))


def build_rows() -> list[list[str]]:
    entries = load_catalog_entries()
    rows_out: list[list[str]] = []
    with tqdm(
        entries,
        desc="导出 catalog",
        unit="数据集",
        dynamic_ncols=True,
        file=sys.stderr,
    ) as progress_bar:
        for name, source in progress_bar:
            if not is_listed_dataset(name):
                continue
            row_count, csv_size = read_dataset_info(name)
            col_count = read_column_count(name)
            total_points = row_count * col_count if row_count and col_count else 0
            tsfile_size = read_tsfile_size_bytes(name)
            timestamp_kind = classify_timestamp_regularity(name)

            progress_bar.set_postfix_str(
                f"{progress_label(name)} -> {timestamp_kind}",
                refresh=False,
            )

            rows_out.append(
                [
                    name,
                    source,
                    "",
                    "csv",
                    timestamp_kind,
                    format_bytes_cell(csv_size),
                    "1",
                    str(col_count),
                    str(row_count),
                    str(total_points),
                    format_bytes_cell(tsfile_size),
                ]
            )

    return rows_out


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    progress_write(f"Exporting {len(DATASET_KEYS)} datasets...")
    rows = build_rows()
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    progress_write(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    raise SystemExit(main())
