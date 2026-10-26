"""Shared helpers for regenerate_staticdata_{java,cpp,python}.py partial algorithm refresh."""

from __future__ import annotations

import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from myapp.lzbench_codec_catalog import canonical_report_algorithm_key  # noqa: E402
from myapp.report_filters import sanitize_compression_report_dataframe  # noqa: E402

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


def parse_algorithms(raw: str | None) -> list[str] | None:
    if raw is None or not str(raw).strip():
        return None
    out: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        name = part.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out or None


def load_report_rows(report_path: Path) -> list[dict]:
    if not report_path.is_file():
        return []
    with report_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_algorithm_key(row: dict) -> str:
    raw = str(row.get("algorithm", "")).strip()
    return canonical_report_algorithm_key(raw) or raw


def _row_physical_key(row: dict) -> tuple[str, str]:
    raw_index = str(row.get("column_index", "")).strip()
    try:
        index = str(int(float(raw_index)))
    except ValueError:
        index = raw_index
    return index, _row_algorithm_key(row)


def _sanitize_report_row_dicts(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    df = sanitize_compression_report_dataframe(pd.DataFrame(rows))
    if df.empty:
        return []
    return [
        {field: row.get(field, "") for field in REPORT_FIELDS}
        for row in df.to_dict(orient="records")
    ]


def merge_report_rows_by_physical_key(
    existing_rows: list[dict],
    new_rows: list[dict],
) -> list[dict]:
    """Upsert rows by (physical column, canonical algorithm)."""
    merged: dict[tuple[str, str], dict] = {}
    for row in [*existing_rows, *new_rows]:
        merged[_row_physical_key(row)] = row
    return _sanitize_report_row_dicts(list(merged.values()))


def merge_partial_algorithm_rows(
    existing_rows: list[dict],
    new_rows: list[dict],
    algorithms: set[str],
) -> list[dict]:
    if os.environ.get("COMPRESSION_BENCHMARK_PARTIAL_MERGE_EXACT_KEYS", "").strip() == "1":
        refresh_keys = {_row_physical_key(row) for row in new_rows}
        kept = [row for row in existing_rows if _row_physical_key(row) not in refresh_keys]
        return _sanitize_report_row_dicts(kept + list(new_rows))

    refresh_keys: set[str] = set()
    for name in algorithms:
        key = str(name or "").strip()
        if not key:
            continue
        refresh_keys.add(key)
        canonical = canonical_report_algorithm_key(key)
        if canonical:
            refresh_keys.add(canonical)

    kept = [row for row in existing_rows if _row_algorithm_key(row) not in refresh_keys]
    return _sanitize_report_row_dicts(kept + list(new_rows))


def write_report_rows(report_path: Path, rows: list[dict], *, format_loss_rate) -> None:
    rows = _sanitize_report_row_dicts(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if "lossRate" in out:
                val = out["lossRate"]
                if isinstance(val, str):
                    text = val.strip()
                    out["lossRate"] = text if text else "0"
                else:
                    out["lossRate"] = format_loss_rate(float(val or 0))
            writer.writerow(out)
    tmp_path.replace(report_path)


@contextmanager
def _report_write_lock(report_path: Path):
    import fcntl

    lock_path = report_path.with_suffix(report_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def upsert_report_rows(
    report_path: Path,
    new_rows: list[dict],
    *,
    format_loss_rate,
    full_rewrite: bool = False,
) -> list[dict]:
    """Atomically retain prior rows and upsert this invocation's results."""
    with _report_write_lock(report_path):
        rows = list(new_rows)
        if not full_rewrite:
            rows = merge_report_rows_by_physical_key(load_report_rows(report_path), rows)
        rows.sort(
            key=lambda row: (
                int(float(str(row.get("column_index", "0") or "0"))),
                _row_algorithm_key(row),
            )
        )
        write_report_rows(report_path, rows, format_loss_rate=format_loss_rate)
        return rows


def apply_partial_algorithm_external_benches(
    env: dict[str, str], algorithms: list[str] | None
) -> dict[str, str]:
    if not algorithms:
        return env
    from myapp.lzbench_codec_catalog import is_lzbench_catalog_algorithm

    names = [a.strip() for a in algorithms if a and str(a).strip()]
    env["COMPRESSION_BENCHMARK_LZBENCH_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_SZ_BENCH"] = "1" if any(a.startswith("sz_") for a in names) else "0"
    env["COMPRESSION_BENCHMARK_SZ3_BENCH"] = "1" if any(a.startswith("sz3_") for a in names) else "0"
    env["COMPRESSION_BENCHMARK_FPZIP_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_BLOSC_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_BLOSC2_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_ZFP_BENCH"] = "1" if any(a.startswith("ext_zfp_") for a in names) else "0"
    return env


def configure_allowlist(algorithms: list[str] | None) -> None:
    if not algorithms:
        return
    os.environ["COMPRESSION_BENCHMARK_RUN_ALGORITHMS"] = ",".join(algorithms)
    apply_partial_algorithm_external_benches(os.environ, algorithms)
