#!/usr/bin/env python3
"""
Append ``ext_zfp_zfpy`` / ``ext_zfp_tol_*`` rows to catalog ``*_float_python_compression_result.csv``.

Uses pip ``zfpy`` with the same patch size as staticdata regen (``WEB_COMPRESSION_CSV_PATCH_ROWS``,
default 10000). Replaces any existing ZFP rows for the same ``column_index``.

    python scripts/append_zfp_float_staticdata.py
    python scripts/append_zfp_float_staticdata.py --dataset Metro-Traffic
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_csv_source import iter_catalog_benchmark_datasets  # noqa: E402

sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)
os.environ.setdefault("WEB_COMPRESSION_REPO_ROOT", str(ROOT.resolve()))
os.environ.setdefault("WEB_COMPRESSION_CSV_PATCH_ROWS", "10000")

from myapp.benchmark_loss import (  # noqa: E402
    format_loss_rate_csv,
    is_lossy_flag,
    loss_rate,
)
from myapp.external_compression_python import (  # noqa: E402
    bench_zfp_lossy_columns,
    bench_zfp_zfpy_columns,
)
from myapp.experiment_data_policy import parse_experiment_float_cell  # noqa: E402

ZFP_ROW_NAMES = (
    "ext_zfp_zfpy",
    "ext_zfp_tol_1e-1",
    "ext_zfp_tol_1e-2",
    "ext_zfp_tol_1e-3",
)
ZFP_ALGOS: tuple[tuple[str, object], ...] = (
    ("ext_zfp_zfpy", bench_zfp_zfpy_columns),
    (1e-1, "ext_zfp_tol_1e-1"),
    (1e-2, "ext_zfp_tol_1e-2"),
    (1e-3, "ext_zfp_tol_1e-3"),
)

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


def _patch_limit() -> int:
    raw = os.environ.get("WEB_COMPRESSION_CSV_PATCH_ROWS", "10000").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 10000


def _float_columns_from_info(column_info_path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with column_info_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            logical = str(row.get("logical_type", "")).strip().lower()
            if logical not in ("float", "double"):
                continue
            try:
                idx = int(row.get("column_index"))
            except (TypeError, ValueError):
                continue
            out.append((idx, str(row.get("column_name", "")).strip()))
    return sorted(out, key=lambda x: x[0])


def _load_column_values(master_csv: Path, col_idx: int) -> list[float]:
    col: list[float] = []
    with master_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or col_idx < 0 or col_idx >= len(header):
            return col
        for row in reader:
            if len(row) <= col_idx:
                continue
            cell = str(row[col_idx]).strip()
            if not cell:
                continue
            value = parse_experiment_float_cell(cell)
            if value is not None:
                col.append(value)
    return col


def _bench_column_patch(
    col: list[float],
    *,
    patch: int,
    bench_fn,
) -> tuple[int, int, int, int, float, float]:
    o = c = te = td = 0
    loss_err = loss_ref = 0.0
    for i in range(0, len(col), patch):
        po, pc, pte, ptd, le, lr = bench_fn([col[i : i + patch]])
        if po <= 0:
            continue
        o += po
        c += pc
        te += pte
        td += ptd
        loss_err += float(le)
        loss_ref += float(lr)
    return o, c, te, td, loss_err, loss_ref


def _zfp_rows_for_column(col_idx: int, col: list[float], patch: int) -> list[dict]:
    rows: list[dict] = []
    for entry in ZFP_ALGOS:
        if entry[0] == "ext_zfp_zfpy":
            name = entry[0]
            bench_fn = entry[1]
        else:
            tol, name = entry
            bench_fn = lambda cols, t=tol: bench_zfp_lossy_columns(cols, t, wire_data_type="float")
        o, c, te, td, le, lr = _bench_column_patch(col, patch=patch, bench_fn=bench_fn)
        if o <= 0:
            continue
        rows.append(
            {
                "column_index": int(col_idx),
                "algorithm": name,
                "originalSize": int(o),
                "compressedSize": int(c),
                "compressionTime": int(te),
                "decompressionTime": int(td),
                "isLossy": int(is_lossy_flag(name, le, lr)),
                "lossRate": format_loss_rate_csv(loss_rate(le, lr)),
            }
        )
    return rows


def _merge_report(report_path: Path, new_rows: list[dict]) -> None:
    if report_path.is_file():
        df = pd.read_csv(report_path)
    else:
        df = pd.DataFrame(columns=list(REPORT_FIELDS))
    zfp_names = set(ZFP_ROW_NAMES)
    col_idxs = {int(r["column_index"]) for r in new_rows}
    mask = df["algorithm"].astype(str).isin(zfp_names) & df["column_index"].astype(int).isin(col_idxs)
    df = df.loc[~mask]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.sort_values(by=["column_index", "algorithm"]).reset_index(drop=True)
    df.to_csv(report_path, index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="SLUG", help="Only update this catalog key.")
    args = parser.parse_args()

    patch = _patch_limit()
    pairs = iter_catalog_benchmark_datasets(BACKEND)
    if args.dataset:
        want = args.dataset.strip()
        pairs = [(s, p) for s, p in pairs if s == want]
        if not pairs:
            raise SystemExit(f"No catalog dataset matches --dataset {want!r}")

    for slug, master in pairs:
        out_root = BACKEND / "dataset" / slug
        col_info = out_root / f"{slug}_column_info.csv"
        report = out_root / f"{slug}_float_python_compression_result.csv"
        if not col_info.is_file():
            print(f"Skip {slug}: missing {col_info.name}")
            continue
        if not report.is_file():
            print(f"Skip {slug}: missing {report.name} (run regenerate_staticdata_python.py first)")
            continue
        float_cols = _float_columns_from_info(col_info)
        if not float_cols:
            print(f"Skip {slug}: no float columns")
            continue
        new_rows: list[dict] = []
        for col_idx, col_name in float_cols:
            col = _load_column_values(master, col_idx)
            if not col:
                print(f"  Skip {slug} col {col_idx} ({col_name}): empty")
                continue
            print(f"  ZFP bench {slug} column_index={col_idx} ({col_name}), rows={len(col)}, patch={patch}")
            new_rows.extend(_zfp_rows_for_column(col_idx, col, patch))
        if not new_rows:
            print(f"Skip {slug}: no ZFP metrics produced (is zfpy installed?)")
            continue
        _merge_report(report, new_rows)
        print(f"Updated {report.relative_to(ROOT)} (+{len(new_rows)} ZFP rows)")


if __name__ == "__main__":
    main()
