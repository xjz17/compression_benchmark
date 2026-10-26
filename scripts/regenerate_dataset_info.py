#!/usr/bin/env python3
"""
Build catalog sidecars under ``backend/dataset/<key>/`` from master CSVs in ``backend/dataset_csv/``.

On every run, (re)generates under ``backend/dataset/<key>/`` (CSV is **not** copied here):

- ``{key}_column_info.csv`` — column_index, column_name, logical_type, max_point_number, column_csv_size_bytes
- ``{key}_dataset_info.csv`` — data_point_count (rows excluding header), csv_size_bytes
- ``{key}_numeric_statistic.csv`` — mean, standard_deviation, spread, entropy (byte), value_entropy, repeat, sortedness
- ``column_distribution_data/column_<idx>_{scaling|leading_zeros|trailing_zeros}.csv`` — bit-width / leading-trailing zero histogram rows for the Dataset feature matrix UI (float64 XOR between consecutive values; float uses scaling for bit width).
- ``float_decimal_places/column_<idx>.csv`` — per float column: decimal_places, count, fraction

Removes legacy unprefixed catalog names.

For ``{key}.tsfile`` and ``{key}_catalog_tsfile_info.csv``, run
``scripts/regenerate_catalog_tsfile.py`` separately (``.tsfile`` under ``backend/dataset_tsfile/<key>/``,
summary under ``backend/dataset/<key>/``).

From repository root::

    python scripts/regenerate_dataset_info.py
    python scripts/regenerate_dataset_info.py --dataset <key>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_BACKEND = _REPO / "backend"
_DATASET = _BACKEND / "dataset"
_MPL_CFG = _BACKEND / ".matplotlib-cache"

_mpl_cfg = _MPL_CFG
_mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg.resolve()))


def _resolve_source_csvs(dataset_filter: str | None) -> list[tuple[str, Path]]:
    """(key, source_csv) pairs from external or in-repo master CSV roots."""
    from dataset_csv_source import (
        describe_dataset_csv_lookup,
        iter_master_csv_paths,
        resolve_benchmark_dataset_csv,
    )

    if dataset_filter:
        csv_path = resolve_benchmark_dataset_csv(_BACKEND, dataset_filter, f"{dataset_filter}.csv")
        if csv_path is not None:
            return [(dataset_filter, csv_path)]
        raise FileNotFoundError(
            f"No source CSV for {dataset_filter!r} under {describe_dataset_csv_lookup(_BACKEND)}"
        )

    pairs = iter_master_csv_paths(_BACKEND)
    if not pairs:
        raise FileNotFoundError(
            f"No master CSV files under {describe_dataset_csv_lookup(_BACKEND)}"
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        metavar="KEY",
        help="Only refresh catalog info for this dataset key.",
    )
    args = parser.parse_args()

    if str(_BACKEND.resolve()) not in sys.path:
        sys.path.insert(0, str(_BACKEND.resolve()))
    if str(_SCRIPTS.resolve()) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS.resolve()))

    try:
        sources = _resolve_source_csvs(args.dataset)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not sources:
        print("No CSV files to process.", file=sys.stderr)
        return 1

    from myapp.dataset_catalog_assets import build_catalog_bundle

    print("Mode: refresh catalog info from dataset_csv (stats always regenerated)")

    _DATASET.mkdir(parents=True, exist_ok=True)
    built = 0
    failed = 0
    for key, csv_path in sources:
        bundle_dir = _DATASET / key
        print(f"Building {key} from {csv_path} …")

        ok = build_catalog_bundle(
            key,
            csv_path,
            bundle_dir,
            generate_dtype_assets=False,
        )
        if ok:
            built += 1
            print(f"OK {key} -> {bundle_dir}")
        else:
            failed += 1

    print(f"Done. built={built} failed={failed} (of {len(sources)} requested)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
