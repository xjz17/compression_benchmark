#!/usr/bin/env python3
"""
Build operator-transformed catalog statistics under ``backend/dataset/<key>/operators/<operator>/``.

Requires an existing catalog bundle (``{key}_column_info.csv``) and master CSV in
``backend/dataset_csv/`` (or external dataset CSV root).

For each supported operator, applies the transform in memory to **numeric columns only**
(from ``{key}_column_info.csv``) and writes:

- ``operators/<operator>/{key}_numeric_statistic.csv`` — same numeric metrics as original catalog (entropy = byte entropy; value_entropy when regenerated)
- ``operators/<operator>/column_distribution_data/column_<idx>_*.csv`` — same distribution histograms on transformed data

No transformed CSV is written on disk.

Supported operators are loaded from ``myapp.dataset_operators.SUPPORTED_OPERATORS``.
They cover the full Methods-page operator catalog, including column transforms and
byte-codec operators from ``benchmark/reference/methods/{encoding,byte_codec}_operators.json``.

Legacy catalog operator ``delta`` / ``xor`` keep the existing catalog transform
semantics (e.g. ``delta`` uses ``np.diff`` without retaining the first value).

From repository root::

    python scripts/regenerate_dataset_operator_stats.py
    python scripts/regenerate_dataset_operator_stats.py --dataset AI4I-Maintenance
    python scripts/regenerate_dataset_operator_stats.py --dataset AI4I-Maintenance --operator delta
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


def _resolve_dataset_keys(dataset_filter: str | None) -> list[str]:
    from dataset_csv_source import describe_dataset_csv_lookup, iter_master_csv_paths

    if dataset_filter:
        from dataset_csv_source import resolve_benchmark_dataset_csv

        csv_path = resolve_benchmark_dataset_csv(_BACKEND, dataset_filter, f"{dataset_filter}.csv")
        if csv_path is None:
            raise FileNotFoundError(
                f"No source CSV for {dataset_filter!r} under {describe_dataset_csv_lookup(_BACKEND)}"
            )
        return [dataset_filter]

    keys: list[str] = []
    for key, _ in iter_master_csv_paths(_BACKEND):
        bundle = _DATASET / key
        col_info = bundle / f"{key}_column_info.csv"
        if col_info.is_file():
            keys.append(key)
    if not keys:
        raise FileNotFoundError(
            "No datasets with both master CSV and column_info sidecar were found."
        )
    return sorted(keys)


def main() -> int:
    if str(_BACKEND.resolve()) not in sys.path:
        sys.path.insert(0, str(_BACKEND.resolve()))
    if str(_SCRIPTS.resolve()) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS.resolve()))

    from dataset_csv_source import resolve_benchmark_dataset_csv
    from myapp.dataset_catalog_assets import column_info_filename
    from myapp.dataset_operators import SUPPORTED_OPERATORS, write_operator_catalog_assets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        metavar="KEY",
        help="Only process this dataset key (must have column_info sidecar).",
    )
    parser.add_argument(
        "--operator",
        choices=tuple(SUPPORTED_OPERATORS) + ("all",),
        default="all",
        help="Operator to apply (default: all supported operators).",
    )
    args = parser.parse_args()

    try:
        keys = _resolve_dataset_keys(args.dataset)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Mode: refresh operator catalog stats (always regenerated)")

    operators = list(SUPPORTED_OPERATORS) if args.operator == "all" else [args.operator]
    built = 0
    failed = 0

    for key in keys:
        bundle_dir = _DATASET / key
        col_info = bundle_dir / column_info_filename(key)
        if not col_info.is_file():
            print(f"Skip {key}: missing {col_info.name}", file=sys.stderr)
            failed += 1
            continue
        master_csv = resolve_benchmark_dataset_csv(_BACKEND, key, f"{key}.csv")
        if master_csv is None:
            print(f"Skip {key}: master CSV not found", file=sys.stderr)
            failed += 1
            continue

        for operator_id in operators:
            print(f"Building {key} operator={operator_id} …")
            try:
                out_dir = write_operator_catalog_assets(
                    key,
                    str(master_csv),
                    bundle_dir,
                    operator_id,
                )
                built += 1
                print(f"OK {key}/{operator_id} -> {out_dir}")
            except Exception as exc:
                failed += 1
                print(f"FAILED {key}/{operator_id}: {exc}", file=sys.stderr)

    print(f"Done. built={built} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
