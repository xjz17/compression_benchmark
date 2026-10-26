#!/usr/bin/env python3
"""
Build ``{key}.tsfile`` under ``backend/dataset_tsfile/<key>/`` from ``dataset_csv`` master + column_info.

Uses **table-model** TsFile (≥ 2.3.x) with fixed encodings/compression:

- int64 → TS_2DIFF
- float → GORILLA
- string → PLAIN
- time column → TS_2DIFF
- chunk compression → LZ4
- TAG columns → none (FIELD columns only; table name identifies the dataset)

Writes ``{key}_catalog_tsfile_info.csv`` documenting the layout (no Java bench CSV lookup).

Requires prior ``regenerate_dataset_info.py`` (``{key}_column_info.csv``).

**Append (default):** keep existing ``{key}.tsfile`` if present.

**Overwrite:** ``WEB_COMPRESSION_STATICDATA_REGEN=1`` — rebuild TsFile.

From repository root::

    python scripts/regenerate_catalog_tsfile.py
    python scripts/regenerate_catalog_tsfile.py --dataset AI4I-Maintenance
    WEB_COMPRESSION_STATICDATA_REGEN=1 python scripts/regenerate_catalog_tsfile.py --dataset AI4I-Maintenance
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
_OVERWRITE_ENV = "WEB_COMPRESSION_STATICDATA_REGEN"
_OVERWRITE_ENV_LEGACY = "WEB_COMPRESSION_DATASET_PROMOTE_OVERWRITE"


def _overwrite_mode() -> bool:
    if os.environ.get(_OVERWRITE_ENV, "").strip() == "1":
        return True
    if os.environ.get(_OVERWRITE_ENV_LEGACY, "").strip() == "1":
        print(
            f"Note: {_OVERWRITE_ENV_LEGACY} is deprecated; use {_OVERWRITE_ENV}=1",
            file=sys.stderr,
        )
        return True
    return False


def _run_java_selected_export(
    master_csv: str,
    column_info: str,
    output_tsfile: str,
    info_csv: str,
    dataset_key: str,
) -> None:
    if str(_BACKEND.resolve()) not in sys.path:
        sys.path.insert(0, str(_BACKEND.resolve()))
    from myapp.typed_tsfile_export import run_java_selected_tsfile_export

    run_java_selected_tsfile_export(
        master_csv,
        column_info,
        output_tsfile_path=output_tsfile,
        info_csv_path=info_csv,
        dataset_key=dataset_key,
    )


def _resolve_master_csv(key: str, bundle_dir: Path) -> Path | None:
    from dataset_csv_source import resolve_benchmark_dataset_csv

    return resolve_benchmark_dataset_csv(_BACKEND, key, f"{key}.csv")


def _tsfile_dir_for_key(key: str) -> Path:
    from dataset_csv_source import tsfile_bundle_dir

    return tsfile_bundle_dir(_BACKEND, key)


def _resolve_catalog_targets(
    dataset_filter: str | None,
) -> list[tuple[str, Path, Path, Path]]:
    """(key, master_csv, info_bundle_dir, tsfile_dir) with column_info present."""
    from myapp.dataset_catalog_assets import column_info_filename

    from dataset_csv_source import describe_dataset_csv_lookup

    if dataset_filter:
        bundle_dir = _DATASET / dataset_filter
        col_info = bundle_dir / column_info_filename(dataset_filter)
        master = _resolve_master_csv(dataset_filter, bundle_dir)
        if master is None:
            raise FileNotFoundError(
                f"Missing master CSV for {dataset_filter!r} under "
                f"{describe_dataset_csv_lookup(_BACKEND)}. "
                f"Run regenerate_dataset_csv.py first."
            )
        if not col_info.is_file():
            raise FileNotFoundError(
                f"Missing {col_info.name}. Run regenerate_dataset_info.py first."
            )
        return [
            (dataset_filter, master, bundle_dir, _tsfile_dir_for_key(dataset_filter))
        ]

    if not _DATASET.is_dir():
        raise FileNotFoundError(f"Missing directory: {_DATASET}")
    targets: list[tuple[str, Path, Path, Path]] = []
    for bundle_dir in sorted(_DATASET.iterdir()):
        if not bundle_dir.is_dir():
            continue
        key = bundle_dir.name
        col_info = bundle_dir / column_info_filename(key)
        if not col_info.is_file():
            continue
        master = _resolve_master_csv(key, bundle_dir)
        if master is None:
            continue
        targets.append((key, master, bundle_dir, _tsfile_dir_for_key(key)))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        metavar="KEY",
        help="Only build TsFile for this catalog dataset key.",
    )
    args = parser.parse_args()

    if str(_BACKEND.resolve()) not in sys.path:
        sys.path.insert(0, str(_BACKEND.resolve()))
    if str(_SCRIPTS.resolve()) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS.resolve()))

    try:
        targets = _resolve_catalog_targets(args.dataset)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if not targets:
        print(
            "No catalog bundles with dataset_csv master + column_info to process.",
            file=sys.stderr,
        )
        return 1

    from myapp.dataset_catalog_assets import build_catalog_tsfile

    from dataset_csv_source import dataset_tsfile_root

    overwrite = _overwrite_mode()
    print(
        f"Mode: {'overwrite' if overwrite else 'append'} "
        f"({f'set {_OVERWRITE_ENV}=1' if overwrite else 'default; keep existing .tsfile'})"
    )

    dataset_tsfile_root(_BACKEND).mkdir(parents=True, exist_ok=True)
    built = 0
    failed = 0
    for key, master_csv, bundle_dir, tsfile_dir in targets:
        print(f"Building table-model TsFile for {key} …")
        ok = build_catalog_tsfile(
            key,
            master_csv,
            bundle_dir,
            tsfile_dir,
            overwrite=overwrite,
            run_selected_tsfile_export=_run_java_selected_export,
        )
        if ok:
            built += 1
            print(f"OK {key} -> {tsfile_dir / f'{key}.tsfile'}")
        else:
            failed += 1

    print(f"Done. built={built} failed={failed} (of {len(targets)} requested)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
