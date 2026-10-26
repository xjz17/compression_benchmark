#!/usr/bin/env python3
"""Append lzbench rows to existing ``*_compression_result.csv`` files (catalog staticdata).

Uses the same patch-chunked lzbench path as Java/C++/Python patch benches (via
``accumulate_lzbench_column_byte_data`` / ``scripts/lzbench_patch_cli.py``). Does not re-run
Java/C++/Python benches — only backfills lzbench rows into existing reports.

From repository root (backend venv active)::

    python scripts/merge_lzbench_staticdata.py
    python scripts/merge_lzbench_staticdata.py --dataset AI4I-Maintenance
    python scripts/merge_lzbench_staticdata.py --dataset AI4I-Maintenance --dtype int
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
_SCRIPTS = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

os.chdir(BACKEND)

from dataset_csv_source import iter_catalog_benchmark_datasets  # noqa: E402
from myapp.lzbench_external import append_lzbench_rows_to_reports, lzbench_bench_enabled  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="SLUG", help="Only this catalog key.")
    parser.add_argument(
        "--dtype",
        choices=("int", "float"),
        help="Only merge into reports for this dtype.",
    )
    parser.add_argument(
        "--language",
        choices=("java", "python", "cpp"),
        help="Only merge into one language report (default: all three).",
    )
    args = parser.parse_args()

    if not lzbench_bench_enabled():
        print(
            "lzbench disabled (WEB_COMPRESSION_LZBENCH_BENCH=0 or binary missing). "
            "Build: python scripts/build_lzbench.py",
            file=sys.stderr,
        )
        return 1

    pairs = iter_catalog_benchmark_datasets(BACKEND)
    if args.dataset:
        want = args.dataset.strip()
        pairs = [(s, p) for s, p in pairs if s == want]
        if not pairs:
            print(f"No catalog dataset matches --dataset {want!r}", file=sys.stderr)
            return 1

    dtypes = (args.dtype,) if args.dtype else ("int", "float")
    langs = (args.language,) if args.language else ("java", "python", "cpp")

    merged = 0
    for slug, src in pairs:
        out_root = BACKEND / "dataset" / slug
        col_info = out_root / f"{slug}_column_info.csv"
        if not col_info.is_file():
            print(f"skip {slug}: missing {col_info.name}", file=sys.stderr)
            continue
        for dt in dtypes:
            paths = [
                str(out_root / f"{slug}_{dt}_{lang}_compression_result.csv")
                for lang in langs
            ]
            missing = [p for p in paths if not os.path.isfile(p)]
            if missing:
                print(f"skip {slug} {dt}: missing {missing[0]}")
                continue
            print(f"lzbench merge {slug} {dt} -> {', '.join(os.path.basename(p) for p in paths)}")
            merged += append_lzbench_rows_to_reports(
                paths,
                master_csv=str(src),
                column_info_path=str(col_info),
                data_type=dt,
            )
    print(f"Done. merged_row_writes={merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
