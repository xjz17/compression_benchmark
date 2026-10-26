#!/usr/bin/env python3
"""
Write ``<dataset>_<int|float>_tsfile_compression_result.csv`` under backend/dataset/<key>/ for each
dataset with master CSV in ``backend/dataset_csv/<key>.csv``.

Per-algorithm TsFiles go to ``backend/dataset_tsfile/<key>/tsfile_artifacts/<dtype>/``.

Requires ``<key>_column_info.csv`` (from ``regenerate_dataset_info.py``). Column selection
matches ``ProcessorTsfileStaticdataBench.buildPlan``:

- ``int``: columns with ``logical_type=integer`` only.
- ``float``: columns with ``logical_type=float`` only.
String/byte staticdata generation is disabled for now.

Datasets with no matching columns for a dtype are **skipped** (no empty all-zero CSVs).

Java reads the master CSV **once per TsFile algorithm**, streaming each row into the file (no
full-dataset buffer). BOS (``ts_2diff_bos``) accumulates columns in patches
(``COMPRESSION_BENCHMARK_CSV_PATCH_ROWS``, default 10000).

**Default (append):** skip existing ``*_tsfile_compression_result.csv`` when present.

**Overwrite:** ``COMPRESSION_BENCHMARK_STATICDATA_REGEN=1`` also removes stale ``*_tsfile`` CSVs and
``dataset_tsfile/<key>/tsfile_artifacts/<dtype>/`` for dtypes that no longer apply.

Prerequisites: built ``benchmark-java-1.0-SNAPSHOT.jar`` (``java/target/`` preferred, or ``backend/lib/``).

Usage (from repo root):

  python scripts/regenerate_tsfile_staticdata.py
  python scripts/regenerate_tsfile_staticdata.py --dataset <key>
  COMPRESSION_BENCHMARK_STATICDATA_REGEN=1 python scripts/regenerate_tsfile_staticdata.py
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_csv_source import (  # noqa: E402
    describe_dataset_csv_lookup,
    resolve_benchmark_dataset_csv,
    tsfile_bundle_dir,
)
from java_bench_common import JAR, java_jvm_prefix  # noqa: E402
from staticdata_regen_common import (  # noqa: E402
    load_report_rows,
    merge_partial_algorithm_rows,
    parse_algorithms,
    write_report_rows,
)

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
DATASET_ROOT = BACKEND / "dataset"
_OVERWRITE_ENV = "COMPRESSION_BENCHMARK_STATICDATA_REGEN"

_INTEGER_LOGICAL = "integer"
_FLOAT_LOGICAL = "float"
_STRING_LOGICAL = "string"
_ALL_TSFILE_DTYPES = ("int", "float")


def _format_loss_rate(value: float) -> str:
    return "0" if value == 0 else f"{value:.12g}"


def _overwrite_mode() -> bool:
    return os.environ.get(_OVERWRITE_ENV, "").strip() == "1"


def _count_benchable_logical_types(column_info: Path) -> tuple[int, int, int]:
    """Return ``(n_integer, n_float, n_string)`` from ``{key}_column_info.csv``."""
    n_int = 0
    n_float = 0
    n_string = 0
    with column_info.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "logical_type" not in reader.fieldnames:
            return 0, 0, 0
        for row in reader:
            logical = (row.get("logical_type") or "").strip().lower()
            if logical == _INTEGER_LOGICAL:
                n_int += 1
            elif logical == _FLOAT_LOGICAL:
                n_float += 1
            elif logical == _STRING_LOGICAL:
                n_string += 1
    return n_int, n_float, n_string


def tsfile_data_types_for_column_info(column_info: Path) -> list[str]:
    """
    Dtypes to bench, aligned with ``ProcessorTsfileStaticdataBench.buildPlan``.

    ``int`` / ``float`` when matching logical_type columns exist.
    """
    n_int, n_float, _n_string = _count_benchable_logical_types(column_info)
    if n_int == 0 and n_float == 0:
        return []
    out: list[str] = []
    if n_int > 0:
        out.append("int")
    if n_float > 0:
        out.append("float")
    return out


def _prune_stale_tsfile_outputs(
    info_dir: Path,
    tsfile_bundle_dir: Path,
    ds: str,
    active_dtypes: set[str],
) -> None:
    """Remove report CSVs and artifact dirs for dtypes that no longer apply (overwrite mode)."""
    for dtype in _ALL_TSFILE_DTYPES:
        if dtype in active_dtypes:
            continue
        out_csv = info_dir / f"{ds}_{dtype}_tsfile_compression_result.csv"
        artifact_dir = tsfile_bundle_dir / "tsfile_artifacts" / dtype
        if out_csv.is_file():
            out_csv.unlink()
            print(f"Removed stale {out_csv.name} (no matching columns in column_info)")
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
            print(f"Removed stale {artifact_dir.relative_to(tsfile_bundle_dir)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="KEY", help="Only regenerate this catalog key.")
    parser.add_argument(
        "--algorithms",
        metavar="NAMES",
        help=(
            "Comma-separated TsFile algorithm row names to benchmark. When set, only these "
            "algorithms run and their rows are merged into existing reports."
        ),
    )
    args = parser.parse_args()
    selected_algorithms = parse_algorithms(args.algorithms)

    if not JAR.is_file():
        print(
            "Missing Java benchmark JAR. Build: cd java && mvn package -DskipTests "
            f"(expected java/target/ or backend/lib/{JAR.name})",
            file=sys.stderr,
        )
        return 1
    if not DATASET_ROOT.is_dir():
        print(f"Missing {DATASET_ROOT}", file=sys.stderr)
        return 1

    os.environ.setdefault("COMPRESSION_BENCHMARK_CSV_PATCH_ROWS", "10000")
    os.environ.setdefault("COMPRESSION_BENCHMARK_REPO_ROOT", str(REPO.resolve()))

    overwrite = _overwrite_mode()
    print(
        f"Mode: {'overwrite' if overwrite else 'append'} "
        f"(patch rows={os.environ.get('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')})"
    )
    if selected_algorithms:
        print(
            f"Allowlist algorithms ({len(selected_algorithms)}): "
            f"{', '.join(selected_algorithms)}"
        )

    wrote = 0
    skipped = 0
    for info_dir in sorted(DATASET_ROOT.iterdir()):
        if not info_dir.is_dir():
            continue
        ds = info_dir.name
        if args.dataset and ds != args.dataset.strip():
            continue
        src = resolve_benchmark_dataset_csv(BACKEND, ds, f"{ds}.csv")
        if src is None:
            print(f"skip {ds}: no master CSV under {describe_dataset_csv_lookup(BACKEND)}")
            continue
        column_info = info_dir / f"{ds}_column_info.csv"
        if not column_info.is_file():
            print(
                f"skip {ds}: missing {column_info.name} (run regenerate_dataset_info.py)",
                file=sys.stderr,
            )
            continue

        n_int, n_float, n_string = _count_benchable_logical_types(column_info)
        dtypes = tsfile_data_types_for_column_info(column_info)
        if not dtypes:
            print(f"skip {ds}: no integer/float columns in {column_info.name}")
            continue

        tsfile_bundle_dir_path = tsfile_bundle_dir(BACKEND, ds)
        tsfile_bundle_dir_path.mkdir(parents=True, exist_ok=True)

        if overwrite:
            _prune_stale_tsfile_outputs(info_dir, tsfile_bundle_dir_path, ds, set(dtypes))

        print(
            f"{ds}: column_info integer={n_int} float={n_float} string={n_string} "
            f"-> tsfile dtypes {','.join(dtypes)}"
        )

        for dtype in dtypes:
            out_csv = info_dir / f"{ds}_{dtype}_tsfile_compression_result.csv"
            if not selected_algorithms and not overwrite and out_csv.is_file():
                print(f"Skip {ds} {dtype}: keep {out_csv.name}")
                skipped += 1
                continue

            artifact_dir = tsfile_bundle_dir_path / "tsfile_artifacts" / dtype
            if not selected_algorithms and artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            report_target = out_csv
            tmp_report: Path | None = None
            if selected_algorithms:
                safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", "_".join(selected_algorithms))
                tmp_report = out_csv.with_name(f"{out_csv.stem}.partial.{safe}.csv")
                tmp_report.unlink(missing_ok=True)
                report_target = tmp_report

            cmd = [
                "java",
                *java_jvm_prefix(),
                "-jar",
                str(JAR),
                "--input",
                str(src.resolve()),
                "--data-type",
                dtype,
                "--column-info",
                str(column_info.resolve()),
                "--tsfile-dir",
                str(artifact_dir.resolve()),
                "--output-report",
                str(report_target.resolve()),
            ]
            if selected_algorithms:
                cmd.extend(["--algorithms", ",".join(selected_algorithms)])
            print("RUN", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, cwd=str(REPO))
                if selected_algorithms and tmp_report is not None:
                    existing_rows = load_report_rows(out_csv)
                    new_rows = load_report_rows(tmp_report)
                    merged_rows = merge_partial_algorithm_rows(
                        existing_rows,
                        new_rows,
                        set(selected_algorithms),
                    )
                    write_report_rows(
                        out_csv,
                        merged_rows,
                        format_loss_rate=_format_loss_rate,
                    )
            finally:
                if tmp_report is not None:
                    tmp_report.unlink(missing_ok=True)
            wrote += 1

    print(f"Done. wrote={wrote} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
