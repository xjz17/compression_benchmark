#!/usr/bin/env python3
"""Refresh native byte-codec family rows as dataset-level summary rows.

The normal staticdata refresh scripts write one row per physical column.  That is
accurate for single-column drill-downs but too slow for wide catalog datasets
when the goal is Home-page visibility.  This script benchmarks every relevant
column and every row, then writes one ``column_index=-1`` summary row per
algorithm/dataset/dtype/language, matching the existing multi-column ``reger``
report convention.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from dataset_csv_source import describe_dataset_csv_lookup, iter_catalog_benchmark_datasets
from java_bench_common import run_java_benchmark
from regenerate_staticdata_cpp import run_cpp_benchmark
from staticdata_regen_common import (
    load_report_rows,
    merge_partial_algorithm_rows,
    parse_algorithms,
    write_report_rows,
)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
os.environ.setdefault("COMPRESSION_BENCHMARK_REPO_ROOT", str(ROOT.resolve()))
os.environ.setdefault("COMPRESSION_BENCHMARK_PYTHON_JAVA_SUPPLEMENT", "0")
os.environ.setdefault("COMPRESSION_BENCHMARK_CPP_SKIP_JAVA_COMPANION", "1")

from myapp.benchmark_loss import format_loss_rate_csv  # noqa: E402
from myapp.active_method_policy import (  # noqa: E402
    active_method_policy_summary,
    is_active_algorithm_key,
)
from myapp.native_lz_family import NATIVE_LZ_FAMILY_ALGORITHMS  # noqa: E402
from myapp.processor_patch_bench import process_csv_simple_patch  # noqa: E402


LEGACY_DEFAULT_ALGORITHMS = (
    "density",
    "gipfeli",
    "glza",
    "lzmat",
    "lzo1",
    "lzo1a",
    "lzo1b",
    "lzo1c",
    "lzo1f",
    "lzo1y",
    "lzo1z",
    "lzo2a",
    "lzrw",
    "lzsse2",
    "lzsse4",
    "lzsse4fast",
    "lzsse8",
    "lzsse8fast",
    "lzvn",
    "pithy",
    "shrinker",
    "ucl_nrv2d",
    "ucl_nrv2e",
    "wflz",
    "xpack",
    "yalz77",
    "yappy",
    "fse",
    "huff0",
    "csc_5",
    "lz4fast_1",
    "lz4hc_12",
    "slz_deflate_3",
    "slz_zlib_3",
    "zlib_9",
)
DEFAULT_ALGORITHMS = tuple(
    algorithm for algorithm in LEGACY_DEFAULT_ALGORITHMS
    if is_active_algorithm_key(algorithm)
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


def _all_python_native_lz_family(algorithms: list[str]) -> bool:
    native = set(NATIVE_LZ_FAMILY_ALGORITHMS)
    return bool(algorithms) and all(algorithm in native for algorithm in algorithms)


def _algorithm_batches(algorithms: list[str], jobs: int, batch_size: int = 0) -> list[list[str]]:
    if batch_size > 0:
        return [
            algorithms[idx : idx + batch_size]
            for idx in range(0, len(algorithms), batch_size)
        ]
    if jobs <= 1 or len(algorithms) <= 1:
        return [list(algorithms)]
    batch_count = min(jobs, len(algorithms))
    batches: list[list[str]] = [[] for _ in range(batch_count)]
    for idx, algorithm in enumerate(algorithms):
        batches[idx % batch_count].append(algorithm)
    return [batch for batch in batches if batch]


def _normalize_logical_type(raw: object) -> str:
    return str(raw or "").strip().lower()


def _iter_column_info_rows(column_info_path: Path) -> list[dict]:
    with column_info_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _columns_for_dtype(column_info_rows: list[dict], data_type: str) -> list[tuple[int, str, dict]]:
    if data_type == "int":
        wanted = {"integer", "int", "long"}
    elif data_type == "float":
        wanted = {"float", "double"}
    elif data_type == "string":
        wanted = {"string"}
    else:
        raise ValueError(data_type)
    out: list[tuple[int, str, dict]] = []
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
        out.append((idx, str(row.get("column_name", "")).strip(), dict(row)))
    return sorted(out, key=lambda item: item[0])


def _read_source_header(src: Path) -> list[str]:
    with src.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _unique_header(raw: str, idx: int, used: set[str]) -> str:
    base = str(raw or "").strip() or f"column_{idx}"
    out = base
    suffix = 2
    while out in used:
        out = f"{base}_{suffix}"
        suffix += 1
    used.add(out)
    return out


def _materialize_typed_csv(
    src: Path,
    slug: str,
    data_type: str,
    columns: list[tuple[int, str, dict]],
    source_header: list[str],
    original_column_info_rows: list[dict],
) -> tuple[Path, Path, int]:
    tmp_csv = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{slug}_{data_type}_native_summary.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )
    tmp_info = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{slug}_{data_type}_native_summary_column_info.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )
    csv_path = Path(tmp_csv.name)
    info_path = Path(tmp_info.name)
    try:
        used: set[str] = set()
        headers: list[str] = []
        for old_idx, col_name, _row in columns:
            raw = col_name
            if not raw and 0 <= old_idx < len(source_header):
                raw = source_header[old_idx]
            headers.append(_unique_header(raw, old_idx, used))

        writer = csv.writer(tmp_csv, lineterminator="\n")
        writer.writerow(headers)
        row_count = 0
        with src.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.reader(input_handle)
            next(reader, None)
            for row in reader:
                writer.writerow([row[idx] if 0 <= idx < len(row) else "" for idx, _name, _r in columns])
                row_count += 1
        tmp_csv.close()

        fieldnames = list(original_column_info_rows[0].keys()) if original_column_info_rows else []
        for field in ("column_index", "column_name", "logical_type", "max_point_number"):
            if field not in fieldnames:
                fieldnames.append(field)
        info_writer = csv.DictWriter(tmp_info, fieldnames=fieldnames, lineterminator="\n")
        info_writer.writeheader()
        for new_idx, (_old_idx, _col_name, row) in enumerate(columns):
            out = dict(row)
            out["column_index"] = new_idx
            out["column_name"] = headers[new_idx]
            info_writer.writerow({field: out.get(field, "") for field in fieldnames})
        tmp_info.close()
        return csv_path, info_path, row_count
    except Exception:
        try:
            tmp_csv.close()
        except Exception:
            pass
        try:
            tmp_info.close()
        except Exception:
            pass
        csv_path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)
        raise


def _read_summary_rows(tmp_report_path: Path) -> list[dict]:
    if not tmp_report_path.is_file():
        return []
    rows: list[dict] = []
    with tmp_report_path.open("r", encoding="utf-8", newline="") as handle:
        for rec in csv.DictReader(handle):
            algo = str(rec.get("algorithm", "")).strip()
            if not algo:
                continue
            rows.append(
                {
                    "column_index": -1,
                    "algorithm": algo,
                    "originalSize": int(float(rec.get("originalSize", 0) or 0)),
                    "compressedSize": int(float(rec.get("compressedSize", 0) or 0)),
                    "compressionTime": int(float(rec.get("compressionTime", 0) or 0)),
                    "decompressionTime": int(float(rec.get("decompressionTime", 0) or 0)),
                    "isLossy": int(float(rec.get("isLossy", 0) or 0)),
                    "lossRate": rec.get("lossRate", 0) or 0,
                }
            )
    return rows


def _merge_language_report(
    report_path: Path,
    rows: list[dict],
    algorithms: list[str],
) -> None:
    merged = merge_partial_algorithm_rows(
        load_report_rows(report_path),
        rows,
        set(algorithms),
    )
    write_report_rows(report_path, merged, format_loss_rate=format_loss_rate_csv)


def _missing_summary_algorithms(report_path: Path, algorithms: list[str]) -> list[str]:
    wanted = set(algorithms)
    seen: set[str] = set()
    for row in load_report_rows(report_path):
        if str(row.get("column_index", "")).strip() != "-1":
            continue
        algorithm = str(row.get("algorithm", "")).strip()
        if algorithm in wanted:
            seen.add(algorithm)
    return [algorithm for algorithm in algorithms if algorithm not in seen]


def _run_language(
    language: str,
    tmp_csv_path: Path,
    data_type: str,
    tmp_info_path: Path,
    algorithms: list[str],
) -> list[dict]:
    fd, tmp_report_name = tempfile.mkstemp(
        prefix=f"{tmp_csv_path.stem}_{language}_",
        suffix="_report.csv",
    )
    os.close(fd)
    tmp_report = Path(tmp_report_name)
    tmp_report.unlink(missing_ok=True)
    old_allow = os.environ.get("COMPRESSION_BENCHMARK_RUN_ALGORITHMS")
    os.environ["COMPRESSION_BENCHMARK_RUN_ALGORITHMS"] = ",".join(algorithms)
    try:
        if language == "java":
            run_java_benchmark(
                tmp_csv_path,
                data_type,
                tmp_report,
                column_info=tmp_info_path,
                algorithms=algorithms,
            )
        elif language == "python":
            cwd = os.getcwd()
            try:
                os.chdir(BACKEND)
                process_csv_simple_patch(
                    str(tmp_csv_path),
                    data_type,
                    str(tmp_report),
                    str(tmp_info_path),
                )
            finally:
                os.chdir(cwd)
        elif language == "cpp":
            run_cpp_benchmark(
                tmp_csv_path,
                data_type,
                tmp_report,
                column_info=tmp_info_path,
                algorithms=algorithms,
            )
        else:
            raise ValueError(language)
        return _read_summary_rows(tmp_report)
    finally:
        if old_allow is None:
            os.environ.pop("COMPRESSION_BENCHMARK_RUN_ALGORITHMS", None)
        else:
            os.environ["COMPRESSION_BENCHMARK_RUN_ALGORITHMS"] = old_allow
        tmp_report.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", metavar="SLUG", help="Only refresh this dataset key.")
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("int", "float"),
        help="Refresh one dtype; may be repeated. Defaults to int and float.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("java", "python", "cpp"),
        help="Refresh one implementation language; may be repeated. Defaults to java, python, cpp.",
    )
    parser.add_argument(
        "--algorithms",
        metavar="NAMES",
        help="Comma-separated native algorithm keys. Defaults to the 35 family rows.",
    )
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip dataset/dtype/language reports that already have all requested summary rows.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run Python algorithm batches in parallel for each dataset/dtype (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Algorithms per Python worker task when --jobs > 1; 0 spreads algorithms across jobs.",
    )
    args = parser.parse_args()

    algorithms = parse_algorithms(args.algorithms) or list(DEFAULT_ALGORITHMS)
    retired = [algorithm for algorithm in algorithms if not is_active_algorithm_key(algorithm)]
    if retired:
        policy = active_method_policy_summary()
        parser.error(
            f"retired algorithm(s) {retired!r} are blocked by active-method policy "
            f"{policy['policy_id']}"
        )
    languages = args.language or ["java", "python", "cpp"]
    dtypes = args.dtype or ["int", "float"]

    pairs = iter_catalog_benchmark_datasets(BACKEND)
    if args.dataset:
        want = args.dataset.strip()
        pairs = [(slug, path) for slug, path in pairs if slug == want]
        if not pairs:
            raise SystemExit(f"No catalog dataset matches --dataset {want!r}")
    if not pairs:
        print(
            f"No datasets found under {describe_dataset_csv_lookup(BACKEND)} "
            "with column_info in backend/dataset/"
        )
        return 0

    print(f"Algorithms ({len(algorithms)}): {', '.join(algorithms)}", flush=True)
    print(f"Languages: {', '.join(languages)}; dtypes: {', '.join(dtypes)}", flush=True)
    print(
        f"patch_rows={os.environ.get('COMPRESSION_BENCHMARK_CSV_PATCH_ROWS', '10000')}",
        flush=True,
    )

    for slug, src in pairs:
        source_header = _read_source_header(src)
        out_root = BACKEND / "dataset" / slug
        col_info = out_root / f"{slug}_column_info.csv"
        if not col_info.is_file():
            raise SystemExit(f"Missing {col_info}; run regenerate_dataset_info.py first.")
        column_info_rows = _iter_column_info_rows(col_info)
        for data_type in dtypes:
            columns = [
                col for col in _columns_for_dtype(column_info_rows, data_type)
                if 0 <= col[0] < len(source_header)
            ]
            report_paths = {
                lang: out_root / f"{slug}_{data_type}_{lang}_compression_result.csv"
                for lang in languages
            }
            missing_by_language = {
                language: _missing_summary_algorithms(report_paths[language], algorithms)
                for language in languages
            }
            if args.skip_complete and all(not missing for missing in missing_by_language.values()):
                print(f"Skip {slug} {data_type}: requested summary rows already present", flush=True)
                continue
            if not columns:
                print(f"Skip {slug} {data_type}: no columns", flush=True)
                for report_path in report_paths.values():
                    _merge_language_report(report_path, [], algorithms)
                continue
            tmp_csv_path, tmp_info_path, row_count = _materialize_typed_csv(
                src,
                slug,
                data_type,
                columns,
                source_header,
                column_info_rows,
            )
            try:
                print(
                    f"Refresh {slug} {data_type}: {len(columns)} columns, {row_count} rows",
                    flush=True,
                )
                for language in languages:
                    language_algorithms = (
                        missing_by_language[language]
                        if args.skip_complete
                        else algorithms
                    )
                    if not language_algorithms:
                        print(f"  -> {language}: already complete", flush=True)
                        continue
                    print(
                        f"  -> {language} ({len(language_algorithms)} algorithms)",
                        flush=True,
                    )
                    if (
                        language == "python"
                        and args.jobs > 1
                        and len(language_algorithms) > 1
                        and not _all_python_native_lz_family(language_algorithms)
                    ):
                        rows: list[dict] = []
                        batches = _algorithm_batches(
                            language_algorithms,
                            args.jobs,
                            max(0, args.batch_size),
                        )
                        with concurrent.futures.ProcessPoolExecutor(
                            max_workers=min(args.jobs, len(batches))
                        ) as executor:
                            future_to_batch = {
                                executor.submit(
                                    _run_language,
                                    language,
                                    tmp_csv_path,
                                    data_type,
                                    tmp_info_path,
                                    batch,
                                ): batch
                                for batch in batches
                            }
                            for future in concurrent.futures.as_completed(future_to_batch):
                                batch = future_to_batch[future]
                                batch_rows = future.result()
                                if not batch_rows:
                                    print(
                                        f"     batch {','.join(batch)} emitted no rows",
                                        flush=True,
                                    )
                                _merge_language_report(
                                    report_paths[language],
                                    batch_rows,
                                    batch,
                                )
                                rows.extend(batch_rows)
                                print(
                                    f"     batch {len(batch)} algorithms complete: "
                                    f"{len(batch_rows)} rows",
                                    flush=True,
                                )
                    else:
                        rows = _run_language(
                            language,
                            tmp_csv_path,
                            data_type,
                            tmp_info_path,
                            language_algorithms,
                        )
                        if not rows:
                            print(f"     no rows emitted", flush=True)
                        _merge_language_report(report_paths[language], rows, language_algorithms)
                    print(
                        f"     wrote {report_paths[language].name}: {len(rows)} summary rows",
                        flush=True,
                    )
            finally:
                tmp_csv_path.unlink(missing_ok=True)
                tmp_info_path.unlink(missing_ok=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
