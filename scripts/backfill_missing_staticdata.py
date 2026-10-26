#!/usr/bin/env python3
"""Backfill missing static benchmark rows without replacing existing metrics.

This is a conservative wrapper around regenerate_staticdata_{java,python,cpp}.py.
Those generators support ``--algorithms`` partial refresh, but their merge step
replaces all existing rows for the selected algorithms. This wrapper restores
the original report and appends only rows whose ``(column_index, algorithm)``
pair was absent before the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DEFAULT_DATASET_ROOT = ROOT / "orignial_datasets"
DEFAULT_LOG_DIR = ROOT / "analysis_outputs" / "backfill_missing_staticdata"
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
LANGUAGE_SCRIPTS = {
    "java": "scripts/regenerate_staticdata_java.py",
    "python": "scripts/regenerate_staticdata_python.py",
    "cpp": "scripts/regenerate_staticdata_cpp.py",
}
REGER_KEY = "reger"


@dataclass(frozen=True)
class Target:
    dataset: str
    dtype: str
    language: str
    report_path: Path
    columns: tuple[int, ...]
    missing_by_algorithm: dict[str, frozenset[int]]

    @property
    def missing_algorithms(self) -> list[str]:
        return [name for name, indexes in self.missing_by_algorithm.items() if indexes]

    @property
    def missing_pair_count(self) -> int:
        return sum(len(indexes) for indexes in self.missing_by_algorithm.values())


def _run_node_catalog() -> list[str]:
    script = (
        "import { getBenchmarkCatalogAlgorithms } "
        "from './frontend/src/utils/benchmarkAlgorithmCatalog.js'; "
        "console.log(JSON.stringify(getBenchmarkCatalogAlgorithms()));"
    )
    out = subprocess.check_output(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        text=True,
    )
    values = json.loads(out)
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        name = str(value).strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _parse_csv_list(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or not raw.strip():
        return default
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        value = part.strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _dataset_names(dataset_root: Path, requested: list[str]) -> list[str]:
    if requested:
        return sorted(dict.fromkeys(requested), key=str.lower)
    return sorted(
        (path.stem for path in dataset_root.glob("*.csv") if path.is_file()),
        key=str.lower,
    )


def _dataset_size(dataset_root: Path, dataset: str) -> int:
    path = dataset_root / f"{dataset}.csv"
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _column_indexes_for_dtype(dataset: str, dtype: str) -> tuple[int, ...]:
    path = BACKEND / "dataset" / dataset / f"{dataset}_column_info.csv"
    if not path.is_file():
        return ()
    numeric_stats_path = BACKEND / "dataset" / dataset / f"{dataset}_numeric_statistic.csv"
    numeric_stat_indexes: set[int] | None = None
    if numeric_stats_path.is_file():
        numeric_stat_indexes = set()
        with numeric_stats_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    numeric_stat_indexes.add(int(row.get("column_index")))
                except (TypeError, ValueError):
                    continue
    wanted = {"integer", "int", "long", "timestamp"} if dtype == "int" else {"float", "double"}
    indexes: list[int] = []
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            logical = str(row.get("logical_type") or "").strip().lower()
            if logical not in wanted:
                continue
            try:
                idx = int(row.get("column_index"))
            except (TypeError, ValueError):
                continue
            if numeric_stat_indexes is not None and idx not in numeric_stat_indexes:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            indexes.append(idx)
    return tuple(sorted(indexes))


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {field: str(row.get(field, "") or "") for field in REPORT_FIELDS}
            for row in csv.DictReader(handle)
        ]


def _pair(row: dict[str, str]) -> tuple[int, str] | None:
    algorithm = str(row.get("algorithm") or "").strip()
    if not algorithm:
        return None
    try:
        column_index = int(float(str(row.get("column_index") or "").strip()))
    except (TypeError, ValueError):
        return None
    return column_index, algorithm


def _is_valid_lossless_row(row: dict[str, str]) -> bool:
    is_lossy = str(row.get("isLossy") or "").strip().lower()
    loss_rate = str(row.get("lossRate") or "").strip().lower()
    return is_lossy in ("", "0", "0.0", "false") and loss_rate in ("", "0", "0.0", "0.00")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".backfill.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "") or "") for field in REPORT_FIELDS})
    tmp.replace(path)


def _restore(path: Path, original_bytes: bytes | None) -> None:
    if original_bytes is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original_bytes)


def _expected_indexes_for_algorithm(algorithm: str, columns: tuple[int, ...]) -> frozenset[int]:
    if algorithm == REGER_KEY:
        return frozenset({-1})
    return frozenset(columns)


def build_targets(
    *,
    datasets: list[str],
    dtypes: list[str],
    languages: list[str],
    algorithms: list[str],
) -> list[Target]:
    targets: list[Target] = []
    for dataset in datasets:
        for dtype in dtypes:
            columns = _column_indexes_for_dtype(dataset, dtype)
            if not columns:
                continue
            for language in languages:
                report_path = (
                    BACKEND
                    / "dataset"
                    / dataset
                    / f"{dataset}_{dtype}_{language}_compression_result.csv"
                )
                existing_pairs = {
                    pair
                    for row in _load_rows(report_path)
                    for pair in [_pair(row)]
                    if pair is not None and _is_valid_lossless_row(row)
                }
                missing_by_algorithm: dict[str, frozenset[int]] = {}
                for algorithm in algorithms:
                    expected = _expected_indexes_for_algorithm(algorithm, columns)
                    missing = frozenset(
                        idx for idx in expected if (idx, algorithm) not in existing_pairs
                    )
                    if missing:
                        missing_by_algorithm[algorithm] = missing
                if missing_by_algorithm:
                    targets.append(
                        Target(
                            dataset=dataset,
                            dtype=dtype,
                            language=language,
                            report_path=report_path,
                            columns=columns,
                            missing_by_algorithm=missing_by_algorithm,
                        )
                    )
    return targets


def _algorithm_batches(algorithms: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0 or batch_size >= len(algorithms):
        return [algorithms]
    return [algorithms[i : i + batch_size] for i in range(0, len(algorithms), batch_size)]


def _subprocess_env(dataset_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["COMPRESSION_BENCHMARK_EXTERNAL_DATASET_CSV_ROOT"] = str(dataset_root.resolve())
    env.setdefault("COMPRESSION_BENCHMARK_REPO_ROOT", str(ROOT.resolve()))
    env.setdefault("COMPRESSION_BENCHMARK_CSV_PATCH_ROWS", "10000")
    return env


def _log_path(
    target: Target,
    algorithms: list[str],
    log_dir: Path,
    column_indexes: list[int] | None = None,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    first = algorithms[0] if algorithms else "none"
    suffix = f"{len(algorithms)}_{first}".replace("/", "_").replace(" ", "_")
    if len(suffix) > 80:
        suffix = suffix[:80]
    if column_indexes:
        suffix += "_cols_" + "_".join(str(idx) for idx in column_indexes)
    return log_dir / f"{target.dataset}_{target.dtype}_{target.language}_{suffix}.log"


def _terminate_process_group(process: subprocess.Popen, grace_sec: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_in_process_group(
    command: list[str],
    *,
    log,
    env: dict[str, str],
    timeout_sec: int,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout_sec if timeout_sec > 0 else None)
    except BaseException:
        _terminate_process_group(process)
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _prepare_column_cache(
    target: Target,
    *,
    dataset_root: Path,
    log_dir: Path,
) -> None:
    if not os.environ.get("COMPRESSION_BENCHMARK_COLUMN_CACHE_DIR", "").strip():
        return
    column_indexes = sorted(
        {
            idx
            for indexes in target.missing_by_algorithm.values()
            for idx in indexes
            if idx >= 0
        }
    )
    if not column_indexes:
        return
    command = [
        sys.executable,
        LANGUAGE_SCRIPTS[target.language],
        "--dataset",
        target.dataset,
        "--dtype",
        target.dtype,
        "--column-indexes",
        ",".join(str(idx) for idx in column_indexes),
        "--prepare-column-cache",
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{target.dataset}_{target.dtype}_{target.language}_prepare_column_cache.log"
    print(
        f"PREPARE_COLUMN_CACHE {target.dataset} {target.dtype}: "
        f"{len(column_indexes)} columns",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        _run_in_process_group(
            command,
            log=log,
            env=_subprocess_env(dataset_root),
            timeout_sec=0,
        )


def run_batch(
    target: Target,
    algorithms: list[str],
    *,
    dataset_root: Path,
    log_dir: Path,
    timeout_sec: int,
    dry_run: bool,
    column_indexes: list[int] | None = None,
) -> tuple[int, int]:
    if not algorithms:
        return 0, 0
    requested_indexes = set(column_indexes or ())
    missing_indexes = {
        algorithm: (
            target.missing_by_algorithm[algorithm] & requested_indexes
            if requested_indexes
            else target.missing_by_algorithm[algorithm]
        )
        for algorithm in algorithms
        if target.missing_by_algorithm.get(algorithm)
    }
    missing_pairs = {
        (idx, algorithm)
        for algorithm, indexes in missing_indexes.items()
        for idx in indexes
    }
    if not missing_pairs:
        return 0, 0

    command = [
        sys.executable,
        LANGUAGE_SCRIPTS[target.language],
        "--dataset",
        target.dataset,
        "--dtype",
        target.dtype,
        "--algorithms",
        ",".join(algorithms),
    ]
    if column_indexes:
        command.extend(["--column-indexes", ",".join(str(idx) for idx in column_indexes)])
    print(
        f"RUN {target.dataset} {target.dtype} {target.language}: "
        f"{len(algorithms)} algorithms, {len(missing_pairs)} missing pairs",
        flush=True,
    )
    if dry_run:
        print("  " + " ".join(command), flush=True)
        return 0, 0

    original_bytes = target.report_path.read_bytes() if target.report_path.is_file() else None
    original_rows = _load_rows(target.report_path)
    original_pairs = {
        pair
        for row in original_rows
        for pair in [_pair(row)]
        if pair is not None and _is_valid_lossless_row(row)
    }
    started = time.time()
    log_path = _log_path(target, algorithms, log_dir, column_indexes)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join(command) + "\n\n")
            log.flush()
            _run_in_process_group(
                command,
                log=log,
                env=_subprocess_env(dataset_root),
                timeout_sec=timeout_sec,
            )
        generated_rows = _load_rows(target.report_path)
    except BaseException:
        _restore(target.report_path, original_bytes)
        print(f"  -> restored original report after failure; log={log_path}", flush=True)
        raise

    additions: list[dict[str, str]] = []
    added_pairs: set[tuple[int, str]] = set()
    for row in generated_rows:
        pair = _pair(row)
        if pair is None:
            continue
        if pair not in missing_pairs:
            continue
        if pair in original_pairs or pair in added_pairs:
            continue
        additions.append(row)
        added_pairs.add(pair)

    if additions:
        merged = [
            row
            for row in original_rows
            if (_pair(row) not in added_pairs)
        ] + additions
        merged.sort(key=lambda row: (_pair(row) or (10**18, str(row.get("algorithm", "")))))
        _write_rows(target.report_path, merged)
    else:
        _restore(target.report_path, original_bytes)

    elapsed = time.time() - started
    print(
        f"  -> added {len(additions)} rows; preserved {len(original_rows)} existing rows "
        f"({elapsed:.1f}s); log={log_path}",
        flush=True,
    )
    return len(additions), len(missing_pairs)


def run_batch_with_split(
    target: Target,
    algorithms: list[str],
    *,
    dataset_root: Path,
    log_dir: Path,
    timeout_sec: int,
) -> tuple[int, int, list[tuple[Target, list[str], str]]]:
    try:
        added, attempted = run_batch(
            target,
            algorithms,
            dataset_root=dataset_root,
            log_dir=log_dir,
            timeout_sec=timeout_sec,
            dry_run=False,
        )
        return added, attempted, []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        reason = (
            f"timeout {timeout_sec}s"
            if isinstance(exc, subprocess.TimeoutExpired)
            else f"exit {exc.returncode}"
        )
        if len(algorithms) <= 1:
            return 0, 0, [(target, algorithms, reason)]
        mid = len(algorithms) // 2
        print(
            f"SPLIT {target.dataset} {target.dtype} {target.language}: "
            f"{len(algorithms)} algorithms -> {mid}+{len(algorithms) - mid} ({reason})",
            flush=True,
        )
        added_left, attempted_left, failures_left = run_batch_with_split(
            target,
            algorithms[:mid],
            dataset_root=dataset_root,
            log_dir=log_dir,
            timeout_sec=timeout_sec,
        )
        added_right, attempted_right, failures_right = run_batch_with_split(
            target,
            algorithms[mid:],
            dataset_root=dataset_root,
            log_dir=log_dir,
            timeout_sec=timeout_sec,
        )
        return (
            added_left + added_right,
            attempted_left + attempted_right,
            failures_left + failures_right,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--dtypes", default="int,float")
    parser.add_argument("--languages", default="java,python,cpp")
    parser.add_argument("--algorithms", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Run each missing (algorithm, column) pair independently and persist it immediately.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    catalog = _run_node_catalog() if not args.algorithms.strip() else []
    algorithms = _parse_csv_list(args.algorithms, catalog)
    dtypes = _parse_csv_list(args.dtypes, ["int", "float"])
    languages = _parse_csv_list(args.languages, ["java", "python", "cpp"])
    unknown_languages = [name for name in languages if name not in LANGUAGE_SCRIPTS]
    if unknown_languages:
        raise SystemExit(f"unsupported languages: {', '.join(unknown_languages)}")

    datasets = _dataset_names(dataset_root, args.dataset)
    targets = build_targets(
        datasets=datasets,
        dtypes=dtypes,
        languages=languages,
        algorithms=algorithms,
    )
    targets.sort(
        key=lambda item: (
            _dataset_size(dataset_root, item.dataset),
            item.dataset.lower(),
            item.dtype,
            item.language,
        )
    )
    if args.max_targets > 0:
        targets = targets[: args.max_targets]

    print(
        f"datasets={len(datasets)} dtypes={','.join(dtypes)} "
        f"languages={','.join(languages)} algorithms={len(algorithms)} "
        f"targets_with_missing={len(targets)}",
        flush=True,
    )
    print(
        "missing_pairs="
        + str(sum(target.missing_pair_count for target in targets)),
        flush=True,
    )
    if args.dry_run:
        for target in targets:
            print(
                f"DRY {target.dataset} {target.dtype} {target.language}: "
                f"{len(target.missing_algorithms)} algorithms, "
                f"{target.missing_pair_count} missing pairs",
                flush=True,
            )
        return 0

    total_added = 0
    total_attempted_pairs = 0
    failures: list[tuple[Target, list[str], str]] = []
    for target in targets:
        if args.pairwise:
            if target.language not in ("cpp", "java"):
                raise SystemExit("--pairwise currently supports only C++ and Java targets")
            try:
                _prepare_column_cache(
                    target,
                    dataset_root=dataset_root,
                    log_dir=args.log_dir,
                )
            except subprocess.CalledProcessError as exc:
                failures.append((target, target.missing_algorithms, f"cache prepare exit {exc.returncode}"))
                continue
            for algorithm in target.missing_algorithms:
                for column_index in sorted(target.missing_by_algorithm[algorithm]):
                    try:
                        # REGER is a multi-column algorithm and uses the sentinel
                        # column_index=-1. Keep it as one atomic special case.
                        pair_columns = None if column_index < 0 else [column_index]
                        added, attempted = run_batch(
                            target,
                            [algorithm],
                            dataset_root=dataset_root,
                            log_dir=args.log_dir,
                            timeout_sec=args.timeout_sec,
                            dry_run=False,
                            column_indexes=pair_columns,
                        )
                        total_added += added
                        total_attempted_pairs += attempted
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                        reason = (
                            f"column {column_index}: timeout {args.timeout_sec}s"
                            if isinstance(exc, subprocess.TimeoutExpired)
                            else f"column {column_index}: exit {exc.returncode}"
                        )
                        failures.append((target, [algorithm], reason))
            continue
        for batch in _algorithm_batches(target.missing_algorithms, args.batch_size):
            added, attempted, batch_failures = run_batch_with_split(
                target,
                batch,
                dataset_root=dataset_root,
                log_dir=args.log_dir,
                timeout_sec=args.timeout_sec,
            )
            total_added += added
            total_attempted_pairs += attempted
            failures.extend(batch_failures)

    print(
        f"backfill complete: added_rows={total_added} attempted_missing_pairs={total_attempted_pairs} "
        f"failed_batches={len(failures)}",
        flush=True,
    )
    if failures:
        for target, batch, reason in failures[:30]:
            print(
                f"FAILED {target.dataset} {target.dtype} {target.language} "
                f"{len(batch)} algorithms: {reason}",
                flush=True,
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
