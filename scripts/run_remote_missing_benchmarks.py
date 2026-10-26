#!/usr/bin/env python3
"""Run only missing benchmark rows for the 19-dataset remote batch.

This is a thin orchestration wrapper around the existing staticdata regenerators.
It never deletes reports. It scans existing result CSVs, builds missing
dataset/dtype/language/algorithm chunks, and lets the existing scripts merge
new rows into the report files.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from project_dataset_scope import CORE_CSV_DATASETS


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATASET_ROOT = BACKEND / "dataset"
CSV_ROOT = BACKEND / "dataset_csv"
LOG_ROOT = Path("/home/user/xjz/compression_benchmark/run_logs")

DATASETS = list(CORE_CSV_DATASETS)

EXCLUDE_ALGORITHMS = {
    "kanzi_9",
    "bzip3_10",
    "ppmd8_9",
    "bzip2_9",
    "DELTA_BINARY_PACKED",
    "lzlib_9",
    "fastlzma2_10",
    "bsc6",
    "brotli",
    "reger",
    "lzf_1",
    "fastlz_2",
    "lzjb",
    "lzav_2",
    "lzfse",
    "quicklz_3",
    "spdp_level_9",
    "alp",
    "ts_2diff_bos",
    "corad_tricklet",
    "lizard_49",
    "zxc_6",
    "libdeflate_12",
    "zlib_ng_9",
    "lzg_9",
}

LOSSY_PREFIXES = (
    "ext_zfp_tol_",
    "machete_rel_",
    "machete_float_rel_",
    "most_minimal",
    "most_float_minimal",
    "sz_",
    "sz3_",
    "serf_",
    "simpiece_",
)

LANGUAGE_SCRIPTS = {
    "java": "scripts/regenerate_staticdata_java.py",
    "python": "scripts/regenerate_staticdata_python.py",
    "cpp": "scripts/regenerate_staticdata_cpp.py",
}


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def read_algorithms(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get("algorithm") or row.get("Algorithm") or "").strip()
            if value:
                out.add(value)
    return out


def target_algorithms() -> list[str]:
    algs: set[str] = set()
    for path in DATASET_ROOT.glob("*/*_compression_result.csv"):
        algs.update(read_algorithms(path))
    return sorted(
        alg
        for alg in algs
        if alg not in EXCLUDE_ALGORITHMS
        and not alg.startswith("lzma_level_")
        and not any(alg.startswith(prefix) for prefix in LOSSY_PREFIXES)
    )


def dataset_csv_exists(dataset: str) -> bool:
    if (CSV_ROOT / f"{dataset}.csv").is_file():
        return True
    external = Path("/data/yn/compression_benchmark/backend_dataset_csv") / f"{dataset}.csv"
    return external.is_file()


def dtype_has_columns(dataset: str, dtype: str) -> bool:
    path = DATASET_ROOT / dataset / f"{dataset}_column_info.csv"
    if not path.is_file():
        return False
    wanted = {"integer", "int", "long"} if dtype == "int" else {"float", "double"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("logical_type") or "").strip().lower() in wanted:
                return True
    return False


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def report_path(dataset: str, dtype: str, language: str) -> Path:
    return DATASET_ROOT / dataset / f"{dataset}_{dtype}_{language}_compression_result.csv"


def command_for(language: str, dataset: str, dtype: str, algorithms: list[str]) -> list[str]:
    return [
        sys.executable,
        LANGUAGE_SCRIPTS[language],
        "--dataset",
        dataset,
        "--dtype",
        dtype,
        "--algorithms",
        ",".join(algorithms),
    ]


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("WEB_COMPRESSION_STATICDATA_REGEN", "0")
    env.setdefault("WEB_COMPRESSION_CSV_PATCH_ROWS", "10000")
    env.setdefault("WEB_COMPRESSION_CMIX_BENCH", "0")
    env.setdefault("WEB_COMPRESSION_MSDZIP_BENCH", "0")
    env.setdefault("WEB_COMPRESSION_MSDLC_BENCH", "0")
    env.setdefault("WEB_COMPRESSION_SERF_BENCH", "0")
    env.setdefault("WEB_COMPRESSION_BUFF_BENCH", "0")
    env.setdefault("WEB_COMPRESSION_REPO_ROOT", str(ROOT))
    env.setdefault("WEB_COMPRESSION_JAVA_OPTS", "-Xmx16g")
    env.setdefault("WEB_COMPRESSION_CPP_BUILD_JOBS", "16")
    return env


def ensure_cpp_build(env: dict[str, str], log_file: Path) -> bool:
    exe = ROOT / "cpp" / "build" / "benchmark_cpp"
    if exe.is_file():
        return True
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] building cpp benchmark\n")
        log.flush()
        configure = ["cmake", "-B", "build", "-S", "."]
        build = ["cmake", "--build", "build", "-j", env.get("WEB_COMPRESSION_CPP_BUILD_JOBS", "16")]
        r1 = subprocess.run(configure, cwd=ROOT / "cpp", env=env, stdout=log, stderr=log)
        if r1.returncode != 0:
            log.write(f"[{now()}] cpp configure failed: {r1.returncode}\n")
            return False
        r2 = subprocess.run(build, cwd=ROOT / "cpp", env=env, stdout=log, stderr=log)
        if r2.returncode != 0:
            log.write(f"[{now()}] cpp build failed: {r2.returncode}\n")
            return False
    return exe.is_file()


def run_chunk(
    *,
    language: str,
    dataset: str,
    dtype: str,
    algorithms: list[str],
    env: dict[str, str],
    run_log: Path,
) -> dict:
    before = read_algorithms(report_path(dataset, dtype, language))
    cmd = command_for(language, dataset, dtype, algorithms)
    started = now()
    with run_log.open("a", encoding="utf-8") as log:
        log.write(f"\n[{started}] RUN {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=log)
    after = read_algorithms(report_path(dataset, dtype, language))
    gained = sorted(after - before)
    still_missing = sorted(set(algorithms) - after)
    return {
        "time": now(),
        "language": language,
        "dataset": dataset,
        "dtype": dtype,
        "algorithms": algorithms,
        "returncode": proc.returncode,
        "added": gained,
        "still_missing": still_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--languages", default="java,python,cpp")
    parser.add_argument("--dtypes", default="int,float")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_path = LOG_ROOT / f"missing_bench_state_{stamp}.jsonl"
    run_log = LOG_ROOT / f"missing_bench_run_{stamp}.log"
    summary_path = LOG_ROOT / f"missing_bench_summary_{stamp}.json"
    env = base_env()

    targets = target_algorithms()
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    dtypes = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    cpp_available = True
    if "cpp" in languages and not args.dry_run:
        cpp_available = ensure_cpp_build(env, run_log)

    summary = {
        "started": now(),
        "target_algorithm_count": len(targets),
        "targets": targets,
        "datasets": datasets,
        "languages": languages,
        "dtypes": dtypes,
        "cpp_available": cpp_available,
        "planned_chunks": 0,
        "skipped": [],
    }

    with state_path.open("a", encoding="utf-8") as state:
        for dataset in datasets:
            if not dataset_csv_exists(dataset):
                item = {"time": now(), "event": "skip_dataset_no_source_csv", "dataset": dataset}
                summary["skipped"].append(item)
                state.write(json.dumps(item, ensure_ascii=False) + "\n")
                state.flush()
                continue
            for dtype in dtypes:
                if not dtype_has_columns(dataset, dtype):
                    item = {"time": now(), "event": "skip_dtype_no_columns", "dataset": dataset, "dtype": dtype}
                    summary["skipped"].append(item)
                    state.write(json.dumps(item, ensure_ascii=False) + "\n")
                    state.flush()
                    continue
                for language in languages:
                    if language == "cpp" and not cpp_available:
                        item = {"time": now(), "event": "skip_cpp_build_unavailable", "dataset": dataset, "dtype": dtype}
                        state.write(json.dumps(item, ensure_ascii=False) + "\n")
                        state.flush()
                        continue
                    existing = read_algorithms(report_path(dataset, dtype, language))
                    missing = [alg for alg in targets if alg not in existing]
                    if not missing:
                        item = {
                            "time": now(),
                            "event": "skip_no_missing",
                            "dataset": dataset,
                            "dtype": dtype,
                            "language": language,
                        }
                        state.write(json.dumps(item, ensure_ascii=False) + "\n")
                        state.flush()
                        continue
                    if args.dry_run:
                        item = {
                            "time": now(),
                            "event": "plan_missing",
                            "dataset": dataset,
                            "dtype": dtype,
                            "language": language,
                            "missing_count": len(missing),
                            "chunk_count": len(chunks(missing, max(1, args.chunk_size))),
                        }
                        state.write(json.dumps(item, ensure_ascii=False) + "\n")
                        state.flush()
                        summary["planned_chunks"] += item["chunk_count"]
                        continue
                    for chunk in chunks(missing, max(1, args.chunk_size)):
                        summary["planned_chunks"] += 1
                        result = run_chunk(
                            language=language,
                            dataset=dataset,
                            dtype=dtype,
                            algorithms=chunk,
                            env=env,
                            run_log=run_log,
                        )
                        state.write(json.dumps(result, ensure_ascii=False) + "\n")
                        state.flush()
                        if result["returncode"] != 0 and len(chunk) > 1:
                            for alg in chunk:
                                retry = run_chunk(
                                    language=language,
                                    dataset=dataset,
                                    dtype=dtype,
                                    algorithms=[alg],
                                    env=env,
                                    run_log=run_log,
                                )
                                retry["retry_after_chunk_failure"] = True
                                state.write(json.dumps(retry, ensure_ascii=False) + "\n")
                                state.flush()

    summary["finished"] = now()
    summary["state_path"] = str(state_path)
    summary["run_log"] = str(run_log)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
