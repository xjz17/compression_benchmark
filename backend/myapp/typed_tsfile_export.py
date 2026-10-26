"""Typed TsFile export — CSV in ``dataset_csv/``, info in ``dataset/``, TsFile in ``dataset_tsfile/``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def catalog_typed_tsfile_paths(
    key: str,
    *,
    tsfile_bundle_dir: str | Path,
    dataset_bundle_dir: str | Path,
) -> dict[str, Path]:
    """Catalog layout: prebuilt under ``dataset_tsfile/<key>/``; info CSV under ``dataset/<key>/``."""
    tsfile_root = Path(tsfile_bundle_dir)
    dataset_root = Path(dataset_bundle_dir)
    return {
        "info_csv": dataset_root / f"{key}_catalog_tsfile_info.csv",
        "prebuilt": tsfile_root / f"{key}.tsfile",
    }


def run_java_selected_tsfile_export(
    csv_path: str,
    column_info_path: str,
    *,
    output_tsfile_path: str,
    info_csv_path: str,
    dataset_key: str | None = None,
    int_encoding: str | None = None,
    float_encoding: str | None = None,
    string_encoding: str | None = None,
) -> None:
    """Run ``--typed-tsfile-export`` (table model, fixed encodings; legacy args ignored)."""
    from .bench_executable_paths import require_benchmark_java_jar, repo_root
    from .catalog_tsfile_selection import (
        CATALOG_TSFILE_FLOAT_ENCODING,
        CATALOG_TSFILE_INT_ENCODING,
        CATALOG_TSFILE_STRING_ENCODING,
    )

    _scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from java_bench_common import (  # noqa: E402
        JAVA_BENCH_CWD,
        _buff_env,
        _serf_env,
        java_cp,
        java_jvm_prefix,
        resolve_benchmark_io_path,
    )

    if not require_benchmark_java_jar().is_file():
        raise FileNotFoundError("Missing Java benchmark JAR for typed TsFile export")

    info_abs = resolve_benchmark_io_path(column_info_path)
    if info_abs.name.endswith("_column_info.csv"):
        column_flag = "--column-info"
    else:
        column_flag = "--column-types-manifest"

    output_path = resolve_benchmark_io_path(output_tsfile_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = resolve_benchmark_io_path(info_csv_path)
    info_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java",
        *java_jvm_prefix(),
        "-cp",
        java_cp(),
        "org.example.Main",
        "--typed-tsfile-export",
        "--input",
        str(resolve_benchmark_io_path(csv_path)),
        column_flag,
        str(info_abs),
        "--typed-tsfile-output",
        str(output_path),
        "--typed-tsfile-info-csv",
        str(info_path),
        "--typed-tsfile-int-encoding",
        str(int_encoding or CATALOG_TSFILE_INT_ENCODING),
        "--typed-tsfile-float-encoding",
        str(float_encoding or CATALOG_TSFILE_FLOAT_ENCODING),
        "--typed-tsfile-string-encoding",
        str(string_encoding or CATALOG_TSFILE_STRING_ENCODING),
    ]

    env = _buff_env(_serf_env(dict(os.environ)))
    env.setdefault("WEB_COMPRESSION_REPO_ROOT", str(repo_root().resolve()))
    env.setdefault("WEB_COMPRESSION_PYTHON", sys.executable)
    subprocess.run(cmd, check=True, timeout=None, cwd=str(JAVA_BENCH_CWD), env=env)


def file_size_bytes_from_catalog_tsfile_info(info_csv_path: str) -> int:
    from .catalog_tsfile_selection import file_size_bytes_from_catalog_tsfile_info as _read

    return _read(info_csv_path)


def min_file_size_bytes_from_encoding_summary(encoding_summary_path: str) -> int:
    """Legacy alias: read ``file_size_bytes`` from ``*_catalog_tsfile_info.csv``."""
    return file_size_bytes_from_catalog_tsfile_info(encoding_summary_path)
