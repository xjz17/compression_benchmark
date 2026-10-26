"""
Merge a small set of Java-emitted float rows into the Python benchmark CSV.

**What gets merged (whitelist in `_supplement_whitelist`):** ``elf_float`` (legacy
Elf+ table label), keys starting with ``sz_`` / ``sz3_`` / ``serf_``, and
``buff_scalar`` / ``buff_simd`` / ``buff_kernel_native`` (BUFF when Python did not emit them).
The supplement runs only for algorithms Python did not already produce.

**Why Camel / Sim-Piece / Chimp / full Elf stack are not on this list:** those rows
are implemented in Python modules (e.g. ``chimp32_python.py``, ``elf32_python.py``,
``camel_python.py``, ``simpiece_python.py``) or use native CLIs; they are not appended
via ``org.example.Main`` here. **SZ / SZ3 / Serf:** when ``sz`` / ``sz3`` / ``sfz``
binaries exist, ``native_research_cli_python.py`` usually emits the same keys; if not,
this module can still pull the matching rows from Java Main for the whitelisted prefixes
(including BUFF rows when ``comp_profiler`` is built but the Python bench skipped them).

Env:
  WEB_COMPRESSION_PYTHON_JAVA_SUPPLEMENT — set to 0/false/no to skip (default: on).
  User-upload pipeline sets this to 0 (Java already ran for that dtype).
  SERF/BUFF follow Django benchmark defaults (Serf/BUFF off unless *_BENCH=1).
  SZ/SZ3 on by default when ``sz`` / ``sz3`` binaries exist (set ``WEB_COMPRESSION_SZ_BENCH=0`` or ``SZ3_BENCH=0`` to skip).
  WEB_COMPRESSION_JAVA_CLASSPATH_APPEND — same as Django upload (`views.py`).
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
try:
    from java_bench_common import (  # type: ignore
        JAVA_BENCH_CWD,
        ROOT,
        _buff_env,
        _serf_env,
        java_jvm_prefix,
        resolve_benchmark_io_path,
    )
except ImportError:
    _buff_env = _serf_env = None  # type: ignore

    def java_jvm_prefix():  # type: ignore
        _raw = os.environ.get("WEB_COMPRESSION_JAVA_OPTS")
        if _raw is None:
            jvm_opts = "-Xmx8g"
        else:
            jvm_opts = _raw.strip()
        return shlex.split(jvm_opts) if jvm_opts else []

logger = logging.getLogger(__name__)

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_BACKEND_ROOT, ".."))

# Keep in sync with views._BENCHMARK_ENV_DEFAULTS (avoid importing views / Django).
_BENCHMARK_ENV_DEFAULTS = {
    "WEB_COMPRESSION_CMIX_BENCH": "0",
    "WEB_COMPRESSION_CMIX_NO_PREPROCESS": "1",
    "WEB_COMPRESSION_CMIX_MAX_COLUMN_BYTES": str(64 * 1024),
    "WEB_COMPRESSION_CMIX_TIMEOUT_SEC": "180",
    "WEB_COMPRESSION_MSDZIP_BENCH": "0",
    "WEB_COMPRESSION_MSDLC_BENCH": "0",
    "WEB_COMPRESSION_SERF_BENCH": "0",
    "WEB_COMPRESSION_SZ_BENCH": "1",
    "WEB_COMPRESSION_SZ3_BENCH": "1",
    "WEB_COMPRESSION_BUFF_BENCH": "0",
    "WEB_COMPRESSION_BUFF_KERNEL_NATIVE": "1",
    "WEB_COMPRESSION_FPZIP_BENCH": "1",
    "WEB_COMPRESSION_STATICDATA_REGEN": "1",
}


def _apply_benchmark_env_defaults(env: dict[str, str] | None = None) -> dict[str, str]:
    out = dict(os.environ if env is None else env)
    for key, value in _BENCHMARK_ENV_DEFAULTS.items():
        out.setdefault(key, value)
    for key in list(out.keys()):
        if key.startswith("WEB_COMPRESSION_CPP_NATIVE"):
            del out[key]
    return out


def _java_benchmark_command(jar_path: str, *main_args: str) -> List[str]:
    from .bench_executable_paths import iter_java_classpath_append_files, repo_root as _bench_repo_root

    parts = [os.path.abspath(jar_path)]
    extra = os.environ.get("WEB_COMPRESSION_JAVA_CLASSPATH_APPEND", "").strip()
    if extra:
        parts.extend(p for p in extra.split(os.pathsep) if p.strip())
    for append_file in iter_java_classpath_append_files(Path(parts[0])):
        with open(append_file, encoding="utf-8") as af:
            for line in af:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts.append(line)
    repo_root = str(_bench_repo_root())
    elf_jar = os.path.join(
        repo_root, "external", "compression", "elf", "target", "start-compress-1.0-SNAPSHOT.jar"
    )
    if os.path.isfile(elf_jar):
        parts.append(elf_jar)
    camel_jar = os.path.join(
        repo_root, "external", "compression", "camel", "target", "start-compress-1.0-SNAPSHOT.jar"
    )
    camel_dep = os.path.join(
        repo_root, "external", "compression", "camel", "target", "dependency"
    )
    if os.path.isfile(camel_jar):
        parts.append(camel_jar)
        if os.path.isdir(camel_dep):
            for name in sorted(os.listdir(camel_dep)):
                if name.endswith(".jar"):
                    parts.append(os.path.join(camel_dep, name))
    return ["java", *java_jvm_prefix(), "-cp", os.pathsep.join(parts), "org.example.Main", *main_args]


def _supplement_whitelist(algorithm: str) -> bool:
    a = str(algorithm).strip()
    if a in {"elf_float", "elf_on_chimp"}:
        return True
    if a.startswith("sz_") or a.startswith("sz3_"):
        return True
    if a.startswith("sz_float_") or a.startswith("sz3_float_"):
        return True
    if a.startswith("serf_"):
        return True
    if a in {
        "buff_scalar",
        "buff_simd",
        "buff_kernel_native",
        "buff_float_scalar",
        "buff_float_simd",
        "buff_float_kernel_native",
    }:
        return True
    return False


def merge_java_float_research_rows_into_records(
    existing_algorithms: Set[str],
    processed_csv_path: str,
    output_csv_path: str,
) -> None:
    """Append whitelisted Java-only float rows not already in ``output_csv_path``."""
    from .compression_report_io import append_compression_report_row

    flag = os.environ.get("WEB_COMPRESSION_PYTHON_JAVA_SUPPLEMENT", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return

    from .bench_executable_paths import benchmark_java_jar_path

    jar_path = benchmark_java_jar_path()
    if not jar_path.is_file():
        logger.info("Java supplement skipped: missing %s", jar_path)
        return

    jar = str(jar_path)

    have: Set[str] = set(existing_algorithms)

    fd, tmp_path = tempfile.mkstemp(suffix="_java_supplement.csv", text=True)
    os.close(fd)
    try:
        in_csv = str(resolve_benchmark_io_path(processed_csv_path))
        out_csv = str(resolve_benchmark_io_path(tmp_path))
        cmd = _java_benchmark_command(
            jar,
            "--input",
            in_csv,
            "--data-type",
            "float",
            "--output-report",
            out_csv,
        )
        env = _apply_benchmark_env_defaults(os.environ.copy())
        env.setdefault("WEB_COMPRESSION_REPO_ROOT", str(ROOT.resolve()))
        env["WEB_COMPRESSION_SKIP_ALGORITHMS"] = ",".join(sorted(have))
        # Same defaults as staticdata regen: Serf/SZ/BUFF off unless explicitly enabled.
        env.setdefault("WEB_COMPRESSION_SERF_BENCH", "0")
        env.setdefault("WEB_COMPRESSION_SZ_BENCH", "1")
        env.setdefault("WEB_COMPRESSION_SZ3_BENCH", "1")
        env.setdefault("WEB_COMPRESSION_BUFF_BENCH", "0")
        env.setdefault("WEB_COMPRESSION_CMIX_BENCH", "0")
        proc = subprocess.run(
            cmd,
            cwd=str(JAVA_BENCH_CWD),
            env=env,
            capture_output=True,
            text=True,
            timeout=None,
        )
        if proc.returncode != 0:
            logger.warning(
                "Java float supplement: subprocess failed (code %s). stderr: %s",
                proc.returncode,
                (proc.stderr or "")[:800],
            )
            return
        try:
            df = pd.read_csv(tmp_path)
        except Exception as ex:
            logger.warning("Java float supplement: could not read %s: %s", tmp_path, ex)
            return

        if "algorithm" not in df.columns:
            return

        added = 0
        for _, row in df.iterrows():
            algo = row.get("algorithm")
            if algo is None or pd.isna(algo):
                continue
            key = str(algo).strip()
            if not _supplement_whitelist(key):
                continue
            if key in have:
                continue
            rec = {
                "algorithm": key,
                "originalSize": int(row["originalSize"]),
                "compressedSize": int(row["compressedSize"]),
                "compressionTime": int(row["compressionTime"]),
                "decompressionTime": int(row["decompressionTime"]),
            }
            append_compression_report_row(output_csv_path, rec)
            have.add(key)
            existing_algorithms.add(key)
            added += 1
        if added:
            logger.info("Java float supplement: merged %s research rows into Python report", added)
    except FileNotFoundError:
        logger.info("Java float supplement skipped: java not on PATH")
    except Exception as ex:
        logger.warning("Java float supplement failed: %s", ex)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
