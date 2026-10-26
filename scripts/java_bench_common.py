"""
Shared Java benchmark CLI (``org.example.Main``) used by:

- ``regenerate_staticdata_java.py``
- ``scripts/cpp_java_companion.py`` (IoTDB + research JAR rows appended by native ``benchmark_cpp``)

``benchmark_cpp`` invokes ``cpp_java_companion.py`` via ``COMPRESSION_BENCHMARK_PYTHON`` when set; otherwise it
prefers ``python3`` on PATH and falls back to ``python``. ``regenerate_staticdata_cpp.py`` sets
``COMPRESSION_BENCHMARK_PYTHON`` to ``sys.executable`` so the active venv/pyenv interpreter is used.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from myapp.bench_executable_paths import (  # noqa: E402
    benchmark_cpp_executable_path,
    benchmark_java_jar_path,
    iter_java_classpath_append_files,
    require_benchmark_cpp_executable,
)

# Java subprocess cwd: repo root (same anchor as resolve_benchmark_io_path for relative paths).
JAVA_BENCH_CWD = ROOT


def resolve_benchmark_io_path(path: str | Path) -> Path:
    """Canonical absolute path for benchmark I/O (anchored at repo root when relative)."""
    p = Path(path)
    if not p.is_absolute():
        return (ROOT / p).resolve()
    return p.resolve()


def benchmark_jar_path() -> Path:
    """``org.example.Main`` JAR — prefers ``java/target/``, then ``backend/lib/``."""
    return benchmark_java_jar_path()


JAR = benchmark_jar_path()


def benchmark_cpp_path() -> Path:
    """Native ``benchmark_cpp`` — prefers ``cpp/build/``, then ``backend/lib/``."""
    exe = benchmark_cpp_executable_path()
    if exe is not None:
        return exe
    fallback = BACKEND / "lib" / "benchmark_cpp"
    if fallback.is_file():
        return fallback.resolve()
    if sys.platform.startswith("win"):
        with_exe = fallback.with_suffix(".exe")
        if with_exe.is_file():
            return with_exe.resolve()
    return fallback.resolve()


def java_jvm_prefix() -> list[str]:
    """JVM flags before ``-cp`` (default ``-Xmx8g`` when env is unset)."""
    _raw = os.environ.get("COMPRESSION_BENCHMARK_JAVA_OPTS")
    if _raw is None:
        jvm_opts = "-Xmx8g"
    else:
        jvm_opts = _raw.strip()
    return shlex.split(jvm_opts) if jvm_opts else []


def java_cp() -> str:
    parts = [str(JAR.resolve())]
    extra = os.environ.get("COMPRESSION_BENCHMARK_JAVA_CLASSPATH_APPEND", "").strip()
    if extra:
        parts.extend(p for p in extra.split(os.pathsep) if p.strip())
    for append_file in iter_java_classpath_append_files(JAR):
        for line in append_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts.append(line)
    elf_jar = (
        ROOT / "external" / "compression" / "elf" / "target" / "start-compress-1.0-SNAPSHOT.jar"
    )
    if elf_jar.is_file():
        parts.append(str(elf_jar.resolve()))
        elf_dep = elf_jar.parent / "dependency"
        if elf_dep.is_dir():
            for name in sorted(elf_dep.iterdir()):
                if name.suffix == ".jar":
                    parts.append(str(name.resolve()))
    camel_jar = (
        ROOT / "external" / "compression" / "camel" / "target" / "start-compress-1.0-SNAPSHOT.jar"
    )
    camel_dep = ROOT / "external" / "compression" / "camel" / "target" / "dependency"
    if camel_jar.is_file():
        parts.append(str(camel_jar.resolve()))
        if camel_dep.is_dir():
            for name in sorted(camel_dep.iterdir()):
                if name.suffix == ".jar":
                    parts.append(str(name.resolve()))
    return os.pathsep.join(parts)


def _staticdata_regen_mode(env: dict) -> bool:
    return env.get("COMPRESSION_BENCHMARK_STATICDATA_REGEN", "").strip() == "1"


def _serf_env(base: dict | None) -> dict:
    env = dict(os.environ if base is None else base)
    if _staticdata_regen_mode(env):
        return env
    if "COMPRESSION_BENCHMARK_SERF_BENCH" not in env:
        sfz = ROOT / "external" / "compression" / "Serf" / "build" / "serf_cmd" / "sfz"
        env["COMPRESSION_BENCHMARK_SERF_BENCH"] = (
            "1" if sfz.is_file() and os.access(sfz, os.X_OK) else "0"
        )
    return env


def _buff_env(base: dict | None) -> dict:
    env = dict(os.environ if base is None else base)
    env.setdefault("COMPRESSION_BENCHMARK_BUFF_KERNEL_NATIVE", "1")
    if _staticdata_regen_mode(env):
        return env
    if "COMPRESSION_BENCHMARK_BUFF_BENCH" not in env:
        on = False
        for rel in (
            "external/compression/buff/database/target/release/comp_profiler",
            "external/compression/buff/database/target/debug/comp_profiler",
        ):
            cand = ROOT / rel
            if cand.is_file() and os.access(cand, os.X_OK):
                on = True
                break
        env["COMPRESSION_BENCHMARK_BUFF_BENCH"] = "1" if on else "0"
    return env


def _fpcompress_env(base: dict | None) -> dict:
    env = dict(os.environ if base is None else base)
    env.setdefault(
        "COMPRESSION_BENCHMARK_FPCOMPRESS_BIN_ROOT",
        str((ROOT / "external" / "runtime" / "fpcompress").resolve()),
    )
    env.setdefault("COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH", "0")
    return env


def _apply_partial_algorithm_external_benches(
    env: dict[str, str], algorithms: list[str] | None
) -> dict[str, str]:
    """Turn off expensive external CLI benches when partial refresh does not request them."""
    if not algorithms:
        return env
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from myapp.lzbench_codec_catalog import is_lzbench_catalog_algorithm

    names = [a.strip() for a in algorithms if a and str(a).strip()]
    env["COMPRESSION_BENCHMARK_LZBENCH_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_SZ_BENCH"] = "1" if any(a.startswith("sz_") for a in names) else "0"
    env["COMPRESSION_BENCHMARK_SZ3_BENCH"] = "1" if any(a.startswith("sz3_") for a in names) else "0"
    env["COMPRESSION_BENCHMARK_FPZIP_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_BLOSC_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_BLOSC2_BENCH"] = "0"
    env["COMPRESSION_BENCHMARK_ZFP_BENCH"] = "1" if any(a.startswith("ext_zfp_") for a in names) else "0"
    return env


def run_java_benchmark(
    input_csv: Path,
    data_type: str,
    output_csv: Path,
    *,
    column_info: Path | None = None,
    column_index: int | None = None,
    algorithms: list[str] | None = None,
) -> None:
    """Run the same entrypoint as Django ``_java_benchmark_command`` / staticdata Java.

    Default JVM heap is **8g** when ``COMPRESSION_BENCHMARK_JAVA_OPTS`` is **unset** (very large
    catalog CSVs can OOM with the JVM default). Override or disable extra flags with::

        export COMPRESSION_BENCHMARK_JAVA_OPTS='-Xmx16g'
        export COMPRESSION_BENCHMARK_JAVA_OPTS=''   # JVM default heap only

    All I/O paths are made absolute against the **repo root** before spawning Java;
    subprocess ``cwd`` is the repo root (not ``backend/``), so relative CLI paths are
    not double-prefixed by mistake.
    """
    input_csv = resolve_benchmark_io_path(input_csv)
    output_csv = resolve_benchmark_io_path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cp = java_cp()
    jvm_prefix = java_jvm_prefix()
    cmd = [
        "java",
        *jvm_prefix,
        "-cp",
        cp,
        "org.example.Main",
        "--input",
        str(input_csv),
        "--data-type",
        data_type,
        "--output-report",
        str(output_csv),
    ]
    if column_info is not None:
        col = resolve_benchmark_io_path(column_info)
        if col.is_file():
            cmd.extend(["--column-info", str(col)])
    if column_index is not None:
        cmd.extend(["--column-index", str(int(column_index))])
    env = _fpcompress_env(_buff_env(_serf_env(dict(os.environ))))
    if algorithms:
        algo_csv = ",".join(a.strip() for a in algorithms if a and str(a).strip())
        if algo_csv:
            env["COMPRESSION_BENCHMARK_RUN_ALGORITHMS"] = algo_csv
            cmd.extend(["--algorithms", algo_csv])
            env = _apply_partial_algorithm_external_benches(env, algorithms)
    env.setdefault("COMPRESSION_BENCHMARK_REPO_ROOT", str(ROOT.resolve()))
    env.setdefault("COMPRESSION_BENCHMARK_PYTHON", sys.executable)
    subprocess.run(cmd, check=True, cwd=str(JAVA_BENCH_CWD), env=env)
