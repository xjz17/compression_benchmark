"""Elf elf_on_chimp / elf_plus on IEEE-754 double via submodule JAR (Elf64Utils), not float32 ports."""
from __future__ import annotations

import io
import os
import struct
import subprocess
import time
from typing import List, Tuple

from .benchmark_wire import column_original_size


def _repo_root() -> str:
    root = os.environ.get("COMPRESSION_BENCHMARK_REPO_ROOT", "").strip()
    if root:
        return root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _elf_jar() -> str | None:
    override = os.environ.get("COMPRESSION_BENCHMARK_ELF_JAR", "").strip()
    if override and os.path.isfile(override):
        return override
    root = _repo_root()
    for rel in (
        "external/compression/elf/target/start-compress-1.0-SNAPSHOT.jar",
        "backend/lib/start-compress-1.0-SNAPSHOT.jar",
    ):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def _benchmark_jar() -> str | None:
    from .bench_executable_paths import benchmark_java_jar_path

    jar = benchmark_java_jar_path()
    return str(jar) if jar.is_file() else None


def _elf_java_classpath(elf_jar: str, bench_jar: str) -> str:
    parts = [bench_jar, elf_jar]
    dep_dir = os.path.join(os.path.dirname(elf_jar), "dependency")
    if os.path.isdir(dep_dir):
        for name in sorted(os.listdir(dep_dir)):
            if name.endswith(".jar"):
                parts.append(os.path.join(dep_dir, name))
    return os.pathsep.join(parts)


def _bench_elf_double_columns(columns: List[List[float]], mode: str) -> Tuple[int, int, int, int]:
    root = _repo_root()
    elf_jar = _elf_jar()
    bench_jar = _benchmark_jar()
    if not elf_jar or not bench_jar:
        return 0, 0, 0, 0
    payload = io.BytesIO()
    payload.write(struct.pack("<i", len(columns)))
    for col in columns:
        payload.write(struct.pack("<i", len(col)))
        for v in col:
            payload.write(struct.pack("<d", float(v)))
    cp = _elf_java_classpath(elf_jar, bench_jar)
    env = os.environ.copy()
    env.setdefault("COMPRESSION_BENCHMARK_REPO_ROOT", root)
    try:
        proc = subprocess.run(
            ["java", "-cp", cp, "org.example.ElfDoubleBatchMain", mode],
            input=payload.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=600,
            check=False,
            cwd=root,
            env=env,
        )
    except (subprocess.SubprocessError, OSError):
        return 0, 0, 0, 0
    if proc.returncode != 0 or len(proc.stdout) < 32:
        return 0, 0, 0, 0
    o, c, te, td = struct.unpack("<qqqq", proc.stdout[:32])
    if o <= 0:
        return 0, 0, 0, 0
    return int(o), int(c), int(te), int(td)


def bench_elf_on_chimp_double_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    return _bench_elf_double_columns(columns, "elf_on_chimp")


def bench_elf_plus_double_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    return _bench_elf_double_columns(columns, "elf_plus")
