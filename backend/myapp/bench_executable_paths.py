"""Resolve benchmark Java JAR and C++ driver paths (build tree first, then ``backend/lib/``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_JAVA_JAR = "benchmark-java-1.0-SNAPSHOT.jar"
_CPP_BIN = "benchmark_cpp"


def repo_root() -> Path:
    return _REPO_ROOT


def backend_root() -> Path:
    return _BACKEND


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _first_existing_file(path: Path) -> Path | None:
    """Return ``path`` if it is a regular file; on Windows also try ``path`` + ``.exe``."""
    if path.is_file():
        return path.resolve()
    if _is_windows():
        name = path.name
        if name and not name.lower().endswith(".exe"):
            with_exe = path.with_name(f"{name}.exe")
            if with_exe.is_file():
                return with_exe.resolve()
    return None


def _cpp_build_candidates() -> list[Path]:
    """Search order: ``cpp/build`` (and MSVC configs), then ``backend/lib``."""
    rels = (
        f"cpp/build/{_CPP_BIN}",
        f"cpp/build/Release/{_CPP_BIN}",
        f"cpp/build/Debug/{_CPP_BIN}",
        f"backend/lib/{_CPP_BIN}",
    )
    return [_REPO_ROOT / rel for rel in rels]


def benchmark_java_jar_path() -> Path:
    """Prefer ``java/target/``, then ``backend/lib/``; honor ``WEB_COMPRESSION_BENCHMARK_JAR``."""
    override = os.environ.get("WEB_COMPRESSION_BENCHMARK_JAR", "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return p.resolve()
    for cand in (
        _REPO_ROOT / "java" / "target" / _JAVA_JAR,
        _BACKEND / "lib" / _JAVA_JAR,
    ):
        if cand.is_file():
            return cand.resolve()
    return (_BACKEND / "lib" / _JAVA_JAR).resolve()


def require_benchmark_java_jar() -> Path:
    jar = benchmark_java_jar_path()
    if not jar.is_file():
        raise FileNotFoundError(
            "Missing Java benchmark JAR. Build with: "
            f"cd {_REPO_ROOT / 'java'} && mvn package -DskipTests "
            f"(expected { _REPO_ROOT / 'java' / 'target' / _JAVA_JAR } "
            f"or {_BACKEND / 'lib' / _JAVA_JAR})"
        )
    return jar


def benchmark_cpp_executable_path() -> Path | None:
    """Prefer ``cpp/build/`` (incl. ``benchmark_cpp.exe`` on Windows), then ``backend/lib/``.

    Override with ``WEB_COMPRESSION_BENCHMARK_CPP`` (with or without ``.exe`` on Windows).
    Copying into ``backend/lib/`` is optional for deployment; local dev can run from ``cpp/build``.
    """
    override = os.environ.get("WEB_COMPRESSION_BENCHMARK_CPP", "").strip()
    if override:
        hit = _first_existing_file(Path(override))
        if hit is not None:
            return hit
    for cand in _cpp_build_candidates():
        hit = _first_existing_file(cand)
        if hit is not None:
            return hit
    return None


def require_benchmark_cpp_executable() -> Path:
    exe = benchmark_cpp_executable_path()
    if exe is None:
        win_hint = (
            f" (on Windows the binary is often {_REPO_ROOT / 'cpp' / 'build' / (_CPP_BIN + '.exe')})"
            if _is_windows()
            else ""
        )
        raise FileNotFoundError(
            "benchmark_cpp not found. Build: "
            f"cd {_REPO_ROOT / 'cpp'} && cmake -B build -S . && cmake --build build "
            f"{win_hint}; optional copy to {_BACKEND / 'lib' / _CPP_BIN}, "
            "or set WEB_COMPRESSION_BENCHMARK_CPP to the executable path."
        )
    return exe


def lzbench_executable_path() -> Path | None:
    """``external/compression/lzbench/lzbench`` (or ``.exe`` on Windows).

    Override with ``WEB_COMPRESSION_LZBENCH``.
    """
    override = os.environ.get("WEB_COMPRESSION_LZBENCH", "").strip()
    if override:
        hit = _first_existing_file(Path(override))
        if hit is not None:
            return hit
    for cand in (
        _REPO_ROOT / "external" / "compression" / "lzbench" / "lzbench",
        _BACKEND / "lib" / "lzbench",
    ):
        hit = _first_existing_file(cand)
        if hit is not None:
            return hit
    return None


def require_lzbench_executable() -> Path:
    exe = lzbench_executable_path()
    if exe is None:
        win_hint = (
            f" (expected {_REPO_ROOT / 'external' / 'compression' / 'lzbench' / 'lzbench.exe'})"
            if _is_windows()
            else ""
        )
        raise FileNotFoundError(
            "lzbench not found. Build: python scripts/build_lzbench.py"
            f"{win_hint}, or set WEB_COMPRESSION_LZBENCH."
        )
    return exe


def zfp_executable_path() -> Path | None:
    """``zfp`` CLI from ``external/compression/zfp/build/`` (standalone cmake). Override with ``WEB_COMPRESSION_ZFP``."""
    override = os.environ.get("WEB_COMPRESSION_ZFP", "").strip()
    if override:
        hit = _first_existing_file(Path(override))
        if hit is not None:
            return hit
    for cand in (
        _REPO_ROOT / "external" / "compression" / "zfp" / "build" / "bin" / "zfp",
        _REPO_ROOT / "external" / "compression" / "zfp" / "build" / "Release" / "zfp",
        _REPO_ROOT / "cpp" / "build" / "ext_zfp" / "bin" / "zfp",
        _REPO_ROOT / "cpp" / "build" / "ext_zfp" / "Release" / "zfp",
    ):
        hit = _first_existing_file(cand)
        if hit is not None:
            return hit
    return None


def iter_java_classpath_append_files(jar_path: Path | None = None) -> list[Path]:
    """``backend/lib/java_classpath.append`` first, then next to the resolved JAR."""
    seen: set[Path] = set()
    out: list[Path] = []
    for cand in (
        _BACKEND / "lib" / "java_classpath.append",
        (jar_path or benchmark_java_jar_path()).parent / "java_classpath.append",
    ):
        resolved = cand.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            out.append(resolved)
    return out
