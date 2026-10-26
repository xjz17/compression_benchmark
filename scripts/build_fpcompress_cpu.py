#!/usr/bin/env python3
"""
Build FPCompress CPU compressors/decompressors only (DPratio, DPspeed).

Sources are read from the upstream checkout (submodule/clone); **binaries are written
outside that tree** so submodule git stays clean::

    external/runtime/fpcompress/double_src/bin/

On Linux/macOS binaries are extensionless; on Windows they are ``*.exe``.

Skips GPU (nvcc) targets and single-precision (SPratio/SPspeed) from upstream ``compile.py``.
Requires ``g++`` with OpenMP (typically MinGW on Windows). On Linux/macOS, GCC/libstdc++
PSTL (``std::execution::par_unseq`` in upstream sources) also needs **Intel oneTBB**
linked (``-ltbb``); the script adds this automatically when ``pkg-config tbb`` or
``libtbb`` is available.

Override source root: ``WEB_COMPRESSION_FPCOMPRESS_UPSTREAM_ROOT``
Override output root: ``WEB_COMPRESSION_FPCOMPRESS_BIN_ROOT``
Override extra link flags: ``WEB_COMPRESSION_FPCOMPRESS_LDFLAGS`` (space-separated)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from myapp.upstream_repo_paths import fpcompress_bin_root, fpcompress_upstream_root  # noqa: E402


def exe_suffix() -> str:
    return ".exe" if sys.platform.startswith("win") else ""


def gxx() -> str:
    return os.environ.get("WEB_COMPRESSION_FPCOMPRESS_GXX", "g++")


def _split_link_flags(raw: str) -> list[str]:
    return [part for part in raw.split() if part]


def tbb_link_flags() -> list[str]:
    """PSTL in libstdc++ (GCC) uses oneTBB; upstream FPCompress needs it at link time."""
    override = os.environ.get("WEB_COMPRESSION_FPCOMPRESS_LDFLAGS", "").strip()
    if override:
        return _split_link_flags(override)

    try:
        proc = subprocess.run(
            ["pkg-config", "--libs", "tbb"],
            check=True,
            capture_output=True,
            text=True,
        )
        flags = _split_link_flags(proc.stdout.strip())
        if flags:
            return flags
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Common on Debian/Ubuntu/macOS Homebrew when pkg-config metadata is missing.
    return ["-ltbb"]


def compile_one(
    out_path: Path,
    source: Path,
    *,
    include_dirs: list[Path],
    link_flags: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gxx(), "-O3", "-march=native", "-fopenmp", "-std=c++17"]
    for inc in include_dirs:
        cmd.extend(["-I", str(inc)])
    cmd.extend(["-o", str(out_path), str(source)])
    cmd.extend(link_flags)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_double_tree(
    upstream_src: Path,
    *,
    upstream: Path,
    bin_root: Path,
) -> None:
    bin_dir = bin_root / "double_src" / "bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    bin_dir.mkdir(parents=True)

    include_dirs = [upstream_src, upstream]
    link_flags = tbb_link_flags()
    suffix = exe_suffix()
    specs = (
        ("ratio-cpu-compress", "ratio-compressor-double.cpp"),
        ("ratio-cpu-decompress", "ratio-decompressor-double.cpp"),
        ("speed-cpu-compress", "speed-compressor-double.cpp"),
        ("speed-cpu-decompress", "speed-decompressor-double.cpp"),
    )
    for stem, src_name in specs:
        compile_one(
            bin_dir / f"{stem}{suffix}",
            upstream_src / src_name,
            include_dirs=include_dirs,
            link_flags=link_flags,
        )


def main() -> None:
    upstream = fpcompress_upstream_root()
    if not upstream.is_dir():
        raise SystemExit(f"FPCompress upstream not found: {upstream}")

    bin_root = fpcompress_bin_root()
    t0 = time.time()
    build_double_tree(upstream / "double_src", upstream=upstream, bin_root=bin_root)
    elapsed = time.time() - t0
    print(f"FPCompress CPU build finished in {elapsed:.1f}s")
    print(f"  {bin_root / 'double_src' / 'bin'}")


if __name__ == "__main__":
    main()
