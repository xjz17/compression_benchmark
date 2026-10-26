"""Shared build logic for Blosc1/Blosc2 column CLI bench tools (Java/C++ subprocess)."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exe_suffix() -> str:
    return ".exe" if sys.platform.startswith("win") else ""


def cmake_bin() -> str:
    return os.environ.get("COMPRESSION_BENCHMARK_CMAKE", "cmake")


def cc_bin() -> str:
    return os.environ.get("CC", "gcc")


def job_count() -> str:
    raw = os.environ.get("COMPRESSION_BENCHMARK_BUILD_JOBS", "").strip()
    if raw:
        return raw
    return str(os.cpu_count() or 4)


def cmake_configure_extra() -> list[str]:
    """Windows: default CMake generator is MSVC; col bench scripts use gcc like the old .sh."""
    if not sys.platform.startswith("win"):
        return []
    cxx = os.environ.get("CXX", "g++")
    return [
        "-G",
        "MinGW Makefiles",
        f"-DCMAKE_C_COMPILER={cc_bin()}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
    ]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


@dataclass(frozen=True)
class ColBenchSpec:
    exe_stem: str
    submodule_rel: str
    cmake_args: tuple[str, ...]
    cmake_target: str
    static_lib_rel: str
    cli_src_rel: str
    include_rel: str


BLOSC1_SPEC = ColBenchSpec(
    exe_stem="blosc_col_bench",
    submodule_rel="external/compression/c-blosc",
    cmake_args=(
        "-DBUILD_STATIC=ON",
        "-DBUILD_SHARED=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_BENCHMARKS=OFF",
        "-DBUILD_FUZZERS=OFF",
        "-DEACTIVATE_LZ4=ON",
        "-DEACTIVATE_SNAPPY=ON",
        "-DEACTIVATE_ZLIB=ON",
        "-DEACTIVATE_ZSTD=ON",
    ),
    cmake_target="blosc_static",
    static_lib_rel="blosc/libblosc.a",
    cli_src_rel="cpp/src/blosc_col_cli.c",
    include_rel="blosc",
)

BLOSC2_SPEC = ColBenchSpec(
    exe_stem="blosc2_col_bench",
    submodule_rel="external/compression/c-blosc2",
    cmake_args=(
        "-DBUILD_LITE=ON",
        "-DBUILD_STATIC=ON",
        "-DBUILD_SHARED=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_FUZZERS=OFF",
        "-DBUILD_BENCHMARKS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_PLUGINS=OFF",
        "-DBLOSC_ENABLE_ZFP=OFF",
    ),
    cmake_target="blosc2_static",
    static_lib_rel="blosc/libblosc2.a",
    cli_src_rel="cpp/src/blosc2_col_cli.c",
    include_rel="include",
)


def col_bench_output_path(spec: ColBenchSpec, root: Path | None = None) -> Path:
    root = root or ROOT
    return root / spec.submodule_rel / "build" / "bin" / f"{spec.exe_stem}{exe_suffix()}"


def build_col_bench(spec: ColBenchSpec, *, root: Path | None = None) -> Path:
    root = root or ROOT
    src = root / spec.submodule_rel
    if not (src / "CMakeLists.txt").is_file():
        raise SystemExit(
            f"Missing {src}. Run: git submodule update --init {spec.submodule_rel}"
        )

    build = src / "build"
    out = col_bench_output_path(spec, root)
    static_lib = build / spec.static_lib_rel
    cli_src = root / spec.cli_src_rel
    include_dir = src / spec.include_rel

    cfg_type = os.environ.get("COMPRESSION_BENCHMARK_CMAKE_BUILD_TYPE", "Release")
    run(
        [
            cmake_bin(),
            "-S",
            str(src),
            "-B",
            str(build),
            f"-DCMAKE_BUILD_TYPE={cfg_type}",
            *cmake_configure_extra(),
            *spec.cmake_args,
        ]
    )
    run(
        [
            cmake_bin(),
            "--build",
            str(build),
            "--target",
            spec.cmake_target,
            "-j",
            job_count(),
        ]
    )
    if not static_lib.is_file():
        raise SystemExit(f"Expected static library missing: {static_lib}")

    out.parent.mkdir(parents=True, exist_ok=True)
    link_cmd = [
        cc_bin(),
        "-O3",
        f"-I{include_dir}",
        str(cli_src),
        str(static_lib),
        "-pthread",
        "-o",
        str(out),
    ]
    run(link_cmd)

    if not out.is_file():
        raise SystemExit(f"Build failed: {out}")
    print(f"Built: {out}")
    return out
