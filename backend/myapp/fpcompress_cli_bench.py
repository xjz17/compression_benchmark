"""Legacy FPCompress CPU CLI helpers.

Runtime benchmark rows for ``fpcompress_dpratio`` and ``fpcompress_dpspeed`` are
now emitted by ``native_lz_family`` in Python/C++/Java.  This module is kept only
so old imports do not break; it must not launch external FPCompress binaries.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

from .benchmark_wire import column_original_size
from .upstream_repo_paths import fpcompress_bin_root, fpcompress_upstream_root

FPCOMPRESS_ROW_NAMES: Tuple[str, ...] = (
    "fpcompress_dpratio",
    "fpcompress_dpspeed",
)

SRC_TREE = "double_src"


def _first_runnable_file(path: Path) -> Path | None:
    candidates: list[Path] = [path]
    if os.name == "nt" and path.suffix.lower() != ".exe":
        candidates.append(path.with_name(path.name + ".exe"))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    return None


def _resolve_exe(stem: str) -> Path | None:
    env_key = {
        "ratio-cpu-compress": "COMPRESSION_BENCHMARK_FPCOMPRESS_DPRATIO_COMPRESS",
        "ratio-cpu-decompress": "COMPRESSION_BENCHMARK_FPCOMPRESS_DPRATIO_DECOMPRESS",
        "speed-cpu-compress": "COMPRESSION_BENCHMARK_FPCOMPRESS_DPSPEED_COMPRESS",
        "speed-cpu-decompress": "COMPRESSION_BENCHMARK_FPCOMPRESS_DPSPEED_DECOMPRESS",
    }.get(stem)
    if env_key:
        raw = os.environ.get(env_key, "").strip()
        if raw:
            hit = _first_runnable_file(Path(raw))
            if hit is not None:
                return hit
    for root in (fpcompress_bin_root(), fpcompress_upstream_root()):
        hit = _first_runnable_file(root / SRC_TREE / "bin" / stem)
        if hit is not None:
            return hit
        hit = _first_runnable_file(root / SRC_TREE / stem)
        if hit is not None:
            return hit
    return None


def fpcompress_bench_enabled() -> bool:
    return False


def _pack_dp_float_column(col: Sequence[float]) -> bytes:
    return struct.pack("<" + "d" * len(col), *[float(x) for x in col])


def _run_cli(executable: Path, input_path: Path, output_path: Path) -> bool:
    try:
        if os.name == "nt":
            cmd = (
                f'cmd /c ""{executable}" "{input_path}" "{output_path}""'
            )
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
        else:
            proc = subprocess.run(
                [str(executable), str(input_path), str(output_path)],
                capture_output=True,
                timeout=300,
                check=False,
            )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _bench_columns_with_pair(
    columns: Iterable[Sequence],
    *,
    pack_fn: Callable[[Sequence], bytes],
    ratio: bool,
) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_byte_loss

    comp_stem = "ratio-cpu-compress" if ratio else "speed-cpu-compress"
    decomp_stem = "ratio-cpu-decompress" if ratio else "speed-cpu-decompress"
    comp_exe = _resolve_exe(comp_stem)
    decomp_exe = _resolve_exe(decomp_stem)
    if comp_exe is None or decomp_exe is None:
        return 0, 0, 0, 0, 0.0, 0.0

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        n = len(col)
        if n < 1:
            continue
        raw = pack_fn(col)
        try:
            with tempfile.TemporaryDirectory(prefix="wc_fpcompress_py_") as tmp:
                tdir = Path(tmp)
                in_path = tdir / "in.bin"
                comp_path = tdir / "comp.bin"
                out_path = tdir / "out.bin"
                in_path.write_bytes(raw)
                t0 = time.perf_counter_ns()
                if not _run_cli(comp_exe, in_path, comp_path):
                    continue
                t1 = time.perf_counter_ns()
                if not comp_path.is_file() or comp_path.stat().st_size <= 0:
                    continue
                csize = comp_path.stat().st_size
                t2 = time.perf_counter_ns()
                if not _run_cli(decomp_exe, comp_path, out_path):
                    continue
                t3 = time.perf_counter_ns()
                if not out_path.is_file():
                    continue
                decoded = out_path.read_bytes()
                if decoded != raw:
                    continue
                accumulate_byte_loss(raw, decoded, loss_acc)
                tot_o += column_original_size(n, "float")
                tot_c += int(csize)
                te += t1 - t0
                td += t3 - t2
        except OSError:
            continue
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_fpcompress_dpratio_double_columns(
    columns: List[List[float]],
) -> Tuple[int, int, int, int, float, float]:
    return 0, 0, 0, 0, 0.0, 0.0


def bench_fpcompress_dpspeed_double_columns(
    columns: List[List[float]],
) -> Tuple[int, int, int, int, float, float]:
    return 0, 0, 0, 0, 0.0, 0.0


def record_fpcompress_benchmarks(
    *,
    double_native_cols: List[List[float]] | None,
    should_run,
    record_fn,
) -> None:
    from .bench_record import record_bench

    if not fpcompress_bench_enabled() or not double_native_cols:
        return
    specs: list[tuple[str, Callable[[], Tuple[int, int, int, int, float, float]]]] = [
        (
            "fpcompress_dpratio",
            lambda: bench_fpcompress_dpratio_double_columns(double_native_cols),
        ),
        (
            "fpcompress_dpspeed",
            lambda: bench_fpcompress_dpspeed_double_columns(double_native_cols),
        ),
    ]
    for name, bench_fn in specs:
        if not should_run(name):
            continue
        result = bench_fn()
        if int(result[0]) > 0:
            record_bench(record_fn, name, result)
