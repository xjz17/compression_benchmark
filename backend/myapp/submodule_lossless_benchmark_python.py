"""
Optional lossless benchmarks for git submodules under ``external/compression/``:

* **cmix** — native **CPU-only** ``cmix`` (``-c``/``-n`` compress, ``-d`` decompress); no GPU. Set
  ``COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS=1`` to use ``-n`` (faster, different ratio). MSDZip/MSDLC need
  PyTorch (CUDA recommended; CPU path patches ``.cuda()`` calls).
* **MSDZip** — PyTorch upstream scripts (``COMPRESSION_BENCHMARK_MSDZIP_BENCH=1``); CUDA recommended. Optional
  ``COMPRESSION_BENCHMARK_MSD_MAX_COLUMNS`` caps how many leading columns are benchmarked (staticdata wall time).
* **MSDLC** — same (``COMPRESSION_BENCHMARK_MSDLC_BENCH=1``).

When binaries / CUDA / deps are missing, functions return ``(0, 0, 0, 0)`` and no CSV row is emitted.

The DAC'23 autoregressive repo (``external/compression/Faster-and-Stronger-Lossless-Compression-with-Optimized-Autoregressive-Framework``) has no standalone
file compress CLI in-tree; do not add a row until a supported entrypoint exists.
"""
from __future__ import annotations

from .benchmark_wire import column_original_size

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return _REPO_ROOT


def _cmix_exe() -> Path | None:
    env = os.environ.get("COMPRESSION_BENCHMARK_CMIX", "").strip()
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    cand = _repo_root() / "external/compression/cmix/cmix"
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    return None


def _cmix_timeout_sec() -> int:
    raw = os.environ.get("COMPRESSION_BENCHMARK_CMIX_TIMEOUT_SEC", "").strip()
    if not raw:
        return 600
    try:
        v = int(raw)
    except ValueError:
        return 600
    if v <= 0:
        return 0
    return min(v, 86400)


def _cmix_compress_argv_prefix() -> Sequence[str]:
    """``-c`` default; ``-n`` if COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS=1 (faster CPU, no preprocessor)."""
    if os.environ.get("COMPRESSION_BENCHMARK_CMIX_NO_PREPROCESS", "").strip() == "1":
        return ["-n"]
    return ["-c"]


def _cmix_max_column_bytes() -> int:
    """If positive, only compress a leading prefix of each column (that many bytes). 0 = full column."""
    raw = os.environ.get("COMPRESSION_BENCHMARK_CMIX_MAX_COLUMN_BYTES", "").strip()
    if not raw:
        return 0
    try:
        v = int(raw)
    except ValueError:
        return 0
    if v <= 0:
        return 0
    return min(v, 256 * 1024 * 1024)


def bench_cmix_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    if os.environ.get("COMPRESSION_BENCHMARK_CMIX_BENCH", "").strip() != "1":
        return 0, 0, 0, 0
    exe = _cmix_exe()
    if exe is None:
        return 0, 0, 0, 0
    timeout = _cmix_timeout_sec()
    max_b = _cmix_max_column_bytes()
    tot_o = tot_c = te = td = 0
    for col in columns:
        n = len(col)
        if n < 1:
            continue
        if max_b > 0:
            max_n = max_b // 4
            if n > max_n:
                n = max_n
        if n < 1:
            continue
        col = col[:n]
        raw = struct.pack("<" + "f" * n, *[float(x) for x in col])
        try:
            with tempfile.TemporaryDirectory(prefix="wc_cmix_") as tmp:
                tdir = Path(tmp)
                fin = tdir / "in.dat"
                fcomp = tdir / "out.cmix"
                fdec = tdir / "dec.dat"
                fin.write_bytes(raw)
                if fcomp.exists():
                    fcomp.unlink()
                if fdec.exists():
                    fdec.unlink()
                enc = [str(exe), *_cmix_compress_argv_prefix(), str(fin), str(fcomp)]
                t0 = time.perf_counter_ns()
                r1 = subprocess.run(
                    enc,
                    capture_output=True,
                    timeout=timeout or None,
                    check=False,
                )
                t1 = time.perf_counter_ns()
                if r1.returncode != 0 or not fcomp.is_file():
                    continue
                csize = fcomp.stat().st_size
                dec = [str(exe), "-d", str(fcomp), str(fdec)]
                t2 = time.perf_counter_ns()
                r2 = subprocess.run(
                    dec,
                    capture_output=True,
                    timeout=timeout or None,
                    check=False,
                )
                t3 = time.perf_counter_ns()
                if r2.returncode != 0 or not fdec.is_file():
                    continue
                if fdec.read_bytes() != raw:
                    continue
                tot_o += column_original_size(n, 'float')
                tot_c += int(csize)
                te += t1 - t0
                td += t3 - t2
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return tot_o, tot_c, te, td


def _msdzip_root() -> Path:
    return _repo_root() / "external/compression/MSDZip"


def _msdlc_root() -> Path:
    return _repo_root() / "external/compression/MSDLC"


def _msd_max_columns(columns: List[List[float]]) -> List[List[float]]:
    """If ``COMPRESSION_BENCHMARK_MSD_MAX_COLUMNS`` is a positive int, only benchmark that many leading columns."""
    raw = os.environ.get("COMPRESSION_BENCHMARK_MSD_MAX_COLUMNS", "").strip()
    if not raw:
        return columns
    try:
        n = int(raw)
    except ValueError:
        return columns
    if n <= 0:
        return columns
    return columns[:n]


def _msd_min_bytes() -> int:
    raw = os.environ.get("COMPRESSION_BENCHMARK_MSD_MIN_COLUMN_BYTES", "").strip()
    if not raw:
        try:
            import torch

            if not torch.cuda.is_available():
                return 200_000
        except ImportError:
            pass
        return 500_000
    try:
        return max(4096, int(raw))
    except ValueError:
        return 500_000


def _inject_msd_torch_device(text: str) -> str:
    """Rewrite ``.cuda()`` → ``.to(device)`` and guard ``torch.cuda.manual_seed`` (MSDZip/MSDLC scripts)."""
    if "device = torch.device" not in text:
        text = text.replace(
            "import torch\n",
            "import torch\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            1,
        )
    text = text.replace(".cuda()", ".to(device)")
    old = "    torch.cuda.manual_seed(args.seed)\n    torch.cuda.manual_seed_all(args.seed)"
    new = (
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.manual_seed(args.seed)\n"
        "        torch.cuda.manual_seed_all(args.seed)"
    )
    text = text.replace(old, new)
    return text


def _patch_msdlc_compress_model_vanilla(text: str) -> str:
    """Use xLSTM ``vanilla`` backend so training runs without CUDA kernels."""
    return text.replace("backend: cuda", "backend: vanilla")


def _write_msdzip_bench_scripts(root: Path, tmp_scripts: Path) -> tuple[Path, Path]:
    c = _inject_msd_torch_device((root / "compress.py").read_text(encoding="utf-8"))
    d = _inject_msd_torch_device((root / "decompress.py").read_text(encoding="utf-8"))
    cp = tmp_scripts / "compress_bench.py"
    dp = tmp_scripts / "decompress_bench.py"
    cp.write_text(c, encoding="utf-8")
    dp.write_text(d, encoding="utf-8")
    return cp, dp


def _write_msdlc_bench_scripts(root: Path, tmp_scripts: Path) -> tuple[Path, Path]:
    cm = _patch_msdlc_compress_model_vanilla((root / "compress_model.py").read_text(encoding="utf-8"))
    (tmp_scripts / "compress_model.py").write_text(cm, encoding="utf-8")
    c = _inject_msd_torch_device((root / "compress.py").read_text(encoding="utf-8"))
    d = _inject_msd_torch_device((root / "decompress.py").read_text(encoding="utf-8"))
    cp = tmp_scripts / "compress_bench.py"
    dp = tmp_scripts / "decompress_bench.py"
    cp.write_text(c, encoding="utf-8")
    dp.write_text(d, encoding="utf-8")
    return cp, dp


def _msd_pythonpath(root: Path, tmp_scripts: Path | None) -> str:
    parts = []
    if tmp_scripts is not None:
        parts.append(str(tmp_scripts.resolve()))
    parts.append(str(root.resolve()))
    extra = os.environ.get("PYTHONPATH", "").strip()
    if extra:
        parts.append(extra)
    return os.pathsep.join(parts)


def _msd_subprocess_timeout() -> int:
    raw = os.environ.get("COMPRESSION_BENCHMARK_MSD_SUBPROCESS_TIMEOUT_SEC", "").strip()
    if not raw:
        return 7200
    try:
        v = int(raw)
    except ValueError:
        return 7200
    if v <= 0:
        return 7200
    return min(v, 86400 * 7)


def bench_msdzip_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    if os.environ.get("COMPRESSION_BENCHMARK_MSDZIP_BENCH", "").strip() != "1":
        return 0, 0, 0, 0
    root = _msdzip_root()
    comp_src = root / "compress.py"
    dec_src = root / "decompress.py"
    if not comp_src.is_file() or not dec_src.is_file():
        return 0, 0, 0, 0
    try:
        import torch
    except ImportError:
        return 0, 0, 0, 0

    use_cuda = torch.cuda.is_available()
    tmp_scripts: Path | None = None
    comp_py = comp_src
    dec_py = dec_src
    if not use_cuda:
        tmp_scripts = Path(tempfile.mkdtemp(prefix="wc_msdzip_scripts_"))
        try:
            comp_py, dec_py = _write_msdzip_bench_scripts(root, tmp_scripts)
        except OSError:
            shutil.rmtree(tmp_scripts, ignore_errors=True)
            return 0, 0, 0, 0

    min_b = _msd_min_bytes()
    timeout = _msd_subprocess_timeout()
    bs = 512 if use_cuda else 64
    ts = 16
    try:
        bs = int(os.environ.get("COMPRESSION_BENCHMARK_MSDZIP_BATCHSIZE", str(bs)).strip())
    except ValueError:
        pass
    try:
        ts = int(os.environ.get("COMPRESSION_BENCHMARK_MSDZIP_TIMESTEPS", "16").strip())
    except ValueError:
        pass
    if bs < 8:
        bs = 8
    if ts < 2:
        ts = 2

    py_path = _msd_pythonpath(root, tmp_scripts)
    tot_o = tot_c = te = td = 0
    for col in _msd_max_columns(columns):
        raw = struct.pack("<" + "f" * len(col), *[float(x) for x in col])
        if len(raw) < min_b:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="wc_msdzip_") as tmp:
                tdir = Path(tmp)
                fin = tdir / "in.dat"
                fcomp = tdir / "out.mz"
                fdec = tdir / "out.dec"
                fin.write_bytes(raw)
                prefix = "wcbench"
                if fcomp.exists():
                    fcomp.unlink()
                if fdec.exists():
                    fdec.unlink()
                for p in tdir.glob(f"{prefix}.params"):
                    p.unlink()
                enc = [
                    sys.executable,
                    str(comp_py),
                    str(fin),
                    str(fcomp),
                    "-p",
                    prefix,
                    "-b",
                    str(bs),
                    "--timesteps",
                    str(ts),
                ]
                env = {**os.environ, "PYTHONPATH": py_path}
                if use_cuda:
                    gpu = os.environ.get("COMPRESSION_BENCHMARK_MSD_GPU", "0").strip() or "0"
                    enc.extend(["--gpu", gpu])
                t0 = time.perf_counter_ns()
                r1 = subprocess.run(
                    enc,
                    cwd=str(tdir),
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
                t1 = time.perf_counter_ns()
                if r1.returncode != 0 or not fcomp.is_file():
                    continue
                csize = fcomp.stat().st_size
                dec = [
                    sys.executable,
                    str(dec_py),
                    str(fcomp),
                    str(fdec),
                    "-p",
                    prefix,
                    "-b",
                    str(bs),
                    "--timesteps",
                    str(ts),
                ]
                if use_cuda:
                    dec.extend(["--gpu", os.environ.get("COMPRESSION_BENCHMARK_MSD_GPU", "0").strip() or "0"])
                t2 = time.perf_counter_ns()
                r2 = subprocess.run(
                    dec,
                    cwd=str(tdir),
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
                t3 = time.perf_counter_ns()
                if r2.returncode != 0 or not fdec.is_file():
                    continue
                if fdec.read_bytes() != raw:
                    continue
                tot_o += len(raw)
                tot_c += int(csize)
                te += t1 - t0
                td += t3 - t2
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    if tmp_scripts is not None:
        shutil.rmtree(tmp_scripts, ignore_errors=True)
    return tot_o, tot_c, te, td


def bench_msdlc_float_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    if os.environ.get("COMPRESSION_BENCHMARK_MSDLC_BENCH", "").strip() != "1":
        return 0, 0, 0, 0
    root = _msdlc_root()
    comp_src = root / "compress.py"
    dec_src = root / "decompress.py"
    cm_src = root / "compress_model.py"
    if not comp_src.is_file() or not dec_src.is_file() or not cm_src.is_file():
        return 0, 0, 0, 0
    try:
        import torch
    except ImportError:
        return 0, 0, 0, 0

    use_cuda = torch.cuda.is_available()
    tmp_scripts: Path | None = None
    comp_py = comp_src
    dec_py = dec_src
    if not use_cuda:
        tmp_scripts = Path(tempfile.mkdtemp(prefix="wc_msdlc_scripts_"))
        try:
            comp_py, dec_py = _write_msdlc_bench_scripts(root, tmp_scripts)
        except OSError:
            shutil.rmtree(tmp_scripts, ignore_errors=True)
            return 0, 0, 0, 0

    min_b = _msd_min_bytes()
    timeout = _msd_subprocess_timeout()
    bs = 512 if use_cuda else 32
    ts = 32
    try:
        bs = int(os.environ.get("COMPRESSION_BENCHMARK_MSDLC_BATCHSIZE", str(bs)).strip())
    except ValueError:
        pass
    try:
        ts = int(os.environ.get("COMPRESSION_BENCHMARK_MSDLC_TIMESTEPS", "32").strip())
    except ValueError:
        pass
    if bs < 8:
        bs = 8
    if ts < 2:
        ts = 2

    py_path = _msd_pythonpath(root, tmp_scripts)
    tot_o = tot_c = te = td = 0
    for col in _msd_max_columns(columns):
        raw = struct.pack("<" + "f" * len(col), *[float(x) for x in col])
        if len(raw) < min_b:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="wc_msdlc_") as tmp:
                tdir = Path(tmp)
                fin = tdir / "in.dat"
                fcomp = tdir / "out.ms"
                fdec = tdir / "out.dec"
                fin.write_bytes(raw)
                prefix = "wcbench"
                if fcomp.exists():
                    fcomp.unlink()
                if fdec.exists():
                    fdec.unlink()
                for p in tdir.glob(f"{prefix}.params"):
                    p.unlink()
                enc = [
                    sys.executable,
                    str(comp_py),
                    str(fin),
                    str(fcomp),
                    "-p",
                    prefix,
                    "-b",
                    str(bs),
                    "--timesteps",
                    str(ts),
                ]
                env = {**os.environ, "PYTHONPATH": py_path}
                t0 = time.perf_counter_ns()
                r1 = subprocess.run(
                    enc,
                    cwd=str(tdir),
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
                t1 = time.perf_counter_ns()
                if r1.returncode != 0 or not fcomp.is_file():
                    continue
                csize = fcomp.stat().st_size
                dec = [
                    sys.executable,
                    str(dec_py),
                    str(fcomp),
                    str(fdec),
                    "-p",
                    prefix,
                    "-b",
                    str(bs),
                    "--timesteps",
                    str(ts),
                ]
                t2 = time.perf_counter_ns()
                r2 = subprocess.run(
                    dec,
                    cwd=str(tdir),
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
                t3 = time.perf_counter_ns()
                if r2.returncode != 0 or not fdec.is_file():
                    continue
                if fdec.read_bytes() != raw:
                    continue
                tot_o += len(raw)
                tot_c += int(csize)
                te += t1 - t0
                td += t3 - t2
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    if tmp_scripts is not None:
        shutil.rmtree(tmp_scripts, ignore_errors=True)
    return tot_o, tot_c, te, td


def submodule_benchmark_float_pairs():
    """(algorithm_key, bench_fn) tuples appended to ``_RESEARCH_FLOAT_BENCHMARK_EXTENSIONS``."""
    return (
        ("ext_cmix", bench_cmix_float_columns),
        ("ext_msdzip_www25", bench_msdzip_float_columns),
        ("ext_msdlc_icassp25", bench_msdlc_float_columns),
    )
