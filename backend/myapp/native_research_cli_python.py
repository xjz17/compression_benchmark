"""
SZ / SZ3 / Serf / BUFF (``comp_profiler``) via the same native CLIs as ``ResearchNativeCliBench`` /
``ResearchSerfCliBench`` / ``ResearchBuffCliBench``.

This mirrors the Java layer in-repo (subprocess to ``sz`` / ``sz3`` / ``sfz``), not a predictor
reimplementation. When binaries or configs are missing, benches return ``(0, 0, 0, 0)`` and the
processor skips rows (same effective behavior as Java when CLIs are absent).

Pure-Python Serf (XOR codec as in ``external/compression/Serf``, ``adjust_digit=0``, window 1000):
``myapp/serf_xor_python.py`` (``compress_floats`` / ``decompress_floats``). SZ remains C/C CLI only here.
"""
from __future__ import annotations

from .benchmark_wire import column_original_size

import os
import platform
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

SZ_FLOAT_PRESETS: Tuple[Tuple[str, str, str, str], ...] = (
    ("sz_ABS_1e-1", "ABS", "-A", "1E-1"),
    ("sz_ABS_1e-2", "ABS", "-A", "1E-2"),
    ("sz_ABS_1e-3", "ABS", "-A", "1E-3"),
    ("sz_ABS_1e-4", "ABS", "-A", "1E-4"),
    ("sz_REL_1e-2", "REL", "-R", "1E-2"),
    ("sz_REL_1e-3", "REL", "-R", "1E-3"),
)

SERF_ABS_PRESETS: Tuple[Tuple[str, str], ...] = (
    ("serf_ABS_1e-1", "1e-1"),
    ("serf_ABS_1e-2", "1e-2"),
    ("serf_ABS_1e-3", "1e-3"),
    ("serf_ABS_1e-4", "1e-4"),
)


def _repo_root() -> Path:
    return _REPO_ROOT


def _first_runnable_file(path: Path) -> Path | None:
    """Resolve a native CLI; on Windows try ``name.exe`` when extensionless ``name`` is absent."""
    candidates: list[Path] = [path]
    if os.name == "nt" and path.suffix.lower() != ".exe":
        candidates.append(path.with_name(path.name + ".exe"))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    return None


def _subprocess_env_for_native_cli(executable: Path) -> dict[str, str]:
    """Prepend SZ3/fpzip DLL dirs to PATH (same dirs as Java ``NativeCliExecutable``)."""
    env = os.environ.copy()
    prepend: list[str] = []
    parent = executable.parent
    if parent.is_dir():
        prepend.append(str(parent.resolve()))
    zstd = _repo_root() / "external/compression/SZ3/build/tools/zstd"
    if zstd.is_dir():
        prepend.append(str(zstd.resolve()))
    mingw = os.environ.get("WEB_COMPRESSION_MINGW_BIN", "").strip()
    if mingw:
        prepend.append(mingw)
    extra = os.environ.get("WEB_COMPRESSION_NATIVE_CLI_PATH", "").strip()
    if extra:
        prepend.extend(p for p in extra.split(os.pathsep) if p.strip())
    if not prepend:
        return env
    path_key = "Path" if os.name == "nt" else "PATH"
    cur = env.get(path_key) or env.get("PATH") or ""
    merged = os.pathsep.join(prepend + ([cur] if cur else []))
    env[path_key] = merged
    if os.name == "nt":
        env["PATH"] = merged
    return env


def _resolve_sz() -> Path | None:
    env = os.environ.get("WEB_COMPRESSION_SZ", "").strip()
    if env:
        hit = _first_runnable_file(Path(env))
        if hit is not None:
            return hit
    root = _repo_root()
    for rel in (
        "external/compression/SZ/example/sz",
        "external/compression/SZ/build/bin/sz",
    ):
        hit = _first_runnable_file(root / rel)
        if hit is not None:
            return hit
    return None


def _resolve_sz3() -> Path | None:
    env = os.environ.get("WEB_COMPRESSION_SZ3", "").strip()
    if env:
        hit = _first_runnable_file(Path(env))
        if hit is not None:
            return hit
    cand = _repo_root() / "external/compression/SZ3/build/tools/sz3/sz3"
    return _first_runnable_file(cand)


def _resolve_sz_config() -> Path | None:
    for key in ("WEB_COMPRESSION_SZ_CONFIG", "WEB_COMPRESSION_SZ3_CONFIG"):
        env = os.environ.get(key, "").strip()
        if env:
            p = Path(env)
            if p.is_file():
                return p
    legacy = _repo_root() / "external/compression/SZ/example/sz.config"
    if legacy.is_file():
        return legacy
    cand = _repo_root() / "external/compression/SZ3/sz.config"
    if cand.is_file():
        return cand
    return None


def _resolve_sfz() -> Path | None:
    env = os.environ.get("WEB_COMPRESSION_SFZ", "").strip()
    if env:
        hit = _first_runnable_file(Path(env))
        if hit is not None:
            return hit
    cand = _repo_root() / "external/compression/Serf/build/serf_cmd/sfz"
    return _first_runnable_file(cand)


def _resolve_sz_dec_path(in_path: Path, out_sz: Path) -> Path | None:
    for cand in (
        Path(str(out_sz) + ".out"),
        Path(str(out_sz) + "."),
        Path(str(in_path) + ".out"),
    ):
        if cand.is_file():
            return cand
    return None


def bench_sz_family_float_columns(
    columns: List[List[float]],
    executable: Path,
    cfg: Path,
    err_mode: str,
    bound_flag: str,
    bound_value: str,
    *,
    use_double_wire: bool = False,
) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_double_list_loss

    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        n = len(col)
        if n < 1:
            continue
        if use_double_wire:
            raw = struct.pack("<" + "d" * n, *[float(x) for x in col])
            dtype_flag = "-d"
        else:
            raw = struct.pack("<" + "f" * n, *[float(x) for x in col])
            dtype_flag = "-f"
        try:
            with tempfile.TemporaryDirectory(prefix="wc_sz_float_py_") as tmp:
                tdir = Path(tmp)
                in_path = tdir / f"in-{os.getpid()}.dat"
                out_sz = Path(str(in_path) + ".sz")
                in_path.write_bytes(raw)
                if out_sz.exists():
                    out_sz.unlink()
                enc_cmd = [
                    str(executable),
                    "-z",
                    dtype_flag,
                    "-c",
                    str(cfg),
                    "-M",
                    err_mode,
                    bound_flag,
                    bound_value,
                    "-i",
                    str(in_path),
                    "-1",
                    str(n),
                ]
                cli_env = _subprocess_env_for_native_cli(executable)
                t0 = time.perf_counter_ns()
                r1 = subprocess.run(
                    enc_cmd,
                    capture_output=True,
                    timeout=120,
                    check=False,
                    env=cli_env,
                    cwd=str(executable.parent),
                )
                t1 = time.perf_counter_ns()
                if r1.returncode != 0 or not out_sz.is_file():
                    continue
                csize = out_sz.stat().st_size
                dec_cmd = [
                    str(executable),
                    "-x",
                    dtype_flag,
                    "-s",
                    str(out_sz),
                    "-1",
                    str(n),
                ]
                t2 = time.perf_counter_ns()
                r2 = subprocess.run(
                    dec_cmd,
                    capture_output=True,
                    timeout=120,
                    check=False,
                    env=cli_env,
                    cwd=str(executable.parent),
                )
                t3 = time.perf_counter_ns()
                if r2.returncode != 0:
                    continue
                dec_path = _resolve_sz_dec_path(in_path, out_sz)
                if dec_path is None:
                    continue
                dec_bytes = dec_path.read_bytes()
                if use_double_wire:
                    dec_vals = [
                        struct.unpack_from("<d", dec_bytes, i * 8)[0]
                        for i in range(min(n, len(dec_bytes) // 8))
                    ]
                else:
                    dec_vals = [
                        float(struct.unpack_from("<f", dec_bytes, i * 4)[0])
                        for i in range(min(n, len(dec_bytes) // 4))
                    ]
                if len(dec_vals) != n:
                    continue
                accumulate_double_list_loss(
                    [float(x) for x in col],
                    dec_vals,
                    loss_acc,
                )
                tot_o += column_original_size(n, 'float')
                tot_c += int(csize)
                te += t1 - t0
                td += t3 - t2
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def bench_serf_abs_float_columns(columns: List[List[float]], bound_str: str) -> Tuple[int, int, int, int, float, float]:
    from .benchmark_loss import accumulate_double_list_loss
    if os.environ.get("WEB_COMPRESSION_SERF_BENCH", "").strip() != "1":
        return 0, 0, 0, 0, 0.0, 0.0
    sfz = _resolve_sfz()
    if sfz is None:
        return 0, 0, 0, 0, 0.0, 0.0
    bound = float(bound_str)
    tot_o = tot_c = te = td = 0
    loss_acc = [0.0, 0.0]
    for col in columns:
        n = len(col)
        if n < 1:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="wc_serf_float_py_") as tmp:
                tdir = Path(tmp)
                in_txt = tdir / "in.txt"
                out_bin = tdir / "cmp.bin"
                out_txt = tdir / "dec.txt"
                lines = "\n".join(str(float(x)) for x in col) + "\n"
                in_txt.write_text(lines, encoding="ascii")
                t0 = time.perf_counter_ns()
                r1 = subprocess.run(
                    [str(sfz), "compress", bound_str, str(in_txt), str(out_bin)],
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                t1 = time.perf_counter_ns()
                if r1.returncode != 0 or not out_bin.is_file():
                    continue
                csize = out_bin.stat().st_size
                t2 = time.perf_counter_ns()
                r2 = subprocess.run(
                    [str(sfz), "decompress", bound_str, str(out_bin), str(out_txt)],
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                t3 = time.perf_counter_ns()
                if r2.returncode != 0 or not out_txt.is_file():
                    continue
                ok = True
                lines_read = 0
                dec_vals: List[float] = []
                with out_txt.open("r", encoding="ascii", errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if i >= n:
                            ok = False
                            break
                        got = float(line.strip())
                        dec_vals.append(got)
                        if abs(got - float(col[i])) > bound * 1.0001 + 1e-6:
                            ok = False
                            break
                        lines_read += 1
                    if lines_read != n:
                        ok = False
                if not ok:
                    continue
                accumulate_double_list_loss([float(x) for x in col], dec_vals, loss_acc)
                tot_o += column_original_size(n, 'float')
                tot_c += int(csize)
                te += t1 - t0
                td += t3 - t2
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return tot_o, tot_c, te, td, loss_acc[0], loss_acc[1]


def _mk_sz(which: str, row_name: str, emode: str, flag: str, val: str):
    def bench(cols: List[List[float]]) -> Tuple[int, int, int, int]:
        ex = _resolve_sz() if which == "sz" else _resolve_sz3()
        cfg = _resolve_sz_config()
        if ex is None or cfg is None:
            return 0, 0, 0, 0, 0.0, 0.0
        return bench_sz_family_float_columns(
            cols,
            ex,
            cfg,
            emode,
            flag,
            val,
            use_double_wire=True,
        )

    bench.__name__ = f"bench_{row_name}"
    return bench


def _mk_serf(row_name: str, bound: str):
    def bench(cols: List[List[float]]) -> Tuple[int, int, int, int]:
        return bench_serf_abs_float_columns(cols, bound)

    bench.__name__ = f"bench_{row_name}"
    return bench


def _resolve_comp_profiler() -> Path | None:
    env = os.environ.get("WEB_COMPRESSION_COMP_PROFILER", "").strip()
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    root = _repo_root()
    for rel in (
        "external/compression/buff/database/target/release/comp_profiler",
        "external/compression/buff/database/target/debug/comp_profiler",
    ):
        cand = root / rel
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def _buff_database_dir() -> Path | None:
    d = _repo_root() / "external/compression/buff/database"
    return d if d.is_dir() else None


def _buff_simd_profiler_mode() -> str:
    """``buff-simd`` on x86_64; ``buff`` on AArch64 (upstream SIMD is x86-only). Override via env."""
    env = os.environ.get("WEB_COMPRESSION_BUFF_SIMD_MODE", "").strip().lower()
    if env in ("buff-simd", "buff"):
        return env
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "buff-simd"
    if m in ("aarch64", "arm64"):
        return "buff"
    return "buff"


def _parse_buff_performance(stdout: str) -> tuple[float, float, float] | None:
    for line in stdout.splitlines():
        if not line.startswith("Performance:"):
            continue
        rest = line[len("Performance:") :]
        parts = rest.split(",")
        if len(parts) < 10:
            return None
        n = len(parts)
        try:
            ratio = float(parts[n - 7].strip())
            enc = float(parts[n - 6].strip())
            dec = float(parts[n - 5].strip())
        except ValueError:
            return None
        if ratio <= 0 or enc <= 0 or dec <= 0:
            return None
        return ratio, enc, dec
    return None


# Reuse scalar totals for SIMD row on AArch64 when driver calls scalar then simd with same columns list.
_BUFF_SCALAR_TOTALS_BY_COLUMNS_ID: dict[int, Tuple[int, int, int, int]] = {}


def _bench_buff_columns_with_mode(
    columns: List[List[float]], profiler_mode: str
) -> Tuple[int, int, int, int]:
    if os.environ.get("WEB_COMPRESSION_BUFF_BENCH", "").strip() != "1":
        return 0, 0, 0, 0
    ex = _resolve_comp_profiler()
    db = _buff_database_dir()
    if ex is None or db is None:
        return 0, 0, 0, 0
    scale = 10000
    es = os.environ.get("WEB_COMPRESSION_BUFF_INT_SCALE", "").strip()
    if es:
        try:
            scale = int(es)
        except ValueError:
            pass
    pred = 1.1509
    ep = os.environ.get("WEB_COMPRESSION_BUFF_PRED", "").strip()
    if ep:
        try:
            pred = float(ep)
        except ValueError:
            pass
    tot_o = tot_c = te = td = 0
    for col in columns:
        n = len(col)
        if n < 1:
            continue
        org_col = n * 8
        try:
            with tempfile.TemporaryDirectory(prefix="wc_buff_float_py_") as tmp:
                tdir = Path(tmp)
                csv_p = tdir / "in.csv"
                csv_p.write_text("\n".join(str(float(x)) for x in col) + "\n", encoding="ascii")
                r = subprocess.run(
                    [str(ex), str(csv_p.resolve()), profiler_mode, str(scale), str(pred)],
                    cwd=str(db),
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                if r.returncode != 0:
                    continue
                parsed = _parse_buff_performance(r.stdout)
                if parsed is None:
                    continue
                ratio, enc_m, dec_m = parsed
                comp_col = int(ratio * org_col)
                enc_ns = int(org_col * 1e9 / (enc_m * 1024 * 1024))
                dec_ns = int(org_col * 1e9 / (dec_m * 1024 * 1024))
                if comp_col < 1 or enc_ns < 1 or dec_ns < 1:
                    continue
                tot_o += org_col
                tot_c += comp_col
                te += enc_ns
                td += dec_ns
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return tot_o, tot_c, te, td


def bench_buff_float_scalar_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    """BUFF scalar path via ``comp_profiler``."""
    r = _bench_buff_columns_with_mode(columns, "buff")
    if r[0] > 0:
        _BUFF_SCALAR_TOTALS_BY_COLUMNS_ID[id(columns)] = r
    return r


def bench_buff_float_simd_columns(columns: List[List[float]]) -> Tuple[int, int, int, int]:
    """BUFF SIMD row: ``buff-simd`` on x86_64; on AArch64 reuses scalar totals (same ``buff`` run)."""
    mode = _buff_simd_profiler_mode()
    if mode == "buff":
        cid = id(columns)
        cached = _BUFF_SCALAR_TOTALS_BY_COLUMNS_ID.pop(cid, None)
        if cached is not None:
            return cached
        return _bench_buff_columns_with_mode(columns, "buff")
    return _bench_buff_columns_with_mode(columns, mode)


def bench_buff_float_kernel_native_columns(
    columns: List[List[float | int]], *, wire_data_type: str = "float"
) -> Tuple[int, int, int, int]:
    """Time actual lossless raw64 encoding and decoding, after filtering NaNs."""
    if os.environ.get("WEB_COMPRESSION_BUFF_KERNEL_NATIVE", "").strip() != "1":
        return 0, 0, 0, 0
    from .buff_kernel_native import decode_raw64, encode_raw64, pack_raw64_values

    if wire_data_type not in {"int", "float"}:
        raise ValueError("BUFF native benchmark requires int or float wire data")
    kind = "int64" if wire_data_type == "int" else "float64"
    tot_o = tot_c = te = td = 0
    for col in columns:
        raw = pack_raw64_values(col, kind, drop_nan=True)
        n = len(raw) // 8
        if n < 1:
            continue
        t0 = time.perf_counter_ns()
        packed = encode_raw64(raw)
        t1 = time.perf_counter_ns()
        restored = decode_raw64(packed)
        t2 = time.perf_counter_ns()
        if restored != raw:
            raise RuntimeError("BUFF native failed original-lane bitwise verification")
        tot_o += column_original_size(n, wire_data_type)
        tot_c += len(packed)
        te += t1 - t0
        td += t2 - t1
    return tot_o, tot_c, te, td


def bench_buff_int_kernel_native_columns(columns: List[List[int]]) -> Tuple[int, int, int, int]:
    """Lossless int64 raw lanes; the caller must supply original integer columns."""
    return bench_buff_float_kernel_native_columns(columns, wire_data_type="int")


def native_cli_float_benchmark_pairs():
    """(algorithm_key, bench_fn) for use in processor FLOAT benchmark lists."""
    out: List[Tuple[str, object]] = []
    sz_flag = os.environ.get("WEB_COMPRESSION_SZ_BENCH", "1").strip()
    sz_on = sz_flag != "0"
    sz3_flag = os.environ.get("WEB_COMPRESSION_SZ3_BENCH", "1").strip()
    sz3_on = sz3_flag != "0"
    for name, em, fl, val in SZ_FLOAT_PRESETS:
        if sz_on:
            out.append((name, _mk_sz("sz", name, em, fl, val)))
        if sz3_on:
            n3 = name.replace("sz_", "sz3_")
            out.append((n3, _mk_sz("sz3", n3, em, fl, val)))
    for row_name, bound in SERF_ABS_PRESETS:
        out.append((row_name, _mk_serf(row_name, bound)))
    out.append(("buff_scalar", bench_buff_float_scalar_columns))
    out.append(("buff_simd", bench_buff_float_simd_columns))
    out.append(("buff_kernel_native", bench_buff_float_kernel_native_columns))
    return out
