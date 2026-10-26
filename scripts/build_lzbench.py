#!/usr/bin/env python3
"""
Build lzbench under ``external/compression/lzbench`` (Makefile project; see lzbench/BUILD.md).

From repository root::

    python scripts/build_lzbench.py
    python scripts/build_lzbench.py --verify-only

Output: ``external/compression/lzbench/lzbench`` (``.exe`` on Windows).

Skips unstable / optional codecs (see lzbench README "BUGGY" list and BUILD.md).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LZ = ROOT / "external" / "compression" / "lzbench"

# Same disable set as the former ``build_lzbench.sh``.
DISABLE_FLAGS = (
    "DONT_BUILD_GLZA=1",
    "DONT_BUILD_ZPAQ=1",
    "DONT_BUILD_LZMAT=1",
    "DONT_BUILD_LZRW=1",
    "DONT_BUILD_WFLZ=1",
    "DONT_BUILD_GIPFELI=1",
    "DONT_BUILD_CSC=1",
    "DONT_BUILD_YAPPY=1",
    "DONT_BUILD_DENSITY=1",
    "DONT_BUILD_ACEAPEX=1",
)


def exe_suffix() -> str:
    return ".exe" if sys.platform.startswith("win") else ""


def make_bin() -> str:
    return os.environ.get("WEB_COMPRESSION_MAKE", "make")


def job_count() -> str:
    raw = os.environ.get("WEB_COMPRESSION_BUILD_JOBS", "").strip()
    if raw:
        return raw
    return str(os.cpu_count() or 4)


def lzbench_output_path() -> Path:
    return LZ / f"lzbench{exe_suffix()}"


def unix_tools_path_prefixes() -> list[str]:
    """Paths for grep/awk/uname used by lzbench's Makefile (same role as /usr/bin on Unix)."""
    override = os.environ.get("WEB_COMPRESSION_UNIX_TOOLS_BIN", "").strip()
    if override:
        return [override]
    if sys.platform.startswith("win"):
        out: list[str] = []
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ):
            git_usr = Path(base) / "Git" / "usr" / "bin"
            if git_usr.is_dir():
                out.append(str(git_usr))
        return out
    return ["/usr/bin"]


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    prepend = unix_tools_path_prefixes()
    if prepend:
        env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    return env


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def verify_lzbench(exe: Path) -> None:
    """Version + in-memory bench with project selectors (realistic payload size)."""
    run([str(exe), "-v"], cwd=LZ, env=os.environ.copy())

    import struct

    sys.path.insert(0, str(ROOT / "backend"))
    from myapp.lzbench_external import _run_lzbench_on_bytes, lzbench_selectors  # noqa: E402

    payload = struct.pack("<1000q", *[(i * 17) % 100_000 for i in range(1000)])
    rows = _run_lzbench_on_bytes(payload, exe)
    if not rows:
        raise SystemExit(
            f"FAIL: {exe} ran but project lzbench integration returned 0 rows "
            f"(selectors len={len(lzbench_selectors())})"
        )
    print(f"Verified: {len(rows)} algorithm rows, originalSize={rows[0]['originalSize']}")


def build_lzbench(*, clean: bool = False) -> Path:
    if not LZ.is_dir():
        raise SystemExit(
            f"Missing {LZ}. Run: git submodule update --init external/compression/lzbench"
        )

    env = build_env()
    if clean:
        run([make_bin(), "clean"], cwd=LZ, env=env)
    run(
        [make_bin(), f"-j{job_count()}", "BUILD_STATIC=1", *DISABLE_FLAGS],
        cwd=LZ,
        env=env,
    )

    out = lzbench_output_path()
    if not out.is_file():
        raise SystemExit(f"Build finished but binary missing: {out}")
    print(f"Built: {out}")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or verify lzbench CLI")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip make; verify existing external/compression/lzbench/lzbench binary",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run make clean before build (full rebuild; only gitignored artifacts in submodule)",
    )
    args = parser.parse_args()

    if args.verify_only:
        out = lzbench_output_path()
        if not out.is_file():
            raise SystemExit(f"Missing {out}; run without --verify-only to build")
        verify_lzbench(out)
        print(f"OK: {out}")
        return

    out = build_lzbench(clean=args.clean)
    verify_lzbench(out)


if __name__ == "__main__":
    main()
