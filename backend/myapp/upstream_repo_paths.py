"""
Resolve optional roots for official upstream GitHub checkouts.

Defaults point at git submodules (see ``.gitmodules`` and ``external/submodules/README.md``).
Override with environment variables when using a custom clone path.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _root() -> Path:
    return _REPO_ROOT


def _resolve(env_key: str, default_under_repo: Path) -> Path:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_under_repo


def streamvbyte_upstream_root() -> Path:
    """Lemire Stream VByte: https://github.com/lemire/streamvbyte"""
    return _resolve(
        "COMPRESSION_BENCHMARK_STREAMVBYTE_UPSTREAM_ROOT",
        _root() / "external" / "compression" / "streamvbyte",
    )


def mostdb_upstream_root() -> Path:
    """MOST / mostdb: https://github.com/schencoding/mostdb"""
    return _resolve(
        "COMPRESSION_BENCHMARK_MOSTDB_UPSTREAM_ROOT",
        _root() / "external" / "compression" / "mostdb",
    )


def corad_upstream_root() -> Path:
    """CORAD: https://github.com/eXascaleInfolab/CORAD"""
    return _resolve(
        "COMPRESSION_BENCHMARK_CORAD_UPSTREAM_ROOT",
        _root() / "external" / "compression" / "CORAD",
    )


def fpcompress_upstream_root() -> Path:
    """Burtscher FPCompress (ASPLOS 2025 DPratio/DPspeed): https://github.com/burtscher/FPCompress"""
    return _resolve(
        "COMPRESSION_BENCHMARK_FPCOMPRESS_UPSTREAM_ROOT",
        _root() / "external" / "compression" / "FPCompress",
    )


def fpcompress_bin_root() -> Path:
    """Built FPCompress CPU CLIs (outside the upstream submodule tree)."""
    return _resolve(
        "COMPRESSION_BENCHMARK_FPCOMPRESS_BIN_ROOT",
        _root() / "external" / "runtime" / "fpcompress",
    )


def has_submodule_checkout(path: Path) -> bool:
    """True if ``path`` exists and looks like a git work tree (submodule or clone)."""
    if not path.is_dir():
        return False
    git_meta = path / ".git"
    return git_meta.is_file() or git_meta.is_dir()
