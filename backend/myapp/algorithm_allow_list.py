"""Allowlist helpers for partial staticdata benchmark runs."""

from __future__ import annotations

import os

from .active_method_policy import is_active_algorithm_key

ALLOWLIST_ENV_KEY = "WEB_COMPRESSION_RUN_ALGORITHMS"
INCREMENTAL_SKIP_ENV_KEY = "WEB_COMPRESSION_SKIP_ALGORITHMS"


def from_allowlist_env() -> set[str]:
    raw = os.environ.get(ALLOWLIST_ENV_KEY, "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def from_incremental_skip_env() -> set[str]:
    raw = os.environ.get(INCREMENTAL_SKIP_ENV_KEY, "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def has_allowlist() -> bool:
    return bool(from_allowlist_env())


def should_skip(incremental_skip: set[str] | None, algorithm: str | None) -> bool:
    """True when the algorithm should be skipped (not benchmarked / not flushed)."""
    if algorithm and not is_active_algorithm_key(algorithm):
        return True
    allowlist = from_allowlist_env()
    if allowlist:
        return not algorithm or algorithm not in allowlist
    if incremental_skip and algorithm:
        return algorithm in incremental_skip
    return False


def should_run(incremental_skip: set[str] | None, algorithm: str) -> bool:
    return not should_skip(incremental_skip, algorithm)
