"""In-process cache for ``DatasetCatalogView`` JSON payloads (same results, faster repeat requests)."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Any

_lock = threading.Lock()
_build_locks: dict[tuple, threading.Lock] = {}
_cache: dict[tuple, dict[str, Any]] = {}


def catalog_response_cache_key(
    lite: bool,
    include_user_uploads: bool,
    user_id: int | None,
    user_uploads_revision: str = '',
    part: str = 'full',
) -> tuple:
    return (
        bool(lite),
        bool(include_user_uploads),
        user_id,
        str(user_uploads_revision or ''),
        str(part or 'full'),
    )


def get_cached_catalog_response(cache_key: tuple) -> dict[str, Any] | None:
    with _lock:
        payload = _cache.get(cache_key)
        return copy.deepcopy(payload) if payload is not None else None


def store_catalog_response(cache_key: tuple, payload: dict[str, Any]) -> None:
    with _lock:
        _cache[cache_key] = copy.deepcopy(payload)


def get_or_build_catalog_response(
    cache_key: tuple,
    builder: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Return a cached catalog payload or build once (single-flight per cache key)."""
    cached = get_cached_catalog_response(cache_key)
    if cached is not None:
        return cached

    build_lock = _build_locks.setdefault(cache_key, threading.Lock())
    with build_lock:
        cached = get_cached_catalog_response(cache_key)
        if cached is not None:
            return cached
        payload = builder()
        store_catalog_response(cache_key, payload)
        return copy.deepcopy(payload)


def clear_catalog_response_cache() -> None:
    with _lock:
        _cache.clear()
