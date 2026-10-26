#!/usr/bin/env python3
"""Active-method projection over the frozen primitive operator runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import primitive_operator_action_runtime as historical  # noqa: E402
from myapp.active_method_policy import (  # noqa: E402
    active_method_policy_summary,
    is_active_method_name,
)


ACTIVE_BENCHMARK_METHODS = tuple(
    method for method in historical.BENCHMARK_86_METHODS if is_active_method_name(method)
)


def require_active_method(method: str) -> str:
    name = str(method or "").strip()
    if not is_active_method_name(name):
        policy = active_method_policy_summary()
        raise ValueError(
            f"method {name!r} is retired by active-method policy {policy['policy_id']}"
        )
    return name


def active_action_pipelines() -> dict[str, Any]:
    historical_rows = historical.benchmark_86_action_pipelines()
    rows = {method: historical_rows[method] for method in ACTIVE_BENCHMARK_METHODS}
    expected = int(active_method_policy_summary()["active_method_count"])
    if len(rows) != expected:
        raise RuntimeError(f"active action pipeline count mismatch: {len(rows)} != {expected}")
    return rows


def active_exact_action_pipelines() -> dict[str, Any]:
    historical_rows = historical.benchmark_86_exact_action_pipelines()
    rows = {method: historical_rows[method] for method in ACTIVE_BENCHMARK_METHODS}
    expected = int(active_method_policy_summary()["active_method_count"])
    if len(rows) != expected:
        raise RuntimeError(f"active exact pipeline count mismatch: {len(rows)} != {expected}")
    return rows
