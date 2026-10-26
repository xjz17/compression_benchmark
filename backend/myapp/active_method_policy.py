"""Versioned active-method policy for new runs and active catalog views.

The frozen method catalog and historical results remain readable.  This module
only decides whether a method participates in newly derived active scopes.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


POLICY_SCHEMA = "tscb-active-method-policy-v1"
POLICY_FILENAME = "active_method_policy_v1.json"
POLICY_REPO_RELATIVE_PATH = "benchmark/reference/methods/active_method_policy_v1.json"


def active_method_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "reference"
        / "methods"
        / POLICY_FILENAME
    )


@lru_cache(maxsize=1)
def load_active_method_policy() -> dict:
    path = active_method_policy_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_SCHEMA:
        raise RuntimeError(f"Unsupported active-method policy schema: {payload.get('schema')!r}")
    retired = payload.get("retired_methods")
    if not isinstance(retired, list) or not retired:
        raise RuntimeError("Active-method policy must declare at least one retired method")
    catalog_count = int(payload.get("catalog_method_count", 0))
    active_count = int(payload.get("active_method_count", 0))
    if catalog_count - len(retired) != active_count:
        raise RuntimeError(
            "Active-method policy count mismatch: "
            f"catalog={catalog_count}, retired={len(retired)}, active={active_count}"
        )
    return payload


@lru_cache(maxsize=1)
def retired_method_names() -> frozenset[str]:
    return frozenset(
        str(alias).strip().casefold()
        for row in load_active_method_policy()["retired_methods"]
        for alias in row.get("method_aliases", [])
        if str(alias).strip()
    )


@lru_cache(maxsize=1)
def retired_algorithm_keys() -> frozenset[str]:
    return frozenset(
        str(key).strip().casefold()
        for row in load_active_method_policy()["retired_methods"]
        for key in row.get("algorithm_keys", [])
        if str(key).strip()
    )


def is_active_method_name(name: object) -> bool:
    value = str(name or "").strip()
    return bool(value) and value.casefold() not in retired_method_names()


def is_active_algorithm_key(key: object) -> bool:
    value = str(key or "").strip().removeprefix("\ufeff")
    return bool(value) and value.casefold() not in retired_algorithm_keys()


def active_method_policy_summary() -> dict:
    payload = load_active_method_policy()
    return {
        "schema": payload["schema"],
        "policy_id": payload["policy_id"],
        "policy_path": POLICY_REPO_RELATIVE_PATH,
        "policy_sha256": active_method_policy_sha256(),
        "catalog_method_count": int(payload["catalog_method_count"]),
        "active_method_count": int(payload["active_method_count"]),
        "retired_methods": [
            {
                "canonical_name": row["canonical_name"],
                "reason_code": row["reason_code"],
                "preserve_historical_results": bool(row["preserve_historical_results"]),
            }
            for row in payload["retired_methods"]
        ],
    }


def active_method_policy_sha256() -> str:
    return hashlib.sha256(active_method_policy_path().read_bytes()).hexdigest()


def active_method_policy_binding() -> dict:
    summary = active_method_policy_summary()
    return {
        "schema": summary["schema"],
        "policy_id": summary["policy_id"],
        "policy_path": summary["policy_path"],
        "policy_sha256": summary["policy_sha256"],
        "catalog_method_count": summary["catalog_method_count"],
        "active_method_count": summary["active_method_count"],
    }
