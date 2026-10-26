"""Layered implementation directory shared with the operator review PDF.

A IDs describe source-equivalence groups, not executable action IDs.  Frozen
canonical and graph registries still own their wire IDs and capability checks.
The 102 A IDs are trace entries, partitioned into separate accounting layers.
Reading this directory must never imply that all items are atomic or executable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from operator_catalog_layers import validate_layers


MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark/reference/actions/operator_implementations_layered_v1.json"
)
_RAW = MANIFEST_PATH.read_bytes()
_MANIFEST = json.loads(_RAW)
REGISTRY_VERSION = _MANIFEST["registry_version"]
MANIFEST_SHA256 = hashlib.sha256(_RAW).hexdigest()
IMPLEMENTATIONS = tuple(_MANIFEST["implementations"])
IMPLEMENTATION_COUNT = len(IMPLEMENTATIONS)
TRACE_ENTRY_COUNT = IMPLEMENTATION_COUNT
LAYER_COUNTS = validate_layers(IMPLEMENTATIONS)
MECHANISMS = tuple(item for item in IMPLEMENTATIONS if item["accounting"]["layer"] == "mechanism")
MECHANISM_COUNT = len(MECHANISMS)
if LAYER_COUNTS != _MANIFEST["layer_counts"] or MECHANISM_COUNT != _MANIFEST["mechanism_count"]:
    raise RuntimeError("layer accounting does not match the versioned catalog")
BY_ID = {item["implementation_id"]: item for item in IMPLEMENTATIONS}
RETIRED_BY_ID = {item["implementation_id"]: item for item in _MANIFEST["retired"]}
RETIRED_IDENTITY_ACTION_NAMES = frozenset(_MANIFEST["retired_identity_action_names"])

if IMPLEMENTATION_COUNT != _MANIFEST["count"] or IMPLEMENTATION_COUNT != 102:
    raise RuntimeError("102-item implementation directory count mismatch")
if len(BY_ID) != IMPLEMENTATION_COUNT or set(BY_ID) & set(RETIRED_BY_ID):
    raise RuntimeError("duplicate or retired implementation in current directory")
if set(BY_ID) != {f"A{i:03}" for i in range(1, 112)} - {"A005", "A017", "A024", "A025", "A026", "A034", "A036", "A071", "A074"}:
    raise RuntimeError("implementation IDs changed without an explicit migration")
if {n for item in IMPLEMENTATIONS for n in item["family_numbers"]} != set(range(1, 31)):
    raise RuntimeError("implementation directory no longer covers 30 families")
_source_entries = [p for item in IMPLEMENTATIONS for p in item["source_entry_ids"]]
if len(_source_entries) != 131 or len(set(_source_entries)) != 131:
    raise RuntimeError("active implementation source-entry mapping is inconsistent")
if set(_source_entries) | {"P006", "P019", "P036", "P088"} != {f"P{i:03}" for i in range(1, 136)}:
    raise RuntimeError("historical source-entry coverage is incomplete")


def describe(implementation_id: str) -> dict:
    """Return a detached current record; retired IDs are never selectable aliases."""
    if implementation_id in RETIRED_BY_ID:
        raise ValueError(f"retired implementation {implementation_id}")
    try:
        return deepcopy(BY_ID[implementation_id])
    except KeyError as exc:
        raise ValueError(f"unknown implementation {implementation_id!r}") from exc


def legacy_replacement(implementation_id: str) -> tuple[str, ...]:
    """Explicit recipe migration only; an empty tuple means identity bypass."""
    if implementation_id in RETIRED_BY_ID:
        if RETIRED_BY_ID[implementation_id].get("kind") in {"branched_pack", "record_dispatch"}:
            raise ValueError("branched migration requires replacement_recipe(), not a linear chain")
        return tuple(RETIRED_BY_ID[implementation_id]["replacement_implementation_ids"])
    describe(implementation_id)
    return (implementation_id,)


def replacement_recipe(implementation_id: str) -> dict:
    """Preserve topology, branch parameters and shared state during migration."""
    if implementation_id in RETIRED_BY_ID:
        return deepcopy(RETIRED_BY_ID[implementation_id])
    describe(implementation_id)
    return {"kind": "sequence", "replacement_implementation_ids": [implementation_id]}


def runtime_action_recipe(action_name: str) -> dict | None:
    """Describe native composition without pretending it exposes graph ports."""
    return deepcopy(_MANIFEST["runtime_action_recipes"].get(action_name))


def inventory() -> dict:
    """Return the versioned document inventory without asserting executability."""
    return deepcopy(_MANIFEST)


if __name__ == "__main__":
    print(json.dumps({
        "registry_version": REGISTRY_VERSION,
        "implementation_count": IMPLEMENTATION_COUNT,
        "trace_entry_count": TRACE_ENTRY_COUNT,
        "mechanism_count": MECHANISM_COUNT,
        "layer_counts": LAYER_COUNTS,
        "family_count": _MANIFEST["family_count"],
        "retired_implementation_ids": list(RETIRED_BY_ID),
        "retired_identity_action_names": sorted(RETIRED_IDENTITY_ACTION_NAMES),
        "manifest_sha256": MANIFEST_SHA256,
        "scope": _MANIFEST["scope"],
    }, ensure_ascii=False, indent=2))
