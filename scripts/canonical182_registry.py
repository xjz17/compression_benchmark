"""Source-controlled algorithm identities and their reversible I/O variants.

Variant names identify adapters, not additional policy logits.  Keeping this
manifest separate from codec dispatch prevents a many-to-one ID migration from
silently choosing a different packet format.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/canonical182.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REGISTRY_VERSION = MANIFEST["registry_version"]
PREVIOUS_REGISTRY_VERSION = MANIFEST["previous_registry_version"]
GROUPS = tuple(MANIFEST["actions"])
GROUP_BY_NAME = {row["name"]: row for row in GROUPS}
VARIANT_BY_NAME = {
    variant["name"]: variant for row in GROUPS for variant in row["io_variants"]
}
VARIANT_TO_ALGORITHM = {
    variant["name"]: row["name"] for row in GROUPS for variant in row["io_variants"]
}
MERGED_NAMES = frozenset(row["name"] for row in GROUPS if len(row["io_variants"]) > 1)


def validate_previous_registry(actions) -> None:
    previous = {item.action_id: item for item in actions}
    if len(previous) != 206 or len(GROUPS) != 182 or len(MERGED_NAMES) != 15:
        raise RuntimeError("canonical182 inventory does not match the audited migration")
    seen = []
    for index, group in enumerate(GROUPS):
        if group["policy_action_id"] != index or group["document_id"] != f"A{index + 1:03d}":
            raise RuntimeError("canonical182 document and dense policy IDs disagree")
        for variant in group["io_variants"]:
            old_id = variant["legacy_206_policy_id"]
            old = previous[old_id]
            if old.name != variant["name"] or old.operator_id != group["operator_id"]:
                raise RuntimeError(f"canonical182 variant changed identity: {old_id}")
            seen.append(old_id)
    if sorted(seen) != list(range(206)) or len(GROUP_BY_NAME) != 182:
        raise RuntimeError("canonical182 variants must partition all 206 previous entries")

