"""Algorithm IDs after native-LZ bounds became invocation parameters.

The immutable canonical182 document inventory remains the migration source.
Old parameter-bearing names resolve to one current I/O profile plus parameters.
"""
from __future__ import annotations

import json
from pathlib import Path
import canonical182_registry as previous
import native_lz_parameters as lz

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/canonical156.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REGISTRY_VERSION = MANIFEST["registry_version"]
PREVIOUS_REGISTRY_VERSION = previous.REGISTRY_VERSION
GROUPS = tuple(MANIFEST["actions"])
GROUP_BY_NAME = {row["name"]: row for row in GROUPS}
VARIANT_BY_NAME = {v["name"]: v for row in GROUPS for v in row["io_variants"]}
VARIANT_TO_ALGORITHM = {v["name"]: row["name"] for row in GROUPS for v in row["io_variants"]}
LEGACY_VARIANT_TO_CURRENT = {
    name: lz.ALGORITHM_NAME if lz.parameters_from_name(name) is not None else name
    for name in previous.VARIANT_BY_NAME
}
LEGACY_VARIANT_TO_ALGORITHM = {
    name: VARIANT_TO_ALGORITHM[current] for name, current in LEGACY_VARIANT_TO_CURRENT.items()
}
PREVIOUS_ALGORITHM_TO_CURRENT = {
    row["name"]: (lz.ALGORITHM_NAME if lz.parameters_from_name(row["name"]) is not None else row["name"])
    for row in previous.GROUPS
}
MERGED_NAMES = frozenset(row["name"] for row in GROUPS if len(row["io_variants"]) > 1)
PARAMETERIZED_NAMES = frozenset({lz.ALGORITHM_NAME})


def validate_previous_registry(actions) -> None:
    previous.validate_previous_registry(actions)
    if len(GROUPS) != 156 or len(VARIANT_BY_NAME) != 180 or len(MERGED_NAMES) != 15:
        raise RuntimeError("canonical156 algorithm, profile or I/O merge counts disagree")
    covered = []
    for index, group in enumerate(GROUPS):
        if group["policy_action_id"] != index:
            raise RuntimeError("canonical156 IDs are not dense")
        for old_id in group["previous_182_policy_ids"]:
            old = previous.GROUPS[old_id]
            if PREVIOUS_ALGORITHM_TO_CURRENT[old["name"]] != group["name"] or old["operator_id"] != group["operator_id"]:
                raise RuntimeError("canonical156 migration changes algorithm family")
            covered.append(old_id)
    if sorted(covered) != list(range(182)):
        raise RuntimeError("canonical156 must cover each previous182 algorithm exactly once")
