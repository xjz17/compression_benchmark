"""Current algorithms: one shared minimum-plus-offsets FOR implementation."""
from __future__ import annotations
import json
from pathlib import Path
import canonical140_registry as previous140
from canonical140_registry import previous143, previous, previous156
import native_action_parameters as parameters
import minimum_for as minimum
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/canonical137.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REGISTRY_VERSION = MANIFEST["registry_version"]
PREVIOUS_REGISTRY_VERSION = previous140.REGISTRY_VERSION
GROUPS = tuple(MANIFEST["actions"])
GROUP_BY_NAME = {row["name"]: row for row in GROUPS}
VARIANT_BY_NAME = {v["name"]: v for row in GROUPS for v in row["io_variants"]}
VARIANT_TO_ALGORITHM = {v["name"]: row["name"] for row in GROUPS for v in row["io_variants"]}
def canonical(name):
    if name in (*minimum.SOURCE_VARIANTS, minimum.PLAIN_VARIANT, minimum.PREVIOUS_WHOLE_NAME):
        return minimum.ALGORITHM_NAME
    return previous140.canonical(name)
LEGACY_VARIANT_TO_CURRENT = {**previous140.LEGACY_VARIANT_TO_CURRENT,
    **{name: name for name in previous140.VARIANT_BY_NAME}}
LEGACY_VARIANT_TO_ALGORITHM = {name: VARIANT_TO_ALGORITHM[current] for name,current in LEGACY_VARIANT_TO_CURRENT.items()}
PREVIOUS_ALGORITHM_TO_CURRENT = {row["name"]: canonical(row["name"]) for row in (*previous.GROUPS,*previous156.GROUPS,*previous143.GROUPS,*previous140.GROUPS)}
MERGED_NAMES = frozenset(row["name"] for row in GROUPS if len(row["io_variants"]) > 1)
PARAMETERIZED_NAMES = frozenset(parameters.BY_NAME)
def validate_previous_registry(actions):
    previous140.validate_previous_registry(actions)
    if (len(GROUPS),len(VARIANT_BY_NAME),len(MERGED_NAMES)) != (137,169,16):
        raise RuntimeError("canonical137 inventory counts disagree")
    covered=[]
    for index,group in enumerate(GROUPS):
        if group["policy_action_id"] != index: raise RuntimeError("canonical137 IDs are not dense")
        for old_id in group["previous_140_policy_ids"]:
            old=previous140.GROUPS[old_id]
            if canonical(old["name"]) != group["name"] or old["operator_id"] != group["operator_id"]:
                raise RuntimeError("canonical137 migration changes algorithm family")
            covered.append(old_id)
    if sorted(covered) != list(range(140)): raise RuntimeError("incomplete canonical140 migration")
