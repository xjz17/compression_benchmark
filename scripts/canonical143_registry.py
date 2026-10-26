"""Current algorithm inventory after native Window and LZ parameter normalization."""
from __future__ import annotations
import json
from pathlib import Path
import canonical182_registry as previous
import canonical156_registry as previous156
import native_action_parameters as parameters
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/canonical143.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REGISTRY_VERSION = MANIFEST["registry_version"]
PREVIOUS_REGISTRY_VERSION = previous156.REGISTRY_VERSION
GROUPS = tuple(MANIFEST["actions"])
GROUP_BY_NAME = {row["name"]: row for row in GROUPS}
VARIANT_BY_NAME = {v["name"]: v for row in GROUPS for v in row["io_variants"]}
VARIANT_TO_ALGORITHM = {v["name"]: row["name"] for row in GROUPS for v in row["io_variants"]}
def canonical(name):
    module = parameters.module_for_name(name)
    return module.ALGORITHM_NAME if module else name
LEGACY_VARIANT_TO_CURRENT = {name: canonical(name) for name in set(previous.VARIANT_BY_NAME) | set(previous156.VARIANT_BY_NAME)}
LEGACY_VARIANT_TO_ALGORITHM = {name: VARIANT_TO_ALGORITHM[current] for name,current in LEGACY_VARIANT_TO_CURRENT.items()}
PREVIOUS_ALGORITHM_TO_CURRENT = {row["name"]: canonical(row["name"]) for row in (*previous.GROUPS,*previous156.GROUPS)}
MERGED_NAMES = frozenset(row["name"] for row in GROUPS if len(row["io_variants"]) > 1)
PARAMETERIZED_NAMES = frozenset(parameters.BY_NAME)
def validate_previous_registry(actions):
    previous156.validate_previous_registry(actions)
    if (len(GROUPS),len(VARIANT_BY_NAME),len(MERGED_NAMES)) != (143,167,15):
        raise RuntimeError("canonical143 inventory counts disagree")
    covered=[]
    for index,group in enumerate(GROUPS):
        if group["policy_action_id"] != index: raise RuntimeError("canonical143 IDs are not dense")
        for old_id in group["previous_156_policy_ids"]:
            old=previous156.GROUPS[old_id]
            if canonical(old["name"]) != group["name"] or old["operator_id"] != group["operator_id"]:
                raise RuntimeError("canonical143 migration changes algorithm family")
            covered.append(old_id)
    if sorted(covered) != list(range(156)): raise RuntimeError("incomplete canonical156 migration")
