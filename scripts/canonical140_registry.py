"""Current algorithms after merging four max-value-width bitpack adapters."""
from __future__ import annotations
import json
from pathlib import Path
import canonical143_registry as previous143
from canonical143_registry import previous, previous156
import native_action_parameters as parameters
import max_value_bitpack as bitpack
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/canonical140.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
REGISTRY_VERSION = MANIFEST["registry_version"]
PREVIOUS_REGISTRY_VERSION = previous143.REGISTRY_VERSION
GROUPS = tuple(MANIFEST["actions"])
GROUP_BY_NAME = {row["name"]: row for row in GROUPS}
VARIANT_BY_NAME = {v["name"]: v for row in GROUPS for v in row["io_variants"]}
VARIANT_TO_ALGORITHM = {v["name"]: row["name"] for row in GROUPS for v in row["io_variants"]}
def canonical(name):
    if name in bitpack.SOURCE_VARIANTS or name == bitpack.PLAIN_VARIANT: return bitpack.ALGORITHM_NAME
    return previous143.canonical(name)
LEGACY_VARIANT_TO_CURRENT = {name: previous143.canonical(name) for name in set(previous.VARIANT_BY_NAME) | set(previous156.VARIANT_BY_NAME) | set(previous143.VARIANT_BY_NAME)}
LEGACY_VARIANT_TO_ALGORITHM = {name: VARIANT_TO_ALGORITHM[current] for name,current in LEGACY_VARIANT_TO_CURRENT.items()}
PREVIOUS_ALGORITHM_TO_CURRENT = {row["name"]: canonical(row["name"]) for row in (*previous.GROUPS,*previous156.GROUPS,*previous143.GROUPS)}
MERGED_NAMES = frozenset(row["name"] for row in GROUPS if len(row["io_variants"]) > 1)
PARAMETERIZED_NAMES = frozenset(parameters.BY_NAME)
def validate_previous_registry(actions):
    previous143.validate_previous_registry(actions)
    if (len(GROUPS),len(VARIANT_BY_NAME),len(MERGED_NAMES)) != (140,168,16):
        raise RuntimeError("canonical140 inventory counts disagree")
    covered=[]
    for index,group in enumerate(GROUPS):
        if group["policy_action_id"] != index: raise RuntimeError("canonical140 IDs are not dense")
        for old_id in group["previous_143_policy_ids"]:
            old=previous143.GROUPS[old_id]
            if canonical(old["name"]) != group["name"] or old["operator_id"] != group["operator_id"]:
                raise RuntimeError("canonical140 migration changes algorithm family")
            covered.append(old_id)
    if sorted(covered) != list(range(143)): raise RuntimeError("incomplete canonical143 migration")
