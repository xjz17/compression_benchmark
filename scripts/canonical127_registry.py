"""Frozen canonical127-v1 slots; the active sparse policy view is v2."""
from __future__ import annotations
import json
from pathlib import Path
import canonical129_registry as previous129
from canonical129_registry import previous137, previous140, previous143, previous, previous156
import current_action_parameters as parameters
import optimal_rle_actions as rle

MANIFEST_PATH = Path(__file__).resolve().parents[1] / 'benchmark/reference/actions/canonical127.json'
MANIFEST = json.loads(MANIFEST_PATH.read_text())
REGISTRY_VERSION = MANIFEST['registry_version']
PREVIOUS_REGISTRY_VERSION = previous129.REGISTRY_VERSION
GROUPS = tuple(MANIFEST['actions'])
GROUP_BY_NAME = {g['name']: g for g in GROUPS}
VARIANT_BY_NAME = {v['name']: v for g in GROUPS for v in g['io_variants']}
VARIANT_TO_ALGORITHM = {v['name']: g['name'] for g in GROUPS for v in g['io_variants']}
# These crosswalks upgrade current recipes. They never select historical wire
# decoders; unified_action_runtime preserves those using the source version.
PREVIOUS_ALGORITHM_TO_CURRENT = {previous129.GROUPS[i]['name']: g['name']
    for g in GROUPS for i in g['previous_129_policy_ids']}
PREVIOUS_ALGORITHM_TO_CURRENT.update({old: PREVIOUS_ALGORITHM_TO_CURRENT[current]
    for old, current in previous129.PREVIOUS_ALGORITHM_TO_CURRENT.items()})
LEGACY_VARIANT_TO_CURRENT = {name: name for name in VARIANT_BY_NAME}
LEGACY_VARIANT_TO_CURRENT.update({old: rle.upgrade_variant(current)
    for old, current in previous129.LEGACY_VARIANT_TO_CURRENT.items()})
LEGACY_VARIANT_TO_CURRENT.update(rle.OLD_TO_CURRENT)
LEGACY_VARIANT_TO_ALGORITHM = {old: VARIANT_TO_ALGORITHM[current]
    for old, current in LEGACY_VARIANT_TO_CURRENT.items()}
def canonical(name):
    return PREVIOUS_ALGORITHM_TO_CURRENT.get(name, LEGACY_VARIANT_TO_ALGORITHM.get(name, name))
MERGED_NAMES = frozenset(g['name'] for g in GROUPS if len(g['io_variants']) > 1)
PARAMETERIZED_NAMES = frozenset(parameters.BY_NAME)
def validate_previous_registry(actions):
    previous129.validate_previous_registry(actions)
    if (len(GROUPS), len(VARIANT_BY_NAME)) != (127, 165):
        raise RuntimeError('canonical127 inventory mismatch')
    seen = []
    for i, group in enumerate(GROUPS):
        if group['policy_action_id'] != i or group['document_id'] != f'P{i+1:03d}':
            raise RuntimeError('canonical127 IDs must be dense')
        for old in group['previous_129_policy_ids']:
            if canonical(previous129.GROUPS[old]['name']) != group['name']:
                raise RuntimeError('incomplete canonical129 crosswalk')
            seen.append(old)
    if sorted(seen) != list(range(129)):
        raise RuntimeError('canonical129 entries are not partitioned')
