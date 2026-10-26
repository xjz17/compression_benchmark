"""Current 129 algorithms; prior registries and concrete wire profiles are frozen."""
from __future__ import annotations
import json
from pathlib import Path
import canonical137_registry as previous137
from canonical137_registry import previous140, previous143, previous, previous156
import current_action_parameters as parameters
MANIFEST_PATH=Path(__file__).resolve().parents[1]/'benchmark/reference/actions/canonical129.json'
MANIFEST=json.loads(MANIFEST_PATH.read_text())
REGISTRY_VERSION=MANIFEST['registry_version'];PREVIOUS_REGISTRY_VERSION=previous137.REGISTRY_VERSION
GROUPS=tuple(MANIFEST['actions']);GROUP_BY_NAME={g['name']:g for g in GROUPS}
VARIANT_BY_NAME={v['name']:v for g in GROUPS for v in g['io_variants']}
VARIANT_TO_ALGORITHM={v['name']:g['name'] for g in GROUPS for v in g['io_variants']}
PREVIOUS_ALGORITHM_TO_CURRENT={previous137.GROUPS[i]['name']:g['name'] for g in GROUPS for i in g['previous_137_policy_ids']}
LEGACY_VARIANT_TO_CURRENT={v['name']:v['name'] for v in VARIANT_BY_NAME.values()}
LEGACY_VARIANT_TO_CURRENT.update({old:v['name'] for v in VARIANT_BY_NAME.values() for old in v.get('legacy_names',[])})
LEGACY_VARIANT_TO_CURRENT.update({old:LEGACY_VARIANT_TO_CURRENT[current] for old,current in previous137.LEGACY_VARIANT_TO_CURRENT.items()})
LEGACY_VARIANT_TO_ALGORITHM={old:VARIANT_TO_ALGORITHM[current] for old,current in LEGACY_VARIANT_TO_CURRENT.items()}
PREVIOUS_ALGORITHM_TO_CURRENT.update({old:PREVIOUS_ALGORITHM_TO_CURRENT[current] for old,current in previous137.PREVIOUS_ALGORITHM_TO_CURRENT.items()})
def canonical(name):return PREVIOUS_ALGORITHM_TO_CURRENT.get(name,LEGACY_VARIANT_TO_ALGORITHM.get(name,name))
MERGED_NAMES=frozenset(g['name'] for g in GROUPS if len(g['io_variants'])>1)
PARAMETERIZED_NAMES=frozenset(parameters.BY_NAME)
def validate_previous_registry(actions):
    previous137.validate_previous_registry(actions)
    if (len(GROUPS),len(VARIANT_BY_NAME))!=(129,165):raise RuntimeError('canonical129 inventory mismatch')
    seen=[]
    for i,g in enumerate(GROUPS):
        if g['policy_action_id']!=i or g['document_id']!=f'P{i+1:03d}':raise RuntimeError('canonical129 IDs must be dense')
        for old in g['previous_137_policy_ids']:
            if canonical(previous137.GROUPS[old]['name'])!=g['name']:raise RuntimeError('incomplete canonical137 migration')
            seen.append(old)
    if sorted(seen)!=list(range(137)):raise RuntimeError('canonical137 entries are not partitioned')
