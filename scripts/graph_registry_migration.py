"""Version-specific graph migrations; no current numeric ID reinterprets old IDs."""
from __future__ import annotations
from pathlib import Path
import json,re
import native_action_parameters as native
import block_state_parameters as block

def frozen_atomic(action_id, version, legacy, *, io_variant=None, parameters=None):
    match=re.fullmatch(r'cpp-canonical-(\d+)-v1',version)
    if version=='legacy-225-v1':match=re.fullmatch(r'cpp-canonical-(\d+)-v1','cpp-canonical-225-v1')
    if match is None:raise ValueError('unknown frozen atomic registry')
    count=int(match.group(1)); path=Path(__file__).resolve().parents[1]/f'benchmark/reference/actions/canonical{count}.json'
    if type(action_id)is not int or not 0<=action_id<=count:raise ValueError('unknown frozen atomic selector')
    if action_id==count:
        if io_variant is not None or parameters:raise ValueError('STOP has no parameters or I/O')
        return {'policy_action_id':len(legacy.GROUPS),'io_variant':None,'parameters':{}}
    if count in (206,225):
        import primitive_operator_action_runtime as primitive
        old=(primitive.LEGACY_206_POLICY_ACTIONS if count==206 else primitive.LEGACY_225_POLICY_ACTIONS)[action_id]
        old_name=(old.name if count==206 else primitive.LEGACY_206_ACTION_CANONICAL_NAMES[old.name])
        source={'name':old.name,'io_variants':[{'name':old_name}]}
    elif path.is_file():source=json.loads(path.read_text())['actions'][action_id]
    else:raise ValueError('unknown frozen atomic registry')
    if version==legacy.REGISTRY_VERSION:target=legacy.GROUPS[action_id]
    else:
        if count in (206,225):
            canonical=legacy.LEGACY_VARIANT_TO_ALGORITHM.get(old_name,legacy.VARIANT_TO_ALGORITHM.get(old_name))
            choices=[g for g in legacy.GROUPS if g['name']==canonical]
        else:choices=[g for g in legacy.GROUPS if action_id in g.get(f'previous_{count}_policy_ids',[])]
        if len(choices)!=1:raise ValueError('no unambiguous mapping into frozen atomic namespace')
        target=choices[0]
    source_variants=[v['name'] for v in source['io_variants']]
    variant=io_variant
    if variant is not None:
        import primitive_operator_action_runtime as primitive
        variant=primitive.LEGACY_206_ACTION_CANONICAL_NAMES.get(variant,variant)
    if variant is not None and variant not in source_variants:raise ValueError('I/O variant does not belong to source ID')
    if variant is None and len(source_variants)==1:variant=source_variants[0]
    if variant is not None:variant=legacy.LEGACY_VARIANT_TO_CURRENT.get(variant,variant)
    if variant is not None and variant not in [v['name'] for v in target['io_variants']]:raise ValueError('I/O variant does not belong to target')
    block_module = block.BY_NAME.get(target['name']) if legacy.REGISTRY_VERSION == 'cpp-canonical-129-v1' else None
    module=native.BY_NAME.get(target['name'])
    if block_module:
        selected = block_module.resolve_parameters(source['name'], io_variant or variant, parameters)
    elif module:
        old=module.parameters_from_name(source['name']) or module.parameters_from_name(io_variant)
        selected=module.normalize_parameters(parameters if parameters is not None else old)
        if old is not None and selected!=old:raise ValueError('historical parameters conflict')
    else:
        if parameters not in (None,{}):raise ValueError('frozen algorithm has no parameters')
        selected={}
    return {'policy_action_id':target['policy_action_id'],'io_variant':variant,'parameters':selected}
