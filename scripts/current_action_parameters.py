"""Current parameter facade; frozen native catalogs remain byte-for-byte stable."""
from __future__ import annotations
import hashlib,json
from itertools import product
import native_action_parameters as native
import block_state_parameters as block
MODULES=(*native.MODULES,*block.MODULES)
BY_NAME={m.ALGORITHM_NAME:m for m in MODULES}
def module_for_name(name):return BY_NAME.get(name) or native.module_for_name(name) or block.module_for_name(name)
def parameters_from_name(name):
    m=module_for_name(name);return m.parameters_from_name(name) if m else None
def schemas():return {n:m.parameter_schema() for n,m in BY_NAME.items()}
def _keys(m):return {k:(m.option_key if m in block.MODULES else k) for k in m.parameter_candidates()[0]}
def _option(m,p):return {_keys(m)[k]:v for k,v in p.items()}
def choices_for_names(names):
    names=set(names)
    selected=[m for m in MODULES if m.ALGORITHM_NAME in names]
    if not selected:return ()
    return tuple({k:v for m,p in zip(selected,ps) for k,v in _option(m,p).items()} for ps in product(*(m.parameter_candidates() for m in selected)))
def all_candidates():return native.all_candidates()+tuple(_option(m,p) for m in block.MODULES for p in m.parameter_candidates())
def for_stage(name,parameters):
    m=BY_NAME.get(name)
    return {k:parameters[v] for k,v in _keys(m).items()} if m else {}
def normalize_option(names,parameters):
    names=set(names)
    selected=[m for m in MODULES if m.ALGORITHM_NAME in names]
    if not isinstance(parameters,dict):raise ValueError('option parameters must be a dictionary')
    expected={v for m in selected for v in _keys(m).values()}
    if set(parameters)!=expected:raise ValueError('option parameters must supply every parameterized stage exactly once')
    return {k:v for m in selected for k,v in _option(m,m.normalize_parameters(for_stage(m.ALGORITHM_NAME,parameters))).items()}
def resolve_parameters(name,variant=None,parameters=None):
    m=module_for_name(name)
    if m is None:
        if parameters not in (None,{}):raise ValueError('algorithm has no parameters')
        return {}
    if m in block.MODULES:return m.resolve_parameters(name,variant,parameters)
    old=m.parameters_from_name(name) or m.parameters_from_name(variant)
    chosen=m.normalize_parameters(parameters if parameters is not None else old)
    if old is not None and chosen!=old:raise ValueError('parameters conflict with historical selector')
    return chosen
def parameter_fingerprint():return hashlib.sha256(json.dumps({n:m.parameter_fingerprint() for n,m in BY_NAME.items()},sort_keys=True).encode()).hexdigest()
def canonical_io_variant(name):
    m=block.module_for_name(name)
    return (m.variant_for_name(name) or name) if m else (native.module_for_name(name).ALGORITHM_NAME if native.module_for_name(name) else name)
def for_option(name, stage_parameters):
    """Validate local stage parameters and return collision-free option keys."""
    module=BY_NAME.get(name)
    if module is None:
        if stage_parameters not in (None,{}):raise ValueError('algorithm has no parameters')
        return {}
    if not isinstance(stage_parameters,dict):raise ValueError('stage parameters must be a dictionary')
    return _option(module,module.normalize_parameters(stage_parameters))
