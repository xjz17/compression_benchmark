"""Parameter dispatch and joint option catalog for native Window and LZ stages."""
from __future__ import annotations
import hashlib
import json
from itertools import product
import native_lz_parameters as lz
import native_window_parameters as window
MODULES = (lz, window)
BY_NAME = {module.ALGORITHM_NAME: module for module in MODULES}

def module_for_name(name):
    if name in BY_NAME: return BY_NAME[name]
    return next((m for m in MODULES if m.parameters_from_name(name) is not None), None)

def parameters_from_name(name):
    module = module_for_name(name)
    return module.parameters_from_name(name) if module else None

def schemas(): return {name: module.parameter_schema() for name, module in BY_NAME.items()}

def choices_for_names(names):
    names = set(names)
    selected = [m for m in MODULES if m.ALGORITHM_NAME in names]
    if not selected: return ()
    return tuple({key: value for choice in choices for key, value in choice.items()}
                 for choices in product(*(m.parameter_candidates() for m in selected)))

def all_candidates():
    return tuple(choice for m in MODULES for choice in m.parameter_candidates()) + choices_for_names(BY_NAME)

def normalize_option(names, parameters):
    names = set(names)
    selected = [m for m in MODULES if m.ALGORITHM_NAME in names]
    if not isinstance(parameters, dict): raise ValueError('option parameters must be a dictionary')
    required = {key for m in selected for key in m.parameter_candidates()[0]}
    if set(parameters) != required: raise ValueError('option parameters must supply every parameterized stage exactly once')
    return {key: value for m in selected for key, value in m.normalize_parameters(
        {k: parameters[k] for k in m.parameter_candidates()[0]}).items()}

def for_stage(name, parameters):
    module = BY_NAME.get(name)
    return {key: parameters[key] for key in module.parameter_candidates()[0]} if module else {}

def parameter_fingerprint():
    return hashlib.sha256(json.dumps({name:m.parameter_fingerprint() for name,m in BY_NAME.items()}, sort_keys=True).encode()).hexdigest()
