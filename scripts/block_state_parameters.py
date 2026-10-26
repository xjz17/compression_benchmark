"""Current block codec parameters; historical selectors retain exact profiles.

These are native codec adapters, not new independent compression algorithms.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOSC1 = 'op.block_state_control.blosc1_residual_core.cpp_v2'
BLOSC2 = 'op.block_state_control.blosc2_residual_core.cpp_v2'
DEFLATE = 'op.block_state_control.deflate_residual_core.cpp_v2'
DEFLATE_VARIANTS = tuple(DEFLATE.replace('.cpp_v2', f'.{mode}.cpp_v2') for mode in ('gzip_prefix','zlib_prefix','corad_ctk2'))
DEFLATE_OLD = ('op.block_state_control.gzip_deflate_residual_core.cpp_v1', 'op.block_state_control.zlib_deflate_residual_core.cpp_v1', 'op.block_state_control.corad_deflate9.cpp_v2')

@dataclass(frozen=True)
class CodecParameters:
    ALGORITHM_NAME: str
    key: str
    option_key: str
    values: tuple[int, ...]

    def parameters_from_name(self, name):
        if not isinstance(name,str): return None
        name=name.replace('.python_v','.cpp_v')
        if self.ALGORITHM_NAME == DEFLATE:
            if name in DEFLATE_OLD:return {'level': (6,9,9)[DEFLATE_OLD.index(name)]}
        else:
            for n in (1,4,8):
                if name == self.ALGORITHM_NAME.replace('.cpp_v2',f'_t{n}.cpp_v1'):return {'typesize':n}
        return None

    def variant_for_name(self, name):
        if not isinstance(name,str):return None
        name=name.replace('.python_v','.cpp_v')
        if self.ALGORITHM_NAME == DEFLATE:
            if name in DEFLATE_VARIANTS:return name
            if name in DEFLATE_OLD:return DEFLATE_VARIANTS[DEFLATE_OLD.index(name)]
        elif name == self.ALGORITHM_NAME or self.parameters_from_name(name) is not None:return self.ALGORITHM_NAME
        return None

    def default_parameters(self, io_variant=None):
        if self.ALGORITHM_NAME == DEFLATE:
            variant=self.variant_for_name(io_variant)
            if variant is None:raise ValueError('DEFLATE requires explicit gzip_prefix, zlib_prefix, or corad_ctk2 I/O variant')
            return {'level': (6,9,9)[DEFLATE_VARIANTS.index(variant)]}
        return {'typesize':1}

    def normalize_parameters(self, parameters=None, *, io_variant=None):
        if parameters is None:return self.default_parameters(io_variant)
        if not isinstance(parameters,Mapping) or set(parameters)!={self.key}:raise ValueError(f'parameters require exactly {self.key}')
        value=parameters[self.key]
        if type(value) is not int or value not in self.values:raise ValueError(f'{self.key} must be an integer in {self.values}, not a boolean')
        return {self.key:value}

    def resolve_parameters(self, action, io_variant=None, parameters=None):
        historical=self.parameters_from_name(action)
        alias=self.parameters_from_name(io_variant)
        if historical is not None and alias is not None and historical!=alias:raise ValueError('historical action and profile parameters conflict')
        old=historical or alias
        selected=self.normalize_parameters(parameters if parameters is not None else old,io_variant=io_variant or self.variant_for_name(action))
        if old is not None and selected!=old:raise ValueError('parameters conflict with historical codec selector')
        return selected

    def parameter_candidates(self):return tuple({self.key:v} for v in self.values)
    def parameter_schema(self):
        schema={'schema_version':'block-codec-parameters-v1','properties':{self.key:{'type':'integer','enum':list(self.values)}},'required':[self.key],'additionalProperties':False,'search_domain':{self.key:list(self.values)},'learning_candidate_count':len(self.values),'option_key':self.option_key}
        if self.ALGORITHM_NAME==DEFLATE:
            schema['defaults_by_io_variant']={v:self.default_parameters(v) for v in DEFLATE_VARIANTS}
        else:
            schema['default']=self.default_parameters()
            schema['input_constraint']='typesize must equal the SHF1 element width recorded inside the reference prefix'
        if self.ALGORITHM_NAME==DEFLATE:schema['input_constraint']='select an explicit gzip_prefix, zlib_prefix, or corad_ctk2 I/O adapter; no magic-based guessing'
        schema['default_scope']='Historical codec-profile defaults; no universal compression-ratio optimum is asserted.'
        return schema
    def parameter_fingerprint(self):return hashlib.sha256(json.dumps(self.parameter_schema(),sort_keys=True).encode()).hexdigest()

MODULES=(CodecParameters(BLOSC1,'typesize','blosc1_typesize',(1,4,8)),CodecParameters(BLOSC2,'typesize','blosc2_typesize',(1,4,8)),CodecParameters(DEFLATE,'level','deflate_level',tuple(range(10))))
BY_NAME={m.ALGORITHM_NAME:m for m in MODULES}
def module_for_name(name):return BY_NAME.get(name) or next((m for m in MODULES if m.variant_for_name(name) is not None),None)
def run(name, data, parameters, *, io_variant=None, reverse=False):
    module=BY_NAME[name];p=module.normalize_parameters(parameters,io_variant=io_variant)
    if name==DEFLATE:
        if io_variant not in DEFLATE_VARIANTS:raise ValueError('DEFLATE requires explicit registered I/O variant')
        mode=('gzip_prefix','zlib_prefix','corad_ctk2')[DEFLATE_VARIANTS.index(io_variant)]
        exe='corad_lossless_cpp_cli' if mode=='corad_ctk2' else 'general_codec_residual_stages_cli'
        args=['deflate-inverse' if reverse else 'deflate-forward',mode]
        if not reverse:args.append(str(p['level']))
    else:
        exe='general_codec_residual_stages_cli'
        args=[('blosc1' if name==BLOSC1 else 'blosc2')+('-inverse' if reverse else '-forward'),str(p['typesize'])]
    process=subprocess.run([str(ROOT/'cpp/build'/exe),*args],input=bytes(data),capture_output=True,check=False)
    if process.returncode:raise ValueError(process.stderr.decode(errors='replace').strip())
    return process.stdout
