#!/usr/bin/env python3
"""Isolated ACF2 extension: explicit block/exception streams and bit models.

Loads an independent instance of the frozen ACF1 framing implementation. It
does not mutate the imported/public ACF1 module or any running v1 experiment.
New inverse streams are stored as binary blobs, never external dictionaries.
"""
import base64
import importlib.util
from pathlib import Path
import subprocess

import atomic_transform_extensions_v2 as transforms

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('_isolated_acf2_core',Path(__file__).with_name('atomic_composition_frame_v1.py'))
_core=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
_core.transforms=transforms
_core.MAGIC=b'ACF2'
_core.VERSION=2
BINARY_CLI=ROOT/'cpp/build/primitive_binary_entropy_v2/primitive_binary_entropy_v2'
BIT_TERMINALS=('bit4','bit8','bit12','bit16')
_core.TERMINALS={**_core.TERMINALS,**{name:5+i for i,name in enumerate(BIT_TERMINALS)}}
_core.PRIMITIVE_TERMINALS=frozenset((*_core.PRIMITIVE_TERMINALS,*BIT_TERMINALS))
_core.TERMINAL_DETAILS={**_core.TERMINAL_DETAILS,**{name:
    'standalone binary arithmetic, preceding '+name[3:]+' bits context; full PBV2 header counted'
    for name in BIT_TERMINALS}}
_original_primitive=_core._primitive


def _primitive(name,data,reverse=False):
    if name not in BIT_TERMINALS:
        return _original_primitive(name,data,reverse)
    result=subprocess.run([str(BINARY_CLI),name+('-decode' if reverse else '-encode')],
        input=data,capture_output=True,timeout=_core.SUBPROCESS_TIMEOUT)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace') or 'binary entropy failure')
    return result.stdout


_core._primitive=_primitive
_serialize=_core.serialize_metadata
_deserialize=_core.deserialize_metadata


def _convert(value,decode=False):
    if isinstance(value,list):
        return [_convert(v,decode) for v in value]
    if not isinstance(value,dict):
        return value
    out={}
    for k,v in value.items():
        if k.endswith('_b64') and k!='palette_b64':
            if decode:
                if not isinstance(v,bytes):raise ValueError('ACF2 side stream must be binary')
                out[k]=base64.b64encode(v).decode('ascii')
            else:
                if not isinstance(v,str):raise ValueError('side stream must be canonical base64')
                blob=base64.b64decode(v,validate=True)
                if base64.b64encode(blob).decode('ascii')!=v:raise ValueError('noncanonical side stream')
                out[k]=blob
        else:
            out[k]=_convert(v,decode)
    return out


_core.serialize_metadata=lambda value:_serialize(_convert(value))
_core.deserialize_metadata=lambda value:_convert(_deserialize(value),True)


def _validate_metadata(metadata,raw_length,payload_length):
    if not isinstance(metadata,list) or len(metadata)>10:
        raise ValueError('invalid inverse metadata sequence')
    kind,count,length='u64',raw_length//8,raw_length
    byte_ops={'compact','bitpack','shuffle','bittranspose','block_bitpack'}
    generic={'op','in_kind','in_count','in_nbytes','out_kind','out_count','out_nbytes'}
    for record in metadata:
        if not isinstance(record,dict) or record.get('op') not in transforms.SUPPORTED_OPERATORS:
            raise ValueError('unknown ACF2 operator')
        op=record['op']
        if set(record)!=generic|set(transforms.INVERSE_FIELDS[op]):
            raise ValueError('missing/unexpected inverse fields')
        if (record['in_kind'],record['in_count'],record['in_nbytes'])!=(kind,count,length):
            raise ValueError('disconnected inverse typed state')
        if op not in ('shuffle','bittranspose') and kind!='u64':
            raise ValueError('numeric operation after bytes')
        kind=record['out_kind']
        if kind!=('bytes' if op in byte_ops else 'u64'):
            raise ValueError('wrong operator output kind')
        count=_core._uint(record['out_count'],_core.MAX_BODY_BYTES)
        length=_core._uint(record['out_nbytes'],_core.MAX_BODY_BYTES)
        if count*(8 if kind=='u64' else 1)!=length:
            raise ValueError('typed output length mismatch')
    if length!=payload_length:
        raise ValueError('final inverse payload length mismatch')


_core._validate_metadata=_validate_metadata
_validate_prepared_original=_core._validate_prepared


def _validate_prepared(raw,specs,payload,metadata,terminals):
    _validate_prepared_original(raw,specs,payload,metadata,terminals)
    for spec,record in zip(specs,metadata):
        if spec['op'] in ('block_for','block_bitpack'):
            if record['block_size']!=spec.get('block_size',128):
                raise ValueError('block size differs from requested spec')
        elif spec['op']=='float_scale_exceptions':
            if record['factor']!=spec.get('factor',1) or record['max_fraction']!=float(spec.get('max_fraction',.05)):
                raise ValueError('exception scaling parameters differ from requested spec')


_core._validate_prepared=_validate_prepared
_read_original=_core._read_frame


def _read_frame(frame):
    info,payload=_read_original(frame)
    info['format']='ACF2'
    return info,payload


_core._read_frame=_read_frame
encode_frame=_core.encode_frame
encode_prepared=_core.encode_prepared
decode_frame=_core.decode_frame
inspect_frame=_core.inspect_frame
make_body=_core.make_body
serialize_metadata=_core.serialize_metadata
deserialize_metadata=_core.deserialize_metadata
TERMINALS=_core.TERMINALS
PRIMITIVE_TERMINALS=_core.PRIMITIVE_TERMINALS
ALGORITHM_TERMINALS=_core.ALGORITHM_TERMINALS
