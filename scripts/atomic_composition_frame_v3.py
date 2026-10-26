#!/usr/bin/env python3
"""ACF3: isolated ACF2 frame with a charged common-value mask primitive."""
import importlib.util
from pathlib import Path
import atomic_transform_extensions_v3 as transforms

_spec=importlib.util.spec_from_file_location('_isolated_acf3_extension',Path(__file__).with_name('atomic_composition_frame_v2.py'))
_v2=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_v2.transforms=transforms
_core=_v2._core
_core.transforms=transforms
_core.MAGIC=b'ACF3'
_core.VERSION=3
_read=_core._read_frame

def _read_frame(buf):
    info,payload=_read(buf)
    info['format']='ACF3'
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
