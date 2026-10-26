#!/usr/bin/env python3
"""Export a hash-verified GRPO checkpoint to portable C++ inference.

The model checksum detects corruption, not malicious authorship. Registry and
feature identity must also be independently pinned by the integrating runtime.
No graph features, graph search, model updates, or compression happen here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from weight_conditioned_grpo_v1 import CHECKPOINT_SCHEMA, load_checkpoint

MAGIC = b'WGPNAT1\0'
MAX_MODEL_BYTES = 256 * 1024 * 1024


def _u32(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= 0xffffffff:
        raise ValueError('uint32 field out of range')
    return struct.pack('<I', value)


def _text(value: str) -> bytes:
    data = value.encode('utf-8', errors='strict')
    if not data or len(data) > 1024 * 1024 or b'\0' in data:
        raise ValueError('invalid native identity string')
    return _u32(len(data)) + data


def _pairs(values) -> bytes:
    if not 0 < len(values) <= 65536:
        raise ValueError('invalid fingerprint count')
    return _u32(len(values)) + b''.join(_text(k) + _text(v) for k, v in values)


def _publish(path: Path, data: bytes) -> None:
    """Atomic no-overwrite publish; completed models are immutable artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # fails if any old model already occupies path
    finally:
        os.unlink(temporary)


def export_checkpoint(checkpoint: Path, destination: Path) -> dict:
    checkpoint, destination = Path(checkpoint), Path(destination)
    if checkpoint.stat().st_size > MAX_MODEL_BYTES * 2:
        raise ValueError('checkpoint file exceeds export limit')
    before = checkpoint.read_bytes()
    loaded = load_checkpoint(checkpoint)
    if checkpoint.read_bytes() != before:
        raise ValueError('checkpoint changed while exporting')
    document = json.loads(before)
    identity = loaded.policy.identity
    if document['payload']['identity'] != identity.to_json():
        raise ValueError('checkpoint identity is not canonical or contains unknown fields')
    if identity.hidden_sizes != (128, 128):
        raise ValueError('native policy v1 requires exactly two 128-wide hidden layers')
    if identity.input_dim > 65536 or identity.action_count > 65536:
        raise ValueError('native model dimension limit exceeded')
    file_sha = hashlib.sha256(before).hexdigest()
    body = b''.join(_text(v) for v in (CHECKPOINT_SCHEMA, loaded.payload_sha256,
                                       file_sha, identity.registry_fingerprint,
                                       identity.feature_schema))
    body += b''.join(_u32(v) for v in (identity.action_count, identity.stop_action,
                                      identity.task_feature_dim, identity.state_feature_dim,
                                      *identity.hidden_sizes))
    body += _pairs(identity.data_fingerprints) + _pairs(identity.root_fingerprints)
    body += _u32(6)
    for value in loaded.policy.params:
        value = np.asarray(value, dtype='<f8', order='C')
        if not np.isfinite(value).all():
            raise ValueError('non-finite model parameters')
        body += (_u32(value.ndim) + b''.join(_u32(v) for v in value.shape)
                 + struct.pack('<Q', value.size) + value.tobytes(order='C'))
    content = MAGIC + struct.pack('<Q', len(body)) + body
    model = content + hashlib.sha256(content).digest()
    if len(model) > MAX_MODEL_BYTES:
        raise ValueError('native model size limit exceeded')
    _publish(destination, model)
    return {
        'schema': 'weighted-grpo-native-policy-export-v1',
        'identity': identity.to_json(),
        'checkpoint_schema': CHECKPOINT_SCHEMA,
        'checkpoint_file_sha256': file_sha,
        'checkpoint_payload_sha256': loaded.payload_sha256,
        'native_model_sha256': hashlib.sha256(model).hexdigest(),
        'native_model_content_sha256': hashlib.sha256(content).hexdigest(),
        'native_model_bytes': len(model),
        'parameter_dtype': 'little_endian_float64',
        'forward': 'tanh(tanh(x @ W1 + b1) @ W2 + b2) @ W3 + b3',
        'scope': 'frozen_mlp_only_features_and_graph_search_are_external',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
