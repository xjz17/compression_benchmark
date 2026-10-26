#!/usr/bin/env python3
"""Derive a diagnostic per-flow-STOP WGPN model from a frozen v1 checkpoint.

This is an identity-changing migration for latency diagnostics.  It preserves
the 6,341 original output columns and initializes 32 new per-flow STOP columns
from legacy global STOP column 0.  The result has not been trained with the new
controls and must not be presented as a matched-training checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from export_weighted_grpo_native_policy_v1 import (
    MAGIC,
    MAX_MODEL_BYTES,
    _pairs,
    _publish,
    _text,
    _u32,
)
from weight_conditioned_grpo_v1 import CHECKPOINT_SCHEMA, FEATURE_SCHEMA, load_checkpoint

BASE_REGISTRY_FINGERPRINT = (
    '50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd'
)
BASE_ACTION_COUNT = 6341
DISABLED_GLOBAL_STOP_ACTION = 0
FLOW_STOP_BEGIN = 6341
FLOW_STOP_COUNT = 32
TARGET_ACTION_COUNT = FLOW_STOP_BEGIN + FLOW_STOP_COUNT
TARGET_STOP_ACTION = FLOW_STOP_BEGIN
MIGRATION_FINGERPRINT_KEY = 'online_control_migration'

REGISTRY_DESCRIPTOR = {
    'schema': 'native-online-flow-stop-controls-v2',
    'base_registry_fingerprint': BASE_REGISTRY_FINGERPRINT,
    'base_action_count': BASE_ACTION_COUNT,
    'disabled_global_stop_action': DISABLED_GLOBAL_STOP_ACTION,
    'flow_stop_begin': FLOW_STOP_BEGIN,
    'flow_stop_count': FLOW_STOP_COUNT,
    'action_count': TARGET_ACTION_COUNT,
    'slot_semantics': 'ascending_open_stream_id',
}
REGISTRY_DESCRIPTOR_CANONICAL = json.dumps(
    REGISTRY_DESCRIPTOR, sort_keys=True, separators=(',', ':'), ensure_ascii=True
).encode('ascii')
TARGET_REGISTRY_FINGERPRINT = hashlib.sha256(REGISTRY_DESCRIPTOR_CANONICAL).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return _sha(np.asarray(value, dtype='<f8', order='C').tobytes(order='C'))


def _migrated_pairs(values) -> tuple[tuple[str, str], ...]:
    pairs = tuple((str(key), str(value)) for key, value in values)
    if any(key == MIGRATION_FINGERPRINT_KEY for key, _ in pairs):
        raise ValueError('source identity already contains online control migration')
    migrated = tuple(sorted((*pairs, (MIGRATION_FINGERPRINT_KEY,
                                      TARGET_REGISTRY_FINGERPRINT))))
    if not migrated or any(migrated[i - 1][0] >= migrated[i][0]
                           for i in range(1, len(migrated))):
        raise ValueError('migrated fingerprints are not sorted and unique')
    return migrated


def _identity_json(source) -> dict:
    return {
        'registry_fingerprint': TARGET_REGISTRY_FINGERPRINT,
        'data_fingerprints': [list(value) for value in _migrated_pairs(
            source.data_fingerprints)],
        'root_fingerprints': [list(value) for value in _migrated_pairs(
            source.root_fingerprints)],
        'action_count': TARGET_ACTION_COUNT,
        'stop_action': TARGET_STOP_ACTION,
        'task_feature_dim': source.task_feature_dim,
        'state_feature_dim': source.state_feature_dim,
        'hidden_sizes': list(source.hidden_sizes),
        'feature_schema': source.feature_schema,
    }


def _validate_source(identity, params) -> None:
    expected = {
        'registry_fingerprint': BASE_REGISTRY_FINGERPRINT,
        'action_count': BASE_ACTION_COUNT,
        'stop_action': DISABLED_GLOBAL_STOP_ACTION,
        'task_feature_dim': 18,
        'state_feature_dim': 28,
        'hidden_sizes': (128, 128),
        'feature_schema': FEATURE_SCHEMA,
    }
    for field, wanted in expected.items():
        actual = getattr(identity, field)
        if actual != wanted:
            raise ValueError(
                f'invalid source {field}: expected {wanted!r}, got {actual!r}')
    shapes = ((49, 128), (128,), (128, 128), (128,),
              (128, BASE_ACTION_COUNT), (BASE_ACTION_COUNT,))
    if len(params) != len(shapes):
        raise ValueError('invalid source parameter count')
    for index, (value, shape) in enumerate(zip(params, shapes)):
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(
                f'invalid source parameter {index} shape: expected {shape}, got {array.shape}')
        if not np.isfinite(array).all():
            raise ValueError(f'non-finite source parameter {index}')


def _migrate_parameters(params) -> tuple[np.ndarray, ...]:
    migrated = [np.asarray(value, dtype='<f8', order='C').copy()
                for value in params]
    source_w3, source_b3 = migrated[4], migrated[5]
    stop_columns = np.repeat(source_w3[:, DISABLED_GLOBAL_STOP_ACTION:
                                      DISABLED_GLOBAL_STOP_ACTION + 1],
                             FLOW_STOP_COUNT, axis=1)
    stop_biases = np.repeat(source_b3[DISABLED_GLOBAL_STOP_ACTION:
                                     DISABLED_GLOBAL_STOP_ACTION + 1],
                            FLOW_STOP_COUNT)
    migrated[4] = np.ascontiguousarray(
        np.concatenate((source_w3, stop_columns), axis=1), dtype='<f8')
    migrated[5] = np.ascontiguousarray(
        np.concatenate((source_b3, stop_biases)), dtype='<f8')
    return tuple(migrated)


def export_diagnostic(checkpoint: Path, destination: Path,
                      receipt_path: Path) -> dict:
    checkpoint, destination, receipt_path = (Path(checkpoint), Path(destination),
                                             Path(receipt_path))
    if destination == receipt_path:
        raise ValueError('model and receipt paths must differ')
    if destination.exists() or receipt_path.exists():
        raise FileExistsError('output model and receipt must both be new paths')
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
    params = tuple(loaded.policy.params)
    _validate_source(identity, params)
    migrated = _migrate_parameters(params)
    target_identity = _identity_json(identity)

    source_file_sha = _sha(before)
    body = b''.join(_text(value) for value in (
        CHECKPOINT_SCHEMA, loaded.payload_sha256, source_file_sha,
        TARGET_REGISTRY_FINGERPRINT, identity.feature_schema,
    ))
    body += b''.join(_u32(value) for value in (
        TARGET_ACTION_COUNT, TARGET_STOP_ACTION, identity.task_feature_dim,
        identity.state_feature_dim, *identity.hidden_sizes,
    ))
    body += _pairs(tuple(map(tuple, target_identity['data_fingerprints'])))
    body += _pairs(tuple(map(tuple, target_identity['root_fingerprints'])))
    body += _u32(len(migrated))
    for value in migrated:
        value = np.asarray(value, dtype='<f8', order='C')
        body += (_u32(value.ndim) + b''.join(_u32(int(v)) for v in value.shape)
                 + struct.pack('<Q', value.size) + value.tobytes(order='C'))
    content = MAGIC + struct.pack('<Q', len(body)) + body
    model = content + hashlib.sha256(content).digest()
    if len(model) > MAX_MODEL_BYTES:
        raise ValueError('native model size limit exceeded')

    receipt = {
        'schema': 'native-online-flow-stop-policy-export-v2',
        'diagnostic_only': True,
        'matched_training': False,
        'source_checkpoint_schema': CHECKPOINT_SCHEMA,
        'source_checkpoint_file_sha256': source_file_sha,
        'source_checkpoint_payload_sha256': loaded.payload_sha256,
        'source_registry_fingerprint': identity.registry_fingerprint,
        'source_action_count': identity.action_count,
        'source_stop_action': identity.stop_action,
        'target_native_model_sha256': _sha(model),
        'target_native_model_content_sha256': _sha(content),
        'target_native_model_bytes': len(model),
        'target_identity': target_identity,
        'registry_descriptor': REGISTRY_DESCRIPTOR,
        'registry_descriptor_canonical_json': REGISTRY_DESCRIPTOR_CANONICAL.decode('ascii'),
        'registry_descriptor_sha256': TARGET_REGISTRY_FINGERPRINT,
        'parameter_source_sha256': [_array_sha(value) for value in params],
        'parameter_target_sha256': [_array_sha(value) for value in migrated],
        'migration': {
            'base_action_ids_preserved': [0, BASE_ACTION_COUNT - 1],
            'disabled_global_stop_action': DISABLED_GLOBAL_STOP_ACTION,
            'flow_stop_action_ids': [FLOW_STOP_BEGIN, TARGET_ACTION_COUNT - 1],
            'flow_stop_slot_semantics': 'ascending_open_stream_id',
            'initialization': 'clone_legacy_global_stop_head_column_and_bias',
            'legacy_global_stop_head_source_id': DISABLED_GLOBAL_STOP_ACTION,
            'training_status': 'not_trained_for_per_flow_stop_controls',
            'binary_origin_semantics': (
                'WGPN origin hashes identify the frozen source checkpoint; '
                'the target is an identity-changing diagnostic derivative'
            ),
        },
        'claim_boundary': (
            'Latency and wiring diagnostics only; no matched-training model-quality claim'
        ),
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    _publish(destination, model)
    _publish(receipt_path, receipt_bytes)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = export_diagnostic(args.checkpoint, args.output, args.receipt)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
