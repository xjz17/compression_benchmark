#!/usr/bin/env python3
"""Explicit smallest32 data binding for native experiments, without v1 relabeling.

Data admission alone never admits training or declares operator coverage. Test
payloads are read only by an explicit test-role consumer, never calibration.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator
import validate_weighted_grpo_formal_dataset_specs_v2 as gate

DTYPES = {'int32': 4, 'int64': 8, 'float32': 4, 'float64': 8}
SCHEMA = 'native-prepared-smallest32-binding-v1'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Block:
    dataset_id: str
    domain: str
    manifest_path: Path
    manifest_sha256: str
    provenance_sha256: str
    role: str
    record: dict[str, Any]

    @property
    def split(self) -> str:
        return self.record['split']

    @property
    def key(self) -> tuple:
        r = self.record
        return self.dataset_id, r['entity_id'], r['column'], r['split'], r['block_index']

    @property
    def fingerprint(self) -> str:
        return digest(canonical({'dataset_id': self.dataset_id, 'manifest_sha256': self.manifest_sha256,
                                 'provenance_sha256': self.provenance_sha256, 'block': self.record}))

    def payload(self) -> tuple[bytes, bytes]:
        r = self.record
        require(r['dtype'] in DTYPES and r['byte_order'] == 'little', 'unsupported original dtype/order')
        require(r['missing_bitmap_semantics'] == 'one_bit_means_missing_lsb_first', 'unknown bitmap semantics')
        values = gate.safe_child(self.manifest_path.parent, r['values_path']).read_bytes()
        bitmap = gate.safe_child(self.manifest_path.parent, r['missing_bitmap_path']).read_bytes()
        require(digest(values) == r['values_sha256'] and digest(bitmap) == r['missing_bitmap_sha256'], 'prepared payload hash drift')
        rows = r['row_count']
        require(type(rows) is int and rows >= 0 and len(bitmap) == (rows+7)//8, 'bitmap row count mismatch')
        require(rows % 8 == 0 or not bitmap or bitmap[-1] >> (rows % 8) == 0, 'nonzero bitmap padding')
        missing = sum(x.bit_count() for x in bitmap)
        valid = rows - missing
        require(valid == r['valid_count'] and missing == r['missing_count'], 'valid/missing counts changed')
        require(len(values) == valid * DTYPES[r['dtype']] == r['original_numeric_bytes'] == r['values_bytes'], 'original numeric byte denominator changed')
        require(len(values)+len(bitmap) == r['prepared_block_bytes'] and len(bitmap) == r['missing_bitmap_bytes'], 'prepared byte count changed')
        return values, bitmap

    def roots(self) -> list[dict[str, Any]]:
        values, bitmap = self.payload()
        return [{'kind': 'numeric', 'width': DTYPES[self.record['dtype']], 'data_hex': values.hex(),
                 'logical_view': self.record['dtype'], 'original_numeric_payload': True},
                {'kind': 'serial', 'width': 1, 'data_hex': bitmap.hex(),
                 'logical_view': 'bytes', 'original_numeric_payload': False}]

    def identity(self) -> dict[str, Any]:
        return {'dataset_id': self.dataset_id, 'domain': self.domain, 'role': self.role,
                'manifest_sha256': self.manifest_sha256, 'provenance_sha256': self.provenance_sha256,
                'block_fingerprint': self.fingerprint,
                **{k: self.record[k] for k in ('block_id','entity_id','column','source_column','dtype','split',
                    'block_index','row_count','valid_count','missing_count','original_numeric_bytes',
                    'prepared_block_bytes','first_timestamp','last_timestamp','values_sha256','missing_bitmap_sha256')}}


@dataclass(frozen=True)
class Corpus:
    receipt: dict[str, Any]
    blocks: tuple[Block, ...]
    role: str
    fingerprint: str

    def split(self, name: str) -> tuple[Block, ...]:
        allowed = {'train','validation'} if self.role == 'train_validation' else {'test'}
        require(name in allowed, f'{name} is not available to {self.role} consumer')
        return tuple(b for b in self.blocks if b.split == name)

    def summary(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'cohort_version': gate.VERSION, 'role': self.role,
                'corpus_fingerprint': self.fingerprint, 'dataset_count': len({b.dataset_id for b in self.blocks}),
                'split_blocks': dict(Counter(b.split for b in self.blocks)),
                'split_B_raw': {s: sum(b.record['original_numeric_bytes'] for b in self.blocks if b.split == s)
                                for s in sorted({b.split for b in self.blocks})},
                'zero_B_raw_blocks': sum(b.record['original_numeric_bytes'] == 0 for b in self.blocks),
                'data_admitted': True, 'formal_training_admitted': False,
                'native_execution_admitted': False}


def bind_receipt(receipt: dict[str, Any], *, role: str) -> Corpus:
    """Bind a freshly verified v2 result; payload hashes are rechecked on access.

    Public loaders must call load() to obtain the receipt from the verifier.
    This function exists for process handoff and deterministic contract tests.
    """
    require(role in {'train_validation','test_candidate'}, 'explicit recognized role required')
    require(receipt.get('schema') == 'weighted-grpo-formal-dataset-spec-gate-result-v2' and
            receipt.get('cohort_version') == gate.VERSION and receipt.get('passed') is True and
            receipt.get('scope') == 'data_admission_only', 'requires verified smallest32 v2 receipt')
    require(receipt.get('test_used_for_training_statistics') is False, 'test statistics contamination')
    require(receipt.get('frozen_contract_sha256') == gate.FROZEN_FILES, 'frozen source contract changed')
    prepared = receipt['prepared']
    ids = [r['dataset_id'] for r in prepared]
    require(len(ids) == len(set(ids)) == 32 and receipt['prepared_count'] == 32, 'missing/duplicate dataset')
    require(Counter(r['role'] for r in prepared) == {'train_validation':16,'test_candidate':16}, 'role counts changed')
    blocks = []
    selected = [r for r in prepared if r['role'] == role]
    for row in selected:
        p = Path(row['manifest']).resolve()
        data = p.read_bytes()
        require(digest(data) == row['manifest_sha256'], 'manifest changed after admission')
        manifest = json.loads(data)
        require(manifest['dataset_id'] == row['dataset_id'] and manifest['provenance_sha256'] == row['provenance_sha256'], 'manifest identity drift')
        require(manifest['cohort']['role'] == role and manifest['cohort']['version'] == gate.VERSION, 'manifest role/version mismatch')
        allowed = {'train','validation'} if role == 'train_validation' else {'test'}
        require({r['split'] for r in manifest['blocks']} == allowed, 'cross-role data leakage')
        for block in manifest['blocks']:
            blocks.append(Block(row['dataset_id'], manifest['domain'], p, row['manifest_sha256'], row['provenance_sha256'], role, block))
    blocks.sort(key=lambda b:b.key)
    require(len({b.key for b in blocks}) == len(blocks), 'duplicate source/entity/column/split/block')
    fingerprint = digest(canonical({'schema':SCHEMA,'cohort':gate.FROZEN_FILES,'role':role,
                                    'block_fingerprints':[b.fingerprint for b in blocks]}))
    return Corpus(receipt, tuple(blocks), role, fingerprint)


def load(cohort_path: Path, materialized: Path, *, role: str) -> Corpus:
    return bind_receipt(gate.validate_bundle(cohort_path=cohort_path, materialized=materialized), role=role)
