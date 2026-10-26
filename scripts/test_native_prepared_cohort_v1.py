#!/usr/bin/env python3
"""Boundary tests for the new cohort binding; no production receipt is forged."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import native_prepared_cohort_v1 as data


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.values = struct.pack('<II', 0x80000000, 0x7fc00017)
        self.mask = b'\x02'
        (self.root/'values').write_bytes(self.values)
        (self.root/'mask').write_bytes(self.mask)
        self.record = dict(dtype='float32', byte_order='little', values_path='values',
            missing_bitmap_path='mask', values_sha256=data.digest(self.values),
            missing_bitmap_sha256=data.digest(self.mask), row_count=3, valid_count=2,
            missing_count=1, original_numeric_bytes=8, values_bytes=8,
            missing_bitmap_bytes=1, prepared_block_bytes=9,
            missing_bitmap_semantics='one_bit_means_missing_lsb_first',
            split='train', entity_id='e', column='value', block_index=0)

    def block(self):
        return data.Block('sample','energy',self.root/'dataset_manifest.json','m','p',
                          'train_validation',self.record)

    def test_float_bits_and_numeric_denominator_preserved(self):
        roots = self.block().roots()
        self.assertEqual(bytes.fromhex(roots[0]['data_hex']),self.values)
        self.assertEqual(roots[0]['width'],4)
        self.assertTrue(roots[0]['original_numeric_payload'])
        self.assertFalse(roots[1]['original_numeric_payload'])

    def test_zero_numeric_bytes_kept(self):
        (self.root/'values').write_bytes(b'')
        (self.root/'mask').write_bytes(b'\x07')
        self.record.update(values_sha256=data.digest(b''), missing_bitmap_sha256=data.digest(b'\x07'),
            valid_count=0,missing_count=3,original_numeric_bytes=0,values_bytes=0,prepared_block_bytes=1)
        self.assertEqual(self.block().payload(),(b'',b'\x07'))

    def test_payload_drift_rejected_after_binding(self):
        (self.root/'values').write_bytes(b'12345678')
        with self.assertRaisesRegex(ValueError,'hash drift'): self.block().payload()

    def test_denominator_cannot_include_mask(self):
        self.record['original_numeric_bytes']=9
        with self.assertRaisesRegex(ValueError,'denominator'): self.block().payload()

    def test_rehashed_bitmap_padding_rejected(self):
        (self.root/'mask').write_bytes(b'\x82')
        self.record['missing_bitmap_sha256']=data.digest(b'\x82')
        with self.assertRaisesRegex(ValueError,'padding'): self.block().payload()

    def test_path_escape_and_symlink_rejected(self):
        for path in ('../outside','escape'):
            if path=='escape': (self.root/path).symlink_to(self.root.parent/'outside')
            self.record['values_path']=path
            with self.assertRaisesRegex(ValueError,'leaves'): self.block().payload()

    def test_train_consumer_cannot_access_test_split(self):
        corpus=data.Corpus({},(self.block(),),'train_validation','f')
        with self.assertRaisesRegex(ValueError,'not available'): corpus.split('test')


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.receipt=dict(schema='weighted-grpo-formal-dataset-spec-gate-result-v2',
            cohort_version=data.gate.VERSION,passed=True,scope='data_admission_only',
            test_used_for_training_statistics=False,frozen_contract_sha256=data.gate.FROZEN_FILES,
            prepared_count=32,prepared=[])
        for i in range(32):
            role='train_validation' if i<16 else 'test_candidate'
            p=self.root/f'{i}.json'
            blocks=[dict(split=s,entity_id='e',column='x',block_index=0,original_numeric_bytes=4)
                    for s in (['train','validation'] if i<16 else ['test'])]
            m=dict(dataset_id=str(i),domain='domain',provenance_sha256='p',
                   cohort=dict(role=role,version=data.gate.VERSION),blocks=blocks)
            p.write_bytes(data.canonical(m))
            self.receipt['prepared'].append(dict(dataset_id=str(i),role=role,manifest=str(p),
                manifest_sha256=data.digest(p.read_bytes()),provenance_sha256='p'))

    def test_train_binding_never_opens_test_manifest(self):
        for row in self.receipt['prepared'][16:]: Path(row['manifest']).unlink()
        c=data.bind_receipt(self.receipt,role='train_validation')
        self.assertEqual(len(c.split('train')),16)
        self.assertFalse(c.summary()['formal_training_admitted'])

    def test_manifest_drift_rejected(self):
        Path(self.receipt['prepared'][0]['manifest']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            data.bind_receipt(self.receipt,role='train_validation')

    def test_duplicate_source_rejected(self):
        self.receipt['prepared'][1]=copy.deepcopy(self.receipt['prepared'][0])
        with self.assertRaisesRegex(ValueError,'duplicate dataset'):
            data.bind_receipt(self.receipt,role='train_validation')

    def test_test_statistics_contamination_rejected(self):
        self.receipt['test_used_for_training_statistics']=True
        with self.assertRaisesRegex(ValueError,'contamination'):
            data.bind_receipt(self.receipt,role='train_validation')

    def test_public_loader_requires_fresh_complete_verification(self):
        with mock.patch.object(data.gate,'validate_bundle',side_effect=ValueError('source hash drift')) as v:
            with self.assertRaisesRegex(ValueError,'source hash drift'):
                data.load(self.root/'cohort',self.root,role='train_validation')
            v.assert_called_once()


if __name__=='__main__': unittest.main()
