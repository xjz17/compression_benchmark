#!/usr/bin/env python3
import copy
import unittest
import calibrate_native_smallest32_v1 as ref


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(split='train',dtype='float32',block_size_bucket_bytes=32,
                       compression_ns_per_original_byte=x,decompression_ns_per_original_byte=2*x,
                       block_fingerprint=str(x)) for x in (1,2,30)]

    def test_group_median_preserves_original_byte_units(self):
        r=ref.references(self.rows)
        self.assertEqual((r[0]['E0'],r[0]['D0'],r[0]['training_block_count']),(2,4,3))

    def test_nontraining_observations_rejected(self):
        for split in ('validation','test'):
            self.rows[0]['split']=split
            with self.assertRaisesRegex(ValueError,'non-training'): ref.references(self.rows)

    def test_missing_bucket_rule_is_frozen_and_uses_same_dtype(self):
        r=ref.references(self.rows)
        r += [dict(dtype='float32',block_size_bucket_bytes=128,E0=7,D0=8),
              dict(dtype='int64',block_size_bucket_bytes=64,E0=123,D0=123)]
        table=dict(schema=ref.SCHEMA,lookup_rule=ref.LOOKUP,references=r)
        out=ref.reference_for(table,'float32',48)
        self.assertEqual(out['block_size_bucket_bytes'],32)
        self.assertFalse(out['exact_size_bucket'])
        self.assertEqual(ref.reference_for(table,'float32',128)['E0'],7)
        with self.assertRaisesRegex(ValueError,'no training reference'):
            ref.reference_for(table,'int32',128)

    def test_invalid_time_or_size_rejected(self):
        for invalid in (float('nan'),-1,0):
            for r in self.rows: r['compression_ns_per_original_byte']=invalid
            with self.assertRaisesRegex(ValueError,'time scale'): ref.references(self.rows)
        for invalid in (0,True,1048577):
            with self.assertRaisesRegex(ValueError,'prepared bytes'): ref.bucket(invalid)


if __name__=='__main__': unittest.main()
