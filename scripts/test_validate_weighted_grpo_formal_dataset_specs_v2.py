#!/usr/bin/env python3
"""Tamper tests for new data binding and preservation of the legacy gate."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import validate_weighted_grpo_formal_dataset_specs_v2 as gate
import validate_weighted_grpo_formal_dataset_specs_v1 as legacy


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.cohort = {'version': gate.VERSION, 'assignment_sha256': 'assignment', 'policy_sha256': 'policy'}
        self.entry = {'id': 'sample', 'domain': 'energy', 'source_group': 'independent:sample', 'revision': 'revision', 'locator': 'https://example.invalid/sample', 'role': 'train_validation'}
        self.pin = {'repo_path': 'sample.tsfile', 'revision': 'revision', 'size': 100, 'sha256': 'source'}
        self.time = {'dataset_id': 'sample', 'timestamp': {'source_unit': 'us', 'output_unit': 'us', 'timezone': 'UTC', 'output_name': 'timestamp_unix_us', 'duplicate_policy': 'reject'}, 'evidence': [{'path': 'evidence', 'sha256': 'old-independent-evidence'}]}
        self.schema = {'table_name': 'sample', 'columns': [{'name': 'sensor', 'category': 'TAG', 'tsfile_type': 'STRING'}, {'name': 'x', 'category': 'FIELD', 'tsfile_type': 'FLOAT'}]}
        self.spec = {'dataset_id': 'sample', 'domain': 'energy', 'source_group': 'independent:sample', 'source': {'path': '/data/sample.tsfile', 'format': 'tsfile', 'expected_sha256': 'source', 'upstream_revision': 'revision', 'upstream_locator': 'https://example.invalid/sample'}, 'model': 'table', 'table': {'table_name': 'sample', 'entity_columns': ['sensor'], 'value_columns': [{'source_name': 'x', 'output_name': 'x', 'tsfile_type': 'FLOAT'}]}, 'timestamp': copy.deepcopy(self.time['timestamp']), 'dataset_byte_cap': 268435456, 'block_byte_cap': 1048576}
        self.manifest = {'dataset_id': 'sample', 'domain': 'energy', 'source_group': 'independent:sample', 'cohort': {**self.cohort, 'role': 'train_validation'}, 'split_contract': {'cohort_role': 'train_validation'}, 'blocks': [{'split': 'train'}, {'split': 'validation'}], 'timestamp_evidence': copy.deepcopy(self.time), 'source_content_pin': copy.deepcopy(self.pin), 'source': {'path': '/data/sample.tsfile', 'sha256': 'source', 'bytes': 100, 'upstream_revision': 'revision', 'upstream_locator': 'https://example.invalid/sample'}, 'input_contract': {'timestamp': copy.deepcopy(self.time['timestamp']), 'entity_columns': ['sensor'], 'numeric_columns': [{'source_name': 'time', 'output_name': 'timestamp_unix_us', 'dtype': 'int64', 'width_bytes': 8, 'role': 'timestamp'}, {'source_name': 'x', 'output_name': 'x', 'dtype': 'float32', 'width_bytes': 4, 'role': 'value'}]}, 'data_admission': {'cohort_adapter_sha256': gate.PRODUCER_SHA256, 'no_timezone_inference': True, 'no_row_index_timestamp_fallback': True, 'test_used_for_training_statistics': False}}
        self.manifest['input_contract']['timestamp'].update(source_name='time', source_kind='epoch', semantic_kind=None, formats=['integer_epoch_from_tsfile'], ambiguous_fold=None, invalid_policy='fail_no_unit_or_timezone_inference')
        self.rehash_spec()

    def rehash_spec(self):
        self.manifest['preparation_spec_sha256'] = hashlib.sha256(gate.canonical(self.spec)).hexdigest()

    def check(self):
        return gate.validate_record(self.entry, self.pin, self.time, self.spec, self.manifest, self.schema, self.cohort)

    def test_matching_record_passes(self):
        self.check()

    def test_wrong_source_hash_is_rejected_even_with_rehashed_spec(self):
        self.spec['source']['expected_sha256'] = 'wrong'
        self.rehash_spec()
        with self.assertRaisesRegex(gate.SpecGateError, 'hash/revision'):
            self.check()

    def test_cross_role_blocks_are_rejected(self):
        self.manifest['blocks'].append({'split': 'test'})
        with self.assertRaisesRegex(gate.SpecGateError, 'leakage'):
            self.check()

    def test_cross_role_cohort_is_rejected(self):
        self.manifest['cohort']['role'] = 'test_candidate'
        with self.assertRaisesRegex(gate.SpecGateError, 'role substitution'):
            self.check()

    def test_time_unit_is_not_widened(self):
        self.spec['timestamp']['source_unit'] = 'ms'
        self.spec['timestamp']['output_unit'] = 'ms'
        self.rehash_spec()
        with self.assertRaisesRegex(gate.SpecGateError, 'timestamp spec'):
            self.check()

    def test_utc_claim_cannot_be_added(self):
        self.spec['timestamp']['event_utc_instant_claim'] = True
        self.rehash_spec()
        with self.assertRaisesRegex(gate.SpecGateError, 'timestamp spec'):
            self.check()

    def test_old_timestamp_evidence_cannot_be_reused(self):
        self.time['evidence'][0]['sha256'] = 'new-independent-evidence'
        with self.assertRaisesRegex(gate.SpecGateError, 'stale or altered timestamp evidence'):
            self.check()

    def test_epoch_cannot_be_reinterpreted_as_row_index(self):
        self.manifest['input_contract']['timestamp']['source_kind'] = 'row_index'
        with self.assertRaisesRegex(gate.SpecGateError, 'timestamp unit/semantics/defaults'):
            self.check()

    def test_epoch_cannot_gain_an_unfrozen_semantic_kind(self):
        self.manifest['input_contract']['timestamp']['semantic_kind'] = 'source_naive_civil_time_coordinate'
        with self.assertRaisesRegex(gate.SpecGateError, 'timestamp unit/semantics/defaults'):
            self.check()

    def test_prepared_timestamp_cannot_enable_row_index_fallback(self):
        self.manifest['input_contract']['timestamp']['invalid_policy'] = 'allow_row_index_fallback'
        with self.assertRaisesRegex(gate.SpecGateError, 'timestamp unit/semantics/defaults'):
            self.check()

    def test_existing_manifest_cannot_bind_new_spec(self):
        self.spec['block_byte_cap'] -= 1
        with self.assertRaisesRegex(gate.SpecGateError, 'stale preparation spec'):
            self.check()

    def test_missing_numeric_column_is_rejected(self):
        self.schema['columns'].append({'name': 'y', 'category': 'FIELD', 'tsfile_type': 'INT32'})
        with self.assertRaisesRegex(gate.SpecGateError, 'missing/altered source numeric columns'):
            self.check()

    def test_tag_coverage_cannot_be_omitted(self):
        self.spec['table']['entity_columns'] = []
        self.rehash_spec()
        with self.assertRaisesRegex(gate.SpecGateError, 'missing/altered source numeric columns or tags'):
            self.check()

    def test_prepared_float_cannot_be_widened(self):
        self.manifest['input_contract']['numeric_columns'][1]['dtype'] = 'float64'
        self.manifest['input_contract']['numeric_columns'][1]['width_bytes'] = 8
        with self.assertRaisesRegex(gate.SpecGateError, 'widened prepared numeric'):
            self.check()


class DispatchTests(unittest.TestCase):
    def test_old_bundle_keeps_exact_old_receipt_and_blockers(self):
        expected = legacy.validate_bundle()
        actual = gate.validate_bundle(cohort_path=legacy.DEFAULT_COHORT)
        self.assertEqual(actual, expected)
        self.assertEqual(actual['schema'], 'weighted-grpo-formal-dataset-spec-gate-result-v1')
        self.assertFalse(actual['complete_cohort_trainable'])
        self.assertGreater(actual['blocked'], 0)

    def test_old_prepared_blocked_sources_still_rejected(self):
        with self.assertRaisesRegex(legacy.SpecGateError, 'FORMAL_DATASET_ADMISSION_BLOCKER'):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'manifest.json'
                p.write_text(json.dumps({'dataset_id': 'ETDataset-ett'}))
                gate.validate_bundle(cohort_path=legacy.DEFAULT_COHORT, prepared_manifests=[p])

    def test_old_frozen_revision_substitution_still_rejected(self):
        revisions = json.loads(legacy.DEFAULT_REVISIONS.read_text())
        revisions['sources'][0]['revision'] = '0' * 40
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'revisions.json'
            p.write_text(json.dumps(revisions))
            with self.assertRaises(legacy.SpecGateError):
                gate.validate_bundle(cohort_path=legacy.DEFAULT_COHORT, revisions_path=p)

    def test_missing_source_in_materialized_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'data_gate.json').write_text(json.dumps({'passed': True, 'failures': [], 'records': []}))
            with mock.patch.object(gate, 'load_contracts', return_value=({'version': gate.VERSION}, {}, {})):
                with self.assertRaisesRegex(gate.SpecGateError, 'missing/duplicate prepared sources'):
                    gate.validate_smallest32(root/'cohort.json', root)

    def test_unknown_version_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'cohort.json'
            p.write_text(json.dumps({'version': 'unreviewed-cohort'}))
            with self.assertRaisesRegex(gate.SpecGateError, 'unknown cohort version'):
                gate.validate_bundle(cohort_path=p)


if __name__ == '__main__':
    unittest.main()
