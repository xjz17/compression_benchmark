#!/usr/bin/env python3
"""Contract tests for the diagnostic 6,373-output flow-STOP migration."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

import weight_conditioned_grpo_v1 as grpo
from export_online_flow_stop_diagnostic_v2 import (
    BASE_ACTION_COUNT,
    BASE_REGISTRY_FINGERPRINT,
    FLOW_STOP_BEGIN,
    FLOW_STOP_COUNT,
    MIGRATION_FINGERPRINT_KEY,
    TARGET_ACTION_COUNT,
    TARGET_REGISTRY_FINGERPRINT,
    export_diagnostic,
)
from export_weighted_grpo_native_policy_v1 import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class FlowStopExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_temp = tempfile.TemporaryDirectory(prefix='flow-stop-policy-build-')
        provided = os.environ.get('NATIVE_POLICY_CLI')
        if provided:
            cls.cli = Path(provided).resolve()
        else:
            build = Path(cls.build_temp.name)
            subprocess.run(['cmake', '-S', str(ROOT / 'cpp/native_policy_v1'),
                            '-B', str(build), '-DCMAKE_BUILD_TYPE=Release'],
                           check=True, capture_output=True)
            subprocess.run(['cmake', '--build', str(build), '--parallel', '2'],
                           check=True, capture_output=True)
            cls.cli = build / 'weighted_grpo_native_policy_cli'
        if not cls.cli.is_file():
            raise RuntimeError('native policy CLI does not exist')

    @classmethod
    def tearDownClass(cls):
        cls.build_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='flow-stop-export-test-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rng = np.random.default_rng(6373)
        self.serial = 0
        self.identity = grpo.PolicyIdentity(
            BASE_REGISTRY_FINGERPRINT,
            (('dataset', 'data-sha'),), (('dataset', 'root-sha'),),
            BASE_ACTION_COUNT, 0, 18, 28, (128, 128), grpo.FEATURE_SCHEMA,
        )

    def checkpoint(self, identity=None) -> tuple[Path, grpo.TwoHiddenMLP]:
        identity = identity or self.identity
        policy = grpo.TwoHiddenMLP(identity, self.rng)
        checkpoint = self.directory / f'source-{identity.action_count}.json'
        grpo.save_checkpoint(grpo.TrainingResult(
            policy, (), grpo.GRPOConfig(hidden_sizes=(128, 128)), 6373, 0), checkpoint)
        return checkpoint, policy

    def native_infer(self, model, registry, actions, stop, features, mask,
                     *, must_pass=True):
        self.serial += 1
        source = self.directory / f'input-{self.serial}.bin'
        output = self.directory / f'output-{self.serial}.bin'
        source.write_bytes(b'WGPINP1\0' + struct.pack('<II', 1, features.size)
                           + np.asarray(features, dtype='<f8').tobytes()
                           + np.asarray(mask, dtype='u1').tobytes())
        command = [str(self.cli), 'infer', str(model), '--registry', registry,
                   '--features', grpo.FEATURE_SCHEMA, '--task-dim', '18',
                   '--state-dim', '28', '--actions', str(actions), '--stop', str(stop),
                   '--input', str(source), '--output', str(output)]
        result = subprocess.run(command, capture_output=True,
                                env={**os.environ, 'PATH': ''})
        if not must_pass:
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(b'identity mismatch', result.stderr)
            self.assertFalse(output.exists())
            return None
        self.assertEqual(result.returncode, 0, result.stderr)
        raw = output.read_bytes()
        rows, observed_actions = struct.unpack_from('<II', raw, 8)
        self.assertEqual((rows, observed_actions), (1, actions))
        return np.frombuffer(raw, dtype='<f8', count=actions, offset=16).copy()

    def test_identity_head_migration_and_old_identity_rejection(self):
        checkpoint, policy = self.checkpoint()
        old_model = self.directory / 'old.wgpn'
        new_model = self.directory / 'new.wgpn'
        receipt_path = self.directory / 'receipt.json'
        export_checkpoint(checkpoint, old_model)
        receipt = export_diagnostic(checkpoint, new_model, receipt_path)
        self.assertEqual(receipt, json.loads(receipt_path.read_text()))
        self.assertFalse(receipt['matched_training'])
        self.assertTrue(receipt['diagnostic_only'])
        self.assertEqual(receipt['registry_descriptor_sha256'],
                         TARGET_REGISTRY_FINGERPRINT)
        for field in ('data_fingerprints', 'root_fingerprints'):
            self.assertIn(MIGRATION_FINGERPRINT_KEY,
                          dict(receipt['target_identity'][field]))

        features = self.rng.normal(size=self.identity.input_dim)
        features[-3:] = [0.4, 0.3, 0.3]
        mask = np.ones(TARGET_ACTION_COUNT, dtype='u1')
        logits = self.native_infer(new_model, TARGET_REGISTRY_FINGERPRINT,
                                   TARGET_ACTION_COUNT, FLOW_STOP_BEGIN,
                                   features, mask)
        old_logits = self.native_infer(
            old_model, BASE_REGISTRY_FINGERPRINT, BASE_ACTION_COUNT, 0,
            features, np.ones(BASE_ACTION_COUNT, dtype='u1'))
        np.testing.assert_array_equal(logits[:BASE_ACTION_COUNT], old_logits)
        source_logits = grpo.FrozenPolicy(policy).forward(features[None, :])[0]
        np.testing.assert_allclose(logits[:BASE_ACTION_COUNT], source_logits,
                                   rtol=2e-12, atol=2e-13)
        np.testing.assert_array_equal(
            logits[FLOW_STOP_BEGIN:FLOW_STOP_BEGIN + FLOW_STOP_COUNT],
            np.repeat(logits[0], FLOW_STOP_COUNT))

        self.native_infer(new_model, BASE_REGISTRY_FINGERPRINT,
                          BASE_ACTION_COUNT, 0, features,
                          np.ones(BASE_ACTION_COUNT, dtype='u1'), must_pass=False)
        inspected = json.loads(subprocess.check_output(
            [str(self.cli), 'inspect', str(new_model)]))
        self.assertEqual(inspected['identity'], receipt['target_identity'])
        self.assertEqual(inspected['checkpoint_file_sha256'],
                         receipt['source_checkpoint_file_sha256'])

    def test_rejects_source_action_dimension_mismatch(self):
        checkpoint, _ = self.checkpoint(replace(
            self.identity, action_count=BASE_ACTION_COUNT - 1))
        with self.assertRaisesRegex(ValueError, 'invalid source action_count'):
            export_diagnostic(checkpoint, self.directory / 'bad.wgpn',
                              self.directory / 'bad.json')


if __name__ == '__main__':
    unittest.main()
