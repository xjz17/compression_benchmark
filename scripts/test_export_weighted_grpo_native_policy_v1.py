#!/usr/bin/env python3
"""Real compiled C++ parity/corruption tests; no real training or compression.

Set NATIVE_POLICY_CLI to reuse an independently built CLI; otherwise build with
the isolated CMake project in a temporary directory. No Python runs in the CLI.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

import weight_conditioned_grpo_v1 as grpo
from export_weighted_grpo_native_policy_v1 import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def seal(content):
    return bytes(content) + hashlib.sha256(content).digest()


def offsets(model):
    """Locate fields independently to exercise parser checks after valid rehash."""
    pos = 16
    texts = []
    for _ in range(5):
        n = struct.unpack_from('<I', model, pos)[0]
        texts.append((pos, pos + 4, n))
        pos += 4 + n
    dims = pos
    pos += 24
    pairs = []
    for _ in range(2):
        pairs.append(pos)
        count = struct.unpack_from('<I', model, pos)[0]
        pos += 4
        for _ in range(count * 2):
            n = struct.unpack_from('<I', model, pos)[0]
            pos += 4 + n
    parameter_count = pos
    pos += 4
    arrays = []
    for _ in range(6):
        rank = struct.unpack_from('<I', model, pos)[0]
        start = pos
        pos += 4 + 4 * rank
        count = struct.unpack_from('<Q', model, pos)[0]
        pos += 8
        arrays.append((start, pos, count))
        pos += 8 * count
    return dict(texts=texts, dims=dims, pairs=pairs, count=parameter_count, arrays=arrays)


class NativePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_temp = tempfile.TemporaryDirectory(prefix='native-policy-build-')
        provided = os.environ.get('NATIVE_POLICY_CLI')
        if provided:
            cls.cli = Path(provided).resolve()
            if not cls.cli.is_file():
                raise RuntimeError('NATIVE_POLICY_CLI does not exist')
        else:
            build = Path(cls.build_temp.name)
            subprocess.run(['cmake', '-S', str(ROOT/'cpp/native_policy_v1'), '-B', str(build),
                            '-DCMAKE_BUILD_TYPE=Release'], check=True, capture_output=True)
            subprocess.run(['cmake', '--build', str(build), '--parallel', '2'], check=True, capture_output=True)
            subprocess.run(['ctest', '--test-dir', str(build), '--output-on-failure'], check=True, capture_output=True)
            cls.cli = build/'weighted_grpo_native_policy_cli'

    @classmethod
    def tearDownClass(cls):
        cls.build_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='native-policy-test-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rng = np.random.default_rng(713)
        self.identity = grpo.PolicyIdentity('registry-v1-sha256-fixed',
            (('dataset-1', 'sha256-data-1'), ('来源-2', 'sha256-data-2')),
            (('dataset-1', 'sha256-root-1'), ('来源-2', 'sha256-root-2')),
            17, 16, 19, 23, (128, 128))
        self.policy = grpo.TwoHiddenMLP(self.identity, self.rng)
        self.serial = 0

    def export(self, policy=None):
        self.serial += 1
        checkpoint = self.directory/f'checkpoint-{self.serial}.json'
        model = self.directory/f'model-{self.serial}.bin'
        p = policy or self.policy
        grpo.save_checkpoint(grpo.TrainingResult(p, (), grpo.GRPOConfig(hidden_sizes=p.identity.hidden_sizes), 713, 0), checkpoint)
        receipt = export_checkpoint(checkpoint, model)
        return model, checkpoint, receipt

    def batch(self, rows=24, identity=None):
        identity = identity or self.identity
        x = self.rng.normal(size=(rows, identity.input_dim))
        x[:, -3:] = self.rng.dirichlet([1, 1, 1], size=rows)
        for i, w in enumerate(grpo.FIXED_WEIGHTS[:min(rows, 6)]):
            x[i, -3:] = w
        mask = (self.rng.random((rows, identity.action_count)) > .4).astype('u1')
        mask[:, identity.stop_action] = 1
        return x, mask

    def run_native(self, model, x, masks, *, expected=None, raw=None, must_pass=True):
        self.serial += 1
        source = self.directory/f'inputs-{self.serial}.bin'
        output = self.directory/f'results-{self.serial}.bin'
        identity = expected or self.identity
        if raw is None:
            raw = b'WGPINP1\0' + struct.pack('<II', len(x), x.shape[1])
            raw += b''.join(np.asarray(a, dtype='<f8').tobytes() + np.asarray(b, dtype='u1').tobytes()
                            for a, b in zip(x, masks))
        source.write_bytes(raw)
        argv = [str(self.cli), 'infer', str(model), '--registry', identity.registry_fingerprint,
                '--features', identity.feature_schema, '--task-dim', str(identity.task_feature_dim),
                '--state-dim', str(identity.state_feature_dim), '--actions', str(identity.action_count),
                '--stop', str(identity.stop_action), '--input', str(source), '--output', str(output)]
        # Empty executable search path proves the CLI does not need a subprocess interpreter.
        result = subprocess.run(argv, capture_output=True, env={**os.environ, 'PATH': ''})
        if not must_pass:
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(b'NATIVE_POLICY_ERROR', result.stderr)
            self.assertFalse(output.exists())
            return result.stderr.decode()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = output.read_bytes()
        self.assertEqual(data[:8], b'WGPRES1\0')
        rows, actions = struct.unpack_from('<II', data, 8)
        self.assertEqual((rows, actions), (len(x), identity.action_count))
        pos, logits, probs, orders = 16, [], [], []
        for _ in range(rows):
            logits.append(np.frombuffer(data, dtype='<f8', count=actions, offset=pos).copy());pos += 8*actions
            probs.append(np.frombuffer(data, dtype='<f8', count=actions, offset=pos).copy());pos += 8*actions
            n = struct.unpack_from('<I', data, pos)[0];pos += 4
            orders.append(list(struct.unpack_from('<' + 'I'*n, data, pos)));pos += 4*n
        self.assertEqual(pos, len(data))
        return np.array(logits), np.array(probs), orders, data

    def parity(self, policy=None, rows=24, x=None, masks=None):
        p = policy or self.policy
        model, _, _ = self.export(p)
        sample, generated_masks = self.batch(rows, p.identity)
        x = sample if x is None else x
        masks = generated_masks if masks is None else masks
        actual, probs, orders, _ = self.run_native(model, x, masks, expected=p.identity)
        expected = grpo.FrozenPolicy(p).forward(x)
        reference, _ = grpo._distribution(grpo.FrozenPolicy(p), x, masks.astype(bool))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(probs, reference, rtol=2e-12, atol=2e-14)
        self.assertTrue(np.all(probs[masks == 0] == 0))
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=3e-15, rtol=0)
        for i, order in enumerate(orders):
            self.assertEqual(order, sorted(np.flatnonzero(masks[i]), key=lambda j: (-actual[i, j], j)))
        return actual, expected

    def test_float64_numpy_parity_all_six_weights_and_random_masks(self):
        self.parity(rows=48)

    def test_large_action_space_4097_and_saturation(self):
        d = self.identity.to_json();d.update(action_count=4097, stop_action=4096)
        identity = grpo.PolicyIdentity.from_json(d)
        p = grpo.TwoHiddenMLP(identity, self.rng)
        x, _ = self.batch(8, identity);x[:, :-3] *= 1e4
        self.parity(p, rows=8, x=x)

    def test_frozen_production_action_dimension_6341(self):
        """The current registry has 6,341 fixed positions and STOP=0."""
        identity = grpo.PolicyIdentity(
            '50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd',
            self.identity.data_fingerprints, self.identity.root_fingerprints,
            6341, 0, 18, 28, (128, 128), grpo.FEATURE_SCHEMA,
        )
        policy = grpo.TwoHiddenMLP(identity, self.rng)
        x, _ = self.batch(2, identity)
        # Keep one legal action at each end and a sparse middle menu.
        masks = np.zeros((2, identity.action_count), dtype='u1')
        masks[0, [0, 17, 1024, 4096, 6340]] = 1
        masks[1, [0, 1, 3141, 6339]] = 1
        self.parity(policy, rows=2, x=x, masks=masks)

    def test_exact_ties_underflow_and_stop_only(self):
        for p in self.policy.params:p.fill(0)
        self.policy.params[-1][:] = -1000
        self.policy.params[-1][0] = 1000
        model, _, _ = self.export()
        x, masks = self.batch(3)
        masks[0] = 1;masks[0, 0] = 0
        masks[1] = 0;masks[1, self.identity.stop_action] = 1
        masks[2] = 1
        _, p, order, a = self.run_native(model, x, masks)
        self.assertEqual(order[0], list(range(1, 17)))
        self.assertEqual(order[1], [16]);self.assertEqual(p[1, 16], 1)
        self.assertEqual(p[2, 0], 1);self.assertTrue(np.all(p[2, 1:] == 0))
        self.assertEqual(a, self.run_native(model, x, masks)[3])

    def test_complete_identity_provenance_and_deterministic_export(self):
        model, checkpoint, receipt = self.export()
        observed = json.loads(subprocess.check_output([str(self.cli), 'inspect', str(model)]))
        self.assertEqual(observed['identity'], self.identity.to_json())
        self.assertEqual(observed['checkpoint_payload_sha256'], receipt['checkpoint_payload_sha256'])
        self.assertEqual(observed['checkpoint_file_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        second = self.directory/'second.bin'
        export_checkpoint(checkpoint, second)
        self.assertEqual(model.read_bytes(), second.read_bytes())
        with self.assertRaises(FileExistsError):export_checkpoint(checkpoint, model)

    def test_runtime_identity_mismatch_rejected_without_remapping(self):
        model, _, _ = self.export();x, masks = self.batch(1)
        for field, value in [('registry_fingerprint','different'),('feature_schema','different'),
                             ('action_count',18),('stop_action',15),('task_feature_dim',20),('state_feature_dim',24)]:
            with self.subTest(field=field):
                d = self.identity.to_json();d[field] = value
                # from_json rejects wrong feature schema earlier; use dataclass directly.
                from dataclasses import replace
                expected = replace(self.identity, **{field:value})
                self.assertIn('identity mismatch', self.run_native(model,x,masks,expected=expected,must_pass=False))

    def test_invalid_inputs_masks_and_objective_weights(self):
        model, _, _ = self.export();x, masks = self.batch(1)
        invalid = []
        a = x.copy();a[0, 0] = np.nan;invalid.append((a,masks))
        a = x.copy();a[0, 0] = np.inf;invalid.append((a,masks))
        a = x.copy();a[0, -3:] = [-.1, .6, .5];invalid.append((a,masks))
        a = x.copy();a[0, -3:] = [1, 1, 0];invalid.append((a,masks))
        a = masks.copy();a[0] = 0;invalid.append((x,a))
        a = masks.copy();a[0, 1] = 2;invalid.append((x,a))
        for a, b in invalid:self.run_native(model, a, b, must_pass=False)
        self.run_native(model, x[:, :-1], masks, must_pass=False)
        self.run_native(model, x, masks, raw=b'WGPINP1\0'+struct.pack('<II',1,x.shape[1])+b'garbage', must_pass=False)

    def reject_model(self, raw):
        self.serial += 1
        path = self.directory/f'corrupt-{self.serial}.bin';path.write_bytes(raw)
        result = subprocess.run([str(self.cli),'inspect',str(path)],capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(b'NATIVE_POLICY_ERROR',result.stderr)

    def test_checksum_truncation_and_trailing_bytes(self):
        model, _, _ = self.export();raw = model.read_bytes()
        for length in [0, 1, 15, 16, 47, len(raw)//2, len(raw)-1]:self.reject_model(raw[:length])
        self.reject_model(raw+b'junk')
        corrupt = bytearray(raw);corrupt[len(raw)//2] ^= 0x40;self.reject_model(corrupt)

    def test_valid_checksum_cannot_hide_wrong_shapes_or_nonfinite_parameters(self):
        model, _, _ = self.export();raw = model.read_bytes();o = offsets(raw)
        edits = [(o['dims'], '<I', 0), (o['dims']+4, '<I', 1000),
                 (o['dims']+16, '<I', 64), (o['count'], '<I', 5),
                 (o['arrays'][0][0], '<I', 1), (o['arrays'][0][0]+4, '<I', 999),
                 (o['arrays'][0][1], '<d', float('nan')),
                 (o['arrays'][5][1], '<d', float('inf')),
                 (o['pairs'][0], '<I', 0xffffffff), (8, '<Q', 0xffffffffffffffff)]
        for offset, fmt, value in edits:
            b = bytearray(raw[:-32]);struct.pack_into(fmt,b,offset,value);self.reject_model(seal(b))
        b = bytearray(raw[:-32]);b[o['texts'][4][1]] = ord('!');self.reject_model(seal(b))
        b = bytearray(raw[:-32]);b[o['texts'][3][1]] = 0xff;self.reject_model(seal(b))
        b = bytearray(raw[:-32])+b'junk';struct.pack_into('<Q',b,8,len(b)-16);self.reject_model(seal(b))

    def test_duplicate_fingerprint_identity_rejected_after_rehash(self):
        model, _, _ = self.export();raw = model.read_bytes();o = offsets(raw)
        # Equal-length keys make a local mutation without changing the envelope.
        b = bytearray(raw[:-32]);p = o['pairs'][0]+4
        first_size=struct.unpack_from('<I',b,p)[0];first=bytes(b[p+4:p+4+first_size]);p += 4+first_size
        value_size=struct.unpack_from('<I',b,p)[0];p += 4+value_size
        second_size=struct.unpack_from('<I',b,p)[0]
        # Splice rather than presume UTF-8 byte length equals character count.
        b[p:p+4+second_size]=struct.pack('<I',len(first))+first
        struct.pack_into('<Q',b,8,len(b)-16)
        self.reject_model(seal(b))

    def test_export_rejects_corrupt_checkpoint_and_unsupported_architecture(self):
        _, checkpoint, _ = self.export()
        doc=json.loads(checkpoint.read_text());doc['payload']['seed']+=1
        checkpoint.write_text(json.dumps(doc))
        with self.assertRaisesRegex(grpo.GraphContractError,'digest mismatch'):
            export_checkpoint(checkpoint,self.directory/'bad.bin')
        from dataclasses import replace
        p=grpo.TwoHiddenMLP(replace(self.identity,hidden_sizes=(32,32)),self.rng)
        with self.assertRaisesRegex(ValueError,'128-wide'):self.export(p)

    def test_nonfinite_affine_overflow_is_explicit_failure(self):
        self.policy.params[0].fill(1e308)
        model, _, _ = self.export();x,masks=self.batch(1);x[0,:-3]=1e308
        self.assertIn('non-finite affine',self.run_native(model,x,masks,must_pass=False))


if __name__ == '__main__':
    unittest.main()
