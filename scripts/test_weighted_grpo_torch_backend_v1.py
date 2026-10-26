"""Numerical policy tests only; these never execute compression or real training."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import weight_conditioned_grpo_v1 as grpo
import weighted_grpo_torch_backend_v1 as backend

try:
    import torch
except ImportError:
    torch = None


class TorchPolicyTests(unittest.TestCase):
    def policy_and_batch(self):
        rng = np.random.default_rng(72)
        identity = grpo.PolicyIdentity("registry", (("toy", "data"),),
                                       (("toy", "root"),), 5, 4, 2, 3, (128, 128))
        policy = grpo.TwoHiddenMLP(identity, rng)
        features = rng.normal(size=(24, identity.input_dim))
        masks = rng.random((24, 5)) > .3
        masks[:, 4] = True
        probabilities, _ = grpo._distribution(policy, features, masks)
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in masks])
        # Include positively and negatively clipped likelihood ratios.
        old_logp = np.log(probabilities[np.arange(24), actions]) + np.tile([-.5, 0, .5], 8)
        batch = dict(features=features, legal=masks, actions=actions, old_logp=old_logp,
                     advantages=rng.normal(size=24), token_weights=rng.uniform(.1, 1, 24))
        return policy, batch

    def assert_update_parity(self, device):
        left, batch = self.policy_and_batch()
        right = copy.deepcopy(left)
        config = grpo.GRPOConfig()
        reference = grpo.Adam(left.params, config.learning_rate)
        actual = grpo.Adam(right.params, config.learning_rate)
        for _ in range(2):
            expected = grpo._update(left, reference, batch, config)
            result = backend.update_policy(right, actual, batch, config, device=device)
            for a, b in zip(left.params + reference.first + reference.second,
                            right.params + actual.first + actual.second):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=2e-12)
            self.assertEqual(reference.steps, actual.steps)
            for key in expected:
                self.assertAlmostEqual(result[key], expected[key], places=10)
            # Restore portable moments exactly as a resume checkpoint does.
            restored = grpo.Adam(right.params, config.learning_rate)
            restored.first = [a.copy() for a in actual.first]
            restored.second = [a.copy() for a in actual.second]
            restored.steps = actual.steps
            actual = restored
        frozen = backend.frozen_parameters(right.params, device)
        output = backend.frozen_forward(batch["features"], frozen, device)
        np.testing.assert_allclose(output, grpo.FrozenPolicy(right).forward(batch["features"]),
                                   rtol=1e-9, atol=2e-12)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_cpu_matches_numpy_clipping_entropy_adam_and_resume(self):
        self.assert_update_parity("cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_numpy_and_dispatch_executes_on_gpu(self):
        self.assert_update_parity("cuda:0")
        policy, batch = self.policy_and_batch()
        before = [p.copy() for p in policy.params]
        optimizer = grpo.Adam(policy.params, 3e-4, device="cuda:0")
        result = grpo._update(policy, optimizer, batch, grpo.GRPOConfig())
        self.assertEqual(result["policy_update_device"], "cuda:0")
        self.assertEqual(optimizer.steps, 4)
        self.assertGreater(sum(float(np.sum((a-b)**2)) for a, b in zip(before, policy.params)), 0.)
        frozen = grpo.FrozenPolicy(policy, device="cuda:0")
        self.assertEqual(frozen._device_params[0].device.type, "cuda")
        np.testing.assert_allclose(frozen.forward(batch["features"]),
                                   grpo.FrozenPolicy(policy).forward(batch["features"]),
                                   rtol=1e-9, atol=2e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_policy.json"
            digest = grpo.save_checkpoint(
                grpo.TrainingResult(policy, (), grpo.GRPOConfig(), 72, 1), path,
                extra={"scope": "synthetic_policy_update_only"})
            loaded = grpo.load_checkpoint(path, expected_registry_fingerprint="registry")
            self.assertEqual(loaded.payload_sha256, digest)
            for expected, actual in zip(policy.params, loaded.policy.params):
                np.testing.assert_array_equal(expected, actual)
            restored = grpo.FrozenPolicy(loaded.policy, device="cuda:0")
            np.testing.assert_array_equal(restored.forward(batch["features"]),
                                          frozen.forward(batch["features"]))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_unavailable_cuda_never_falls_back_to_cpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            policy, _ = self.policy_and_batch()
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                grpo.Adam(policy.params, 3e-4, device="cuda:0")
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                grpo.FrozenPolicy(policy, device="cuda:0")

    def test_only_one_explicit_device_is_accepted(self):
        self.assertEqual(backend.normalize_device("cuda"), "cuda:0")
        for value in ("cuda:0,1", "mps", "cuda:-1", "auto"):
            with self.assertRaises(ValueError):
                backend.normalize_device(value)


if __name__ == "__main__":
    unittest.main()
