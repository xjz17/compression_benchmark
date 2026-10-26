#!/usr/bin/env python3
"""Bounded real-source smoke tests. Never instantiate the 32 GB cmix model."""
import math
import random
from pathlib import Path
import subprocess
import unittest

from prediction_probability import (
    ROOT, PROVIDERS, PredictionSession, PredictorUnavailable, build_driver,
    predict_probabilities,
)


class ProbabilityProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = build_driver()

    def test_all_real_bounded_providers_are_deterministic_and_causal(self):
        for provider in ("ppmd", "dmc", "paq"):
            with self.subTest(provider=provider):
                first = predict_probabilities(b"ABABABAB", provider)
                repeated = predict_probabilities(b"ABABABAB", provider)
                changed_suffix = predict_probabilities(b"ABABxxxx", provider)
                self.assertEqual(len(first), 64)
                self.assertEqual(first, repeated)
                # Even the first changed bit's probability must be prefix-only.
                self.assertEqual(first[:33], changed_suffix[:33])
                self.assertEqual(first[0], 0.5)
                if provider != "paq":
                    self.assertEqual(first[:8], [0.5] * 8)
                self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in first))
                self.assertTrue(any(p != 0.5 for p in first[8:]))

    def test_interactive_matches_batch_and_predict_is_idempotent(self):
        data = b"ABA"
        for provider in ("ppmd", "dmc", "paq"):
            with self.subTest(provider=provider):
                output = []
                with PredictionSession(provider) as session:
                    for byte in data:
                        for shift in range(7, -1, -1):
                            p = session.predict()
                            self.assertEqual(p, session.predict())
                            output.append(p)
                            session.observe((byte >> shift) & 1)
                self.assertEqual(output, predict_probabilities(data, provider))

    def test_ppmd_exposes_actual_byte_distribution(self):
        with PredictionSession("ppmd") as session:
            self.assertEqual(session.byte_distribution(), [1 / 256] * 256)
            for shift in range(7, -1, -1):
                session.predict()
                session.observe((65 >> shift) & 1)
            distribution = session.byte_distribution()
            self.assertAlmostEqual(sum(distribution), 1, places=5)
            self.assertGreater(distribution[65], distribution[66])
            self.assertTrue(all(p > 0 for p in distribution))
            self.assertAlmostEqual(session.predict(), sum(distribution[128:]) / sum(distribution), places=5)

    def test_protocol_order_and_validation(self):
        with PredictionSession("dmc") as session:
            with self.assertRaises(ValueError):
                session.observe(0)
            session.predict()
            with self.assertRaises(ValueError):
                session.observe(2)
        result = subprocess.run([str(self.driver), "dmc"], input=b"O\x00Q", capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"predict must precede observe", result.stderr)

    def test_empty_input(self):
        self.assertEqual(predict_probabilities(b"", "ppmd"), [])

    def test_binary_alphabet_and_random_inputs(self):
        rng = random.Random(1729)
        data = bytes(range(256)) + bytes(rng.randrange(256) for _ in range(512))
        for provider in ("ppmd", "dmc", "paq"):
            with self.subTest(provider=provider):
                values = predict_probabilities(data, provider)
                self.assertEqual(len(values), len(data) * 8)
                self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in values))

    def test_sessions_do_not_share_upstream_global_state(self):
        for provider in ("dmc", "paq"):
            with self.subTest(provider=provider):
                with PredictionSession(provider) as left, PredictionSession(provider) as right:
                    for bit in (0, 1, 0, 0, 1, 1, 0, 1) * 3:
                        self.assertEqual(left.predict(), right.predict())
                        left.observe(bit)
                        right.observe(bit)

    def test_cmix_resource_gate_and_nncp_symbol_unit_are_explicit(self):
        with self.assertRaises(PredictorUnavailable):
            predict_probabilities(b"A", "cmix")
        with self.assertRaisesRegex(ValueError, "byte symbols"):
            predict_probabilities(b"A", "nncp")
        result = subprocess.run([str(self.driver), "cmix"], input=b"Q", capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"allow-large-memory", result.stderr)
        self.assertIn("not_runtime_verified", PROVIDERS["cmix"]["status"])
        self.assertTrue((ROOT / "external/prediction_probability/nncp-2024-06-05/nncp.c").is_file())


if __name__ == "__main__":
    unittest.main()
