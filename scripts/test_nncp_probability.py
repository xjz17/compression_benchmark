#!/usr/bin/env python3
"""Bounded tests of genuine NNCP models, never synthetic replacement networks."""
import math
import struct
import subprocess
import unittest

from build_nncp_probability import nncp_command
from prediction_probability import NNCPSymbolSession, predict_symbol_probabilities, predict_probabilities


class NNCPProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.command = nncp_command()

    def test_distribution_is_real_normalized_and_causal(self):
        first = predict_symbol_probabilities(b"ABA")
        changed = predict_symbol_probabilities(b"ABZ")
        repeat = predict_symbol_probabilities(b"ABA")
        self.assertEqual(first, changed)
        self.assertEqual(first, repeat)
        self.assertEqual(len(first), 3)
        self.assertNotEqual(first[0], [1 / 256] * 256)
        self.assertNotEqual(first[0], first[1])
        for row in first:
            self.assertEqual(len(row), 256)
            self.assertAlmostEqual(sum(row), 1, places=5)
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in row))

    def test_interactive_prediction_is_cached_and_matches_batch(self):
        rows = []
        with NNCPSymbolSession() as session:
            for symbol in b"ABA":
                value = session.predict()
                self.assertEqual(value, session.predict())
                rows.append(value)
                session.observe(symbol)
        self.assertEqual(rows, predict_symbol_probabilities(b"ABA"))

    def test_fresh_sessions_are_isolated(self):
        with NNCPSymbolSession() as left, NNCPSymbolSession() as right:
            self.assertEqual(left.predict(), right.predict())
            left.observe(0)
            right.observe(255)
            lvalue, rvalue = left.predict(), right.predict()
            self.assertNotEqual(lvalue, rvalue)
        self.assertEqual(lvalue, predict_symbol_probabilities(b"\0A")[1])
        self.assertEqual(rvalue, predict_symbol_probabilities(b"\xffA")[1])

    def test_protocol_and_symbol_unit(self):
        with NNCPSymbolSession() as session:
            with self.assertRaises(ValueError):
                session.observe(0)
            session.predict()
            with self.assertRaises(ValueError):
                session.observe(256)
        with self.assertRaisesRegex(ValueError, "byte symbols"):
            predict_probabilities(b"A", "nncp")
        result = subprocess.run(self.command, input=b"O\0", capture_output=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"predict must precede observe", result.stderr)

    def test_entropy_roundtrip_uses_independent_cold_decoder(self):
        for raw in (b"", b"\0\xffABA\0"):
            with self.subTest(raw=raw):
                encoded = subprocess.run([*self.command, "encode"], input=raw, capture_output=True, check=True, timeout=120).stdout
                self.assertEqual(encoded[:4], b"NNP1")
                self.assertGreaterEqual(len(encoded), 24)
                decoded = subprocess.run([*self.command, "decode"], input=encoded, capture_output=True, check=True, timeout=120).stdout
                self.assertEqual(decoded, raw)
                for invalid in (encoded[:-1], encoded + b"x", encoded[:16] + bytes([encoded[16] ^ 1]) + encoded[17:]):
                    result = subprocess.run([*self.command, "decode"], input=invalid, capture_output=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)
                for body in (encoded[24:] + b"x", encoded[24:-1], b"arbitrary"):
                    invalid = encoded[:20] + struct.pack("<I", len(body)) + body
                    result = subprocess.run([*self.command, "decode"], input=invalid, capture_output=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0)

    def test_bounds_and_empty_trace(self):
        self.assertEqual(predict_symbol_probabilities(b""), [])
        with self.assertRaises(ValueError):
            predict_symbol_probabilities(bytes(4097))
        result = subprocess.run(self.command, input=b"T" + struct.pack("<Q", 4097), capture_output=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"limit", result.stderr)


if __name__ == "__main__":
    unittest.main()
