#!/usr/bin/env python3
"""Cold-process lossless tests for PEC1 prediction-conditioned entropy paths."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest import mock

import prediction_entropy_codec as codec
from typed_graph_runtime import GraphState, Stream, restore_frame


class PredictionEntropyCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = codec._binary()

    def test_ppmd_dmc_paq_cold_roundtrip_and_determinism(self):
        rng = random.Random(1729)
        samples = [b"", b"\0\xffABABABA\0", bytes(range(256)),
                   bytes(rng.randrange(256) for _ in range(513))]
        for provider in ("ppmd", "dmc", "paq"):
            for raw in samples:
                with self.subTest(provider=provider, size=len(raw)):
                    first = codec.forward(raw, provider)
                    second = codec.forward(raw, provider)
                    self.assertEqual(first, second)
                    self.assertEqual(codec.frame_provider(first), provider)
                    # inverse starts a separate native process and reconstructs
                    # the causal predictor without encoder state.
                    self.assertEqual(codec.inverse(first), raw)

    def test_provider_parameters_are_framed_and_cold_decoded(self):
        raw = b"parameter-sensitive-data" * 3
        ppmd = codec.forward(raw, "ppmd", ppmd_order=4, ppmd_memory_mib=4)
        paq = codec.forward(raw, "paq", paq_level=1)
        self.assertEqual(codec.inverse(ppmd), raw)
        self.assertEqual(codec.inverse(paq), raw)
        self.assertNotEqual(ppmd, codec.forward(raw, "ppmd"))
        self.assertNotEqual(paq, codec.forward(raw, "paq"))

    def test_truncation_trailing_bytes_and_corruption_are_rejected(self):
        raw = b"corruption-check" * 5
        for provider in ("ppmd", "dmc", "paq"):
            encoded = codec.forward(raw, provider)
            corrupt_header = bytearray(encoded)
            corrupt_header[8] ^= 1
            corrupt_body = bytearray(encoded)
            corrupt_body[-1] ^= 0x40
            for invalid in (encoded[:-1], encoded + b"x", bytes(corrupt_header),
                            bytes(corrupt_body)):
                with self.subTest(provider=provider, invalid_length=len(invalid)):
                    with self.assertRaises(ValueError):
                        codec.inverse(invalid)

    def test_expected_provider_prevents_graph_relabelling(self):
        packet = codec.forward(b"abc", "ppmd")
        with self.assertRaisesRegex(ValueError, "does not match"):
            codec.inverse(packet, expected_action_or_provider="dmc")

    def test_cmix_remains_explicitly_large_memory_gated(self):
        result = subprocess.run([str(self.binary), "encode", "cmix"], input=b"x",
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"allow-large-memory", result.stderr)
        # A real full-cmix encode/decode probe is intentionally not run on the
        # shared development host: upstream recommends at least 32 GiB.

    def test_cmix_receipt_is_missing_stale_or_current_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "prediction_entropy_codec_cli"
            binary.write_bytes(b"current-binary")
            validation = root / "prediction_entropy_validation.json"
            cmix_validation = root / "cmix_prediction_entropy_validation.json"
            sources = {"scripts/current.py": "a" * 64}
            receipt = {
                "schema": codec.CMIX_RECEIPT_SCHEMA,
                "result": "passed",
                "provider": "cmix",
                "action": codec.PROVIDER_TO_ACTION["cmix"],
                "frame": "PEC1",
                "complete_reversible_path": True,
                "cold_decode_independent_process": True,
                "explicit_large_memory_opt_in": True,
                "single_process_sequential": True,
                "input_size": 1,
                "input_sha256": hashlib.sha256(b"A").hexdigest(),
                "frame_size": 66,
                "frame_sha256": "b" * 64,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source_fingerprints": sources,
                "encode": {"returncode": 0},
                "cold_decode": {"returncode": 0},
            }
            with (
                mock.patch.object(codec, "ROOT", root),
                mock.patch.object(codec, "BINARY", binary),
                mock.patch.object(codec, "VALIDATION", validation),
                mock.patch.object(codec, "CMIX_VALIDATION", cmix_validation),
                mock.patch.object(codec, "source_fingerprints", return_value=sources),
                mock.patch.dict(
                    os.environ,
                    {"COMPRESSION_BENCHMARK_PREDICTION_ENTROPY_CODEC": str(binary)},
                ),
            ):
                self.assertFalse(codec.cmix_receipt_status()["current"])
                cmix_validation.write_text(json.dumps(receipt))
                receipt_sha = hashlib.sha256(cmix_validation.read_bytes()).hexdigest()
                base = {
                    "result": "passed",
                    "source_fingerprints": sources,
                    "binary_sha256": receipt["binary_sha256"],
                    "validated_providers": ["ppmd", "dmc", "paq", "cmix"],
                    "cmix_probe_receipt": cmix_validation.name,
                    "cmix_probe_sha256": "0" * 64,
                }
                validation.write_text(json.dumps(base))
                self.assertFalse(codec.cmix_receipt_status()["current"])
                base["cmix_probe_sha256"] = receipt_sha
                validation.write_text(json.dumps(base))
                current = codec.cmix_receipt_status()
                self.assertTrue(current["current"])
                self.assertEqual(current["action"], codec.PROVIDER_TO_ACTION["cmix"])
                self.assertTrue(current["cold_decode_independent_process"])

                receipt["action"] = codec.PROVIDER_TO_ACTION["ppmd"]
                cmix_validation.write_text(json.dumps(receipt))
                base["cmix_probe_sha256"] = hashlib.sha256(
                    cmix_validation.read_bytes()
                ).hexdigest()
                validation.write_text(json.dumps(base))
                self.assertFalse(codec.cmix_receipt_status()["current"])

    def test_typed_graph_roundtrip_uses_complete_frame(self):
        raw = b"\0\0\1\0" * 17
        for action in tuple(codec.ACTION_TO_PROVIDER)[:3]:
            with self.subTest(action=action):
                state = GraphState.from_streams([Stream("numeric", 4, raw)])
                encoded_state = state.apply(action, ["s0"])
                leaf = next(iter(encoded_state.streams.values()))
                self.assertEqual(leaf.kind, "serial")
                self.assertEqual(codec.frame_provider(leaf.data), codec.ACTION_TO_PROVIDER[action])
                restored = restore_frame(encoded_state.stop())
                self.assertEqual(restored["s0"], Stream("numeric", 4, raw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
