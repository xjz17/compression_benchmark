#!/usr/bin/env python3
"""Local synthetic tests only: no formal dataset search or legacy file edits."""
import ast
import base64
import copy
import hashlib
import itertools
import json
from pathlib import Path
import unittest

import numpy as np

import atomic_transform_runtime_v1 as v1
import atomic_transform_extensions_v2 as v2


def raw(values):
    return np.asarray(values, dtype="<u8").tobytes()


def b64(values):
    return base64.b64encode(raw(values)).decode("ascii")


class AtomicExtensionTests(unittest.TestCase):
    def roundtrip(self, source, specs):
        payload, metadata = v2.apply_ops(source, specs)
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.assertEqual(v2.reverse_ops(payload, metadata), source)
        return payload, metadata

    def test_v1_file_and_registry_unchanged(self):
        self.assertEqual(hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest(),
                         "904e8934026ff0c8d08a8a2f9ddcdfe9e6ae9bff8551a61842db67c1e75a4c10")
        self.assertEqual(len(v1.SUPPORTED_OPERATORS), 13)
        self.assertEqual(len(v2.SUPPORTED_OPERATORS), 16)
        self.assertIs(v2.State, v1.State)
        self.assertNotIn("block_for", v1.INVERSE_FIELDS)

    def test_v1_operators_identical(self):
        source = raw([0, 1, 5, (1 << 64) - 1, 0, 0])
        for op in v1.SUPPORTED_OPERATORS:
            if op == "float_scale":
                continue
            specs = [{"op": op}]
            self.assertEqual(v2.apply_ops(source, specs), v1.apply_ops(source, specs))
            self.roundtrip(source, specs)

    def test_empty_new_operations(self):
        for op in v2.EXTENSION_OPERATORS:
            self.roundtrip(b"", [{"op": op}])

    def test_float_all_factors_and_exact_tolerance(self):
        for factor in v2.SCALE_FACTORS:
            source = (np.arange(-100, 100, dtype="<f8") / factor).tobytes()
            payload, metadata = self.roundtrip(source, [{"op": "float_scale_exceptions", "factor": factor}])
            self.assertEqual(len(payload), len(source))
            self.assertEqual(metadata[0]["exceptions_positions_b64"], "")
            self.roundtrip(source, [{"op": "float_scale_exceptions", "factor": factor, "max_fraction": 0},
                                    {"op": "delta"}, {"op": "zigzag"},
                                    {"op": "block_for", "block_size": 32}, {"op": "block_bitpack", "block_size": 32}])

    def test_float_sparse_nan_minuszero_infinity_and_bit_patterns(self):
        bits = np.zeros(200, dtype="<u8")
        specials = [0x8000000000000000, 0x7ff8000000000123, 0xfff8000000001234,
                    0x7ff0000000000001, 0x7ff0000000000000, 0xfff0000000000000,
                    0x0000000000000001, 0x8000000000000001]
        positions = np.array([1, 19, 33, 47, 61, 80, 97, 190])
        bits[positions] = specials
        payload, metadata = self.roundtrip(bits.tobytes(), [{"op": "float_scale_exceptions", "factor": 1000}])
        record = metadata[0]
        stored_positions = np.frombuffer(base64.b64decode(record["exceptions_positions_b64"]), dtype="<u8")
        stored_values = np.frombuffer(base64.b64decode(record["exceptions_values_b64"]), dtype="<u8")
        self.assertEqual(stored_positions.tolist(), positions.tolist())
        self.assertEqual(stored_values.tolist(), specials)
        self.assertTrue(np.all(np.frombuffer(payload, dtype="<u8")[positions] == 0))
        self.assertEqual(len(base64.b64decode(record["exceptions_positions_b64"])) +
                         len(base64.b64decode(record["exceptions_values_b64"])), 8 * 16)

    def test_float_exception_fraction_gate_and_boundary(self):
        values = np.zeros(100, dtype="<f8")
        values[:5] = [0.125, -0.0, float("nan"), float("inf"), 1e300]
        self.roundtrip(values.tobytes(), [{"op": "float_scale_exceptions", "factor": 10}])
        with self.assertRaises(ValueError):
            v2.apply_ops(values.tobytes(), [{"op": "float_scale_exceptions", "factor": 10, "max_fraction": 0.049}])
        values[99] = -float("inf")
        with self.assertRaises(ValueError):
            v2.apply_ops(values.tobytes(), [{"op": "float_scale_exceptions", "factor": 10}])
        with self.assertRaises(ValueError):
            v2.apply_ops(np.asarray([float("nan")] * 100, dtype="<f8").tobytes(), [{"op": "float_scale_exceptions"}])
        boundary = np.zeros(200, dtype="<f8")
        boundary[:5] = [-(2.0 ** 63), np.nextafter(2.0 ** 63, 0), 2.0 ** 63, -1e300, 1e300]
        _, metadata = self.roundtrip(boundary.tobytes(), [{"op": "float_scale_exceptions", "factor": 1}])
        self.assertEqual(np.frombuffer(base64.b64decode(metadata[0]["exceptions_positions_b64"]), dtype="<u8").tolist(), [2, 3, 4])

    def test_block_for_full_domain_and_short_blocks(self):
        rng = np.random.default_rng(2048)
        for block_size, count in itertools.product(v2.BLOCK_SIZES, (0, 1, 15, 16, 17, 1023, 1024, 1025)):
            values = rng.integers(0, 1 << 64, count, dtype=np.uint64)
            if count:
                values[0] = (1 << 64) - 1
            payload, metadata = self.roundtrip(values.astype("<u8").tobytes(), [{"op": "block_for", "block_size": block_size}])
            self.assertEqual(len(payload), count * 8)
            self.assertEqual(len(base64.b64decode(metadata[0]["minima_b64"])), ((count + block_size - 1) // block_size) * 8)
        payload, meta = self.roundtrip(raw([100] * 16 + [10000, 10003, 10002]), [{"op": "block_for", "block_size": 16}])
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [0] * 16 + [0, 3, 2])
        self.assertEqual(np.frombuffer(base64.b64decode(meta[0]["minima_b64"]), dtype="<u8").tolist(), [100, 10000])

    def test_block_bitpack_every_width_and_tail(self):
        for bits, block_size in itertools.product(range(65), v2.BLOCK_SIZES):
            values = [(1 << bits) - 1] * (block_size + 3)
            payload, metadata = self.roundtrip(raw(values), [{"op": "block_bitpack", "block_size": block_size}])
            self.assertEqual(base64.b64decode(metadata[0]["widths_b64"]), bytes([bits, bits]))
            self.assertEqual(len(payload), (block_size * bits + 7) // 8 + (3 * bits + 7) // 8)
        mixed = raw([0] * 16 + [1] * 16 + [255] * 16 + [(1 << 64) - 1] * 3)
        payload, metadata = self.roundtrip(mixed, [{"op": "block_bitpack", "block_size": 16}])
        self.assertEqual(base64.b64decode(metadata[0]["widths_b64"]), bytes([0, 1, 8, 64]))
        self.assertEqual(len(payload), 0 + 2 + 16 + 24)

    def test_mixed_chains_random_full_domain(self):
        rng = np.random.default_rng(2006)
        for count, block_size, prefix in itertools.product((1, 19, 133, 2001), (16, 128, 1024),
                                                           ([], [{"op": "xor", "lag": 4}], [{"op": "delta"}, {"op": "zigzag"}])):
            source = rng.integers(0, 1 << 64, count, dtype=np.uint64).astype("<u8").tobytes()
            self.roundtrip(source, prefix + [{"op": "block_for", "block_size": block_size},
                                             {"op": "delta", "lag": 24}, {"op": "zigzag"},
                                             {"op": "block_bitpack", "block_size": block_size},
                                             {"op": "shuffle", "width": 8}, {"op": "bittranspose"}])

    def test_malformed_exception_positions_values_and_placeholders(self):
        values = np.zeros(100, dtype="<f8")
        values[[1, 7]] = [-0.0, float("nan")]
        payload, metadata = v2.apply_ops(values.tobytes(), [{"op": "float_scale_exceptions"}])
        for replacement in (b64([1, 1]), b64([7, 1]), b64([1, 100]), "%%%", b64([1])):
            bad = copy.deepcopy(metadata)
            bad[0]["exceptions_positions_b64"] = replacement
            with self.assertRaises(ValueError): v2.reverse_ops(payload, bad)
        bad = copy.deepcopy(metadata)
        bad[0]["exceptions_values_b64"] = b64([0])
        with self.assertRaises(ValueError): v2.reverse_ops(payload, bad)
        bad = copy.deepcopy(metadata)
        bad[0]["max_fraction"] = 0.01
        with self.assertRaises(ValueError): v2.reverse_ops(payload, bad)
        corrupt = np.frombuffer(payload, dtype="<u8").copy()
        corrupt[1] = 1
        with self.assertRaises(ValueError): v2.reverse_ops(corrupt.tobytes(), metadata)

    def test_malformed_block_minima_widths_and_padding(self):
        source = raw([1] * 19)
        payload, metadata = v2.apply_ops(source, [{"op": "block_bitpack", "block_size": 16}])
        for widths in (bytes([65, 1]), bytes([1]), bytes([1, 1, 1]), bytes([0, 0])):
            bad = copy.deepcopy(metadata)
            bad[0]["widths_b64"] = base64.b64encode(widths).decode("ascii")
            with self.assertRaises(ValueError): v2.reverse_ops(payload, bad)
        corrupt = payload[:-1] + bytes([payload[-1] | 0x80])
        with self.assertRaises(ValueError): v2.reverse_ops(corrupt, metadata)
        payload, metadata = v2.apply_ops(source, [{"op": "block_for", "block_size": 16}])
        bad = copy.deepcopy(metadata)
        bad[0]["minima_b64"] = b64([1])
        with self.assertRaises(ValueError): v2.reverse_ops(payload, bad)
        bad = copy.deepcopy(metadata)
        bad[0]["minima_b64"] = b64([(1 << 64) - 1, 1])
        corrupt = np.frombuffer(payload, dtype="<u8").copy()
        corrupt[0] = 1
        with self.assertRaises(ValueError): v2.reverse_ops(corrupt.tobytes(), bad)

    def test_invalid_grammar_and_options(self):
        invalid = ([{"op": "compact"}, {"op": "block_for"}],
                   [{"op": "block_bitpack"}, {"op": "zigzag"}],
                   [{"op": "block_for", "block_size": 24}],
                   [{"op": "block_bitpack", "block_size": 2048}],
                   [{"op": "float_scale_exceptions", "factor": 3}],
                   [{"op": "float_scale_exceptions", "max_fraction": 0.051}],
                   [{"op": "float_scale_exceptions", "max_fraction": -0.01}],
                   [{"op": "float_scale_exceptions", "max_fraction": float("nan")}],
                   [{"op": "float_scale_exceptions", "max_fraction": True}],
                   [{"op": "block_for", "compressor": "zlib"}])
        for specs in invalid:
            with self.subTest(specs=specs), self.assertRaises(ValueError): v2.apply_ops(raw([0]), specs)
        self.assertEqual(v2.validate_specs([{"op": "block_for"}, {"op": "delta"}]), "u64")
        self.assertEqual(v2.validate_specs([{"op": "block_bitpack"}, {"op": "shuffle"}]), "bytes")

    def test_no_implicit_codecs_or_original_fallback(self):
        tree = ast.parse(Path(v2.__file__).read_text())
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.alias)}
        self.assertFalse(names & {"bz2", "zlib", "lzma", "subprocess", "ctypes"})
        self.assertEqual(v2.apply_ops(raw([1]), []), (raw([1]), []))
        for op in v2.EXTENSION_OPERATORS:
            _, metadata = v2.apply_ops(raw([0] * 100), [{"op": op}])
            expected = {"op", "in_kind", "out_kind", "in_count", "out_count", "in_nbytes", "out_nbytes"} | set(v2.INVERSE_FIELDS[op])
            self.assertEqual(set(metadata[0]), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
