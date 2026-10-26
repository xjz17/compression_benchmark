#!/usr/bin/env python3
"""Deterministic reversibility/typing tests; no experimental column search."""
import copy
import itertools
import json
import unittest

import numpy as np

import atomic_transform_runtime_v1 as runtime


def raw(values):
    return np.asarray(values, dtype="<u8").tobytes()


class AtomicTransformTests(unittest.TestCase):
    def roundtrip(self, source, specs):
        payload, metadata = runtime.apply_ops(source, specs)
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.assertEqual(runtime.reverse_ops(payload, metadata), source)
        self.assertEqual(len(metadata), len(specs))
        for spec, record in zip(specs, metadata):
            self.assertEqual(spec["op"], record["op"])
            self.assertEqual(record["out_count"] * (8 if record["out_kind"] == "u64" else 1), record["out_nbytes"])
        return payload, metadata

    def test_zigzag_only_maps_nonnegative_same_width(self):
        source = np.asarray([0, -1, 1, -2, 2, -(1 << 63), (1 << 63) - 1], dtype="<i8").tobytes()
        payload, _ = self.roundtrip(source, [{"op": "zigzag"}])
        self.assertEqual(len(payload), len(source))
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [0, 1, 2, 3, 4, (1 << 64) - 1, (1 << 64) - 2])

    def test_every_operator_empty(self):
        for op in runtime.SUPPORTED_OPERATORS:
            with self.subTest(op=op):
                self.roundtrip(b"", [{"op": op}])

    def test_numeric_full_u64_domain_and_float_bits(self):
        rng = np.random.default_rng(20260906)
        values = rng.integers(0, 1 << 64, size=4096, dtype=np.uint64)
        values[:10] = [0, 1, 2, (1 << 63) - 1, 1 << 63, (1 << 64) - 1,
                       0x8000000000000000, 0x7ff0000000000000,
                       0x7ff8000000000123, 0xfff8000000001234]
        source = values.astype("<u8").tobytes()
        specs = ([{"op": "zigzag"}], [{"op": "for"}], [{"op": "gcd"}],
                 [{"op": "dictionary", "max_entries": 5000}], [{"op": "rle"}],
                 [{"op": "sparsezero"}], [{"op": "compact"}], [{"op": "bitpack"}])
        for spec in specs:
            with self.subTest(spec=spec):
                self.roundtrip(source, spec)
        for lag, order in itertools.product((1, 2, 4, 8, 24, 1024), (1, 2)):
            self.roundtrip(source, [{"op": "delta", "lag": lag, "order": order}])
            self.roundtrip(source, [{"op": "xor", "lag": lag}])

    def test_compact_and_bitpack_all_widths(self):
        rng = np.random.default_rng(17)
        for bits in range(65):
            maximum = (1 << bits) - 1
            values = rng.integers(0, maximum + 1, size=31, dtype=np.uint64) if bits else np.zeros(31, dtype=np.uint64)
            values[0] = maximum
            payload, _ = self.roundtrip(values.astype("<u8").tobytes(), [{"op": "bitpack", "bits": bits}])
            self.assertEqual(len(payload), (31 * bits + 7) // 8)
        for width in (1, 2, 4, 8):
            source = raw([0, (1 << (width * 8)) - 1])
            payload, _ = self.roundtrip(source, [{"op": "compact", "width": width}])
            self.assertEqual(len(payload), 2 * width)

    def test_transpose_bytes_tails_and_repetition(self):
        source = raw(list(range(37)))
        for op, width in itertools.product(("shuffle", "bittranspose"), (1, 2, 4, 8, 16)):
            self.roundtrip(source, [{"op": op, "width": width}])
            self.roundtrip(source, [{"op": "bitpack", "bits": 6}, {"op": op, "width": width}])
            self.roundtrip(source, [{"op": "compact"}, {"op": op, "width": width}, {"op": op, "width": 2}])

    def test_rle_sparse_are_explicit_numeric_two_streams(self):
        source = raw([0, 0, 5, 5, 5, 0, (1 << 64) - 1])
        payload, metadata = self.roundtrip(source, [{"op": "rle"}])
        self.assertEqual(metadata[0]["out_kind"], "u64")
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [0, 5, 0, (1 << 64) - 1, 2, 3, 1, 1])
        payload, metadata = self.roundtrip(source, [{"op": "sparsezero"}])
        self.assertEqual(metadata[0]["out_kind"], "u64")
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [5, 5, 5, (1 << 64) - 1, 2, 3, 4, 6])
        self.roundtrip(source, [{"op": "rle"}, {"op": "delta"}, {"op": "zigzag"}, {"op": "bitpack"}, {"op": "shuffle"}])

    def test_gcd_for_dictionary_metadata_not_hidden_payload(self):
        payload, metadata = self.roundtrip(raw([600, 1200, 1800]), [{"op": "gcd"}])
        self.assertEqual(metadata[0]["factor"], 600)
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [1, 2, 3])
        payload, metadata = self.roundtrip(raw([600, 1200, 1800]), [{"op": "for"}])
        self.assertEqual(metadata[0]["minimum"], 600)
        self.assertEqual(len(payload), 24)
        payload, metadata = self.roundtrip(raw([7, 8, 7, 8]), [{"op": "dictionary"}])
        self.assertEqual(len(payload), 32)
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [0, 1, 0, 1])
        self.assertEqual(set(metadata[0]), {"op", "in_kind", "out_kind", "in_count", "out_count", "in_nbytes", "out_nbytes", "palette_b64"})

    def test_exact_float_scaling_and_rejection(self):
        for factor in (1, 10, 100, 1000, 10000, 100000, 1000000):
            source = (np.asarray([0, 1, -3, 1928, 5807], dtype="<f8") / factor).tobytes()
            self.roundtrip(source, [{"op": "float_scale", "factor": factor}, {"op": "delta"}, {"op": "zigzag"}, {"op": "compact"}])
        for value in (-0.0, float("nan"), float("inf"), -float("inf"), 0.125, 1e40):
            source = np.asarray([value], dtype="<f8").tobytes()
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.apply_ops(source, [{"op": "float_scale", "factor": 10}])

    def test_many_compositions(self):
        rng = np.random.default_rng(1107)
        seeds = [raw([0] * 131), raw([7] * 131), raw([1, 2, 2, 0, 4, 0, 0] * 19),
                 (rng.integers(-500, 500, 131, dtype=np.int64)).astype("<i8").tobytes()]
        numerics = [{"op": "zigzag"}, {"op": "delta", "lag": 2}, {"op": "delta", "order": 2},
                    {"op": "xor", "lag": 4}, {"op": "for"}, {"op": "gcd"},
                    {"op": "dictionary"}, {"op": "rle"}, {"op": "sparsezero"}]
        combinations = 0
        for source, first, second in itertools.product(seeds, numerics, numerics):
            self.roundtrip(source, [first, second, {"op": "bitpack"}, {"op": "bittranspose", "width": 8}])
            combinations += 1
        self.assertEqual(combinations, 324)

    def test_invalid_types_parameters_and_truncation(self):
        for specs in ([{"op": "compact"}, {"op": "zigzag"}], [{"op": "bitpack"}, {"op": "delta"}],
                      [{"op": "compact", "width": 3}], [{"op": "zigzag", "backend": "zlib"}],
                      [{"op": "delta", "lag": 0}], [{"op": "float_scale", "factor": 3}],
                      [{"op": "dictionary", "max_entries": 1}], [{"op": "compact", "width": 1}],
                      [{"op": "bitpack", "bits": 3}]):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                runtime.apply_ops(raw([0, 1000]), specs)
        with self.assertRaises(ValueError): runtime.apply_ops(b"abc", [])
        with self.assertRaises(ValueError): runtime.apply_ops(raw([1]), [{"op": "unknown"}])

    def test_malformed_reverse_records_rejected(self):
        with self.assertRaises(ValueError): runtime.reverse_ops(b"", [None])
        payload, records = runtime.apply_ops(raw([1, 2, 3]), [{"op": "bitpack", "bits": 2}])
        with self.assertRaises(ValueError): runtime.reverse_ops(payload + b"extra", records)
        with self.assertRaises(ValueError): runtime.reverse_ops(bytes([payload[0] | 0x80]), records)
        bad = copy.deepcopy(records)
        bad[0]["in_count"] = -1
        with self.assertRaises(ValueError): runtime.reverse_ops(payload, bad)
        payload, records = runtime.apply_ops(raw([1, 1, 2]), [{"op": "rle"}])
        corrupt = np.frombuffer(payload, dtype="<u8").copy()
        corrupt[-1] = 0
        with self.assertRaises(ValueError): runtime.reverse_ops(corrupt.tobytes(), records)
        payload, records = runtime.apply_ops(raw([0, 2, 0, 3]), [{"op": "sparsezero"}])
        corrupt = np.frombuffer(payload, dtype="<u8").copy()
        corrupt[-1] = corrupt[-2]
        with self.assertRaises(ValueError): runtime.reverse_ops(corrupt.tobytes(), records)

    def test_identity_and_no_compressor_imports(self):
        import ast
        from pathlib import Path
        source = raw([0, (1 << 64) - 1])
        self.assertEqual(runtime.apply_ops(source, []), (source, []))
        self.assertEqual(runtime.reverse_ops(source, []), source)
        tree = ast.parse(Path(runtime.__file__).read_text())
        imported = {node.name for node in ast.walk(tree) if isinstance(node, ast.alias)}
        self.assertFalse(imported & {"zlib", "bz2", "lzma", "subprocess", "ctypes"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
