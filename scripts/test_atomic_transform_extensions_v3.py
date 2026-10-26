#!/usr/bin/env python3
"""Synthetic tests for general common-symbol splitting; no dataset search."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import atomic_transform_runtime_v1 as v1
import atomic_transform_extensions_v2 as v2
import atomic_transform_extensions_v3 as v3


def raw(values):
    return np.asarray(values, dtype="<u8").tobytes()


class SplitCommonTests(unittest.TestCase):
    def roundtrip(self, source, specs=None):
        if specs is None:
            specs = [{"op": "split_common"}]
        payload, metadata = v3.apply_ops(source, specs)
        self.assertEqual(v3.reverse_ops(payload, json.loads(json.dumps(metadata))), source)
        return payload, metadata

    def test_frozen_v1_v2_unchanged(self):
        expected = [(v1, "904e8934026ff0c8d08a8a2f9ddcdfe9e6ae9bff8551a61842db67c1e75a4c10"),
                    (v2, "9346d4df1d75e301384ec96731be3a74066dacce12c5804dfb709f7e9ab38ee7")]
        for module, digest in expected:
            self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), digest)
        self.assertIs(v3.State, v1.State)
        self.assertEqual(len(v1.SUPPORTED_OPERATORS), 13)
        self.assertEqual(len(v2.SUPPORTED_OPERATORS), 16)
        self.assertEqual(len(v3.SUPPORTED_OPERATORS), 17)

    def test_automatic_common_and_bit_order(self):
        source = raw([7, 3, 7, 7, 2, 3, 7, 4, 7])
        payload, metadata = self.roundtrip(source)
        self.assertEqual(metadata[0]["common_value"], 7)
        self.assertEqual(base64.b64decode(metadata[0]["mask_b64"]), bytes([0b01001101, 1]))
        self.assertEqual(np.frombuffer(payload, dtype="<u8").tolist(), [3, 2, 3, 4])
        self.assertEqual(metadata[0]["out_kind"], "u64")
        self.assertEqual(set(metadata[0]), {"op", "in_kind", "in_count", "in_nbytes", "out_kind", "out_count", "out_nbytes", "common_value", "mask_b64"})

    def test_ties_use_smallest_unsigned_bits(self):
        _, metadata = self.roundtrip(raw([(1 << 64) - 1, 7, 5, 7, (1 << 64) - 1, 5]))
        self.assertEqual(metadata[0]["common_value"], 5)
        _, metadata = self.roundtrip(raw([(1 << 64) - 1, 1 << 63, 3]))
        self.assertEqual(metadata[0]["common_value"], 3)

    def test_empty_and_all_common(self):
        payload, metadata = self.roundtrip(b"")
        self.assertEqual(payload, b"")
        self.assertEqual(metadata[0]["common_value"], 0)
        self.assertEqual(metadata[0]["mask_b64"], "")
        for count in (1, 7, 8, 9, 33, 1024):
            for value in (0, 1, 1 << 63, (1 << 64) - 1):
                payload, metadata = self.roundtrip(raw([value] * count))
                self.assertEqual(payload, b"")
                self.assertEqual(len(base64.b64decode(metadata[0]["mask_b64"])), (count + 7) // 8)

    def test_nan_payload_signedzero_and_infinities(self):
        nan = 0x7ff8000000000123
        source = raw([nan, 0x8000000000000000, nan, 0, 0x7ff0000000000000,
                      0xfff0000000000000, nan, 0x7ff8000000000124,
                      0x7ff0000000000001, 0x8000000000000001, nan])
        payload, metadata = self.roundtrip(source)
        self.assertEqual(metadata[0]["common_value"], nan)
        self.assertEqual(len(payload), 7 * 8)

    def test_random_full_domain(self):
        rng = np.random.default_rng(3096)
        for count in (0, 1, 15, 16, 17, 127, 128, 129, 1025, 9357):
            values = rng.integers(0, 1 << 64, count, dtype=np.uint64)
            if count:
                values[::4] = (1 << 64) - 1
            self.roundtrip(values.astype("<u8").tobytes())

    def test_real_composition_not_hidden_pipeline(self):
        source = np.asarray([-200, 93, 94, -200, 99, 100, -200, -200] * 37, dtype="<i8").tobytes()
        payload, metadata = self.roundtrip(source, [{"op": "split_common"}, {"op": "delta"},
                                                   {"op": "zigzag"}, {"op": "block_bitpack", "block_size": 32},
                                                   {"op": "bittranspose", "width": 4}])
        self.assertEqual(len(metadata), 5)
        self.assertEqual(metadata[0]["common_value"], (1 << 64) - 200)
        self.assertEqual(metadata[0]["out_nbytes"], len(source) // 2)
        self.roundtrip(source, [{"op": "split_common"}, {"op": "split_common"},
                               {"op": "delta"}, {"op": "zigzag"}, {"op": "compact"}])

    def test_dirty_short_long_and_false_masks(self):
        payload, metadata = v3.apply_ops(raw([9, 1, 9, 2, 9, 3, 9, 4, 9]), [{"op": "split_common"}])
        for mask in (b"", b"\x55", b"\x55\x01\x00", b"\x55\x81", b"\x00\x00", b"\xff\x01"):
            bad = copy.deepcopy(metadata)
            bad[0]["mask_b64"] = base64.b64encode(mask).decode("ascii")
            with self.subTest(mask=mask), self.assertRaises(ValueError): v3.reverse_ops(payload, bad)
        bad = copy.deepcopy(metadata)
        bad[0]["mask_b64"] = "@@@"
        with self.assertRaises(ValueError): v3.reverse_ops(payload, bad)
        bad = copy.deepcopy(metadata)
        bad[0]["common_value"] = 1 << 64
        with self.assertRaises(ValueError): v3.reverse_ops(payload, bad)
        corrupt = np.frombuffer(payload, dtype="<u8").copy()
        corrupt[0] = 9
        with self.assertRaises(ValueError): v3.reverse_ops(corrupt.tobytes(), metadata)

    def test_grammar_and_old_operator_compatibility(self):
        for specs in ([{"op": "split_common", "value": -200}], [{"op": "compact"}, {"op": "split_common"}],
                      [{"op": "block_bitpack"}, {"op": "split_common"}]):
            with self.assertRaises(ValueError): v3.apply_ops(raw([0]), specs)
        source = raw([1, 3, 5, 7, 7, 9])
        specs = [{"op": "block_for", "block_size": 16}, {"op": "delta"}, {"op": "zigzag"}, {"op": "bitpack"}]
        self.assertEqual(v3.apply_ops(source, specs), v2.apply_ops(source, specs))
        self.assertEqual(v3.validate_specs([{"op": "split_common"}]), "u64")

    def test_no_compressor_imports(self):
        tree = ast.parse(Path(v3.__file__).read_text())
        imported = {node.name for node in ast.walk(tree) if isinstance(node, ast.alias)}
        self.assertFalse(imported & {"zlib", "bz2", "lzma", "subprocess", "ctypes"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
