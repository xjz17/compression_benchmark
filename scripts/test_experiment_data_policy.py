#!/usr/bin/env python3
"""Regression checks for NaN removal, retained wire bits, and shared loaders."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

from myapp.experiment_data_policy import (  # noqa: E402
    drop_nan_numeric, parse_experiment_float_cell, prepare_numeric_column,
)


class ExperimentDataPolicyTest(unittest.TestCase):
    def test_float_wire_drops_nan_only(self):
        source = [1.25, float("nan"), -0.0, float("inf"), -float("inf"), 2.5]
        result = prepare_numeric_column(source, "float")
        expected = struct.pack("<5d", 1.25, -0.0, float("inf"), -float("inf"), 2.5)
        self.assertEqual(result.raw, expected)
        self.assertEqual((result.source_count, result.count, result.removed_nan_count), (6, 5, 1))
        self.assertEqual(result.raw_sha256, hashlib.sha256(expected).hexdigest())
        self.assertEqual(result.manifest()["raw_bytes"], result.count * 8)
        self.assertTrue(math.isnan(source[1]))  # Input container was not edited.

    def test_int64_never_passes_through_float(self):
        source = [str((1 << 63) - 1), "nan", str(-(1 << 63)), "", str((1 << 53) + 1)]
        result = prepare_numeric_column(source, "integer")
        self.assertEqual(result.values, ((1 << 63) - 1, -(1 << 63), (1 << 53) + 1))
        self.assertEqual(result.raw, struct.pack("<3q", *result.values))
        self.assertEqual(result.removed_nan_count, 2)
        with self.assertRaises(ValueError):
            prepare_numeric_column([1.25], "int64")
        with self.assertRaises(struct.error):
            prepare_numeric_column([1 << 63], "integer")

    def test_all_nan_and_missing_cells(self):
        result = prepare_numeric_column([float("nan"), "NaN", None, ""], "float64")
        self.assertEqual(result.values, ())
        self.assertEqual(result.raw, b"")
        self.assertEqual(result.removed_nan_count, 4)
        retained, count = drop_nan_numeric([1 << 200, float("nan"), -0.0])
        self.assertEqual(count, 1)
        self.assertEqual(retained[0], 1 << 200)
        self.assertEqual(struct.pack("<d", retained[1]), struct.pack("<d", -0.0))

    def test_streaming_parser_keeps_infinities(self):
        for cell in ("NaN", "nan", "", None, "not-a-number"):
            self.assertIsNone(parse_experiment_float_cell(cell))
        self.assertEqual(parse_experiment_float_cell("inf"), float("inf"))
        self.assertEqual(parse_experiment_float_cell("-inf"), -float("inf"))
        self.assertEqual(struct.pack("<d", parse_experiment_float_cell("-0.0")), struct.pack("<d", -0.0))

    def test_old_operator_runner_uses_the_same_filter(self):
        import pandas as pd
        from generate_air_operator_pair_heatmap import column_values
        metadata = {"logical_type": "integer", "max_point_number": "0"}
        values = column_values(pd.Series([str((1 << 53) + 1), "nan", "2"]), metadata)
        self.assertEqual(values.tolist(), [(1 << 53) + 1, 2])
        self.assertEqual(metadata["removed_nan_count"], "1")
        float_meta = {"logical_type": "float", "max_point_number": "2"}
        self.assertEqual(column_values(pd.Series(["1.25", "nan", "2.5"]), float_meta).tolist(), [125, 250])
        with self.assertRaisesRegex(ValueError, "cannot represent infinity"):
            column_values(pd.Series(["inf"]), float_meta)

    def test_patch_and_grouping_use_identical_retained_values(self):
        from myapp.processor_patch_bench import _build_patch_column_arrays
        from myapp.grouping_native import streams_from_patch_cells
        cells = [["NaN"], ["-0.0"], ["inf"], ["1.25"]]
        columns = [("value", 2, 0)]
        byte_columns, _, floats, _, _, _ = _build_patch_column_arrays("float", columns, cells)
        expected = struct.pack("<3d", -0.0, float("inf"), 1.25)
        self.assertEqual(byte_columns, [expected])
        self.assertEqual(struct.pack("<3d", *floats[0]), expected)
        streams = streams_from_patch_cells("float", columns, cells, row_offset=100)
        self.assertEqual(streams[0].timestamps, [101, 102, 103])
        self.assertEqual(struct.pack("<3Q", *streams[0].values), expected)

    def test_matrix_mask_keeps_columns_aligned(self):
        from prepare_system_codec_inputs import load_numeric_columns, write_scdbin
        audit = {}
        names, columns, rows = load_numeric_columns(ROOT / "scripts/fixtures/experiment_nan_matrix.csv", audit=audit)
        self.assertEqual(names, ["left", "right"])
        self.assertEqual((audit["source_rows"], rows, audit["rows_removed"]), (5, 3, 2))
        self.assertEqual(audit["retained_columns"], names)
        self.assertEqual(columns[1], [10.0, 40.0, -float("inf")])
        self.assertEqual(struct.pack(">3d", *columns[0]), struct.pack(">3d", 1, -0.0, float("inf")))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.scdbin"
            write_scdbin(path, names, columns, rows)
            payload = path.read_bytes()
            self.assertEqual(struct.unpack(">Iq", payload[8:20]), (2, 3))
            self.assertEqual(payload[-48:], struct.pack(">6d", 1, -0.0, float("inf"), 10, 40, -float("inf")))
            with self.assertRaisesRegex(ValueError, "contains NaN"):
                write_scdbin(path, ["bad"], [[float("nan")]], 1)

    def test_legacy_processor_keeps_int64_after_nan_removal(self):
        from myapp import processor
        seen = []
        expected = struct.pack("<2q", (1 << 53) + 1, 2)

        def fake_single(raw, *_args):
            seen.append(raw)
            return len(raw), len(raw), 1, 1

        def fake_columns(columns, *_args):
            seen.extend(columns)
            size = sum(map(len, columns))
            return size, size, 1, 1

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(processor, "test_common_compression", fake_single), \
                patch.object(processor, "test_lzma_compression", fake_single), \
                patch.object(processor, "test_zstd_compression", fake_single), \
                patch.object(processor, "bench_ans_bytes", fake_columns), \
                patch.object(processor, "bench_huffman_bytes", fake_columns), \
                patch.object(processor, "bench_arithmetic_bytes", fake_columns), \
                patch("myapp.native_lz_family.bench_native_lz_family_bytes", fake_columns):
            processor.process_csv(str(ROOT / "scripts/fixtures/experiment_nan_int64.csv"),
                                  "int", 1, 1, str(Path(directory) / "result.csv"))
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(raw == expected for raw in seen))


if __name__ == "__main__":
    unittest.main()
