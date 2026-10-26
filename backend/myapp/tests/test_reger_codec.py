from __future__ import annotations

import math
import os
import struct
import unittest
from unittest.mock import patch

from myapp.reger_codec import (
    bench_reger_double_columns,
    bench_reger_int_columns,
    _default_reorder_iterations,
    _row_order_candidates,
    decode_int_columns,
    decode_float_column,
    decode_int_column,
    encode_int_columns,
    encode_float_column,
    encode_int_column,
)


def _double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


class RegerCodecTests(unittest.TestCase):
    def test_int_roundtrip_and_benchmark_metrics_are_lossless(self) -> None:
        values = [
            1000 + i * 3 + (i % 5)
            for i in range(96)
        ] + [-(2**40), 2**42, -17, 0, 17]

        encoded = encode_int_column(values)
        self.assertLess(len(encoded), len(values) * 16)
        self.assertEqual(decode_int_column(encoded), values)

        original, compressed, comp_ns, decomp_ns, loss_err, loss_ref = bench_reger_int_columns(
            [values]
        )
        self.assertEqual(original, len(values) * 8)
        self.assertGreater(compressed, 0)
        self.assertGreaterEqual(comp_ns, 0)
        self.assertGreaterEqual(decomp_ns, 0)
        self.assertEqual(loss_err, 0.0)
        self.assertGreater(loss_ref, 0.0)

    def test_float_roundtrip_preserves_ieee754_bits(self) -> None:
        values = [
            -0.0,
            1.25,
            -13.5,
            math.pi,
            2.0**40,
            -2.0**-20,
            float("inf"),
            float("-inf"),
        ] * 12

        encoded = encode_float_column(values)
        decoded = decode_float_column(encoded)
        self.assertEqual([_double_bits(v) for v in decoded], [_double_bits(v) for v in values])

        original, compressed, comp_ns, decomp_ns, loss_err, loss_ref = bench_reger_double_columns(
            [values]
        )
        self.assertEqual(original, len(values) * 8)
        self.assertGreater(compressed, 0)
        self.assertGreaterEqual(comp_ns, 0)
        self.assertGreaterEqual(decomp_ns, 0)
        self.assertEqual(loss_err, 0.0)
        self.assertGreaterEqual(loss_ref, 0.0)

    def test_multi_column_roundtrip_uses_shared_row_order(self) -> None:
        rows = list(range(180))
        col_a = [10_000 - i * 17 + (i % 3) for i in rows]
        col_b = [v * 2 + 7 for v in col_a]
        col_c = [i % 11 for i in rows]
        columns = [col_a, col_b, col_c]

        encoded = encode_int_columns(columns)
        self.assertEqual(decode_int_columns(encoded), columns)

        original, compressed, _comp_ns, _decomp_ns, loss_err, loss_ref = bench_reger_int_columns(
            columns
        )
        self.assertEqual(original, len(rows) * len(columns) * 8)
        self.assertGreater(compressed, 0)
        self.assertEqual(loss_err, 0.0)
        self.assertGreater(loss_ref, 0.0)

    def test_multi_column_reorder_uses_explicit_times_and_compact_order_payload(self) -> None:
        rows = list(range(512))
        times = [1_700_000_000_000_000 + i * 60_000_000 for i in rows]
        col_a = [50_000 - (i // 4) * 11 + (i % 4) for i in rows]
        col_b = [v * 3 - 17 for v in col_a]
        col_c = [(i % 16) * 5 for i in rows]
        columns = [col_a, col_b, col_c]

        encoded = encode_int_columns(columns, times=times)
        self.assertEqual(decode_int_columns(encoded), columns)

        original = len(rows) * len(columns) * 8
        self.assertLess(len(encoded), int(original * 0.18))

        original, compressed, _comp_ns, _decomp_ns, loss_err, loss_ref = bench_reger_int_columns(
            columns,
            times=times,
        )
        self.assertEqual(original, len(rows) * len(columns) * 8)
        self.assertEqual(loss_err, 0.0)
        self.assertGreater(loss_ref, 0.0)
        self.assertEqual(compressed, len(encoded))

    def test_time_stream_restore_handles_non_identity_value_order(self) -> None:
        rows = list(range(512))
        times = rows[:]
        col_a = [
            (i // 2) if i % 2 == 0 else (1_000_000 + i // 2)
            for i in rows
        ]
        col_b = [value * 3 + 11 for value in col_a]
        columns = [col_a, col_b]

        encoded = encode_int_columns(columns, times=times)
        self.assertEqual(decode_int_columns(encoded), columns)
        self.assertLess(len(encoded), int(len(rows) * len(columns) * 8 / 4))

        original, compressed, _comp_ns, _decomp_ns, loss_err, _loss_ref = bench_reger_int_columns(
            columns,
            times=times,
        )
        self.assertEqual(original, len(rows) * len(columns) * 8)
        self.assertEqual(compressed, len(encoded))
        self.assertEqual(loss_err, 0.0)

    def test_parallel_blocks_match_serial_payload(self) -> None:
        rows = list(range(520))
        columns = [
            [i * 7 + (i % 9) for i in rows],
            [100_000 - i * 11 + (i % 5) for i in rows],
        ]
        with patch.dict(os.environ, {"WEB_COMPRESSION_REGER_WORKERS": "1"}):
            serial = encode_int_columns(columns, times=rows)
        with patch.dict(os.environ, {"WEB_COMPRESSION_REGER_WORKERS": "2"}):
            parallel = encode_int_columns(columns, times=rows)

        self.assertEqual(parallel, serial)
        self.assertEqual(decode_int_columns(parallel), columns)

    def test_fast_profile_limits_row_orders_and_roundtrips(self) -> None:
        rows = list(range(96))
        times = [1_700_000_000_000_000 + i * 1_000 for i in rows]
        columns = [
            [i * 13 + (i % 7) for i in rows],
            [10_000 - i * 5 + (i % 3) for i in rows],
        ]
        with patch.dict(os.environ, {"WEB_COMPRESSION_REGER_PROFILE": "fast"}):
            self.assertEqual(_default_reorder_iterations(2), 0)
            self.assertLessEqual(len(_row_order_candidates(columns, times)), 2)
            encoded = encode_int_columns(columns, times=times)
            self.assertEqual(decode_int_columns(encoded), columns)
            original, compressed, _comp_ns, _decomp_ns, loss_err, loss_ref = (
                bench_reger_int_columns(columns, times=times)
            )
        self.assertEqual(original, len(rows) * len(columns) * 8)
        self.assertGreater(compressed, 0)
        self.assertEqual(loss_err, 0.0)
        self.assertGreater(loss_ref, 0.0)


if __name__ == "__main__":
    unittest.main()
