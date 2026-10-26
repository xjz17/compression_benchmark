from __future__ import annotations

import struct
import unittest

from backend.myapp.dbp_float64_codec import (
    HEADER_SIZE,
    MAX_VALUES,
    MAGIC32,
    MODE_DECIMAL,
    MODE_IEEE_BITS,
    bench_delta_zigzag_varint_float_columns,
    decode,
    decode32,
    encode,
    encode32,
)


HEADER = struct.Struct("<4sBBI")
U64 = struct.Struct("<Q")


def floats_from_words(words: list[int]) -> list[float]:
    return list(struct.unpack(f"<{len(words)}d", b"".join(U64.pack(v) for v in words)))


def packed(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}d", *values) if values else b""


class DBPFloat64CodecTest(unittest.TestCase):
    def assert_bitwise_round_trip(
        self, values: list[float], max_point: int = 18
    ) -> bytes:
        frame = encode(values, max_point=max_point)
        self.assertEqual(packed(decode(frame)), packed(values))
        return frame

    def test_decimal_golden_and_header(self) -> None:
        values = [1.25, 1.5, -2.75, 0.0]
        frame = self.assert_bitwise_round_trip(values, max_point=2)
        self.assertEqual(
            frame.hex(),
            "444246310002040000007d0000000000000032d106a604",
        )
        self.assertEqual(HEADER.unpack_from(frame), (b"DBF1", MODE_DECIMAL, 2, 4))

    def test_empty_is_canonical_raw_bits_frame(self) -> None:
        frame = encode([], max_point=0)
        self.assertEqual(frame, b"DBF1\x01\xff\x00\x00\x00\x00")
        self.assertEqual(decode(frame), [])

    def test_raw_mode_preserves_special_ieee_words(self) -> None:
        words = [
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000001,
            0x8000000000000001,
            0x7FEFFFFFFFFFFFFF,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000001,
            0x7FF0000000000001,
            0xFFF8000000001234,
        ]
        frame = self.assert_bitwise_round_trip(floats_from_words(words))
        self.assertEqual(frame[4], MODE_IEEE_BITS)
        self.assertEqual(frame[5], 0xFF)

    def test_float32_decimal_golden(self) -> None:
        raw = struct.pack("<4f", 1.25, 1.5, -2.75, 0.0)
        frame = encode32(raw, max_point=2)
        self.assertEqual(decode32(frame), raw)
        self.assertEqual(
            frame.hex(),
            "444246320002040000007d0000000000000032d106a604",
        )
        self.assertEqual(HEADER.unpack_from(frame), (MAGIC32, MODE_DECIMAL, 2, 4))

    def test_float32_raw_mode_preserves_special_words_and_nan_payloads(self) -> None:
        words = [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
            0xFFC01234,
        ]
        raw = struct.pack(f"<{len(words)}I", *words)
        frame = encode32(raw)
        self.assertEqual(frame[4], MODE_IEEE_BITS)
        self.assertEqual(frame[5], 0xFF)
        self.assertEqual(decode32(frame), raw)

    def test_float32_empty_and_input_validation(self) -> None:
        frame = encode32(b"", max_point=0)
        self.assertEqual(frame, b"DBF2\x01\xff\x00\x00\x00\x00")
        self.assertEqual(decode32(frame), b"")
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            encode32(b"\0")
        with self.assertRaisesRegex(ValueError, "maximum"):
            encode32(b"\0" * (4 * (MAX_VALUES + 1)))

    def test_float32_decoder_is_strict(self) -> None:
        raw_word_too_wide = HEADER.pack(MAGIC32, MODE_IEEE_BITS, 0xFF, 1) + U64.pack(
            1 << 32
        )
        malformed = [
            encode([1.0], 0),
            HEADER.pack(MAGIC32, MODE_IEEE_BITS, 0, 1) + U64.pack(0),
            HEADER.pack(MAGIC32, MODE_DECIMAL, 19, 1) + U64.pack(0),
            HEADER.pack(MAGIC32, MODE_DECIMAL, 0, 0),
            raw_word_too_wide,
            encode32(struct.pack("<f", 1.0), 0) + b"\0",
        ]
        for frame in malformed:
            with self.subTest(frame=frame.hex()):
                with self.assertRaises(ValueError):
                    decode32(frame)

    def test_modulo_2_64_delta_wrap_round_trips(self) -> None:
        values = [
            float(-9_000_000_000_000_000_000),
            float(9_000_000_000_000_000_000),
            float(-9_000_000_000_000_000_000),
        ]
        frame = self.assert_bitwise_round_trip(values, max_point=0)
        self.assertEqual(frame[4], MODE_DECIMAL)

    def test_negative_or_missing_max_point_uses_eighteen(self) -> None:
        values = [1.25]
        negative = encode(values, max_point=-1)
        missing = encode(values, max_point=None)
        self.assertEqual(negative, missing)
        self.assertEqual(negative[5], 18)

    def test_encode_rejects_oversized_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum"):
            encode([0.0] * (MAX_VALUES + 1))

    def test_benchmark_chunks_and_counts_complete_frames(self) -> None:
        values = [float(i % 97) for i in range(MAX_VALUES + 1)]
        result = bench_delta_zigzag_varint_float_columns([values], [0])
        self.assertEqual(result[0], len(values) * 8)
        self.assertGreaterEqual(result[1], 2 * HEADER_SIZE)
        self.assertGreater(result[2], 0)
        self.assertGreater(result[3], 0)
        self.assertEqual(result[4], 0.0)

    def test_decoder_rejects_malformed_frames(self) -> None:
        good_one = encode([1.0], max_point=0)
        malformed = [
            b"",
            b"NOPE" + good_one[4:],
            HEADER.pack(b"DBF1", 2, 0, 0),
            HEADER.pack(b"DBF1", MODE_DECIMAL, 19, 1) + b"\0" * 8,
            HEADER.pack(b"DBF1", MODE_IEEE_BITS, 0, 1) + b"\0" * 8,
            HEADER.pack(b"DBF1", MODE_DECIMAL, 0, 0),
            encode([]) + b"\0",
            HEADER.pack(b"DBF1", MODE_IEEE_BITS, 0xFF, 1) + b"\0" * 7,
            HEADER.pack(b"DBF1", MODE_IEEE_BITS, 0xFF, 2) + b"\0" * 8 + b"\x80",
            HEADER.pack(b"DBF1", MODE_IEEE_BITS, 0xFF, 2) + b"\0" * 8 + b"\x80\0",
            HEADER.pack(b"DBF1", MODE_IEEE_BITS, 0xFF, 2) + b"\0" * 8 + b"\x80" * 10,
            HEADER.pack(b"DBF1", MODE_IEEE_BITS, 0xFF, MAX_VALUES + 1),
            good_one + b"\0",
        ]
        for frame in malformed:
            with self.subTest(frame=frame.hex()):
                with self.assertRaises(ValueError):
                    decode(frame)


if __name__ == "__main__":
    unittest.main()
