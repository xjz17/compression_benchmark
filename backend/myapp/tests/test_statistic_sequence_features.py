import math
import struct
from collections import Counter

from django.test import SimpleTestCase

from myapp.statistic import byte_entropy, repeat_words, sortedness


class StatisticSequenceFeatureTests(SimpleTestCase):
    def test_non_finite_values_break_runs_and_adjacent_pairs(self) -> None:
        sequence = [1.0] * 7 + [math.nan] + [1.0] * 7

        self.assertEqual(repeat_words(sequence, limit=8), 0.0)
        self.assertEqual(sortedness([2.0, math.nan, 1.0]), 0.0)

    def test_repeat_ratio_counts_complete_maximal_run(self) -> None:
        self.assertEqual(repeat_words([3.0] * 8 + [4.0], limit=8), 8 / 9)

    def test_byte_entropy_uses_little_endian_float64(self) -> None:
        raw = b"".join(struct.pack("<d", value) for value in (1.0, 2.0))
        counts = Counter(raw)
        expected = -sum((count / len(raw)) * math.log2(count / len(raw)) for count in counts.values())

        self.assertAlmostEqual(byte_entropy([1.0, 2.0]), expected)
