from __future__ import annotations

import struct
import unittest

from myapp.tsfile_encoding import p04_i64


def _double_bits(values: list[float]) -> list[bytes]:
    return [struct.pack("<d", float(value)) for value in values]


class TsFileP04I64Tests(unittest.TestCase):
    def test_sprintz_double_preserves_problematic_sign_bits(self) -> None:
        values = [
            -200.0,
            200.0,
            0.7,
            1.1,
            255.93,
            -255.93,
            2.16,
            -2.16,
            3.39,
            -3.39,
        ] * 17

        decoded = p04_i64.decode_sprintz_double(p04_i64.encode_sprintz_double(values))

        self.assertEqual(_double_bits(decoded), _double_bits(values))


if __name__ == "__main__":
    unittest.main()
