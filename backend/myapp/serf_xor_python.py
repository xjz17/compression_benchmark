"""
Pure-Python Serf XOR float64 compressor / decompressor aligned with
``external/compression/Serf`` (``SerfXORCompressor`` / ``SerfXORDecompressor``,
``OutputBitStream`` / ``InputBitStream``, ``SerfUtils64::FindAppLong``,
``PostOfficeSolver``).

CLI parity: same parameters as ``Serf/serf_cmd/cmd_interface.cc`` — window 1000,
``adjust_digit=0``, ``max_diff`` = absolute error bound.

Not a Python reimplementation of SZ; SZ lives under ``external/compression/SZ`` (C).
"""
from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple

# --- IEEE double bits (little-endian pack matches typical Linux / Serf tests) ---
def _double_to_long_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _long_bits_to_double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & ((1 << 64) - 1)))[0]


_NAN_PACK = struct.pack("<d", float("nan"))
NAN_BITS = struct.unpack("<Q", _NAN_PACK)[0]


_K_BIT_WEIGHT: Tuple[int, ...] = tuple(
    1 << i for i in range(64)
)

K_POSITION_LENGTH_2_BITS: Tuple[int, ...] = (
    0, 0, 1, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
)

K_POW2Z: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)

_BRANCH_LESS_TABLE: Tuple[int, ...] = (
    32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31,
)


def _clz_u64(x: int) -> int:
    x &= (1 << 64) - 1
    if x == 0:
        return 64
    return 64 - x.bit_length()


def _ctz_u64(x: int) -> int:
    x &= (1 << 64) - 1
    if x == 0:
        return 64
    return (x & -x).bit_length() - 1


def _find_app_long_inner(
    min_double: float,
    max_double: float,
    sign: int,
    original: float,
    last_long: int,
    max_diff: float,
    adjust_digit: float,
) -> int:
    min_u = _double_to_long_bits(min_double) & 0x7FFFFFFFFFFFFFFF
    max_u = _double_to_long_bits(max_double)
    xor_mv = (min_u ^ max_u) & ((1 << 64) - 1)
    leading_zeros = _clz_u64(xor_mv)
    front_mask = ((1 << 64) - 1) & (((1 << 64) - 1) << (64 - leading_zeros))
    shift = 64 - leading_zeros
    sign_u = sign & ((1 << 64) - 1)

    while shift >= 0:
        front = front_mask & min_u
        rear = (~front_mask) & (last_long & ((1 << 64) - 1))
        append = (front | rear) & ((1 << 64) - 1)

        condition1 = append >= min_u and append <= max_u
        result_long = (append ^ sign_u) & ((1 << 64) - 1) if condition1 else 0
        diff = _long_bits_to_double(result_long) - adjust_digit - original
        diff_ok = -max_diff <= diff <= max_diff
        if condition1 and diff_ok:
            return int(result_long)

        append = (append + _K_BIT_WEIGHT[shift]) & 0x7FFFFFFFFFFFFFFF
        condition2 = append <= max_u
        result_long = (append ^ sign_u) & ((1 << 64) - 1) if condition2 else 0
        diff = _long_bits_to_double(result_long) - adjust_digit - original
        diff_ok = -max_diff <= diff <= max_diff
        if condition2 and diff_ok:
            return int(result_long)

        front_mask = (front_mask >> 1) & ((1 << 64) - 1)
        shift -= 1

    return _double_to_long_bits(original + adjust_digit)


def find_app_long(
    min_v: float,
    max_v: float,
    v: float,
    last_long: int,
    max_diff: float,
    adjust_digit: float,
) -> int:
    if min_v >= 0:
        return _find_app_long_inner(min_v, max_v, 0, v, last_long, max_diff, adjust_digit)
    if max_v <= 0:
        return _find_app_long_inner(-max_v, -min_v, 0x8000000000000000, v, last_long, max_diff, adjust_digit)
    if (last_long >> 63) == 0:
        return _find_app_long_inner(0, max_v, 0, v, last_long, max_diff, adjust_digit)
    return _find_app_long_inner(0, -min_v, 0x8000000000000000, v, last_long, max_diff, adjust_digit)


class _PostOfficeResult:
    __slots__ = ("office_positions", "total_app_cost")

    def __init__(self, office_positions: List[int], total_app_cost: int) -> None:
        self.office_positions = office_positions
        self.total_app_cost = total_app_cost


def _cal_total_count_and_non_zeros_counts(
    arr: List[int],
) -> Tuple[int, int, List[int], List[int]]:
    n = len(arr)
    pre_non = [0] * n
    post_non = [0] * n
    non_zeros_count = n
    total_count = arr[0]
    pre_non[0] = 1
    for i in range(1, n):
        total_count += arr[i]
        if arr[i] == 0:
            non_zeros_count -= 1
            pre_non[i] = pre_non[i - 1]
        else:
            pre_non[i] = pre_non[i - 1] + 1
    for i in range(n):
        post_non[i] = non_zeros_count - pre_non[i]
    return total_count, non_zeros_count, pre_non, post_non


def _build_post_office(
    arr: List[int],
    num: int,
    non_zeros_count: int,
    pre_non: List[int],
    post_non: List[int],
) -> _PostOfficeResult:
    original_num = num
    num = min(num, non_zeros_count)
    alen = len(arr)
    dp = [[0] * num for _ in range(alen)]
    pre = [[-1] * num for _ in range(alen)]
    dp[0][0] = 0
    pre[0][0] = -1

    for i in range(1, alen):
        if arr[i] == 0:
            continue
        j_lo = max(1, num + i - alen)
        j_hi = min(i, num - 1)
        for j in range(j_lo, j_hi + 1):
            if j >= num:
                break
            if i > 1 and j == 1:
                s = 0
                for k in range(1, i):
                    s += arr[k] * k
                dp[i][j] = s
                pre[i][j] = 0
            else:
                if pre_non[i] < j + 1 or post_non[i] < num - 1 - j:
                    continue
                app_cost = 2**62  # large
                pre_k = 0
                for k in range(j - 1, i):
                    if (arr[k] == 0 and k > 0) or pre_non[k] < j or post_non[k] < num - j:
                        continue
                    sm = dp[k][j - 1]
                    for p in range(k + 1, i):
                        sm += arr[p] * (p - k)
                    if app_cost > sm:
                        app_cost = sm
                        pre_k = k
                        if sm == 0:
                            break
                if app_cost < 2**62:
                    dp[i][j] = app_cost
                    pre[i][j] = pre_k

    temp_total_app_cost = 2**62
    temp_best_last = 2**62
    for i in range(num - 1, alen):
        if num - 1 == 0 and i > 0:
            break
        if (arr[i] == 0 and i > 0) or pre_non[i] < num:
            continue
        sm = dp[i][num - 1]
        for j in range(i + 1, alen):
            sm += arr[j] * (j - i)
        if temp_total_app_cost > sm:
            temp_total_app_cost = sm
            temp_best_last = i

    office_positions = [0] * num
    ii = 1
    t = temp_best_last
    while t != -1 and t != 2**62:
        office_positions[num - ii] = t
        t = pre[t][num - ii]
        ii += 1

    if original_num > non_zeros_count:
        modifying = [0] * original_num
        j = 0
        k = 0
        while j < original_num and k < num:
            if j - k < original_num - num and j < office_positions[k]:
                modifying[j] = j
                j += 1
            else:
                modifying[j] = office_positions[k]
                j += 1
                k += 1
        office_positions = modifying

    return _PostOfficeResult(office_positions, temp_total_app_cost if temp_total_app_cost < 2**62 else 0)


def _init_round_and_representation(
    distribution: List[int],
    representation: List[int],
    round_: List[int],
) -> List[int]:
    n = len(distribution)
    total_count, non_zeros_count, pre_non, post_non = _cal_total_count_and_non_zeros_counts(distribution)
    max_z = min(K_POSITION_LENGTH_2_BITS[non_zeros_count] if non_zeros_count < len(K_POSITION_LENGTH_2_BITS) else 5, 5)
    total_cost = 2**62
    positions: List[int] = []
    z = 0
    while z <= max_z:
        present_cost = total_count * z
        if present_cost >= total_cost:
            break
        num = K_POW2Z[z]
        por = _build_post_office(distribution, num, non_zeros_count, pre_non, post_non)
        temp_total_cost = por.total_app_cost + present_cost
        if temp_total_cost < total_cost:
            total_cost = temp_total_cost
            positions = list(por.office_positions)
        z += 1

    representation[0] = 0
    round_[0] = 0
    i_pt = 1
    for j in range(1, n):
        magic = 1 if (i_pt < len(positions) and j == positions[i_pt]) else 0
        representation[j] = representation[j - 1] + magic
        round_[j] = j if magic else round_[j - 1]
        i_pt += magic
    return positions


def _write_positions(positions: List[int], out: "OutputBitStream") -> int:
    s = out.write_int(len(positions), 5)
    for p in positions:
        s += out.write_int(p, 6)
    return s


class OutputBitStream:
    __slots__ = ("data", "buffer", "cursor", "bit_in_buffer")

    def __init__(self, buffer_size_words: int) -> None:
        self.data = [0] * (buffer_size_words + 1)
        self.buffer = 0
        self.cursor = 0
        self.bit_in_buffer = 0

    def write(self, content: int, length: int) -> int:
        if length == 0:
            return 0
        content &= (1 << 64) - 1
        content <<= 64 - length
        self.buffer |= content >> self.bit_in_buffer
        self.bit_in_buffer += length
        if self.bit_in_buffer >= 32:
            self.data[self.cursor] = (self.buffer >> 32) & 0xFFFFFFFF
            self.cursor += 1
            self.buffer = (self.buffer << 32) & ((1 << 64) - 1)
            self.bit_in_buffer -= 32
        return length

    def write_long(self, content: int, length: int) -> int:
        if length == 0:
            return 0
        if length > 32:
            self.write(content >> (length - 32), 32)
            self.write(content & ((1 << (length - 32)) - 1), length - 32)
            return length
        return self.write(content, length)

    def write_int(self, content: int, length: int) -> int:
        return self.write(content & 0xFFFFFFFF, length)

    def flush(self) -> None:
        if self.bit_in_buffer:
            self.data[self.cursor] = (self.buffer >> 32) & 0xFFFFFFFF
            self.cursor += 1
            self.buffer = 0
            self.bit_in_buffer = 0

    def get_buffer(self, nbytes: int) -> bytes:
        nwords = (nbytes + 3) // 4 + 2
        raw = b"".join(
            struct.pack(">I", self.data[i] & 0xFFFFFFFF) for i in range(min(nwords, len(self.data)))
        )
        return raw[:nbytes]

    def refresh(self) -> None:
        self.cursor = 0
        self.bit_in_buffer = 0
        self.buffer = 0


class InputBitStream:
    """
    Bit order matches Serf ``OutputBitStream::GetBuffer`` / ``InputBitStream``:
    big-endian 32-bit words, MSB first within each word.
    """

    __slots__ = ("_bits", "_i")

    def __init__(self, raw: bytes) -> None:
        self.set_buffer(raw)

    def set_buffer(self, raw: bytes) -> None:
        pad = raw + bytes((4 - len(raw) % 4) % 4)
        bits: List[int] = []
        for j in range(0, len(pad), 4):
            w = int.from_bytes(pad[j : j + 4], "big")
            for k in range(32):
                bits.append((w >> (31 - k)) & 1)
        self._bits = bits
        self._i = 0

    def read_long(self, length: int) -> int:
        if length == 0:
            return 0
        v = 0
        for _ in range(length):
            v = (v << 1) | self._bits[self._i]
            self._i += 1
        return v

    def read_int(self, length: int) -> int:
        return int(self.read_long(length))

    def read_bit(self) -> int:
        return int(self.read_long(1))


LEADING_REPRESENTATION: List[int] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
]
LEADING_ROUND: List[int] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    8, 8, 8, 8, 12, 12, 12, 12,
    16, 16, 18, 18, 20, 20, 22, 22,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
    24, 24, 24, 24, 24, 24, 24, 24,
]
TRAILING_REPRESENTATION: List[int] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 6, 6, 6, 6, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
]
TRAILING_ROUND: List[int] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 22, 22,
    22, 22, 22, 22, 28, 28, 28, 28,
    32, 32, 32, 32, 36, 36, 36, 36,
    40, 40, 42, 42, 42, 42, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46,
]


class SerfXORCompressor:
    def __init__(self, window_size: int, max_diff: float, adjust_digit: int) -> None:
        self.k_max_diff = max_diff
        self.k_adjust_digit = float(adjust_digit)
        self.k_window_size = window_size
        self.stored_val = _double_to_long_bits(2.0)
        buf_words = int(math.floor(((window_size + 1) * 8 + window_size // 8 + 1) * 1.2))
        self.output_buffer = OutputBitStream(buf_words)
        self.compressed_size_this_block = self.output_buffer.write_int(0, 1)
        self.compressed_bytes_last_block = b""
        self.compressed_size_last_block = 0
        self.compressed_size_this_window = 0
        self.number_of_values_this_window = 0
        self.compression_ratio_last_window = 0.0
        self.leading_representation = list(LEADING_REPRESENTATION)
        self.leading_round = list(LEADING_ROUND)
        self.trailing_representation = list(TRAILING_REPRESENTATION)
        self.trailing_round = list(TRAILING_ROUND)
        self.leading_bits_per_value = 3
        self.trailing_bits_per_value = 3
        self.lead_distribution = [0] * 64
        self.trail_distribution = [0] * 64
        self.stored_leading_zeros = 2147483647  # std::numeric_limits<int>::max()
        self.stored_trailing_zeros = 2147483647

    def add_value(self, v: float) -> None:
        adjust_d = self.k_adjust_digit
        if abs(_long_bits_to_double(self.stored_val) - adjust_d - v) > self.k_max_diff:
            adjust_value = v + adjust_d
            this_val = find_app_long(
                adjust_value - self.k_max_diff,
                adjust_value + self.k_max_diff,
                v,
                self.stored_val,
                self.k_max_diff,
                adjust_d,
            )
        else:
            this_val = self.stored_val
        self.compressed_size_this_block += self._compress_value(this_val)
        self.stored_val = this_val
        self.number_of_values_this_window += 1

    def close(self) -> None:
        self.compressed_size_this_block += self._compress_value(NAN_BITS)
        self.output_buffer.flush()
        nbytes = int(math.ceil(self.compressed_size_this_block / 8.0))
        self.compressed_bytes_last_block = self.output_buffer.get_buffer(nbytes)
        self.output_buffer.refresh()
        self.compressed_size_last_block = self.compressed_size_this_block
        self.compressed_size_this_block = self._update_positions_if_needed()

    def _compress_value(self, value: int) -> int:
        this_size = 0
        xor_result = (self.stored_val ^ value) & ((1 << 64) - 1)
        out = self.output_buffer

        if xor_result == 0:
            this_size += out.write_int(1, 2)
        else:
            lc = _clz_u64(xor_result)
            tc = _ctz_u64(xor_result)
            leading_zeros = self.leading_round[lc]
            trailing_zeros = self.trailing_round[tc]
            self.lead_distribution[lc] += 1
            self.trail_distribution[tc] += 1

            cond1 = (
                leading_zeros >= self.stored_leading_zeros
                and trailing_zeros >= self.stored_trailing_zeros
                and (leading_zeros - self.stored_leading_zeros) + (trailing_zeros - self.stored_trailing_zeros)
                < 1 + self.leading_bits_per_value + self.trailing_bits_per_value
            )
            if cond1:
                center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros
                length = 1 + center_bits
                if length > 64:
                    out.write_int(1, 1)
                    out.write_long(xor_result >> self.stored_trailing_zeros, center_bits)
                else:
                    out.write_long((1 << center_bits) | (xor_result >> self.stored_trailing_zeros), 1 + center_bits)
                this_size += length
            else:
                self.stored_leading_zeros = leading_zeros
                self.stored_trailing_zeros = trailing_zeros
                center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros
                length = 2 + self.leading_bits_per_value + self.trailing_bits_per_value + center_bits
                pack_lead = self.leading_representation[self.stored_leading_zeros]
                pack_trail = self.trailing_representation[self.stored_trailing_zeros]
                low = ((pack_lead << self.trailing_bits_per_value) | pack_trail) << center_bits | (
                    xor_result >> self.stored_trailing_zeros
                )
                if length > 64:
                    out.write_int(
                        (pack_lead << self.trailing_bits_per_value) | pack_trail,
                        2 + self.leading_bits_per_value + self.trailing_bits_per_value,
                    )
                    out.write_long(xor_result >> self.stored_trailing_zeros, center_bits)
                else:
                    out.write_long(low, length)
                this_size += length
        return this_size

    def _update_positions_if_needed(self) -> int:
        ob = self.output_buffer
        if self.number_of_values_this_window < self.k_window_size:
            self.compressed_size_this_window += self.compressed_size_last_block
            return ob.write_int(0, 1)
        compression_ratio_this_window = self.compressed_size_this_window / (
            self.number_of_values_this_window * 64
        )
        if self.compression_ratio_last_window < compression_ratio_this_window:
            lead_positions = _init_round_and_representation(
                self.lead_distribution, self.leading_representation, self.leading_round
            )
            self.leading_bits_per_value = K_POSITION_LENGTH_2_BITS[len(lead_positions)]
            trail_positions = _init_round_and_representation(
                self.trail_distribution, self.trailing_representation, self.trailing_round
            )
            self.trailing_bits_per_value = K_POSITION_LENGTH_2_BITS[len(trail_positions)]
            length = (
                ob.write_int(1, 1)
                + _write_positions(lead_positions, ob)
                + _write_positions(trail_positions, ob)
            )
        else:
            length = ob.write_int(0, 1)
        self.compression_ratio_last_window = compression_ratio_this_window
        self.lead_distribution = [0] * 64
        self.trail_distribution = [0] * 64
        self.compressed_size_this_window = 0
        self.number_of_values_this_window = 0
        return length


class SerfXORDecompressor:
    def __init__(self, adjust_digit: int) -> None:
        self.adjust_digit = float(adjust_digit)
        self.stored_val = _double_to_long_bits(2.0)
        self.stored_leading_zeros = 2147483647
        self.stored_trailing_zeros = 2147483647
        self.input_bit_stream: InputBitStream | None = None
        self.leading_representation = [0, 8, 12, 16, 18, 20, 22, 24]
        self.trailing_representation = [0, 22, 28, 32, 36, 40, 42, 46]
        self.leading_bits_per_value = 3
        self.trailing_bits_per_value = 3

    def decompress(self, bs: bytes) -> List[float]:
        ibs = InputBitStream(bs)
        self.input_bit_stream = ibs
        self._update_positions_if_needed()
        values: List[float] = []
        while True:
            value = self._read_value()
            if value == NAN_BITS:
                break
            values.append(_long_bits_to_double(value) - self.adjust_digit)
            self.stored_val = value
        return values

    def _read_value(self) -> int:
        ibs = self.input_bit_stream
        assert ibs is not None
        value = self.stored_val
        if ibs.read_int(1) == 1:
            center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros
            n = ibs.read_long(center_bits)
            value = self.stored_val ^ (n << self.stored_trailing_zeros)
        elif ibs.read_int(1) == 0:
            lead_and_trail = ibs.read_int(self.leading_bits_per_value + self.trailing_bits_per_value)
            trail_mask = ~(0xFFFFFFFF << self.trailing_bits_per_value) & 0xFFFFFFFF
            lead = lead_and_trail >> self.trailing_bits_per_value
            trail = trail_mask & lead_and_trail
            self.stored_leading_zeros = self.leading_representation[lead]
            self.stored_trailing_zeros = self.trailing_representation[trail]
            center_bits = 64 - self.stored_leading_zeros - self.stored_trailing_zeros
            n = ibs.read_long(center_bits)
            value = self.stored_val ^ (n << self.stored_trailing_zeros)
        return value & ((1 << 64) - 1)

    def _update_positions_if_needed(self) -> None:
        ibs = self.input_bit_stream
        assert ibs is not None
        if ibs.read_bit():
            self._update_leading_representation()
            self._update_trailing_representation()

    def _update_leading_representation(self) -> None:
        ibs = self.input_bit_stream
        assert ibs is not None
        num = ibs.read_int(5)
        num = _BRANCH_LESS_TABLE[num]
        self.leading_bits_per_value = K_POSITION_LENGTH_2_BITS[num]
        self.leading_representation = [ibs.read_int(6) for _ in range(num)]

    def _update_trailing_representation(self) -> None:
        ibs = self.input_bit_stream
        assert ibs is not None
        num = ibs.read_int(5)
        num = _BRANCH_LESS_TABLE[num]
        self.trailing_bits_per_value = K_POSITION_LENGTH_2_BITS[num]
        self.trailing_representation = [ibs.read_int(6) for _ in range(num)]


def compress_floats(
    values: Sequence[float],
    *,
    window_size: int = 1000,
    max_diff: float,
    adjust_digit: int = 0,
) -> bytes:
    c = SerfXORCompressor(window_size, max_diff, adjust_digit)
    for v in values:
        c.add_value(float(v))
    c.close()
    return c.compressed_bytes_last_block


def decompress_floats(blob: bytes, *, adjust_digit: int = 0) -> List[float]:
    return SerfXORDecompressor(adjust_digit).decompress(blob)


if __name__ == "__main__":
    _sample = [1.0, 1.1, 1.2, 3.5, -0.25]
    _bound = 0.01
    _b = compress_floats(_sample, max_diff=_bound)
    _out = decompress_floats(_b, adjust_digit=0)
    for _a, _e in zip(_out, _sample):
        assert abs(_a - _e) <= _bound * 1.0001 + 1e-6