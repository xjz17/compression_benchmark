"""Exact int64/byte adapters used by the PV2F 86-method action catalog."""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
import struct
import zlib
from typing import Sequence

import numpy as np


_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_MASK64 = (1 << 64) - 1


def _raw_i64(values: Sequence[int]) -> bytes:
    return np.asarray(values, dtype="<i8").tobytes(order="C")


def _raw_to_i64(raw: bytes, count: int) -> list[int]:
    if len(raw) != count * 8:
        raise ValueError(f"int64 payload length mismatch: {len(raw)} != {count * 8}")
    return np.frombuffer(raw, dtype="<i8").astype(np.int64, copy=False).tolist()


def _u32_limbs(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        bits = int(value) & _MASK64
        out.extend((bits & 0xFFFFFFFF, bits >> 32))
    return out


def _u16_limbs(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        bits = int(value) & _MASK64
        out.extend((bits >> shift) & 0xFFFF for shift in (0, 16, 32, 48))
    return out


def _limbs_to_i64(limbs: Sequence[int], count: int, limb_bits: int) -> list[int]:
    per_value = 64 // limb_bits
    if len(limbs) != count * per_value:
        raise ValueError(f"limb count mismatch: {len(limbs)} != {count * per_value}")
    mask = (1 << limb_bits) - 1
    out: list[int] = []
    for start in range(0, len(limbs), per_value):
        bits = 0
        for index in range(per_value):
            limb = int(limbs[start + index])
            if limb < 0 or limb > mask:
                raise ValueError("decoded limb is out of range")
            bits |= limb << (index * limb_bits)
        out.append(bits - (1 << 64) if bits & (1 << 63) else bits)
    return out


class _PongoBitWriter:
    def __init__(self, prefix: bytes = b"") -> None:
        self.data = bytearray(prefix)
        self.used = 0

    def write_bit(self, value: bool) -> None:
        if self.used == 0:
            self.data.append(0)
        if value:
            self.data[-1] |= 1 << (7 - self.used)
        self.used = (self.used + 1) & 7

    def write_bits(self, value: int, count: int) -> None:
        for bit in range(count - 1, -1, -1):
            self.write_bit(bool((int(value) >> bit) & 1))


class _PongoBitReader:
    def __init__(self, data: bytes, offset: int = 0) -> None:
        self.data = data
        self.byte_index = offset
        self.used = 0

    def read_bit(self) -> bool:
        if self.byte_index >= len(self.data):
            raise ValueError("truncated Pongo bit stream")
        value = bool((self.data[self.byte_index] >> (7 - self.used)) & 1)
        self.used += 1
        if self.used == 8:
            self.used = 0
            self.byte_index += 1
        return value

    def read_bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | int(self.read_bit())
        return value


def _double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _bits_double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits) & _MASK64))[0]


def _reverse_low_bits(value: int, count: int) -> int:
    reversed_value = 0
    for index in range(count):
        reversed_value = (reversed_value << 1) | ((int(value) >> index) & 1)
    return reversed_value


def _pongo_decimal_fraction(value: float) -> tuple[int, int] | None:
    text = repr(abs(float(value)))
    mantissa, separator, exponent_text = text.lower().partition("e")
    exponent = int(exponent_text) if separator else 0
    if "." in mantissa:
        left, right = mantissa.split(".", 1)
        digits = left + right
        decimal_index = len(left) + exponent
    else:
        digits = mantissa
        decimal_index = len(digits) + exponent
    if decimal_index <= 0:
        fraction = "0" * (-decimal_index) + digits
    elif decimal_index >= len(digits):
        fraction = ""
    else:
        fraction = digits[decimal_index:]
    fraction = fraction.rstrip("0")
    if not fraction or len(fraction) > 16:
        return None
    first = len(fraction) - len(fraction.lstrip("0"))
    if first > 15 or first == len(fraction):
        return None
    decimal_fraction = int(fraction[first:])
    return (first, decimal_fraction) if decimal_fraction else None


def _pongo_restore(reencoded_bits: int, leading_decimal_zeros: int) -> int | None:
    biased_exponent = (int(reencoded_bits) >> 52) & 0x7FF
    if biased_exponent == 0x7FF:
        return None
    available = 52 - max(0, biased_exponent - 1023)
    if available <= 0:
        return None
    mask = (1 << available) - 1
    field = int(reencoded_bits) & mask
    if field == 0:
        return None
    trailing = (field & -field).bit_length() - 1
    fraction_bits = available - trailing
    decimal_fraction = _reverse_low_bits(field >> trailing, fraction_bits)
    if decimal_fraction == 0:
        return None
    reencoded = _bits_double(reencoded_bits)
    sign = "-" if (int(reencoded_bits) >> 63) else ""
    decimal = f"{sign}{math.trunc(abs(reencoded))}.{'0' * leading_decimal_zeros}{decimal_fraction}"
    try:
        return _double_bits(float(decimal))
    except ValueError:
        return None


def _pongo_reencode(value: float) -> tuple[int, int] | None:
    if not math.isfinite(value) or value == 0.0:
        return None
    decimal = _pongo_decimal_fraction(value)
    if decimal is None:
        return None
    leading_zeros, decimal_fraction = decimal
    original_bits = _double_bits(value)
    biased_exponent = (original_bits >> 52) & 0x7FF
    available = 52 - max(0, biased_exponent - 1023)
    fraction_bits = decimal_fraction.bit_length()
    if available <= 0 or fraction_bits > available:
        return None
    field = _reverse_low_bits(decimal_fraction, fraction_bits) << (available - fraction_bits)
    mask = (1 << available) - 1
    reencoded_bits = (original_bits & ~mask) | field
    restored = _pongo_restore(reencoded_bits, leading_zeros)
    if restored != original_bits:
        return None
    return reencoded_bits, leading_zeros


def _pongo_write_xor(writer: _PongoBitWriter, xor_value: int, window: list[int]) -> None:
    if xor_value == 0:
        writer.write_bit(False)
        return
    writer.write_bit(True)
    leading = 64 - int(xor_value).bit_length()
    trailing = (int(xor_value) & -int(xor_value)).bit_length() - 1
    if window[0] >= 0 and leading >= window[0] and trailing >= window[1]:
        writer.write_bit(False)
        writer.write_bits(int(xor_value) >> window[1], 64 - window[0] - window[1])
        return
    writer.write_bit(True)
    significant = 64 - leading - trailing
    writer.write_bits(leading, 6)
    writer.write_bits(0 if significant == 64 else significant, 6)
    writer.write_bits(int(xor_value) >> trailing, significant)
    window[:] = (leading, trailing)


def _pongo_read_xor(reader: _PongoBitReader, stored: int, window: list[int]) -> int:
    if not reader.read_bit():
        return stored
    new_window = reader.read_bit()
    if new_window:
        leading = reader.read_bits(6)
        significant = reader.read_bits(6) or 64
        trailing = 64 - leading - significant
        if trailing < 0:
            raise ValueError("invalid Pongo XOR window")
        window[:] = (leading, trailing)
    elif window[0] < 0:
        raise ValueError("Pongo reused an unset XOR window")
    significant = 64 - window[0] - window[1]
    return stored ^ (reader.read_bits(significant) << window[1])


def _pongo_encode(values: Sequence[float]) -> bytes:
    writer = _PongoBitWriter(_U64.pack(len(values)))
    if not values:
        return bytes(writer.data)
    stored = _double_bits(values[0])
    writer.write_bits(stored, 64)
    window = [-1, 0]
    for value in values[1:]:
        original = _double_bits(value)
        reencoded = _pongo_reencode(float(value))
        if reencoded is None:
            writer.write_bits(1, 2)
            target = original
        else:
            target, decimal_zeros = reencoded
            if decimal_zeros == 0:
                writer.write_bit(True)
            else:
                writer.write_bits(0, 2)
                writer.write_bits(decimal_zeros, 4)
        _pongo_write_xor(writer, stored ^ target, window)
        stored = target
    return bytes(writer.data)


def _pongo_decode(payload: bytes) -> list[float]:
    if len(payload) < 8:
        raise ValueError("truncated Pongo header")
    count = _U64.unpack_from(payload, 0)[0]
    if count == 0:
        if len(payload) != 8:
            raise ValueError("trailing Pongo bytes for empty stream")
        return []
    if count > len(payload) * 8 + 1:
        raise ValueError("invalid Pongo count")
    reader = _PongoBitReader(payload, 8)
    stored = reader.read_bits(64)
    out = [_bits_double(stored)]
    window = [-1, 0]
    for _ in range(1, count):
        decimal_zeros = 0
        if reader.read_bit():
            reencoded = True
        elif reader.read_bit():
            reencoded = False
        else:
            reencoded = True
            decimal_zeros = reader.read_bits(4)
        stored = _pongo_read_xor(reader, stored, window)
        original = _pongo_restore(stored, decimal_zeros) if reencoded else stored
        if original is None:
            raise ValueError("invalid Pongo decimal restoration")
        out.append(_bits_double(original))
    return out


def _pack_float64_limb_codec(values: Sequence[int], method: str) -> bytes:
    if method == "Pongo":
        return _pongo_encode([_bits_double(int(value) & _MASK64) for value in values])
    limbs = _u16_limbs(values) if method == "Camel" else _u32_limbs(values)
    floats = [float(value) for value in limbs]
    if method == "Camel":
        from .camel_python import CamelCompressor

        out = bytearray(b"CAMI" + _U32.pack(len(floats)))
        for start in range(0, len(floats), 1000):
            chunk = floats[start : start + 1000]
            encoder = CamelCompressor()
            for value in chunk:
                encoder.add_value(value)
            encoder.close()
            body = encoder.get_bytes()
            out.extend(struct.pack("<II", len(chunk), len(body)))
            out.extend(body)
        return bytes(out)
    if method == "FPZIP":
        import fpzip

        return fpzip.compress(np.asarray(floats, dtype="<f8"), precision=0, order="C")
    raise KeyError(method)


def _unpack_float64_limb_codec(payload: bytes, count: int, method: str) -> list[int]:
    if method == "Pongo":
        floats = _pongo_decode(payload)
        if len(floats) != count:
            raise ValueError(f"Pongo decoded count mismatch: {len(floats)} != {count}")
        bits = [_double_bits(value) for value in floats]
        return [value - (1 << 64) if value & (1 << 63) else value for value in bits]
    elif method == "Camel":
        from .camel_python import CamelDecompressor

        if len(payload) < 8 or payload[:4] != b"CAMI":
            raise ValueError("invalid Camel int64 wrapper")
        limb_count = _U32.unpack_from(payload, 4)[0]
        position = 8
        floats: list[float] = []
        while len(floats) < limb_count:
            if position + 8 > len(payload):
                raise ValueError("truncated Camel chunk header")
            chunk_count, size = struct.unpack_from("<II", payload, position)
            position += 8
            if position + size > len(payload):
                raise ValueError("truncated Camel chunk")
            decoder = CamelDecompressor(payload[position : position + size])
            position += size
            floats.extend(decoder.read_pair() for _ in range(chunk_count))
        if position != len(payload) or len(floats) != limb_count:
            raise ValueError("Camel payload boundary mismatch")
    elif method == "FPZIP":
        import fpzip

        floats = fpzip.decompress(payload, order="C").reshape(-1).tolist()
    else:
        raise KeyError(method)
    limbs = [int(value) for value in floats]
    if any(float(limb) != float(value) for limb, value in zip(limbs, floats)):
        raise ValueError(f"{method} decoded a non-integral limb")
    return _limbs_to_i64(limbs, count, 16 if method == "Camel" else 32)


def _pack_elf(values: Sequence[int], method: str) -> bytes:
    from .elf32_python import ElfCompressor32, ElfOnChimpCompressor32

    limbs = _u16_limbs(values)
    out = bytearray(b"ELFI" + _U32.pack(len(limbs)))
    for start in range(0, len(limbs), 2048):
        chunk = limbs[start : start + 2048]
        encoder = ElfOnChimpCompressor32() if method == "Elf" else ElfCompressor32()
        for value in chunk:
            encoder.add_value(float(value))
        encoder.close()
        body = encoder.get_bytes()
        out.extend(struct.pack("<II", len(chunk), len(body)))
        out.extend(body)
    return bytes(out)


def _unpack_elf(payload: bytes, count: int, method: str) -> list[int]:
    from .elf32_python import ElfDecompressor32, ElfOnChimpDecompressor32

    if len(payload) < 8 or payload[:4] != b"ELFI":
        raise ValueError("invalid Elf int64 wrapper")
    limb_count = _U32.unpack_from(payload, 4)[0]
    position = 8
    limbs: list[int] = []
    while len(limbs) < limb_count:
        if position + 8 > len(payload):
            raise ValueError("truncated Elf chunk header")
        chunk_count, size = struct.unpack_from("<II", payload, position)
        position += 8
        if position + size > len(payload):
            raise ValueError("truncated Elf chunk")
        body = payload[position : position + size]
        position += size
        decoder = ElfOnChimpDecompressor32(body) if method == "Elf" else ElfDecompressor32(body)
        decoded = decoder.decompress(chunk_count)
        chunk_limbs = [int(value) for value in decoded]
        if any(float(limb) != float(value) for limb, value in zip(chunk_limbs, decoded)):
            raise ValueError(f"{method} decoded a non-integral limb")
        limbs.extend(chunk_limbs)
    if position != len(payload) or len(limbs) != limb_count:
        raise ValueError("Elf payload boundary mismatch")
    return _limbs_to_i64(limbs, count, 16)


def _pack_buff_int64(values: Sequence[int]) -> bytes:
    values = [int(value) for value in values]
    base = min(values)
    base_bits = base & _MASK64
    offsets = [((value & _MASK64) - base_bits) & _MASK64 for value in values]
    width = max((offset.bit_length() for offset in offsets), default=0)
    byte_slices = (width + 7) // 8
    padding = byte_slices * 8 - width
    out = bytearray(_U32.pack(len(values)) + _U64.pack(base_bits) + bytes((width, 1)))
    for slice_index in range(byte_slices):
        shift = (byte_slices - 1 - slice_index) * 8
        for offset in offsets:
            aligned = offset if padding == 0 else (offset << padding) & _MASK64
            out.append((aligned >> shift) & 0xFF)
    return bytes(out)


def _unpack_buff_int64(payload: bytes, count: int) -> list[int]:
    if len(payload) < 14:
        raise ValueError("truncated BUFF int64 header")
    stored_count = _U32.unpack_from(payload, 0)[0]
    base = _U64.unpack_from(payload, 4)[0]
    width, packed = payload[12], payload[13]
    if stored_count != count or width > 64 or packed != 1:
        raise ValueError("invalid BUFF int64 header")
    byte_slices = (width + 7) // 8
    padding = byte_slices * 8 - width
    expected = 14 + byte_slices * count
    if len(payload) != expected:
        raise ValueError(f"BUFF payload length mismatch: {len(payload)} != {expected}")
    offsets = [0] * count
    position = 14
    for _ in range(byte_slices):
        for index in range(count):
            offsets[index] = (offsets[index] << 8) | payload[position]
            position += 1
    if padding:
        offsets = [offset >> padding for offset in offsets]
    out = []
    for offset in offsets:
        bits = (base + offset) & _MASK64
        out.append(bits - (1 << 64) if bits & (1 << 63) else bits)
    return out


def pack_specialist_int64(values: Sequence[int], method: str) -> bytes:
    values = [int(value) for value in values]
    if not values:
        return b""
    if method == "Chimp":
        from .tsfile_encoding.p03_i64 import decode_chimp_int64, encode_chimp_int64

        body = encode_chimp_int64(values)
        if decode_chimp_int64(body) == values:
            return b"C" + body
        return b"R" + _raw_i64(values)
    if method == "Pongo":
        return _pack_float64_limb_codec(values, method)
    if method == "BOS":
        from .iotdb_style_python import _encode_long_delta_bitpack

        return _encode_long_delta_bitpack(values)
    if method == "REGER":
        from .reger_codec import encode_int_column

        return encode_int_column(values)
    if method in {"FastPFor", "ALP", "LECO"}:
        from .int64_column_codec import fast_pfor_encode_int64

        return fast_pfor_encode_int64(values)
    if method == "RLE":
        from .tsfile_encoding.p02 import encode_rle_int64

        return encode_rle_int64(values)
    if method == "RLBE":
        from .tsfile_encoding.p04_i64 import encode_rlbe_int64

        return encode_rlbe_int64(values)
    if method == "Dict.":
        from .column_encodings import encode_dictionary_int

        return encode_dictionary_int(values)
    if method == "DBP":
        from .column_encodings import encode_delta_zigzag_varint_int

        return encode_delta_zigzag_varint_int(values)
    if method == "Camel" or method == "FPZIP":
        return _pack_float64_limb_codec(values, method)
    if method in {"Elf", "Elf+"}:
        return _pack_elf(values, method)
    if method == "FPC":
        from .research_extended_codecs import _fpc_compress_memory

        return _fpc_compress_memory(_raw_i64(values), 16)
    if method == "PFPC":
        from .research_extended_codecs import _pfpc_compress_memory

        raw = _raw_i64(values)
        return _pfpc_compress_memory(raw, 16, 1, max(1, min(len(raw), 1 << 20)))
    if method == "SPDP":
        from .research_extended_codecs import _spdp_roundtrip

        body, _, _, restored = _spdp_roundtrip(_raw_i64(values), 9, 8)
        if restored != _raw_i64(values):
            raise ValueError("SPDP encoder self-check failed")
        return body
    if method == "BUFF":
        return _pack_buff_int64(values)
    if method in {"FPCmp-R", "FPCmp-S"}:
        from .native_lz_family import compress_native_lz_family

        key = "fpcompress_dpratio" if method == "FPCmp-R" else "fpcompress_dpspeed"
        return compress_native_lz_family(_raw_i64(values), key)
    raise KeyError(f"unsupported specialist method {method}")


def unpack_specialist_int64(payload: bytes, count: int, method: str) -> list[int]:
    if count == 0:
        if payload:
            raise ValueError(f"non-empty {method} payload for empty input")
        return []
    if method == "Chimp":
        from .tsfile_encoding.p03_i64 import decode_chimp_int64

        if not payload:
            raise ValueError("missing Chimp safety mode")
        if payload[:1] == b"C":
            values = decode_chimp_int64(payload[1:])
        elif payload[:1] == b"R":
            values = _raw_to_i64(payload[1:], count)
        else:
            raise ValueError("invalid Chimp safety mode")
    elif method == "Pongo":
        values = _unpack_float64_limb_codec(payload, count, method)
    elif method == "BOS":
        from .iotdb_style_python import _decode_long_delta_bitpack

        values = _decode_long_delta_bitpack(payload)
    elif method == "REGER":
        from .reger_codec import decode_int_column

        values = decode_int_column(payload)
    elif method in {"FastPFor", "ALP", "LECO"}:
        from .int64_column_codec import fast_pfor_decode_int64

        values = fast_pfor_decode_int64(payload, count)
    elif method == "RLE":
        from .tsfile_encoding.p02 import decode_rle_int64

        values = decode_rle_int64(payload)
    elif method == "RLBE":
        from .tsfile_encoding.p04_i64 import decode_rlbe_int64

        values = decode_rlbe_int64(payload)
    elif method == "Dict.":
        from .column_encodings import decode_dictionary_int

        values = decode_dictionary_int(payload)
    elif method == "DBP":
        from .column_encodings import decode_delta_zigzag_varint_int

        values = decode_delta_zigzag_varint_int(payload)
    elif method in {"Camel", "FPZIP"}:
        values = _unpack_float64_limb_codec(payload, count, method)
    elif method in {"Elf", "Elf+"}:
        values = _unpack_elf(payload, count, method)
    elif method == "FPC":
        from .research_extended_codecs import _fpc_decompress_memory

        values = _raw_to_i64(_fpc_decompress_memory(payload), count)
    elif method == "PFPC":
        from .research_extended_codecs import _pfpc_decompress_memory

        values = _raw_to_i64(_pfpc_decompress_memory(payload), count)
    elif method == "SPDP":
        from .research_extended_codecs import _spdp_decompress_core

        raw = bytearray(count * 8)
        _spdp_decompress_core(9, len(payload), payload, raw, 8)
        values = _raw_to_i64(bytes(raw), count)
    elif method == "BUFF":
        values = _unpack_buff_int64(payload, count)
    elif method in {"FPCmp-R", "FPCmp-S"}:
        from .native_lz_family import decompress_native_lz_family

        key = "fpcompress_dpratio" if method == "FPCmp-R" else "fpcompress_dpspeed"
        values = _raw_to_i64(decompress_native_lz_family(payload, key, count * 8), count)
    else:
        raise KeyError(f"unsupported specialist method {method}")
    if len(values) != count:
        raise ValueError(f"{method} decoded count mismatch: {len(values)} != {count}")
    return [int(value) for value in values]


def _compress_general_body(data: bytes, key: str) -> bytes:
    if key == "ans":
        from .entropy_cpp import ans_encode

        return ans_encode(data)
    if key == "huffman":
        from .entropy_cpp import huffman_encode

        return huffman_encode(data)
    if key == "arithmetic":
        from .entropy_cpp import arithmetic_encode

        return arithmetic_encode(data)
    if key == "gz":
        return gzip.compress(data, compresslevel=6, mtime=0)
    if key == "brotli":
        import brotli

        return brotli.compress(data, quality=6)
    if key.startswith("zstd_level_"):
        import zstandard

        return zstandard.ZstdCompressor(level=int(key.rsplit("_", 1)[1])).compress(data)
    if key.startswith("xz_level_"):
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=int(key.rsplit("_", 1)[1]))
    if key == "lz4-framed":
        import lz4.frame

        return lz4.frame.compress(data)
    if key == "snappy-framed":
        import snappy

        return snappy.compress(data)
    if key.startswith("bzip2_"):
        return bz2.compress(data, compresslevel=int(key.rsplit("_", 1)[1]))
    if key.startswith("zlib_") and not key.startswith("zlib_ng_"):
        return zlib.compress(data, int(key.rsplit("_", 1)[1]))
    from .native_lz_family import compress_native_lz_family

    return compress_native_lz_family(data, key)


def _decompress_general_body(data: bytes, key: str, expected_size: int) -> bytes:
    if key == "ans":
        from .entropy_cpp import ans_decode

        return ans_decode(data)
    if key == "huffman":
        from .entropy_cpp import huffman_decode

        return huffman_decode(data)
    if key == "arithmetic":
        from .entropy_cpp import arithmetic_decode

        return arithmetic_decode(data)
    if key == "gz":
        return gzip.decompress(data)
    if key == "brotli":
        import brotli

        return brotli.decompress(data)
    if key.startswith("zstd_level_"):
        import zstandard

        return zstandard.ZstdDecompressor().decompress(data)
    if key.startswith("xz_level_"):
        return lzma.decompress(data, format=lzma.FORMAT_XZ)
    if key == "lz4-framed":
        import lz4.frame

        return lz4.frame.decompress(data)
    if key == "snappy-framed":
        import snappy

        return snappy.decompress(data)
    if key.startswith("bzip2_"):
        return bz2.decompress(data)
    if key.startswith("zlib_") and not key.startswith("zlib_ng_"):
        return zlib.decompress(data)
    from .native_lz_family import decompress_native_lz_family

    return decompress_native_lz_family(data, key, expected_size)


def compress_general(data: bytes, key: str) -> bytes:
    raw = bytes(data)
    return _U64.pack(len(raw)) + _compress_general_body(raw, key)


def decompress_general(payload: bytes, key: str) -> bytes:
    if len(payload) < 8:
        raise ValueError("truncated catalog byte-codec header")
    expected_size = _U64.unpack_from(payload, 0)[0]
    restored = _decompress_general_body(payload[8:], key, expected_size)
    if len(restored) != expected_size:
        raise ValueError(f"{key} decoded length mismatch: {len(restored)} != {expected_size}")
    return restored
