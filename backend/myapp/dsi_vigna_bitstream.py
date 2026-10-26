"""
DSI / Sebastiano Vigna-style big-endian bit streams over ``bytes`` / ``bytearray``,
matching ``gr.aueb.delorean.chimp.{Input,Output}BitStream`` byte-array mode
(used by Chimp, Elf, Camel in this repo).
"""
from __future__ import annotations

from typing import Union

Buffer = Union[bytes, bytearray]


class OutputBitStreamArray:
    __slots__ = ("buffer", "written_bits", "current", "free", "pos", "avail")

    def __init__(self, buf: bytearray) -> None:
        self.buffer = buf
        self.written_bits = 0
        self.current = 0
        self.free = 8
        self.pos = 0
        self.avail = len(buf)

    @property
    def writtenBits(self) -> int:  # Java-style name for benches
        return self.written_bits

    def _write_byte(self, b: int) -> None:
        if self.avail == 0:
            raise RuntimeError("OutputBitStream buffer overflow")
        self.avail -= 1
        self.buffer[self.pos] = b & 0xFF
        self.pos += 1

    def _write_in_current(self, b: int, length: int) -> int:
        if length == 0:
            return 0
        self.free -= length
        self.current |= (b & ((1 << length) - 1)) << self.free
        if self.free == 0:
            self._write_byte(self.current)
            self.free = 8
            self.current = 0
        self.written_bits += length
        return length

    def write_bit(self, bit: int) -> int:
        return self._write_in_current(1 if bit else 0, 1)

    def write_int(self, x: int, length: int) -> int:
        x &= 0xFFFFFFFF
        if length == 0:
            return 0
        if length <= self.free:
            return self._write_in_current(x, length)

        i = length - self.free
        queue = i & 7

        if self.free != 0:
            self._write_in_current(x >> i, self.free)

        if queue != 0:
            i -= queue
            self._write_in_current(x & ((1 << queue) - 1), queue)
            x >>= queue

        if i == 32:
            self._write_byte((x >> 24) & 0xFF)
        if i > 23:
            self._write_byte((x >> 16) & 0xFF)
        if i > 15:
            self._write_byte((x >> 8) & 0xFF)
        if i > 7:
            self._write_byte(x & 0xFF)

        self.written_bits += i
        return length

    def write_long(self, x: int, length: int) -> int:
        """Match ``OutputBitStream.writeLong`` (lower ``length`` bits, MSB first)."""
        if length == 0:
            return 0
        if length < 64:
            x &= (1 << length) - 1
        else:
            x &= (1 << 64) - 1
        if length <= self.free:
            return self._write_in_current(int(x), length)

        i = length - self.free
        queue = i & 7

        if self.free != 0:
            self._write_in_current(int(x >> i), self.free)

        if queue != 0:
            i -= queue
            self._write_in_current(int(x & ((1 << queue) - 1)), queue)
            x >>= queue

        if i == 64:
            self._write_byte(int((x >> 56) & 0xFF))
        if i > 55:
            self._write_byte(int((x >> 48) & 0xFF))
        if i > 47:
            self._write_byte(int((x >> 40) & 0xFF))
        if i > 39:
            self._write_byte(int((x >> 32) & 0xFF))
        if i > 31:
            self._write_byte(int((x >> 24) & 0xFF))
        if i > 23:
            self._write_byte(int((x >> 16) & 0xFF))
        if i > 15:
            self._write_byte(int((x >> 8) & 0xFF))
        if i > 7:
            self._write_byte(int(x & 0xFF))

        self.written_bits += i
        return length

    def align(self) -> int:
        if self.free != 8:
            return self._write_in_current(0, self.free)
        return 0

    def flush(self) -> None:
        self.align()


class InputBitStreamBytes:
    __slots__ = ("buffer", "read_bits", "current", "fill", "pos", "avail", "position")

    def __init__(self, data: Buffer) -> None:
        self.buffer = data
        self.read_bits = 0
        self.current = 0
        self.fill = 0
        self.pos = 0
        self.avail = len(data)
        self.position = 0

    def read(self) -> int:
        if self.avail == 0:
            self.position += self.pos
            self.pos = 0
            self.avail = -1
        self.avail -= 1
        b = self.buffer[self.pos] & 0xFF
        self.pos += 1
        return b

    def refill(self) -> int:
        if self.avail > 1:
            self.avail -= 2
            b0 = self.buffer[self.pos] & 0xFF
            self.pos += 1
            b1 = self.buffer[self.pos] & 0xFF
            self.pos += 1
            self.current = ((self.current & 0xFFFFFFFF) << 16) | (b0 << 8) | b1
            self.current &= 0xFFFFFFFF
            self.fill += 16
            return self.fill
        self.current = ((self.current & 0xFFFFFFFF) << 8) | self.read()
        self.current &= 0xFFFFFFFF
        self.fill += 8
        self.current = ((self.current & 0xFFFFFFFF) << 8) | self.read()
        self.current &= 0xFFFFFFFF
        self.fill += 8
        return self.fill

    def read_from_current(self, length: int) -> int:
        if self.fill == 0:
            self.current = self.read()
            self.fill = 8
        self.fill -= length
        return (self.current >> self.fill) & ((1 << length) - 1)

    def read_bit(self) -> int:
        return self.read_from_current(1)

    def read_int(self, length: int) -> int:
        if self.fill < 16:
            self.refill()
        if length <= self.fill:
            return self.read_from_current(length)
        length -= self.fill
        x = self.read_from_current(self.fill)
        i = length >> 3
        while i > 0:
            i -= 1
            x = ((x << 8) | self.read()) & 0xFFFFFFFF
        self.read_bits += length & ~7
        length &= 7
        return (((x << length) & 0xFFFFFFFF) | self.read_from_current(length)) & 0xFFFFFFFF

    def read_long(self, length: int) -> int:
        if self.fill < 16:
            self.refill()
        if length <= self.fill:
            return self.read_from_current(length)
        length -= self.fill
        x = self.read_from_current(self.fill)
        i = length >> 3
        while i > 0:
            i -= 1
            x = (x << 8) | self.read()
        length &= 7
        return ((x << length) | self.read_from_current(length)) & ((1 << 64) - 1)
