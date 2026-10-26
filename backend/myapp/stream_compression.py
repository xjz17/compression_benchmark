"""Framed stream codecs aligned with C++ ``native_streams`` / Java Commons Compress."""

from __future__ import annotations

import gzip
import struct

import lz4.frame
import snappy

_SNAPPY_STREAM_ID = b"sNaPpY"
_SNAPPY_MAX_UNCOMP = 65536


def compress_gz(data: bytes) -> bytes:
    # zlib default level (6), stable header — matches Java/C++ gzip benches.
    return gzip.compress(data, compresslevel=6, mtime=0)


def decompress_gz(data: bytes) -> bytes:
    return gzip.decompress(data)


def _append_u24(buf: bytearray, length: int) -> None:
    buf.extend(struct.pack("<I", length)[:3])


def _write_framed_chunk(buf: bytearray, chunk_type: int, payload: bytes) -> None:
    if len(payload) > 0xFFFFFF:
        raise ValueError("snappy chunk too large")
    buf.append(chunk_type)
    _append_u24(buf, len(payload))
    buf.extend(payload)


def compress_snappy_framed(data: bytes) -> bytes:
    out = bytearray()
    _write_framed_chunk(out, 0xFF, _SNAPPY_STREAM_ID)
    offset = 0
    while offset < len(data):
        n = min(_SNAPPY_MAX_UNCOMP, len(data) - offset)
        block = snappy.compress(data[offset : offset + n])
        _write_framed_chunk(out, 0x00, block)
        offset += n
    return bytes(out)


def _read_u24(buf: bytes, pos: int) -> tuple[int, int]:
    length = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16)
    return length, pos + 3


def decompress_snappy_framed(data: bytes) -> bytes:
    decoded = bytearray()
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError("snappy frame truncated")
        chunk_type = data[pos]
        length, pos = _read_u24(data, pos + 1)
        if pos + length > len(data):
            raise ValueError("snappy frame payload truncated")
        payload = data[pos : pos + length]
        pos += length
        if chunk_type == 0xFF:
            if payload != _SNAPPY_STREAM_ID:
                raise ValueError("snappy stream id bad")
        elif chunk_type == 0x00:
            decoded.extend(snappy.decompress(payload))
    return bytes(decoded)


def compress_lz4_framed(data: bytes) -> bytes:
    # Apache Commons FramedLZ4CompressorOutputStream.Parameters.DEFAULT
    return lz4.frame.compress(
        data,
        block_size=lz4.frame.BLOCKSIZE_MAX4MB,
        block_linked=False,
        compression_level=0,
        content_checksum=True,
    )


def decompress_lz4_framed(data: bytes) -> bytes:
    return lz4.frame.decompress(data)
