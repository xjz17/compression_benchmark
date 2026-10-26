"""Lossless storage compaction of C206's exposed UAR1 metadata only.

CCF1 stores one compressed metadata block, the original C206 header, and the
UNCHANGED codec payload. Expanding reproduces the exact old frame, including
JSON whitespace/key order. If the new whole frame is not smaller, keep C206.
This does not atomize codecs or rewrite metadata already inside a payload.
"""
from __future__ import annotations

import struct
import zlib


MAGIC = b'CCF1'
LEGACY_MAGIC = b'C206'
UAR_MAGIC = b'UAR1'
LEGACY_HEADER = struct.Struct('<4sBBQI')
# magic, original JSON bytes, compressed JSON bytes, payload bytes, legacy CRC
PREFIX = struct.Struct('<4sIIQI')
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 512 * 1024 * 1024
MAX_LAYERS = 10
WIRE_FORMAT = 'ccf1-metadata-zlib9-or-legacy-c206-v1'


def _legacy_parts(frame):
    if not LEGACY_HEADER.size <= len(frame) <= MAX_FRAME_BYTES:
        raise ValueError('C206 frame size is outside the supported bound')
    magic, kind, layers, count, checksum = LEGACY_HEADER.unpack_from(frame)
    if magic != LEGACY_MAGIC or kind not in (1, 2) or layers > MAX_LAYERS:
        raise ValueError('invalid C206 header')
    if count * 8 > MAX_FRAME_BYTES:
        raise ValueError('C206 original size exceeds the block bound')
    header = frame[:LEGACY_HEADER.size]
    body = frame[LEGACY_HEADER.size:]
    if layers == 0:
        if len(body) != count * 8 or zlib.crc32(body) != checksum:
            raise ValueError('invalid uncompressed C206 frame')
        return header, None, body
    if len(body) < 8 or body[:4] != UAR_MAGIC:
        raise ValueError('C206 encoded frame requires an exposed UAR1 header')
    meta_size = struct.unpack_from('<I', body, 4)[0]
    if meta_size > MAX_METADATA_BYTES or 8 + meta_size > len(body):
        raise ValueError('UAR1 metadata length exceeds available bytes or bound')
    return header, body[8:8 + meta_size], body[8 + meta_size:]


def compact_frame(frame: bytes) -> bytes:
    """Store the entire frame no larger than C206; codecs are not re-executed."""
    frame = bytes(frame)
    header, metadata, payload = _legacy_parts(frame)
    if metadata is None:
        return frame
    encoded = zlib.compress(metadata, level=9)
    total = PREFIX.size + len(header) + len(encoded) + len(payload)
    if total >= len(frame):
        return frame
    return (PREFIX.pack(MAGIC, len(metadata), len(encoded), len(payload),
                        zlib.crc32(frame)) + header + encoded + payload)


def expand_frame(wire: bytes) -> bytes:
    """Expand a bounded CCF1 or validate/pass through a legacy C206 frame.

    CRC detects accidental damage, not malicious authentication. Limits cover
    this metadata envelope; the underlying codecs retain their own limits.
    """
    wire = bytes(wire)
    if wire.startswith(LEGACY_MAGIC):
        _legacy_parts(wire)
        return wire
    if not PREFIX.size + LEGACY_HEADER.size <= len(wire) <= MAX_FRAME_BYTES:
        raise ValueError('compact frame size is outside the supported bound')
    magic, raw_size, stored_size, payload_size, checksum = PREFIX.unpack_from(wire)
    if magic != MAGIC:
        raise ValueError('unknown compact pipeline frame version')
    if raw_size > MAX_METADATA_BYTES:
        raise ValueError('metadata expansion exceeds the bound')
    offset = PREFIX.size + LEGACY_HEADER.size
    if stored_size > MAX_METADATA_BYTES or offset + stored_size + payload_size != len(wire):
        raise ValueError('compact frame extent mismatch')
    if LEGACY_HEADER.size + 8 + raw_size + payload_size > MAX_FRAME_BYTES:
        raise ValueError('expanded C206 frame exceeds the bound')
    header = wire[PREFIX.size:offset]
    encoded = wire[offset:offset + stored_size]
    # ponytail: standard zlib, no custom JSON codec or external dictionary.
    inflater = zlib.decompressobj()
    try:
        metadata = inflater.decompress(encoded, raw_size + 1)
    except zlib.error as exc:
        raise ValueError('invalid compressed metadata') from exc
    if (len(metadata) != raw_size or not inflater.eof or inflater.unused_data
            or inflater.unconsumed_tail):
        raise ValueError('metadata length, termination, or trailing stream mismatch')
    legacy = (header + UAR_MAGIC + struct.pack('<I', raw_size) + metadata
              + wire[offset + stored_size:])
    if zlib.crc32(legacy) != checksum:
        raise ValueError('compact pipeline checksum mismatch')
    _legacy_parts(legacy)
    return legacy


def frame_accounting(frame: bytes) -> dict:
    """Byte accounting only; no encoded payload content is emitted."""
    legacy = expand_frame(frame)
    header, metadata, payload = _legacy_parts(legacy)
    compressed = frame.startswith(MAGIC)
    return {
        'wire_magic': frame[:4].decode('ascii'),
        'stored_bytes': len(frame), 'legacy_bytes': len(legacy),
        'bytes_saved': len(legacy) - len(frame),
        'codec_payload_bytes': len(payload),
        'original_metadata_bytes': 0 if metadata is None else len(metadata),
        'stored_metadata_bytes': (PREFIX.unpack_from(frame)[2] if compressed
                                  else 0 if metadata is None else len(metadata)),
        'all_bytes_outside_codec_payload': len(frame) - len(payload),
        'legacy_header_bytes': len(header),
        'compact_prefix_bytes': PREFIX.size if compressed else 0,
        'legacy_uar_prefix_bytes': 0 if compressed or metadata is None else 8,
        'payload_changed': False,
    }
