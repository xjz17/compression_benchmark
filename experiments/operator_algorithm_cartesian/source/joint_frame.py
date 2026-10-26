#!/usr/bin/env python3
"""Complete JOA1 frame for one measured operator x algorithm block."""

from __future__ import annotations

import hashlib
import json
import struct


MAGIC = b"JOA1"
VERSION = 1
HEADER = struct.Struct("<4sB3xQQQ32s")
CHECKSUM_BYTES = 32


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def build_frame(binding: dict, carrier_metadata: bytes,
                algorithm_frame: bytes, source_column_sha256: str) -> bytes:
    if len(source_column_sha256) != 64:
        raise ValueError("source column SHA-256 must be lowercase hexadecimal")
    try:
        source_digest = bytes.fromhex(source_column_sha256)
    except ValueError as error:
        raise ValueError("source column SHA-256 is not hexadecimal") from error
    if len(source_digest) != 32 or source_column_sha256 != source_column_sha256.lower():
        raise ValueError("source column SHA-256 is not canonical")
    binding_bytes = _canonical_json(binding)
    header = HEADER.pack(
        MAGIC, VERSION, len(binding_bytes), len(carrier_metadata),
        len(algorithm_frame), source_digest,
    )
    body = header + binding_bytes + carrier_metadata + algorithm_frame
    return body + hashlib.sha256(body).digest()


def parse_frame(frame: bytes) -> dict:
    minimum = HEADER.size + CHECKSUM_BYTES
    if len(frame) < minimum:
        raise ValueError("JOA1 frame is truncated")
    body, checksum = frame[:-CHECKSUM_BYTES], frame[-CHECKSUM_BYTES:]
    if hashlib.sha256(body).digest() != checksum:
        raise ValueError("JOA1 checksum mismatch")
    magic, version, binding_length, metadata_length, algorithm_length, \
        source_digest = HEADER.unpack_from(body)
    if magic != MAGIC or version != VERSION:
        raise ValueError("invalid JOA1 header")
    expected = HEADER.size + binding_length + metadata_length + algorithm_length
    if expected != len(body):
        raise ValueError("JOA1 component lengths do not consume the frame")
    position = HEADER.size
    binding_bytes = body[position:position + binding_length]
    position += binding_length
    carrier_metadata = body[position:position + metadata_length]
    position += metadata_length
    algorithm_frame = body[position:position + algorithm_length]
    try:
        binding = json.loads(binding_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JOA1 binding is not canonical JSON") from error
    if _canonical_json(binding) != binding_bytes:
        raise ValueError("JOA1 binding JSON is not canonical")
    return {
        "binding": binding,
        "carrier_metadata": carrier_metadata,
        "algorithm_frame": algorithm_frame,
        "source_column_sha256": source_digest.hex(),
        "complete_frame_bytes": len(frame),
        "complete_frame_sha256": hashlib.sha256(frame).hexdigest(),
    }
