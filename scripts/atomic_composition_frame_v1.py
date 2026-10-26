#!/usr/bin/env python3
"""Self-contained ACF1 files for explicit primitive and codec compositions.

Body = varint(metadata_size) || compact_binary(metadata) || transformed_payload.
Every explicit terminal processes the WHOLE body; metadata has no hidden codec.
The versioned header records kind, raw size/SHA256, terminal IDs and body sizes.
All bytes, palettes, parameters, entropy headers and this envelope are charged.
No original input, external palette or experiment checkpoint is used by decode.
"""
from __future__ import annotations

import base64
import bz2
import hashlib
import lzma
import math
from pathlib import Path
import struct
import subprocess
import zlib

import atomic_transform_runtime_v1 as transforms

ROOT = Path(__file__).resolve().parents[1]
ENTROPY_CLI = ROOT / 'cpp/build/primitive_entropy_v1/primitive_entropy_v1'
MAGIC = b'ACF1'
VERSION = 1
MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_BODY_BYTES = 512 * 1024 * 1024
MAX_FRAME_BYTES = MAX_BODY_BYTES + 1024
MAX_METADATA_BYTES = MAX_RAW_BYTES
MAX_NODES = 1000000
MAX_DEPTH = 10
SUBPROCESS_TIMEOUT = 10
KINDS = {'int64': 0, 'float64': 1}
TERMINALS = {
    'range0': 1, 'range1': 2, 'sparse0': 3, 'sparse1': 4,
    'zlib6': 16, 'bzip2': 17, 'xz': 18, 'zstd': 19, 'brotli': 20,
}
PRIMITIVE_TERMINALS = frozenset(('range0', 'range1', 'sparse0', 'sparse1'))
ALGORITHM_TERMINALS = frozenset(TERMINALS) - PRIMITIVE_TERMINALS
TERMINAL_DETAILS = {
    'range0': 'standalone adaptive order-0 arithmetic primitive; PEV1 framing counted',
    'range1': 'standalone adaptive order-1 arithmetic primitive; PEV1 framing counted',
    'sparse0': 'standalone adaptive order-0 sparse-alphabet arithmetic with escape symbol; PEV1 framing counted',
    'sparse1': 'standalone adaptive previous-byte sparse-alphabet arithmetic with escape symbol; PEV1 framing counted',
    'zlib6': 'complete zlib algorithm, level 6; not a pure primitive',
    'bzip2': 'complete bzip2 algorithm, level 9; not a pure primitive',
    'xz': 'complete XZ/LZMA2 algorithm, preset 6 / CRC64; not a pure primitive',
    'zstd': 'complete Zstandard algorithm, level 9; not a pure primitive',
    'brotli': 'complete Brotli algorithm, quality 6; not a pure primitive',
}

# Frozen ACF1 metadata symbol table: changing order requires a new frame version.
# Entries are literal strings, not probabilistic dictionaries learned from input.
SYMBOLS = (
    'op', 'in_kind', 'in_count', 'in_nbytes', 'out_kind', 'out_count', 'out_nbytes',
    'u64', 'bytes', 'zigzag', 'delta', 'xor', 'for', 'gcd', 'dictionary', 'compact',
    'bitpack', 'shuffle', 'bittranspose', 'rle', 'sparsezero', 'float_scale',
    'lag', 'order', 'minimum', 'factor', 'palette_b64', 'width', 'bits', 'split',
    'max_entries', 'auto',
)
SYMBOL_IDS = {value: index + 32 for index, value in enumerate(SYMBOLS)}


def _uint(value, limit=(1 << 64) - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
        raise ValueError('integer is outside frame bounds')
    return value


def _varint(value):
    value = _uint(value)
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


class _Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, size):
        if size < 0 or size > len(self.data) - self.pos:
            raise ValueError('truncated frame or metadata')
        part = self.data[self.pos:self.pos + size]
        self.pos += size
        return bytes(part)

    def uint(self):
        value = 0
        for index in range(10):
            byte = self.take(1)[0]
            value |= (byte & 127) << (index * 7)
            if not byte & 128:
                if value > (1 << 64) - 1 or (index and byte == 0):
                    raise ValueError('oversized or noncanonical varint')
                return value
        raise ValueError('unterminated varint')


def serialize_metadata(value):
    """Lossless compact generic values; palette base64 text is stored as a blob."""
    output = bytearray()
    nodes = 0

    def emit(item, depth=0, key=None):
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > MAX_NODES:
            raise ValueError('metadata nesting/element limit exceeded')
        if key == 'palette_b64':
            if not isinstance(item, str):
                raise ValueError('palette_b64 must be canonical base64 text')
            try:
                blob = base64.b64decode(item, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError('invalid base64 palette') from exc
            if base64.b64encode(blob).decode('ascii') != item:
                raise ValueError('noncanonical base64 palette')
            output.append(10)
            output.extend(_varint(len(blob)))
            output.extend(blob)
        elif item is None:
            output.append(0)
        elif isinstance(item, bool):
            output.append(2 if item else 1)
        elif isinstance(item, int):
            output.append(3 if item >= 0 else 4)
            output.extend(_varint(item if item >= 0 else -item - 1))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError('nonfinite metadata float')
            output.append(5)
            output.extend(struct.pack('<d', item))
        elif isinstance(item, str):
            if item in SYMBOL_IDS:
                output.append(SYMBOL_IDS[item])
            else:
                encoded = item.encode('utf-8')
                output.append(6)
                output.extend(_varint(len(encoded)))
                output.extend(encoded)
        elif isinstance(item, bytes):
            output.append(7)
            output.extend(_varint(len(item)))
            output.extend(item)
        elif isinstance(item, (list, tuple)):
            output.append(8)
            output.extend(_varint(len(item)))
            for child in item:
                emit(child, depth + 1)
        elif isinstance(item, dict):
            if not all(isinstance(k, str) for k in item):
                raise ValueError('metadata map keys must be strings')
            output.append(9)
            output.extend(_varint(len(item)))
            for child_key in sorted(item):
                emit(child_key, depth + 1)
                emit(item[child_key], depth + 1, child_key)
        else:
            raise ValueError('unsupported metadata type')
        if len(output) > MAX_METADATA_BYTES:
            raise ValueError('metadata exceeds safety size limit')

    emit(value)
    return bytes(output)


def deserialize_metadata(encoded):
    if not isinstance(encoded, bytes) or len(encoded) > MAX_METADATA_BYTES:
        raise ValueError('invalid metadata buffer')
    reader = _Reader(encoded)
    nodes = 0

    def parse(depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > MAX_NODES:
            raise ValueError('metadata nesting/element limit exceeded')
        tag = reader.take(1)[0]
        if 32 <= tag < 32 + len(SYMBOLS):
            return SYMBOLS[tag - 32]
        if tag in (0, 1, 2):
            return (None, False, True)[tag]
        if tag in (3, 4):
            value = reader.uint()
            return value if tag == 3 else -value - 1
        if tag == 5:
            value, = struct.unpack('<d', reader.take(8))
            if not math.isfinite(value):
                raise ValueError('nonfinite metadata float')
            return value
        if tag in (6, 7, 10):
            length = reader.uint()
            if length > MAX_METADATA_BYTES:
                raise ValueError('metadata item exceeds safety size limit')
            value = reader.take(length)
            if tag == 6:
                return value.decode('utf-8')
            if tag == 10:
                return base64.b64encode(value).decode('ascii')
            return value
        if tag in (8, 9):
            count = reader.uint()
            if count > MAX_NODES:
                raise ValueError('metadata container exceeds safety size limit')
            if tag == 8:
                return [parse(depth + 1) for _ in range(count)]
            result = {}
            for _ in range(count):
                key = parse(depth + 1)
                if not isinstance(key, str) or key in result:
                    raise ValueError('duplicate/nonstring metadata map key')
                result[key] = parse(depth + 1)
            return result
        raise ValueError('unknown metadata tag')

    result = parse()
    if reader.pos != len(encoded):
        raise ValueError('metadata has trailing bytes')
    return result


def _check_raw(raw, kind):
    if not isinstance(raw, bytes) or len(raw) > MAX_RAW_BYTES or len(raw) % 8:
        raise ValueError('raw input must be 8-byte aligned bytes, at most 256 MiB')
    if kind not in KINDS:
        raise ValueError('source kind must be int64 or float64')


def _check_terminals(terminals):
    if not isinstance(terminals, (list, tuple)) or len(terminals) > 2:
        raise ValueError('at most two explicit complete-algorithm terminals are allowed')
    if any(not isinstance(t, str) or t not in TERMINALS for t in terminals):
        raise ValueError('unknown terminal codec')
    if len(terminals) == 2 and any(t in PRIMITIVE_TERMINALS for t in terminals):
        raise ValueError('pure range/sparse terminal must be the only terminal')


def _validate_metadata(metadata, raw_length, payload_length):
    if not isinstance(metadata, list) or len(metadata) > MAX_DEPTH:
        raise ValueError('invalid metadata operator sequence')
    kind, count, length = 'u64', raw_length // 8, raw_length
    for record in metadata:
        if not isinstance(record, dict) or record.get('op') not in transforms.SUPPORTED_OPERATORS:
            raise ValueError('unknown primitive in inverse metadata')
        op = record['op']
        if (record.get('in_kind'), record.get('in_count'), record.get('in_nbytes')) != (kind, count, length):
            raise ValueError('disconnected metadata input state')
        if op not in ('shuffle', 'bittranspose') and kind != 'u64':
            raise ValueError('numeric operator after byte-producing operator')
        allowed = {'op', 'in_kind', 'in_count', 'in_nbytes', 'out_kind', 'out_count', 'out_nbytes'} | set(transforms.INVERSE_FIELDS[op])
        if set(record) != allowed:
            raise ValueError('missing or unexpected inverse metadata fields')
        kind = record.get('out_kind')
        if kind not in ('u64', 'bytes'):
            raise ValueError('invalid output state kind')
        count = _uint(record.get('out_count'), MAX_BODY_BYTES)
        length = _uint(record.get('out_nbytes'), MAX_BODY_BYTES)
        if count * (8 if kind == 'u64' else 1) != length:
            raise ValueError('metadata typed length mismatch')
        expected_kind = 'bytes' if op in ('compact', 'bitpack', 'shuffle', 'bittranspose') else 'u64'
        if kind != expected_kind:
            raise ValueError('operator output kind mismatch')
    if length != payload_length:
        raise ValueError('metadata final payload length mismatch')


def _validate_prepared(raw, specs, payload, metadata, terminals):
    if not isinstance(specs, (list, tuple)) or len(specs) + len(terminals) > MAX_DEPTH:
        raise ValueError('specs plus terminals exceed maximum depth 10')
    transforms.validate_specs(specs)
    if not isinstance(payload, bytes) or len(payload) > MAX_BODY_BYTES:
        raise ValueError('invalid transformed payload')
    _validate_metadata(metadata, len(raw), len(payload))
    if len(specs) != len(metadata):
        raise ValueError('specs and metadata operator counts differ')
    for spec, record in zip(specs, metadata):
        op = spec['op']
        if op != record['op']:
            raise ValueError('spec and metadata operator differ')
        requested = {}
        if op == 'delta':
            requested = {'lag': spec.get('lag', 1), 'order': spec.get('order', 1)}
        elif op == 'xor':
            requested = {'lag': spec.get('lag', 1)}
        elif op in ('shuffle', 'bittranspose'):
            requested = {'width': spec.get('width', 8)}
        elif op == 'float_scale':
            requested = {'factor': spec.get('factor', 1)}
        elif op in ('compact', 'bitpack'):
            key = 'width' if op == 'compact' else 'bits'
            if spec.get(key, 'auto') != 'auto':
                requested[key] = spec[key]
        elif op == 'dictionary':
            try:
                palette = base64.b64decode(record['palette_b64'], validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError('invalid prepared dictionary palette') from exc
            if len(palette) % 8 or len(palette) // 8 > spec.get('max_entries', 4096):
                raise ValueError('prepared dictionary exceeds requested cardinality')
        if any(record.get(k) != value for k, value in requested.items()):
            raise ValueError('prepared metadata parameters do not match specs')


def make_body(payload, metadata):
    """Join the complete compact metadata and payload, without applying a codec."""
    if not isinstance(payload, bytes) or len(payload) > MAX_BODY_BYTES:
        raise ValueError('invalid body payload')
    encoded = serialize_metadata(metadata)
    body = _varint(len(encoded)) + encoded + payload
    if len(body) > MAX_BODY_BYTES:
        raise ValueError('uncompressed body exceeds safety size limit')
    return body


def _primitive(name, data, reverse=False):
    if not ENTROPY_CLI.is_file():
        raise RuntimeError('missing explicit primitive entropy CLI: ' + str(ENTROPY_CLI))
    result = subprocess.run([str(ENTROPY_CLI), name + ('-decode' if reverse else '-encode')],
                            input=data, capture_output=True, timeout=SUBPROCESS_TIMEOUT)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace').strip() or 'primitive codec failed')
    return result.stdout


def _compress(name, data):
    if name in PRIMITIVE_TERMINALS:
        result = _primitive(name, data)
    elif name == 'zlib6':
        result = zlib.compress(data, 6)
    elif name == 'bzip2':
        result = bz2.compress(data, compresslevel=9)
    elif name == 'xz':
        result = lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)
    elif name == 'zstd':
        import zstandard
        result = zstandard.ZstdCompressor(level=9).compress(data)
    elif name == 'brotli':
        import brotli
        result = brotli.compress(data, quality=6)
    else:
        raise ValueError('unknown terminal codec')
    if len(result) > MAX_BODY_BYTES:
        raise ValueError('encoded body exceeds safety size limit')
    return result


def _decompress(name, data, limit=MAX_BODY_BYTES):
    if name in PRIMITIVE_TERMINALS:
        result = _primitive(name, data, reverse=True)
    elif name == 'zlib6':
        decoder = zlib.decompressobj()
        result = decoder.decompress(data, limit + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('zlib body is truncated, trailing, or too large')
    elif name in ('bzip2', 'xz'):
        decoder = bz2.BZ2Decompressor() if name == 'bzip2' else lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=MAX_BODY_BYTES)
        result = decoder.decompress(data, max_length=limit + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError(name + ' body is truncated, trailing, or too large')
    elif name == 'zstd':
        import zstandard
        content_size = zstandard.frame_content_size(data)
        if content_size not in (zstandard.CONTENTSIZE_UNKNOWN, zstandard.CONTENTSIZE_ERROR) and content_size > limit:
            raise ValueError('zstd declared decoded body is too large')
        result = zstandard.ZstdDecompressor().decompress(data, max_output_size=limit + 1, allow_extra_data=False)
    elif name == 'brotli':
        import brotli
        decoder = brotli.Decompressor()
        pieces, size = [], 0
        for start in range(0, len(data), 16384):
            piece = decoder.process(data[start:start + 16384])
            size += len(piece)
            if size > limit:
                raise ValueError('brotli decoded body exceeds safety limit')
            pieces.append(piece)
        if not decoder.is_finished():
            raise ValueError('brotli body is truncated')
        result = b''.join(pieces)
    else:
        raise ValueError('unknown terminal codec')
    if len(result) > limit:
        raise ValueError('decoded body exceeds safety limit')
    return result


def encode_prepared(raw, kind, specs, terminals, payload, metadata):
    """Reuse a prepared transform; validate its explicit specs/metadata contract.

    This avoids recomputing transforms across terminal candidates. It is not a
    substitute for independent final encode_frame/decode_frame bytewise replay.
    """
    _check_raw(raw, kind)
    _check_terminals(terminals)
    _validate_prepared(raw, specs, payload, metadata, terminals)
    body = make_body(payload, metadata)
    encoded = body
    for name in terminals:
        encoded = _compress(name, encoded)
    header = (MAGIC + bytes((KINDS[kind],)) + _varint(len(raw)) + hashlib.sha256(raw).digest()
              + bytes((len(terminals),)) + bytes(TERMINALS[name] for name in terminals)
              + _varint(len(body)) + _varint(len(encoded)))
    checksum = zlib.crc32(encoded, zlib.crc32(header))
    return header + struct.pack('<I', checksum) + encoded


def encode_frame(raw, kind, specs, terminals):
    """Encode ONLY the supplied primitive specs and terminal sequence."""
    _check_raw(raw, kind)
    _check_terminals(terminals)
    if not isinstance(specs, (list, tuple)) or len(specs) + len(terminals) > MAX_DEPTH:
        raise ValueError('specs plus terminals exceed maximum depth 10')
    payload, metadata = transforms.apply_ops(raw, specs)
    return encode_prepared(raw, kind, specs, terminals, payload, metadata)


def _read_frame(frame):
    if not isinstance(frame, bytes) or len(frame) > MAX_FRAME_BYTES:
        raise ValueError('invalid or oversized frame buffer')
    reader = _Reader(frame)
    if reader.take(4) != MAGIC:
        raise ValueError('unknown frame magic/version')
    kind_id = reader.take(1)[0]
    if kind_id not in KINDS.values():
        raise ValueError('unknown source kind')
    raw_size = reader.uint()
    if raw_size > MAX_RAW_BYTES or raw_size % 8:
        raise ValueError('invalid raw byte length')
    raw_sha = reader.take(32)
    terminal_count = reader.take(1)[0]
    if terminal_count > 2:
        raise ValueError('too many terminal codecs')
    ids = reader.take(terminal_count)
    names = {code: name for name, code in TERMINALS.items()}
    try:
        terminals = [names[code] for code in ids]
    except KeyError as exc:
        raise ValueError('unknown terminal ID') from exc
    _check_terminals(terminals)
    body_size, encoded_size = reader.uint(), reader.uint()
    if body_size > MAX_BODY_BYTES or encoded_size > MAX_BODY_BYTES:
        raise ValueError('declared body length exceeds safety bound')
    checksum_prefix = reader.pos
    checksum, = struct.unpack('<I', reader.take(4))
    header_size = reader.pos
    encoded = reader.take(encoded_size)
    if reader.pos != len(frame):
        raise ValueError('trailing bytes after frame')
    if zlib.crc32(encoded, zlib.crc32(frame[:checksum_prefix])) != checksum:
        raise ValueError('frame checksum mismatch')
    body = encoded
    for index, name in enumerate(reversed(terminals)):
        # Only the final inverse terminal has the declared uncompressed size.
        limit = body_size if index == len(terminals) - 1 else MAX_BODY_BYTES
        body = _decompress(name, body, limit)
    if len(body) != body_size:
        raise ValueError('uncompressed body length mismatch')
    body_reader = _Reader(body)
    metadata_size = body_reader.uint()
    prefix_size = body_reader.pos
    if metadata_size > MAX_METADATA_BYTES:
        raise ValueError('declared metadata length exceeds safety bound')
    metadata = deserialize_metadata(body_reader.take(metadata_size))
    payload = body_reader.take(len(body) - body_reader.pos)
    _validate_metadata(metadata, raw_size, len(payload))
    if len(metadata) + len(terminals) > MAX_DEPTH:
        raise ValueError('decoded chain exceeds maximum depth 10')
    kind = next(name for name, code in KINDS.items() if code == kind_id)
    has_algorithm = bool(set(terminals) & ALGORITHM_TERMINALS)
    group = ('mixed_primitives_and_algorithms' if metadata else 'complete_algorithms_only') if has_algorithm else 'pure_primitives'
    return {
        'format': 'ACF1', 'version': VERSION, 'kind': kind, 'raw_bytes': raw_size,
        'raw_sha256': raw_sha.hex(), 'frame_bytes': len(frame), 'header_bytes': header_size,
        'encoded_body_bytes': encoded_size, 'uncompressed_body_bytes': body_size,
        'metadata_bytes': metadata_size, 'metadata_length_prefix_bytes': prefix_size,
        'payload_bytes': len(payload), 'metadata': metadata,
        'params': [{key: value for key, value in record.items() if key not in ('in_kind', 'in_count', 'in_nbytes', 'out_kind', 'out_count', 'out_nbytes')} for record in metadata],
        'terminals': terminals, 'terminal_ids': list(ids),
        'terminal_details': [TERMINAL_DETAILS[name] for name in terminals],
        'registered_depth': len(metadata) + len(terminals), 'comparison_group': group,
        'checksum_verified': True, 'metadata_compressed_separately': False,
    }, payload


def decode_frame(frame):
    info, payload = _read_frame(frame)
    raw = transforms.reverse_ops(payload, info['metadata'])
    _check_raw(raw, info['kind'])
    if len(raw) != info['raw_bytes'] or hashlib.sha256(raw).hexdigest() != info['raw_sha256']:
        raise ValueError('reconstructed source length/SHA256 mismatch')
    return raw


def inspect_frame(frame):
    """Inspect charged sizes and all inverse parameters; checks frame CRC.

    Does not claim source SHA256 verification: call decode_frame for that check.
    """
    info, _ = _read_frame(frame)
    return info
