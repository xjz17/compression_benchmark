#!/usr/bin/env python3
"""Standalone DBF1 encode/decode/benchmark and golden-vector utility.

Unified cross-runtime interface::

    dbp_float64_python_cli.py encode <maxPoint> < input.f64le > frame.dbf1
    dbp_float64_python_cli.py encode32 <maxPoint> < input.f32le > frame.dbf2
    dbp_float64_python_cli.py decode < frame.dbf1 > output.f64le
    dbp_float64_python_cli.py bench <maxPoint> [input.f64le ...]
    dbp_float64_python_cli.py bench-manifest --manifest blocks.tsv \
        --aggregates aggregates.jsonl --output results.jsonl --repeats 1

``bench`` reads stdin when no input files are named.  It streams arbitrary-size
little-endian float64 input and encodes blocks of at most 10,000 values without
loading a complete source file.  ``encode`` accepts exactly one block.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import BinaryIO, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.myapp.dbp_float64_codec import (  # noqa: E402
    HEADER_SIZE,
    MAGIC,
    MAGIC32,
    MAX_VALUES,
    MODE_DECIMAL,
    MODE_IEEE_BITS,
    decode,
    decode32,
    encode,
    encode32,
)


DEFAULT_GOLDEN_DIR = ROOT / "benchmark" / "reference" / "golden" / "dbp_float64"
CONTRACT_PATH = (
    ROOT / "analysis_outputs" / "dbp_float_e2e_20261009" / "WIRE_CONTRACT.md"
)
_HEADER = struct.Struct("<4sBBI")
_U64 = struct.Struct("<Q")
_BLOCK_BYTES = MAX_VALUES * 8
_MANIFEST_FIELDS = (
    "aggregate_id",
    "width",
    "max_point",
    "values_path",
    "values_offset",
    "values_length",
    "values_sha256",
    "missing_path",
    "missing_offset",
    "missing_length",
    "missing_sha256",
    "expected_sequence_sha256",
    "row_count",
    "valid_count",
    "missing_count",
)


def _unpack_float64(raw: bytes) -> list[float]:
    if len(raw) % 8:
        raise ValueError("raw float64 input length must be divisible by 8")
    if not raw:
        return []
    return list(struct.unpack(f"<{len(raw) // 8}d", raw))


def _pack_float64(values: list[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}d", *values)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _frame_metadata(frame: bytes) -> dict[str, int | str]:
    magic, mode, scale, count = _HEADER.unpack_from(frame, 0)
    return {
        "magic": magic.decode("ascii"),
        "mode": mode,
        "scale": scale,
        "count": count,
        "header_bytes": HEADER_SIZE,
        "frame_bytes": len(frame),
    }


def _require_bitwise_round_trip(raw: bytes, frame: bytes) -> bytes:
    restored = _pack_float64(decode(frame))
    if restored != raw:
        raise RuntimeError("DBF1 bitwise round-trip check failed")
    return restored


def _require_bitwise_round_trip32(raw: bytes, frame: bytes) -> bytes:
    restored = decode32(frame)
    if restored != raw:
        raise RuntimeError("DBF2 bitwise round-trip check failed")
    return restored


def _stream_blocks(stream: BinaryIO) -> Iterator[bytes]:
    """Yield full DBF1-size blocks while tolerating short pipe reads."""

    pending = bytearray()
    while True:
        chunk = stream.read(_BLOCK_BYTES - len(pending))
        if not chunk:
            break
        pending.extend(chunk)
        if len(pending) == _BLOCK_BYTES:
            yield bytes(pending)
            pending.clear()
    if len(pending) % 8:
        raise ValueError("raw float64 input length must be divisible by 8")
    if pending:
        yield bytes(pending)


def _cmd_encode(args: argparse.Namespace) -> int:
    raw = sys.stdin.buffer.read()
    values = _unpack_float64(raw)
    if len(values) > MAX_VALUES:
        raise ValueError(f"encode accepts at most {MAX_VALUES} float64 values")
    frame = encode(values, max_point=args.max_point)
    _require_bitwise_round_trip(raw, frame)
    sys.stdout.buffer.write(frame)
    return 0


def _cmd_encode32(args: argparse.Namespace) -> int:
    raw = sys.stdin.buffer.read()
    if len(raw) // 4 > MAX_VALUES:
        raise ValueError(f"encode32 accepts at most {MAX_VALUES} float32 values")
    frame = encode32(raw, max_point=args.max_point)
    _require_bitwise_round_trip32(raw, frame)
    sys.stdout.buffer.write(frame)
    return 0


def _cmd_decode(_args: argparse.Namespace) -> int:
    frame = sys.stdin.buffer.read()
    if frame[:4] == MAGIC:
        restored = _pack_float64(decode(frame))
    elif frame[:4] == MAGIC32:
        restored = decode32(frame)
    else:
        raise ValueError("unsupported DBF magic")
    sys.stdout.buffer.write(restored)
    return 0


def _bench_stream(stream: BinaryIO, max_point: int, totals: dict[str, int]) -> None:
    for raw in _stream_blocks(stream):
        values = _unpack_float64(raw)
        encode_started = time.perf_counter_ns()
        frame = encode(values, max_point=max_point)
        encode_finished = time.perf_counter_ns()
        restored = decode(frame)
        decode_finished = time.perf_counter_ns()

        if _pack_float64(restored) != raw:
            raise RuntimeError("DBF1 bitwise round-trip check failed")
        mode = frame[4]
        if mode == MODE_DECIMAL:
            totals["scaled_blocks"] += 1
        elif mode == MODE_IEEE_BITS:
            totals["raw_blocks"] += 1
        else:
            raise RuntimeError(f"DBF1 encoder produced invalid mode {mode}")

        count = len(values)
        totals["valid_count"] += count
        totals["original_bytes"] += len(raw)
        totals["compressed_bytes"] += len(frame)
        totals["encode_ns"] += encode_finished - encode_started
        totals["decode_ns"] += decode_finished - encode_finished
        totals["verified_values"] += count


def _cmd_bench(args: argparse.Namespace) -> int:
    totals = {
        "valid_count": 0,
        "original_bytes": 0,
        "compressed_bytes": 0,
        "encode_ns": 0,
        "decode_ns": 0,
        "scaled_blocks": 0,
        "raw_blocks": 0,
        "verified_values": 0,
    }
    if args.inputs:
        for source in args.inputs:
            with Path(source).open("rb") as stream:
                _bench_stream(stream, args.max_point, totals)
    else:
        _bench_stream(sys.stdin.buffer, args.max_point, totals)

    report = {
        "codec": "DBF1",
        "max_point": args.max_point,
        "clock": "time.perf_counter_ns",
        "block_value_limit": MAX_VALUES,
        "source_count": len(args.inputs) if args.inputs else 1,
        **totals,
    }
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0


@dataclass(frozen=True)
class _ManifestBlock:
    aggregate_id: str
    width: int
    max_point: int
    values_path: Path
    values_offset: int
    values_length: int
    values_sha256: str
    missing_path: Path
    missing_offset: int
    missing_length: int
    missing_sha256: str
    expected_sequence_sha256: str
    row_count: int
    valid_count: int
    missing_count: int
    line_number: int


class _SegmentReader:
    def __init__(self, limit: int = 32) -> None:
        self.limit = limit
        self._handles: OrderedDict[Path, BinaryIO] = OrderedDict()

    def read(self, path: Path, offset: int, length: int) -> bytes:
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = path.open("rb")
        self._handles[path] = handle
        while len(self._handles) > self.limit:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        size = os.fstat(handle.fileno()).st_size
        if offset + length > size:
            raise ValueError(
                f"segment exceeds file: {path} offset={offset} length={length} size={size}"
            )
        handle.seek(offset)
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError(f"short segment read from {path}")
        return payload

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "_SegmentReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _parse_nonnegative(text: str, field: str, line_number: int) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"manifest line {line_number}: invalid {field}") from exc
    if value < 0:
        raise ValueError(f"manifest line {line_number}: negative {field}")
    return value


def _parse_sha256(text: str, field: str, line_number: int) -> str:
    value = text.lower()
    if len(value) != 64:
        raise ValueError(f"manifest line {line_number}: invalid {field}")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"manifest line {line_number}: invalid {field}") from exc
    return value


def _manifest_path(text: str, manifest_dir: Path, line_number: int, field: str) -> Path:
    if not text:
        raise ValueError(f"manifest line {line_number}: empty {field}")
    path = Path(text)
    return path if path.is_absolute() else manifest_dir / path


def _iter_manifest(path: Path) -> Iterator[_ManifestBlock]:
    """Stream validated block metadata without retaining the manifest in memory."""

    manifest_dir = path.resolve().parent
    previous_aggregate = ""
    aggregate_shape: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(
                    f"manifest line {line_number}: blank rows are forbidden"
                )
            fields = line.split("\t")
            if len(fields) != len(_MANIFEST_FIELDS):
                raise ValueError(
                    f"manifest line {line_number}: expected {len(_MANIFEST_FIELDS)} fields, "
                    f"got {len(fields)}"
                )
            row = dict(zip(_MANIFEST_FIELDS, fields))
            aggregate_id = row["aggregate_id"]
            if not aggregate_id:
                raise ValueError(f"manifest line {line_number}: empty aggregate_id")
            if aggregate_id < previous_aggregate:
                raise ValueError("manifest aggregate_id order is not canonical")
            previous_aggregate = aggregate_id

            width = _parse_nonnegative(row["width"], "width", line_number)
            if width not in (4, 8):
                raise ValueError(f"manifest line {line_number}: width must be 4 or 8")
            try:
                max_point = int(row["max_point"])
            except ValueError as exc:
                raise ValueError(
                    f"manifest line {line_number}: invalid max_point"
                ) from exc
            shape = aggregate_shape.setdefault(aggregate_id, (width, max_point))
            if shape != (width, max_point):
                raise ValueError(
                    f"manifest line {line_number}: aggregate width/max_point changed"
                )

            values_length = _parse_nonnegative(
                row["values_length"], "values_length", line_number
            )
            missing_length = _parse_nonnegative(
                row["missing_length"], "missing_length", line_number
            )
            row_count = _parse_nonnegative(row["row_count"], "row_count", line_number)
            valid_count = _parse_nonnegative(
                row["valid_count"], "valid_count", line_number
            )
            missing_count = _parse_nonnegative(
                row["missing_count"], "missing_count", line_number
            )
            if row_count > MAX_VALUES:
                raise ValueError(
                    f"manifest line {line_number}: row_count exceeds {MAX_VALUES}"
                )
            if valid_count + missing_count != row_count:
                raise ValueError(
                    f"manifest line {line_number}: valid_count + missing_count != row_count"
                )
            if values_length != valid_count * width:
                raise ValueError(
                    f"manifest line {line_number}: values_length != valid_count * width"
                )
            expected_missing_length = (row_count + 7) // 8
            if missing_length != expected_missing_length:
                raise ValueError(
                    f"manifest line {line_number}: missing_length != ceil(row_count / 8)"
                )

            yield _ManifestBlock(
                aggregate_id=aggregate_id,
                width=width,
                max_point=max_point,
                values_path=_manifest_path(
                    row["values_path"], manifest_dir, line_number, "values_path"
                ),
                values_offset=_parse_nonnegative(
                    row["values_offset"], "values_offset", line_number
                ),
                values_length=values_length,
                values_sha256=_parse_sha256(
                    row["values_sha256"], "values_sha256", line_number
                ),
                missing_path=_manifest_path(
                    row["missing_path"], manifest_dir, line_number, "missing_path"
                ),
                missing_offset=_parse_nonnegative(
                    row["missing_offset"], "missing_offset", line_number
                ),
                missing_length=missing_length,
                missing_sha256=_parse_sha256(
                    row["missing_sha256"], "missing_sha256", line_number
                ),
                expected_sequence_sha256=_parse_sha256(
                    row["expected_sequence_sha256"],
                    "expected_sequence_sha256",
                    line_number,
                ),
                row_count=row_count,
                valid_count=valid_count,
                missing_count=missing_count,
                line_number=line_number,
            )


def _load_verified_block(
    block: _ManifestBlock, reader: _SegmentReader
) -> tuple[bytes, bytes]:
    values = reader.read(block.values_path, block.values_offset, block.values_length)
    missing = reader.read(
        block.missing_path, block.missing_offset, block.missing_length
    )
    if _sha256(values) != block.values_sha256:
        raise ValueError(f"manifest line {block.line_number}: values SHA mismatch")
    if _sha256(missing) != block.missing_sha256:
        raise ValueError(f"manifest line {block.line_number}: missing SHA mismatch")
    if sum(byte.bit_count() for byte in missing) != block.missing_count:
        raise ValueError(f"manifest line {block.line_number}: bitmap popcount mismatch")
    remainder = block.row_count % 8
    if remainder and missing and missing[-1] & ~((1 << remainder) - 1):
        raise ValueError(
            f"manifest line {block.line_number}: bitmap unused high bits are nonzero"
        )
    return values, missing


def _run_manifest_codec_block(
    block: _ManifestBlock, raw: bytes
) -> tuple[bytes, int, int, int]:
    if block.width == 8:
        values = _unpack_float64(raw)
        encode_started = time.perf_counter_ns()
        frame = encode(values, max_point=block.max_point)
        encode_finished = time.perf_counter_ns()
        restored = decode(frame)
        decode_finished = time.perf_counter_ns()
        restored_raw = _pack_float64(restored)
    else:
        encode_started = time.perf_counter_ns()
        frame = encode32(raw, max_point=block.max_point)
        encode_finished = time.perf_counter_ns()
        restored_raw = decode32(frame)
        decode_finished = time.perf_counter_ns()
    if restored_raw != raw:
        raise RuntimeError(
            f"manifest line {block.line_number}: DBF bitwise round-trip failure"
        )
    return (
        frame,
        encode_finished - encode_started,
        decode_finished - encode_finished,
        frame[4],
    )


def _new_aggregate_result(block: _ManifestBlock) -> dict[str, int | str]:
    return {
        "status": "n_a",
        "aggregate_id": block.aggregate_id,
        "width": block.width,
        "blocks": 0,
        "rows": 0,
        "missing_count": 0,
        "validity_bytes": 0,
        "header_bytes": 0,
        "valid_count": 0,
        "original_bytes": 0,
        "compressed_bytes": 0,
        "encode_ns": 0,
        "decode_ns": 0,
        "scaled_blocks": 0,
        "raw_blocks": 0,
        "verified_values": 0,
        "frame_sequence_sha256": "",
    }


def _write_jsonl_atomic(path: Path, rows: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cmd_bench_manifest(args: argparse.Namespace) -> int:
    if args.repeats != 1:
        raise ValueError("bench-manifest currently requires --repeats 1")
    manifest_path = Path(args.manifest)
    aggregates_path = Path(args.aggregates)
    if not aggregates_path.is_file():
        raise ValueError(f"aggregates file does not exist: {aggregates_path}")
    results_by_id: OrderedDict[str, dict[str, int | str]] = OrderedDict()
    frame_digests: OrderedDict[str, object] = OrderedDict()
    with _SegmentReader() as reader:
        # Fixed, bounded warmup.  Counters are intentionally discarded.
        for block in itertools.islice(_iter_manifest(manifest_path), 20):
            raw, _missing = _load_verified_block(block, reader)
            _run_manifest_codec_block(block, raw)

        # Reopen and stream the complete manifest.  Only aggregate summaries
        # remain resident; multi-million-row block metadata is never listed.
        for block in _iter_manifest(manifest_path):
            raw, _missing = _load_verified_block(block, reader)
            frame, encode_ns, decode_ns, mode = _run_manifest_codec_block(block, raw)
            result = results_by_id.setdefault(
                block.aggregate_id, _new_aggregate_result(block)
            )
            result["blocks"] += 1
            result["rows"] += block.row_count
            result["missing_count"] += block.missing_count
            result["validity_bytes"] += block.missing_length
            result["header_bytes"] += HEADER_SIZE
            result["valid_count"] += block.valid_count
            result["original_bytes"] += block.values_length
            result["compressed_bytes"] += len(frame)
            result["encode_ns"] += encode_ns
            result["decode_ns"] += decode_ns
            result["verified_values"] += block.valid_count
            if mode == MODE_DECIMAL:
                result["scaled_blocks"] += 1
            elif mode == MODE_IEEE_BITS:
                result["raw_blocks"] += 1
            else:
                raise RuntimeError(f"encoder produced invalid mode {mode}")
            digest = frame_digests.setdefault(block.aggregate_id, hashlib.sha256())
            digest.update(struct.pack("<I", len(frame)))
            digest.update(frame)

    rows = list(results_by_id.values())
    for result in rows:
        if result["valid_count"]:
            result["status"] = "ok"
        result["frame_sequence_sha256"] = frame_digests[
            str(result["aggregate_id"])
        ].hexdigest()
    _write_jsonl_atomic(Path(args.output), rows)
    return 0


def _floats_from_words(words: list[int]) -> list[float]:
    raw = b"".join(_U64.pack(word) for word in words)
    return _unpack_float64(raw)


def _golden_cases() -> list[tuple[str, bytes, int, bytes]]:
    return [
        ("empty", b"", 18, MAGIC),
        ("decimal_scale_2", _pack_float64([1.25, 1.5, -2.75, 0.0]), 2, MAGIC),
        (
            "decimal_modulo_wrap",
            _pack_float64(
                [
                    float(-9_000_000_000_000_000_000),
                    float(9_000_000_000_000_000_000),
                    float(-9_000_000_000_000_000_000),
                ]
            ),
            0,
            MAGIC,
        ),
        (
            "raw_ieee_specials",
            _pack_float64(
                _floats_from_words(
                    [
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
                )
            ),
            18,
            MAGIC,
        ),
        (
            "float32_decimal_scale_2",
            struct.pack("<4f", 1.25, 1.5, -2.75, 0.0),
            2,
            MAGIC32,
        ),
        (
            "float32_raw_specials",
            struct.pack(
                "<8I",
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x7F7FFFFF,
                0x7F800000,
                0xFF800000,
                0xFF7FFFFF,
            ),
            18,
            MAGIC32,
        ),
        (
            "float32_nan_payloads",
            struct.pack(
                "<6I",
                0x7FC00001,
                0x7F800001,
                0x7FFFFFFF,
                0xFFC01234,
                0xFF800001,
                0xFFFFFFFF,
            ),
            18,
            MAGIC32,
        ),
    ]


def _cmd_goldens(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict] = []

    for name, raw, max_point, magic in _golden_cases():
        if magic == MAGIC:
            frame = encode(_unpack_float64(raw), max_point=max_point)
            _require_bitwise_round_trip(raw, frame)
            input_name = f"{name}.f64le"
            frame_name = f"{name}.dbf1"
        else:
            frame = encode32(raw, max_point=max_point)
            _require_bitwise_round_trip32(raw, frame)
            input_name = f"{name}.f32le"
            frame_name = f"{name}.dbf2"
        (output_dir / input_name).write_bytes(raw)
        (output_dir / frame_name).write_bytes(frame)
        manifest_cases.append(
            {
                "name": name,
                "max_point": max_point,
                **_frame_metadata(frame),
                "input_file": input_name,
                "frame_file": frame_name,
                "input_sha256": _sha256(raw),
                "frame_sha256": _sha256(frame),
                "input_hex": raw.hex(),
                "frame_hex": frame.hex(),
                "bitwise_round_trip": True,
            }
        )

    manifest = {
        "formats": ["DBF1", "DBF2"],
        "byte_order": "little-endian",
        "wire_contract": str(CONTRACT_PATH.relative_to(ROOT)),
        "wire_contract_sha256": (
            _sha256(CONTRACT_PATH.read_bytes()) if CONTRACT_PATH.is_file() else None
        ),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "cases": len(manifest_cases),
                "all_bitwise_round_trip": True,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="stdin f64le -> stdout DBF1")
    encode_parser.add_argument("max_point", type=int)
    encode_parser.set_defaults(func=_cmd_encode)

    encode32_parser = subparsers.add_parser(
        "encode32", help="stdin f32le -> stdout DBF2"
    )
    encode32_parser.add_argument("max_point", type=int)
    encode32_parser.set_defaults(func=_cmd_encode32)

    decode_parser = subparsers.add_parser(
        "decode", help="stdin DBF1/DBF2 -> stdout native float bytes"
    )
    decode_parser.set_defaults(func=_cmd_decode)

    bench_parser = subparsers.add_parser("bench", help="stream f64le and print JSON")
    bench_parser.add_argument("max_point", type=int)
    bench_parser.add_argument(
        "inputs",
        nargs="*",
        help="optional f64le files; stdin is used when no files are named",
    )
    bench_parser.set_defaults(func=_cmd_bench)

    manifest_parser = subparsers.add_parser(
        "bench-manifest", help="benchmark validated float32/float64 source blocks"
    )
    manifest_parser.add_argument("--manifest", required=True)
    manifest_parser.add_argument("--aggregates", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--repeats", type=int, default=1)
    manifest_parser.set_defaults(func=_cmd_bench_manifest)

    golden_parser = subparsers.add_parser("goldens", help="write cross-runtime vectors")
    golden_parser.add_argument("output_dir", nargs="?", default=str(DEFAULT_GOLDEN_DIR))
    golden_parser.set_defaults(func=_cmd_goldens)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
