"""Current byte-optimal RLE profiles; historical RLE codecs remain replay-only.

Optimal means minimum complete URL1 bytes within the native declared grammar,
not an optimum over arbitrary compressors. Python only validates/bridges I/O.
"""
from __future__ import annotations
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_NAME = "op.run_length.byte_optimal.cpp_v1"
WORDS_VARIANT = "op.run_length.byte_optimal.words64.cpp_v1"
BYTES_VARIANT = "op.run_length.byte_optimal.uint8.cpp_v1"
SCALED_VARIANT = "op.run_length.byte_optimal.tsfile_scaled.cpp_v1"
RLBE_VARIANT = "op.run_length.byte_optimal.rlbe_lengths.cpp_v1"
OLD_SPRINTZ_VARIANT = "op.bitpack_int.sprintz.cpp_v1"
SPRINTZ_VARIANT = "op.bitpack_int.sprintz.unified_rle_tail.cpp_v2"
VARIANTS = (WORDS_VARIANT, BYTES_VARIANT, SCALED_VARIANT, RLBE_VARIANT)
OLD_ALGORITHMS = frozenset(("op.run_length.value_count.cpp_v2", "op.run_length.tsfile_int64.v1", "op.run_length.rlbe.cpp_v1"))
OLD_TO_CURRENT = {
    "op.run_length.reference_v1": WORDS_VARIANT,
    "op.run_length.bzip2_uint8.cpp_reference_v1": BYTES_VARIANT,
    "op.run_length.tsfile_int64.v1": WORDS_VARIANT,
    "op.run_length.rlbe.cpp_v1": RLBE_VARIANT,
}
SOURCE = "cpp/src/optimal_rle.cpp"
PRODUCERS = {SCALED_VARIANT: "op.scaling.tsfile_rle.cpp_v1", RLBE_VARIANT: "op.delta.rlbe.cpp_v1"}
CLI = ROOT / "cpp/build/optimal_rle_cli"
RLBE_CLI = ROOT / "cpp/build/unified_rle_numeric_cli"


def _run(executable, args, payload):
    if not executable.is_file():
        raise ValueError(f"missing native RLE executable: {executable}")
    result = subprocess.run([str(executable), *args], input=payload, capture_output=True, check=False)
    if result.returncode:
        raise ValueError("native RLE rejected input: " + result.stderr.decode(errors="replace").strip())
    return result.stdout


def upgrade_variant(name, previous_name=None):
    """Explicit current-recipe replacement, distinct from historical replay."""
    if name == "op.run_length.tsfile_int64.v1" and previous_name == PRODUCERS[SCALED_VARIANT]:
        return SCALED_VARIANT
    if name == OLD_SPRINTZ_VARIANT: return SPRINTZ_VARIANT
    return OLD_TO_CURRENT.get(name, name)


def forward(payload: bytes, kind: str, variant: str) -> bytes:
    if variant == WORDS_VARIANT:
        if kind not in ("int64", "float64", "bytes") or len(payload) % 8:
            raise ValueError("words64 requires aligned int64/float64 raw bits or explicit 64-bit bytes")
        return _run(CLI, ["encode", "8"], payload)
    if variant == BYTES_VARIANT:
        if kind != "bytes": raise ValueError("uint8 RLE requires bytes")
        return _run(CLI, ["encode", "1"], payload)
    if kind != "bytes": raise ValueError("RLE source adapters require their preceding byte packet")
    if variant == SCALED_VARIANT:
        if len(payload) < 16 or payload[:4] != b"RLS1": raise ValueError("expected RLS1 scaling packet")
        scale, count = struct.unpack_from("<IQ", payload, 4)
        if scale > 18 or len(payload) != 16 + count * 8: raise ValueError("invalid RLS1 scaling packet")
        return b"URS1" + struct.pack("<I", scale) + _run(CLI, ["encode", "8"], payload[16:])
    if variant == RLBE_VARIANT:
        if len(payload) < 12 or payload[:4] != b"RLD1": raise ValueError("expected RLD1 delta packet")
        count = struct.unpack_from("<Q", payload, 4)[0]
        if len(payload) != 12 + count * 8: raise ValueError("invalid RLD1 delta packet")
        return _run(RLBE_CLI, ["encode", "rlbe_deltas", "i64"], payload[12:])
    raise ValueError("unknown current RLE I/O profile")


def inverse(payload: bytes, variant: str) -> bytes:
    if variant in (WORDS_VARIANT, BYTES_VARIANT):
        raw = _run(CLI, ["decode"], payload)
        # The native frame carries width; the wrapper also validates the
        # expected profile before allowing a self-describing frame to cross it.
        width = symbol_width(payload)
        if width != (8 if variant == WORDS_VARIANT else 1): raise ValueError("RLE symbol width/profile mismatch")
        return raw
    if variant == SCALED_VARIANT:
        if len(payload) < 8 or payload[:4] != b"URS1": raise ValueError("expected URS1 scaled RLE packet")
        scale = struct.unpack_from("<I", payload, 4)[0]
        if scale > 18 or symbol_width(payload[8:]) != 8: raise ValueError("invalid scaled RLE profile")
        raw = _run(CLI, ["decode"], payload[8:])
        return b"RLS1" + struct.pack("<IQ", scale, len(raw) // 8) + raw
    if variant == RLBE_VARIANT:
        if len(payload) < 20 or payload[:4] != b'URB1':
            raise ValueError('current RLBE lengths profile requires URB1')
        raw = _run(RLBE_CLI, ["decode", "rlbe_deltas", "i64"], payload)
        return b"RLD1" + struct.pack("<Q", len(raw) // 8) + raw
    raise ValueError("unknown current RLE I/O profile")


def symbol_width(payload):
    if len(payload) < 6 or payload[:4] != b"URL1": raise ValueError("expected URL1 RLE frame")
    # Kept in sync with the native frame contract, never inferred from values.
    return payload[4]
