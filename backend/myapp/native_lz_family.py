"""Native-LZ public API and preset metadata; codec kernels execute in C++.

Both historical fast and chain/lazy paths retain their original wire bytes.
Python only selects the configured path, marshals token IPC, and aggregates
benchmark results. There is no Python matcher, ctrl8 codec, or decoder fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import os
import struct

from .benchmark_loss import accumulate_byte_loss
from .cpp_action_bridge import run as _cpp_run
from .lzbench_codec_catalog import all_catalog_algorithm_keys
from .active_method_policy import (
    active_method_policy_summary,
    is_active_algorithm_key,
)


@dataclass(frozen=True)
class NativeLzPreset:
    min_match: int
    max_match: int
    window: int
    max_candidates: int
    lazy: bool = False


@dataclass(frozen=True)
class NativeLzToken:
    """One independently packable token from the native LZ parse stage.

    A literal has ``literal`` set and zero ``offset``/``length``.  A match has
    ``literal`` unset and carries the already selected offset and full match
    length.  Keeping this representation outside the byte packer makes the
    match/parse policy and token-format stages independently testable without
    changing the historical stream format.
    """

    literal: int | None = None
    offset: int = 0
    length: int = 0

    @property
    def is_match(self) -> bool:
        return self.literal is None


LEGACY_NATIVE_LZ_PRESETS: dict[str, NativeLzPreset] = {
    "density": NativeLzPreset(4, 66, 4096, 8),
    "gipfeli": NativeLzPreset(4, 130, 32768, 16, True),
    "glza": NativeLzPreset(5, 255, 65535, 64, True),
    "lzmat": NativeLzPreset(3, 255, 8192, 16),
    "lzo1": NativeLzPreset(3, 33, 4096, 4),
    "lzo1a": NativeLzPreset(3, 64, 4096, 6),
    "lzo1b": NativeLzPreset(3, 96, 8192, 8),
    "lzo1c": NativeLzPreset(3, 128, 8192, 12, True),
    "lzo1f": NativeLzPreset(3, 32, 2048, 2),
    "lzo1y": NativeLzPreset(3, 160, 16384, 16, True),
    "lzo1z": NativeLzPreset(3, 255, 32768, 24, True),
    "lzo2a": NativeLzPreset(3, 255, 65535, 32, True),
    "lzrw": NativeLzPreset(3, 18, 4096, 1),
    "lzsse2": NativeLzPreset(4, 64, 8192, 4),
    "lzsse4": NativeLzPreset(4, 96, 16384, 8),
    "lzsse4fast": NativeLzPreset(4, 48, 8192, 2),
    "lzsse8": NativeLzPreset(4, 160, 32768, 16, True),
    "lzsse8fast": NativeLzPreset(4, 80, 16384, 4),
    "lzvn": NativeLzPreset(3, 258, 65535, 8),
    "pithy": NativeLzPreset(4, 128, 32768, 8),
    "shrinker": NativeLzPreset(3, 255, 4096, 32, True),
    "ucl_nrv2d": NativeLzPreset(3, 255, 8192, 16, True),
    "ucl_nrv2e": NativeLzPreset(3, 255, 16384, 24, True),
    "wflz": NativeLzPreset(3, 64, 4096, 4),
    "xpack": NativeLzPreset(4, 255, 65535, 64, True),
    "yalz77": NativeLzPreset(3, 255, 32768, 16, True),
    "yappy": NativeLzPreset(4, 32, 4096, 2),
}

# Public/default selection contains active algorithms only. The legacy table is
# retained so historical streams, including retired ones, remain decodable.
NATIVE_LZ_PRESETS: dict[str, NativeLzPreset] = {
    key: preset
    for key, preset in LEGACY_NATIVE_LZ_PRESETS.items()
    if is_active_algorithm_key(key)
}

STRICT_NATIVE_GENERIC_ALGORITHMS = (
    "brotli",
    "gz",
    "lz4-framed",
    "snappy-framed",
    "blosc_blosclz",
    "blosc2_blosclz",
    "fpzip",
    "fpcompress_dpratio",
    "fpcompress_dpspeed",
    "elf_on_chimp",
    "elf_plus",
    "camel",
    "ext_zfp_zfpy",
    "fse",
    "huff0",
)


def _generated_native_algorithm_keys() -> tuple[str, ...]:
    candidates = tuple(
        dict.fromkeys(
            [
                *NATIVE_LZ_PRESETS.keys(),
                *STRICT_NATIVE_GENERIC_ALGORITHMS,
                *(f"xz_level_{level}" for level in range(10)),
                *(f"zstd_level_{level}" for level in range(1, 10)),
                *all_catalog_algorithm_keys(),
            ]
        )
    )
    return tuple(key for key in candidates if is_active_algorithm_key(key))


NATIVE_LZ_FAMILY_ALGORITHMS = _generated_native_algorithm_keys()
_DEFAULT_FAST_PATH_BYTES = 1_000_000


def _algorithm_signature(algorithm: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(str(algorithm)))


def _algorithm_level(algorithm: str) -> int:
    s = str(algorithm)
    if s.startswith("bsc") and len(s) > 3 and s[3:].isdigit():
        return int(s[3:])
    if "_" in s:
        tail = s.rsplit("_", 1)[1]
        if tail.isdigit():
            return int(tail)
    return {
        "brotli": 6,
        "gz": 6,
        "lz4-framed": 1,
        "snappy-framed": 1,
        "blosc_blosclz": 5,
        "blosc2_blosclz": 5,
        "fpzip": 6,
        "fpcompress_dpratio": 8,
        "fpcompress_dpspeed": 2,
        "ext_zfp_zfpy": 7,
        "lzfse": 6,
        "lzjb": 2,
    }.get(s, 5)


def _algorithm_profile(algorithm: str) -> str:
    s = str(algorithm).lower()
    if s == "fpcompress_dpspeed" or s in {"lz4-framed", "snappy-framed", "lzfse", "lzjb"} or s.startswith(
        ("lz4fast_", "fastlz_", "quicklz_", "lzf_")
    ):
        return "speed"
    if s == "gz" or s.startswith(
        ("zlib_", "zlib_ng_", "slz_deflate_", "slz_gzip_", "slz_zlib_", "libdeflate_")
    ):
        return "deflate"
    if s in {"fpzip", "ext_zfp_zfpy"}:
        return "float"
    if s == "fpcompress_dpratio" or s == "brotli" or s.startswith(
        (
            "xz_level_",
            "zstd_level_",
            "fastlzma2_",
            "lzlib_",
            "csc_",
            "bzip2_",
            "bzip3_",
            "bsc",
            "kanzi_",
            "zling_",
            "ppmd8_",
            "lzg_",
            "lzham_",
            "tornado_",
        )
    ):
        return "ratio"
    return "balanced"


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _generated_preset(algorithm: str) -> NativeLzPreset:
    level = _algorithm_level(algorithm)
    sig = _algorithm_signature(algorithm)
    profile = _algorithm_profile(algorithm)

    if profile == "speed":
        min_match = 3 + (sig % 2)
        max_match = _clamp(18 + min(80, level * 4) + (sig % 13), min_match, 258)
        window = _clamp(1024 * (2 + min(30, max(1, level))), 1024, 65535)
        candidates = _clamp(1 + (level // 4) + (sig % 3), 1, 8)
        return NativeLzPreset(min_match, max_match, window, candidates, False)

    if profile == "deflate":
        min_match = 3
        max_match = _clamp(66 + min(120, level * 10) + (sig % 23), min_match, 258)
        window = _clamp(8192 * (1 + min(7, max(1, level) // 2)), 4096, 65535)
        candidates = _clamp(6 + level * 2 + (sig % 5), 4, 32)
        return NativeLzPreset(min_match, max_match, window, candidates, level >= 5 or sig % 4 == 0)

    if profile == "ratio":
        min_match = 3 + (sig % 3)
        max_match = _clamp(130 + min(100, level * 12) + (sig % 29), min_match, 258)
        window = 65535 if level >= 4 else 32768
        candidates = _clamp(16 + level * 4 + (sig % 9), 8, 64)
        return NativeLzPreset(min_match, max_match, window, candidates, True)

    if profile == "float":
        min_match = 4
        max_match = _clamp(96 + level * 9 + (sig % 41), min_match, 258)
        candidates = _clamp(12 + level * 3 + (sig % 11), 8, 48)
        return NativeLzPreset(min_match, max_match, 32768, candidates, True)

    min_match = 3 + (sig % 2)
    max_match = _clamp(64 + min(120, level * 8) + (sig % 31), min_match, 258)
    window = _clamp(4096 * (1 + min(15, max(1, level))), 4096, 65535)
    candidates = _clamp(4 + level * 2 + (sig % 7), 2, 40)
    return NativeLzPreset(min_match, max_match, window, candidates, level >= 6)


def _preset(algorithm: str) -> NativeLzPreset:
    """Resolve legacy metadata for replay and decode compatibility."""
    key = str(algorithm)
    if key in LEGACY_NATIVE_LZ_PRESETS:
        return LEGACY_NATIVE_LZ_PRESETS[key]
    if key in NATIVE_LZ_FAMILY_ALGORITHMS:
        return _generated_preset(key)
    raise ValueError(f"unknown native LZ-family algorithm: {algorithm}")


def _active_preset(algorithm: str) -> NativeLzPreset:
    key = str(algorithm)
    if not is_active_algorithm_key(key):
        policy = active_method_policy_summary()
        raise ValueError(
            f"algorithm {key!r} is retired by active-method policy "
            f"{policy['policy_id']}"
        )
    return _preset(key)


def _fast_path_threshold() -> int:
    raw = os.environ.get("WEB_COMPRESSION_NATIVE_LZ_FAST_BYTES", "").strip()
    if not raw:
        return _DEFAULT_FAST_PATH_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_FAST_PATH_BYTES
    return max(0, value)


def _literal_fast_enabled() -> bool:
    return os.environ.get("WEB_COMPRESSION_NATIVE_LZ_LITERAL_FAST", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def native_lz_literal_fast_enabled() -> bool:
    return _literal_fast_enabled()


def _run(command: str, data: bytes, *args) -> bytes:
    return _cpp_run("native_lz_action_stages_cli", command, bytes(data or b""), *args)


def _tokens_from_packet(raw: bytes) -> tuple[NativeLzToken, ...]:
    """Read C++ token IPC metadata; this does not decode an LZ byte stream."""
    if len(raw) < 8:
        raise ValueError("truncated C++ token IPC")
    count = struct.unpack_from("<Q", raw)[0]
    pos, tokens = 8, []
    if count > (len(raw)-8)//2:
        raise ValueError("truncated C++ token list")
    for _ in range(count):
        tag = raw[pos]
        pos += 1
        if tag == 0:
            if pos >= len(raw):
                raise ValueError("truncated token IPC literal")
            tokens.append(NativeLzToken(literal=raw[pos]))
            pos += 1
        elif tag == 1:
            if pos+4 > len(raw):
                raise ValueError("truncated token IPC match fields")
            offset, length = struct.unpack_from("<HH", raw, pos)
            tokens.append(NativeLzToken(offset=offset, length=length))
            pos += 4
        else:
            raise ValueError("invalid token IPC tag")
    if pos != len(raw):
        raise ValueError("trailing token IPC bytes")
    return tuple(tokens)


def _tokens_to_packet(tokens: tuple[NativeLzToken, ...]) -> bytes:
    """Marshal public token records to C++; ctrl8 packing occurs only there."""
    fields = [struct.pack("<Q", len(tokens))]
    for token in tokens:
        if token.is_match:
            fields.append(struct.pack("<BHH", 1, token.offset, token.length))
        else:
            fields.append(struct.pack("<BB", 0, token.literal))
    return b"".join(fields)


def _compress_native_lz_family_literal(data: bytes) -> bytes:
    return _run("literal", data)


def _compress_native_lz_family_fast(data: bytes, preset: NativeLzPreset) -> bytes:
    return _run("compress", data, preset.window, preset.min_match, preset.max_match)


def _parse_native_lz_family_fast(data: bytes, preset: NativeLzPreset) -> tuple[NativeLzToken, ...]:
    return _tokens_from_packet(_run("parse", data, preset.window, preset.min_match, preset.max_match))


def _parse_native_lz_family_chain(data: bytes, preset: NativeLzPreset) -> tuple[NativeLzToken, ...]:
    return _tokens_from_packet(_run("parse-chain", data, preset.window, preset.min_match,
                                   preset.max_match, preset.max_candidates, int(preset.lazy)))


def parse_native_lz_family(data: bytes, algorithm: str) -> tuple[NativeLzToken, ...]:
    preset = _active_preset(algorithm)
    raw = bytes(data or b"")
    if len(raw) >= _fast_path_threshold():
        if _literal_fast_enabled():
            return _tokens_from_packet(_run("unpack", _run("literal", raw), preset.min_match, len(raw)))
        return _parse_native_lz_family_fast(raw, preset)
    return _parse_native_lz_family_chain(raw, preset)


def pack_native_lz_tokens(tokens: tuple[NativeLzToken, ...], algorithm: str) -> bytes:
    return _run("pack", _tokens_to_packet(tokens), _active_preset(algorithm).min_match)


def compress_native_lz_family_staged(data: bytes, algorithm: str) -> bytes:
    return pack_native_lz_tokens(parse_native_lz_family(data, algorithm), algorithm)


def compress_native_lz_family_staged_fast(data: bytes, algorithm: str) -> bytes:
    return _compress_native_lz_family_fast(bytes(data or b""), _active_preset(algorithm))


def compress_native_lz_family(data: bytes, algorithm: str) -> bytes:
    preset = _active_preset(algorithm)
    raw = bytes(data or b"")
    if len(raw) >= _fast_path_threshold():
        if _literal_fast_enabled():
            return _compress_native_lz_family_literal(raw)
        return _compress_native_lz_family_fast(raw, preset)
    return _run("compress-chain", raw, preset.window, preset.min_match,
                preset.max_match, preset.max_candidates, int(preset.lazy))


def decompress_native_lz_family(data: bytes, algorithm: str, expected_size: int) -> bytes:
    return _run("decompress", data, _preset(algorithm).min_match, expected_size)


def _bench_columns(columns: list[bytes], algorithm: str, literal: bool = False) -> list[float]:
    # Public API timing remains end-to-end and includes the CLI process bridge.
    # Do not present these figures as isolated C++ kernel timing.
    total_o = total_c = total_te = total_td = 0
    loss_acc = [0.0, 0.0]
    for data in columns:
        raw = bytes(data or b"")
        start = time.perf_counter_ns()
        compressed = (_compress_native_lz_family_literal(raw) if literal
                      else compress_native_lz_family(raw, algorithm))
        total_te += time.perf_counter_ns()-start
        start = time.perf_counter_ns()
        restored = decompress_native_lz_family(compressed, algorithm, len(raw))
        total_td += time.perf_counter_ns()-start
        total_o += len(raw)
        total_c += len(compressed)
        if restored == raw:
            loss_acc[1] += float(len(raw))
        else:
            accumulate_byte_loss(raw, restored, loss_acc)
    return [total_o, total_c, total_te, total_td, loss_acc[0], loss_acc[1]]


def bench_native_lz_family_literal_bytes(column_byte_data: list[bytes]) -> list[float]:
    return _bench_columns(column_byte_data, "zlib_1", literal=True)


def bench_native_lz_family_bytes(column_byte_data: list[bytes], algorithm: str) -> list[float]:
    return _bench_columns(column_byte_data, algorithm)
