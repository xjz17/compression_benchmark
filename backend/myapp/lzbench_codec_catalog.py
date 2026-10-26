"""lzbench codec catalog: level sampling, algorithm keys, and -e selector strings."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

MAX_SAMPLED_LEVELS = 10

# lzbench selectors we benchmark (byte codecs not covered by the core suite).
_BSC_PRESETS: tuple[str, ...] = ("bsc0", "bsc1", "bsc2", "bsc3", "bsc4", "bsc5", "bsc6")

_BSC_OUTPUT_PATTERNS: dict[str, str] = {
    "bsc0": r"-m0\s+-e2",
    "bsc1": r"-m0\s+-e1",
    "bsc2": r"-m0\s+-e0",
    "bsc3": r"-m3\s+-e1",
    "bsc4": r"-m4\s+-e1",
    "bsc5": r"-m5\s+-e1",
    "bsc6": r"-m6\s+-e1",
}


@dataclass(frozen=True)
class LzbenchCodecEntry:
    codec_id: str
    display_name: str
    first_level: int | None = None
    last_level: int | None = None
    presets: tuple[str, ...] | None = None
    lzbench_selector: str | None = None
    sampled_levels: tuple[int, ...] | None = None
    default_preset: str | None = None

    @property
    def selector(self) -> str:
        return self.lzbench_selector or self.codec_id


LZBENCH_CODEC_ENTRIES: tuple[LzbenchCodecEntry, ...] = (
    LzbenchCodecEntry("brieflz", "BriefLZ", 1, 9),
    LzbenchCodecEntry("bsc", "BSC", presets=_BSC_PRESETS),
    LzbenchCodecEntry("bzip2", "Bzip2", 1, 9),
    LzbenchCodecEntry("bzip3", "Bzip3", 1, 10),
    LzbenchCodecEntry("crush", "Crush", 0, 2),
    LzbenchCodecEntry("csc", "CSC", 1, 5),
    LzbenchCodecEntry("fastlz", "FastLZ", 1, 2),
    LzbenchCodecEntry("fastlzma2", "FastLZMA2", 1, 10),
    LzbenchCodecEntry("kanzi", "Kanzi", 1, 9),
    LzbenchCodecEntry("libdeflate", "Libdeflate", 1, 12),
    LzbenchCodecEntry("lizard", "Lizard", 10, 49),
    LzbenchCodecEntry("lz4fast", "LZ4 Fast", sampled_levels=(1, 3, 9, 17)),
    LzbenchCodecEntry("lz4hc", "LZ4HC", 1, 12),
    LzbenchCodecEntry("lzav", "LZAV", 1, 2),
    LzbenchCodecEntry("lzf", "LZF", 0, 1),
    LzbenchCodecEntry("lzfse", "LZFSE", presets=("lzfse",)),
    LzbenchCodecEntry("lzg", "LZG", 1, 9),
    LzbenchCodecEntry("lzham", "LZHAM", 0, 4),
    LzbenchCodecEntry("lzjb", "LZJB", presets=("lzjb",)),
    LzbenchCodecEntry("lzlib", "Lzlib", 0, 9),
    LzbenchCodecEntry("lzo1x", "LZO1X", 1, 999),
    LzbenchCodecEntry("ppmd8", "PPMD8", 1, 9),
    LzbenchCodecEntry("quicklz", "QuickLZ", 1, 3),
    LzbenchCodecEntry("slz_deflate", "Slz_deflate", 1, 3),
    LzbenchCodecEntry("slz_gzip", "Slz_gzip", 1, 3),
    LzbenchCodecEntry("slz_zlib", "Slz_zlib", 1, 3),
    LzbenchCodecEntry("tornado", "Tornado", 1, 16),
    LzbenchCodecEntry("ucl_nrv2b", "UCL NRV2b", 1, 9),
    LzbenchCodecEntry("zlib", "Zlib", 1, 9),
    LzbenchCodecEntry("zlib-ng", "Zlib-ng", 1, 9, lzbench_selector="zlib-ng"),
    LzbenchCodecEntry("zling", "Zling", 0, 4),
    LzbenchCodecEntry("zxc", "ZXC", 1, 6),
    LzbenchCodecEntry("tamp", "TAMP", 8, 15),
)

_PARAMETER_VARIANT_ENTRIES: tuple[LzbenchCodecEntry, ...] = (
    LzbenchCodecEntry(
        "lzo1_variant",
        "LZO1",
        presets=("lzo1", "lzo1a", "lzo1b", "lzo1c", "lzo1f", "lzo1x_999", "lzo1y", "lzo1z"),
        default_preset="lzo1z",
    ),
    LzbenchCodecEntry(
        "slz_variant",
        "SLZ",
        presets=("slz_deflate_3", "slz_gzip_3", "slz_zlib_3"),
        default_preset="slz_zlib_3",
    ),
    LzbenchCodecEntry(
        "ucl_nrv_variant",
        "UCL NRV",
        presets=("ucl_nrv2b_9", "ucl_nrv2d", "ucl_nrv2e"),
        default_preset="ucl_nrv2e",
    ),
)

_PARAMETER_VARIANT_BACKING_CODEC_IDS = frozenset(
    {"lzo1x", "slz_deflate", "slz_gzip", "slz_zlib", "ucl_nrv2b"}
)

PARAMETER_CODEC_ENTRIES: tuple[LzbenchCodecEntry, ...] = (
    *_PARAMETER_VARIANT_ENTRIES,
    *(
        entry
        for entry in LZBENCH_CODEC_ENTRIES
        if entry.codec_id not in _PARAMETER_VARIANT_BACKING_CODEC_IDS
    ),
)

_LZBENCH_SKIP_OUTPUT_PREFIXES = (
    "brotli",
    "lz4 ",
    "snappy",
    "zstd",
    "lzma ",
    "xz ",
)

_GENERAL_ALGORITHM_ALIASES = {
    "lzma": "xz",
    "Chimp128": "CHIMP128",
    "CHIMP128": "CHIMP128",
    "chimp128": "CHIMP128",
    "chimp_128": "CHIMP128",
    "Chimp": "CHIMP",
    "chimp": "CHIMP",
}


def sample_levels(first: int, last: int, max_count: int = MAX_SAMPLED_LEVELS) -> list[int]:
    """If the inclusive range has <= max_count values, return all; else pick max_count evenly spaced."""
    if first > last:
        return []
    total = last - first + 1
    if total <= max_count:
        return list(range(first, last + 1))
    if max_count <= 1:
        return [first]
    out: list[int] = []
    seen: set[int] = set()
    for i in range(max_count):
        lv = last if i == max_count - 1 else round(first + (last - first) * i / (max_count - 1))
        lv = max(first, min(last, int(lv)))
        if lv not in seen:
            out.append(lv)
            seen.add(lv)
    if len(out) < max_count:
        for lv in range(first, last + 1):
            if lv in seen:
                continue
            out.append(lv)
            seen.add(lv)
            if len(out) >= max_count:
                break
    return sorted(out)


def _algorithm_key_for_level(codec_id: str, level: int) -> str:
    safe_id = codec_id.replace("-", "_")
    return f"{safe_id}_{level}"


def entry_sampled_levels(entry: LzbenchCodecEntry) -> list[int]:
    if entry.sampled_levels is not None:
        return list(entry.sampled_levels)
    if entry.presets is not None:
        if entry.codec_id == "bsc":
            return list(range(len(entry.presets)))
        return [0]
    assert entry.first_level is not None and entry.last_level is not None
    return sample_levels(entry.first_level, entry.last_level)


def entry_algorithm_keys(entry: LzbenchCodecEntry) -> list[str]:
    if entry.presets is not None:
        if entry.codec_id == "bsc":
            return list(entry.presets)
        return list(entry.presets)
    levels = entry_sampled_levels(entry)
    return [_algorithm_key_for_level(entry.codec_id, lv) for lv in levels]


def all_catalog_algorithm_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for entry in LZBENCH_CODEC_ENTRIES:
        keys.extend(entry_algorithm_keys(entry))
    return tuple(keys)


@lru_cache(maxsize=1)
def all_parameter_controlled_algorithm_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for entry in PARAMETER_CODEC_ENTRIES:
        keys.extend(entry_algorithm_keys(entry))
    return tuple(dict.fromkeys(keys))


def build_lzbench_selector(entry: LzbenchCodecEntry) -> str:
    if entry.presets is not None:
        return "/".join(entry.presets)
    levels = entry_sampled_levels(entry)
    return f"{entry.selector},{','.join(str(lv) for lv in levels)}"


def default_lzbench_e_string() -> str:
    return "/".join(build_lzbench_selector(entry) for entry in LZBENCH_CODEC_ENTRIES)


def iter_lzbench_selector_batches(selectors: str | None = None) -> list[str]:
    """One lzbench ``-e`` argument per catalog entry (or per ``/`` segment when overridden)."""
    raw = (selectors if selectors is not None else default_lzbench_e_string()).strip()
    if not raw:
        return []
    if "/" in raw:
        return [part.strip() for part in raw.split("/") if part.strip()]
    return [raw]


def default_lzbench_level_for_codec(codec_id: str) -> int | str:
    """Default UI / filter selection: highest-compression sampled level/preset."""
    entry = parameter_codec_entry_by_id(codec_id) or codec_entry_by_id(codec_id)
    if entry is None:
        return 0
    if entry.default_preset:
        return entry.default_preset
    if entry.codec_id == "bsc":
        presets = entry.presets or _BSC_PRESETS
        return presets[-1]
    if entry.codec_id == "lz4fast":
        return min(entry_sampled_levels(entry) or [1])
    levels = entry_sampled_levels(entry)
    if not levels:
        return 0
    return levels[-1]


def default_lzbench_level_selection() -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    for entry in PARAMETER_CODEC_ENTRIES:
        out[entry.codec_id] = default_lzbench_level_for_codec(entry.codec_id)
    return out


def codec_entry_by_id(codec_id: str) -> LzbenchCodecEntry | None:
    want = str(codec_id or "").strip()
    for entry in LZBENCH_CODEC_ENTRIES:
        if entry.codec_id == want:
            return entry
    return None


def parameter_codec_entry_by_id(codec_id: str) -> LzbenchCodecEntry | None:
    want = str(codec_id or "").strip()
    for entry in PARAMETER_CODEC_ENTRIES:
        if entry.codec_id == want:
            return entry
    return None


def parse_lzbench_levels_param(raw: Any) -> dict[str, int | str]:
    defaults = default_lzbench_level_selection()
    if raw is None or raw == "":
        return defaults
    parsed: dict[str, Any]
    if isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return defaults
    if not isinstance(parsed, dict):
        return defaults
    out = dict(defaults)
    for entry in PARAMETER_CODEC_ENTRIES:
        val = parsed.get(entry.codec_id)
        if val is None:
            continue
        if entry.presets is not None:
            presets = entry.presets or _BSC_PRESETS
            if isinstance(val, str) and val in presets:
                out[entry.codec_id] = val
            elif isinstance(val, int) and 0 <= val < len(presets):
                out[entry.codec_id] = presets[val]
            continue
        try:
            out[entry.codec_id] = int(val)
        except (TypeError, ValueError):
            continue
    return out


def resolve_selected_algorithm_key(entry: LzbenchCodecEntry, selected: int | str | None) -> str | None:
    if entry.presets is not None:
        presets = entry.presets or _BSC_PRESETS
        if isinstance(selected, str) and selected in presets:
            return selected
        if isinstance(selected, int) and 0 <= selected < len(presets):
            return presets[selected]
        if entry.default_preset and entry.default_preset in presets:
            return entry.default_preset
        return presets[-1] if entry.codec_id == "bsc" else presets[0]
    levels = entry_sampled_levels(entry)
    if not levels:
        return None
    if selected is None:
        selected = levels[-1]
    try:
        want = int(selected)
    except (TypeError, ValueError):
        want = levels[-1]
    if want in levels:
        return _algorithm_key_for_level(entry.codec_id, want)
    nearest = min(levels, key=lambda lv: (abs(lv - want), lv))
    return _algorithm_key_for_level(entry.codec_id, nearest)


def selected_lzbench_algorithm_keys(lzbench_levels: Mapping[str, int | str] | None = None) -> set[str]:
    levels = parse_lzbench_levels_param(lzbench_levels)
    keys: set[str] = set()
    for entry in PARAMETER_CODEC_ENTRIES:
        chosen = resolve_selected_algorithm_key(entry, levels.get(entry.codec_id))
        if chosen:
            keys.add(chosen)
    return keys


def parse_lzbench_algorithm_key(key: str) -> dict[str, Any] | None:
    """Parse a catalog algorithm key into codec entry and level (mirrors frontend)."""
    s = str(key or '').strip()
    if not s:
        return None
    if s.startswith('lzbench_'):
        return parse_lzbench_algorithm_key(s[len('lzbench_'):])
    for entry in PARAMETER_CODEC_ENTRIES:
        if entry.presets is not None and s in entry.presets:
            return {'entry': entry, 'level': s, 'preset': s}
    for entry in PARAMETER_CODEC_ENTRIES:
        if entry.presets is not None:
            continue
        prefix = entry.codec_id.replace('-', '_') + '_'
        if s.startswith(prefix):
            tail = s[len(prefix):]
            try:
                level = int(tail)
            except ValueError:
                continue
            return {'entry': entry, 'level': level}
    return None


def is_lzbench_catalog_algorithm(algo: str) -> bool:
    s = str(algo or "").strip()
    if not s:
        return False
    controlled = set(all_parameter_controlled_algorithm_keys())
    if s.startswith("lzbench_"):
        inner = s[len("lzbench_") :]
        return inner in controlled or inner in all_catalog_algorithm_keys()
    return s in controlled or s in all_catalog_algorithm_keys()


def is_lzbench_selector_garbage_algorithm(algo: str) -> bool:
    """Placeholder rows from failed lzbench subprocesses (must never enter rankings)."""
    return str(algo or "").strip().lower().startswith("lzbench selector")


def canonical_report_algorithm_key(name: str) -> str | None:
    """Normalize report ``algorithm`` values to catalog keys; drop garbage names."""
    raw = str(name or "").strip()
    if not raw or is_lzbench_selector_garbage_algorithm(raw):
        return None
    if raw in _GENERAL_ALGORITHM_ALIASES:
        return _GENERAL_ALGORITHM_ALIASES[raw]
    if raw.startswith("lzma_level_"):
        return "xz_level_" + raw[len("lzma_level_") :]
    if raw.startswith("lzbench_"):
        inner = raw[len("lzbench_") :]
        if inner in all_catalog_algorithm_keys():
            return inner
        mapped = lzbench_output_to_algorithm_key(raw) or lzbench_output_to_algorithm_key(inner)
        return mapped
    if raw in all_catalog_algorithm_keys():
        return raw
    mapped = lzbench_output_to_algorithm_key(raw)
    if mapped:
        return mapped
    return raw


def is_lzbench_report_row_algorithm(name: str) -> bool:
    """Whether a report row belongs to the lzbench catalog (raw stdout name or canonical key)."""
    raw = str(name or "").strip()
    if not raw:
        return False
    if is_lzbench_selector_garbage_algorithm(raw):
        return True
    catalog = all_catalog_algorithm_keys()
    if raw in catalog:
        return True
    if raw.startswith("lzbench_"):
        inner = raw[len("lzbench_") :]
        if inner in catalog:
            return True
    return lzbench_output_to_algorithm_key(raw) is not None


def lzbench_output_to_algorithm_key(lzbench_name: str) -> str | None:
    """Map lzbench CSV name column to catalog algorithm key (no ``lzbench_`` prefix)."""
    raw = str(lzbench_name or "").strip()
    lower = raw.lower()
    if not lower or lower.startswith("memcpy"):
        return None
    if lower.startswith("lz4 ") and "--fast" in lower:
        level_match = re.search(r"-(\d+)\s*$", raw)
        if level_match:
            level = int(level_match.group(1))
            key = _algorithm_key_for_level("lz4fast", level)
            entry = codec_entry_by_id("lz4fast")
            if entry is not None and level in entry_sampled_levels(entry):
                return key
        return None
    if any(lower.startswith(prefix) for prefix in _LZBENCH_SKIP_OUTPUT_PREFIXES):
        return None
    if lower.startswith("bsc "):
        for preset, pattern in _BSC_OUTPUT_PATTERNS.items():
            if re.search(pattern, raw, flags=re.IGNORECASE):
                return preset
        return None
    if lower.startswith("lzfse"):
        return "lzfse"
    if lower.startswith("lzjb"):
        return "lzjb"
    level_match = re.search(r"-(\d+)\s*$", raw)
    level = int(level_match.group(1)) if level_match else None
    for entry in LZBENCH_CODEC_ENTRIES:
        if entry.presets is not None:
            continue
        sel = entry.selector.lower()
        if lower.startswith(sel + " ") or lower.startswith(sel.replace("-", " ") + " "):
            if level is None:
                return None
            key = _algorithm_key_for_level(entry.codec_id, level)
            if level in entry_sampled_levels(entry):
                return key
            return None
    return None
