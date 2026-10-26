"""Native-LZ search parameters, independent of algorithm and token-format IDs.

The learned domain uses effective lengths to avoid duplicate aligned-8 search
settings. Historical raw values remain accepted for exact packet replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping

ALGORITHM_NAME = "op.lz_match.native_lz.aligned8.cpp_v2"
PARAMETER_SCHEMA_VERSION = "native-lz-search-parameters-v1"
DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/native_lz_defaults.json"
MIN_CANDIDATES = (8, 16, 24, 32, 64)
MAX_CANDIDATES = tuple(range(8, 257, 8))
_LEGACY_NAME = re.compile(r"op\.lz_match\.native_lz\.aligned8\.min(\d+)\.max(\d+)$")


def _validate(parameters: Mapping) -> dict[str, int]:
    if not isinstance(parameters, Mapping) or set(parameters) != {"min_match", "max_match"}:
        raise ValueError("LZ parameters require exactly min_match and max_match")
    minimum, maximum = parameters["min_match"], parameters["max_match"]
    if type(minimum) is not int or type(maximum) is not int:
        raise ValueError("LZ min_match and max_match must be integers, not booleans")
    if not (3 <= minimum <= 256 and 8 <= maximum <= 258 and minimum <= maximum):
        raise ValueError("LZ requires 3<=min_match<=256, 8<=max_match<=258, min_match<=max_match")
    if max(8, (minimum + 7) // 8 * 8) > maximum // 8 * 8:
        raise ValueError("LZ effective minimum exceeds effective maximum after 8-byte alignment")
    return {"min_match": minimum, "max_match": maximum}


def default_parameters() -> dict[str, int]:
    """Return the frozen calibration winner; never search on inference input."""
    if not DEFAULTS_PATH.is_file():
        raise RuntimeError("native-LZ default calibration is missing; specify parameters explicitly")
    record = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    return _validate(record["parameters"])


def normalize_parameters(parameters: Mapping | None = None) -> dict[str, int]:
    return default_parameters() if parameters is None else _validate(parameters)


def parameters_from_name(name: str) -> dict[str, int] | None:
    match = _LEGACY_NAME.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        return None
    return _validate(dict(zip(("min_match", "max_match"), map(int, match.groups()))))


def legacy_action_name(parameters: Mapping | None = None) -> str:
    selected = normalize_parameters(parameters)
    return f"op.lz_match.native_lz.aligned8.min{selected['min_match']}.max{selected['max_match']}"


def parameter_candidates() -> tuple[dict[str, int], ...]:
    return tuple({"min_match": minimum, "max_match": maximum}
                 for minimum in MIN_CANDIDATES for maximum in MAX_CANDIDATES
                 if minimum <= maximum)


def parameter_schema() -> dict:
    return {
        "schema_version": PARAMETER_SCHEMA_VERSION,
        "properties": {
            "min_match": {"type": "integer", "minimum": 3, "maximum": 256},
            "max_match": {"type": "integer", "minimum": 8, "maximum": 258},
        },
        "required": ["min_match", "max_match"],
        "additionalProperties": False,
        "constraints": ["min_match <= max_match", "ceil(min_match/8)*8 <= floor(max_match/8)*8"],
        "search_domain": {"min_match": list(MIN_CANDIDATES), "max_match": list(MAX_CANDIDATES)},
        "learning_candidate_count": len(parameter_candidates()),
        "default": default_parameters(),
        "default_provenance": str(DEFAULTS_PATH.relative_to(DEFAULTS_PATH.parents[3])),
        "default_scope": "minimum aggregate complete-frame bytes on the recorded calibration samples and candidate grid",
        "token_length_bias": "independent I/O-format parameter; decoder uses the selected token-format bias",
    }


def parameter_fingerprint() -> str:
    payload = {"schema": parameter_schema(), "defaults_sha256": hashlib.sha256(DEFAULTS_PATH.read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
