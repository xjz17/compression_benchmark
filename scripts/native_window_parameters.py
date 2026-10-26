"""Native-LZ window size is an invocation parameter, not an algorithm ID."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from collections.abc import Mapping

ALGORITHM_NAME = "op.match_window_model.native_lz.window.cpp_v2"
PARAMETER_SCHEMA_VERSION = "native-lz-window-parameters-v1"
DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "benchmark/reference/actions/native_window_defaults.json"
WINDOW_CANDIDATES = (3072, 4096, 5120, 8192, 12288, 16384, 19456, 24576, 28672, 32768, 40960, 53248, 57344, 65535)
_LEGACY_NAME = re.compile(r"op\.match_window_model\.native_lz\.window(\d+)$")

def _validate(parameters):
    if not isinstance(parameters, Mapping) or set(parameters) != {"window_bytes"}:
        raise ValueError("Window parameters require exactly window_bytes")
    value = parameters["window_bytes"]
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("window_bytes must be an integer in 1..65535, not a boolean")
    return {"window_bytes": value}

def default_parameters():
    if not DEFAULTS_PATH.is_file():
        raise RuntimeError("native-LZ window calibration is missing; specify window_bytes explicitly")
    return _validate(json.loads(DEFAULTS_PATH.read_text())["parameters"])

def normalize_parameters(parameters=None):
    return default_parameters() if parameters is None else _validate(parameters)

def parameters_from_name(name):
    match = _LEGACY_NAME.fullmatch(name) if isinstance(name, str) else None
    return _validate({"window_bytes": int(match.group(1))}) if match else None

def legacy_action_name(parameters=None):
    return f"op.match_window_model.native_lz.window{normalize_parameters(parameters)['window_bytes']}"

def parameter_candidates():
    return tuple({"window_bytes": value} for value in WINDOW_CANDIDATES)

def parameter_schema():
    return {"schema_version": PARAMETER_SCHEMA_VERSION,
            "properties": {"window_bytes": {"type": "integer", "minimum": 1, "maximum": 65535}},
            "required": ["window_bytes"], "additionalProperties": False,
            "search_domain": {"window_bytes": list(WINDOW_CANDIDATES)},
            "learning_candidate_count": len(WINDOW_CANDIDATES), "default": default_parameters(),
            "default_provenance": str(DEFAULTS_PATH.relative_to(DEFAULTS_PATH.parents[3])),
            "default_scope": "minimum aggregate complete NLC1 frame bytes on recorded calibration samples and candidate grid",
            "semantics": "maximum backward byte distance; stores NLS1 context, search executes in Match Parse Policy",
            "wire_constraint": "offset16 limits maximum backward distance to 65535 bytes"}

def parameter_fingerprint():
    record = {"schema": parameter_schema(), "defaults_sha256": hashlib.sha256(DEFAULTS_PATH.read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
