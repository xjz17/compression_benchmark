"""Thin adapter for the C++ legacy numeric actions; no Python codec kernels.

The stable action names and LNS1 metadata contract are preserved for recorded
pipelines. Decimal scaling, ALP exception selection, limb transforms, native-LZ
search/packing, framing and all inverses live in legacy_numeric_action_stages.cpp.
"""
from __future__ import annotations

import json
from .cpp_action_bridge import run as _cpp_run

KEY = "zstd_level_3"
MAGIC = b"LNS1"
SCALE = "op.scaling.legacy_exact_decimal.python_v1"
ALP_SCALE = "op.scaling.alp_legacy_fe_exceptions.python_v1"
LIMBS = "op.block_transform.int64_lo_hi32.python_v1"
MATCH = "op.lz_match.native_lz_limb_batch.aligned8_greedy.python_v1"
PACK = "op.match_token_format.native_lz_limb_batch.ctrl8.python_v1"
FRAME = "op.container_frame.legacy_numeric_limbs.python_v1"
ALP_FRAME = "op.container_frame.alp2_legacy.python_v1"
SCALED_ACTIONS = (SCALE, LIMBS, MATCH, PACK, FRAME)
ALP_ACTIONS = (ALP_SCALE, LIMBS, MATCH, PACK, ALP_FRAME)
ACTION_NAMES = tuple(dict.fromkeys(SCALED_ACTIONS + ALP_ACTIONS))


def _read(raw: bytes, stage: str) -> dict:
    """Inspect stage metadata for inverse context; this is not a codec."""
    if not raw.startswith(MAGIC):
        raise ValueError("expected a legacy numeric stage packet")
    obj = json.loads(raw[len(MAGIC):])
    if obj.get("stage") != stage:
        raise ValueError(f"expected legacy numeric {stage} packet")
    return obj


def forward(action: str, raw: bytes, kind: str = "bytes", max_point: int | None = None) -> bytes:
    return _cpp_run("legacy_numeric_action_stages_cli", "forward", raw, action,
                    kind, "none" if max_point is None else max_point)


def inverse(action: str, raw: bytes, *, count: int = 0) -> bytes:
    return _cpp_run("legacy_numeric_action_stages_cli", "inverse", raw, action, count)
