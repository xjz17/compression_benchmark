#!/usr/bin/env python3
"""Complete NNP1 NNCP entropy path used by the typed graph.

This module is deliberately separate from the symbol-probability API.  A110 is
selectable only as its existing complete, terminal NNP1 compression path; the
policy never treats a probability vector as compressed bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any

from build_nncp_probability import (
    DRIVER,
    ROOT,
    nncp_availability,
    nncp_command,
    sha256,
    source_fingerprints as nncp_source_fingerprints,
)


ACTION = "op.prediction_probability.nncp.provider_v1"
PROVIDER = "nncp"
PROFILE = "transformer-stream1-v1"
FRAME_MAGIC = b"NNP1"
FRAME_PROFILE = 1
VALIDATION = (
    ROOT
    / "analysis_outputs/weight_conditioned_grpo_20260917_v1/preflight/nncp_graph_validation.json"
)


def _validate_frame(packet: bytes) -> None:
    packet = bytes(packet)
    if len(packet) < 24 or packet[:4] != FRAME_MAGIC:
        raise ValueError("invalid NNP1 frame")
    profile = struct.unpack_from("<I", packet, 4)[0]
    raw_size = struct.unpack_from("<Q", packet, 8)[0]
    body_size = struct.unpack_from("<I", packet, 20)[0]
    if profile != FRAME_PROFILE:
        raise ValueError("unsupported NNP1 streaming profile")
    if raw_size > 4096 or body_size != len(packet) - 24:
        raise ValueError("invalid NNP1 frame lengths")


def _run(mode: str, payload: bytes) -> bytes:
    result = subprocess.run(
        [*nncp_command(), mode], input=bytes(payload), capture_output=True, check=False,
        timeout=180,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def forward(payload: bytes, action: str = ACTION) -> bytes:
    if action != ACTION:
        raise ValueError("NNP1 action identity does not match A110")
    packet = _run("encode", payload)
    _validate_frame(packet)
    return packet


def inverse(packet: bytes, *, expected_action: str = ACTION) -> bytes:
    if expected_action != ACTION:
        raise ValueError("NNP1 action identity does not match A110")
    _validate_frame(packet)
    return _run("decode", packet)


def source_paths() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/test_nncp_entropy_graph.py",
        ROOT / "scripts/validate_nncp_entropy_graph.py",
        ROOT / "scripts/typed_graph_runtime.py",
        ROOT / "scripts/unified_action_runtime.py",
        ROOT / "scripts/weight_conditioned_graph_task_v1.py",
        ROOT / "scripts/weight_conditioned_operator_gate_v1.py",
    ]


def source_fingerprints() -> dict[str, str]:
    result = dict(nncp_source_fingerprints())
    result.update(
        (str(path.relative_to(ROOT)), sha256(path)) for path in source_paths()
    )
    return dict(sorted(result.items()))


def availability() -> dict[str, Any]:
    """Read-only graph capability check; never builds or initializes NNCP."""
    standalone = nncp_availability()
    try:
        receipt = json.loads(VALIDATION.read_text())
    except (OSError, ValueError):
        receipt = {}
    driver = Path(standalone.get("driver", DRIVER))
    try:
        current = (
            standalone.get("available") is True
            and standalone.get("validated_current_build") is True
            and driver.is_file()
            and receipt.get("result") == "passed"
            and receipt.get("action") == ACTION
            and receipt.get("frame") == "NNP1"
            and receipt.get("profile") == PROFILE
            and receipt.get("source_fingerprints") == source_fingerprints()
            and receipt.get("driver_sha256") == sha256(driver)
        )
    except OSError:
        current = False
    return {
        "action": ACTION,
        "provider": PROVIDER,
        "frame": "NNP1",
        "profile": PROFILE,
        "driver": str(driver),
        "standalone": standalone,
        "complete_reversible_path": bool(current),
        "cold_decode_validated": bool(current),
        "graph_eligible": bool(current),
        "reason": (
            "current NNP1 driver and typed-graph cold-decode receipt match"
            if current
            else "A110 requires current standalone and typed-graph NNP1 validation receipts"
        ),
    }
