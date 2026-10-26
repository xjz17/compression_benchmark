#!/usr/bin/env python3
"""Native complete entropy paths for causal PPMd/DMC/PAQ/cmix providers.

Python only selects the C++ executable and graph action. PEC1 encoding,
decoding, model updates, probability quantization and framing stay in C++.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "cpp/build/prediction_entropy_codec_cli"
VALIDATION = ROOT / "analysis_outputs/weight_conditioned_grpo_20260917_v1/preflight/prediction_entropy_validation.json"
CMIX_VALIDATION = (
    ROOT
    / "analysis_outputs/weight_conditioned_grpo_20260917_v1/preflight/"
      "cmix_prediction_entropy_validation.json"
)
CMIX_RECEIPT_SCHEMA = "cmix-prediction-entropy-validation-v1"

ACTION_TO_PROVIDER = {
    "op.prediction_probability.ppm.provider_v1": "ppmd",
    "op.prediction_probability.dmc.provider_v1": "dmc",
    "op.prediction_probability.paq.provider_v1": "paq",
    "op.prediction_probability.cmix.provider_v1": "cmix",
}
PROVIDER_TO_ACTION = {value: key for key, value in ACTION_TO_PROVIDER.items()}
PROVIDER_CODES = {"ppmd": 1, "dmc": 2, "paq": 3, "cmix": 4}
CODE_TO_PROVIDER = {value: key for key, value in PROVIDER_CODES.items()}


def _binary() -> Path:
    override = os.environ.get("COMPRESSION_BENCHMARK_PREDICTION_ENTROPY_CODEC")
    result = Path(override).expanduser().resolve() if override else BINARY
    if not result.is_file():
        raise RuntimeError(f"missing C++ prediction entropy binary {result}; build prediction_entropy_codec_cli")
    return result


def _provider(action_or_provider: str) -> str:
    result = ACTION_TO_PROVIDER.get(action_or_provider, action_or_provider)
    if result not in PROVIDER_CODES:
        raise ValueError("provider must be ppmd, dmc, paq, or cmix")
    return result


def _bounded_integer(name: str, value: int, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def frame_provider(packet: bytes) -> str:
    packet = bytes(packet)
    if len(packet) < 64 or packet[:4] != b"PEC1" or packet[4] != 1:
        raise ValueError("invalid PEC1 frame")
    try:
        return CODE_TO_PROVIDER[packet[5]]
    except KeyError as error:
        raise ValueError("unknown PEC1 provider") from error


def _run(arguments: list[str], payload: bytes) -> bytes:
    result = subprocess.run([str(_binary()), *arguments], input=bytes(payload),
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def forward(payload: bytes, action_or_provider: str, *,
            ppmd_order: int = 6, ppmd_memory_mib: int = 16,
            paq_level: int = 0, allow_large_memory: bool = False) -> bytes:
    provider = _provider(action_or_provider)
    arguments = ["encode", provider]
    if provider == "ppmd":
        arguments += ["--ppmd-order", str(_bounded_integer("ppmd_order", ppmd_order, 2, 64)),
                      "--ppmd-memory-mib", str(_bounded_integer(
                          "ppmd_memory_mib", ppmd_memory_mib, 4, 256))]
    elif provider == "paq":
        arguments += ["--paq-level", str(_bounded_integer("paq_level", paq_level, 0, 3))]
    if type(allow_large_memory) is not bool:
        raise ValueError("allow_large_memory must be boolean")
    if allow_large_memory:
        arguments.append("--allow-large-memory")
    packet = _run(arguments, payload)
    if frame_provider(packet) != provider:
        raise RuntimeError("native PEC1 encoder returned the wrong provider")
    return packet


def inverse(packet: bytes, *, expected_action_or_provider: str | None = None,
            allow_large_memory: bool = False) -> bytes:
    provider = frame_provider(packet)
    if expected_action_or_provider is not None and provider != _provider(expected_action_or_provider):
        raise ValueError("PEC1 provider does not match graph action")
    arguments = ["decode"]
    if allow_large_memory:
        arguments.append("--allow-large-memory")
    return _run(arguments, packet)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> list[Path]:
    from prediction_probability import native_sources, CMIX
    return [
        ROOT / "cpp/CMakeLists.txt",
        ROOT / "cpp/src/prediction_probability_model.hpp",
        ROOT / "cpp/src/prediction_entropy_codec_cli.cpp",
        Path(__file__),
        ROOT / "scripts/test_prediction_entropy_codec.py",
        ROOT / "scripts/validate_cmix_prediction_entropy.py",
        ROOT / "scripts/typed_graph_runtime.py",
        ROOT / "scripts/unified_action_runtime.py",
        ROOT / "scripts/weight_conditioned_graph_task_v1.py",
        *native_sources(),
        CMIX / "src/models/paq8.cpp",
        CMIX / "src/models/ppmd.cpp",
    ]


def source_fingerprints() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(source_paths()))}


def cmix_receipt_status() -> dict[str, Any]:
    """Validate the separate full-cmix resource/cold-decode receipt.

    No cmix model is initialized here.  Eligibility requires the ordinary PEC1
    validation receipt to pin this exact receipt and both receipts to match the
    current binary and source fingerprints.
    """
    binary = Path(os.environ.get("COMPRESSION_BENCHMARK_PREDICTION_ENTROPY_CODEC", BINARY))
    try:
        base = json.loads(VALIDATION.read_text())
    except (OSError, ValueError):
        base = {}
    try:
        current_sources = source_fingerprints()
        binary_sha256 = sha256(binary) if binary.is_file() else None
    except OSError:
        current_sources = {}
        binary_sha256 = None
    relative = base.get("cmix_probe_receipt")
    receipt_path: Path | None = None
    if isinstance(relative, str) and relative:
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                candidate = Path()
            if candidate == CMIX_VALIDATION.resolve():
                receipt_path = candidate
    try:
        receipt_bytes = receipt_path.read_bytes() if receipt_path is not None else b""
        receipt = json.loads(receipt_bytes) if receipt_bytes else {}
    except (OSError, ValueError):
        receipt_bytes = b""
        receipt = {}
    expected_receipt_sha = base.get("cmix_probe_sha256")
    actual_receipt_sha = hashlib.sha256(receipt_bytes).hexdigest() if receipt_bytes else None
    action = PROVIDER_TO_ACTION["cmix"]
    encode = receipt.get("encode") if isinstance(receipt, dict) else None
    cold = receipt.get("cold_decode") if isinstance(receipt, dict) else None
    current = bool(
        binary.is_file()
        and base.get("result") == "passed"
        and base.get("source_fingerprints") == current_sources
        and base.get("binary_sha256") == binary_sha256
        and "cmix" in set(base.get("validated_providers", ()))
        and isinstance(expected_receipt_sha, str)
        and len(expected_receipt_sha) == 64
        and actual_receipt_sha == expected_receipt_sha
        and receipt.get("schema") == CMIX_RECEIPT_SCHEMA
        and receipt.get("result") == "passed"
        and receipt.get("provider") == "cmix"
        and receipt.get("action") == action
        and receipt.get("frame") == "PEC1"
        and receipt.get("complete_reversible_path") is True
        and receipt.get("cold_decode_independent_process") is True
        and receipt.get("explicit_large_memory_opt_in") is True
        and receipt.get("single_process_sequential") is True
        and receipt.get("binary_sha256") == binary_sha256
        and receipt.get("source_fingerprints") == current_sources
        and isinstance(receipt.get("input_size"), int)
        and not isinstance(receipt.get("input_size"), bool)
        and receipt.get("input_size") > 0
        and isinstance(receipt.get("input_sha256"), str)
        and len(receipt.get("input_sha256")) == 64
        and isinstance(receipt.get("frame_size"), int)
        and not isinstance(receipt.get("frame_size"), bool)
        and receipt.get("frame_size") >= 64
        and isinstance(receipt.get("frame_sha256"), str)
        and len(receipt.get("frame_sha256")) == 64
        and isinstance(encode, dict) and encode.get("returncode") == 0
        and isinstance(cold, dict) and cold.get("returncode") == 0
    )
    return {
        "current": current,
        "action": action,
        "provider": "cmix",
        "frame": "PEC1",
        "receipt": str(receipt_path) if receipt_path is not None else None,
        "receipt_sha256": actual_receipt_sha,
        "binary_sha256": binary_sha256,
        "frame_size": receipt.get("frame_size") if isinstance(receipt, dict) else None,
        "frame_sha256": receipt.get("frame_sha256") if isinstance(receipt, dict) else None,
        "cold_decode_independent_process": (
            receipt.get("cold_decode_independent_process") is True
            if isinstance(receipt, dict) else False
        ),
        "reason": (
            "current full-cmix PEC1 binary/source/action receipt proves independent cold decode"
            if current else
            "cmix requires a pinned current full-cmix PEC1 large-memory cold-decode receipt"
        ),
    }


def availability() -> dict[str, Any]:
    """Read-only validated capability; never builds or initializes a model."""
    binary = Path(os.environ.get("COMPRESSION_BENCHMARK_PREDICTION_ENTROPY_CODEC", BINARY))
    try:
        receipt = json.loads(VALIDATION.read_text())
    except (OSError, ValueError):
        receipt = {}
    try:
        current = (binary.is_file()
                   and receipt.get("result") == "passed"
                   and receipt.get("source_fingerprints") == source_fingerprints()
                   and receipt.get("binary_sha256") == sha256(binary))
    except OSError:
        current = False
    validated = set(receipt.get("validated_providers", ())) if current else set()
    cmix = cmix_receipt_status()
    if not cmix["current"]:
        validated.discard("cmix")
    return {
        "binary": str(binary),
        "binary_exists": binary.is_file(),
        "validated_current_build": current,
        "providers": {
            provider: {
                "complete_reversible_path": provider in validated,
                "cold_decode_validated": provider in validated,
                "graph_eligible": provider in validated,
                "large_memory_opt_in": provider == "cmix",
                **({"large_memory_receipt": cmix} if provider == "cmix" else {}),
            }
            for provider in PROVIDER_CODES
        },
        "reason": ("current PEC1 binary and validation receipt match"
                   if current else "PEC1 requires a current binary and validation receipt"),
    }
