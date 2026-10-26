#!/usr/bin/env python3
"""Current source-backed native executor for prepared-block baseline86.

The public 86-label catalog is broader than the set of implementations that
can presently be executed under the project's C++-only measurement contract.
This bridge therefore admits only method labels with an explicitly enumerated
current C++ path.  Everything else is returned as unsupported by ``support``;
there is no fallback to a Python compression kernel and no reuse of historical
benchmark rows.

The outer ``B86F1`` frame is self-contained.  A compact binary header records
the public method label, input dtype and byte length, registry/catalog/pipeline
fingerprints, the complete missing bitmap, and the current method payload.  It
does not repeat long action names in every block.  A trailing SHA-256 protects
the whole frame.  ``decode`` launches a fresh Python dispatcher process which
parses the frame and invokes only the registered C++ inverse path; the original
request bytes are not passed to that process.

Timing intentionally includes outer-frame construction/parsing plus all C++
CLI process launch and IPC used by the current action runtime.  It excludes the
launch and IPC of the independent cold-decoder dispatcher itself.  The caller
normalizes these elapsed nanoseconds by the original numeric bytes.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Mapping

import prepared_block_baseline86_v1 as protocol
import primitive_operator_action_runtime as runtime


EXECUTOR_IDENTITY = "prepared-block-baseline86-native-executor-v1"
FRAME_MAGIC = b"B86F1\x00\x00\x00"
DECODE_MAGIC = b"B86D1\x00\x00\x00"
FRAME_SCHEMA = "prepared-block-baseline86-complete-frame-v1"
DECODE_SCHEMA = "prepared-block-baseline86-cold-decode-v1"
MAX_HEADER_BYTES = 1 << 20
MAX_COLD_DECODE_SECONDS = 300
_FRAME_FIXED = struct.Struct("<HHHHQQQQQ")
_FRAME_DIGEST_BYTES = 32 * 3
_MAX_FRAME_TEXT_BYTES = 4096


# These labels use the current C++ staged native-LZ implementation selected by
# EXACT_NATIVE_METHOD_KEYS.  The label remains a project implementation label;
# this bridge does not claim identity with an independent upstream library.
_NATIVE_LZ_LABELS = frozenset(runtime.EXACT_NATIVE_METHOD_KEYS)

# These additional methods have an audited current C++ forward and inverse
# path in primitive_operator_action_runtime.  They operate on the source bytes
# without widening a prepared column's dtype.
_BYTE_EXACT_LABELS = frozenset({
    "Huffman", "Arithmetic", "ANS",
    "LZ4", "Snappy",
    "Brotli", "Zstd", "Bzip2", "gzip", "Zlib", "XZ",
})

_TYPED_BOTH64_LABELS = frozenset({
    "Chimp", "Chimp128", "Pongo", "Gorilla", "RLBE", "Sprintz",
    "Dict.", "Camel", "CORAD", "BUFF",
})
_TYPED_INT64_LABELS = frozenset({
    "2DIFF", "BOS", "REGER", "SVB", "FastPFor", "Simple8b", "RLE",
    "DBP", "LECO",
})
_TYPED_FLOAT64_LABELS = frozenset({
    "ALP", "Elf", "Elf+", "FPC", "PFPC", "SPDP", "FPZIP",
})
_BLOSC_LABELS = frozenset({"Blosc", "Blosc2"})

SUPPORTED_LABELS = (
    _NATIVE_LZ_LABELS | _BYTE_EXACT_LABELS | _TYPED_BOTH64_LABELS
    | _TYPED_INT64_LABELS | _TYPED_FLOAT64_LABELS | _BLOSC_LABELS
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@functools.lru_cache(maxsize=None)
def _file_sha256(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary(name: str) -> Path:
    return runtime.REPO_ROOT / "cpp" / "build" / name


def _required_binaries(label: str) -> tuple[Path, ...]:
    if label in _NATIVE_LZ_LABELS:
        return (_binary("native_lz_action_stages_cli"),)
    if label in {"Huffman", "Arithmetic", "ANS"}:
        return (_binary("general_codec_reference_stages_cli"),)
    if label == "LZ4":
        return (_binary("lz4_exact_stages_cli"),)
    if label == "Snappy":
        return (_binary("snappy_exact_stages_cli"),)
    typed = {
        "Chimp": ("chimp128_exact_stages_cli",),
        "Chimp128": ("chimp128_exact_stages_cli",),
        "Pongo": ("pongo_exact_stages_cli",),
        "Gorilla": ("gorilla_exact_stages_cli",),
        "2DIFF": ("ts2diff_exact_stages_cli",),
        "BOS": ("bos_exact_stages_cli",),
        "REGER": ("learning_codec_exact_stages_cli",),
        "SVB": ("svb_exact_stages_cli",),
        "FastPFor": ("fastpfor_exact_stages_cli",),
        "Simple8b": ("simple8b_exact_stages_cli",),
        "RLE": ("rle_int64_action_cli",),
        "RLBE": ("p04_exact_stages_cli",),
        "Sprintz": ("p04_exact_stages_cli",),
        "Dict.": ("dictionary64_action_cli",),
        "DBP": ("dbp_exact_stages_cli", "unsigned_leb128_cli"),
        "ALP": ("learning_codec_exact_stages_cli",),
        "LECO": ("learning_codec_exact_stages_cli",),
        "Camel": ("camel_exact_stages_cli",),
        "Elf": ("elf64_exact_stages_cli",),
        "Elf+": ("elf64_exact_stages_cli",),
        "FPC": ("fpc_exact_stages_cli",),
        "PFPC": ("pfpc_exact_stages_cli",),
        "SPDP": ("spdp_exact_stages_cli",),
        "CORAD": ("corad_lossless_cpp_cli",),
        "BUFF": ("buff_native_exact_stages_cli",),
        "FPZIP": ("fpzip_exact_stages_cli",),
    }
    if label in typed:
        return tuple(_binary(name) for name in typed[label])
    base = (
        _binary("general_codec_reference_stages_cli"),
        _binary("general_codec_residual_stages_cli"),
    )
    if label in {"gzip", "Zlib"}:
        return base + (_binary("deflate_container_stages_cli"),)
    if label == "XZ":
        return base + (_binary("xz_container_stage_cli"),)
    if label in {"Brotli", "Zstd", "Bzip2", "Blosc", "Blosc2"}:
        if label == "Bzip2":
            return base + (_binary("optimal_rle_cli"),)
        return base
    return ()


def _implementation_kind(label: str) -> str:
    if label in _NATIVE_LZ_LABELS:
        return "project_cpp_native_lz_registered_label_path"
    if label in {"Huffman", "Arithmetic", "ANS"}:
        return "project_cpp_entropy_operator"
    if label in {"LZ4", "Snappy"}:
        return "current_cpp_exact_method_path"
    if label in (_TYPED_BOTH64_LABELS | _TYPED_INT64_LABELS | _TYPED_FLOAT64_LABELS):
        return "current_cpp_exact_typed_method_path"
    if label in _BLOSC_LABELS:
        return "current_cpp_composed_typed_general_method_path"
    return "current_cpp_composed_general_method_path"


@functools.lru_cache(maxsize=1)
def _current_registry_identity() -> tuple[str, str]:
    current = runtime.registry()
    return (
        str(current["policy_registry_version"]),
        str(current["registry_sha256"]),
    )


@functools.lru_cache(maxsize=1)
def _current_catalog_snapshot_sha256() -> str:
    return protocol.build_catalog_snapshot().snapshot_sha256


def _logical_type(label: str, dtype: str) -> str:
    if label in (_NATIVE_LZ_LABELS | _BYTE_EXACT_LABELS):
        return "bytes"
    if label in _BLOSC_LABELS:
        if dtype in {"float32", "float64", "int64"}:
            return dtype
        return "bytes"
    if dtype == "int64" and label in (_TYPED_BOTH64_LABELS | _TYPED_INT64_LABELS):
        return "int"
    if dtype == "float64" and label in (_TYPED_BOTH64_LABELS | _TYPED_FLOAT64_LABELS):
        return "float"
    raise ValueError(f"{label} has no audited current path for prepared dtype {dtype}")


def _applicability_reason(
    label: str, dtype: str, values: bytes
) -> str | None:
    """Return a known data-dependent admission failure without encoding."""

    if label in {"Elf", "Elf+"} and dtype == "float64":
        for (bits,) in struct.iter_unpack("<Q", values):
            exponent = (bits >> 52) & 0x7ff
            fraction = bits & ((1 << 52) - 1)
            if exponent == 0x7ff and fraction:
                return (
                    "current Elf64 wire reserves its NaN marker and cannot restore "
                    "arbitrary source NaN payloads"
                )
    if label == "Camel" and dtype == "int64":
        integers = [value for (value,) in struct.iter_unpack("<q", values)]
        if any(abs(current - previous) > 0xffff
               for previous, current in zip(integers, integers[1:])):
            return "current Camel integer wire accepts only adjacent deltas up to 65535"
    return None


def _encode_method_payload(label: str, values: bytes, dtype: str) -> bytes:
    # General methods receive bytes deliberately so prepared int8/int32 and
    # float32 payloads are not widened.  Typed methods are admitted only for a
    # registered source width and receive that logical type explicitly.
    logical_type = _logical_type(label, dtype)
    if label == "Dict.":
        return runtime._cpp_action_run("dictionary64_action_cli", "encode", values)
    if label in _BLOSC_LABELS:
        pipelines = (
            runtime.BLOSC_EXACT_ACTIONS_BY_KIND
            if label == "Blosc" else runtime.BLOSC2_EXACT_ACTIONS_BY_KIND
        )
        return runtime.encode_exact_general_residual_action_pipeline(
            values, pipelines[logical_type]
        )
    return runtime.encode_exact_benchmark_method(
        values, label, logical_type=logical_type
    )


def _reverse_reference_chain(data: bytes, commands: tuple[str, ...]) -> bytes:
    current = bytes(data)
    for command in commands:
        current = runtime._run_general_codec_reference_stage_cli(command, current)
    return current


def _decode_method_payload(
    label: str, payload: bytes, expected_size: int, dtype: str
) -> bytes:
    logical_type = _logical_type(label, dtype)
    if label in _NATIVE_LZ_LABELS:
        return runtime.decompress_native_lz_family(
            payload, runtime.EXACT_NATIVE_METHOD_KEYS[label], expected_size
        )
    if label == "Huffman":
        return runtime.huffman_decode(payload)
    if label == "Arithmetic":
        return runtime.arithmetic_decode(payload)
    if label == "ANS":
        return runtime.ans_decode(payload)
    if label == "LZ4":
        return runtime.decode_exact_lz4_action_pipeline(payload)
    if label == "Snappy":
        return runtime.decode_exact_snappy_action_pipeline(payload)
    if label in {"Chimp", "Chimp128"}:
        return runtime.decode_exact_chimp128_action_pipeline(payload, logical_type)
    if label == "Pongo":
        return runtime.decode_exact_pongo_action_pipeline(payload)
    if label == "Gorilla":
        return runtime.decode_exact_gorilla_action_pipeline(payload, logical_type)
    if label == "2DIFF":
        return runtime.decode_exact_ts2diff_action_pipeline(payload, logical_type)
    if label == "BOS":
        return runtime.decode_exact_bos64_action_pipeline(payload, logical_type)
    if label == "REGER":
        return runtime.decode_exact_reger_action_pipeline(payload)
    if label == "SVB":
        return runtime.decode_exact_svb_action_pipeline(payload, logical_type)
    if label == "FastPFor":
        return runtime.decode_exact_fastpfor_action_pipeline(
            payload, logical_type, expected_size // 8
        )
    if label == "Simple8b":
        return runtime.decode_exact_simple8b_action_pipeline(payload, logical_type)
    if label == "RLE":
        return runtime._run_rle_int64_action_cli("decode", payload)
    if label == "RLBE":
        delta_packet = runtime.optimal_rle.inverse(
            payload, runtime.optimal_rle.RLBE_VARIANT
        )
        return runtime._run_p04_stage_cli("rlbe-delta-inverse", delta_packet)
    if label == "Sprintz":
        return runtime.decode_exact_p04_action_pipeline(
            payload, label, logical_type
        )
    if label == "Dict.":
        return runtime._cpp_action_run("dictionary64_action_cli", "decode", payload)
    if label == "DBP":
        return runtime.decode_exact_dbp_action_pipeline(payload, logical_type)
    if label == "ALP":
        return runtime.decode_exact_alp_action_pipeline(payload)
    if label == "LECO":
        return runtime.decode_exact_leco_action_pipeline(payload)
    if label == "Camel":
        return runtime.decode_exact_camel_action_pipeline(payload, logical_type)
    if label in {"Elf", "Elf+"}:
        return runtime.elf64_exact_stages.decode(payload, label)
    if label == "FPC":
        return runtime.decode_exact_fpc_action_pipeline(payload)
    if label == "PFPC":
        return runtime.decode_exact_pfpc_action_pipeline(payload)
    if label == "SPDP":
        return runtime.decode_exact_spdp_action_pipeline(payload, expected_size)
    if label == "CORAD":
        kind = "int64" if logical_type == "int" else "float64"
        return runtime.corad_lossless_stages.decode(payload, kind)
    if label == "BUFF":
        return runtime._run_exact_stage_cli(
            runtime.BUFF_NATIVE_STAGE_CLI, "BUFF native", "decompose", payload
        )
    if label == "FPZIP":
        return runtime.decode_exact_fpzip_action_pipeline(payload)
    if label in _BLOSC_LABELS:
        type_size = {"bytes": 1, "float32": 4, "float64": 8, "int64": 8}[
            logical_type
        ]
        prefix = runtime._run_general_codec_residual_stage_cli(
            f"{'blosc1' if label == 'Blosc' else 'blosc2'}-inverse-{type_size}",
            payload,
        )
        return _reverse_reference_chain(
            prefix, ("token-inverse", "lz-inverse", "shuffle-inverse")
        )
    if label == "Brotli":
        current = runtime._run_general_codec_residual_stage_cli(
            "brotli-inverse", payload
        )
        return _reverse_reference_chain(
            current, ("huffman-inverse", "lz-inverse", "dictionary-inverse")
        )
    if label == "Zstd":
        current = runtime._run_general_codec_residual_stage_cli(
            "zstd-inverse", payload
        )
        return _reverse_reference_chain(
            current,
            ("rans-inverse", "huffman-inverse", "parse-inverse", "lz-inverse"),
        )
    if label == "Bzip2":
        current = runtime._run_general_codec_residual_stage_cli(
            "bzip2-optimal-inverse", payload
        )
        current = _reverse_reference_chain(current, ("huffman-inverse",))
        current = runtime.optimal_rle.inverse(
            current, runtime.optimal_rle.BYTES_VARIANT
        )
        return _reverse_reference_chain(current, ("mtf-inverse", "bwt-inverse"))
    if label == "gzip":
        current = runtime._run_deflate_container_stage_cli("gzip-inverse", payload)
        current = runtime._run_general_codec_residual_stage_cli(
            "gzip-inverse", current
        )
        return _reverse_reference_chain(
            current, ("huffman-inverse", "lz-inverse")
        )
    if label == "Zlib":
        current = runtime._run_deflate_container_stage_cli("zlib-inverse", payload)
        current = runtime._run_general_codec_residual_stage_cli(
            "zlib-inverse", current
        )
        return _reverse_reference_chain(
            current, ("huffman-inverse", "parse-inverse", "lz-inverse")
        )
    if label == "XZ":
        current = runtime._run_xz_container_stage_cli("inverse", payload)
        current = runtime._run_general_codec_residual_stage_cli("xz-inverse", current)
        return _reverse_reference_chain(
            current, ("range-inverse", "parse-inverse", "lz-inverse")
        )
    raise ValueError(f"no admitted native decoder for label {label!r}")


def _pipeline_contract_sha256(request: protocol.LabelExecutionRequest) -> str:
    return _sha256(_canonical_json({
        "registered_pipeline": list(request.registered_pipeline),
        "io_variant_pipeline": list(request.io_variant_pipeline),
        "parameters": [dict(item) for item in request.parameters],
    }))


def _digest_bytes(value: str, description: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{description} must be a 64-character SHA-256")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{description} is not hexadecimal") from error


def _make_complete_frame(request: protocol.LabelExecutionRequest) -> bytes:
    payload = _encode_method_payload(request.label, request.values, request.dtype)
    fields = tuple(
        value.encode("utf-8") for value in (
            request.label, request.dtype, request.registry_version,
            request.selected_variant,
        )
    )
    if any(len(value) > _MAX_FRAME_TEXT_BYTES for value in fields):
        raise ValueError("baseline86 complete-frame text field is too large")
    label, dtype, registry_version, selected_variant = fields
    body = (
        FRAME_MAGIC
        + _FRAME_FIXED.pack(
            len(label), len(dtype), len(registry_version), len(selected_variant),
            request.row_count, request.valid_count, len(request.values),
            len(request.missing_bitmap), len(payload),
        )
        + _digest_bytes(request.registry_sha256, "registry SHA-256")
        + _digest_bytes(request.catalog_snapshot_sha256, "catalog SHA-256")
        + bytes.fromhex(_pipeline_contract_sha256(request))
        + label + dtype + registry_version + selected_variant
        + request.missing_bitmap
        + payload
    )
    return body + hashlib.sha256(body).digest()


def _parse_complete_frame(frame: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    raw = bytes(frame)
    minimum = len(FRAME_MAGIC) + _FRAME_FIXED.size + _FRAME_DIGEST_BYTES + 32
    if len(raw) < minimum or raw[:len(FRAME_MAGIC)] != FRAME_MAGIC:
        raise ValueError("invalid baseline86 complete-frame magic")
    expected_digest = raw[-32:]
    body = raw[:-32]
    if hashlib.sha256(body).digest() != expected_digest:
        raise ValueError("baseline86 complete-frame checksum mismatch")
    cursor = len(FRAME_MAGIC)
    (
        label_size, dtype_size, registry_version_size, selected_variant_size,
        row_count, valid_count, values_size, bitmap_size, payload_size,
    ) = _FRAME_FIXED.unpack_from(body, cursor)
    cursor += _FRAME_FIXED.size
    if any(value > _MAX_FRAME_TEXT_BYTES for value in (
        label_size, dtype_size, registry_version_size, selected_variant_size
    )):
        raise ValueError("baseline86 complete-frame text field exceeds limit")
    if values_size <= 0 or valid_count > row_count:
        raise ValueError("invalid baseline86 complete-frame lengths")
    if len(body) - cursor < _FRAME_DIGEST_BYTES:
        raise ValueError("truncated baseline86 complete-frame fingerprints")
    registry_sha256 = body[cursor:cursor + 32].hex()
    catalog_sha256 = body[cursor + 32:cursor + 64].hex()
    pipeline_sha256 = body[cursor + 64:cursor + 96].hex()
    cursor += _FRAME_DIGEST_BYTES
    text_total = label_size + dtype_size + registry_version_size + selected_variant_size
    if cursor + text_total + bitmap_size + payload_size != len(body):
        raise ValueError("baseline86 complete-frame length mismatch")
    text_values: list[str] = []
    try:
        for size in (label_size, dtype_size, registry_version_size, selected_variant_size):
            text_values.append(body[cursor:cursor + size].decode("utf-8"))
            cursor += size
    except UnicodeDecodeError as error:
        raise ValueError("invalid baseline86 complete-frame text") from error
    label, dtype, registry_version, selected_variant = text_values
    if label not in SUPPORTED_LABELS:
        raise ValueError("complete frame names an unsupported method label")
    bitmap = body[cursor:cursor + bitmap_size]
    payload = body[cursor + bitmap_size:]
    header = {
        "schema": FRAME_SCHEMA,
        "executor_identity": EXECUTOR_IDENTITY,
        "label": label,
        "implementation_kind": _implementation_kind(label),
        "upstream_native_claim": False,
        "dtype": dtype,
        "row_count": row_count,
        "valid_count": valid_count,
        "original_numeric_bytes": values_size,
        "values_bytes": values_size,
        "missing_bitmap_bytes": bitmap_size,
        "method_payload_bytes": payload_size,
        "registry_version": registry_version,
        "registry_sha256": registry_sha256,
        "catalog_snapshot_sha256": catalog_sha256,
        "pipeline_contract_sha256": pipeline_sha256,
        "selected_variant": selected_variant,
        "compact_header_bytes": cursor,
    }
    return header, bitmap, payload


def _cold_decode_frame(frame: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    header, bitmap, payload = _parse_complete_frame(frame)
    registry_version, registry_sha256 = _current_registry_identity()
    if (
        header.get("registry_version") != registry_version
        or header.get("registry_sha256") != registry_sha256
    ):
        raise ValueError("complete frame registry identity is stale")
    if header.get("catalog_snapshot_sha256") != _current_catalog_snapshot_sha256():
        raise ValueError("complete frame catalog snapshot is stale")
    values = _decode_method_payload(
        str(header["label"]), payload, int(header["values_bytes"]), str(header["dtype"])
    )
    if len(values) != int(header["values_bytes"]):
        raise ValueError("native decoder restored an unexpected byte count")
    return values, bitmap, header


def _pack_decode_response(
    values: bytes, bitmap: bytes, *, elapsed_ns: int, header: Mapping[str, Any]
) -> bytes:
    metadata = _canonical_json({
        "schema": DECODE_SCHEMA,
        "elapsed_ns": int(elapsed_ns),
        "values_bytes": len(values),
        "missing_bitmap_bytes": len(bitmap),
        "decoder_identity": EXECUTOR_IDENTITY + ":fresh-process",
        "label": header["label"],
        "implementation_kind": header["implementation_kind"],
        "timing_scope": (
            "complete-frame parse/checksum plus current C++ inverse including "
            "C++ CLI launch/IPC; outer cold-worker launch/IPC excluded"
        ),
    })
    return (
        DECODE_MAGIC + struct.pack("<I", len(metadata)) + metadata
        + values + bitmap
    )


def _unpack_decode_response(payload: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    raw = bytes(payload)
    if len(raw) < len(DECODE_MAGIC) + 4 or raw[:len(DECODE_MAGIC)] != DECODE_MAGIC:
        raise ValueError("invalid cold-decoder response magic")
    size = struct.unpack_from("<I", raw, len(DECODE_MAGIC))[0]
    start = len(DECODE_MAGIC) + 4
    end = start + size
    if size > MAX_HEADER_BYTES or end > len(raw):
        raise ValueError("invalid cold-decoder response header")
    try:
        metadata = json.loads(raw[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid cold-decoder response JSON") from error
    if not isinstance(metadata, dict) or metadata.get("schema") != DECODE_SCHEMA:
        raise ValueError("unexpected cold-decoder response schema")
    values_size = int(metadata.get("values_bytes", -1))
    bitmap_size = int(metadata.get("missing_bitmap_bytes", -1))
    if values_size < 0 or bitmap_size < 0 or end + values_size + bitmap_size != len(raw):
        raise ValueError("cold-decoder response length mismatch")
    return (
        raw[end:end + values_size],
        raw[end + values_size:],
        metadata,
    )


class NativeBaseline86Executor:
    """Execute the admitted current C++ subset of the public 86 labels."""

    identity = EXECUTOR_IDENTITY

    def support(
        self, request: protocol.LabelExecutionRequest
    ) -> protocol.SupportDecision:
        if request.label not in SUPPORTED_LABELS:
            return protocol.SupportDecision(
                supported=False,
                reason=(
                    "no current C++-only complete-frame forward/inverse bridge "
                    "is admitted for this label; no proxy or historical result is substituted"
                ),
                executor_identity=self.identity,
                implementation_evidence={
                    "label": request.label,
                    "catalog_implementation_class": request.implementation_class,
                    "upstream_native_claim": False,
                    "supported_label_count": len(SUPPORTED_LABELS),
                },
            )
        try:
            _logical_type(request.label, request.dtype)
        except ValueError as error:
            return protocol.SupportDecision(
                supported=False,
                reason=str(error),
                executor_identity=self.identity,
                implementation_evidence={
                    "label": request.label,
                    "dtype": request.dtype,
                    "catalog_implementation_class": request.implementation_class,
                    "upstream_native_claim": False,
                },
            )
        applicability = _applicability_reason(
            request.label, request.dtype, request.values
        )
        if applicability is not None:
            return protocol.SupportDecision(
                supported=False,
                reason=applicability,
                executor_identity=self.identity,
                implementation_evidence={
                    "label": request.label,
                    "dtype": request.dtype,
                    "catalog_implementation_class": request.implementation_class,
                    "known_data_dependent_limit": applicability,
                    "upstream_native_claim": False,
                },
            )
        registry_version, registry_sha256 = _current_registry_identity()
        if (
            request.registry_version != registry_version
            or request.registry_sha256 != registry_sha256
        ):
            return protocol.SupportDecision(
                supported=False,
                reason="execution request registry identity is stale",
                executor_identity=self.identity,
                implementation_evidence={
                    "request_registry_version": request.registry_version,
                    "request_registry_sha256": request.registry_sha256,
                    "current_registry_version": registry_version,
                    "current_registry_sha256": registry_sha256,
                },
            )
        current_catalog_sha256 = _current_catalog_snapshot_sha256()
        if request.catalog_snapshot_sha256 != current_catalog_sha256:
            return protocol.SupportDecision(
                supported=False,
                reason="execution request catalog snapshot is stale",
                executor_identity=self.identity,
                implementation_evidence={
                    "request_catalog_snapshot_sha256": request.catalog_snapshot_sha256,
                    "current_catalog_snapshot_sha256": current_catalog_sha256,
                    "upstream_native_claim": False,
                },
            )
        required = _required_binaries(request.label)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return protocol.SupportDecision(
                supported=False,
                reason="required current C++ executable is not built",
                executor_identity=self.identity,
                implementation_evidence={
                    "label": request.label,
                    "missing_executables": missing,
                    "upstream_native_claim": False,
                },
            )
        return protocol.SupportDecision(
            supported=True,
            reason="current source-backed C++ forward and inverse are available",
            executor_identity=self.identity,
            implementation_evidence={
                "label": request.label,
                "catalog_implementation_class": request.implementation_class,
                "implementation_kind": _implementation_kind(request.label),
                "upstream_native_claim": False,
                "registered_pipeline": list(request.registered_pipeline),
                "required_executables": [
                    {"path": str(path), "sha256": _file_sha256(str(path))}
                    for path in required
                ],
                "bridge_source": str(Path(__file__).resolve()),
                "bridge_source_sha256": _file_sha256(str(Path(__file__).resolve())),
                "timing_scope": (
                    "complete outer frame plus current C++ CLI launch/IPC; "
                    "cold-worker launch/IPC excluded"
                ),
            },
        )

    def encode(
        self, request: protocol.LabelExecutionRequest
    ) -> protocol.EncodeReceipt:
        decision = self.support(request)
        if not decision.supported:
            raise protocol.BaselineContractError(decision.reason)
        started = time.perf_counter_ns()
        frame = _make_complete_frame(request)
        elapsed = time.perf_counter_ns() - started
        return protocol.EncodeReceipt(
            complete_frame=frame,
            elapsed_ns=elapsed,
            self_contained=True,
            metadata={
                "schema": FRAME_SCHEMA,
                "label": request.label,
                "implementation_kind": _implementation_kind(request.label),
                "upstream_native_claim": False,
                "timing_scope": (
                    "method encode plus complete-frame construction/checksum, including "
                    "current C++ CLI launch/IPC"
                ),
            },
        )

    def decode(
        self, request: protocol.LabelExecutionRequest, frame: bytes
    ) -> protocol.DecodeReceipt:
        environment = dict(os.environ)
        environment.setdefault("PYTHONHASHSEED", "0")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--cold-decode"],
            input=bytes(frame), capture_output=True, check=False,
            timeout=MAX_COLD_DECODE_SECONDS, env=environment,
        )
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise protocol.BaselineContractError(
                f"independent cold decoder failed: {detail[:2000]}"
            )
        values, bitmap, metadata = _unpack_decode_response(completed.stdout)
        if metadata.get("label") != request.label:
            raise protocol.BaselineContractError(
                "independent cold decoder restored a different method label"
            )
        return protocol.DecodeReceipt(
            values=values,
            missing_bitmap=bitmap,
            elapsed_ns=int(metadata["elapsed_ns"]),
            independent_process=True,
            decoder_identity=str(metadata["decoder_identity"]),
            metadata=metadata,
        )


def native_executor() -> NativeBaseline86Executor:
    """Factory used by prepared_block_baseline86_v1.py --executor."""

    return NativeBaseline86Executor()


def _cold_decode_main() -> int:
    frame = sys.stdin.buffer.read()
    try:
        started = time.perf_counter_ns()
        values, bitmap, header = _cold_decode_frame(frame)
        elapsed = time.perf_counter_ns() - started
        sys.stdout.buffer.write(_pack_decode_response(
            values, bitmap, elapsed_ns=elapsed, header=header
        ))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--cold-decode"]:
        return _cold_decode_main()
    print(
        json.dumps({
            "executor_identity": EXECUTOR_IDENTITY,
            "supported_label_count": len(SUPPORTED_LABELS),
            "supported_labels": sorted(SUPPORTED_LABELS),
            "claim_boundary": (
                "current source-backed project paths only; no 86-upstream-native claim"
            ),
        }, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
