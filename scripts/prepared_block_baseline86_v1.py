#!/usr/bin/env python3
"""Auditable 86-label baselines over verified prepared numeric blocks.

This module deliberately separates the *protocol* from the compression
implementation.  The current project registry supplies the 86 public labels,
their strongest registered pipelines, I/O profiles, and parameters.  An
injected executor supplies the actual current encode and independent cold
decode operations.  Consequently, a label is never reported as measured when
the executor cannot establish a complete, self-contained, bit-exact frame.

The 86 labels are a research catalog, not 86 independent upstream-native
libraries.  Every saved row carries one of these implementation classes:

``native_or_current_implementation``
    A current project execution path is registered.  This is not an upstream
    provenance claim and still requires a successful run receipt.
``declared_proxy_alias_or_composite``
    The current registry explicitly shares, aliases, or composes another
    implementation for the public label.
``unsupported``
    No eligible registered variant or executor support exists for this block.

Metrics follow ``PROJECT_CONSTRAINTS.md``: the complete frame size and both
elapsed times are divided by the original valid numeric bytes.  The missing
bitmap is included in the complete frame but excluded from ``B_raw``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

import prepare_weighted_grpo_timeseries_v1 as preparation
import primitive_operator_action_runtime as operator_runtime


SCHEMA = "prepared-block-baseline86-v1"
CATALOG_SCHEMA = "prepared-block-baseline86-catalog-v1"
EXECUTION_REQUEST_SCHEMA = "prepared-block-baseline86-execution-request-v1"
IMPLEMENTATION_NATIVE_CURRENT = "native_or_current_implementation"
IMPLEMENTATION_PROXY = "declared_proxy_alias_or_composite"
IMPLEMENTATION_UNSUPPORTED = "unsupported"
IMPLEMENTATION_CLASSES = frozenset({
    IMPLEMENTATION_NATIVE_CURRENT,
    IMPLEMENTATION_PROXY,
    IMPLEMENTATION_UNSUPPORTED,
})


class BaselineContractError(RuntimeError):
    """The registry, prepared data, executor, or result violated the protocol."""


# Methods explicitly replaced by current exact/native registrations in
# benchmark_86_exact_action_pipelines().  Native-LZ methods are read from the
# runtime's EXACT_NATIVE_METHOD_KEYS below; this set names the additional
# typed/current overrides in that function.  Chimp is intentionally absent:
# the runtime documents that its public label reuses Chimp128 byte-identically.
_CURRENT_TYPED_OVERRIDES = frozenset({
    "Huffman", "Arithmetic", "ANS", "Dict.", "RLE", "2DIFF", "SVB",
    "Simple8b", "RLBE", "Sprintz", "DBP", "Gorilla", "Chimp128",
    "Pongo", "BOS", "SPDP", "FPC", "PFPC", "Elf", "Elf+", "Camel",
    "FPZIP", "LECO", "ALP", "REGER", "BUFF", "CORAD", "FastPFor",
    "Brotli", "Zstd", "Blosc", "Blosc2", "Bzip2", "gzip", "Zlib",
    "XZ", "LZ4", "Snappy",
})
_DECLARED_SHARED_LABELS = {
    "Chimp": (
        "the current registry intentionally routes Chimp through the same "
        "byte-identical TSFile Chimp implementation as Chimp128"
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BaselineContractError("metadata contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "json_record") and callable(value.json_record):
        return _json_safe(value.json_record())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    raise BaselineContractError(f"cannot serialize metadata type {type(value).__name__}")


@dataclass(frozen=True)
class LabelRecord:
    label: str
    implementation_class: str
    classification_reason: str
    upstream_native_claim: bool
    registered_pipeline: Any
    io_variant_pipeline: Any
    parameters: Any

    def __post_init__(self) -> None:
        if self.implementation_class not in IMPLEMENTATION_CLASSES:
            raise ValueError("invalid implementation classification")
        if self.upstream_native_claim:
            raise ValueError("v1 never asserts upstream-native identity from a label")


@dataclass(frozen=True)
class CatalogSnapshot:
    registry_version: str
    registry_sha256: str
    labels: tuple[LabelRecord, ...]
    snapshot_sha256: str

    def json_record(self) -> dict[str, Any]:
        return {
            "schema": CATALOG_SCHEMA,
            "registry_version": self.registry_version,
            "registry_sha256": self.registry_sha256,
            "label_count": len(self.labels),
            "labels": [asdict(label) for label in self.labels],
            "snapshot_sha256": self.snapshot_sha256,
            "claim_boundary": (
                "labels denote current registered project executions, aliases, "
                "proxies, or composites; no independent upstream-native86 claim"
            ),
        }


def build_catalog_snapshot() -> CatalogSnapshot:
    """Freeze the current 86-label registry with an explicit claim class."""

    registry = operator_runtime.registry()
    labels = tuple(operator_runtime.BENCHMARK_86_METHODS)
    if len(labels) != 86 or len(set(labels)) != 86:
        raise BaselineContractError("current benchmark catalog is not exactly 86 unique labels")
    exact = registry.get("benchmark_86_exact_action_pipelines")
    io_variants = registry.get("benchmark_86_io_variant_pipelines")
    parameters = registry.get("benchmark_86_action_parameters")
    if not all(isinstance(value, Mapping) for value in (exact, io_variants, parameters)):
        raise BaselineContractError("registry lacks fixed-86 pipeline metadata")
    exact_native = frozenset(operator_runtime.EXACT_NATIVE_METHOD_KEYS)
    current = exact_native | _CURRENT_TYPED_OVERRIDES
    rows: list[LabelRecord] = []
    for label in labels:
        missing = [name for name, value in (
            ("exact pipeline", exact.get(label)),
            ("I/O pipeline", io_variants.get(label)),
            ("parameters", parameters.get(label)),
        ) if value is None]
        if missing:
            implementation_class = IMPLEMENTATION_UNSUPPORTED
            reason = "current registry lacks " + ", ".join(missing)
        elif label in _DECLARED_SHARED_LABELS:
            implementation_class = IMPLEMENTATION_PROXY
            reason = _DECLARED_SHARED_LABELS[label]
        elif label in current:
            implementation_class = IMPLEMENTATION_NATIVE_CURRENT
            reason = (
                "current registry provides an exact/native or current typed execution path; "
                "successful external exact-byte execution is still required"
            )
        else:
            implementation_class = IMPLEMENTATION_PROXY
            reason = (
                "registered fixed-label pipeline is retained as a declared project "
                "proxy/alias/composite and carries no exact upstream implementation claim"
            )
        rows.append(LabelRecord(
            label=label,
            implementation_class=implementation_class,
            classification_reason=reason,
            upstream_native_claim=False,
            registered_pipeline=exact.get(label),
            io_variant_pipeline=io_variants.get(label),
            parameters=parameters.get(label),
        ))
    basis = {
        "schema": CATALOG_SCHEMA,
        "registry_version": registry["policy_registry_version"],
        "registry_sha256": registry["registry_sha256"],
        "labels": [asdict(row) for row in rows],
    }
    digest = _sha256(_canonical_json(basis))
    return CatalogSnapshot(
        registry_version=str(registry["policy_registry_version"]),
        registry_sha256=str(registry["registry_sha256"]),
        labels=tuple(rows),
        snapshot_sha256=digest,
    )


@dataclass(frozen=True)
class PreparedBlock:
    dataset_id: str
    dataset_provenance_sha256: str
    block_id: str
    entity_id: str
    split: str
    column: str
    dtype: str
    row_count: int
    valid_count: int
    block_index: int
    values: bytes
    missing_bitmap: bytes
    original_numeric_bytes: int
    values_sha256: str
    missing_bitmap_sha256: str

    def __post_init__(self) -> None:
        if self.original_numeric_bytes <= 0:
            raise ValueError("baseline blocks require positive B_raw")
        if len(self.values) != self.original_numeric_bytes:
            raise ValueError("B_raw must equal valid numeric payload bytes")
        if _sha256(self.values) != self.values_sha256:
            raise ValueError("values digest mismatch")
        if _sha256(self.missing_bitmap) != self.missing_bitmap_sha256:
            raise ValueError("missing bitmap digest mismatch")
        if not 0 <= self.valid_count <= self.row_count:
            raise ValueError("invalid row/valid counts")


@dataclass(frozen=True)
class LabelExecutionRequest:
    label: str
    implementation_class: str
    classification_reason: str
    registry_version: str
    registry_sha256: str
    catalog_snapshot_sha256: str
    registered_pipeline: tuple[str, ...]
    io_variant_pipeline: tuple[str, ...]
    parameters: tuple[Mapping[str, Any], ...]
    selected_variant: str
    dataset_id: str
    dataset_provenance_sha256: str
    block_id: str
    entity_id: str
    split: str
    column: str
    dtype: str
    row_count: int
    valid_count: int
    block_index: int
    values: bytes
    missing_bitmap: bytes
    original_numeric_bytes: int
    schema: str = EXECUTION_REQUEST_SCHEMA


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    reason: str
    executor_identity: str
    implementation_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class EncodeReceipt:
    complete_frame: bytes
    elapsed_ns: int
    self_contained: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DecodeReceipt:
    values: bytes
    missing_bitmap: bytes
    elapsed_ns: int
    independent_process: bool
    decoder_identity: str
    metadata: Mapping[str, Any]


class BaselineExecutor(Protocol):
    """Execution boundary for native/current fixed-label implementations."""

    def support(self, request: LabelExecutionRequest) -> SupportDecision: ...
    def encode(self, request: LabelExecutionRequest) -> EncodeReceipt: ...
    def decode(self, request: LabelExecutionRequest, frame: bytes) -> DecodeReceipt: ...


def _logical_type(dtype: str) -> str:
    if dtype.startswith("int") or dtype.startswith("uint"):
        return "int"
    if dtype.startswith("float"):
        return "float"
    return "bytes"


def _select_variant(value: Any, dtype: str) -> tuple[str, Any] | None:
    if not isinstance(value, Mapping):
        return "shared", value
    logical = _logical_type(dtype)
    for key in (dtype, logical, "bytes"):
        if key in value:
            return key, value[key]
    # Decimal float variants require an audited max-point value that prepared
    # manifests intentionally do not infer.  They remain unsupported here.
    return None


def _as_pipeline(value: Any, description: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BaselineContractError(f"{description} must be a non-empty action sequence")
    if any(not isinstance(item, str) or not item for item in value):
        raise BaselineContractError(f"{description} contains an invalid action name")
    return tuple(value)


def _as_parameters(value: Any, count: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise BaselineContractError("registered parameters do not match pipeline depth")
    if any(not isinstance(item, Mapping) for item in value):
        raise BaselineContractError("registered action parameters must be objects")
    return tuple(dict(item) for item in value)


def make_execution_request(
    catalog: CatalogSnapshot, label: LabelRecord, block: PreparedBlock,
) -> tuple[LabelExecutionRequest | None, str | None]:
    if label.implementation_class == IMPLEMENTATION_UNSUPPORTED:
        return None, label.classification_reason
    exact_variant = _select_variant(label.registered_pipeline, block.dtype)
    io_variant = _select_variant(label.io_variant_pipeline, block.dtype)
    parameter_variant = _select_variant(label.parameters, block.dtype)
    if exact_variant is None or io_variant is None or parameter_variant is None:
        return None, f"no audited registered variant for dtype {block.dtype}"
    exact_key, exact_value = exact_variant
    io_key, io_value = io_variant
    parameter_key, parameter_value = parameter_variant
    keys = {key for key in (exact_key, io_key, parameter_key) if key != "shared"}
    if len(keys) > 1:
        return None, (
            "registered exact/I/O/parameter variants disagree: "
            f"{exact_key}, {io_key}, {parameter_key}"
        )
    exact_pipeline = _as_pipeline(exact_value, "registered exact pipeline")
    io_pipeline = _as_pipeline(io_value, "registered I/O pipeline")
    params = _as_parameters(parameter_value, len(io_pipeline))
    return LabelExecutionRequest(
        label=label.label,
        implementation_class=label.implementation_class,
        classification_reason=label.classification_reason,
        registry_version=catalog.registry_version,
        registry_sha256=catalog.registry_sha256,
        catalog_snapshot_sha256=catalog.snapshot_sha256,
        registered_pipeline=exact_pipeline,
        io_variant_pipeline=io_pipeline,
        parameters=params,
        selected_variant=(next(iter(keys)) if keys else "shared"),
        dataset_id=block.dataset_id,
        dataset_provenance_sha256=block.dataset_provenance_sha256,
        block_id=block.block_id,
        entity_id=block.entity_id,
        split=block.split,
        column=block.column,
        dtype=block.dtype,
        row_count=block.row_count,
        valid_count=block.valid_count,
        block_index=block.block_index,
        values=block.values,
        missing_bitmap=block.missing_bitmap,
        original_numeric_bytes=block.original_numeric_bytes,
    ), None


def _validate_encode(receipt: EncodeReceipt) -> None:
    if not isinstance(receipt, EncodeReceipt):
        raise BaselineContractError("executor encode did not return EncodeReceipt")
    if not receipt.self_contained:
        raise BaselineContractError("executor did not certify a self-contained frame")
    if not isinstance(receipt.complete_frame, bytes) or not receipt.complete_frame:
        raise BaselineContractError("executor returned an empty/non-byte complete frame")
    if isinstance(receipt.elapsed_ns, bool) or receipt.elapsed_ns < 0:
        raise BaselineContractError("invalid compression elapsed_ns")


def _validate_decode(
    receipt: DecodeReceipt, request: LabelExecutionRequest,
) -> None:
    if not isinstance(receipt, DecodeReceipt):
        raise BaselineContractError("executor decode did not return DecodeReceipt")
    if not receipt.independent_process:
        raise BaselineContractError("decode did not establish an independent cold process")
    if not receipt.decoder_identity:
        raise BaselineContractError("decode receipt lacks decoder identity")
    if isinstance(receipt.elapsed_ns, bool) or receipt.elapsed_ns < 0:
        raise BaselineContractError("invalid decompression elapsed_ns")
    if receipt.values != request.values:
        raise BaselineContractError("decoded numeric bytes are not bit-exact")
    if receipt.missing_bitmap != request.missing_bitmap:
        raise BaselineContractError("decoded missing bitmap differs")


def _write_complete_frame_artifact(
    root: Path, request: LabelExecutionRequest, frame: bytes,
) -> dict[str, Any]:
    """Persist one content-addressed baseline frame for later cold audit."""
    root = root.expanduser().resolve()
    digest = _sha256(frame)
    artifact_id = _sha256(_canonical_json({
        "dataset_id": request.dataset_id,
        "dataset_provenance_sha256": request.dataset_provenance_sha256,
        "block_id": request.block_id,
        "label": request.label,
        "catalog_snapshot_sha256": request.catalog_snapshot_sha256,
        "complete_frame_sha256": digest,
    }))
    destination = (root / request.dataset_id / request.label / (artifact_id + ".bin")).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise BaselineContractError("baseline artifact path leaves artifact root") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(frame)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    if destination.stat().st_size != len(frame) or _sha256_file(destination) != digest:
        raise BaselineContractError("persisted baseline frame failed hash verification")
    return {
        "relative_path": destination.relative_to(root).as_posix(),
        "absolute_path": str(destination),
        "bytes": len(frame), "sha256": digest,
        "content": "self-contained complete frame reopened after atomic persistence",
    }


def benchmark_request(
    executor: BaselineExecutor,
    request: LabelExecutionRequest,
    *,
    warmups: int = 1,
    repeats: int = 5,
    artifact_writer: Callable[[bytes], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")
    decision = executor.support(request)
    if not isinstance(decision, SupportDecision):
        raise BaselineContractError("executor support did not return SupportDecision")
    if not decision.supported:
        return {
            "status": "unsupported",
            "reason": decision.reason,
            "executor_identity": decision.executor_identity,
            "implementation_evidence": _json_safe(decision.implementation_evidence),
        }
    frames: list[bytes] = []
    compression_times: list[int] = []
    decompression_times: list[int] = []
    encode_metadata: list[Any] = []
    decode_metadata: list[Any] = []
    for iteration in range(warmups + repeats):
        encoded = executor.encode(request)
        _validate_encode(encoded)
        decoded = executor.decode(request, encoded.complete_frame)
        _validate_decode(decoded, request)
        if iteration >= warmups:
            frames.append(encoded.complete_frame)
            compression_times.append(encoded.elapsed_ns)
            decompression_times.append(decoded.elapsed_ns)
            encode_metadata.append(_json_safe(encoded.metadata))
            decode_metadata.append(_json_safe(decoded.metadata))
    if any(frame != frames[0] for frame in frames[1:]):
        raise BaselineContractError("complete frame is not deterministic across repeats")
    complete_bytes = len(frames[0])
    compression_ns = int(statistics.median(compression_times))
    decompression_ns = int(statistics.median(decompression_times))
    raw = request.original_numeric_bytes
    return {
        "status": "measured",
        "executor_identity": decision.executor_identity,
        "implementation_evidence": _json_safe(decision.implementation_evidence),
        "complete_compressed_bytes": complete_bytes,
        "complete_frame_sha256": _sha256(frames[0]),
        "complete_frame_artifact": (
            _json_safe(artifact_writer(frames[0]))
            if artifact_writer is not None else None
        ),
        "original_numeric_bytes": raw,
        "missing_bitmap_bytes_in_complete_frame_contract": len(request.missing_bitmap),
        "rho_complete_bytes_per_original_numeric_byte": complete_bytes / raw,
        "compression_elapsed_ns": compression_ns,
        "decompression_elapsed_ns": decompression_ns,
        "E_compression_ns_per_original_numeric_byte": compression_ns / raw,
        "D_decompression_ns_per_original_numeric_byte": decompression_ns / raw,
        "compression_elapsed_ns_repeats": compression_times,
        "decompression_elapsed_ns_repeats": decompression_times,
        "warmups": warmups,
        "repeats": repeats,
        "self_contained_frame": True,
        "independent_cold_decode": True,
        "bit_exact_values_and_missing_bitmap": True,
        "encode_metadata_repeats": encode_metadata,
        "decode_metadata_repeats": decode_metadata,
    }


def benchmark_block(
    executor: BaselineExecutor,
    catalog: CatalogSnapshot,
    block: PreparedBlock,
    *,
    warmups: int = 1,
    repeats: int = 5,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label in catalog.labels:
        base = {
            "label": label.label,
            "implementation_class": label.implementation_class,
            "classification_reason": label.classification_reason,
            "upstream_native_claim": False,
        }
        try:
            request, reason = make_execution_request(catalog, label, block)
            if request is None:
                row = {**base, "status": "unsupported", "reason": reason}
            else:
                row = {
                    **base,
                    "selected_variant": request.selected_variant,
                    "registered_pipeline": list(request.registered_pipeline),
                    "io_variant_pipeline": list(request.io_variant_pipeline),
                    "parameters": _json_safe(request.parameters),
                    **benchmark_request(
                        executor, request, warmups=warmups, repeats=repeats,
                        artifact_writer=(
                            (lambda frame, request=request:
                             _write_complete_frame_artifact(
                                 artifact_root, request, frame
                             ))
                            if artifact_root is not None else None
                        ),
                    ),
                }
        except Exception as error:  # keep all per-label failures in the receipt
            row = {
                **base,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
            failures.append({
                "label": label.label,
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            })
        results.append(row)
    if len(results) != 86 or {row["label"] for row in results} != {
        label.label for label in catalog.labels
    }:
        raise BaselineContractError("block receipt does not cover all 86 labels")
    return {
        "schema": SCHEMA,
        "dataset_id": block.dataset_id,
        "dataset_provenance_sha256": block.dataset_provenance_sha256,
        "block_id": block.block_id,
        "entity_id": block.entity_id,
        "split": block.split,
        "column": block.column,
        "dtype": block.dtype,
        "block_index": block.block_index,
        "row_count": block.row_count,
        "valid_count": block.valid_count,
        "original_numeric_bytes": block.original_numeric_bytes,
        "missing_bitmap_bytes": len(block.missing_bitmap),
        "catalog_snapshot_sha256": catalog.snapshot_sha256,
        "registry_version": catalog.registry_version,
        "registry_sha256": catalog.registry_sha256,
        "metric_contract": {
            "rho": "complete_compressed_bytes / original_numeric_bytes",
            "E": "compression_elapsed_ns / original_numeric_bytes",
            "D": "decompression_elapsed_ns / original_numeric_bytes",
            "direction": "all lower is better",
            "missing_bitmap": "included in complete frame but excluded from B_raw",
        },
        "label_results": results,
        "failures": failures,
        "counts": {
            "labels": len(results),
            "measured": sum(row["status"] == "measured" for row in results),
            "unsupported": sum(row["status"] == "unsupported" for row in results),
            "failed": sum(row["status"] == "failed" for row in results),
        },
    }


def load_prepared_blocks(
    manifest_path: Path,
    *,
    splits: Sequence[str] = ("validation",),
    verify_source: bool = True,
) -> tuple[dict[str, Any], list[PreparedBlock]]:
    resolved = manifest_path.resolve()
    root = resolved if resolved.is_dir() else resolved.parent
    path = root / "dataset_manifest.json" if resolved.is_dir() else resolved
    if path.name != "dataset_manifest.json":
        raise BaselineContractError("prepared manifest must be named dataset_manifest.json")
    wanted = frozenset(splits)
    if not wanted or not wanted.issubset({"train", "validation", "test"}):
        raise ValueError("at least one valid split is required")
    if "test" in wanted and wanted != {"test"}:
        raise BaselineContractError("test cannot be mixed with train/validation splits")
    try:
        verification = preparation.verify_prepared_dataset(
            root, verify_source=verify_source,
            expected_role="test_candidate" if wanted == {"test"} else None,
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            preparation.PreparationError) as error:
        raise BaselineContractError(f"prepared dataset verification failed: {path}") from error
    blocks: list[PreparedBlock] = []
    for row in manifest["blocks"]:
        if row["split"] not in wanted:
            continue
        values = (root / row["values_path"]).read_bytes()
        bitmap = (root / row["missing_bitmap_path"]).read_bytes()
        blocks.append(PreparedBlock(
            dataset_id=str(manifest["dataset_id"]),
            dataset_provenance_sha256=str(manifest["provenance_sha256"]),
            block_id=str(row["block_id"]), entity_id=str(row["entity_id"]),
            split=str(row["split"]), column=str(row["column"]),
            dtype=str(row["dtype"]), row_count=int(row["row_count"]),
            valid_count=int(row["valid_count"]),
            block_index=int(row["block_index"]), values=values,
            missing_bitmap=bitmap,
            original_numeric_bytes=int(row["original_numeric_bytes"]),
            values_sha256=str(row["values_sha256"]),
            missing_bitmap_sha256=str(row["missing_bitmap_sha256"]),
        ))
    return {
        "manifest_path": str(path),
        "verification": verification,
        "selected_splits": sorted(wanted),
        "selected_blocks": len(blocks),
    }, blocks


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_safe(value), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def run_prepared_baselines(
    executor: BaselineExecutor,
    manifests: Sequence[Path],
    output_dir: Path,
    *,
    splits: Sequence[str] = ("validation",),
    verify_source: bool = True,
    warmups: int = 1,
    repeats: int = 5,
    persist_complete_frames: bool = True,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise BaselineContractError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    artifact_root = (output / "complete_frames").resolve()
    catalog = build_catalog_snapshot()
    _atomic_json(output / "catalog.json", catalog.json_record())
    datasets: list[dict[str, Any]] = []
    block_receipts: list[dict[str, Any]] = []
    for manifest in manifests:
        dataset, blocks = load_prepared_blocks(
            manifest, splits=splits, verify_source=verify_source
        )
        datasets.append(dataset)
        for block in blocks:
            receipt = benchmark_block(
                executor, catalog, block, warmups=warmups, repeats=repeats,
                artifact_root=(artifact_root if persist_complete_frames else None),
            )
            relative = Path("blocks") / block.dataset_id / f"{block.block_id}.json"
            _atomic_json(output / relative, receipt)
            block_receipts.append({
                "dataset_id": block.dataset_id,
                "block_id": block.block_id,
                "receipt_path": relative.as_posix(),
                "counts": receipt["counts"],
                "failures": receipt["failures"],
            })
    summary = {
        "schema": SCHEMA,
        "status": "complete_with_explicit_failures",
        "catalog_snapshot_sha256": catalog.snapshot_sha256,
        "registry_version": catalog.registry_version,
        "registry_sha256": catalog.registry_sha256,
        "datasets": datasets,
        "blocks": block_receipts,
        "counts": {
            "datasets": len(datasets),
            "blocks": len(block_receipts),
            "label_block_pairs": sum(row["counts"]["labels"] for row in block_receipts),
            "measured": sum(row["counts"]["measured"] for row in block_receipts),
            "unsupported": sum(row["counts"]["unsupported"] for row in block_receipts),
            "failed": sum(row["counts"]["failed"] for row in block_receipts),
        },
        "timing_protocol": {
            "warmups": warmups,
            "repeats": repeats,
            "aggregate": "median elapsed nanoseconds",
            "encode": "executor-reported complete-frame encoding boundary",
            "decode": "executor-reported independent cold-process complete-frame decoding boundary",
        },
        "complete_frame_artifacts": {
            "enabled": persist_complete_frames,
            "root": str(artifact_root) if persist_complete_frames else None,
            "measured_artifacts": sum(
                row["counts"]["measured"] for row in block_receipts
            ) if persist_complete_frames else 0,
        },
        "claim_boundary": (
            "only status=measured rows are benchmark results; unsupported and failed "
            "rows are retained and no old result is relabeled"
        ),
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def load_executor(specification: str) -> BaselineExecutor:
    if ":" not in specification:
        raise ValueError("executor must use module:factory syntax")
    module_name, attribute_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    factory: Callable[[], BaselineExecutor] = getattr(module, attribute_name)
    return factory()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor", required=True, help="module:factory")
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-source-verification", action="store_true")
    arguments = parser.parse_args(argv)
    summary = run_prepared_baselines(
        load_executor(arguments.executor), arguments.manifest, arguments.output,
        splits=tuple(arguments.split or ("validation",)),
        verify_source=not arguments.skip_source_verification,
        warmups=arguments.warmups, repeats=arguments.repeats,
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
