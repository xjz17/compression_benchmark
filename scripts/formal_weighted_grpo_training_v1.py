#!/usr/bin/env python3
"""Formal, resumable training orchestration for weight-conditioned GRPO.

This module is deliberately an orchestrator, not a compression implementation.
An injected task factory owns native operator execution, complete framing,
measurement, and independent lossless decoding.  The orchestrator verifies the
prepared train/validation artifacts and frozen zlib6 time-reference table,
admits a run only after four explicit evidence gates pass, and trains one shared
critic-free two-layer 128-dimensional policy.

The three campaign seeds are trained independently.  Each seed has at most 16
hours of cumulative training wall clock.  Validation freezes a policy snapshot
and evaluates its selected complete representations with the exact physical
objective.  Test data is neither accepted nor inspected here, and the delivery
checkpoint is selected only by validation score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import base64
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

import prepare_weighted_grpo_timeseries_v1 as preparation
import weight_conditioned_grpo_v1 as grpo
import weight_conditioned_search_v1 as search
import weighted_grpo_experiment_v1 as experiment


SCHEMA = "formal-weight-conditioned-grpo-training-v1"
GATE_BUNDLE_SCHEMA = "weighted-grpo-formal-training-gates-v1"
GATE_RECEIPT_SCHEMA = "weighted-grpo-training-gate-receipt-v1"
RESUME_SCHEMA = "weighted-grpo-resume-checkpoint-v1"
CAMPAIGN_SCHEMA = "weighted-grpo-three-seed-campaign-v1"
TASK_REQUEST_SCHEMA = "weighted-grpo-training-task-factory-request-v1"
FORMAL_SEEDS = (0, 1, 2)
FORMAL_SEED_BUDGET_SECONDS = 16 * 60 * 60
MAX_MEASUREMENT_WORKERS = 4
FIRST_BLOCK_EVALUATION_BUDGET = 64
CONTINUATION_EVALUATION_BUDGET = 16
MAX_CONTINUATION_EDITS = 2

REQUIRED_GATE_ASSERTIONS: Mapping[str, tuple[str, ...]] = {
    "mechanisms": (
        "all_67_retained_mechanisms_current",
        "source_fingerprints_match",
        "cold_bit_exact_roundtrip",
        "no_python_compression_fallback",
    ),
    "graph": (
        "typed_graph_correctness_passed",
        "independent_cold_decode_passed",
        "depth10_nodes64_streams32_enforced",
        "exactly_stop_concat_continuous_split_specials",
    ),
    "data": (
        "prepared_manifests_verified",
        "chronological_train_validation_split",
        "timestamp_contracts_resolved",
        "source_family_leakage_excluded",
    ),
    "reference": (
        "zlib6_table_frozen",
        "training_blocks_only",
        "native_complete_frame_timings",
        "positive_group_references",
    ),
}


class TrainingAdmissionError(RuntimeError):
    """Formal training evidence or immutable identities do not match."""


class TrainingExecutionError(RuntimeError):
    """The injected task factory or measured rollout violated the contract."""


@dataclass(frozen=True)
class TrainingBlock:
    dataset_id: str
    dataset_manifest_path: str
    prepared_root: str
    provenance_sha256: str
    block_id: str
    entity_id: str
    split: str
    column: str
    dtype: str
    block_index: int
    values_path: str
    missing_bitmap_path: str
    values_sha256: str
    missing_bitmap_sha256: str
    original_numeric_bytes: int
    prepared_block_bytes: int
    compression_reference_ns_per_byte: float
    decompression_reference_ns_per_byte: float
    block_fingerprint: str


@dataclass(frozen=True)
class TrainingCorpus:
    train_blocks: tuple[TrainingBlock, ...]
    validation_blocks: tuple[TrainingBlock, ...]
    dataset_provenance_sha256: tuple[tuple[str, str], ...]
    calibration_sha256: str
    corpus_sha256: str


@dataclass(frozen=True)
class _BlockSequence:
    sequence_id: str
    dataset_id: str
    entity_id: str
    split: str
    column: str
    blocks: tuple[TrainingBlock, ...]


@dataclass(frozen=True)
class GateEvidence:
    source_fingerprint: str
    action_registry_fingerprint: str
    receipt_sha256: tuple[tuple[str, str], ...]
    bundle_sha256: str


@dataclass(frozen=True)
class TrainingTaskFactoryRequest:
    stage: str
    dataset_id: str
    dataset_manifest_path: str
    prepared_root: str
    provenance_sha256: str
    block_id: str
    entity_id: str
    split: str
    column: str
    dtype: str
    block_index: int
    series_block_ordinal: int
    values_path: str
    missing_bitmap_path: str
    values_sha256: str
    missing_bitmap_sha256: str
    original_numeric_bytes: int
    prepared_block_bytes: int
    compression_reference_ns_per_byte: float
    decompression_reference_ns_per_byte: float
    weight: tuple[float, float, float]
    seed: int
    iteration: int
    group_index: int
    episode_index: int
    previous_recipe: tuple[int, ...]
    previous_recipe_source_block_id: str | None
    evaluation_budget: int
    max_local_edits: int | None
    reuse_previous_recipe: bool
    start_recipe_mode: str
    expected_action_registry_fingerprint: str
    schema: str = TASK_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.stage not in {"identity", "train", "validation"}:
            raise ValueError("unknown training task-factory stage")
        if self.split not in {"train", "validation"}:
            raise ValueError("formal training cannot construct test tasks")
        if self.stage == "train" and self.split != "train":
            raise ValueError("train stage requires a train block")
        if self.stage == "validation" and self.split != "validation":
            raise ValueError("validation stage requires a validation block")
        grpo.validate_weight(self.weight)
        if min(self.original_numeric_bytes, self.prepared_block_bytes) <= 0:
            raise ValueError("task request requires positive byte counts")
        if self.series_block_ordinal < 0:
            raise ValueError("series block ordinal must be non-negative")
        if any(isinstance(action, bool) or not isinstance(action, (int, np.integer))
               or int(action) < 0 for action in self.previous_recipe):
            raise ValueError("previous recipe must contain non-negative integer action IDs")
        if self.series_block_ordinal == 0:
            if (self.evaluation_budget != FIRST_BLOCK_EVALUATION_BUDGET
                    or self.max_local_edits is not None
                    or self.reuse_previous_recipe
                    or self.previous_recipe
                    or self.previous_recipe_source_block_id is not None
                    or self.start_recipe_mode != "training_solution_library_only"):
                raise ValueError("the first series block must start from the training-only library")
        else:
            if (self.evaluation_budget != CONTINUATION_EVALUATION_BUDGET
                    or self.max_local_edits != MAX_CONTINUATION_EDITS):
                raise ValueError("continuation blocks must reuse the preceding successful recipe")
            if self.previous_recipe_source_block_id is None:
                if (self.previous_recipe or self.reuse_previous_recipe
                        or self.start_recipe_mode != "no_successful_predecessor_empty_recipe"):
                    raise ValueError("missing successful predecessor must be explicit, empty and budgeted")
            elif (not self.reuse_previous_recipe or
                  self.start_recipe_mode != "previous_successful_complete_recipe"):
                raise ValueError("continuation blocks must reuse the preceding successful recipe")
        if not self.expected_action_registry_fingerprint:
            raise ValueError("expected action registry fingerprint is required")

    def json_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["weight"] = list(self.weight)
        return row


class TrainingTaskFactory(Protocol):
    def __call__(self, request: TrainingTaskFactoryRequest) -> grpo.GraphCompressionTask: ...


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, int, int] = FORMAL_SEEDS
    per_seed_budget_seconds: float = FORMAL_SEED_BUDGET_SECONDS
    measurement_workers: int = MAX_MEASUREMENT_WORKERS
    validation_interval: int = 10
    checkpoint_interval: int = 1
    max_iterations: int | None = None
    policy_device: str = "cpu"
    grpo_config: grpo.GRPOConfig = grpo.GRPOConfig(
        hidden_sizes=(128, 128), learning_rate=3.0e-4, clip_ratio=0.2,
        entropy_coefficient=0.01, epochs=4, group_size=8,
        groups_per_iteration=8,
    )

    def __post_init__(self) -> None:
        from weighted_grpo_torch_backend_v1 import normalize_device
        object.__setattr__(self, "policy_device", normalize_device(self.policy_device))
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("exactly three distinct seeds are required")
        if not math.isfinite(self.per_seed_budget_seconds) or self.per_seed_budget_seconds <= 0:
            raise ValueError("per-seed wall-clock budget must be finite and positive")
        if self.per_seed_budget_seconds > FORMAL_SEED_BUDGET_SECONDS:
            raise ValueError("per-seed budget must not exceed 16 hours")
        if not 1 <= self.measurement_workers <= MAX_MEASUREMENT_WORKERS:
            raise ValueError("measurement workers must be in 1..4")
        if self.validation_interval < 1 or self.checkpoint_interval < 1:
            raise ValueError("validation/checkpoint intervals must be positive")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be positive when supplied")
        expected = grpo.GRPOConfig(
            hidden_sizes=(128, 128), learning_rate=3.0e-4, clip_ratio=0.2,
            entropy_coefficient=0.01, epochs=4, group_size=8,
            groups_per_iteration=8,
        )
        if self.grpo_config != expected:
            raise ValueError("formal v1 freezes the requested GRPO hyperparameters")


@dataclass(frozen=True)
class _TaskContract:
    task_id: str
    registry_fingerprint: str
    data_fingerprint: str
    root_fingerprint: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite metadata")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"metadata is not JSON-safe: {type(value).__name__}")


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype="<f8")
    return {
        "dtype": "<f8", "shape": list(array.shape),
        "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _array_from_record(value: Mapping[str, Any]) -> np.ndarray:
    try:
        if value["dtype"] != "<f8":
            raise ValueError("array dtype mismatch")
        shape = tuple(int(part) for part in value["shape"])
        raw = base64.b64decode(str(value["base64"]), validate=True)
        result = np.frombuffer(raw, dtype="<f8").reshape(shape).astype(np.float64, copy=True)
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingAdmissionError("invalid resume array") from error
    if not np.all(np.isfinite(result)):
        raise TrainingAdmissionError("resume array contains non-finite parameters")
    return result


def _block_sort_key(block: TrainingBlock) -> tuple[Any, ...]:
    return (
        block.dataset_id, block.entity_id, block.column, block.split,
        block.block_index, block.block_id,
    )


def _ordered_sequences(blocks: Sequence[TrainingBlock]) -> tuple[_BlockSequence, ...]:
    """Group blocks without exposing any later block to an earlier request."""
    grouped: dict[tuple[str, str, str, str], list[TrainingBlock]] = {}
    for block in blocks:
        key = (block.dataset_id, block.entity_id, block.split, block.column)
        grouped.setdefault(key, []).append(block)
    sequences: list[_BlockSequence] = []
    for key, members in sorted(grouped.items()):
        ordered = tuple(sorted(members, key=lambda item: (item.block_index, item.block_id)))
        source_indices = [item.block_index for item in ordered]
        if len(source_indices) != len(set(source_indices)):
            raise TrainingAdmissionError(
                "a dataset/entity/split/column sequence has duplicate source block_index values"
            )
        sequence_basis = {
            "dataset_id": key[0], "entity_id": key[1], "split": key[2],
            "column": key[3],
            "ordered_block_fingerprints": [item.block_fingerprint for item in ordered],
        }
        sequences.append(_BlockSequence(
            sequence_id=_sha256_bytes(_canonical_json(sequence_basis)),
            dataset_id=key[0], entity_id=key[1], split=key[2], column=key[3],
            blocks=ordered,
        ))
    if not sequences or any(not sequence.blocks for sequence in sequences):
        raise TrainingAdmissionError("at least one non-empty ordered block sequence is required")
    return tuple(sequences)


def _load_corpus(prepared_paths: Sequence[Path], reference_path: Path) -> TrainingCorpus:
    if not prepared_paths:
        raise TrainingAdmissionError("at least one prepared dataset is required")
    datasets = [experiment._load_prepared_dataset(
                    path, verify_source=True, expected_role="train_validation",
                )
                for path in prepared_paths]
    preparation.validate_prepared_cohort_manifests(
        [dataset.manifest for dataset in datasets],
        preparation.load_cohort_manifest(),
        require_complete_train=True,
    )
    reference = experiment._load_reference_table(reference_path)
    blocks: list[TrainingBlock] = []
    for dataset in datasets:
        root = dataset.root.resolve()
        for raw in dataset.manifest.get("blocks", ()):
            split = str(raw.get("split", ""))
            if split not in {"train", "validation"}:
                continue
            e0_bucket, e0, d0 = experiment._reference_for_block(reference, raw)
            del e0_bucket

            def child(field: str) -> str:
                candidate = (root / str(raw[field])).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as error:
                    raise TrainingAdmissionError("prepared payload leaves dataset root") from error
                if not candidate.is_file():
                    raise TrainingAdmissionError(f"missing prepared payload: {candidate}")
                return str(candidate)

            basis = {
                "dataset_id": str(dataset.manifest["dataset_id"]),
                "provenance_sha256": str(dataset.manifest["provenance_sha256"]),
                "block_id": str(raw["block_id"]), "entity_id": str(raw["entity_id"]),
                "split": split, "column": str(raw["column"]), "dtype": str(raw["dtype"]),
                "block_index": int(raw["block_index"]),
                "values_sha256": str(raw["values_sha256"]),
                "missing_bitmap_sha256": str(raw["missing_bitmap_sha256"]),
                "original_numeric_bytes": int(raw["original_numeric_bytes"]),
                "prepared_block_bytes": int(raw["prepared_block_bytes"]),
                "E0": e0, "D0": d0,
            }
            if min(basis["original_numeric_bytes"], basis["prepared_block_bytes"]) <= 0:
                raise TrainingAdmissionError("formal training excludes empty prepared blocks")
            blocks.append(TrainingBlock(
                dataset_id=basis["dataset_id"],
                dataset_manifest_path=str(dataset.manifest_path.resolve()),
                prepared_root=str(root), provenance_sha256=basis["provenance_sha256"],
                block_id=basis["block_id"], entity_id=basis["entity_id"], split=split,
                column=basis["column"], dtype=basis["dtype"],
                block_index=basis["block_index"], values_path=child("values_path"),
                missing_bitmap_path=child("missing_bitmap_path"),
                values_sha256=basis["values_sha256"],
                missing_bitmap_sha256=basis["missing_bitmap_sha256"],
                original_numeric_bytes=basis["original_numeric_bytes"],
                prepared_block_bytes=basis["prepared_block_bytes"],
                compression_reference_ns_per_byte=e0,
                decompression_reference_ns_per_byte=d0,
                block_fingerprint=_sha256_bytes(_canonical_json(basis)),
            ))
    train = tuple(sorted((block for block in blocks if block.split == "train"), key=_block_sort_key))
    validation = tuple(sorted(
        (block for block in blocks if block.split == "validation"), key=_block_sort_key
    ))
    if not train or not validation:
        raise TrainingAdmissionError("verified corpus requires both train and validation blocks")
    provenance = tuple(sorted(
        (str(dataset.manifest["dataset_id"]), str(dataset.manifest["provenance_sha256"]))
        for dataset in datasets
    ))
    if len(dict(provenance)) != len(provenance):
        raise TrainingAdmissionError("prepared dataset IDs must be unique")
    corpus_basis = {
        "dataset_provenance_sha256": [list(row) for row in provenance],
        "calibration_sha256": reference.calibration_sha256,
        "train_block_fingerprints": [block.block_fingerprint for block in train],
        "validation_block_fingerprints": [block.block_fingerprint for block in validation],
    }
    return TrainingCorpus(
        train, validation, provenance, reference.calibration_sha256,
        _sha256_bytes(_canonical_json(corpus_basis)),
    )


def _load_gate_bundle(path: Path, corpus: TrainingCorpus) -> GateEvidence:
    try:
        raw_bytes = path.read_bytes()
        bundle = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingAdmissionError("cannot read formal-training gate bundle") from error
    if not isinstance(bundle, Mapping) or bundle.get("schema") != GATE_BUNDLE_SCHEMA:
        raise TrainingAdmissionError("unsupported formal-training gate bundle")
    receipt_specs = bundle.get("receipts")
    if not isinstance(receipt_specs, Mapping) or set(receipt_specs) != set(REQUIRED_GATE_ASSERTIONS):
        raise TrainingAdmissionError("gate bundle must name exactly four required receipts")
    source_fingerprints: set[str] = set()
    action_fingerprints: set[str] = set()
    receipt_hashes: list[tuple[str, str]] = []
    loaded: dict[str, Mapping[str, Any]] = {}
    for gate, assertions in REQUIRED_GATE_ASSERTIONS.items():
        spec = receipt_specs[gate]
        if not isinstance(spec, Mapping):
            raise TrainingAdmissionError(f"invalid {gate} receipt descriptor")
        receipt_path = (path.resolve().parent / str(spec.get("path", ""))).resolve()
        expected_sha = str(spec.get("sha256", ""))
        if len(expected_sha) != 64 or not receipt_path.is_file():
            raise TrainingAdmissionError(f"missing or unpinned {gate} receipt")
        actual_sha = _sha256_file(receipt_path)
        if actual_sha != expected_sha:
            raise TrainingAdmissionError(f"{gate} receipt fingerprint mismatch")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingAdmissionError(f"invalid {gate} receipt JSON") from error
        if (not isinstance(receipt, Mapping)
                or receipt.get("schema") != GATE_RECEIPT_SCHEMA
                or receipt.get("gate") != gate
                or receipt.get("passed") is not True):
            raise TrainingAdmissionError(f"{gate} receipt does not explicitly pass")
        flags = receipt.get("assertions")
        if (not isinstance(flags, Mapping)
                or any(flags.get(assertion) is not True for assertion in assertions)):
            raise TrainingAdmissionError(f"{gate} receipt lacks required true assertions")
        if gate in {"mechanisms", "graph"}:
            source_fingerprints.add(str(receipt.get("source_fingerprint", "")))
            action_fingerprints.add(str(receipt.get("action_registry_fingerprint", "")))
        loaded[gate] = receipt
        receipt_hashes.append((gate, actual_sha))
    if len(source_fingerprints) != 1 or "" in source_fingerprints:
        raise TrainingAdmissionError("mechanism/graph source fingerprints do not match")
    if len(action_fingerprints) != 1 or "" in action_fingerprints:
        raise TrainingAdmissionError("mechanism/graph action-registry fingerprints do not match")
    data_map = loaded["data"].get("dataset_provenance_sha256")
    expected_data = dict(corpus.dataset_provenance_sha256)
    if not isinstance(data_map, Mapping) or {
        str(key): str(value) for key, value in data_map.items()
    } != expected_data:
        raise TrainingAdmissionError("data gate is not bound to the prepared corpus")
    if loaded["reference"].get("calibration_sha256") != corpus.calibration_sha256:
        raise TrainingAdmissionError("reference gate is not bound to the frozen E0/D0 table")
    try:
        experiment._validate_current_runtime_measurement()
        experiment._validate_formal_measurement_evidence(loaded["graph"])
        experiment._validate_formal_measurement_contract(
            loaded["graph"].get("measurement_protocol_schema"),
            loaded["reference"].get("measurement_contract"),
        )
    except experiment.ExperimentContractError as error:
        raise TrainingAdmissionError(str(error)) from error
    return GateEvidence(
        source_fingerprint=next(iter(source_fingerprints)),
        action_registry_fingerprint=next(iter(action_fingerprints)),
        receipt_sha256=tuple(sorted(receipt_hashes)),
        bundle_sha256=_sha256_bytes(raw_bytes),
    )


def _factory_request(
    block: TrainingBlock, *, stage: str, weight: Sequence[float], seed: int,
    iteration: int, group_index: int, episode_index: int, registry: str,
    series_block_ordinal: int, previous_recipe: Sequence[int] = (),
    previous_recipe_source_block_id: str | None = None,
) -> TrainingTaskFactoryRequest:
    continuation = series_block_ordinal > 0
    return TrainingTaskFactoryRequest(
        stage=stage, dataset_id=block.dataset_id,
        dataset_manifest_path=block.dataset_manifest_path,
        prepared_root=block.prepared_root, provenance_sha256=block.provenance_sha256,
        block_id=block.block_id, entity_id=block.entity_id, split=block.split,
        column=block.column, dtype=block.dtype, block_index=block.block_index,
        series_block_ordinal=series_block_ordinal,
        values_path=block.values_path, missing_bitmap_path=block.missing_bitmap_path,
        values_sha256=block.values_sha256,
        missing_bitmap_sha256=block.missing_bitmap_sha256,
        original_numeric_bytes=block.original_numeric_bytes,
        prepared_block_bytes=block.prepared_block_bytes,
        compression_reference_ns_per_byte=block.compression_reference_ns_per_byte,
        decompression_reference_ns_per_byte=block.decompression_reference_ns_per_byte,
        weight=tuple(float(value) for value in grpo.validate_weight(weight)),
        seed=seed, iteration=iteration, group_index=group_index,
        episode_index=episode_index,
        previous_recipe=tuple(int(action) for action in previous_recipe),
        previous_recipe_source_block_id=previous_recipe_source_block_id,
        evaluation_budget=(
            CONTINUATION_EVALUATION_BUDGET if continuation
            else FIRST_BLOCK_EVALUATION_BUDGET
        ),
        max_local_edits=MAX_CONTINUATION_EDITS if continuation else None,
        reuse_previous_recipe=continuation and previous_recipe_source_block_id is not None,
        start_recipe_mode=(
            "previous_successful_complete_recipe" if continuation
            else "training_solution_library_only"
        ) if not continuation or previous_recipe_source_block_id is not None else (
            "no_successful_predecessor_empty_recipe"
        ),
        expected_action_registry_fingerprint=registry,
    )


def _task_contract(task: grpo.GraphCompressionTask) -> _TaskContract:
    return _TaskContract(
        str(task.task_id), str(task.registry_fingerprint),
        str(task.data_fingerprint), str(task.root_fingerprint),
    )


def _build_policy_identity(
    corpus: TrainingCorpus, factory: TrainingTaskFactory, gate: GateEvidence,
    config: TrainingConfig,
) -> grpo.PolicyIdentity:
    # Identity discovery uses only the chronological first block of each train
    # sequence.  A later block is never fabricated as an independent first
    # block merely to inspect dimensions.
    representatives: list[grpo.GraphCompressionTask] = []
    for sequence in _ordered_sequences(corpus.train_blocks):
        block = sequence.blocks[0]
        task = factory(_factory_request(
            block, stage="identity", weight=(0.4, 0.3, 0.3), seed=0,
            iteration=0, group_index=-1, episode_index=-1,
            registry=gate.action_registry_fingerprint, series_block_ordinal=0,
        ))
        contract = _task_contract(task)
        if contract.registry_fingerprint != gate.action_registry_fingerprint:
            raise TrainingAdmissionError("factory action registry differs from gate evidence")
        representatives.append(task)
    try:
        return grpo._inspect_tasks(representatives, config.grpo_config.hidden_sizes)
    finally:
        for task in representatives:
            close = getattr(task, "close", None)
            if callable(close): close()


def _validate_fresh_task(
    policy: grpo.TwoHiddenMLP | grpo.FrozenPolicy,
    task: grpo.GraphCompressionTask,
    request: TrainingTaskFactoryRequest,
) -> None:
    grpo._validate_task_for_policy(policy, task)
    contract = _task_contract(task)
    if contract.registry_fingerprint != request.expected_action_registry_fingerprint:
        raise TrainingExecutionError("task factory changed the frozen action registry")
    if not contract.data_fingerprint or not contract.root_fingerprint:
        raise TrainingExecutionError("task factory omitted data/root fingerprints")


def _extract_complete_recipe(outcome: grpo.TerminalOutcome) -> tuple[int, ...]:
    graph = outcome.operator_graph
    if not isinstance(graph, Mapping):
        raise TrainingExecutionError(
            "operator_graph must expose recipe_action_ids for block-to-block reuse"
        )
    raw = graph.get("recipe_action_ids")
    if (not isinstance(raw, (list, tuple))
            or any(isinstance(action, bool) or not isinstance(action, (int, np.integer))
                   or int(action) < 0 for action in raw)):
        raise TrainingExecutionError("operator_graph.recipe_action_ids is invalid")
    return tuple(int(action) for action in raw)


@contextmanager
def _single_core_environment() -> Any:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _sequence_rollout_job(
    factory: TrainingTaskFactory, sequence: _BlockSequence,
    weight: Sequence[float], policy: grpo.FrozenPolicy,
    *, seed: int, iteration: int, group_index: int, episode_index: int,
    rollout_seed: int, registry_fingerprint: str,
) -> dict[str, Any]:
    """Roll out one episode in time order, exposing only the current block."""
    rng = np.random.default_rng(rollout_seed)
    previous_recipe: tuple[int, ...] = ()
    previous_source: str | None = None
    rows: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for ordinal, block in enumerate(sequence.blocks):
        request = _factory_request(
            block, stage="train", weight=weight, seed=seed,
            iteration=iteration, group_index=group_index,
            episode_index=episode_index, registry=registry_fingerprint,
            series_block_ordinal=ordinal, previous_recipe=previous_recipe,
            previous_recipe_source_block_id=previous_source,
        )
        # The request contains exactly this block and the preceding successful
        # recipe.  No later block path, payload, statistic, or recipe is passed.
        task = factory(request)
        try:
            _validate_fresh_task(policy, task, request)
            part = grpo._rollout(
                policy, task, np.asarray(request.weight, dtype=np.float64),
                rng, greedy=False, timing=False,
            )
        finally:
            close = getattr(task, "close", None)
            if callable(close): close()
        recipe = _extract_complete_recipe(part["outcome"])
        rows.append({
            "block_id": block.block_id,
            "source_block_index": block.block_index,
            "series_block_ordinal": ordinal,
            "previous_recipe": list(previous_recipe),
            "previous_recipe_source_block_id": previous_source,
            "selected_complete_recipe": list(recipe),
            "evaluation_budget": request.evaluation_budget,
            "max_local_edits": request.max_local_edits,
            "reuse_previous_recipe": request.reuse_previous_recipe,
            "start_recipe_mode": request.start_recipe_mode,
            "future_block_data_visible": False,
            "J": (None if isinstance(part["outcome"], grpo.TerminalFailure) else float(part["objective"])),
            "terminal_status": "failed" if isinstance(part["outcome"], grpo.TerminalFailure) else "ok",
            "training_terminal_cost": float(part["objective"]),
        })
        parts.append(part)
        if not isinstance(part["outcome"], grpo.TerminalFailure):
            previous_recipe = recipe
            previous_source = block.block_id
    objectives = np.asarray([float(part["objective"]) for part in parts], dtype=np.float64)
    if not parts or np.any(~np.isfinite(objectives)):
        raise TrainingExecutionError("sequence rollout did not produce finite complete-block objectives")
    return {
        "features": np.concatenate([part["features"] for part in parts]),
        "legal": np.concatenate([part["legal"] for part in parts]),
        "actions": np.concatenate([part["actions"] for part in parts]),
        "old_logp": np.concatenate([part["old_logp"] for part in parts]),
        "return": -float(np.mean(objectives)),
        "objective": float(np.mean(objectives)),
        "block_objectives": objectives.tolist(),
        "sequence_receipt": rows,
    }


def _training_iteration(
    *, policy: grpo.TwoHiddenMLP, optimizer: grpo.Adam,
    factory: TrainingTaskFactory, corpus: TrainingCorpus,
    rng: np.random.Generator, seed: int, iteration: int,
    config: TrainingConfig, registry_fingerprint: str, progress_callback=None,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    hyper = config.grpo_config
    sampled_weights = grpo.sample_training_weights(
        hyper.groups_per_iteration, rng, hyper.dirichlet_alpha
    )
    sequences = _ordered_sequences(corpus.train_blocks)
    sequence_indices = rng.integers(0, len(sequences), size=hyper.groups_per_iteration)
    frozen = grpo.FrozenPolicy(policy)
    jobs = []
    group_rows: list[dict[str, Any]] = []
    for group_index, ((weight, source), sequence_index) in enumerate(
        zip(sampled_weights, sequence_indices)
    ):
        sequence = sequences[int(sequence_index)]
        seeds = rng.integers(0, np.iinfo(np.int64).max, size=hyper.group_size, dtype=np.int64)
        group_rows.append({
            "group": group_index, "sequence_id": sequence.sequence_id,
            "dataset_id": sequence.dataset_id, "entity_id": sequence.entity_id,
            "column": sequence.column, "split": sequence.split,
            "ordered_block_ids": [block.block_id for block in sequence.blocks],
            "source_block_indices": [block.block_index for block in sequence.blocks],
            "sequence_block_count": len(sequence.blocks),
            "weight": [float(value) for value in weight], "weight_source": source,
        })
        for episode_index, rollout_seed in enumerate(seeds):
            jobs.append((sequence, weight, episode_index, int(rollout_seed), group_index))
    episodes: list[dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=config.measurement_workers) as pool:
        futures = [
            pool.submit(
                _sequence_rollout_job, factory, sequence, weight, frozen,
                seed=seed, iteration=iteration, group_index=group_index,
                episode_index=episode_index, rollout_seed=rollout_seed,
                registry_fingerprint=registry_fingerprint,
            )
            for sequence, weight, episode_index, rollout_seed, group_index in jobs
        ]
        indices = {future: index for index, future in enumerate(futures)}
        for completed_count, future in enumerate(as_completed(futures), 1):
            index = indices[future]
            episodes[index] = future.result()
            if progress_callback is not None:
                progress_callback({"event": "rollout_sequence_complete", "seed": seed,
                    "iteration": iteration, "completed_sequences": completed_count,
                    "total_sequences": len(jobs), "episode_index_in_batch": index,
                    "transitions": int(episodes[index]["actions"].size),
                    "wall_ns": time.perf_counter_ns() - started,
                    "gradient_update_completed": False})
    completed = [episode for episode in episodes if episode is not None]
    if len(completed) != hyper.groups_per_iteration * hyper.group_size:
        raise TrainingExecutionError("incomplete GRPO rollout batch")
    returns = np.asarray([episode["return"] for episode in completed], dtype=np.float64)
    episode_advantages = grpo.group_advantages(returns, hyper.group_size)
    lengths = np.asarray([episode["actions"].size for episode in completed], dtype=np.int64)
    episode_ids = np.repeat(np.arange(len(completed)), lengths)
    batch = {
        "features": np.concatenate([episode["features"] for episode in completed]),
        "legal": np.concatenate([episode["legal"] for episode in completed]),
        "actions": np.concatenate([episode["actions"] for episode in completed]),
        "old_logp": np.concatenate([episode["old_logp"] for episode in completed]),
        "advantages": episode_advantages[episode_ids],
        "token_weights": 1.0 / lengths[episode_ids],
    }
    before = [value.copy() for value in policy.params]
    update_started = time.perf_counter_ns()
    update = grpo._update(policy, optimizer, batch, hyper)
    update_ns = time.perf_counter_ns() - update_started
    if any(not np.all(np.isfinite(value)) for value in policy.params):
        raise FloatingPointError("non-finite GRPO policy parameters")
    for group_index, row in enumerate(group_rows):
        group_returns = returns[
            group_index * hyper.group_size:(group_index + 1) * hyper.group_size
        ]
        row["mean_return"] = float(group_returns.mean())
        row["std_return"] = float(group_returns.std())
        row["episode_sequences"] = [
            completed[index]["sequence_receipt"]
            for index in range(
                group_index * hyper.group_size, (group_index + 1) * hyper.group_size
            )
        ]
    return {
        "iteration": iteration,
        "groups": hyper.groups_per_iteration,
        "group_size": hyper.group_size,
        "episodes": len(completed),
        "transitions": int(sum(lengths)),
        "fixed_weight_groups": sum(row["weight_source"] == "fixed" for row in group_rows),
        "continuous_simplex_groups": sum(
            row["weight_source"] == "dirichlet" for row in group_rows
        ),
        "mean_terminal_reward": float(returns.mean()),
        "min_terminal_reward": float(returns.min()),
        "max_terminal_reward": float(returns.max()),
        "informative_groups": sum(row["std_return"] > 1.0e-8 for row in group_rows),
        "parameter_change_l2": float(math.sqrt(sum(
            float(np.sum((old - new) ** 2)) for old, new in zip(before, policy.params)
        ))),
        "reward_definition": (
            "each_block_terminal_-J_nonterminal_0; "
            "infeasible_candidate_cost_1_plus_a_Lmax_over_Braw_plus_b_plus_c; "
            "sequence_return_negative_macro_mean_block_terminal_cost"
        ),
        "group_advantage": "same_ordered_sequence_weight_group_of_8",
        "continuation_contract": (
            "ordinal0_library_budget64; later_previous_successful_recipe_budget16_edits2"
        ),
        "future_block_data_visible": False,
        "weight_sampling": "half_fixed_six_half_seeded_dirichlet_simplex",
        "group_detail": group_rows,
        "update_time_ns": update_ns,
        "policy_execution": dict(optimizer.execution_receipt),
        "rollout_policy_device": "cpu",
        "iteration_wall_time_ns": time.perf_counter_ns() - started,
        **update,
    }


def _validation_sequence_job(
    factory: TrainingTaskFactory, sequence: _BlockSequence,
    weight: Sequence[float], policy: grpo.FrozenPolicy,
    *, seed: int, iteration: int, weight_index: int,
    registry_fingerprint: str,
) -> dict[str, Any]:
    previous_recipe: tuple[int, ...] = ()
    previous_source: str | None = None
    rows: list[dict[str, Any]] = []
    for ordinal, block in enumerate(sequence.blocks):
        request = _factory_request(
            block, stage="validation", weight=weight, seed=seed,
            iteration=iteration, group_index=weight_index, episode_index=-1,
            registry=registry_fingerprint, series_block_ordinal=ordinal,
            previous_recipe=previous_recipe,
            previous_recipe_source_block_id=previous_source,
        )
        task = factory(request)
        _validate_fresh_task(policy, task, request)
        search_config = search.SearchConfig.for_block(ordinal)
        if search_config.max_evaluations != request.evaluation_budget:
            raise TrainingExecutionError(
                "validation search budget differs from the serving protocol"
            )
        bound_provider = getattr(factory, "bound_provider", None)
        if bound_provider is None and callable(
            getattr(task, "certified_bound_state", None)
        ):
            bound_provider = lambda current_task, state: (
                current_task.certified_bound_state(state)
            )
        result = search.policy_candidate_search(
            policy,
            task,
            request.weight,
            config=search_config,
            bound_provider=bound_provider,
        )
        outcome = result.best_outcome
        if outcome.metrics.original_numeric_bytes != request.original_numeric_bytes:
            raise TrainingExecutionError("validation B_raw differs from prepared block")
        if (outcome.metrics.compression_reference_ns_per_byte
                != request.compression_reference_ns_per_byte
                or outcome.metrics.decompression_reference_ns_per_byte
                != request.decompression_reference_ns_per_byte):
            raise TrainingExecutionError("validation E0/D0 differs from frozen reference")
        recipe = _extract_complete_recipe(outcome)
        rows.append({
            "sequence_id": sequence.sequence_id,
            "dataset_id": request.dataset_id, "entity_id": request.entity_id,
            "column": request.column, "split": request.split,
            "block_id": request.block_id,
            "source_block_index": request.block_index,
            "series_block_ordinal": request.series_block_ordinal,
            "previous_recipe": list(previous_recipe),
            "previous_recipe_source_block_id": previous_source,
            "selected_complete_recipe": list(recipe),
            "evaluation_budget": request.evaluation_budget,
            "max_local_edits": request.max_local_edits,
            "reuse_previous_recipe": request.reuse_previous_recipe,
            "start_recipe_mode": request.start_recipe_mode,
            "future_block_data_visible": False,
            "block_fingerprint": _sha256_bytes(_canonical_json({
                "dataset": request.dataset_id, "block": request.block_id,
                "values": request.values_sha256,
                "bitmap": request.missing_bitmap_sha256,
            })),
            "weight": list(request.weight), "J": result.best_objective,
            "rho": outcome.compression_ratio,
            "E_ns_per_original_byte": outcome.metrics.compression_ns_per_byte,
            "D_ns_per_original_byte": outcome.metrics.decompression_ns_per_byte,
            "complete_bytes": len(outcome.compressed_payload),
            "complete_payload_sha256": _sha256_bytes(outcome.compressed_payload),
            "prediction_time_ns": result.timing.total_prediction_time_ns,
            "selection_protocol": "frozen_policy_budgeted_candidate_search",
            "evaluated_complete_candidates": result.evaluations,
            "certified_pruned_states": result.certified_pruned_states,
            "budget_unexplored_actions": result.budget_unexplored_actions,
            "duplicate_states_skipped": result.duplicate_states_skipped,
        })
        previous_recipe = recipe
        previous_source = block.block_id
    return {
        "sequence_id": sequence.sequence_id,
        "weight": list(grpo.validate_weight(weight)),
        "rows": rows,
    }


def _validate_policy(
    *, policy: grpo.FrozenPolicy, factory: TrainingTaskFactory,
    corpus: TrainingCorpus, seed: int, iteration: int, config: TrainingConfig,
    registry_fingerprint: str,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    sequences = _ordered_sequences(corpus.validation_blocks)
    jobs = [
        (sequence, weight_index, weight)
        for sequence in sequences
        for weight_index, weight in enumerate(grpo.FIXED_WEIGHTS)
    ]
    rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.measurement_workers) as pool:
        futures = [
            pool.submit(
                _validation_sequence_job, factory, sequence, weight, policy,
                seed=seed, iteration=iteration, weight_index=weight_index,
                registry_fingerprint=registry_fingerprint,
            )
            for sequence, weight_index, weight in jobs
        ]
        for future in futures:
            sequence_row = future.result()
            sequence_rows.append(sequence_row)
            rows.extend(sequence_row["rows"])
    objectives = [float(row["J"]) for row in rows]
    if not objectives or any(not math.isfinite(value) for value in objectives):
        raise TrainingExecutionError("validation produced no finite exact objectives")
    by_weight: dict[str, list[float]] = {}
    for row in rows:
        key = "-".join(format(float(value), ".12g") for value in row["weight"])
        by_weight.setdefault(key, []).append(float(row["J"]))
    return {
        "iteration": iteration,
        "selection_statistic": "macro_mean_exact_J_over_validation_block_weight_pairs",
        "checkpoint_selection_inference_protocol": (
            "frozen_policy_budgeted_candidate_search_64_first_16_continuation"
        ),
        "validation_pair_count": len(rows),
        "validation_block_count": len(corpus.validation_blocks),
        "validation_sequence_count": len(sequences),
        "weight_count": len(grpo.FIXED_WEIGHTS),
        "mean_exact_J": float(np.mean(objectives)),
        "median_exact_J": float(np.median(objectives)),
        "max_exact_J": float(np.max(objectives)),
        "mean_exact_J_by_weight": {
            key: float(np.mean(values)) for key, values in sorted(by_weight.items())
        },
        "rows": rows,
        "sequence_rows": sequence_rows,
        "continuation_contract": (
            "ordinal0_library_budget64; later_previous_successful_recipe_budget16_edits2"
        ),
        "future_block_data_visible": False,
        "validation_wall_time_ns": time.perf_counter_ns() - started,
    }


def _run_identity(
    corpus: TrainingCorpus, gate: GateEvidence, policy_identity: grpo.PolicyIdentity,
    config: TrainingConfig, factory: TrainingTaskFactory,
    solution_library_identity: Mapping[str, str],
) -> dict[str, Any]:
    try:
        factory_file = inspect.getsourcefile(factory)
    except (TypeError, OSError):
        factory_file = None
    import weighted_grpo_torch_backend_v1 as torch_backend
    source_paths = [Path(__file__).resolve(), Path(grpo.__file__).resolve(),
                    Path(torch_backend.__file__).resolve()]
    if factory_file:
        source_paths.append(Path(factory_file).resolve())
    config_identity = asdict(config)
    # ``max_iterations`` is an operator-requested early-stop ceiling, not a
    # learning or selection semantic.  Excluding it permits an exact resume
    # from a bounded preflight into the same formal run.
    config_identity.pop("max_iterations", None)
    return {
        "schema": SCHEMA,
        "corpus_sha256": corpus.corpus_sha256,
        "dataset_provenance_sha256": [list(row) for row in corpus.dataset_provenance_sha256],
        "calibration_sha256": corpus.calibration_sha256,
        "gate_bundle_sha256": gate.bundle_sha256,
        "gate_receipt_sha256": [list(row) for row in gate.receipt_sha256],
        "source_fingerprint": gate.source_fingerprint,
        "action_registry_fingerprint": gate.action_registry_fingerprint,
        "solution_library_identity": dict(solution_library_identity),
        "policy_identity": policy_identity.to_json(),
        "sequence_contract": {
            "group_key": ["dataset_id", "entity_id", "split", "column"],
            "order": ["source_block_index", "block_id"],
            "train_sequence_count": len(_ordered_sequences(corpus.train_blocks)),
            "validation_sequence_count": len(_ordered_sequences(corpus.validation_blocks)),
            "first_block": "training_solution_library_only_budget64",
            "continuation": "immediate_previous_successful_complete_recipe_budget16_edits2",
            "future_block_data_visible": False,
        },
        "training_config": _json_safe(config_identity),
        "orchestrator_sources": {
            str(path): _sha256_file(path) for path in source_paths if path.is_file()
        },
    }


def _factory_solution_library_identity(
    factory: TrainingTaskFactory, *, expected_registry_fingerprint: str,
) -> dict[str, str]:
    """Resolve and independently hash the factory's frozen training library."""
    provider = getattr(factory, "solution_library_identity", None)
    if not callable(provider):
        raise TrainingAdmissionError(
            "formal task factory lacks solution_library_identity evidence"
        )
    raw = provider()
    if not isinstance(raw, Mapping):
        raise TrainingAdmissionError("solution library identity must be an object")
    path_text = raw.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise TrainingAdmissionError("solution library identity lacks an absolute path")
    path = Path(path_text).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise TrainingAdmissionError("solution library path is not an existing absolute file")
    path = path.resolve()
    actual_sha256 = _sha256_file(path)
    if raw.get("file_sha256") != actual_sha256:
        raise TrainingAdmissionError("solution library bytes changed after factory binding")
    registry = raw.get("action_registry_fingerprint")
    if registry != expected_registry_fingerprint:
        raise TrainingAdmissionError(
            "solution library action registry differs from formal gate evidence"
        )
    expected_path = os.environ.get("WEIGHTED_GRPO_SOLUTION_LIBRARY_MANIFEST")
    expected_sha = os.environ.get("WEIGHTED_GRPO_SOLUTION_LIBRARY_SHA256")
    if expected_path is not None and Path(expected_path).expanduser().resolve() != path:
        raise TrainingAdmissionError(
            "task factory solution library differs from the injected manifest path"
        )
    if expected_sha is not None and expected_sha != actual_sha256:
        raise TrainingAdmissionError(
            "task factory solution library differs from the injected manifest hash"
        )
    return {
        "path": str(path),
        "file_sha256": actual_sha256,
        "action_registry_fingerprint": str(registry),
    }


def _resume_document(
    *, run_identity: Mapping[str, Any], seed: int, iteration: int,
    policy: grpo.TwoHiddenMLP, optimizer: grpo.Adam, rng: np.random.Generator,
    cumulative_wall_ns: int, best_score: float | None,
    best_checkpoint: str | None, trace: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]], status: str,
) -> dict[str, Any]:
    payload = {
        "run_identity": run_identity, "seed": seed, "completed_iterations": iteration,
        "policy_parameters": [_array_record(value) for value in policy.params],
        "optimizer_first": [_array_record(value) for value in optimizer.first],
        "optimizer_second": [_array_record(value) for value in optimizer.second],
        "optimizer_steps": optimizer.steps, "rng_state": _json_safe(rng.bit_generator.state),
        "cumulative_training_wall_time_ns": cumulative_wall_ns,
        "best_validation_score": best_score,
        "best_policy_checkpoint": best_checkpoint,
        "trace": list(trace), "validations": list(validations), "status": status,
    }
    return {
        "schema": RESUME_SCHEMA,
        "payload_sha256": _sha256_bytes(_canonical_json(payload)),
        "payload": payload,
    }


def _load_resume(
    path: Path, *, expected_identity: Mapping[str, Any], config: TrainingConfig,
) -> tuple[
    grpo.TwoHiddenMLP, grpo.Adam, np.random.Generator, int, int,
    float | None, str | None, list[dict[str, Any]], list[dict[str, Any]],
] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingAdmissionError("cannot read resume checkpoint") from error
    if document.get("schema") != RESUME_SCHEMA or not isinstance(document.get("payload"), Mapping):
        raise TrainingAdmissionError("unsupported resume checkpoint")
    payload = document["payload"]
    if document.get("payload_sha256") != _sha256_bytes(_canonical_json(payload)):
        raise TrainingAdmissionError("resume checkpoint digest mismatch")
    if payload.get("run_identity") != expected_identity:
        raise TrainingAdmissionError("resume checkpoint immutable identity mismatch")
    identity = grpo.PolicyIdentity.from_json(expected_identity["policy_identity"])
    policy = grpo.TwoHiddenMLP(identity, np.random.default_rng(0))
    policy.params = [_array_from_record(value) for value in payload["policy_parameters"]]
    optimizer = grpo.Adam(policy.params, config.grpo_config.learning_rate,
                          device=config.policy_device)
    optimizer.first = [_array_from_record(value) for value in payload["optimizer_first"]]
    optimizer.second = [_array_from_record(value) for value in payload["optimizer_second"]]
    optimizer.steps = int(payload["optimizer_steps"])
    if ([value.shape for value in policy.params] != [value.shape for value in optimizer.first]
            or [value.shape for value in policy.params] != [value.shape for value in optimizer.second]):
        raise TrainingAdmissionError("resume optimizer shape mismatch")
    rng = np.random.default_rng(0)
    try:
        rng.bit_generator.state = payload["rng_state"]
    except (TypeError, ValueError) as error:
        raise TrainingAdmissionError("invalid resume RNG state") from error
    score = payload.get("best_validation_score")
    if score is not None:
        score = float(score)
        if not math.isfinite(score):
            raise TrainingAdmissionError("invalid best validation score")
    return (
        policy, optimizer, rng, int(payload["completed_iterations"]),
        int(payload["cumulative_training_wall_time_ns"]), score,
        payload.get("best_policy_checkpoint"), list(payload.get("trace", [])),
        list(payload.get("validations", [])),
    )


def train_one_seed(
    *, corpus: TrainingCorpus, gate: GateEvidence, factory: TrainingTaskFactory,
    output_dir: Path, seed: int, config: TrainingConfig,
    initialize_only: bool = False, require_initialized: bool = False,
) -> dict[str, Any]:
    """Train or exactly resume one seed; never reads a test split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_identity = _factory_solution_library_identity(
        factory,
        expected_registry_fingerprint=gate.action_registry_fingerprint,
    )
    identity = _build_policy_identity(corpus, factory, gate, config)
    run_identity = _run_identity(
        corpus, gate, identity, config, factory, solution_identity,
    )
    resume_path = output_dir / "resume_checkpoint.json"
    loaded = _load_resume(resume_path, expected_identity=run_identity, config=config)
    if loaded is None and require_initialized:
        raise TrainingAdmissionError("supervised training requires an initialized seed checkpoint")
    if loaded is None:
        policy = grpo.TwoHiddenMLP(identity, np.random.default_rng(seed))
        optimizer = grpo.Adam(policy.params, config.grpo_config.learning_rate,
                              device=config.policy_device)
        rng = np.random.default_rng(seed + 1_000_003)
        iteration = cumulative_wall_ns = 0
        best_score = None
        best_checkpoint = None
        trace: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
    else:
        (policy, optimizer, rng, iteration, cumulative_wall_ns, best_score,
         best_checkpoint, trace, validations) = loaded
    budget_ns = int(config.per_seed_budget_seconds * 1_000_000_000)
    status = "running"
    session_started = time.perf_counter_ns()
    def persist_resume() -> None:
        _atomic_json(resume_path, _resume_document(
            run_identity=run_identity, seed=seed, iteration=iteration,
            policy=policy, optimizer=optimizer, rng=rng,
            cumulative_wall_ns=cumulative_wall_ns,
            best_score=best_score, best_checkpoint=best_checkpoint,
            trace=trace, validations=validations, status="running",
        ))
    with _single_core_environment():
        if not validations:
            initial_validation_started = time.perf_counter_ns()
            validation = _validate_policy(
                policy=grpo.FrozenPolicy(policy), factory=factory, corpus=corpus,
                seed=seed, iteration=iteration, config=config,
                registry_fingerprint=gate.action_registry_fingerprint,
            )
            validations.append(validation)
            validation["accounting_scope"] = "initial_frozen_validation_outside_training_budget"
            best_score = float(validation["mean_exact_J"])
            best_path = output_dir / f"best_policy_iter_{iteration:08d}.json"
            best_checkpoint = str(best_path.resolve())
            grpo.save_checkpoint(
                grpo.TrainingResult(policy, (), config.grpo_config, seed, iteration), best_path,
                extra={
                    "selection": validation, "run_identity": run_identity,
                    "solution_library_identity": solution_identity,
                    "selection_used_test_data": False,
                },
            )
            validation["initial_validation_wall_time_ns"] = (
                time.perf_counter_ns() - initial_validation_started)
            # The separate initialize phase commits iteration zero before the
            # supervised training process starts its hard wall-clock budget.
            persist_resume()
        if initialize_only:
            status = "initialized"
        while not initialize_only and cumulative_wall_ns < budget_ns:
            if config.max_iterations is not None and iteration >= config.max_iterations:
                status = "max_iterations_reached"
                break
            iteration_started = time.perf_counter_ns()
            iteration += 1
            row = _training_iteration(
                policy=policy, optimizer=optimizer, factory=factory, corpus=corpus,
                rng=rng, seed=seed, iteration=iteration, config=config,
                registry_fingerprint=gate.action_registry_fingerprint,
            )
            trace.append(row)
            cumulative_wall_ns += time.perf_counter_ns() - iteration_started
            # Commit every complete update before a possibly lengthy validation.
            # Atomic replace leaves the preceding complete checkpoint intact if
            # the supervisor kills this process during serialization.
            persist_resume()
            if iteration % config.validation_interval == 0:
                validation_started = time.perf_counter_ns()
                validation = _validate_policy(
                    policy=grpo.FrozenPolicy(policy), factory=factory, corpus=corpus,
                    seed=seed, iteration=iteration, config=config,
                    registry_fingerprint=gate.action_registry_fingerprint,
                )
                validation_wall = time.perf_counter_ns() - validation_started
                cumulative_wall_ns += validation_wall
                validations.append(validation)
                score = float(validation["mean_exact_J"])
                if best_score is None or score < best_score:
                    best_score = score
                    best_path = output_dir / f"best_policy_iter_{iteration:08d}.json"
                    best_checkpoint = str(best_path.resolve())
                    grpo.save_checkpoint(
                        grpo.TrainingResult(
                            policy, tuple(trace), config.grpo_config, seed, iteration
                        ),
                        best_path,
                        extra={
                            "selection": validation, "run_identity": run_identity,
                            "solution_library_identity": solution_identity,
                            "selection_used_test_data": False,
                        },
                    )
            if iteration % config.checkpoint_interval == 0:
                _atomic_json(resume_path, _resume_document(
                    run_identity=run_identity, seed=seed, iteration=iteration,
                    policy=policy, optimizer=optimizer, rng=rng,
                    cumulative_wall_ns=cumulative_wall_ns,
                    best_score=best_score, best_checkpoint=best_checkpoint,
                    trace=trace, validations=validations, status="running",
                ))
        else:
            if not initialize_only:
                status = "wall_clock_budget_exhausted"
    if status == "running":
        status = "wall_clock_budget_exhausted"
    _atomic_json(resume_path, _resume_document(
        run_identity=run_identity, seed=seed, iteration=iteration,
        policy=policy, optimizer=optimizer, rng=rng,
        cumulative_wall_ns=cumulative_wall_ns,
        best_score=best_score, best_checkpoint=best_checkpoint,
        trace=trace, validations=validations, status=status,
    ))
    receipt = {
        "schema": SCHEMA, "seed": seed, "status": status,
        "completed_iterations": iteration,
        "cumulative_training_wall_time_ns": cumulative_wall_ns,
        "current_session_wall_time_ns": time.perf_counter_ns() - session_started,
        "training_budget_ns": budget_ns,
        "policy_execution": dict(optimizer.execution_receipt),
        "training_wall_clock_includes_periodic_validation": True,
        "initial_frozen_validation_in_training_budget": False,
        "online_selection_time_included": False,
        "final_encoding_time_included": False,
        "best_validation_score": best_score,
        "best_policy_checkpoint": best_checkpoint,
        "best_policy_selected_by": "validation_macro_mean_exact_J",
        "test_data_used_for_selection": False,
        "run_identity": run_identity,
        "trace": trace, "validations": validations,
        "resume_checkpoint": str(resume_path.resolve()),
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_json(receipt))
    _atomic_json(output_dir / "training_receipt.json", receipt)
    return receipt


def validate_last_complete_seed(
    *, corpus: TrainingCorpus, gate: GateEvidence, factory: TrainingTaskFactory,
    output_dir: Path, seed: int, config: TrainingConfig,
) -> dict[str, Any]:
    """Read the last atomic update after worker exit; never resume optimization."""
    solution_identity = _factory_solution_library_identity(
        factory, expected_registry_fingerprint=gate.action_registry_fingerprint)
    identity = _build_policy_identity(corpus, factory, gate, config)
    run_identity = _run_identity(corpus, gate, identity, config, factory, solution_identity)
    loaded = _load_resume(output_dir / "resume_checkpoint.json",
                          expected_identity=run_identity, config=config)
    if loaded is None:
        raise TrainingAdmissionError("no last complete checkpoint exists for frozen validation")
    policy, _, _, iteration, wall_ns, _, _, trace, _ = loaded
    started = time.perf_counter_ns()
    before = [p.copy() for p in policy.params]
    with _single_core_environment():
        validation = _validate_policy(
            policy=grpo.FrozenPolicy(policy), factory=factory, corpus=corpus,
            seed=seed, iteration=iteration, config=config,
            registry_fingerprint=gate.action_registry_fingerprint)
    if not all(np.array_equal(a, b) for a, b in zip(before, policy.params)):
        raise TrainingExecutionError("frozen post-training validation changed policy parameters")
    checkpoint = output_dir / f"last_complete_policy_iter_{iteration:08d}.json"
    digest = grpo.save_checkpoint(
        grpo.TrainingResult(policy, tuple(trace), config.grpo_config, seed, iteration),
        checkpoint, extra={"selection": validation, "run_identity": run_identity,
                           "solution_library_identity": solution_identity,
                           "selection_used_test_data": False,
                           "phase": "frozen_validation_after_worker_exit"})
    receipt = {
        "schema": "weighted-grpo-last-complete-frozen-validation-v1",
        "seed": seed, "completed_iterations": iteration,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_payload_sha256": digest,
        "checkpoint_file_sha256": _sha256_file(checkpoint),
        "validation": validation, "optimizer_steps": 0,
        "model_parameters_frozen": True, "selection_used_test_data": False,
        "checkpoint_recorded_training_wall_ns": wall_ns,
        "validation_wall_time_ns": time.perf_counter_ns() - started,
        "validation_in_training_budget": False,
        "training_budget_authority": "independent seed supervisor ledger",
    }
    _atomic_json(output_dir / "frozen_validation_receipt.json", receipt)
    return receipt


def run_campaign(
    *, corpus: TrainingCorpus, gate: GateEvidence, factory: TrainingTaskFactory,
    output_dir: Path, config: TrainingConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for seed in config.seeds:
        receipts.append(train_one_seed(
            corpus=corpus, gate=gate, factory=factory,
            output_dir=output_dir / f"seed_{seed}", seed=seed, config=config,
        ))
    eligible = [receipt for receipt in receipts
                if receipt.get("best_policy_checkpoint") is not None
                and receipt.get("best_validation_score") is not None]
    if not eligible:
        raise TrainingExecutionError("no seed produced a validation-selected checkpoint")
    selected = min(eligible, key=lambda receipt: (receipt["best_validation_score"], receipt["seed"]))
    solution_identities = {
        _sha256_bytes(_canonical_json(
            receipt["run_identity"]["solution_library_identity"]
        ))
        for receipt in receipts
    }
    if len(solution_identities) != 1:
        raise TrainingExecutionError("training seeds consumed different solution libraries")
    solution_identity = dict(
        selected["run_identity"]["solution_library_identity"]
    )
    if solution_identity.get("action_registry_fingerprint") != gate.action_registry_fingerprint:
        raise TrainingExecutionError(
            "selected solution library action registry differs from gate evidence"
        )
    selected_path = Path(str(selected["best_policy_checkpoint"]))
    destination = output_dir / "delivery_policy.json"
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(selected_path, temporary)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    delivery = grpo.load_checkpoint(destination)
    if (delivery.extra.get("solution_library_identity") != solution_identity
            or not isinstance(delivery.extra.get("run_identity"), Mapping)
            or delivery.extra["run_identity"].get("solution_library_identity")
            != solution_identity
            or delivery.extra["run_identity"].get("action_registry_fingerprint")
            != gate.action_registry_fingerprint):
        raise TrainingExecutionError(
            "delivery checkpoint is not bound to the selected solution library/registry"
        )
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(config.seeds),
        "per_seed_budget_seconds": config.per_seed_budget_seconds,
        "total_requested_training_budget_seconds": (
            len(config.seeds) * config.per_seed_budget_seconds
        ),
        "measurement_workers_per_seed": config.measurement_workers,
        "policy_device": config.policy_device,
        "rollout_policy_device": "cpu",
        "training_seeds_execute": "sequential_on_one_policy_device",
        "selected_seed": selected["seed"],
        "selected_validation_score": selected["best_validation_score"],
        "selection_used_test_data": False,
        "source_checkpoint": str(selected_path.resolve()),
        "delivery_checkpoint": str(destination.resolve()),
        "delivery_file_sha256": _sha256_file(destination),
        "solution_library_identity": solution_identity,
        "action_registry_fingerprint": gate.action_registry_fingerprint,
        "seed_receipts": [
            {
                "seed": receipt["seed"], "status": receipt["status"],
                "completed_iterations": receipt["completed_iterations"],
                "training_wall_time_ns": receipt["cumulative_training_wall_time_ns"],
                "best_validation_score": receipt["best_validation_score"],
                "receipt_sha256": receipt["receipt_sha256"],
                "solution_library_identity": receipt["run_identity"][
                    "solution_library_identity"
                ],
            }
            for receipt in receipts
        ],
    }
    campaign["campaign_sha256"] = _sha256_bytes(_canonical_json(campaign))
    _atomic_json(output_dir / "campaign_receipt.json", campaign)
    return campaign


def _load_factory(spec: str) -> TrainingTaskFactory:
    if ":" not in spec:
        raise ValueError("task factory must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError("task factory target is not callable")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", action="append", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--admission-bundle", type=Path, required=True)
    parser.add_argument("--task-factory", required=True, help="module:function")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--single-seed", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--phase", choices=("initialize", "train", "frozen-validation"), default="train")
    parser.add_argument("--require-initialized", action="store_true")
    parser.add_argument("--policy-device", default="cpu",
                        help="cpu or one CUDA device, e.g. cuda:0; no implicit fallback")
    parser.add_argument("--measurement-workers", type=int,
                        default=int(os.environ.get("WEIGHTED_GRPO_MAX_MEASUREMENT_WORKERS", "4")))
    args = parser.parse_args(argv)
    corpus = _load_corpus(args.prepared, args.reference)
    gate = _load_gate_bundle(args.admission_bundle, corpus)
    config = TrainingConfig(
        seeds=FORMAL_SEEDS,
        per_seed_budget_seconds=FORMAL_SEED_BUDGET_SECONDS,
        measurement_workers=args.measurement_workers,
        policy_device=args.policy_device,
        validation_interval=args.validation_interval,
        checkpoint_interval=args.checkpoint_interval,
        max_iterations=args.max_iterations,
    )
    if args.single_seed is not None:
        common = dict(corpus=corpus, gate=gate, factory=_load_factory(args.task_factory),
                      output_dir=args.output.resolve(), seed=args.single_seed, config=config)
        if args.phase == "frozen-validation":
            receipt = validate_last_complete_seed(**common)
        else:
            receipt = train_one_seed(**common, initialize_only=args.phase == "initialize",
                                     require_initialized=args.require_initialized)
        print(json.dumps({"schema": receipt["schema"], "seed": args.single_seed,
                          "phase": args.phase, "output": str(args.output.resolve())}))
        return 0
    if args.phase != "train" or args.require_initialized:
        parser.error("separate initialization/validation phases require --single-seed")
    campaign = run_campaign(
        corpus=corpus, gate=gate, factory=_load_factory(args.task_factory),
        output_dir=args.output.resolve(), config=config,
    )
    print(json.dumps({
        "schema": campaign["schema"],
        "campaign_sha256": campaign["campaign_sha256"],
        "delivery_checkpoint": campaign["delivery_checkpoint"],
        "selected_seed": campaign["selected_seed"],
        "selected_validation_score": campaign["selected_validation_score"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
