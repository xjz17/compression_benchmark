#!/usr/bin/env python3
"""Independent weighted-GRPO/Beam30 experiment orchestration protocol.

The module consumes prepared ``dataset_manifest.json`` artifacts, an immutable
zlib-6 E0/D0 reference table, the six project weight vectors, and either an
already frozen policy or a checkpoint that is frozen immediately after load.
It delegates executable graph construction to an injected ``task_factory`` and
delegates online search to :mod:`weight_conditioned_search_v1`.

Three stages are deliberately kept separate in every receipt:

* training provenance -- consumed checkpoint/policy only; no update occurs;
* online selection -- policy/Beam ordering plus exact candidate evaluations;
* final independent encoding -- a new factory task re-encodes the selected
  complete recipe once and must reproduce the selected self-contained bytes.

This file does not manufacture 86-method, OpenZL, or AMMMO measurements.
Those comparisons remain absent until independently measured under the same
input, complete-byte, and timing contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

import calibrate_weighted_grpo_references_v1 as calibration
import prepare_weighted_grpo_timeseries_v1 as preparation
from weight_conditioned_grpo_v1 import (
    FIXED_WEIGHTS,
    FrozenPolicy,
    GraphCompressionTask,
    TerminalOutcome,
    load_checkpoint,
    objective_value,
    validate_weight,
)
from weight_conditioned_search_v1 import (
    BEAM_WIDTH,
    SearchConfig,
    SearchResult,
    beam30_search,
    policy_candidate_search,
)


SCHEMA = "weighted-grpo-experiment-orchestration-v1"
TASK_FACTORY_SCHEMA = "weighted-grpo-task-factory-request-v1"
CAMPAIGN_PLAN_SCHEMA = "formal-weighted-grpo-campaign-plan-v1"
FORMAL_MEASUREMENT_PROTOCOL_SCHEMA = "native-in-process-complete-frame-timing-v1"
FORMAL_MEASUREMENT_CONTRACT = {
    "schema": "weighted-grpo-formal-measurement-contract-v1",
    "execution_layer": "native_in_process",
    "frame_scope": "complete_frame_encode_and_decode",
    "operator_process_launch_and_ipc": "excluded",
    "python_scheduling": "excluded",
    "source_io_and_policy_inference": "excluded",
    "initialization": "outside_timing_same_process_warmups",
    "aggregation": "median_of_repeated_native_elapsed_ns",
    "independent_cold_decode": "correctness_only_outside_timing",
}
FORMAL_TIMING_BOUNDARY = (
    "native in-process complete frame; excludes Python scheduling, operator "
    "process launch/IPC, source I/O, policy inference and cold validation; "
    "same-process warmups precede measured repeats"
)
MAX_CONTINUATION_EDITS = 2
METHODS = ("grpo", "beam30")
TIMING_FIELDS = (
    "online_task_factory_time_ns",
    "feature_time_ns",
    "model_inference_time_ns",
    "action_ordering_time_ns",
    "environment_transition_time_ns",
    "certified_pruning_time_ns",
    "candidate_evaluation_time_ns",
    "search_only_total_prediction_time_ns",
    "total_prediction_time_ns",
)


class ExperimentContractError(RuntimeError):
    """An input, factory, or result violated the experiment protocol."""


class TaskFactory(Protocol):
    def __call__(self, request: "TaskFactoryRequest") -> GraphCompressionTask: ...


@dataclass(frozen=True)
class AblationConfig:
    """Named ablations; all remain visible in the saved result."""

    no_certified_prune: bool = False
    no_heuristic_order: bool = False
    no_reuse: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in (
            self.no_certified_prune, self.no_heuristic_order, self.no_reuse
        )):
            raise TypeError("ablation fields must be booleans")


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen v1 experiment dimensions and evidence behavior."""

    methods: tuple[str, ...] = METHODS
    weights: tuple[tuple[float, float, float], ...] = FIXED_WEIGHTS
    splits: tuple[str, ...] = ("validation",)
    beam_width: int = BEAM_WIDTH
    verify_source: bool = True
    fail_fast: bool = False
    require_independent_payload_match: bool = True
    require_formal_evidence: bool = False
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self) -> None:
        if any(not isinstance(value, bool) for value in (
            self.verify_source, self.fail_fast,
            self.require_independent_payload_match,
            self.require_formal_evidence,
        )):
            raise TypeError("experiment evidence flags must be booleans")
        if (not self.methods or len(set(self.methods)) != len(self.methods)
                or any(method not in METHODS for method in self.methods)):
            raise ValueError(f"methods must be a unique subset of {METHODS}")
        canonical = tuple(tuple(float(value) for value in validate_weight(weight))
                          for weight in self.weights)
        if canonical != FIXED_WEIGHTS:
            raise ValueError("v1 requires exactly the six frozen project weight vectors")
        if (not self.splits or len(set(self.splits)) != len(self.splits)
                or any(split not in {"train", "validation", "test"}
                       for split in self.splits)):
            raise ValueError(
                "splits must be a unique non-empty subset of train/validation/test"
            )
        if isinstance(self.beam_width, bool) or self.beam_width != BEAM_WIDTH:
            raise ValueError("v1 Beam search width is fixed at 30")


@dataclass(frozen=True)
class TaskFactoryRequest:
    """Everything a factory needs to build one auditable executable task.

    For ``online_selection``, later blocks receive the preceding successful
    full recipe and ``max_local_edits=2``.  With the ``no_reuse`` ablation the
    previous recipe is empty, while the two-edit continuation budget remains
    explicit.  For ``final_independent_encoding``, ``selected_recipe`` is the
    complete fixed recipe and the factory must return a task whose initial
    state evaluates exactly that recipe without policy search.
    """

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
    method: str
    weight: tuple[float, float, float]
    previous_recipe: tuple[int, ...]
    selected_recipe: tuple[int, ...]
    max_local_edits: int | None
    evaluation_budget: int
    reuse_previous_recipe: bool
    ablation: AblationConfig
    schema: str = TASK_FACTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.stage not in {"online_selection", "final_independent_encoding"}:
            raise ValueError("unknown task-factory stage")
        if self.method not in METHODS:
            raise ValueError("unknown search method")
        validate_weight(self.weight)
        if self.original_numeric_bytes <= 0 or self.prepared_block_bytes <= 0:
            raise ValueError("factory request requires positive block byte counts")
        if (self.block_index < 0 or self.series_block_ordinal < 0
                or self.evaluation_budget < 1):
            raise ValueError("factory request block index/ordinal/budget is invalid")
        refs = (
            self.compression_reference_ns_per_byte,
            self.decompression_reference_ns_per_byte,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in refs):
            raise ValueError("factory request references must be finite and positive")
        if self.stage == "online_selection":
            expected = (
                None if self.series_block_ordinal == 0
                else MAX_CONTINUATION_EDITS
            )
            if self.max_local_edits != expected or self.selected_recipe:
                raise ValueError("online factory request has an invalid edit/recipe contract")
        elif self.max_local_edits != 0:
            raise ValueError("final encoding requires a fixed selected recipe")

    def json_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["weight"] = list(self.weight)
        row["previous_recipe"] = list(self.previous_recipe)
        row["selected_recipe"] = list(self.selected_recipe)
        return row


@dataclass(frozen=True)
class _PreparedDataset:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    verification: Mapping[str, Any]


@dataclass(frozen=True)
class _ReferenceTable:
    path: Path
    calibration_sha256: str
    references: Mapping[tuple[str, int], tuple[float, float]]
    document: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentContractError("result metadata contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": _sha256(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "json_record") and callable(value.json_record):
        return _json_safe(value.json_record())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    raise ExperimentContractError(
        f"operator/result metadata is not JSON serializable: {type(value).__name__}"
    )


def _weight_id(weight: Sequence[float]) -> str:
    return "w_" + "_".join(format(float(value), ".12g").replace(".", "p")
                            for value in weight)


def _policy_sha256(policy: FrozenPolicy) -> str:
    digest = hashlib.sha256(_canonical_json(policy.identity.to_json()))
    for parameter in policy.params:
        array = np.ascontiguousarray(parameter, dtype="<f8")
        digest.update(_canonical_json({"shape": list(array.shape), "dtype": "<f8"}))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_frozen_policy(
    *,
    methods: Sequence[str],
    policy: FrozenPolicy | None,
    checkpoint_path: Path | None,
    policy_device: str | None = None,
) -> tuple[FrozenPolicy | None, dict[str, Any]]:
    needs_policy = "grpo" in methods
    if policy is not None and checkpoint_path is not None:
        raise ValueError("provide a FrozenPolicy or checkpoint, not both")
    if not needs_policy:
        if policy is not None or checkpoint_path is not None:
            raise ValueError("a policy/checkpoint is unused when GRPO is not selected")
        return None, {
            "stage": "training",
            "status": "not_applicable_beam_only",
            "optimizer_steps": 0,
            "model_parameters_frozen": True,
        }
    if policy is None and checkpoint_path is None:
        raise ValueError("GRPO search requires a FrozenPolicy or checkpoint")
    if checkpoint_path is not None:
        loaded = load_checkpoint(checkpoint_path)
        policy = FrozenPolicy(loaded.policy, device=(policy_device or os.environ.get(
            "WEIGHTED_GRPO_POLICY_INFERENCE_DEVICE", "cpu")))
        receipt: dict[str, Any] = {
            "stage": "training",
            "status": "not_run_checkpoint_consumed",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_payload_sha256": loaded.payload_sha256,
            "checkpoint_file_sha256": _sha256(checkpoint_path.resolve().read_bytes()),
            "checkpoint_seed": loaded.seed,
            "checkpoint_completed_iterations": loaded.completed_iterations,
            "checkpoint_config": _json_safe(asdict(loaded.config)),
            "checkpoint_extra": _json_safe(loaded.extra),
            "optimizer_steps": 0,
            "model_parameters_frozen": True,
        }
    else:
        if not isinstance(policy, FrozenPolicy):
            raise TypeError("policy must be a FrozenPolicy")
        receipt = {
            "stage": "training",
            "status": "not_run_frozen_policy_consumed",
            "checkpoint_path": None,
            "checkpoint_payload_sha256": None,
            "optimizer_steps": 0,
            "model_parameters_frozen": True,
        }
    assert policy is not None
    if any(value.flags.writeable for value in policy.params):
        raise ExperimentContractError("FrozenPolicy parameters are unexpectedly writeable")
    receipt["policy_identity"] = policy.identity.to_json()
    receipt["policy_inference_device"] = policy.execution_device
    receipt["policy_parameter_sha256_before"] = _policy_sha256(policy)
    return policy, receipt


def _load_reference_table(path: Path) -> _ReferenceTable:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentContractError("cannot read zlib6 reference table") from error
    if not isinstance(document, dict) or document.get("schema") != calibration.SCHEMA:
        raise ExperimentContractError("unsupported zlib6 reference-table schema")
    expected = document.get("calibration_sha256")
    basis = {key: value for key, value in document.items()
             if key != "calibration_sha256"}
    actual = _sha256(_canonical_json(basis))
    if expected != actual:
        raise ExperimentContractError("zlib6 reference-table digest mismatch")
    if document.get("objective_normalization") != "E/(E+E0), D/(D+D0)":
        raise ExperimentContractError("zlib6 reference objective contract mismatch")
    references: dict[tuple[str, int], tuple[float, float]] = {}
    rows = document.get("references")
    if not isinstance(rows, list) or not rows:
        raise ExperimentContractError("zlib6 reference table has no groups")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExperimentContractError("zlib6 reference group must be an object")
        try:
            key = (str(row["dtype"]), int(row["block_size_bucket_bytes"]))
            values = (
                float(row["E0_compression_ns_per_original_byte"]),
                float(row["D0_decompression_ns_per_original_byte"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ExperimentContractError("invalid zlib6 reference group") from error
        if (key in references or key[1] < 1
                or any(not math.isfinite(value) or value <= 0.0 for value in values)):
            raise ExperimentContractError("duplicate or invalid zlib6 reference group")
        references[key] = values
    return _ReferenceTable(path.resolve(), actual, references, document)


def _load_campaign_binding(
    path: Path | None,
    *,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    """Load an immutable formal-campaign identity without trusting labels.

    The plan is written by the fail-closed campaign launcher before any frozen
    test command runs.  Recomputing its digest here prevents a result from
    being attached to a different cohort, gate bundle, action registry, or
    delivery model after measurement.
    """
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentContractError("cannot read formal campaign plan") from error
    if not isinstance(document, Mapping) or document.get("schema") != CAMPAIGN_PLAN_SCHEMA:
        raise ExperimentContractError("unsupported formal campaign plan schema")
    expected_plan_sha256 = document.get("plan_sha256")
    basis = dict(document)
    basis.pop("plan_sha256", None)
    actual_plan_sha256 = _sha256(_canonical_json(basis))
    if expected_plan_sha256 != actual_plan_sha256:
        raise ExperimentContractError("formal campaign plan digest mismatch")
    admission = document.get("admission")
    if not isinstance(admission, Mapping):
        raise ExperimentContractError("formal campaign plan lacks admission identity")
    required = (
        "corpus_sha256", "calibration_sha256", "gate_bundle_sha256",
        "source_fingerprint", "action_registry_fingerprint", "test_cohort_sha256",
    )
    if any(not isinstance(admission.get(key), str) or not admission.get(key)
           for key in required):
        raise ExperimentContractError("formal campaign admission identity is incomplete")
    delivery_text = document.get("delivery_checkpoint")
    if not isinstance(delivery_text, str) or not delivery_text:
        raise ExperimentContractError("formal campaign plan lacks delivery checkpoint")
    delivery = Path(delivery_text).expanduser().resolve()
    if not delivery.is_file():
        raise ExperimentContractError("formal delivery checkpoint is missing")
    delivery_file_sha256 = _sha256(delivery.read_bytes())
    if checkpoint_path is not None and checkpoint_path.expanduser().resolve() != delivery:
        raise ExperimentContractError(
            "GRPO checkpoint differs from the campaign delivery checkpoint"
        )
    library_text = document.get("solution_library_manifest")
    if not isinstance(library_text, str) or not library_text:
        raise ExperimentContractError(
            "formal campaign plan lacks a solution-library manifest path"
        )
    library = Path(library_text).expanduser()
    if not library.is_absolute() or not library.is_file():
        raise ExperimentContractError("formal solution library is missing")
    library = library.resolve()
    library_file_sha256 = _sha256(library.read_bytes())
    return {
        "campaign_plan_path": str(resolved),
        "campaign_plan_sha256": actual_plan_sha256,
        "corpus_sha256": str(admission["corpus_sha256"]),
        "calibration_sha256": str(admission["calibration_sha256"]),
        "gate_bundle_sha256": str(admission["gate_bundle_sha256"]),
        "source_fingerprint": str(admission["source_fingerprint"]),
        "action_registry_fingerprint": str(
            admission["action_registry_fingerprint"]
        ),
        "solution_library_path": str(library),
        "solution_library_file_sha256": library_file_sha256,
        "test_cohort_sha256": str(admission["test_cohort_sha256"]),
        "delivery_checkpoint_path": str(delivery),
        "delivery_checkpoint_file_sha256": delivery_file_sha256,
        "delivery_checkpoint_role": (
            "consumed_frozen_policy" if checkpoint_path is not None
            else "bound_campaign_model_not_used_by_non_policy_method"
        ),
    }


def _formal_solution_library_identity(
    task_factory: TaskFactory, campaign_binding: Mapping[str, Any],
) -> dict[str, str]:
    provider = getattr(task_factory, "solution_library_identity", None)
    if not callable(provider):
        raise ExperimentContractError(
            "formal task factory lacks solution_library_identity evidence"
        )
    raw = provider()
    if not isinstance(raw, Mapping):
        raise ExperimentContractError("solution library identity must be an object")
    identity = {
        "path": str(raw.get("path", "")),
        "file_sha256": str(raw.get("file_sha256", "")),
        "action_registry_fingerprint": str(
            raw.get("action_registry_fingerprint", "")
        ),
    }
    expected = {
        "path": str(campaign_binding["solution_library_path"]),
        "file_sha256": str(campaign_binding["solution_library_file_sha256"]),
        "action_registry_fingerprint": str(
            campaign_binding["action_registry_fingerprint"]
        ),
    }
    if identity != expected:
        raise ExperimentContractError(
            "task factory solution library differs from the formal campaign"
        )
    path = Path(identity["path"])
    if not path.is_file() or _sha256(path.read_bytes()) != identity["file_sha256"]:
        raise ExperimentContractError("formal solution library changed during binding")
    return identity


def _task_solution_library_evidence(
    task: GraphCompressionTask, expected: Mapping[str, str],
) -> dict[str, str]:
    evidence = {
        "path": str(getattr(task, "solution_library_path", "")),
        "file_sha256": str(getattr(task, "solution_library_sha256", "")),
        "action_registry_fingerprint": str(task.registry_fingerprint),
    }
    if evidence != dict(expected):
        raise ExperimentContractError(
            "formal task consumed a different solution library or action registry"
        )
    return evidence


def _validate_checkpoint_solution_library(
    training: Mapping[str, Any], expected: Mapping[str, str],
) -> None:
    extra = training.get("checkpoint_extra")
    if not isinstance(extra, Mapping):
        raise ExperimentContractError(
            "formal GRPO checkpoint lacks immutable extra metadata"
        )
    direct = extra.get("solution_library_identity")
    run_identity = extra.get("run_identity")
    nested = (
        run_identity.get("solution_library_identity")
        if isinstance(run_identity, Mapping) else None
    )
    registry = (
        run_identity.get("action_registry_fingerprint")
        if isinstance(run_identity, Mapping) else None
    )
    if (direct != dict(expected) or nested != dict(expected)
            or registry != expected["action_registry_fingerprint"]):
        raise ExperimentContractError(
            "formal GRPO checkpoint is bound to a different solution library/registry"
        )


def _load_prepared_dataset(
    path: Path, *, verify_source: bool, expected_role: str | None = None,
) -> _PreparedDataset:
    resolved = path.resolve()
    root = resolved if resolved.is_dir() else resolved.parent
    manifest_path = root / "dataset_manifest.json" if resolved.is_dir() else resolved
    if manifest_path.name != "dataset_manifest.json":
        raise ExperimentContractError("prepared manifests must be named dataset_manifest.json")
    try:
        verification = preparation.verify_prepared_dataset(
            root, verify_source=verify_source, expected_role=expected_role,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            preparation.PreparationError) as error:
        raise ExperimentContractError(
            f"prepared dataset verification failed: {manifest_path}"
        ) from error
    if (not isinstance(manifest, dict)
            or verification.get("dataset_id") != manifest.get("dataset_id")):
        raise ExperimentContractError("prepared dataset identity mismatch")
    return _PreparedDataset(root, manifest_path, manifest, verification)


def _reference_for_block(
    table: _ReferenceTable, block: Mapping[str, Any]
) -> tuple[int, float, float]:
    try:
        prepared_bytes = int(block["prepared_block_bytes"])
        dtype = str(block["dtype"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentContractError("prepared block lacks dtype/byte count") from error
    bucket = calibration.block_size_bucket(prepared_bytes)
    try:
        e0, d0 = table.references[(dtype, bucket)]
    except KeyError as error:
        raise ExperimentContractError(
            f"no frozen zlib6 reference for dtype={dtype}, bucket={bucket}"
        ) from error
    return bucket, e0, d0


def _extract_recipe(outcome: TerminalOutcome) -> tuple[int, ...]:
    graph = outcome.operator_graph
    if not isinstance(graph, Mapping):
        raise ExperimentContractError(
            "operator_graph must expose recipe_action_ids for independent replay"
        )
    raw = graph.get("recipe_action_ids")
    if (not isinstance(raw, (list, tuple))
            or any(isinstance(action, bool) or not isinstance(action, (int, np.integer))
                   or int(action) < 0 for action in raw)):
        raise ExperimentContractError("operator_graph.recipe_action_ids is invalid")
    return tuple(int(action) for action in raw)


def _validate_outcome(
    outcome: TerminalOutcome,
    expected_raw_bytes: int,
    expected_e0: float,
    expected_d0: float,
    *,
    require_formal_evidence: bool = False,
) -> dict[str, Any]:
    if not isinstance(outcome, TerminalOutcome):
        raise ExperimentContractError("task evaluate_terminal did not return TerminalOutcome")
    if outcome.metrics.original_numeric_bytes != expected_raw_bytes:
        raise ExperimentContractError("terminal B_raw differs from prepared block contract")
    if (outcome.metrics.compression_reference_ns_per_byte != expected_e0
            or outcome.metrics.decompression_reference_ns_per_byte != expected_d0):
        raise ExperimentContractError("terminal E0/D0 differs from the frozen reference")
    values = (
        outcome.compression_ratio,
        outcome.metrics.compression_ns_per_byte,
        outcome.metrics.decompression_ns_per_byte,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ExperimentContractError("terminal physical metrics are invalid")
    graph = outcome.operator_graph
    if not isinstance(graph, Mapping):
        raise ExperimentContractError("terminal operator_graph must be an object")
    evidence = {
        "self_contained_complete_frame": bool(
            graph.get("complete_frame_includes_all_root_flows") is True
        ),
        "independent_cold_decode": bool(
            graph.get("independent_cold_validation") is True
        ),
        "bit_exact_roundtrip": bool(
            graph.get("schema") == "weight-conditioned-graph-task-v1"
            and graph.get("independent_cold_validation") is True
        ),
        "action_registry_fingerprint": graph.get("registry_fingerprint"),
        "typed_graph_registry": graph.get("typed_graph_registry"),
    }
    if require_formal_evidence:
        if not outcome.compressed_payload:
            raise ExperimentContractError("formal complete frame is empty")
        if not all(evidence[key] is True for key in (
            "self_contained_complete_frame", "independent_cold_decode",
            "bit_exact_roundtrip",
        )):
            raise ExperimentContractError(
                "formal outcome lacks self-contained bit-exact cold-decode evidence"
            )
        fingerprint = evidence["action_registry_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ExperimentContractError(
                "formal outcome lacks an action-registry fingerprint"
            )
        _validate_formal_measurement_evidence(graph)
        evidence["measurement_contract"] = dict(FORMAL_MEASUREMENT_CONTRACT)
    return evidence


def _validate_formal_measurement_evidence(graph: Mapping[str, Any]) -> None:
    """Keep the current Python/CLI timing path outside formal native results."""
    _validate_formal_measurement_contract(
        graph.get("measurement_protocol_schema"), graph.get("measurement_contract"),
    )
    for field in ("compression_timing_boundary", "decompression_timing_boundary"):
        if graph.get(field) != FORMAL_TIMING_BOUNDARY:
            raise ExperimentContractError(f"formal {field} differs from its measurement contract")


def _validate_formal_measurement_contract(protocol: Any, contract: Any) -> None:
    if protocol != FORMAL_MEASUREMENT_PROTOCOL_SCHEMA:
        raise ExperimentContractError(
            "formal measurement requires native in-process complete-frame timing; "
            "the Python/child-process hybrid protocol is not admitted"
        )
    if (not isinstance(contract, Mapping)
            or dict(contract) != FORMAL_MEASUREMENT_CONTRACT):
        raise ExperimentContractError("formal measurement_contract is missing or incompatible")


def _validate_current_runtime_measurement() -> None:
    import weight_conditioned_graph_task_v1 as runtime

    _validate_formal_measurement_contract(
        getattr(runtime, "MEASUREMENT_PROTOCOL_SCHEMA", None),
        getattr(runtime, "FORMAL_MEASUREMENT_CONTRACT", None),
    )


def _outcome_record(
    outcome: TerminalOutcome, weight: tuple[float, float, float],
    *, evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "rho_complete_bytes_per_original_numeric_byte": outcome.compression_ratio,
        "E_compression_ns_per_original_numeric_byte": (
            outcome.metrics.compression_ns_per_byte
        ),
        "D_decompression_ns_per_original_numeric_byte": (
            outcome.metrics.decompression_ns_per_byte
        ),
        "Ehat": outcome.metrics.normalized_compression_time,
        "Dhat": outcome.metrics.normalized_decompression_time,
        "J": objective_value(outcome, weight),
        "original_numeric_bytes": outcome.metrics.original_numeric_bytes,
        "complete_bytes": len(outcome.compressed_payload),
        "complete_payload_sha256": _sha256(outcome.compressed_payload),
        "compression_time_ns": outcome.metrics.compression_time_ns,
        "decompression_time_ns": outcome.metrics.decompression_time_ns,
        "E0_compression_ns_per_original_numeric_byte": (
            outcome.metrics.compression_reference_ns_per_byte
        ),
        "D0_decompression_ns_per_original_numeric_byte": (
            outcome.metrics.decompression_reference_ns_per_byte
        ),
        "operator_graph": _json_safe(outcome.operator_graph),
    }
    if evidence is not None:
        row["correctness_evidence"] = _json_safe(evidence)
    return row


def _payload_artifact_relative_path(
    identity: Mapping[str, Any], payload_sha256: str,
) -> Path:
    """Return a content-addressed, path-safe name for one final complete frame.

    The identity alone is not immutable across checkpoints or ablations.  The
    payload digest prevents a later run for the same block/method/weight from
    overwriting bytes referenced by an earlier result receipt.
    """
    artifact_id = _sha256(_canonical_json({
        "dataset_id": identity["dataset_id"],
        "dataset_provenance_sha256": identity["dataset_provenance_sha256"],
        "block_id": identity["block_id"],
        "entity_id": identity["entity_id"],
        "split": identity["split"],
        "column": identity["column"],
        "dtype": identity["dtype"],
        "block_index": identity["block_index"],
        "series_block_ordinal": identity["series_block_ordinal"],
        "method": identity["method"],
        "weight_id": identity["weight_id"],
        "complete_payload_sha256": payload_sha256,
    }))
    return Path(str(identity["method"])) / str(identity["weight_id"]) / (
        artifact_id + ".bin"
    )


def _write_payload_artifact_atomic(
    root: Path,
    identity: Mapping[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    """Atomically persist one independently encoded self-contained frame."""
    root = root.expanduser().resolve()
    expected_sha256 = _sha256(payload)
    relative = _payload_artifact_relative_path(identity, expected_sha256)
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ExperimentContractError("payload artifact path leaves artifact root") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    written = destination.read_bytes()
    if written != payload or _sha256(written) != expected_sha256:
        raise ExperimentContractError("persisted payload artifact failed byte verification")
    return {
        "artifact_id": relative.stem,
        "relative_path": relative.as_posix(),
        "absolute_path": str(destination),
        "bytes": len(payload),
        "sha256": expected_sha256,
        "content": "independently encoded self-contained complete frame",
    }


def _canonical_action_order(
    _task: GraphCompressionTask,
    _state: Any,
    action: int,
    _weight: tuple[float, float, float],
) -> float:
    """Ablation order: stable action ID only, with no semantic heuristic."""
    return float(action)


def _block_identity(
    dataset: _PreparedDataset,
    block: Mapping[str, Any],
    method: str,
    weight: tuple[float, float, float],
    series_block_ordinal: int,
) -> dict[str, Any]:
    return {
        "dataset_id": str(dataset.manifest["dataset_id"]),
        "dataset_provenance_sha256": str(dataset.manifest["provenance_sha256"]),
        "block_id": str(block.get("block_id", "")),
        "entity_id": str(block.get("entity_id", "")),
        "split": str(block.get("split", "")),
        "column": str(block.get("column", "")),
        "dtype": str(block.get("dtype", "")),
        "block_index": int(block.get("block_index", -1)),
        "series_block_ordinal": series_block_ordinal,
        "method": method,
        "weight_id": _weight_id(weight),
        "weight": list(weight),
    }


def _factory_request(
    *,
    stage: str,
    dataset: _PreparedDataset,
    block: Mapping[str, Any],
    method: str,
    weight: tuple[float, float, float],
    e0: float,
    d0: float,
    previous_recipe: tuple[int, ...],
    selected_recipe: tuple[int, ...],
    series_block_ordinal: int,
    evaluation_budget: int,
    reuse: bool,
    ablation: AblationConfig,
) -> TaskFactoryRequest:
    root = dataset.root.resolve()

    def child(relative: Any) -> str:
        candidate = (root / str(relative)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ExperimentContractError("prepared block path leaves dataset root") from error
        return str(candidate)

    block_index = int(block["block_index"])
    return TaskFactoryRequest(
        stage=stage,
        dataset_id=str(dataset.manifest["dataset_id"]),
        dataset_manifest_path=str(dataset.manifest_path.resolve()),
        prepared_root=str(root),
        provenance_sha256=str(dataset.manifest["provenance_sha256"]),
        block_id=str(block["block_id"]),
        entity_id=str(block["entity_id"]),
        split=str(block["split"]),
        column=str(block["column"]),
        dtype=str(block["dtype"]),
        block_index=block_index,
        series_block_ordinal=series_block_ordinal,
        values_path=child(block["values_path"]),
        missing_bitmap_path=child(block["missing_bitmap_path"]),
        values_sha256=str(block["values_sha256"]),
        missing_bitmap_sha256=str(block["missing_bitmap_sha256"]),
        original_numeric_bytes=int(block["original_numeric_bytes"]),
        prepared_block_bytes=int(block["prepared_block_bytes"]),
        compression_reference_ns_per_byte=e0,
        decompression_reference_ns_per_byte=d0,
        method=method,
        weight=weight,
        previous_recipe=previous_recipe,
        selected_recipe=selected_recipe,
        max_local_edits=(
            0 if stage == "final_independent_encoding"
            else None if series_block_ordinal == 0
            else MAX_CONTINUATION_EDITS
        ),
        evaluation_budget=evaluation_budget,
        reuse_previous_recipe=reuse,
        ablation=ablation,
    )


def _run_search(
    *,
    method: str,
    task: GraphCompressionTask,
    policy: FrozenPolicy | None,
    weight: tuple[float, float, float],
    search_config: SearchConfig,
    bound_provider: Callable[..., Any] | None,
    beam_action_ordering: Callable[..., float] | None,
    ablation: AblationConfig,
) -> SearchResult:
    active_bound = None if ablation.no_certified_prune else bound_provider
    if method == "grpo":
        if policy is None:
            raise ExperimentContractError("GRPO method has no frozen policy")
        return policy_candidate_search(
            policy, task, weight, config=search_config,
            bound_provider=active_bound,
        )
    ordering = (
        _canonical_action_order if ablation.no_heuristic_order
        else beam_action_ordering
    )
    return beam30_search(
        task, weight, config=search_config,
        bound_provider=active_bound, action_ordering=ordering,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "p95": _percentile(values, 0.95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["method"]), str(record["weight_id"]))].append(record)
    output: list[dict[str, Any]] = []
    for (method, weight_id), members in sorted(groups.items()):
        successful = [record for record in members if record["status"] == "success"]
        failures = [record for record in members if record["status"] != "success"]
        recipes: Counter[tuple[int, ...]] = Counter()
        actions: Counter[int] = Counter()
        reused_unchanged = 0
        for record in successful:
            recipe = tuple(int(value) for value in record["selected_recipe"])
            recipes[recipe] += 1
            actions.update(recipe)
            if record["reuse"]["selected_recipe_equals_previous_recipe"]:
                reused_unchanged += 1
        metric_fields = {
            "rho": "rho_complete_bytes_per_original_numeric_byte",
            "E_ns_per_original_numeric_byte": "E_compression_ns_per_original_numeric_byte",
            "D_ns_per_original_numeric_byte": "D_decompression_ns_per_original_numeric_byte",
            "complete_bytes": "complete_bytes",
            "J": "J",
        }
        metrics = {
            label: _stats([float(record["final_independent_encoding"][field_name])
                           for record in successful])
            for label, field_name in metric_fields.items()
        }
        evaluations = _stats([
            float(record["online_selection"]["evaluations"])
            for record in successful
        ])
        timing = {
            field_name: _stats([
                float(record["online_selection"]["prediction_timing_ns"][field_name])
                for record in successful
            ])
            for field_name in TIMING_FIELDS
        }
        total_raw = sum(int(record["final_independent_encoding"]["original_numeric_bytes"])
                        for record in successful)
        total_complete = sum(int(record["final_independent_encoding"]["complete_bytes"])
                             for record in successful)
        failure_reasons = Counter(
            (str(record["failure"]["stage"]), str(record["failure"]["type"]),
             str(record["failure"]["message"]))
            for record in failures
        )
        output.append({
            "method": method,
            "weight_id": weight_id,
            "weight": members[0]["weight"],
            "attempted_blocks": len(members),
            "successful_blocks": len(successful),
            "failed_blocks": len(failures),
            "metrics_across_successful_blocks": metrics,
            "prediction_timing_across_successful_blocks_ns": timing,
            "candidate_evaluations_across_successful_blocks": evaluations,
            "aggregate_complete_bytes": total_complete,
            "aggregate_original_numeric_bytes": total_raw,
            "aggregate_rho": (total_complete / total_raw if total_raw else None),
            "combination_usage": {
                "distinct_selected_recipes": len(recipes),
                "selected_recipe_counts": [
                    {"recipe_action_ids": list(recipe), "blocks": count}
                    for recipe, count in sorted(recipes.items())
                ],
                "operator_action_counts": [
                    {"action_id": action, "selected_recipe_occurrences": count}
                    for action, count in sorted(actions.items())
                ],
                "previous_recipe_reused_unchanged_blocks": reused_unchanged,
            },
            "failure_reasons": [
                {"stage": key[0], "type": key[1], "message": key[2], "blocks": count}
                for key, count in sorted(failure_reasons.items())
            ],
            "summary_statistics": {
                "mean": "arithmetic mean across successful block records",
                "p95": "linear interpolation at index (n-1)*0.95",
            },
        })
    return output


def _column_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize repeated blocks without merging unrelated dataset columns."""
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset_id"]), str(record["entity_id"]),
            str(record["split"]), str(record["column"]), str(record["dtype"]),
            str(record["method"]), str(record["weight_id"]),
        )
        groups[key].append(record)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        summary = _summaries(members)
        if len(summary) != 1:
            raise AssertionError("one column group must produce one method/weight summary")
        output.append({
            "dataset_id": key[0],
            "entity_id": key[1],
            "split": key[2],
            "column": key[3],
            "dtype": key[4],
            **summary[0],
        })
    return output


def _dataset_column_summaries(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate entities only after preserving their auditable detail rows."""
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset_id"]), str(record["split"]),
            str(record["column"]), str(record["dtype"]),
            str(record["method"]), str(record["weight_id"]),
        )
        groups[key].append(record)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        summary = _summaries(members)
        if len(summary) != 1:
            raise AssertionError("one dataset-column group must produce one summary")
        output.append({
            "dataset_id": key[0],
            "split": key[1],
            "column": key[2],
            "dtype": key[3],
            "entity_count": len({str(row["entity_id"]) for row in members}),
            "entity_ids": sorted({str(row["entity_id"]) for row in members}),
            **summary[0],
        })
    return output


def _amortization_curves(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep prediction, final encoding, and decoding costs separate per prefix."""
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] != "success":
            continue
        key = (
            str(record["dataset_id"]), str(record["entity_id"]),
            str(record["split"]), str(record["column"]),
            str(record["dtype"]), str(record["method"]),
            str(record["weight_id"]),
        )
        groups[key].append(record)
    curves: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: int(row["series_block_ordinal"]))
        cumulative_raw = 0
        cumulative_complete = 0
        cumulative_prediction = 0
        cumulative_compression = 0
        cumulative_decompression = 0
        cumulative_weighted_j = 0.0
        points: list[dict[str, Any]] = []
        for row in members:
            final = row["final_independent_encoding"]
            timing = row["online_selection"]["prediction_timing_ns"]
            raw = int(final["original_numeric_bytes"])
            cumulative_raw += raw
            cumulative_complete += int(final["complete_bytes"])
            cumulative_prediction += int(timing["total_prediction_time_ns"])
            cumulative_compression += int(final["compression_time_ns"])
            cumulative_decompression += int(final["decompression_time_ns"])
            cumulative_weighted_j += float(final["J"]) * raw
            points.append({
                "block_count": len(points) + 1,
                "through_block_id": row["block_id"],
                "cumulative_original_numeric_bytes": cumulative_raw,
                "cumulative_complete_bytes": cumulative_complete,
                "cumulative_rho": cumulative_complete / cumulative_raw,
                "cumulative_online_selection_time_ns": cumulative_prediction,
                "cumulative_final_encoding_time_ns": cumulative_compression,
                "cumulative_decompression_time_ns": cumulative_decompression,
                "amortized_online_selection_ns_per_original_numeric_byte": (
                    cumulative_prediction / cumulative_raw
                ),
                "final_encoding_ns_per_original_numeric_byte": (
                    cumulative_compression / cumulative_raw
                ),
                "decompression_ns_per_original_numeric_byte": (
                    cumulative_decompression / cumulative_raw
                ),
                "B_raw_weighted_mean_J": cumulative_weighted_j / cumulative_raw,
            })
        curves.append({
            "dataset_id": key[0], "entity_id": key[1], "split": key[2],
            "column": key[3], "dtype": key[4], "method": key[5],
            "weight_id": key[6], "weight": members[0]["weight"],
            "points": points,
            "cost_boundary": {
                "online_selection": (
                    "features + model/order + transitions + pruning + exact candidate "
                    "evaluations + online task construction"
                ),
                "final_encoding": "one independent encode of the selected complete recipe",
                "training": "excluded here and reported only in training_provenance",
            },
        })
    return curves


def run_experiment(
    prepared_manifests: Sequence[Path],
    reference_table_path: Path,
    task_factory: TaskFactory,
    *,
    policy: FrozenPolicy | None = None,
    checkpoint_path: Path | None = None,
    config: ExperimentConfig = ExperimentConfig(),
    bound_provider: Callable[..., Any] | None = None,
    beam_action_ordering: Callable[..., float] | None = None,
    payload_artifact_root: Path | None = None,
    campaign_plan_path: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen experiment protocol and return a JSON-safe document."""
    if not prepared_manifests:
        raise ValueError("at least one prepared dataset manifest is required")
    if not callable(task_factory):
        raise TypeError("task_factory must be callable")
    frozen, training = _load_frozen_policy(
        methods=config.methods, policy=policy, checkpoint_path=checkpoint_path
    )
    campaign_binding = _load_campaign_binding(
        campaign_plan_path, checkpoint_path=checkpoint_path
    )
    formal_solution_library: dict[str, str] | None = None
    if config.require_formal_evidence:
        if campaign_binding is None:
            raise ExperimentContractError(
                "formal evidence requires an immutable campaign plan"
            )
        if config.splits != ("test",):
            raise ExperimentContractError("formal frozen evaluation must use only test")
        if not config.verify_source:
            raise ExperimentContractError(
                "formal frozen evaluation cannot skip source verification"
            )
        if payload_artifact_root is None:
            raise ExperimentContractError(
                "formal frozen evaluation must persist every complete frame"
            )
        formal_solution_library = _formal_solution_library_identity(
            task_factory, campaign_binding,
        )
        if "grpo" in config.methods:
            _validate_checkpoint_solution_library(
                training, formal_solution_library,
            )
    reference = _load_reference_table(reference_table_path)
    if "test" in config.splits and config.splits != ("test",):
        raise ExperimentContractError("test cannot be mixed with train/validation splits")
    expected_role = "test_candidate" if config.splits == ("test",) else None
    prepared = [
        _load_prepared_dataset(
            path, verify_source=config.verify_source, expected_role=expected_role,
        )
        for path in prepared_manifests
    ]
    if expected_role is not None:
        try:
            preparation.validate_prepared_cohort_manifests(
                [dataset.manifest for dataset in prepared],
                preparation.load_cohort_manifest(), require_complete_train=False,
            )
        except preparation.PreparationError as error:
            raise ExperimentContractError(str(error)) from error
    dataset_ids = [str(item.manifest["dataset_id"]) for item in prepared]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ExperimentContractError("prepared dataset IDs must be unique")
    selected_block_count = sum(
        block.get("split") in config.splits
        for item in prepared
        for block in item.manifest.get("blocks", ())
    )
    if selected_block_count == 0:
        raise ExperimentContractError("selected splits contain no prepared blocks")

    records: list[dict[str, Any]] = []
    previous_recipes: dict[tuple[str, str, str, str, str, str], tuple[int, ...]] = {}
    previous_blocks: dict[tuple[str, str, str, str, str, str], str] = {}
    series_ordinals: dict[tuple[str, str, str, str, str, str], int] = {}
    for dataset in sorted(prepared, key=lambda item: str(item.manifest["dataset_id"])):
        raw_blocks = dataset.manifest.get("blocks")
        if not isinstance(raw_blocks, list):
            raise ExperimentContractError("prepared manifest blocks must be a list")
        blocks = [block for block in raw_blocks if block.get("split") in config.splits]
        blocks.sort(key=lambda block: (
            str(block.get("entity_id", "")), str(block.get("split", "")),
            str(block.get("column", "")), int(block.get("block_index", -1)),
        ))
        for block in blocks:
            for method in config.methods:
                for weight in config.weights:
                    series_key = (
                        method, _weight_id(weight), str(dataset.manifest["dataset_id"]),
                        str(block.get("entity_id", "")), str(block.get("split", "")),
                        str(block.get("column", "")),
                    )
                    series_block_ordinal = series_ordinals.get(series_key, 0)
                    series_ordinals[series_key] = series_block_ordinal + 1
                    identity = _block_identity(
                        dataset, block, method, weight, series_block_ordinal
                    )
                    previous = (() if config.ablation.no_reuse
                                else previous_recipes.get(series_key, ()))
                    previous_block = (None if config.ablation.no_reuse
                                      else previous_blocks.get(series_key))
                    stage = "reference_lookup"
                    reference_record: dict[str, Any] | None = None
                    request: TaskFactoryRequest | None = None
                    final_request: TaskFactoryRequest | None = None
                    online_task_factory_time_ns: int | None = None
                    online_library_evidence: dict[str, str] | None = None
                    final_library_evidence: dict[str, str] | None = None
                    try:
                        bucket, e0, d0 = _reference_for_block(reference, block)
                        reference_record = {
                            "table_calibration_sha256": reference.calibration_sha256,
                            "block_size_bucket_bytes": bucket,
                            "E0_compression_ns_per_original_numeric_byte": e0,
                            "D0_decompression_ns_per_original_numeric_byte": d0,
                        }
                        search_config = SearchConfig.for_block(
                            series_block_ordinal, beam_width=config.beam_width
                        )
                        if config.ablation.no_heuristic_order:
                            search_config = replace(
                                search_config,
                                use_objective_guided_ordering=False,
                            )
                        stage = "online_task_factory_request"
                        request = _factory_request(
                            stage="online_selection", dataset=dataset, block=block,
                            method=method, weight=weight, e0=e0, d0=d0,
                            previous_recipe=previous, selected_recipe=(),
                            series_block_ordinal=series_block_ordinal,
                            evaluation_budget=search_config.max_evaluations,
                            reuse=(previous_block is not None), ablation=config.ablation,
                        )
                        stage = "online_task_factory"
                        factory_started = time.perf_counter_ns()
                        task = task_factory(request)
                        if formal_solution_library is not None:
                            online_library_evidence = _task_solution_library_evidence(
                                task, formal_solution_library,
                            )
                        online_task_factory_time_ns = (
                            time.perf_counter_ns() - factory_started
                        )
                        stage = "online_selection"
                        result = _run_search(
                            method=method, task=task, policy=frozen, weight=weight,
                            search_config=search_config,
                            bound_provider=bound_provider,
                            beam_action_ordering=beam_action_ordering,
                            ablation=config.ablation,
                        )
                        online_evidence = _validate_outcome(
                            result.best_outcome, int(block["original_numeric_bytes"]),
                            e0, d0,
                            require_formal_evidence=config.require_formal_evidence,
                        )
                        selected_recipe = _extract_recipe(result.best_outcome)
                        stage = "final_task_factory_request"
                        final_request = _factory_request(
                            stage="final_independent_encoding", dataset=dataset,
                            block=block, method=method, weight=weight, e0=e0, d0=d0,
                            previous_recipe=selected_recipe,
                            selected_recipe=selected_recipe,
                            series_block_ordinal=series_block_ordinal,
                            evaluation_budget=1, reuse=False,
                            ablation=config.ablation,
                        )
                        stage = "final_task_factory"
                        final_task = task_factory(final_request)
                        if formal_solution_library is not None:
                            final_library_evidence = _task_solution_library_evidence(
                                final_task, formal_solution_library,
                            )
                        stage = "final_independent_encoding"
                        final_outcome = final_task.evaluate_terminal(final_task.initial_state())
                        final_evidence = _validate_outcome(
                            final_outcome, int(block["original_numeric_bytes"]),
                            e0, d0,
                            require_formal_evidence=config.require_formal_evidence,
                        )
                        final_recipe = _extract_recipe(final_outcome)
                        if final_recipe != selected_recipe:
                            raise ExperimentContractError(
                                "independent encoding replayed a different recipe"
                            )
                        payload_matches = (
                            final_outcome.compressed_payload
                            == result.best_outcome.compressed_payload
                        )
                        if config.require_independent_payload_match and not payload_matches:
                            raise ExperimentContractError(
                                "independent encoding complete payload differs from selection"
                            )
                        online = result.json_record()
                        online["prediction_timing_ns"] = online.pop("timing")
                        search_only_total = int(
                            online["prediction_timing_ns"]["total_prediction_time_ns"]
                        )
                        assert online_task_factory_time_ns is not None
                        online["prediction_timing_ns"][
                            "online_task_factory_time_ns"
                        ] = online_task_factory_time_ns
                        online["prediction_timing_ns"][
                            "search_only_total_prediction_time_ns"
                        ] = search_only_total
                        online["prediction_timing_ns"][
                            "total_prediction_time_ns"
                        ] = search_only_total + online_task_factory_time_ns
                        online["best_candidate_measurement"] = _outcome_record(
                            result.best_outcome, weight, evidence=online_evidence
                        )
                        online["evaluated_candidates"] = _json_safe(
                            online["evaluated_candidates"]
                        )
                        final_record = _outcome_record(
                            final_outcome, weight, evidence=final_evidence
                        )
                        final_record["payload_matches_online_selection"] = payload_matches
                        if payload_artifact_root is not None:
                            stage = "final_payload_artifact"
                            final_record["complete_payload_artifact"] = (
                                _write_payload_artifact_atomic(
                                    payload_artifact_root,
                                    identity,
                                    final_outcome.compressed_payload,
                                )
                            )
                        else:
                            final_record["complete_payload_artifact"] = None
                        records.append({
                            **identity,
                            "status": "success",
                            "reference": reference_record,
                            "reuse": {
                                "enabled": not config.ablation.no_reuse,
                                "previous_recipe_source_block_id": previous_block,
                                "previous_recipe_action_ids": list(previous),
                                "max_local_edits": (
                                    None if series_block_ordinal == 0
                                    else MAX_CONTINUATION_EDITS
                                ),
                                "local_search_action_ids": list(result.best_actions),
                                "selected_recipe_equals_previous_recipe": (
                                    previous_block is not None
                                    and selected_recipe == previous
                                ),
                            },
                            "selected_recipe": list(selected_recipe),
                            "online_task_factory_request": request.json_record(),
                            "online_selection": _json_safe(online),
                            "final_task_factory_request": final_request.json_record(),
                            "solution_library_evidence": {
                                "online_selection": online_library_evidence,
                                "final_independent_encoding": final_library_evidence,
                            },
                            "final_independent_encoding": final_record,
                            "failure": None,
                        })
                        if not config.ablation.no_reuse:
                            previous_recipes[series_key] = selected_recipe
                            previous_blocks[series_key] = str(block["block_id"])
                    except Exception as error:
                        if config.fail_fast:
                            raise
                        records.append({
                            **identity,
                            "status": "failed",
                            "reference": reference_record,
                            "reuse": {
                                "enabled": not config.ablation.no_reuse,
                                "previous_recipe_source_block_id": previous_block,
                                "previous_recipe_action_ids": list(previous),
                                "max_local_edits": (
                                    None if series_block_ordinal == 0
                                    else MAX_CONTINUATION_EDITS
                                ),
                            },
                            "selected_recipe": [],
                            "online_task_factory_request": (
                                request.json_record() if request is not None else None
                            ),
                            "online_selection": None,
                            "online_task_factory_time_ns": online_task_factory_time_ns,
                            "final_task_factory_request": (
                                final_request.json_record()
                                if final_request is not None else None
                            ),
                            "solution_library_evidence": {
                                "online_selection": online_library_evidence,
                                "final_independent_encoding": final_library_evidence,
                            },
                            "final_independent_encoding": None,
                            "failure": {
                                "stage": stage,
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        })

    if frozen is not None:
        after = _policy_sha256(frozen)
        training["policy_parameter_sha256_after"] = after
        training["model_parameters_unchanged"] = (
            after == training["policy_parameter_sha256_before"]
        )
        if not training["model_parameters_unchanged"]:
            raise ExperimentContractError("frozen policy parameters changed during search")

    action_registry_fingerprints = sorted({
        str(record["final_independent_encoding"]["correctness_evidence"]
            ["action_registry_fingerprint"])
        for record in records
        if record["status"] == "success"
        and isinstance(record.get("final_independent_encoding"), Mapping)
        and isinstance(record["final_independent_encoding"].get(
            "correctness_evidence"), Mapping)
        and record["final_independent_encoding"]["correctness_evidence"].get(
            "action_registry_fingerprint")
    })
    if config.require_formal_evidence:
        assert campaign_binding is not None
        if action_registry_fingerprints != [
            campaign_binding["action_registry_fingerprint"]
        ]:
            raise ExperimentContractError(
                "formal result action registry differs from campaign admission"
            )

    failures = sum(record["status"] != "success" for record in records)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if failures == 0 else "complete_with_failures",
        "contract": {
            "compression_ratio": "complete_compressed_bytes / original_numeric_bytes",
            "compression_time": "compression_elapsed_ns / original_numeric_bytes",
            "decompression_time": "decompression_elapsed_ns / original_numeric_bytes",
            "optimization_direction": "rho_E_D_all_lower_is_better",
            "objective": "J = a*rho + b*E/(E+E0) + c*D/(D+D0)",
            "weight_vectors": [list(weight) for weight in FIXED_WEIGHTS],
            "first_block_complete_candidate_budget": 64,
            "continuation_complete_candidate_budget": 16,
            "continuation_previous_recipe_max_local_edits": MAX_CONTINUATION_EDITS,
            "beam_width": BEAM_WIDTH,
            "stages": ["training_provenance", "online_selection",
                       "final_independent_encoding"],
            "training_claim": "no training or optimizer update is performed here",
            "online_selection_claim": (
                "prediction includes verified task construction/current-block features, "
                "policy or Beam ordering, local edits, and exact complete-candidate evaluations"
            ),
            "final_encoding_claim": (
                "selected recipe is re-encoded by a new factory task and payload checked"
            ),
            "payload_artifact_claim": (
                "when enabled, only the independently re-encoded self-contained complete "
                "frame is atomically persisted and byte-verified"
            ),
            "excluded_unmeasured_comparisons": [
                "best_of_86_compression_algorithms", "OpenZL", "AMMMO"
            ],
        },
        "config": {
            "methods": list(config.methods),
            "splits": list(config.splits),
            "verify_source": config.verify_source,
            "fail_fast": config.fail_fast,
            "require_independent_payload_match": (
                config.require_independent_payload_match
            ),
            "require_formal_evidence": config.require_formal_evidence,
            "ablation": asdict(config.ablation),
            "effective_certified_pruning": (
                bound_provider is not None and not config.ablation.no_certified_prune
            ),
            "effective_objective_guided_ordering": (
                not config.ablation.no_heuristic_order
            ),
            "effective_beam_ordering": (
                "canonical_action_id_ablation" if config.ablation.no_heuristic_order
                else "injected_ordering" if beam_action_ordering is not None
                else "weight_conditioned_search_v1_default_heuristic"
            ),
            "effective_previous_recipe_reuse": not config.ablation.no_reuse,
        },
        "training_provenance": training,
        "campaign_binding": campaign_binding,
        "solution_library_identity": formal_solution_library,
        "action_registry_fingerprints": action_registry_fingerprints,
        "payload_artifacts": {
            "enabled": payload_artifact_root is not None,
            "root": (
                str(payload_artifact_root.expanduser().resolve())
                if payload_artifact_root is not None else None
            ),
            "successful_artifacts": sum(
                record["status"] == "success"
                and record["final_independent_encoding"][
                    "complete_payload_artifact"
                ] is not None
                for record in records
            ),
        },
        "reference_table": {
            "path": str(reference.path),
            "schema": reference.document["schema"],
            "calibration_sha256": reference.calibration_sha256,
            "reference_group_count": len(reference.references),
        },
        "prepared_datasets": [
            {
                "dataset_id": item.manifest["dataset_id"],
                "manifest_path": str(item.manifest_path.resolve()),
                "provenance_sha256": item.manifest["provenance_sha256"],
                "verification": _json_safe(item.verification),
                "selected_splits": list(config.splits),
                "selected_block_count": sum(
                    block.get("split") in config.splits
                    for block in item.manifest["blocks"]
                ),
            }
            for item in prepared
        ],
        "records": records,
        "summaries": _summaries(records),
        "column_summaries": _column_summaries(records),
        "dataset_column_summaries": _dataset_column_summaries(records),
        "amortization_curves": _amortization_curves(records),
        "cost_phase_separation": {
            "training": "training_provenance only; no optimizer step occurs in this run",
            "online_selection": (
                "online_selection.prediction_timing_ns; excludes final independent encoding"
            ),
            "final_encoding": (
                "final_independent_encoding.compression_time_ns; excludes selection"
            ),
            "decompression": (
                "final_independent_encoding.decompression_time_ns; frame-only timed decode"
            ),
        },
        "attempted_records": len(records),
        "successful_records": len(records) - failures,
        "failed_records": failures,
    }
    document["result_sha256"] = _sha256(_canonical_json(document))
    return document


def write_result_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically save one immutable JSON result protocol document."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_task_factory(specification: str) -> TaskFactory:
    """Load ``module:attribute`` without assigning compression semantics here."""
    if specification.count(":") != 1:
        raise ValueError("task factory must use module:attribute syntax")
    module_name, attribute = specification.split(":")
    if not module_name or not attribute:
        raise ValueError("task factory module and attribute must be non-empty")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("loaded task factory is not callable")
    return factory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="required when --method includes grpo")
    parser.add_argument("--task-factory", required=True,
                        help="importable module:callable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--payload-artifacts", type=Path,
        help=(
            "directory for independently encoded complete frames; defaults to "
            "<output-stem>.payloads beside --output"
        ),
    )
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument(
        "--split", action="append", choices=("train", "validation", "test")
    )
    parser.add_argument("--skip-source-verification", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--campaign-plan", type=Path,
        help="immutable formal campaign plan used to bind cohort/gates/model/registry",
    )
    parser.add_argument("--require-formal-evidence", action="store_true")
    parser.add_argument("--no-certified-prune", action="store_true")
    parser.add_argument("--no-heuristic-order", action="store_true")
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args(argv)

    factory = load_task_factory(args.task_factory)
    config = ExperimentConfig(
        methods=tuple(args.method or METHODS),
        splits=tuple(args.split or ("validation",)),
        verify_source=not args.skip_source_verification,
        fail_fast=args.fail_fast,
        require_formal_evidence=args.require_formal_evidence,
        ablation=AblationConfig(
            no_certified_prune=args.no_certified_prune,
            no_heuristic_order=args.no_heuristic_order,
            no_reuse=args.no_reuse,
        ),
    )
    payload_artifacts = (
        args.payload_artifacts
        if args.payload_artifacts is not None
        else args.output.parent / (args.output.stem + ".payloads")
    )
    document = run_experiment(
        args.prepared,
        args.reference_table,
        factory,
        checkpoint_path=args.checkpoint,
        config=config,
        bound_provider=getattr(factory, "bound_provider", None),
        beam_action_ordering=getattr(factory, "beam_action_ordering", None),
        payload_artifact_root=payload_artifacts,
        campaign_plan_path=args.campaign_plan,
    )
    write_result_atomic(args.output, document)
    print(json.dumps({
        "schema": document["schema"],
        "status": document["status"],
        "result_sha256": document["result_sha256"],
        "successful_records": document["successful_records"],
        "failed_records": document["failed_records"],
        "payload_artifacts": document["payload_artifacts"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0 if document["failed_records"] == 0 else 2


__all__ = [
    "AblationConfig", "ExperimentConfig", "ExperimentContractError",
    "MAX_CONTINUATION_EDITS", "METHODS", "SCHEMA", "TASK_FACTORY_SCHEMA",
    "TaskFactory", "TaskFactoryRequest", "load_task_factory", "run_experiment",
    "write_result_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
