#!/usr/bin/env python3
"""Production task factory for formal weighted-GRPO graph experiments.

The factory is intentionally a narrow bridge.  It reads exactly the current
prepared value payload and missing-value bitmap named by a training or frozen
experiment request, verifies both payload hashes, and constructs one
``TypedGraphCompressionTask``.  Compression remains in the typed graph/native
operator runtime; this module contains no compression fallback.

First blocks start from an explicitly frozen training-only solution library and
may spend the 64-candidate budget on multi-step recipe edits.  Continuation
blocks replay the preceding successful recipe verbatim before the policy may
make at most two local edits.  The module-level ``task_factory``
entrypoint freezes its library on first use from
``WEIGHTED_GRPO_SOLUTION_LIBRARY_MANIFEST`` and is suitable for both CLI
orchestrators' ``module:callable`` option.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

import prepare_weighted_grpo_timeseries_v1 as preparation
import typed_graph_runtime as graph_runtime
import weight_conditioned_graph_task_v1 as graph_task
from weight_conditioned_grpo_v1 import validate_weight


SCHEMA = "weighted-grpo-production-task-factory-v1"
SOLUTION_LIBRARY_SCHEMA = "weighted-grpo-training-solution-library-v1"
TRAINING_REQUEST_SCHEMA = "weighted-grpo-training-task-factory-request-v1"
EXPERIMENT_REQUEST_SCHEMA = "weighted-grpo-task-factory-request-v1"
SOLUTION_LIBRARY_ENV = "WEIGHTED_GRPO_SOLUTION_LIBRARY_MANIFEST"
SOLUTION_LIBRARY_SHA256_ENV = "WEIGHTED_GRPO_SOLUTION_LIBRARY_SHA256"
SOLUTION_LIBRARY_FORMAT = "feature-bucket-measured-recipes-v1"
FEATURE_BUCKET_SCHEMA = "typed-root-statistics-decile-v1"

FIRST_BLOCK_EVALUATION_BUDGET = 64
CONTINUATION_EVALUATION_BUDGET = 16
MAX_LOCAL_EDITS = 2
FIRST_BLOCK_MAX_RECIPE_EDITS = graph_task.MAX_GRAPH_NODES


class ProductionTaskFactoryError(RuntimeError):
    """A frozen request, payload, library, or registry contract was invalid."""


@dataclass(frozen=True)
class _LibraryMeasurement:
    block_id: str
    rho: float
    compression_ns_per_byte: float
    decompression_ns_per_byte: float


@dataclass(frozen=True)
class _LibraryBucketEvidence:
    bucket_id: str
    feature_vector: tuple[int, ...]
    measurements: tuple[_LibraryMeasurement, ...]


@dataclass(frozen=True)
class _LibraryRecipe:
    recipe_id: str
    dtype: str
    action_ids: tuple[int, ...]
    source_block_ids: tuple[str, ...]
    bucket_evidence: tuple[_LibraryBucketEvidence, ...] = ()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionTaskFactoryError(f"{label} must be a non-empty string")
    return value


def _valid_sha256(value: Any, label: str) -> str:
    text = _required_text(value, label).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ProductionTaskFactoryError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _recipe_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProductionTaskFactoryError(f"{label} must be an action-ID array")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ProductionTaskFactoryError(f"{label} contains a non-integer action ID")
        action = int(raw)
        if not 0 <= action < graph_task.ACTION_COUNT:
            raise ProductionTaskFactoryError(f"{label} contains an unknown action ID")
        descriptor = graph_task.ACTION_CATALOG[action]
        if (action == graph_task.STOP_ACTION
                or descriptor.category not in {"mechanism", "split_segments", "concat"}):
            raise ProductionTaskFactoryError(
                f"{label} contains STOP, an edit token, or a non-recipe action"
            )
        result.append(action)
    if len(result) > graph_task.MAX_GRAPH_NODES:
        raise ProductionTaskFactoryError(f"{label} exceeds the 64-node graph budget")
    return tuple(result)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def solution_library_feature_bucket(
    dtype: str, values: bytes, bitmap: bytes
) -> tuple[str, tuple[int, ...]]:
    """Return the frozen, label-free data-feature bucket for one current block.

    The descriptor uses only the current prepared payload.  It never inspects a
    split label, a future block, or a measured objective.  Quantising the same
    typed-root features used by the policy makes the manifest stable while
    avoiding false precision in nearest-bucket matching.
    """
    if dtype not in preparation.SUPPORTED_DTYPES:
        raise ProductionTaskFactoryError(f"unsupported prepared dtype {dtype!r}")
    width = int(preparation.SUPPORTED_DTYPES[dtype][0])
    if not values or len(values) % width:
        raise ProductionTaskFactoryError(
            "prepared values are empty or not aligned to the declared dtype"
        )
    roots = (
        graph_runtime.Stream("numeric", width, values),
        graph_runtime.Stream("serial", 1, bitmap),
    )
    features = graph_task._task_features(roots, (dtype, "bytes"))
    # Size plus the seven content-shape statistics: byte entropy, zero/equal
    # bytes, zero/repeated/smooth numeric values, and non-finite fraction.
    selected = (0, 11, 12, 13, 14, 15, 16, 17)
    quantised = tuple(
        max(0, min(20, int(math.floor(float(features[index]) * 20.0 + 1.0e-12))))
        for index in selected
    )
    payload = {
        "schema": FEATURE_BUCKET_SCHEMA,
        "dtype": dtype,
        "quantised_features": list(quantised),
    }
    bucket_id = hashlib.sha256(_canonical_json(payload)).hexdigest()[:24]
    return bucket_id, quantised


def _bucket_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise ProductionTaskFactoryError("solution-library feature vector dimension mismatch")
    return sum(abs(a - b) for a, b in zip(left, right))


def _measurement_objective(
    measurement: _LibraryMeasurement,
    weight: tuple[float, float, float],
    e0: float,
    d0: float,
) -> float:
    a, b, c = weight
    e = measurement.compression_ns_per_byte
    d = measurement.decompression_ns_per_byte
    return a * measurement.rho + b * (e / (e + e0)) + c * (d / (d + d0))


class ProductionTaskFactory:
    """Frozen, callable bridge shared by formal training and test execution."""

    def __init__(
        self,
        solution_library_manifest: Path,
        *,
        expected_library_sha256: str | None = None,
        independent_cold_validation: bool = True,
        timing_warmups: int = 1,
        timing_repetitions: int = 5,
        allow_large_memory: bool = False,
        require_formal_training_ready: bool = True,
    ) -> None:
        manifest_path = Path(solution_library_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise ProductionTaskFactoryError(
                f"training-only solution library is missing: {manifest_path}"
            )
        try:
            raw = manifest_path.read_bytes()
            document = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProductionTaskFactoryError(
                "cannot read the training-only solution library"
            ) from error
        actual_sha256 = _sha256_bytes(raw)
        if expected_library_sha256 is not None:
            expected = _valid_sha256(
                expected_library_sha256, "expected solution-library SHA-256"
            )
            if actual_sha256 != expected:
                raise ProductionTaskFactoryError(
                    "training-only solution-library fingerprint mismatch"
                )
        if not isinstance(document, Mapping):
            raise ProductionTaskFactoryError("solution library must be a JSON object")
        if document.get("schema") != SOLUTION_LIBRARY_SCHEMA:
            raise ProductionTaskFactoryError("unsupported solution-library schema")
        if document.get("source_scope") != "training_only":
            raise ProductionTaskFactoryError(
                "solution library must explicitly declare source_scope=training_only"
            )
        extended = document.get("library_format") == SOLUTION_LIBRARY_FORMAT
        if document.get("library_format") not in {None, SOLUTION_LIBRARY_FORMAT}:
            raise ProductionTaskFactoryError("unsupported solution-library format")
        if extended:
            if document.get("feature_bucket_schema") != FEATURE_BUCKET_SCHEMA:
                raise ProductionTaskFactoryError(
                    "unsupported solution-library feature-bucket schema"
                )
            expected_digest = _valid_sha256(
                document.get("library_content_sha256"),
                "solution-library content SHA-256",
            )
            digest_basis = {
                key: value for key, value in document.items()
                if key != "library_content_sha256"
            }
            if _sha256_bytes(_canonical_json(digest_basis)) != expected_digest:
                raise ProductionTaskFactoryError(
                    "solution-library content fingerprint mismatch"
                )
        registry = _required_text(
            document.get("action_registry_fingerprint"),
            "solution-library action registry fingerprint",
        )
        if registry != graph_task.REGISTRY_FINGERPRINT:
            raise ProductionTaskFactoryError(
                "solution library was built for a different action registry"
            )
        recipes_raw = document.get("recipes")
        defaults_raw = document.get("default_recipe_by_dtype")
        if not isinstance(recipes_raw, list) or not isinstance(defaults_raw, Mapping):
            raise ProductionTaskFactoryError(
                "solution library requires recipes and default_recipe_by_dtype"
            )
        recipes: dict[str, _LibraryRecipe] = {}
        for index, row in enumerate(recipes_raw):
            if not isinstance(row, Mapping):
                raise ProductionTaskFactoryError(f"recipes[{index}] must be an object")
            recipe_id = _required_text(row.get("recipe_id"), f"recipes[{index}].recipe_id")
            if recipe_id in recipes:
                raise ProductionTaskFactoryError("solution-library recipe IDs must be unique")
            dtype = _required_text(row.get("dtype"), f"recipes[{index}].dtype")
            if dtype not in preparation.SUPPORTED_DTYPES:
                raise ProductionTaskFactoryError(
                    f"recipes[{index}] has unsupported dtype {dtype!r}"
                )
            if row.get("source_split") != "train":
                raise ProductionTaskFactoryError(
                    f"recipes[{index}] must be derived only from source_split=train"
                )
            source_blocks = row.get("source_block_ids")
            if (not isinstance(source_blocks, list) or not source_blocks
                    or any(not isinstance(item, str) or not item for item in source_blocks)):
                raise ProductionTaskFactoryError(
                    f"recipes[{index}].source_block_ids must be a non-empty string array"
                )
            bucket_evidence: list[_LibraryBucketEvidence] = []
            if extended:
                evidence_raw = row.get("bucket_evidence")
                if not isinstance(evidence_raw, list) or not evidence_raw:
                    raise ProductionTaskFactoryError(
                        f"recipes[{index}].bucket_evidence must be non-empty"
                    )
                seen_buckets: set[str] = set()
                for evidence_index, evidence in enumerate(evidence_raw):
                    label = f"recipes[{index}].bucket_evidence[{evidence_index}]"
                    if not isinstance(evidence, Mapping):
                        raise ProductionTaskFactoryError(f"{label} must be an object")
                    bucket_id = _required_text(evidence.get("bucket_id"), f"{label}.bucket_id")
                    if bucket_id in seen_buckets:
                        raise ProductionTaskFactoryError(
                            f"{label} duplicates a feature bucket"
                        )
                    seen_buckets.add(bucket_id)
                    feature_vector_raw = evidence.get("feature_vector")
                    if (not isinstance(feature_vector_raw, list)
                            or len(feature_vector_raw) != 8
                            or any(isinstance(value, bool) or not isinstance(value, int)
                                   or not 0 <= value <= 20
                                   for value in feature_vector_raw)):
                        raise ProductionTaskFactoryError(
                            f"{label}.feature_vector is invalid"
                        )
                    measurements_raw = evidence.get("measurements")
                    if not isinstance(measurements_raw, list) or not measurements_raw:
                        raise ProductionTaskFactoryError(
                            f"{label}.measurements must be non-empty"
                        )
                    measurements: list[_LibraryMeasurement] = []
                    for measurement_index, measurement in enumerate(measurements_raw):
                        measurement_label = (
                            f"{label}.measurements[{measurement_index}]"
                        )
                        if not isinstance(measurement, Mapping):
                            raise ProductionTaskFactoryError(
                                f"{measurement_label} must be an object"
                            )
                        block_id = _required_text(
                            measurement.get("block_id"),
                            f"{measurement_label}.block_id",
                        )
                        try:
                            metrics = tuple(float(measurement[field]) for field in (
                                "rho_complete_bytes_per_original_numeric_byte",
                                "E_compression_ns_per_original_numeric_byte",
                                "D_decompression_ns_per_original_numeric_byte",
                            ))
                        except (KeyError, TypeError, ValueError) as error:
                            raise ProductionTaskFactoryError(
                                f"{measurement_label} has invalid physical metrics"
                            ) from error
                        if any(not math.isfinite(value) or value < 0.0 for value in metrics):
                            raise ProductionTaskFactoryError(
                                f"{measurement_label} has invalid physical metrics"
                            )
                        measurements.append(_LibraryMeasurement(
                            block_id=block_id,
                            rho=metrics[0],
                            compression_ns_per_byte=metrics[1],
                            decompression_ns_per_byte=metrics[2],
                        ))
                    bucket_evidence.append(_LibraryBucketEvidence(
                        bucket_id=bucket_id,
                        feature_vector=tuple(int(value) for value in feature_vector_raw),
                        measurements=tuple(measurements),
                    ))
            action_ids = _recipe_tuple(
                row.get("recipe_action_ids"),
                f"recipes[{index}].recipe_action_ids",
            )
            if extended and not action_ids:
                raise ProductionTaskFactoryError(
                    f"recipes[{index}] cannot be an unmeasured STOP-only recipe"
                )
            recipes[recipe_id] = _LibraryRecipe(
                recipe_id=recipe_id,
                dtype=dtype,
                action_ids=action_ids,
                source_block_ids=tuple(source_blocks),
                bucket_evidence=tuple(bucket_evidence),
            )
        defaults: dict[str, str] = {}
        for dtype, recipe_id in defaults_raw.items():
            if dtype not in preparation.SUPPORTED_DTYPES:
                raise ProductionTaskFactoryError(
                    f"default_recipe_by_dtype has unsupported dtype {dtype!r}"
                )
            selected = _required_text(recipe_id, f"default recipe for {dtype}")
            if selected not in recipes or recipes[selected].dtype != dtype:
                raise ProductionTaskFactoryError(
                    f"default recipe for {dtype} is missing or has the wrong dtype"
                )
            defaults[str(dtype)] = selected
        if not defaults:
            raise ProductionTaskFactoryError(
                "solution library must select at least one explicit dtype default"
            )

        if type(independent_cold_validation) is not bool:
            raise TypeError("independent_cold_validation must be boolean")
        if (isinstance(timing_warmups, bool) or not isinstance(timing_warmups, int)
                or not 0 <= timing_warmups <= 31):
            raise ValueError("timing_warmups must be in [0,31]")
        if (isinstance(timing_repetitions, bool) or not isinstance(timing_repetitions, int)
                or not 1 <= timing_repetitions <= 31):
            raise ValueError("timing_repetitions must be in [1,31]")
        if type(allow_large_memory) is not bool:
            raise TypeError("allow_large_memory must be boolean")
        if type(require_formal_training_ready) is not bool:
            raise TypeError("require_formal_training_ready must be boolean")

        self.solution_library_path = manifest_path
        self.solution_library_sha256 = actual_sha256
        self.action_registry_fingerprint = registry
        self.library_format = document.get("library_format")
        self.feature_bucket_schema = document.get("feature_bucket_schema")
        self._recipes = recipes
        self._defaults = defaults
        self._independent_cold_validation = independent_cold_validation
        self._timing_warmups = timing_warmups
        self._timing_repetitions = timing_repetitions
        self._allow_large_memory = allow_large_memory
        self._require_formal_training_ready = require_formal_training_ready

    def _library_recipe(
        self,
        dtype: str,
        *,
        values: bytes | None = None,
        bitmap: bytes | None = None,
        weight: tuple[float, float, float] | None = None,
        e0: float | None = None,
        d0: float | None = None,
    ) -> tuple[_LibraryRecipe, str | None, float | None]:
        if self.library_format == SOLUTION_LIBRARY_FORMAT:
            if values is None or bitmap is None or weight is None or e0 is None or d0 is None:
                raise ProductionTaskFactoryError(
                    "measured solution-library selection requires current block and objective"
                )
            current_bucket, current_vector = solution_library_feature_bucket(
                dtype, values, bitmap
            )
            candidates: list[
                tuple[int, float, tuple[int, ...], str, _LibraryRecipe, str]
            ] = []
            for recipe in self._recipes.values():
                if recipe.dtype != dtype:
                    continue
                for evidence in recipe.bucket_evidence:
                    distance = _bucket_distance(current_vector, evidence.feature_vector)
                    objectives = [
                        _measurement_objective(measurement, weight, e0, d0)
                        for measurement in evidence.measurements
                    ]
                    objective = float(statistics.median(objectives))
                    candidates.append((
                        distance, objective, recipe.action_ids, recipe.recipe_id,
                        recipe, evidence.bucket_id,
                    ))
            if not candidates:
                raise ProductionTaskFactoryError(
                    f"training-only solution library has no measured recipe for dtype {dtype!r}"
                )
            exact = [row for row in candidates if row[5] == current_bucket]
            pool = exact if exact else [
                row for row in candidates
                if row[0] == min(candidate[0] for candidate in candidates)
            ]
            selected = min(pool, key=lambda row: (row[1], row[2], row[3]))
            return selected[4], selected[5], selected[1]
        recipe_id = self._defaults.get(dtype)
        if recipe_id is None:
            raise ProductionTaskFactoryError(
                f"training-only solution library has no default for dtype {dtype!r}"
            )
        return self._recipes[recipe_id], None, None

    @staticmethod
    def _payload(path_value: Any, prepared_root: Path, expected_sha256: Any, label: str) -> bytes:
        path = Path(_required_text(path_value, f"{label} path")).expanduser().resolve()
        try:
            path.relative_to(prepared_root)
        except ValueError as error:
            raise ProductionTaskFactoryError(
                f"{label} path leaves the prepared dataset root"
            ) from error
        if not path.is_file():
            raise ProductionTaskFactoryError(f"missing prepared {label}: {path}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ProductionTaskFactoryError(f"cannot read prepared {label}") from error
        expected = _valid_sha256(expected_sha256, f"{label} SHA-256")
        if _sha256_bytes(payload) != expected:
            raise ProductionTaskFactoryError(f"prepared {label} fingerprint mismatch")
        return payload

    def __call__(self, request: Any) -> graph_task.TypedGraphCompressionTask:
        schema = getattr(request, "schema", None)
        if schema not in {TRAINING_REQUEST_SCHEMA, EXPERIMENT_REQUEST_SCHEMA}:
            raise ProductionTaskFactoryError("unsupported task-factory request schema")
        stage = _required_text(getattr(request, "stage", None), "request stage")
        dtype = _required_text(getattr(request, "dtype", None), "request dtype")
        if dtype not in preparation.SUPPORTED_DTYPES:
            raise ProductionTaskFactoryError(f"unsupported prepared dtype {dtype!r}")
        prepared_root = Path(
            _required_text(getattr(request, "prepared_root", None), "prepared root")
        ).expanduser().resolve()
        if not prepared_root.is_dir():
            raise ProductionTaskFactoryError("prepared root is not a directory")
        values = self._payload(
            getattr(request, "values_path", None), prepared_root,
            getattr(request, "values_sha256", None), "values payload",
        )
        bitmap = self._payload(
            getattr(request, "missing_bitmap_path", None), prepared_root,
            getattr(request, "missing_bitmap_sha256", None), "missing bitmap",
        )
        width = int(preparation.SUPPORTED_DTYPES[dtype][0])
        if not values or len(values) % width:
            raise ProductionTaskFactoryError(
                "prepared values are empty or not aligned to the declared dtype"
            )
        original_numeric_bytes = int(getattr(request, "original_numeric_bytes", -1))
        prepared_block_bytes = int(getattr(request, "prepared_block_bytes", -1))
        if original_numeric_bytes != len(values):
            raise ProductionTaskFactoryError(
                "original_numeric_bytes must equal valid-value payload bytes"
            )
        if prepared_block_bytes != len(values) + len(bitmap):
            raise ProductionTaskFactoryError(
                "prepared_block_bytes must include values and the missing bitmap"
            )

        weight = tuple(float(value) for value in validate_weight(
            getattr(request, "weight", None)
        ))
        e0 = float(getattr(request, "compression_reference_ns_per_byte", math.nan))
        d0 = float(getattr(request, "decompression_reference_ns_per_byte", math.nan))
        if any(not math.isfinite(value) or value <= 0.0 for value in (e0, d0)):
            raise ProductionTaskFactoryError("frozen E0/D0 references must be positive")
        ordinal = int(getattr(request, "series_block_ordinal", -1))
        source_block_index = int(getattr(request, "block_index", -1))
        if ordinal < 0 or source_block_index < 0:
            raise ProductionTaskFactoryError("block indices must be non-negative")
        budget = int(getattr(request, "evaluation_budget", -1))

        library_recipe: _LibraryRecipe | None = None
        library_bucket_id: str | None = None
        library_selection_objective: float | None = None
        base_recipe: tuple[int, ...]
        continuation_edit_mode: bool
        max_recipe_edits = MAX_LOCAL_EDITS
        if schema == TRAINING_REQUEST_SCHEMA:
            expected_registry = _required_text(
                getattr(request, "expected_action_registry_fingerprint", None),
                "expected action registry fingerprint",
            )
            if expected_registry != self.action_registry_fingerprint:
                raise ProductionTaskFactoryError(
                    "training request and solution library use different action registries"
                )
            if stage not in {"identity", "train", "validation"}:
                raise ProductionTaskFactoryError("invalid formal-training request stage")
            if ordinal == 0:
                if (budget != FIRST_BLOCK_EVALUATION_BUDGET
                        or getattr(request, "max_local_edits", None) is not None
                        or getattr(request, "reuse_previous_recipe", None) is not False
                        or tuple(getattr(request, "previous_recipe", ()))
                        or getattr(request, "start_recipe_mode", None)
                        != "training_solution_library_only"):
                    raise ProductionTaskFactoryError(
                        "first training block must use only the budget-64 training library"
                    )
                (library_recipe, library_bucket_id,
                 library_selection_objective) = self._library_recipe(
                    dtype, values=values, bitmap=bitmap, weight=weight, e0=e0, d0=d0,
                )
                base_recipe = library_recipe.action_ids
                max_recipe_edits = FIRST_BLOCK_MAX_RECIPE_EDITS
            else:
                if (budget != CONTINUATION_EVALUATION_BUDGET
                        or getattr(request, "max_local_edits", None) != MAX_LOCAL_EDITS
                        or getattr(request, "reuse_previous_recipe", None) is not True
                        or not getattr(request, "previous_recipe_source_block_id", None)
                        or getattr(request, "start_recipe_mode", None)
                        != "previous_successful_complete_recipe"):
                    raise ProductionTaskFactoryError(
                        "continuation training blocks require budget 16 and two-edit reuse"
                    )
                base_recipe = _recipe_tuple(
                    getattr(request, "previous_recipe", None), "previous_recipe"
                )
            continuation_edit_mode = True
        else:
            if stage == "online_selection":
                if ordinal == 0:
                    if (budget != FIRST_BLOCK_EVALUATION_BUDGET
                            or getattr(request, "max_local_edits", None) is not None
                            or getattr(request, "reuse_previous_recipe", None) is not False
                            or tuple(getattr(request, "previous_recipe", ()))):
                        raise ProductionTaskFactoryError(
                            "first experiment block must use only the budget-64 training library"
                        )
                    (library_recipe, library_bucket_id,
                     library_selection_objective) = self._library_recipe(
                        dtype, values=values, bitmap=bitmap, weight=weight, e0=e0, d0=d0,
                    )
                    base_recipe = library_recipe.action_ids
                    max_recipe_edits = FIRST_BLOCK_MAX_RECIPE_EDITS
                else:
                    if (budget != CONTINUATION_EVALUATION_BUDGET
                            or getattr(request, "max_local_edits", None) != MAX_LOCAL_EDITS):
                        raise ProductionTaskFactoryError(
                            "continuation selection requires budget 16 and at most two edits"
                        )
                    if getattr(request, "reuse_previous_recipe", None) is True:
                        base_recipe = _recipe_tuple(
                            getattr(request, "previous_recipe", None), "previous_recipe"
                        )
                    else:
                        if tuple(getattr(request, "previous_recipe", ())):
                            raise ProductionTaskFactoryError(
                                "no-reuse ablation cannot carry a preceding recipe"
                            )
                        (library_recipe, library_bucket_id,
                         library_selection_objective) = self._library_recipe(
                            dtype, values=values, bitmap=bitmap, weight=weight, e0=e0, d0=d0,
                        )
                        base_recipe = library_recipe.action_ids
                continuation_edit_mode = True
            elif stage == "final_independent_encoding":
                if budget != 1 or getattr(request, "max_local_edits", None) != 0:
                    raise ProductionTaskFactoryError(
                        "final independent encoding requires one fixed evaluation"
                    )
                selected = _recipe_tuple(
                    getattr(request, "selected_recipe", None), "selected_recipe"
                )
                previous = _recipe_tuple(
                    getattr(request, "previous_recipe", None), "previous_recipe"
                )
                if previous != selected:
                    raise ProductionTaskFactoryError(
                        "final independent encoding must replay the exact selected recipe"
                    )
                base_recipe = selected
                continuation_edit_mode = False
                max_recipe_edits = 1
            else:
                raise ProductionTaskFactoryError("invalid frozen-experiment request stage")

        roots = (
            graph_runtime.Stream("numeric", width, values),
            graph_runtime.Stream("serial", 1, bitmap),
        )
        config = graph_task.GraphTaskConfig(
            max_ordinary_depth=graph_task.MAX_ORDINARY_DEPTH,
            max_graph_nodes=graph_task.MAX_GRAPH_NODES,
            max_active_streams=graph_task.MAX_ACTIVE_STREAMS,
            max_policy_edits=(max_recipe_edits if continuation_edit_mode else 1),
            timing_warmups=self._timing_warmups,
            timing_repetitions=self._timing_repetitions,
            independent_cold_validation=self._independent_cold_validation,
            allow_large_memory=self._allow_large_memory,
            require_formal_training_ready=self._require_formal_training_ready,
        )
        task_id = ":".join((
            _required_text(getattr(request, "dataset_id", None), "dataset_id"),
            _required_text(getattr(request, "block_id", None), "block_id"),
            stage,
        ))
        task = graph_task.TypedGraphCompressionTask(
            task_id,
            roots,
            root_views=(dtype, "bytes"),
            compression_reference_ns_per_byte=e0,
            decompression_reference_ns_per_byte=d0,
            objective_weight=weight,
            config=config,
            previous_recipe=base_recipe,
            candidate_budget=graph_task.CandidateBudget(budget, hard_limit=True),
            # This is the chronological ordinal, not the source block number.
            block_index=ordinal,
            original_numeric_bytes=original_numeric_bytes,
            continuation_edit_mode=continuation_edit_mode,
            max_recipe_edits=max_recipe_edits,
        )
        if task.registry_fingerprint != self.action_registry_fingerprint:
            raise ProductionTaskFactoryError("constructed task changed the frozen action registry")
        if task.previous_recipe != base_recipe:
            raise ProductionTaskFactoryError("constructed task did not preserve the base recipe")

        # Evidence attached to the task is descriptive only and is not used by
        # encoding or decoding.  It allows receipts to identify the frozen
        # library and distinguish the source block number from sequence order.
        task.factory_schema = SCHEMA
        task.solution_library_path = str(self.solution_library_path)
        task.solution_library_sha256 = self.solution_library_sha256
        task.solution_library_recipe_id = (
            library_recipe.recipe_id if library_recipe is not None else None
        )
        task.solution_library_feature_bucket_id = library_bucket_id
        task.solution_library_selection_objective = library_selection_objective
        task.source_block_index = source_block_index
        task.series_block_ordinal = ordinal
        task.prepared_values_sha256 = _sha256_bytes(values)
        task.prepared_missing_bitmap_sha256 = _sha256_bytes(bitmap)
        return task


@lru_cache(maxsize=1)
def _factory_from_environment() -> ProductionTaskFactory:
    manifest = os.environ.get(SOLUTION_LIBRARY_ENV)
    if not manifest:
        raise ProductionTaskFactoryError(
            f"set {SOLUTION_LIBRARY_ENV} to an explicit training-only manifest"
        )
    path = Path(manifest).expanduser()
    if not path.is_absolute():
        raise ProductionTaskFactoryError(
            f"{SOLUTION_LIBRARY_ENV} must be an absolute path"
        )
    expected_sha = os.environ.get(SOLUTION_LIBRARY_SHA256_ENV)
    return ProductionTaskFactory(path, expected_library_sha256=expected_sha)


def task_factory(request: Any) -> graph_task.TypedGraphCompressionTask:
    """CLI ``module:callable`` entrypoint for identity/train/validation/test."""
    return _factory_from_environment()(request)


def solution_library_identity() -> dict[str, str]:
    """Return the exact training-only library consumed by the cached factory."""
    factory = _factory_from_environment()
    return {
        "path": str(factory.solution_library_path),
        "file_sha256": factory.solution_library_sha256,
        "action_registry_fingerprint": factory.action_registry_fingerprint,
    }


def bound_provider(
    task: graph_task.TypedGraphCompressionTask, state: graph_task.TaskState
) -> graph_task.BoundState:
    """Expose the task's suffix-safe lower bound to the experiment CLI."""
    if not isinstance(task, graph_task.TypedGraphCompressionTask):
        raise ProductionTaskFactoryError(
            "bound_provider requires a TypedGraphCompressionTask"
        )
    return task.certified_bound_state(state)


# ``weighted_grpo_experiment_v1.load_task_factory`` returns the named callable,
# so CLI capability discovery occurs on this function rather than its module.
task_factory.bound_provider = bound_provider  # type: ignore[attr-defined]
task_factory.solution_library_identity = solution_library_identity  # type: ignore[attr-defined]


__all__ = [
    "CONTINUATION_EVALUATION_BUDGET",
    "EXPERIMENT_REQUEST_SCHEMA",
    "FIRST_BLOCK_EVALUATION_BUDGET",
    "MAX_LOCAL_EDITS",
    "ProductionTaskFactory",
    "ProductionTaskFactoryError",
    "SCHEMA",
    "SOLUTION_LIBRARY_ENV",
    "SOLUTION_LIBRARY_FORMAT",
    "FEATURE_BUCKET_SCHEMA",
    "SOLUTION_LIBRARY_SCHEMA",
    "SOLUTION_LIBRARY_SHA256_ENV",
    "TRAINING_REQUEST_SCHEMA",
    "bound_provider",
    "solution_library_identity",
    "solution_library_feature_bucket",
    "task_factory",
]
