#!/usr/bin/env python3
"""Fail-closed audit and aggregation for the formal weighted-GRPO campaign.

Only independently materialized result files are accepted.  This module does
not execute compression, fill missing rows, relabel historical measurements,
or infer success from a command exit status.  It verifies the frozen campaign
identity, exact test-block/weight coverage, metric identities, persisted
complete frames, and bit-exact independent cold-decode evidence before it
emits any aggregate comparison.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import calibrate_weighted_grpo_references_v1 as calibration
import formal_weighted_grpo_training_v1 as training
import prepared_block_baseline86_v1 as baseline86
import weighted_grpo_experiment_v1 as experiment


SCHEMA = "formal-weighted-grpo-result-aggregation-v1"
DETAIL_SCHEMA = "formal-weighted-grpo-result-details-v1"
SUMMARY_SCHEMA = "formal-weighted-grpo-result-summary-v1"
PAYLOAD_AUDIT_SCHEMA = "complete-payload-audit-v2"
PLAN_SCHEMA = "formal-weighted-grpo-campaign-plan-v1"
WEIGHTS = experiment.FIXED_WEIGHTS
WEIGHT_IDS = tuple(experiment._weight_id(weight) for weight in WEIGHTS)
J_ABS_TOLERANCE = 1e-12
J_REL_TOLERANCE = 1e-9


class AggregationError(RuntimeError):
    """Formal evidence is missing, inconsistent, or not independently auditable."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AggregationError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise AggregationError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2,
                allow_nan=False,
            ).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _verify_embedded_digest(
    document: Mapping[str, Any], field: str, label: str,
) -> str:
    expected = document.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise AggregationError(f"{label} lacks {field}")
    basis = dict(document)
    basis.pop(field, None)
    actual = _sha256_bytes(_canonical_json(basis))
    if expected != actual:
        raise AggregationError(f"{label} {field} mismatch")
    return actual


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AggregationError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AggregationError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < 0.0:
        raise AggregationError(f"{label} must be finite and nonnegative")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AggregationError(f"{label} must be a nonnegative integer")
    return value


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AggregationError(f"{label} does not match its numerator/denominator")


def _load_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _load_json(path.resolve(), "campaign plan")
    if document.get("schema") != PLAN_SCHEMA:
        raise AggregationError("unsupported campaign plan schema")
    plan_sha256 = _verify_embedded_digest(document, "plan_sha256", "campaign plan")
    admission = document.get("admission")
    if not isinstance(admission, Mapping):
        raise AggregationError("campaign plan lacks admission identity")
    required = (
        "corpus_sha256", "calibration_sha256", "gate_bundle_sha256",
        "source_fingerprint", "action_registry_fingerprint", "test_cohort_sha256",
    )
    if any(not isinstance(admission.get(key), str) or not admission.get(key)
           for key in required):
        raise AggregationError("campaign admission identity is incomplete")
    training_prepared = document.get("training_prepared_manifests")
    test_prepared = document.get("test_prepared_manifests")
    if not isinstance(training_prepared, list) or len(training_prepared) != 16:
        raise AggregationError("formal plan must bind exactly 16 training manifests")
    if not isinstance(test_prepared, list) or not test_prepared:
        raise AggregationError("formal plan must bind a non-empty frozen test cohort")
    if document.get("weights") != [list(weight) for weight in WEIGHTS]:
        raise AggregationError("formal plan does not bind the six reporting weights")
    delivery = document.get("delivery_checkpoint")
    if not isinstance(delivery, str) or not delivery:
        raise AggregationError("formal plan lacks delivery checkpoint")
    library_text = document.get("solution_library_manifest")
    if not isinstance(library_text, str) or not library_text:
        raise AggregationError("formal plan lacks solution-library manifest path")
    library = Path(library_text).expanduser()
    if not library.is_absolute() or not library.is_file():
        raise AggregationError("formal solution library is missing")
    library = library.resolve()
    binding = {
        "campaign_plan_path": str(path.resolve()),
        "campaign_plan_sha256": plan_sha256,
        **{key: str(admission[key]) for key in required},
        "delivery_checkpoint": str(Path(delivery).expanduser().resolve()),
        "solution_library_path": str(library),
        "solution_library_file_sha256": _sha256_file(library),
    }
    return document, binding


def _expected_blocks(
    prepared_paths: Sequence[str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    blocks: dict[tuple[str, str], dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for text in prepared_paths:
        path = Path(text).expanduser().resolve()
        manifest_path = path / "dataset_manifest.json" if path.is_dir() else path
        manifest = _load_json(manifest_path, "prepared dataset manifest")
        dataset_id = str(manifest.get("dataset_id", ""))
        digest = str(manifest.get("provenance_sha256", ""))
        if not dataset_id or len(digest) != 64 or dataset_id in provenance:
            raise AggregationError("prepared cohort has invalid/duplicate dataset identity")
        provenance[dataset_id] = digest
        raw_blocks = manifest.get("blocks")
        if not isinstance(raw_blocks, list):
            raise AggregationError("prepared manifest blocks must be a list")
        for row in raw_blocks:
            if not isinstance(row, Mapping) or row.get("split") != "test":
                continue
            key = (dataset_id, str(row.get("block_id", "")))
            if not key[1] or key in blocks:
                raise AggregationError("duplicate/empty prepared test block identity")
            blocks[key] = {
                "dataset_id": dataset_id,
                "dataset_provenance_sha256": digest,
                "block_id": key[1],
                "entity_id": str(row.get("entity_id", "")),
                "split": "test",
                "column": str(row.get("column", "")),
                "dtype": str(row.get("dtype", "")),
                "block_index": int(row.get("block_index", -1)),
                "original_numeric_bytes": int(row.get("original_numeric_bytes", -1)),
                "prepared_block_bytes": int(row.get("prepared_block_bytes", -1)),
            }
    if not blocks:
        raise AggregationError("formal prepared cohort has no test blocks")
    return blocks, provenance


def _load_references(path: Path, calibration_sha256: str) -> dict[tuple[str, int], tuple[float, float]]:
    document = _load_json(path.resolve(), "reference table")
    if document.get("schema") != calibration.SCHEMA:
        raise AggregationError("unsupported reference-table schema")
    if document.get("calibration_sha256") != calibration_sha256:
        raise AggregationError("reference table differs from campaign calibration")
    basis = dict(document)
    basis.pop("calibration_sha256", None)
    if _sha256_bytes(_canonical_json(basis)) != calibration_sha256:
        raise AggregationError("reference-table content digest mismatch")
    result: dict[tuple[str, int], tuple[float, float]] = {}
    for row in document.get("references", ()):
        try:
            key = (str(row["dtype"]), int(row["block_size_bucket_bytes"]))
            values = (
                float(row["E0_compression_ns_per_original_byte"]),
                float(row["D0_decompression_ns_per_original_byte"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AggregationError("invalid reference-table row") from error
        if key in result or any(not math.isfinite(value) or value <= 0 for value in values):
            raise AggregationError("invalid/duplicate reference-table group")
        result[key] = values
    return result


def _artifact(
    artifact: Any, *, root: Path, expected_sha256: str, expected_bytes: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise AggregationError(f"{label} lacks a complete-frame artifact")
    relative = artifact.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise AggregationError(f"{label} artifact lacks relative_path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AggregationError(f"{label} artifact leaves its declared root") from error
    if artifact.get("absolute_path") not in (None, str(path)):
        raise AggregationError(f"{label} artifact absolute/relative paths disagree")
    if not path.is_file():
        raise AggregationError(f"{label} complete-frame artifact is missing")
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256_file(path)
    if (actual_bytes != expected_bytes or actual_sha256 != expected_sha256
            or _nonnegative_int(artifact.get("bytes"), label + " artifact bytes")
            != actual_bytes
            or artifact.get("sha256") != actual_sha256):
        raise AggregationError(f"{label} complete-frame artifact hash/size mismatch")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def _validate_metrics(
    row: Mapping[str, Any], *, raw: int, weight: Sequence[float], label: str,
) -> dict[str, float | int]:
    if raw <= 0 or int(row.get("original_numeric_bytes", -1)) != raw:
        raise AggregationError(f"{label} B_raw differs from prepared valid numeric bytes")
    complete = _nonnegative_int(row.get("complete_bytes"), label + " complete_bytes")
    compression_ns = _nonnegative_int(
        row.get("compression_time_ns"), label + " compression_time_ns"
    )
    decompression_ns = _nonnegative_int(
        row.get("decompression_time_ns"), label + " decompression_time_ns"
    )
    rho = _finite_nonnegative(
        row.get("rho_complete_bytes_per_original_numeric_byte"), label + " rho"
    )
    e_value = _finite_nonnegative(
        row.get("E_compression_ns_per_original_numeric_byte"), label + " E"
    )
    d_value = _finite_nonnegative(
        row.get("D_decompression_ns_per_original_numeric_byte"), label + " D"
    )
    _close(rho, complete / raw, label + " rho")
    _close(e_value, compression_ns / raw, label + " E")
    _close(d_value, decompression_ns / raw, label + " D")
    e0 = _finite_nonnegative(
        row.get("E0_compression_ns_per_original_numeric_byte"), label + " E0"
    )
    d0 = _finite_nonnegative(
        row.get("D0_decompression_ns_per_original_numeric_byte"), label + " D0"
    )
    if e0 <= 0 or d0 <= 0:
        raise AggregationError(f"{label} E0/D0 must be positive")
    ehat = e_value / (e_value + e0)
    dhat = d_value / (d_value + d0)
    objective = weight[0] * rho + weight[1] * ehat + weight[2] * dhat
    _close(float(row.get("Ehat", math.nan)), ehat, label + " Ehat")
    _close(float(row.get("Dhat", math.nan)), dhat, label + " Dhat")
    _close(float(row.get("J", math.nan)), objective, label + " J")
    return {
        "original_numeric_bytes": raw, "complete_bytes": complete,
        "compression_time_ns": compression_ns,
        "decompression_time_ns": decompression_ns,
        "rho": rho, "E": e_value, "D": d_value,
        "E0": e0, "D0": d0, "Ehat": ehat, "Dhat": dhat,
        "J": objective,
    }


def _weighted_expectation(source: str) -> tuple[str, dict[str, bool]]:
    if source == "frozen_grpo":
        return "grpo", {
            "no_certified_prune": False, "no_heuristic_order": False,
            "no_reuse": False,
        }
    if source == "beam30":
        return "beam30", {
            "no_certified_prune": False, "no_heuristic_order": False,
            "no_reuse": False,
        }
    if source == "no_target_pruning_ranking":
        return "grpo", {
            "no_certified_prune": True, "no_heuristic_order": True,
            "no_reuse": False,
        }
    if source == "no_block_reuse":
        return "grpo", {
            "no_certified_prune": False, "no_heuristic_order": False,
            "no_reuse": True,
        }
    raise AggregationError(f"unknown weighted result source: {source}")


def validate_weighted_result(
    path: Path, source: str, *, plan_binding: Mapping[str, Any],
    expected_blocks: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_provenance: Mapping[str, str], delivery_sha256: str,
) -> dict[str, Any]:
    document = _load_json(path.resolve(), source + " result")
    if document.get("schema") != experiment.SCHEMA:
        raise AggregationError(f"{source} has an unsupported result schema")
    result_sha256 = _verify_embedded_digest(document, "result_sha256", source)
    if (document.get("status") != "complete"
            or int(document.get("failed_records", -1)) != 0):
        failures = document.get("records", ())
        missing_reason = any(
            isinstance(row, Mapping) and row.get("status") != "success"
            and not isinstance(row.get("failure"), Mapping)
            for row in failures if isinstance(row, Mapping)
        )
        suffix = " (a failure lacks an explicit reason)" if missing_reason else ""
        raise AggregationError(f"{source} is incomplete or has failures{suffix}")
    method, ablation = _weighted_expectation(source)
    config = document.get("config")
    if not isinstance(config, Mapping):
        raise AggregationError(f"{source} lacks experiment config")
    if (config.get("methods") != [method] or config.get("splits") != ["test"]
            or config.get("ablation") != ablation
            or config.get("verify_source") is not True
            or config.get("require_formal_evidence") is not True):
        raise AggregationError(f"{source} does not implement its formal protocol")
    binding = document.get("campaign_binding")
    if (not isinstance(binding, Mapping)
            or binding.get("solution_library_path")
            != plan_binding.get("solution_library_path")
            or binding.get("solution_library_file_sha256")
            != plan_binding.get("solution_library_file_sha256")):
        raise AggregationError(f"{source} is bound to a different solution library")
    required_binding = (
        "campaign_plan_sha256", "corpus_sha256", "calibration_sha256",
        "gate_bundle_sha256", "source_fingerprint",
        "action_registry_fingerprint", "test_cohort_sha256",
        "solution_library_path", "solution_library_file_sha256",
    )
    if any(
        binding.get(key) != plan_binding.get(key) for key in required_binding
    ):
        raise AggregationError(f"{source} is not bound to this formal campaign")
    if binding.get("delivery_checkpoint_file_sha256") != delivery_sha256:
        raise AggregationError(f"{source} is bound to a different delivery model")
    if document.get("action_registry_fingerprints") != [
        plan_binding["action_registry_fingerprint"]
    ]:
        raise AggregationError(f"{source} action registry is not frozen")
    expected_library_identity = {
        "path": plan_binding["solution_library_path"],
        "file_sha256": plan_binding["solution_library_file_sha256"],
        "action_registry_fingerprint": plan_binding[
            "action_registry_fingerprint"
        ],
    }
    if document.get("solution_library_identity") != expected_library_identity:
        raise AggregationError(f"{source} used a different solution library")
    training_provenance = document.get("training_provenance")
    if not isinstance(training_provenance, Mapping):
        raise AggregationError(f"{source} lacks frozen-model provenance")
    if method == "grpo":
        checkpoint_extra = training_provenance.get("checkpoint_extra")
        run_identity = (
            checkpoint_extra.get("run_identity")
            if isinstance(checkpoint_extra, Mapping) else None
        )
        if (training_provenance.get("model_parameters_frozen") is not True
                or training_provenance.get("model_parameters_unchanged") is not True
                or training_provenance.get("checkpoint_file_sha256") != delivery_sha256
                or int(training_provenance.get("optimizer_steps", -1)) != 0
                or not isinstance(checkpoint_extra, Mapping)
                or checkpoint_extra.get("solution_library_identity")
                != expected_library_identity
                or not isinstance(run_identity, Mapping)
                or run_identity.get("solution_library_identity")
                != expected_library_identity
                or run_identity.get("action_registry_fingerprint")
                != plan_binding["action_registry_fingerprint"]):
            raise AggregationError(f"{source} updated or changed the frozen delivery model")
    elif (training_provenance.get("status") != "not_applicable_beam_only"
          or training_provenance.get("model_parameters_frozen") is not True
          or int(training_provenance.get("optimizer_steps", -1)) != 0):
        raise AggregationError("Beam30 result has invalid no-model training provenance")
    datasets = document.get("prepared_datasets")
    observed_provenance = {
        str(row.get("dataset_id")): str(row.get("provenance_sha256"))
        for row in datasets if isinstance(row, Mapping)
    } if isinstance(datasets, list) else {}
    if observed_provenance != dict(expected_provenance):
        raise AggregationError(f"{source} cohort provenance differs from the plan")
    payloads = document.get("payload_artifacts")
    if not isinstance(payloads, Mapping) or payloads.get("enabled") is not True:
        raise AggregationError(f"{source} did not persist complete frames")
    root_text = payloads.get("root")
    if not isinstance(root_text, str) or not root_text:
        raise AggregationError(f"{source} lacks its payload root")
    artifact_root = Path(root_text).expanduser().resolve()

    records = document.get("records")
    if not isinstance(records, list):
        raise AggregationError(f"{source} records must be a list")
    expected_keys = {
        (dataset_id, block_id, weight_id)
        for dataset_id, block_id in expected_blocks
        for weight_id in WEIGHT_IDS
    }
    expected_sequences: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for block in expected_blocks.values():
        expected_sequences[(str(block["dataset_id"]), str(block["entity_id"]),
                            str(block["column"]), str(block["dtype"]))].append(block)
    expected_ordinals: dict[tuple[str, str], int] = {}
    for members in expected_sequences.values():
        members.sort(key=lambda row: (int(row["block_index"]), str(row["block_id"])))
        for ordinal, block in enumerate(members):
            expected_ordinals[(str(block["dataset_id"]), str(block["block_id"]))] = ordinal
    seen: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("status") != "success":
            raise AggregationError(f"{source} record {index} is not successful")
        key = (
            str(record.get("dataset_id", "")), str(record.get("block_id", "")),
            str(record.get("weight_id", "")),
        )
        if key not in expected_keys or key in seen or record.get("method") != method:
            raise AggregationError(f"{source} has unexpected/duplicate record {key}")
        seen.add(key)
        expected = expected_blocks[(key[0], key[1])]
        if any(record.get(field) != expected[field] for field in (
            "dataset_provenance_sha256", "entity_id", "split", "column", "dtype",
            "block_index",
        )):
            raise AggregationError(f"{source} record identity differs from prepared block")
        weight = tuple(float(value) for value in record.get("weight", ()))
        if weight not in WEIGHTS or experiment._weight_id(weight) != key[2]:
            raise AggregationError(f"{source} record has an invalid weight identity")
        final = record.get("final_independent_encoding")
        online = record.get("online_selection")
        if not isinstance(final, Mapping) or not isinstance(online, Mapping):
            raise AggregationError(f"{source} lacks separated selection/final stages")
        expected_objective_ordering = not ablation["no_heuristic_order"]
        if online.get("objective_guided_ordering") is not expected_objective_ordering:
            raise AggregationError(
                f"{source} objective-guided ordering evidence differs from its ablation"
            )
        evidence = final.get("correctness_evidence")
        if (not isinstance(evidence, Mapping)
                or any(evidence.get(flag) is not True for flag in (
                    "self_contained_complete_frame", "independent_cold_decode",
                    "bit_exact_roundtrip",
                ))
                or evidence.get("action_registry_fingerprint")
                != plan_binding["action_registry_fingerprint"]
                or final.get("payload_matches_online_selection") is not True):
            raise AggregationError(f"{source} lacks bit-exact cold-decode evidence")
        metrics = _validate_metrics(
            final, raw=int(expected["original_numeric_bytes"]), weight=weight,
            label=f"{source}:{key}",
        )
        timing = online.get("prediction_timing_ns")
        if not isinstance(timing, Mapping) or set(experiment.TIMING_FIELDS) - set(timing):
            raise AggregationError(f"{source} prediction timing is incomplete")
        normalized_timing = {
            field: _nonnegative_int(timing[field], f"{source} {field}")
            for field in experiment.TIMING_FIELDS
        }
        if normalized_timing["total_prediction_time_ns"] != (
            normalized_timing["online_task_factory_time_ns"]
            + normalized_timing["search_only_total_prediction_time_ns"]
        ):
            raise AggregationError(f"{source} prediction total has a different boundary")
        evaluations = _nonnegative_int(online.get("evaluations"), source + " evaluations")
        ordinal = _nonnegative_int(
            record.get("series_block_ordinal"), source + " series_block_ordinal"
        )
        if ordinal != expected_ordinals[(key[0], key[1])]:
            raise AggregationError(f"{source} block order differs from chronological manifest order")
        budget = 64 if ordinal == 0 else 16
        declared_budget = _nonnegative_int(
            online.get("evaluation_budget"), source + " evaluation_budget"
        )
        request = record.get("online_task_factory_request")
        if (declared_budget != budget or not isinstance(request, Mapping)
                or _nonnegative_int(
                    request.get("evaluation_budget"),
                    source + " task request evaluation_budget",
                ) != budget
                or _nonnegative_int(
                    request.get("series_block_ordinal"),
                    source + " task request series_block_ordinal",
                ) != ordinal):
            raise AggregationError(
                f"{source} candidate evaluation budget differs from the formal 64/16 contract"
            )
        evaluated_candidates = online.get("evaluated_candidates")
        if (not isinstance(evaluated_candidates, list)
                or len(evaluated_candidates) != evaluations):
            raise AggregationError(
                f"{source} evaluated-candidate evidence does not match evaluations"
            )
        selected_recipe = record.get("selected_recipe")
        if (not isinstance(selected_recipe, list)
                or any(isinstance(action, bool) or not isinstance(action, int)
                       or action < 0 for action in selected_recipe)):
            raise AggregationError(f"{source} selected recipe is invalid")
        reuse = record.get("reuse")
        previous_recipe = (
            reuse.get("previous_recipe_action_ids")
            if isinstance(reuse, Mapping) else None
        )
        expected_max_edits = None if ordinal == 0 else 2
        expected_reuse = (
            ordinal > 0 and not ablation["no_reuse"]
        )
        request_identity = {
            "dataset_id": expected["dataset_id"],
            "provenance_sha256": expected["dataset_provenance_sha256"],
            "block_id": expected["block_id"],
            "entity_id": expected["entity_id"],
            "split": expected["split"],
            "column": expected["column"],
            "dtype": expected["dtype"],
            "block_index": expected["block_index"],
            "original_numeric_bytes": expected["original_numeric_bytes"],
            "prepared_block_bytes": expected["prepared_block_bytes"],
        }
        if (request.get("schema") != experiment.TASK_FACTORY_SCHEMA
                or request.get("stage") != "online_selection"
                or request.get("method") != method
                or request.get("weight") != list(weight)
                or request.get("series_block_ordinal") != ordinal
                or request.get("evaluation_budget") != budget
                or request.get("max_local_edits") != expected_max_edits
                or request.get("reuse_previous_recipe") is not expected_reuse
                or request.get("selected_recipe") not in ([], ())
                or request.get("previous_recipe") != previous_recipe
                or request.get("ablation") != ablation
                or any(request.get(field) != value
                       for field, value in request_identity.items())):
            raise AggregationError(
                f"{source} online TaskFactoryRequest differs from the formal contract"
            )
        final_request = record.get("final_task_factory_request")
        if (not isinstance(final_request, Mapping)
                or final_request.get("schema") != experiment.TASK_FACTORY_SCHEMA
                or final_request.get("stage") != "final_independent_encoding"
                or final_request.get("method") != method
                or final_request.get("weight") != list(weight)
                or final_request.get("series_block_ordinal") != ordinal
                or final_request.get("evaluation_budget") != 1
                or final_request.get("max_local_edits") != 0
                or final_request.get("reuse_previous_recipe") is not False
                or final_request.get("previous_recipe") != selected_recipe
                or final_request.get("selected_recipe") != selected_recipe
                or final_request.get("ablation") != ablation
                or any(final_request.get(field) != value
                       for field, value in request_identity.items())):
            raise AggregationError(
                f"{source} final TaskFactoryRequest differs from the selected recipe"
            )
        library_evidence = record.get("solution_library_evidence")
        if (not isinstance(library_evidence, Mapping)
                or library_evidence.get("online_selection")
                != expected_library_identity
                or library_evidence.get("final_independent_encoding")
                != expected_library_identity):
            raise AggregationError(
                f"{source} record is not bound to the campaign solution library"
            )
        if not 1 <= evaluations <= budget:
            raise AggregationError(f"{source} candidate evaluations exceed formal budget")
        payload_sha256 = str(final.get("complete_payload_sha256", ""))
        artifact = _artifact(
            final.get("complete_payload_artifact"), root=artifact_root,
            expected_sha256=payload_sha256,
            expected_bytes=int(metrics["complete_bytes"]),
            label=f"{source}:{key}",
        )
        normalized.append({
            "source": source,
            "selection_role": "online_prediction_and_bounded_candidate_evaluation",
            **{field: expected[field] for field in (
                "dataset_id", "dataset_provenance_sha256", "block_id", "entity_id",
                "split", "column", "dtype", "block_index",
            )},
            "series_block_ordinal": ordinal,
            "weight_id": key[2], "weight": list(weight),
            "selected_recipe": selected_recipe,
            "metrics": metrics,
            "prediction_timing_ns": normalized_timing,
            "candidate_evaluations": evaluations,
            "candidate_evaluation_budget": declared_budget,
            "complete_frame_artifact": artifact,
            "reuse": record.get("reuse"),
        })
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)[:10]
        raise AggregationError(f"{source} does not cover every test block x six weights: {missing}")
    series: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        series[(str(row["dataset_id"]), str(row["entity_id"]),
                str(row["column"]), str(row["dtype"]),
                str(row["weight_id"]))].append(row)
    for series_key, members in series.items():
        members.sort(key=lambda row: int(row["series_block_ordinal"]))
        if [int(row["series_block_ordinal"]) for row in members] != list(range(len(members))):
            raise AggregationError(f"{source} has a non-contiguous block sequence {series_key}")
        previous_block_id = None
        for row in members:
            reuse = row.get("reuse")
            if not isinstance(reuse, Mapping):
                raise AggregationError(f"{source} lacks block-reuse evidence")
            ordinal = int(row["series_block_ordinal"])
            if ablation["no_reuse"]:
                if (reuse.get("enabled") is not False
                        or reuse.get("previous_recipe_source_block_id") is not None
                        or reuse.get("previous_recipe_action_ids") not in ([], ())):
                    raise AggregationError("no-block-reuse ablation carried previous state")
            else:
                if reuse.get("enabled") is not True:
                    raise AggregationError(f"{source} unexpectedly disabled block reuse")
                if ordinal == 0:
                    if reuse.get("previous_recipe_source_block_id") is not None:
                        raise AggregationError(f"{source} first block used future/prior state")
                elif (reuse.get("previous_recipe_source_block_id") != previous_block_id
                      or reuse.get("max_local_edits") != 2):
                    raise AggregationError(f"{source} continuation reuse contract changed")
            previous_block_id = str(row["block_id"])
    if int(payloads.get("successful_artifacts", -1)) != len(records):
        raise AggregationError(f"{source} payload count does not match records")
    return {
        "source": source, "path": str(path.resolve()),
        "file_sha256": _sha256_file(path.resolve()),
        "evidence_sha256": _sha256_file(path.resolve()),
        "result_sha256": result_sha256, "records": normalized,
        "record_count": len(normalized),
    }


def validate_baseline86_result(
    path: Path, *, expected_blocks: Mapping[tuple[str, str], Mapping[str, Any]],
    references: Mapping[tuple[str, int], tuple[float, float]],
) -> dict[str, Any]:
    root = path.expanduser().resolve()
    summary_path = root / "summary.json" if root.is_dir() else root
    root = summary_path.parent
    summary = _load_json(summary_path, "baseline86 summary")
    if summary.get("schema") != baseline86.SCHEMA:
        raise AggregationError("baseline86 has an unsupported schema")
    live_catalog = baseline86.build_catalog_snapshot()
    if (summary.get("catalog_snapshot_sha256") != live_catalog.snapshot_sha256
            or summary.get("registry_sha256") != live_catalog.registry_sha256
            or summary.get("registry_version") != live_catalog.registry_version):
        raise AggregationError("baseline86 catalog/registry differs from the current frozen catalog")
    expected_labels = {row.label for row in live_catalog.labels}
    block_rows = summary.get("blocks")
    if not isinstance(block_rows, list):
        raise AggregationError("baseline86 summary lacks block receipts")
    seen_blocks: set[tuple[str, str]] = set()
    best_records: list[dict[str, Any]] = []
    measured_artifacts = 0
    measured_label_rows = unsupported_label_rows = failed_label_rows = 0
    receipt_evidence: list[dict[str, Any]] = []
    frame_evidence: list[dict[str, Any]] = []
    for descriptor in block_rows:
        if not isinstance(descriptor, Mapping):
            raise AggregationError("baseline86 block descriptor is invalid")
        key = (str(descriptor.get("dataset_id", "")), str(descriptor.get("block_id", "")))
        if key not in expected_blocks or key in seen_blocks:
            raise AggregationError(f"baseline86 has unexpected/duplicate block {key}")
        seen_blocks.add(key)
        receipt_path = (root / str(descriptor.get("receipt_path", ""))).resolve()
        try:
            receipt_path.relative_to(root)
        except ValueError as error:
            raise AggregationError("baseline86 block receipt leaves output root") from error
        receipt = _load_json(receipt_path, "baseline86 block receipt")
        receipt_evidence.append({
            "dataset_id": key[0], "block_id": key[1],
            "path": str(receipt_path), "sha256": _sha256_file(receipt_path),
        })
        expected = expected_blocks[key]
        if any(receipt.get(field) != expected[field] for field in (
            "dataset_id", "dataset_provenance_sha256", "block_id", "entity_id",
            "split", "column", "dtype", "block_index", "original_numeric_bytes",
        )):
            raise AggregationError(f"baseline86 block identity differs from prepared block {key}")
        labels = receipt.get("label_results")
        if (not isinstance(labels, list) or len(labels) != 86
                or {row.get("label") for row in labels if isinstance(row, Mapping)}
                != expected_labels):
            raise AggregationError(f"baseline86 block {key} does not cover 86 labels")
        measured: list[Mapping[str, Any]] = []
        for row in labels:
            if not isinstance(row, Mapping) or row.get("status") not in {
                "measured", "unsupported", "failed",
            }:
                raise AggregationError(f"baseline86 block {key} has an invalid label row")
            if row["status"] != "measured":
                if row["status"] == "failed":
                    failed_label_rows += 1
                    raise AggregationError(
                        f"baseline86 block {key} contains a failed label result"
                    )
                unsupported_label_rows += 1
                reason = row.get("reason")
                if not isinstance(reason, str) or not reason:
                    raise AggregationError("baseline86 non-measurement lacks an explicit reason")
                continue
            measured_label_rows += 1
            raw = int(expected["original_numeric_bytes"])
            complete = _nonnegative_int(
                row.get("complete_compressed_bytes"), "baseline86 complete bytes"
            )
            compression_ns = _nonnegative_int(
                row.get("compression_elapsed_ns"), "baseline86 compression elapsed ns"
            )
            decompression_ns = _nonnegative_int(
                row.get("decompression_elapsed_ns"), "baseline86 decompression elapsed ns"
            )
            _close(float(row.get("rho_complete_bytes_per_original_numeric_byte", math.nan)),
                   complete / raw, "baseline86 rho")
            _close(float(row.get("E_compression_ns_per_original_numeric_byte", math.nan)),
                   compression_ns / raw, "baseline86 E")
            _close(float(row.get("D_decompression_ns_per_original_numeric_byte", math.nan)),
                   decompression_ns / raw, "baseline86 D")
            if any(row.get(flag) is not True for flag in (
                "self_contained_frame", "independent_cold_decode",
                "bit_exact_values_and_missing_bitmap",
            )):
                raise AggregationError("baseline86 measured row lacks cold roundtrip evidence")
            artifact_root_text = summary.get("complete_frame_artifacts", {}).get("root")
            if not isinstance(artifact_root_text, str) or not artifact_root_text:
                raise AggregationError(
                    "baseline86 does not persist complete frames; hash-only rows are inadmissible"
                )
            artifact_evidence = _artifact(
                row.get("complete_frame_artifact"),
                root=Path(artifact_root_text).expanduser().resolve(),
                expected_sha256=str(row.get("complete_frame_sha256", "")),
                expected_bytes=complete,
                label=f"baseline86:{key}:{row.get('label')}",
            )
            frame_evidence.append({
                "dataset_id": key[0], "block_id": key[1],
                "label": str(row.get("label")), **artifact_evidence,
            })
            measured_artifacts += 1
            measured.append(row)
        if not measured:
            raise AggregationError(f"baseline86 has no measured method for block {key}")
        bucket = calibration.block_size_bucket(int(expected["prepared_block_bytes"]))
        try:
            e0, d0 = references[(str(expected["dtype"]), bucket)]
        except KeyError as error:
            raise AggregationError(f"baseline86 block {key} lacks frozen E0/D0") from error
        for weight in WEIGHTS:
            candidates = []
            for row in measured:
                rho = float(row["rho_complete_bytes_per_original_numeric_byte"])
                e_value = float(row["E_compression_ns_per_original_numeric_byte"])
                d_value = float(row["D_decompression_ns_per_original_numeric_byte"])
                j_value = (
                    weight[0] * rho + weight[1] * e_value / (e_value + e0)
                    + weight[2] * d_value / (d_value + d0)
                )
                candidates.append((j_value, str(row["label"]), row))
            j_value, label, winner = min(candidates, key=lambda item: (item[0], item[1]))
            best_records.append({
                "source": "best_of_86_posthoc",
                "selection_role": "posthoc_best_of_86_reference_not_free_prediction",
                **{field: expected[field] for field in (
                    "dataset_id", "dataset_provenance_sha256", "block_id", "entity_id",
                    "split", "column", "dtype", "block_index",
                )},
                "weight_id": experiment._weight_id(weight), "weight": list(weight),
                "selected_compression_algorithm_label": label,
                "metrics": {
                    "original_numeric_bytes": int(expected["original_numeric_bytes"]),
                    "complete_bytes": int(winner["complete_compressed_bytes"]),
                    "compression_time_ns": int(winner["compression_elapsed_ns"]),
                    "decompression_time_ns": int(winner["decompression_elapsed_ns"]),
                    "rho": float(winner["rho_complete_bytes_per_original_numeric_byte"]),
                    "E": float(winner["E_compression_ns_per_original_numeric_byte"]),
                    "D": float(winner["D_decompression_ns_per_original_numeric_byte"]),
                    "E0": e0, "D0": d0, "J": j_value,
                },
                "prediction_timing": "not_applicable_posthoc_exhaustive_reference",
            })
    if seen_blocks != set(expected_blocks):
        raise AggregationError("baseline86 does not cover every prepared test block")
    summary_file_sha256 = _sha256_file(summary_path)
    evidence_sha256 = _sha256_bytes(_canonical_json({
        "summary_file_sha256": summary_file_sha256,
        "block_receipts": sorted(
            receipt_evidence,
            key=lambda row: (row["dataset_id"], row["block_id"], row["path"]),
        ),
        "complete_frames": sorted(
            frame_evidence,
            key=lambda row: (
                row["dataset_id"], row["block_id"], row["label"], row["path"]
            ),
        ),
    }))
    return {
        "source": "best_of_86_posthoc", "path": str(summary_path),
        "file_sha256": summary_file_sha256,
        "evidence_sha256": evidence_sha256,
        "catalog_snapshot_sha256": summary.get("catalog_snapshot_sha256"),
        "registry_sha256": summary.get("registry_sha256"),
        "measured_complete_frame_artifacts": measured_artifacts,
        "label_coverage": {
            "total_rows": measured_label_rows + unsupported_label_rows,
            "measured_rows": measured_label_rows,
            "unsupported_rows": unsupported_label_rows,
            "failed_rows": failed_label_rows,
            "best_of_86_name_scope": (
                "posthoc best among measured labels; unsupported labels remain explicit"
            ),
        },
        "records": best_records, "record_count": len(best_records),
        "claim_boundary": (
            "best label is selected after measuring eligible labels on the same test block; "
            "it is a posthoc reference and not a zero-cost predictor"
        ),
    }


def _source_hashes(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        name: str(value["evidence_sha256"])
        for name, value in sorted(sources.items())
    }


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def validate_payload_audit(
    path: Path, sources: Mapping[str, Mapping[str, Any]], artifact_count: int,
) -> dict[str, Any]:
    document = _load_json(path.resolve(), "complete payload audit")
    if (document.get("schema") != PAYLOAD_AUDIT_SCHEMA
            or document.get("status") != "passed"
            or document.get("failures") != []):
        raise AggregationError("complete payload audit did not pass cleanly")
    _verify_embedded_digest(document, "audit_sha256", "complete payload audit")
    if document.get("audited_source_evidence_sha256") != _source_hashes(sources):
        raise AggregationError("payload audit is bound to different result files")
    if int(document.get("checked_complete_frame_artifacts", -1)) != artifact_count:
        raise AggregationError("payload audit artifact count differs from aggregation")
    return {
        "path": str(path.resolve()), "file_sha256": _sha256_file(path.resolve()),
        "checked_complete_frame_artifacts": artifact_count,
    }


def _group_summary(records: Sequence[Mapping[str, Any]], *, include_entity: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        key_parts = [str(row["dataset_id"])]
        if include_entity:
            key_parts.append(str(row["entity_id"]))
        key_parts.extend((str(row["column"]), str(row["dtype"]),
                          str(row["source"]), str(row["weight_id"])))
        groups[tuple(key_parts)].append(row)
    output = []
    for key, members in sorted(groups.items()):
        raw = sum(int(row["metrics"]["original_numeric_bytes"]) for row in members)
        complete = sum(int(row["metrics"]["complete_bytes"]) for row in members)
        compression = sum(int(row["metrics"]["compression_time_ns"]) for row in members)
        decompression = sum(int(row["metrics"]["decompression_time_ns"]) for row in members)
        weighted_j = sum(
            float(row["metrics"]["J"]) * int(row["metrics"]["original_numeric_bytes"])
            for row in members
        ) / raw
        offset = 1 if include_entity else 0
        entry = {
            "dataset_id": key[0],
            "column": key[1 + offset], "dtype": key[2 + offset],
            "source": key[3 + offset], "weight_id": key[4 + offset],
            "weight": members[0]["weight"],
            "block_count": len(members), "original_numeric_bytes": raw,
            "complete_bytes": complete, "compression_time_ns": compression,
            "decompression_time_ns": decompression,
            "rho": complete / raw, "E_ns_per_original_numeric_byte": compression / raw,
            "D_ns_per_original_numeric_byte": decompression / raw,
            "B_raw_weighted_mean_block_J": weighted_j,
        }
        if include_entity:
            entry["entity_id"] = key[1]
        else:
            entry["entity_count"] = len({str(row["entity_id"]) for row in members})
        prediction = [
            int(row["prediction_timing_ns"]["total_prediction_time_ns"])
            for row in members if isinstance(row.get("prediction_timing_ns"), Mapping)
        ]
        entry["online_selection_time_ns"] = sum(prediction) if len(prediction) == len(members) else None
        entry["selection_cost_status"] = (
            "measured_all_blocks" if len(prediction) == len(members)
            else "not_applicable_posthoc_reference"
        )
        if len(prediction) == len(members):
            entry["prediction_timing_by_component_ns"] = {
                field: {
                    "total": sum(values),
                    "block_mean": sum(values) / len(values),
                    "block_p95": _percentile(values, 0.95),
                }
                for field in experiment.TIMING_FIELDS
                for values in [[
                    int(row["prediction_timing_ns"][field]) for row in members
                ]]
            }
        else:
            entry["prediction_timing_by_component_ns"] = None
        output.append(entry)
    return output


def _amortization(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if not isinstance(row.get("prediction_timing_ns"), Mapping):
            continue
        groups[(str(row["source"]), str(row["dataset_id"]), str(row["entity_id"]),
                str(row["column"]), str(row["dtype"]), str(row["weight_id"]))].append(row)
    output = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda row: int(row.get("series_block_ordinal", -1)))
        raw = complete = selection = encoding = decoding = 0
        points = []
        for row in members:
            metrics = row["metrics"]
            raw += int(metrics["original_numeric_bytes"])
            complete += int(metrics["complete_bytes"])
            selection += int(row["prediction_timing_ns"]["total_prediction_time_ns"])
            encoding += int(metrics["compression_time_ns"])
            decoding += int(metrics["decompression_time_ns"])
            points.append({
                "block_count": len(points) + 1,
                "through_block_id": row["block_id"],
                "cumulative_original_numeric_bytes": raw,
                "cumulative_complete_bytes": complete,
                "cumulative_online_selection_time_ns": selection,
                "cumulative_final_encoding_time_ns": encoding,
                "cumulative_decompression_time_ns": decoding,
                "rho": complete / raw,
                "selection_ns_per_original_numeric_byte": selection / raw,
                "final_encoding_ns_per_original_numeric_byte": encoding / raw,
                "decompression_ns_per_original_numeric_byte": decoding / raw,
            })
        output.append({
            "source": key[0], "dataset_id": key[1], "entity_id": key[2],
            "column": key[3], "dtype": key[4], "weight_id": key[5],
            "points": points,
        })
    return output


def _comparison(candidate: float, reference: float) -> str:
    tolerance = max(
        J_ABS_TOLERANCE,
        J_REL_TOLERANCE * max(abs(candidate), abs(reference)),
    )
    if candidate < reference - tolerance:
        return "win"
    if candidate > reference + tolerance:
        return "loss"
    return "tie"


def _win_tie_loss(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[tuple[str, str, str], Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        key = (str(row["dataset_id"]), str(row["block_id"]), str(row["weight_id"]))
        by_source[str(row["source"])][key] = row
    candidate = by_source.get("frozen_grpo", {})
    output = []
    for reference_name in sorted(name for name in by_source if name != "frozen_grpo"):
        reference = by_source[reference_name]
        if set(candidate) != set(reference):
            raise AggregationError(f"J comparison coverage differs for {reference_name}")
        counts = {"win": 0, "tie": 0, "loss": 0}
        for key in sorted(candidate):
            result = _comparison(
                float(candidate[key]["metrics"]["J"]),
                float(reference[key]["metrics"]["J"]),
            )
            counts[result] += 1
        output.append({
            "candidate": "frozen_grpo", "reference": reference_name,
            **counts, "compared_block_weight_pairs": len(candidate),
            "direction": "lower_J_is_better",
            "tie_tolerance": {
                "formula": "max(abs_tol, rel_tol*max(abs(J_candidate),abs(J_reference)))",
                "absolute": J_ABS_TOLERANCE, "relative": J_REL_TOLERANCE,
            },
        })
    return output


def _training_cost(
    path: Path, delivery_sha256: str, plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_json(path.resolve(), "training campaign receipt")
    if receipt.get("schema") != training.CAMPAIGN_SCHEMA:
        raise AggregationError("unsupported training campaign receipt")
    _verify_embedded_digest(receipt, "campaign_sha256", "training campaign receipt")
    if receipt.get("delivery_file_sha256") != delivery_sha256:
        raise AggregationError("training receipt is bound to a different delivery model")
    expected_library_identity = {
        "path": plan_binding["solution_library_path"],
        "file_sha256": plan_binding["solution_library_file_sha256"],
        "action_registry_fingerprint": plan_binding[
            "action_registry_fingerprint"
        ],
    }
    if (receipt.get("solution_library_identity") != expected_library_identity
            or receipt.get("action_registry_fingerprint")
            != plan_binding["action_registry_fingerprint"]):
        raise AggregationError(
            "training receipt is bound to a different solution library/registry"
        )
    seed_receipts = receipt.get("seed_receipts")
    if (receipt.get("seeds") != [0, 1, 2]
            or receipt.get("selection_used_test_data") is not False
            or not isinstance(seed_receipts, list) or len(seed_receipts) != 3):
        raise AggregationError("training receipt violates frozen three-seed selection")
    total_training_wall_time_ns = 0
    for row in seed_receipts:
        if not isinstance(row, Mapping):
            raise AggregationError("training seed receipt summary is invalid")
        if row.get("solution_library_identity") != expected_library_identity:
            raise AggregationError("training seeds consumed different solution libraries")
        elapsed = int(row.get("training_wall_time_ns", -1))
        if elapsed < 0:
            raise AggregationError("training wall time is invalid")
        total_training_wall_time_ns += elapsed
    return {
        "receipt_path": str(path.resolve()),
        "receipt_file_sha256": _sha256_file(path.resolve()),
        "formal_seeds": receipt.get("seeds"),
        "per_seed_budget_seconds": receipt.get("per_seed_budget_seconds"),
        "measurement_workers_per_seed": receipt.get("measurement_workers_per_seed"),
        "seed_runs": seed_receipts,
        "total_training_wall_time_ns": total_training_wall_time_ns,
        "delivery_selection": {
            "selected_seed": receipt.get("selected_seed"),
            "selected_validation_score": receipt.get("selected_validation_score"),
            "selection_used_test_data": receipt.get("selection_used_test_data"),
        },
        "claim_boundary": "offline training is reported separately from online selection",
        "solution_library_identity": expected_library_identity,
    }


def aggregate(
    *, campaign_plan: Path, reference_table: Path, training_receipt: Path,
    frozen_grpo: Path, beam30: Path, baseline86_path: Path,
    no_target_pruning_ranking: Path, no_block_reuse: Path,
    payload_audit: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]], int]:
    plan, binding = _load_plan(campaign_plan)
    delivery = Path(binding["delivery_checkpoint"])
    if not delivery.is_file():
        raise AggregationError("delivery checkpoint is missing")
    delivery_sha256 = _sha256_file(delivery)
    binding["delivery_checkpoint_file_sha256"] = delivery_sha256
    expected_blocks, provenance = _expected_blocks(plan["test_prepared_manifests"])
    test_identity = [
        dict(row) for _, row in sorted(expected_blocks.items(), key=lambda item: item[0])
    ]
    if _sha256_bytes(_canonical_json(test_identity)) != binding["test_cohort_sha256"]:
        raise AggregationError("frozen test manifests differ from the admitted test cohort")
    references = _load_references(reference_table, binding["calibration_sha256"])
    sources: dict[str, Mapping[str, Any]] = {}
    for name, path in (
        ("frozen_grpo", frozen_grpo), ("beam30", beam30),
        ("no_target_pruning_ranking", no_target_pruning_ranking),
        ("no_block_reuse", no_block_reuse),
    ):
        sources[name] = validate_weighted_result(
            path, name, plan_binding=binding, expected_blocks=expected_blocks,
            expected_provenance=provenance, delivery_sha256=delivery_sha256,
        )
    sources["best_of_86_posthoc"] = validate_baseline86_result(
        baseline86_path, expected_blocks=expected_blocks, references=references,
    )
    records = [row for source in sources.values() for row in source["records"]]
    artifact_count = sum(
        1 for row in records if isinstance(row.get("complete_frame_artifact"), Mapping)
    ) + int(sources["best_of_86_posthoc"]["measured_complete_frame_artifacts"])
    audit = None
    if payload_audit is not None:
        audit = validate_payload_audit(payload_audit, sources, artifact_count)
    detail = {
        "schema": DETAIL_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_binding": binding,
        "metric_contract": {
            "rho": "complete frame bytes / original valid numeric bytes",
            "E": "compression elapsed ns / same original valid numeric bytes",
            "D": "decompression elapsed ns / same original valid numeric bytes",
            "direction": "rho_E_D_J_all_lower_is_better",
        },
        "source_evidence": {
            name: {
                **{key: value for key, value in source.items() if key != "records"},
                "formal_campaign_binding": {
                    "campaign_plan_sha256": binding["campaign_plan_sha256"],
                    "corpus_sha256": binding["corpus_sha256"],
                    "test_cohort_sha256": binding["test_cohort_sha256"],
                    "gate_bundle_sha256": binding["gate_bundle_sha256"],
                    "source_fingerprint": binding["source_fingerprint"],
                    "action_registry_fingerprint": binding[
                        "action_registry_fingerprint"
                    ],
                    "solution_library_path": binding["solution_library_path"],
                    "solution_library_file_sha256": binding[
                        "solution_library_file_sha256"
                    ],
                    "delivery_checkpoint_file_sha256": binding[
                        "delivery_checkpoint_file_sha256"
                    ],
                },
            }
            for name, source in sources.items()
        },
        "payload_audit": audit,
        "records": records,
        "dataset_entity_column_summaries": _group_summary(records, include_entity=True),
        "dataset_column_summaries": _group_summary(records, include_entity=False),
        "amortization_curves": _amortization(records),
    }
    detail["detail_sha256"] = _sha256_bytes(_canonical_json(detail))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "created_utc": detail["created_utc"],
        "campaign_binding": binding,
        "detail_sha256": detail["detail_sha256"],
        "coverage": {
            "datasets": len(provenance), "test_blocks": len(expected_blocks),
            "weights": [list(weight) for weight in WEIGHTS],
            "weighted_source_records_each": len(expected_blocks) * len(WEIGHTS),
            "sources": {name: int(source["record_count"])
                        for name, source in sources.items()},
        },
        "win_tie_loss_by_block_weight_J": _win_tie_loss(records),
        "training_cost": _training_cost(
            training_receipt, delivery_sha256, binding,
        ),
        "cost_separation": {
            "offline_training": "training_cost only",
            "online_selection": "prediction_timing_ns totals and amortization curves",
            "final_encoding": "compression_time_ns after recipe selection",
            "decompression": "decompression_time_ns from the complete frame",
            "best_of_86": "posthoc reference; its exhaustive evaluation is not free prediction",
        },
        "dataset_column_summaries": detail["dataset_column_summaries"],
        "status": "passed_fail_closed_aggregation",
    }
    summary["summary_sha256"] = _sha256_bytes(_canonical_json(summary))
    return detail, summary, sources, artifact_count


def write_payload_audit(
    path: Path, sources: Mapping[str, Mapping[str, Any]], artifact_count: int,
) -> dict[str, Any]:
    document = {
        "schema": PAYLOAD_AUDIT_SCHEMA, "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audited_source_evidence_sha256": _source_hashes(sources),
        "checked_complete_frame_artifacts": artifact_count,
        "failures": [],
        "claim": "every result artifact used by the aggregate was reopened and hash checked",
    }
    document["audit_sha256"] = _sha256_bytes(_canonical_json(document))
    _atomic_json(path, document)
    return document


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--frozen-grpo", type=Path, required=True)
    parser.add_argument("--beam30", type=Path, required=True)
    parser.add_argument("--baseline86", type=Path, required=True)
    parser.add_argument("--no-target-pruning-ranking", type=Path, required=True)
    parser.add_argument("--no-block-reuse", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit-payloads")
    _common_arguments(audit_parser)
    audit_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    _common_arguments(aggregate_parser)
    aggregate_parser.add_argument("--payload-audit", type=Path, required=True)
    aggregate_parser.add_argument("--detail-output", type=Path, required=True)
    aggregate_parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        detail, summary, sources, artifact_count = aggregate(
            campaign_plan=args.campaign_plan,
            reference_table=args.reference_table,
            training_receipt=args.training_receipt,
            frozen_grpo=args.frozen_grpo, beam30=args.beam30,
            baseline86_path=args.baseline86,
            no_target_pruning_ranking=args.no_target_pruning_ranking,
            no_block_reuse=args.no_block_reuse,
            payload_audit=(args.payload_audit if args.command == "aggregate" else None),
        )
    except AggregationError as error:
        parser.exit(2, f"formal aggregation refused: {error}\n")
    if args.command == "audit-payloads":
        receipt = write_payload_audit(args.output, sources, artifact_count)
        print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve())}))
        return 0
    _atomic_json(args.detail_output, detail)
    _atomic_json(args.summary_output, summary)
    print(json.dumps({
        "status": summary["status"], "detail": str(args.detail_output.resolve()),
        "summary": str(args.summary_output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
