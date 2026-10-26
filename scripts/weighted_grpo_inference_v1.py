#!/usr/bin/env python3
"""Frozen single-block inference for weight-conditioned operator composition.

Input is one verified prepared numeric block, an arbitrary valid objective
weight ``(a,b,c)``, and a frozen policy checkpoint.  The policy orders exact
complete-candidate evaluations; it does not replace physical measurement.
The selected recipe is then replayed by a new factory task and the independently
encoded self-contained frame is atomically written to ``--compressed-output``.

For the first block of a series, omit ``--previous-*`` and the complete-candidate
budget is 64.  For a later block, provide the preceding result receipt, its
externally retained result SHA-256, the preceding block ID, and its complete
recipe.  The receipt hash is verified and must bind the supplied block and
recipe before at most two local edits and 16 complete evaluations are used.
No optimizer update or test-time neural-parameter update occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from weight_conditioned_grpo_v1 import objective_value, validate_weight
from weight_conditioned_search_v1 import SearchConfig
import weighted_grpo_experiment_v1 as experiment


SCHEMA = "weighted-grpo-frozen-single-block-inference-v1"


class InferenceContractError(RuntimeError):
    """A block, continuation, payload, or frozen-model contract was invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> dict[str, Any]:
    destination = path.expanduser().resolve()
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
    if written != payload:
        raise InferenceContractError("persisted complete frame differs from encoder output")
    return {
        "absolute_path": str(destination),
        "bytes": len(written),
        "sha256": _sha256(written),
        "content": "independently encoded self-contained complete frame",
    }


def _recipe(value: Sequence[int] | None, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise InferenceContractError(f"{label} must contain non-negative action IDs")
        result.append(int(raw))
    return tuple(result)


def _sha256_text(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise InferenceContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_previous_result_receipt(
    receipt_path: Path,
    expected_result_sha256: str,
    *,
    expected_block: Mapping[str, Any],
    expected_recipe: tuple[int, ...],
) -> dict[str, Any]:
    """Verify a preceding inference receipt and its continuation bindings.

    ``result_sha256`` is the digest of the canonical receipt with that field
    omitted.  Requiring the caller to supply the expected digest prevents an
    edited receipt from authorizing a different previous block or recipe.
    """
    expected_digest = _sha256_text(
        expected_result_sha256, "previous_result_sha256"
    )
    resolved = receipt_path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        receipt = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceContractError("cannot read previous-result receipt") from error
    if (not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA
            or receipt.get("status") != "complete"):
        raise InferenceContractError("unsupported or incomplete previous-result receipt")
    embedded = _sha256_text(
        receipt.get("result_sha256"), "previous receipt result_sha256"
    )
    basis = {key: value for key, value in receipt.items() if key != "result_sha256"}
    actual = _sha256(_canonical_json(basis))
    if embedded != actual:
        raise InferenceContractError("previous-result receipt self-digest mismatch")
    if expected_digest != embedded:
        raise InferenceContractError("previous-result receipt SHA-256 mismatch")
    if receipt.get("block") != dict(expected_block):
        raise InferenceContractError(
            "previous-result receipt does not bind the expected preceding block"
        )
    if not isinstance(receipt.get("selected_recipe_action_ids"), list):
        raise InferenceContractError(
            "previous-result receipt lacks a selected recipe binding"
        )
    recorded_recipe = _recipe(
        receipt["selected_recipe_action_ids"], "previous receipt selected recipe"
    )
    if recorded_recipe != expected_recipe:
        raise InferenceContractError(
            "previous-result receipt does not bind the supplied previous recipe"
        )
    return {
        "absolute_path": str(resolved),
        "receipt_file_sha256": _sha256(raw),
        "verified_result_sha256": actual,
        "bound_previous_block_id": str(expected_block["block_id"]),
        "bound_previous_recipe_action_ids": list(expected_recipe),
    }


def _default_bound_provider(task: Any, state: Any) -> Any:
    provider = getattr(task, "certified_bound_state", None)
    if not callable(provider):
        raise InferenceContractError(
            "the production task does not expose its certified objective bound"
        )
    return provider(state)


def run_single_block_inference(
    prepared_manifest: Path,
    block_id: str,
    reference_table_path: Path,
    task_factory: experiment.TaskFactory,
    checkpoint_path: Path,
    weight: Sequence[float],
    compressed_output: Path,
    *,
    previous_recipe: Sequence[int] | None = None,
    previous_block_id: str | None = None,
    previous_result_receipt: Path | None = None,
    previous_result_sha256: str | None = None,
    series_block_ordinal: int = 0,
    verify_source: bool = True,
    policy_device: str | None = None,
    bound_provider: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Select, replay, and persist one complete frame with frozen parameters."""
    weights_array = validate_weight(weight)
    weights = tuple(float(value) for value in weights_array)
    if not isinstance(block_id, str) or not block_id:
        raise ValueError("block_id must be non-empty")
    if (isinstance(series_block_ordinal, bool)
            or not isinstance(series_block_ordinal, int)
            or series_block_ordinal < 0):
        raise ValueError("series_block_ordinal must be a non-negative integer")
    continuation = previous_block_id is not None
    if continuation != (series_block_ordinal > 0):
        raise InferenceContractError(
            "continuation requires both a positive series ordinal and previous block ID"
        )
    if previous_block_id is not None and (
        not isinstance(previous_block_id, str) or not previous_block_id
    ):
        raise InferenceContractError("previous_block_id must be a non-empty string")
    if not continuation and previous_recipe is not None:
        raise InferenceContractError("first-block inference cannot carry a previous recipe")
    receipt_supplied = previous_result_receipt is not None
    digest_supplied = previous_result_sha256 is not None
    if not continuation and (receipt_supplied or digest_supplied):
        raise InferenceContractError(
            "first-block inference cannot carry a previous-result receipt"
        )
    if continuation and not (receipt_supplied and digest_supplied):
        raise InferenceContractError(
            "continuation inference requires a previous-result receipt and SHA-256"
        )
    preceding = _recipe(previous_recipe, "previous_recipe")
    if continuation and previous_recipe is None:
        raise InferenceContractError("continuation inference requires the previous recipe")
    if not callable(task_factory):
        raise TypeError("task_factory must be callable")

    frozen, training = experiment._load_frozen_policy(
        methods=("grpo",), policy=None, checkpoint_path=checkpoint_path,
        policy_device=policy_device,
    )
    assert frozen is not None
    dataset = experiment._load_prepared_dataset(
        prepared_manifest, verify_source=verify_source
    )
    matching = [
        block for block in dataset.manifest.get("blocks", ())
        if block.get("block_id") == block_id
    ]
    if len(matching) != 1:
        raise InferenceContractError(
            f"prepared manifest must contain exactly one block_id={block_id!r}"
        )
    block = matching[0]
    reference = experiment._load_reference_table(reference_table_path)
    bucket, e0, d0 = experiment._reference_for_block(reference, block)
    identity = experiment._block_identity(
        dataset, block, "grpo", weights, series_block_ordinal
    )
    previous_receipt_evidence: dict[str, Any] | None = None
    if continuation:
        previous_matches = [
            candidate for candidate in dataset.manifest.get("blocks", ())
            if candidate.get("block_id") == previous_block_id
        ]
        if len(previous_matches) != 1:
            raise InferenceContractError(
                "prepared manifest must contain exactly one previous_block_id"
            )
        previous_identity = experiment._block_identity(
            dataset, previous_matches[0], "grpo", weights,
            series_block_ordinal - 1,
        )
        for field in ("dataset_id", "dataset_provenance_sha256", "entity_id",
                      "split", "column", "dtype", "method", "weight_id"):
            if previous_identity[field] != identity[field]:
                raise InferenceContractError(
                    "previous block is not in the current block's method/weight series"
                )
        if previous_identity["block_index"] >= identity["block_index"]:
            raise InferenceContractError(
                "previous block must precede the current block chronologically"
            )
        if previous_result_receipt is None or previous_result_sha256 is None:
            raise InferenceContractError(
                "continuation receipt arguments disappeared after validation"
            )
        previous_receipt_evidence = _verify_previous_result_receipt(
            Path(previous_result_receipt),
            previous_result_sha256,
            expected_block=previous_identity,
            expected_recipe=preceding,
        )
    search_config = SearchConfig.for_block(series_block_ordinal)
    online_request = experiment._factory_request(
        stage="online_selection", dataset=dataset, block=block,
        method="grpo", weight=weights, e0=e0, d0=d0,
        previous_recipe=preceding, selected_recipe=(),
        series_block_ordinal=series_block_ordinal,
        evaluation_budget=search_config.max_evaluations,
        reuse=continuation, ablation=experiment.AblationConfig(),
    )
    factory_started = time.perf_counter_ns()
    task = task_factory(online_request)
    online_task_factory_time_ns = time.perf_counter_ns() - factory_started
    result = experiment._run_search(
        method="grpo", task=task, policy=frozen, weight=weights,
        search_config=search_config,
        bound_provider=(bound_provider or _default_bound_provider),
        beam_action_ordering=None,
        ablation=experiment.AblationConfig(),
    )
    experiment._validate_outcome(
        result.best_outcome, int(block["original_numeric_bytes"]), e0, d0
    )
    selected_recipe = experiment._extract_recipe(result.best_outcome)
    final_request = experiment._factory_request(
        stage="final_independent_encoding", dataset=dataset, block=block,
        method="grpo", weight=weights, e0=e0, d0=d0,
        previous_recipe=selected_recipe, selected_recipe=selected_recipe,
        series_block_ordinal=series_block_ordinal,
        evaluation_budget=1, reuse=False, ablation=experiment.AblationConfig(),
    )
    final_task = task_factory(final_request)
    final_outcome = final_task.evaluate_terminal(final_task.initial_state())
    experiment._validate_outcome(
        final_outcome, int(block["original_numeric_bytes"]), e0, d0
    )
    if experiment._extract_recipe(final_outcome) != selected_recipe:
        raise InferenceContractError("independent encoding replayed a different recipe")
    if final_outcome.compressed_payload != result.best_outcome.compressed_payload:
        raise InferenceContractError(
            "independent encoding complete payload differs from online selection"
        )
    artifact = _write_bytes_atomic(compressed_output, final_outcome.compressed_payload)
    final_record = experiment._outcome_record(final_outcome, weights)
    final_record["complete_payload_artifact"] = artifact
    online = result.json_record()
    online["prediction_timing_ns"] = online.pop("timing")
    search_only_total = int(
        online["prediction_timing_ns"]["total_prediction_time_ns"]
    )
    online["prediction_timing_ns"][
        "online_task_factory_time_ns"
    ] = online_task_factory_time_ns
    online["prediction_timing_ns"][
        "search_only_total_prediction_time_ns"
    ] = search_only_total
    online["prediction_timing_ns"][
        "total_prediction_time_ns"
    ] = search_only_total + online_task_factory_time_ns
    online["best_candidate_measurement"] = experiment._outcome_record(
        result.best_outcome, weights
    )
    online["evaluated_candidates"] = experiment._json_safe(
        online["evaluated_candidates"]
    )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "contract": {
            "model_parameters_frozen": True,
            "optimizer_steps": 0,
            "weight_scope": "any non-negative finite triple summing to one",
            "objective": "J = a*rho + b*E/(E+E0) + c*D/(D+D0)",
            "first_block_complete_candidate_budget": 64,
            "continuation_complete_candidate_budget": 16,
            "continuation_max_local_recipe_edits": 2,
            "final_output": "independently encoded self-contained complete frame",
            "prediction_time": (
                "verified task construction/current-block features + frozen-model or "
                "ordering work + local edits + exact complete-candidate evaluations"
            ),
        },
        "block": identity,
        "weight": list(weights),
        "reference": {
            "table_calibration_sha256": reference.calibration_sha256,
            "block_size_bucket_bytes": bucket,
            "E0_compression_ns_per_original_numeric_byte": e0,
            "D0_decompression_ns_per_original_numeric_byte": d0,
        },
        "continuation": {
            "enabled": continuation,
            "previous_block_id": previous_block_id,
            "previous_recipe_action_ids": list(preceding),
            "previous_result_receipt": previous_receipt_evidence,
            "series_block_ordinal": series_block_ordinal,
            "max_local_edits": 2 if continuation else None,
        },
        "selected_recipe_action_ids": list(selected_recipe),
        "training_provenance": training,
        "online_task_factory_request": online_request.json_record(),
        "online_selection": experiment._json_safe(online),
        "final_task_factory_request": final_request.json_record(),
        "final_independent_encoding": final_record,
    }
    if not math.isfinite(final_record["J"]):
        raise InferenceContractError("final objective is not finite")
    document["result_sha256"] = _sha256(_canonical_json(document))
    return document


def _parse_recipe(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    if text == "":
        return ()
    try:
        return tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "previous recipe must be a comma-separated action-ID list"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-device", help="cpu or a single cuda:N device")
    parser.add_argument("--task-factory", required=True, help="module:callable")
    parser.add_argument("--weight", type=float, nargs=3, required=True,
                        metavar=("A", "B", "C"))
    parser.add_argument("--compressed-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--previous-recipe",
                        help="comma-separated complete recipe action IDs")
    parser.add_argument("--previous-block-id")
    parser.add_argument("--previous-result-receipt", type=Path)
    parser.add_argument("--previous-result-sha256")
    parser.add_argument("--series-block-ordinal", type=int, default=0)
    parser.add_argument("--skip-source-verification", action="store_true")
    args = parser.parse_args(argv)

    factory = experiment.load_task_factory(args.task_factory)
    document = run_single_block_inference(
        args.prepared, args.block_id, args.reference_table, factory,
        args.checkpoint, args.weight, args.compressed_output,
        previous_recipe=_parse_recipe(args.previous_recipe),
        previous_block_id=args.previous_block_id,
        previous_result_receipt=args.previous_result_receipt,
        previous_result_sha256=args.previous_result_sha256,
        series_block_ordinal=args.series_block_ordinal,
        verify_source=not args.skip_source_verification,
        policy_device=args.policy_device,
        bound_provider=getattr(factory, "bound_provider", None),
    )
    experiment.write_result_atomic(args.result_output, document)
    print(json.dumps({
        "schema": document["schema"],
        "status": document["status"],
        "result_sha256": document["result_sha256"],
        "selected_recipe_action_ids": document["selected_recipe_action_ids"],
        "compressed_output": document["final_independent_encoding"]
        ["complete_payload_artifact"],
        "result_output": str(args.result_output.expanduser().resolve()),
    }, sort_keys=True))
    return 0


__all__ = [
    "InferenceContractError", "SCHEMA", "run_single_block_inference",
]


if __name__ == "__main__":
    raise SystemExit(main())
