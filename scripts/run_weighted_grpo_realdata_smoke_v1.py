#!/usr/bin/env python3
"""Non-formal real-training-block GRPO interface/resource smoke.

The raw-only start, two-operator horizon and one greedy proposal per weight are
diagnostic limits, not the formal solution library or 64-candidate experiment.
No validation/test values enter policy observations, rewards or calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import calibrate_weighted_grpo_references_v1 as calibration
import prepare_weighted_grpo_timeseries_v1 as preparation
import typed_graph_runtime as runtime
import weight_conditioned_graph_task_v1 as graph
from weight_conditioned_grpo_v1 import (
    FIXED_WEIGHTS, FrozenPolicy, GRPOConfig, initialize_policy, load_checkpoint,
    objective_value, propose_frozen, save_checkpoint, train_grpo,
)

SCHEMA = "weighted-grpo-realdata-resource-smoke-v1"
DEFAULT_ADMISSION = Path(__file__).resolve().parents[1] / (
    "analysis_outputs/weight_conditioned_grpo_20260917_v1/data_audit/"
    "training_source_admissibility_v1.json"
)


class SmokeError(RuntimeError):
    pass


class _RecordingTask(graph.TypedGraphCompressionTask):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.terminal_outcomes: list[Any] = []

    def evaluate_terminal(self, state: Any) -> Any:
        outcome = super().evaluate_terminal(state)
        self.terminal_outcomes.append(outcome)
        return outcome


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def choose_training_block(
    manifest: Mapping[str, Any], *, block_id: str | None, max_block_bytes: int
) -> Mapping[str, Any]:
    if manifest.get("cohort", {}).get("role") != "train_validation":
        raise SmokeError("smoke requires an explicitly bound train_validation dataset")
    eligible = [
        row for row in manifest.get("blocks", [])
        if row.get("split") == "train"
        and 0 < int(row.get("original_numeric_bytes", 0)) <= max_block_bytes
        and row.get("dtype") in {"int32", "int64", "float32", "float64"}
        and (block_id is None or row.get("block_id") == block_id)
    ]
    if not eligible:
        raise SmokeError("no eligible complete training block; no truncation or split substitution")
    # Prefer a measured value column; the timestamp remains eligible if explicitly selected.
    return min(eligible, key=lambda row: (
        str(row.get("column", "")).startswith("timestamp"),
        int(row["original_numeric_bytes"]), str(row["block_id"]),
    ))


def load_training_input(
    prepared: Path, admission: Path, *, block_id: str | None, max_block_bytes: int
) -> tuple[dict[str, Any], bytes, bytes]:
    root = prepared.resolve() if prepared.is_dir() else prepared.resolve().parent
    manifest_path = root / "dataset_manifest.json"
    verification = preparation.verify_prepared_dataset(
        root, verify_source=True, expected_role="train_validation",
    )
    manifest = json.loads(manifest_path.read_bytes())
    preparation.validate_prepared_cohort_manifests(
        [manifest], preparation.load_cohort_manifest(), require_complete_train=False,
    )
    admission_document = json.loads(admission.read_bytes())
    matches = [row for row in admission_document.get("sources", [])
               if row.get("dataset_id") == manifest.get("dataset_id")]
    if len(matches) != 1 or matches[0].get("status") != "approved":
        raise SmokeError("dataset is not approved in the source admission audit")
    if matches[0].get("source_group") != manifest.get("source_group"):
        raise SmokeError("source admission group differs from prepared source")
    if matches[0].get("revision") != manifest.get("source", {}).get("upstream_revision"):
        raise SmokeError("source admission revision differs from prepared source")
    block = choose_training_block(manifest, block_id=block_id, max_block_bytes=max_block_bytes)
    values_path = calibration._safe_child(root, str(block["values_path"]))
    bitmap_path = calibration._safe_child(root, str(block["missing_bitmap_path"]))
    values, bitmap = values_path.read_bytes(), bitmap_path.read_bytes()
    if _hash(values) != block["values_sha256"] or _hash(bitmap) != block["missing_bitmap_sha256"]:
        raise SmokeError("selected training payload changed after prepared verification")
    if len(values) != int(block["original_numeric_bytes"]):
        raise SmokeError("valid-value denominator differs from numeric payload size")
    record = {
        "dataset_id": manifest["dataset_id"], "source_group": manifest["source_group"],
        "manifest_path": str(manifest_path), "manifest_sha256": _hash(manifest_path.read_bytes()),
        "admission_path": str(admission.resolve()), "admission_sha256": _hash(admission.read_bytes()),
        "verification": verification, "block": dict(block),
        "model_data_scope": "one_complete_train_block_only",
        "verification_may_read_other_splits": True,
    }
    return record, values, bitmap


def parameter_sha256(policy: Any) -> str:
    return _hash(b"".join(np.ascontiguousarray(value, dtype="<f8").tobytes()
                         for value in policy.params))


def run_smoke(
    prepared: Path, admission: Path, zlib6_cli: Path, output: Path, *,
    seed: int = 17, block_id: str | None = None, max_block_bytes: int = 16384,
) -> dict[str, Any]:
    if not 1 <= max_block_bytes <= 65536:
        raise SmokeError("resource smoke block cap must be in 1..65536 numeric bytes")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    total_started = time.perf_counter_ns()
    _event("verify_training_input")
    source, values, bitmap = load_training_input(
        prepared, admission, block_id=block_id, max_block_bytes=max_block_bytes,
    )
    calibration.write_atomic(output / "input.json", source)
    raw_bytes = len(values)
    _event("calibrate_zlib6", dataset_id=source["dataset_id"], raw_bytes=raw_bytes)
    native = calibration._run_native(zlib6_cli, values + bitmap, warmups=1, repeats=5)
    e0 = native["compression_median_ns"] / raw_bytes
    d0 = native["decompression_median_ns"] / raw_bytes
    if not all(math.isfinite(value) and value > 0 for value in (e0, d0)):
        raise SmokeError("same-input measured zlib6 references must be positive")
    references = {
        "scope": "same_train_block_nonformal_smoke", "E0": e0, "D0": d0,
        "unit": "ns/original_numeric_byte", "native_receipt": native,
        "native_cli": str(zlib6_cli.resolve()), "native_cli_sha256": _hash(zlib6_cli.read_bytes()),
        "payload_sha256": _hash(values + bitmap), "original_numeric_bytes": raw_bytes,
        "reference_boundary": "in_process_native_ZR61_frame",
        "graph_boundary_matches_reference": False,
    }
    calibration.write_atomic(output / "references.json", references)
    width = preparation.SUPPORTED_DTYPES[source["block"]["dtype"]][0]
    roots = (runtime.Stream("numeric", width, values), runtime.Stream("serial", 1, bitmap))
    config = graph.GraphTaskConfig(
        max_policy_edits=2, timing_warmups=1, timing_repetitions=5,
        independent_cold_validation=True, require_formal_training_ready=False,
    )

    def make_task(suffix: str, recipe: Sequence[int] = ()) -> _RecordingTask:
        return _RecordingTask(
            f"{source['dataset_id']}:{source['block']['block_id']}:{suffix}", roots,
            root_views=(source["block"]["dtype"], "bytes"), config=config,
            compression_reference_ns_per_byte=e0, decompression_reference_ns_per_byte=d0,
            previous_recipe=recipe, original_numeric_bytes=raw_bytes,
        )

    _event("initialize_native_graph")
    training_task = make_task("train")
    initial = initialize_policy([training_task], seed=seed, hidden_sizes=(128, 128))
    before = parameter_sha256(initial)
    training_config = GRPOConfig(groups_per_iteration=2)
    _event("train_start", groups=2, group_size=8, iterations=1)
    training_started = time.perf_counter_ns()
    trained = train_grpo([training_task], seed=seed, iterations=1,
                         config=training_config, policy=initial)
    training_ns = time.perf_counter_ns() - training_started
    after = parameter_sha256(trained.policy)
    trace = trained.trace[0]
    if before == after or trace["parameter_change_l2"] <= 0 or trace["optimizer_steps"] != 4:
        raise SmokeError("real GRPO optimizer did not change model parameters")
    if trace["fixed_weight_groups"] != 1 or trace["dirichlet_weight_groups"] != 1:
        raise SmokeError("smoke must train one fixed-weight and one continuous-weight group")
    if len(training_task.terminal_outcomes) != 16:
        raise SmokeError("one training iteration must measure exactly 16 terminal candidates")
    mechanisms_executed = sum(
        any(graph.ACTION_CATALOG[action].category == "mechanism"
            for action in outcome.operator_graph["recipe_action_ids"])
        for outcome in training_task.terminal_outcomes
    )
    if mechanisms_executed == 0:
        raise SmokeError("training did not execute a native mechanism candidate")
    checkpoint = output / "smoke_policy.json"
    checkpoint_payload_hash = save_checkpoint(trained, checkpoint, extra={
        "scope": SCHEMA, "formal_training_eligible": False,
        "starting_solution": "raw_only_diagnostic_not_formal_solution_library",
        "source_manifest_sha256": source["manifest_sha256"],
    })
    calibration.write_atomic(output / "training_trace.json", {
        "trace": list(trained.trace), "training_walltime_ns": training_ns,
        "parameter_sha256_before": before, "parameter_sha256_after": after,
        "terminal_graphs": [outcome.operator_graph for outcome in training_task.terminal_outcomes],
        "native_mechanism_terminal_candidates": mechanisms_executed,
    })
    loaded = load_checkpoint(checkpoint, expected_registry_fingerprint=graph.REGISTRY_FINGERPRINT,
                             expected_data_fingerprints={training_task.task_id: training_task.data_fingerprint})
    frozen = FrozenPolicy(loaded.policy)
    frozen_hash = parameter_sha256(frozen)
    rows = []
    for weight_index, weight in enumerate(FIXED_WEIGHTS):
        _event("frozen_inference_start", weight=list(weight))
        selection_started = time.perf_counter_ns()
        inference_task = make_task(f"inference{weight_index}")
        proposal = propose_frozen(frozen, inference_task, weight, greedy=True, seed=seed)
        selection_ns = time.perf_counter_ns() - selection_started
        recipe = tuple(proposal.operator_graph["recipe_action_ids"])
        final_task = make_task(f"final{weight_index}", recipe)
        final = final_task.evaluate_terminal(final_task.initial_state())
        if final.compressed_payload != proposal.compressed_payload:
            raise SmokeError("independent final replay changed compressed frame bytes")
        frame_path = output / f"weight_{weight_index}.tgf"
        frame_path.write_bytes(final.compressed_payload)
        if frame_path.read_bytes() != final.compressed_payload:
            raise SmokeError("persisted complete frame changed")
        # Recheck the persisted frame without using the inference-side graph state.
        final_task._independent_cold_validate(frame_path.read_bytes())
        if parameter_sha256(frozen) != frozen_hash:
            raise SmokeError("frozen inference changed policy parameters")
        rows.append({
            "weight": list(weight), "recipe_action_ids": list(recipe),
            "operator_graph": final.operator_graph, "original_numeric_bytes": raw_bytes,
            "complete_compressed_bytes": len(final.compressed_payload),
            "rho": final.compression_ratio, "E_ns_per_byte": final.metrics.compression_ns_per_byte,
            "D_ns_per_byte": final.metrics.decompression_ns_per_byte, "J": objective_value(final, weight),
            "selection_walltime_ns": selection_ns,
            "selection_includes": "task_features_model_proposal_transitions_candidate_evaluation_cold_check",
            "proposal_timing_ns": {
                name: getattr(proposal, name) for name in (
                    "feature_time_ns", "model_inference_time_ns", "environment_transition_time_ns",
                    "terminal_evaluation_time_ns", "prediction_time_ns",
                )
            },
            "complete_candidate_evaluations": 1,
            "frame_path": str(frame_path), "frame_sha256": _hash(final.compressed_payload),
            "bit_exact": True, "independent_persisted_frame_cold_decode": True,
            "final_replay_excluded_from_selection_time": True,
        })
        calibration.write_atomic(output / "frozen_results.json", {"results": rows})
    result = {
        "schema": SCHEMA, "status": "complete", "formal_training_eligible": False,
        "claim_scope": "interface_resource_smoke_on_one_training_block_not_test_quality",
        "frozen_inference_data_scope": "same_seen_training_block_no_generalization_claim",
        "starting_solution": "raw_only_diagnostic_not_formal_solution_library",
        "search_scope": "two_operator_horizon_one_greedy_proposal_per_weight",
        "formal_search_or_budget_claim": False, "graph_config": asdict(config),
        "grpo_config": asdict(training_config), "iterations": 1, "seed": seed,
        "source": source, "references": references, "action_registry_fingerprint": graph.REGISTRY_FINGERPRINT,
        "training_walltime_ns": training_ns, "parameter_sha256_before": before,
        "parameter_sha256_after": after, "parameter_change_l2": trace["parameter_change_l2"],
        "optimizer_steps": trace["optimizer_steps"], "training_groups": trace["groups_detail"],
        "training_native_mechanism_terminal_candidates": mechanisms_executed,
        "model_parameters": sum(value.size for value in frozen.params),
        "model_parameter_bytes": sum(value.nbytes for value in frozen.params),
        "checkpoint_path": str(checkpoint), "checkpoint_payload_sha256": checkpoint_payload_hash,
        "checkpoint_file_sha256": _hash(checkpoint.read_bytes()), "frozen_parameters_unchanged": True,
        "results": rows, "total_walltime_ns": time.perf_counter_ns() - total_started,
    }
    calibration.write_atomic(output / "REPORT.json", result)
    _event("smoke_complete", report=str(output / "REPORT.json"), weights=len(rows))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--zlib6-cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-id")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-block-bytes", type=int, default=16384)
    args = parser.parse_args(argv)
    try:
        run_smoke(args.prepared_manifest, args.admission, args.zlib6_cli, args.output_dir,
                  seed=args.seed, block_id=args.block_id, max_block_bytes=args.max_block_bytes)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"REALDATA_SMOKE_FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
