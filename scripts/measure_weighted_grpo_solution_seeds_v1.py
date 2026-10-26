#!/usr/bin/env python3
"""Measure training-only Beam30 seed recipes for the weighted-GRPO library.

This is the non-learning bootstrap between prepared training blocks and
``build_weighted_grpo_solution_library_v1.py``.  It never reads validation or
test blocks.  For one deterministic representative per current-data feature
bucket, Beam30 evaluates at most 64 complete candidates for each of the six
formal weights.  The best non-empty recipe observed for each weight is replayed
from the original roots, independently cold-decoded, and saved as a complete
typed-graph frame with physical rho/E/D measurements.

STOP-only representations are intentionally excluded because the production
first-block factory requires an actually measured operator composition.  No
candidate measurement, frame, or recipe from validation/test data is admitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import calibrate_weighted_grpo_references_v1 as calibration
import prepare_weighted_grpo_timeseries_v1 as preparation
import typed_graph_runtime as graph_runtime
import weight_conditioned_graph_task_v1 as graph_task
from weight_conditioned_grpo_v1 import FIXED_WEIGHTS, objective_value
from weight_conditioned_search_v1 import (
    CandidateSummary,
    SearchConfig,
    beam30_search,
)
import weighted_grpo_experiment_v1 as experiment
import weighted_grpo_graph_task_factory_v1 as factory


SCHEMA = "weighted-grpo-training-recipe-measurements-v1"
MAX_BLOCKS_PER_DTYPE = 4
SEARCH_EVALUATIONS = 64
SEARCH_DEPTH = 10
SEARCH_MAX_GRAPH_ACTIONS = graph_task.MAX_GRAPH_NODES
SEARCH_TIMING_WARMUPS = 1
SEARCH_TIMING_REPETITIONS = 5


class SeedMeasurementError(RuntimeError):
    """Prepared data, search, replay, or output violated the seed contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _measurement_digest(row: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json({
        key: value for key, value in row.items() if key != "measurement_sha256"
    }))


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SeedMeasurementError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise SeedMeasurementError(f"{label} leaves the prepared root") from error
    if not candidate.is_file():
        raise SeedMeasurementError(f"missing prepared payload: {candidate}")
    return candidate


def _best_nonempty_recipe(candidates: Sequence[CandidateSummary]) -> tuple[int, ...]:
    eligible = [candidate for candidate in candidates if candidate.actions]
    if not eligible:
        raise SeedMeasurementError("Beam30 evaluated no non-empty operator recipe")
    best = min(
        eligible,
        key=lambda row: (
            row.objective, row.complete_bytes, row.compression_ns_per_byte,
            row.decompression_ns_per_byte, row.actions,
        ),
    )
    return tuple(int(action) for action in best.actions)


def _roots(values: bytes, bitmap: bytes, width: int) -> tuple[graph_runtime.Stream, ...]:
    if not values or len(values) % width:
        raise SeedMeasurementError("prepared values are empty or dtype-misaligned")
    return (
        graph_runtime.Stream("numeric", width, values),
        graph_runtime.Stream("serial", 1, bitmap),
    )


def _search_task(
    *, task_id: str, values: bytes, bitmap: bytes, dtype: str,
    original_numeric_bytes: int, e0: float, d0: float,
    weight: tuple[float, float, float],
) -> graph_task.TypedGraphCompressionTask:
    width = int(preparation.SUPPORTED_DTYPES[dtype][0])
    return graph_task.TypedGraphCompressionTask(
        task_id,
        _roots(values, bitmap, width),
        root_views=(dtype, "bytes"),
        compression_reference_ns_per_byte=e0,
        decompression_reference_ns_per_byte=d0,
        objective_weight=weight,
        config=graph_task.GraphTaskConfig(
            max_ordinary_depth=graph_task.MAX_ORDINARY_DEPTH,
            max_graph_nodes=graph_task.MAX_GRAPH_NODES,
            max_active_streams=graph_task.MAX_ACTIVE_STREAMS,
            # Depth is enforced per ordinary data-flow path by
            # max_ordinary_depth.  A branched graph may contain up to 64 policy
            # nodes without any one path exceeding depth ten.
            max_policy_edits=SEARCH_MAX_GRAPH_ACTIONS,
            timing_warmups=SEARCH_TIMING_WARMUPS,
            timing_repetitions=SEARCH_TIMING_REPETITIONS,
            independent_cold_validation=False,
            allow_large_memory=False,
            require_formal_training_ready=True,
        ),
        candidate_budget=graph_task.CandidateBudget(
            SEARCH_EVALUATIONS, hard_limit=True
        ),
        original_numeric_bytes=original_numeric_bytes,
    )


def _replay_task(
    *, task_id: str, values: bytes, bitmap: bytes, dtype: str,
    original_numeric_bytes: int, e0: float, d0: float,
    weight: tuple[float, float, float], recipe: tuple[int, ...],
    timing_repetitions: int,
) -> graph_task.TypedGraphCompressionTask:
    width = int(preparation.SUPPORTED_DTYPES[dtype][0])
    return graph_task.TypedGraphCompressionTask(
        task_id,
        _roots(values, bitmap, width),
        root_views=(dtype, "bytes"),
        compression_reference_ns_per_byte=e0,
        decompression_reference_ns_per_byte=d0,
        objective_weight=weight,
        config=graph_task.GraphTaskConfig(
            max_ordinary_depth=graph_task.MAX_ORDINARY_DEPTH,
            max_graph_nodes=graph_task.MAX_GRAPH_NODES,
            max_active_streams=graph_task.MAX_ACTIVE_STREAMS,
            max_policy_edits=1,
            timing_warmups=1,
            timing_repetitions=timing_repetitions,
            independent_cold_validation=True,
            allow_large_memory=False,
            require_formal_training_ready=True,
        ),
        previous_recipe=recipe,
        candidate_budget=graph_task.CandidateBudget(1, hard_limit=True),
        original_numeric_bytes=original_numeric_bytes,
    )


def _candidate_blocks(
    prepared_manifests: Sequence[Path], *, max_blocks_per_dtype: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (isinstance(max_blocks_per_dtype, bool)
            or not isinstance(max_blocks_per_dtype, int)
            or max_blocks_per_dtype < 1):
        raise ValueError("max_blocks_per_dtype must be a positive integer")
    representatives: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []

    def order_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["dtype"], row["feature_bucket_id"],
            str(row["dataset"].manifest["dataset_id"]),
            str(row["block"].get("entity_id", "")),
            str(row["block"].get("column", "")),
            int(row["block"].get("block_index", -1)),
        )

    for supplied in prepared_manifests:
        dataset = experiment._load_prepared_dataset(
            supplied, verify_source=True, expected_role="train_validation",
        )
        manifest = dataset.manifest
        cohort = manifest.get("cohort")
        if not isinstance(cohort, Mapping) or cohort.get("role") != "train_validation":
            raise SeedMeasurementError("seed source is not a train_validation dataset")
        dataset_id = str(manifest.get("dataset_id", ""))
        if not dataset_id:
            raise SeedMeasurementError("prepared dataset_id is missing")
        provenance.append({
            "dataset_id": dataset_id,
            "manifest_path": str(dataset.manifest_path.resolve()),
            "provenance_sha256": str(manifest.get("provenance_sha256", "")),
            "verified_train_blocks": sum(
                block.get("split") == "train" for block in manifest.get("blocks", ())
            ),
        })
        for block in manifest.get("blocks", ()):
            if block.get("split") != "train":
                continue
            dtype = str(block.get("dtype", ""))
            if dtype not in preparation.SUPPORTED_DTYPES:
                raise SeedMeasurementError(f"unsupported prepared dtype {dtype!r}")
            values_path = _safe_child(dataset.root, block.get("values_path"), "values_path")
            bitmap_path = _safe_child(
                dataset.root, block.get("missing_bitmap_path"), "missing_bitmap_path"
            )
            values = values_path.read_bytes()
            bitmap = bitmap_path.read_bytes()
            if (_sha256(values) != block.get("values_sha256")
                    or _sha256(bitmap) != block.get("missing_bitmap_sha256")):
                raise SeedMeasurementError("prepared block payload fingerprint mismatch")
            if int(block.get("original_numeric_bytes", -1)) != len(values):
                raise SeedMeasurementError(
                    "prepared B_raw must equal the valid numeric value payload bytes"
                )
            if int(block.get("prepared_block_bytes", -1)) != len(values) + len(bitmap):
                raise SeedMeasurementError(
                    "prepared_block_bytes must include values and the missing bitmap"
                )
            bucket_id, feature_vector = factory.solution_library_feature_bucket(
                dtype, values, bitmap
            )
            row = {
                "dataset": dataset,
                "block": block,
                "dtype": dtype,
                "values": values,
                "bitmap": bitmap,
                "feature_bucket_id": bucket_id,
                "feature_vector": list(feature_vector),
            }
            buckets = representatives.setdefault(dtype, {})
            previous = buckets.get(bucket_id)
            if previous is not None:
                # Strict comparison preserves the first occurrence of an exact
                # tie, matching the former stable sort over all TRAIN blocks.
                if order_key(row) < order_key(previous):
                    buckets[bucket_id] = row
            elif len(buckets) < max_blocks_per_dtype:
                buckets[bucket_id] = row
            else:
                largest_bucket = max(buckets)
                if bucket_id < largest_bucket:
                    del buckets[largest_bucket]
                    buckets[bucket_id] = row
            # Every payload is verified above, including discarded buckets.
            # Only the selected representatives survive to the next block.
            del row, previous, values, bitmap
    selected = sorted(
        (row for buckets in representatives.values() for row in buckets.values()),
        key=order_key,
    )
    if not selected:
        raise SeedMeasurementError("no TRAIN blocks were selected")
    return selected, sorted(provenance, key=lambda row: row["dataset_id"])


def measure_seed_recipes(
    prepared_manifests: Sequence[Path],
    reference_table_path: Path,
    output_directory: Path,
    *,
    max_blocks_per_dtype: int = MAX_BLOCKS_PER_DTYPE,
    timing_repetitions: int = 5,
) -> dict[str, Any]:
    """Return a builder-ready training-only measured-recipe receipt."""
    if not prepared_manifests:
        raise ValueError("at least one prepared manifest is required")
    if (isinstance(timing_repetitions, bool)
            or not isinstance(timing_repetitions, int)
            or timing_repetitions < 1):
        raise ValueError("timing_repetitions must be a positive integer")
    graph_task.assert_formal_training_ready()
    reference = experiment._load_reference_table(reference_table_path)
    if reference.document.get("formal_training_eligible") is not True:
        raise SeedMeasurementError("seed measurement requires the formal zlib6 table")
    selected, provenance = _candidate_blocks(
        prepared_manifests, max_blocks_per_dtype=max_blocks_per_dtype
    )
    output_directory = output_directory.expanduser().resolve()
    frame_root = output_directory / "frames"
    measurements: list[dict[str, Any]] = []
    search_receipts: list[dict[str, Any]] = []
    for item in selected:
        dataset = item["dataset"]
        block = item["block"]
        values = item["values"]
        bitmap = item["bitmap"]
        dtype = item["dtype"]
        block_id = str(block["block_id"])
        dataset_id = str(dataset.manifest["dataset_id"])
        bucket, e0, d0 = experiment._reference_for_block(reference, block)
        discovered: dict[tuple[int, ...], tuple[float, float, float]] = {}
        for weight in FIXED_WEIGHTS:
            task = _search_task(
                task_id=f"seed-search:{dataset_id}:{block_id}",
                values=values, bitmap=bitmap, dtype=dtype,
                original_numeric_bytes=int(block["original_numeric_bytes"]),
                e0=e0, d0=d0, weight=weight,
            )
            result = beam30_search(
                task, weight,
                config=SearchConfig(
                    max_evaluations=SEARCH_EVALUATIONS,
                    beam_width=30,
                    max_steps=SEARCH_MAX_GRAPH_ACTIONS + 1,
                ),
                bound_provider=lambda current_task, state: (
                    current_task.certified_bound_state(state)
                ),
            )
            recipe = _best_nonempty_recipe(result.evaluated_candidates)
            discovered.setdefault(recipe, weight)
            search_receipts.append({
                "dataset_id": dataset_id,
                "block_id": block_id,
                "dtype": dtype,
                "feature_bucket_id": item["feature_bucket_id"],
                "weight": list(weight),
                "selected_nonempty_recipe_action_ids": list(recipe),
                "evaluations": result.evaluations,
                "certified_pruned_states": result.certified_pruned_states,
                "budget_unexplored_actions": result.budget_unexplored_actions,
                "prediction_timing_ns": asdict(result.timing),
            })
        for recipe, discovery_weight in sorted(discovered.items()):
            replay = _replay_task(
                task_id=f"seed-replay:{dataset_id}:{block_id}",
                values=values, bitmap=bitmap, dtype=dtype,
                original_numeric_bytes=int(block["original_numeric_bytes"]),
                e0=e0, d0=d0, weight=discovery_weight, recipe=recipe,
                timing_repetitions=timing_repetitions,
            )
            outcome = replay.evaluate_terminal(replay.initial_state())
            replayed_recipe = experiment._extract_recipe(outcome)
            if replayed_recipe != recipe:
                raise SeedMeasurementError("seed replay changed the discovered recipe")
            experiment._validate_outcome(
                outcome, int(block["original_numeric_bytes"]), e0, d0
            )
            identity_digest = _sha256(_canonical_json({
                "dataset_provenance_sha256": dataset.manifest["provenance_sha256"],
                "block_id": block_id,
                "dtype": dtype,
                "recipe_action_ids": list(recipe),
            }))
            frame_path = frame_root / (identity_digest + ".tgf")
            _atomic_bytes(frame_path, outcome.compressed_payload)
            if frame_path.read_bytes() != outcome.compressed_payload:
                raise SeedMeasurementError("seed frame failed post-write verification")
            graph_record = experiment._json_safe(outcome.operator_graph)
            row: dict[str, Any] = {
                "status": "passed",
                "source_split": "train",
                "dataset_id": dataset_id,
                "dataset_provenance_sha256": dataset.manifest["provenance_sha256"],
                "block_id": block_id,
                "dtype": dtype,
                "values_sha256": block["values_sha256"],
                "missing_bitmap_sha256": block["missing_bitmap_sha256"],
                "feature_bucket_id": item["feature_bucket_id"],
                "feature_vector": item["feature_vector"],
                "discovery_weight": list(discovery_weight),
                "recipe_action_ids": list(recipe),
                "operator_graph": graph_record,
                "complete_frame_path": str(frame_path),
                "complete_frame_sha256": _sha256(outcome.compressed_payload),
                "complete_frame_bytes": len(outcome.compressed_payload),
                "original_numeric_bytes": outcome.metrics.original_numeric_bytes,
                "compression_time_ns": outcome.metrics.compression_time_ns,
                "decompression_time_ns": outcome.metrics.decompression_time_ns,
                "rho_complete_bytes_per_original_numeric_byte": outcome.compression_ratio,
                "E_compression_ns_per_original_numeric_byte": (
                    outcome.metrics.compression_ns_per_byte
                ),
                "D_decompression_ns_per_original_numeric_byte": (
                    outcome.metrics.decompression_ns_per_byte
                ),
                "E0_compression_ns_per_original_numeric_byte": e0,
                "D0_decompression_ns_per_original_numeric_byte": d0,
                "J_at_discovery_weight": objective_value(outcome, discovery_weight),
                "block_size_bucket_bytes": bucket,
            }
            if any(
                not math.isfinite(float(row[field]))
                for field in (
                    "rho_complete_bytes_per_original_numeric_byte",
                    "E_compression_ns_per_original_numeric_byte",
                    "D_decompression_ns_per_original_numeric_byte",
                    "J_at_discovery_weight",
                )
            ):
                raise SeedMeasurementError("seed measurement contains non-finite metrics")
            row["measurement_sha256"] = _measurement_digest(row)
            measurements.append(row)
    if not measurements:
        raise SeedMeasurementError("no non-empty seed recipe completed measurement")
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training_only",
        "action_registry_fingerprint": graph_task.REGISTRY_FINGERPRINT,
        "typed_graph_registry": graph_runtime.CURRENT_GRAPH_REGISTRY,
        "operator_source": {
            "families": 30,
            "trace_entries": 102,
            "retained_mechanisms": 67,
            "special_operations": ["STOP", "Concat", "continuous Split"],
        },
        "search_contract": {
            "method": "Beam30 bootstrap only; no learned policy",
            "weights": [list(weight) for weight in FIXED_WEIGHTS],
            "complete_candidate_evaluations_per_weight": SEARCH_EVALUATIONS,
            "candidate_timing_warmups": SEARCH_TIMING_WARMUPS,
            "candidate_timing_repetitions": SEARCH_TIMING_REPETITIONS,
            "candidate_timing_aggregation": "median",
            "maximum_ordinary_depth": SEARCH_DEPTH,
            "maximum_total_graph_actions": SEARCH_MAX_GRAPH_ACTIONS,
            "certified_pruning": True,
            "stop_only_recipes_excluded": True,
        },
        "reference_table": {
            "path": str(reference.path),
            "calibration_sha256": reference.calibration_sha256,
        },
        "prepared_datasets": provenance,
        "selected_training_blocks": [
            {
                "dataset_id": item["dataset"].manifest["dataset_id"],
                "block_id": item["block"]["block_id"],
                "dtype": item["dtype"],
                "feature_bucket_id": item["feature_bucket_id"],
                "feature_vector": item["feature_vector"],
            }
            for item in selected
        ],
        "search_receipts": search_receipts,
        "measurements": sorted(
            measurements,
            key=lambda row: (
                row["dtype"], row["dataset_id"], row["block_id"],
                row["recipe_action_ids"],
            ),
        ),
    }
    document["receipt_sha256"] = _sha256(_canonical_json(document))
    return document


def write_receipt(path: Path, document: Mapping[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(
        document, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-blocks-per-dtype", type=int,
                        default=MAX_BLOCKS_PER_DTYPE)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    args = parser.parse_args(argv)
    document = measure_seed_recipes(
        args.prepared, args.reference_table, args.output_directory,
        max_blocks_per_dtype=args.max_blocks_per_dtype,
        timing_repetitions=args.timing_repetitions,
    )
    write_receipt(args.output, document)
    print(json.dumps({
        "schema": document["schema"],
        "receipt_sha256": document["receipt_sha256"],
        "selected_training_blocks": len(document["selected_training_blocks"]),
        "measured_recipes": len(document["measurements"]),
        "output": str(args.output.expanduser().resolve()),
    }, sort_keys=True))
    return 0


__all__ = [
    "MAX_BLOCKS_PER_DTYPE", "SCHEMA", "SEARCH_DEPTH", "SEARCH_EVALUATIONS",
    "SEARCH_MAX_GRAPH_ACTIONS", "SEARCH_TIMING_REPETITIONS",
    "SEARCH_TIMING_WARMUPS",
    "SeedMeasurementError", "measure_seed_recipes", "write_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
