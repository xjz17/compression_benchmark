#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import aggregate_formal_weighted_grpo_results_v1 as aggregate
import formal_weighted_grpo_training_v1 as training
import prepared_block_baseline86_v1 as baseline86
import weighted_grpo_experiment_v1 as experiment


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FormalAggregationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        delivery = root / "delivery.json"
        delivery.write_text("delivery", encoding="utf-8")
        delivery_sha = hashlib.sha256(delivery.read_bytes()).hexdigest()
        library = root / "solution_library.json"
        library.write_bytes(b"solution-library")
        library_identity = {
            "path": str(library.resolve()),
            "file_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "action_registry_fingerprint": "f" * 64,
        }
        test = root / "test"
        test.mkdir(parents=True)
        manifest = {
            "dataset_id": "test-dataset", "provenance_sha256": "d" * 64,
            "blocks": [{
                "block_id": "test__e__value__b-000000", "entity_id": "e",
                "split": "test", "column": "value", "dtype": "int64",
                "block_index": 0, "original_numeric_bytes": 8,
                "prepared_block_bytes": 9,
            }],
        }
        write_json(test / "dataset_manifest.json", manifest)
        admission = {
            "corpus_sha256": "a" * 64, "calibration_sha256": "b" * 64,
            "gate_bundle_sha256": "c" * 64, "source_fingerprint": "e" * 64,
            "action_registry_fingerprint": "f" * 64,
        }
        test_identity = [{
            "dataset_id": "test-dataset", "dataset_provenance_sha256": "d" * 64,
            "block_id": manifest["blocks"][0]["block_id"], "entity_id": "e",
            "split": "test", "column": "value", "dtype": "int64",
            "block_index": 0, "original_numeric_bytes": 8,
            "prepared_block_bytes": 9,
        }]
        admission["test_cohort_sha256"] = hashlib.sha256(canonical(test_identity)).hexdigest()
        plan = {
            "schema": aggregate.PLAN_SCHEMA, "admission": admission,
            "training_prepared_manifests": [f"/train/{index}" for index in range(16)],
            "test_prepared_manifests": [str(test)],
            "delivery_checkpoint": str(delivery),
            "solution_library_manifest": str(library),
            "weights": [list(weight) for weight in experiment.FIXED_WEIGHTS],
        }
        plan["plan_sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
        plan_path = root / "plan.json"
        write_json(plan_path, plan)
        references = {
            "schema": "weighted-grpo-zlib6-reference-table-v1",
            "objective_normalization": "E/(E+E0), D/(D+D0)",
            "references": [{
                "dtype": "int64", "block_size_bucket_bytes": 16,
                "E0_compression_ns_per_original_byte": 2.0,
                "D0_decompression_ns_per_original_byte": 3.0,
            }],
        }
        references["calibration_sha256"] = hashlib.sha256(canonical(references)).hexdigest()
        admission["calibration_sha256"] = references["calibration_sha256"]
        plan["admission"] = admission
        plan.pop("plan_sha256")
        plan["plan_sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
        write_json(plan_path, plan)
        reference_path = root / "reference.json"
        write_json(reference_path, references)
        training_receipt = {
            "schema": training.CAMPAIGN_SCHEMA, "delivery_file_sha256": delivery_sha,
            "seeds": [0, 1, 2], "per_seed_budget_seconds": 57600,
            "measurement_workers_per_seed": 4,
            "selection_used_test_data": False, "selected_seed": 0,
            "selected_validation_score": 1.0,
            "solution_library_identity": library_identity,
            "action_registry_fingerprint": admission["action_registry_fingerprint"],
            "seed_receipts": [
                {"seed": seed, "training_wall_time_ns": 10, "status": "complete",
                 "solution_library_identity": library_identity}
                for seed in (0, 1, 2)
            ],
        }
        training_receipt["campaign_sha256"] = hashlib.sha256(
            canonical(training_receipt)
        ).hexdigest()
        training_path = root / "training.json"
        write_json(training_path, training_receipt)

        campaign_binding = {
            "campaign_plan_path": str(plan_path),
            "campaign_plan_sha256": plan["plan_sha256"],
            **admission,
            "delivery_checkpoint_path": str(delivery),
            "delivery_checkpoint_file_sha256": delivery_sha,
            "delivery_checkpoint_role": "consumed_frozen_policy",
            "solution_library_path": library_identity["path"],
            "solution_library_file_sha256": library_identity["file_sha256"],
        }

        def weighted(source, method, ablation):
            artifact_root = root / (source + "_frames")
            records = []
            for weight in experiment.FIXED_WEIGHTS:
                weight_id = experiment._weight_id(weight)
                payload = (source + weight_id).encode()
                payload_sha = hashlib.sha256(payload).hexdigest()
                frame = artifact_root / weight_id / (payload_sha + ".bin")
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(payload)
                complete = len(payload)
                rho, e_value, d_value = complete / 8, 10.0, 5.0
                ehat, dhat = e_value / 12.0, d_value / 8.0
                j_value = weight[0] * rho + weight[1] * ehat + weight[2] * dhat
                final = {
                    "rho_complete_bytes_per_original_numeric_byte": rho,
                    "E_compression_ns_per_original_numeric_byte": e_value,
                    "D_decompression_ns_per_original_numeric_byte": d_value,
                    "Ehat": ehat, "Dhat": dhat, "J": j_value,
                    "original_numeric_bytes": 8, "complete_bytes": complete,
                    "complete_payload_sha256": payload_sha,
                    "compression_time_ns": 80, "decompression_time_ns": 40,
                    "E0_compression_ns_per_original_numeric_byte": 2.0,
                    "D0_decompression_ns_per_original_numeric_byte": 3.0,
                    "payload_matches_online_selection": True,
                    "correctness_evidence": {
                        "self_contained_complete_frame": True,
                        "independent_cold_decode": True, "bit_exact_roundtrip": True,
                        "action_registry_fingerprint": admission["action_registry_fingerprint"],
                    },
                    "complete_payload_artifact": {
                        "relative_path": frame.relative_to(artifact_root).as_posix(),
                        "absolute_path": str(frame.resolve()), "bytes": complete,
                        "sha256": payload_sha,
                    },
                }
                timing = {field: 1 for field in experiment.TIMING_FIELDS}
                timing["online_task_factory_time_ns"] = 2
                timing["search_only_total_prediction_time_ns"] = 7
                timing["total_prediction_time_ns"] = 9
                records.append({
                    "dataset_id": "test-dataset", "dataset_provenance_sha256": "d" * 64,
                    "block_id": manifest["blocks"][0]["block_id"], "entity_id": "e",
                    "split": "test", "column": "value", "dtype": "int64",
                    "block_index": 0, "series_block_ordinal": 0,
                    "method": method, "weight_id": weight_id, "weight": list(weight),
                    "status": "success", "selected_recipe": [1],
                    "reuse": ({
                        "enabled": False, "previous_recipe_source_block_id": None,
                        "previous_recipe_action_ids": [], "max_local_edits": None,
                    } if ablation["no_reuse"] else {
                        "enabled": True, "previous_recipe_source_block_id": None,
                        "previous_recipe_action_ids": [], "max_local_edits": None,
                    }),
                    "online_task_factory_request": {
                        "schema": experiment.TASK_FACTORY_SCHEMA,
                        "stage": "online_selection",
                        "dataset_id": "test-dataset",
                        "provenance_sha256": "d" * 64,
                        "block_id": manifest["blocks"][0]["block_id"],
                        "entity_id": "e", "split": "test", "column": "value",
                        "dtype": "int64", "block_index": 0,
                        "original_numeric_bytes": 8, "prepared_block_bytes": 9,
                        "method": method, "weight": list(weight),
                        "evaluation_budget": 64,
                        "series_block_ordinal": 0,
                        "max_local_edits": None,
                        "reuse_previous_recipe": False,
                        "previous_recipe": [], "selected_recipe": [],
                        "ablation": ablation,
                    },
                    "online_selection": {
                        "evaluations": 1,
                        "evaluated_candidates": [{}],
                        "evaluation_budget": 64,
                        "objective_guided_ordering": not ablation["no_heuristic_order"],
                        "prediction_timing_ns": timing,
                    },
                    "final_task_factory_request": {
                        "schema": experiment.TASK_FACTORY_SCHEMA,
                        "stage": "final_independent_encoding",
                        "dataset_id": "test-dataset",
                        "provenance_sha256": "d" * 64,
                        "block_id": manifest["blocks"][0]["block_id"],
                        "entity_id": "e", "split": "test", "column": "value",
                        "dtype": "int64", "block_index": 0,
                        "original_numeric_bytes": 8, "prepared_block_bytes": 9,
                        "method": method, "weight": list(weight),
                        "series_block_ordinal": 0, "evaluation_budget": 1,
                        "max_local_edits": 0, "reuse_previous_recipe": False,
                        "previous_recipe": [1], "selected_recipe": [1],
                        "ablation": ablation,
                    },
                    "solution_library_evidence": {
                        "online_selection": library_identity,
                        "final_independent_encoding": library_identity,
                    },
                    "final_independent_encoding": final, "failure": None,
                })
            document = {
                "schema": experiment.SCHEMA, "status": "complete",
                "config": {
                    "methods": [method], "splits": ["test"], "verify_source": True,
                    "require_formal_evidence": True, "ablation": ablation,
                },
                "campaign_binding": campaign_binding,
                "solution_library_identity": library_identity,
                "training_provenance": ({
                    "status": "not_run_checkpoint_consumed",
                    "model_parameters_frozen": True,
                    "model_parameters_unchanged": True,
                    "optimizer_steps": 0,
                    "checkpoint_file_sha256": delivery_sha,
                    "checkpoint_extra": {
                        "solution_library_identity": library_identity,
                        "run_identity": {
                            "solution_library_identity": library_identity,
                            "action_registry_fingerprint": admission[
                                "action_registry_fingerprint"
                            ],
                        },
                    },
                } if method == "grpo" else {
                    "status": "not_applicable_beam_only",
                    "model_parameters_frozen": True, "optimizer_steps": 0,
                }),
                "action_registry_fingerprints": [admission["action_registry_fingerprint"]],
                "prepared_datasets": [{
                    "dataset_id": "test-dataset", "provenance_sha256": "d" * 64,
                }],
                "payload_artifacts": {
                    "enabled": True, "root": str(artifact_root.resolve()),
                    "successful_artifacts": 6,
                },
                "records": records, "attempted_records": 6,
                "successful_records": 6, "failed_records": 0,
            }
            document["result_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
            path = root / (source + ".json")
            write_json(path, document)
            return path

        normal = {"no_certified_prune": False, "no_heuristic_order": False,
                  "no_reuse": False}
        paths = {
            "frozen_grpo": weighted("frozen_grpo", "grpo", normal),
            "beam30": weighted("beam30", "beam30", normal),
            "no_target_pruning_ranking": weighted(
                "no_target_pruning_ranking", "grpo",
                {"no_certified_prune": True, "no_heuristic_order": True,
                 "no_reuse": False},
            ),
            "no_block_reuse": weighted(
                "no_block_reuse", "grpo",
                {"no_certified_prune": False, "no_heuristic_order": False,
                 "no_reuse": True},
            ),
        }

        catalog = baseline86.build_catalog_snapshot()
        baseline_root = root / "baseline86"
        frames = baseline_root / "complete_frames"
        payload = b"baseline"
        payload_sha = hashlib.sha256(payload).hexdigest()
        frame = frames / "test-dataset" / "gzip" / "frame.bin"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(payload)
        label_rows = []
        for row in catalog.labels:
            if row.label == "gzip":
                label_rows.append({
                    "label": row.label, "status": "measured",
                    "complete_compressed_bytes": len(payload),
                    "complete_frame_sha256": payload_sha,
                    "complete_frame_artifact": {
                        "relative_path": frame.relative_to(frames).as_posix(),
                        "absolute_path": str(frame.resolve()), "bytes": len(payload),
                        "sha256": payload_sha,
                    },
                    "original_numeric_bytes": 8,
                    "rho_complete_bytes_per_original_numeric_byte": len(payload) / 8,
                    "compression_elapsed_ns": 80, "decompression_elapsed_ns": 40,
                    "E_compression_ns_per_original_numeric_byte": 10.0,
                    "D_decompression_ns_per_original_numeric_byte": 5.0,
                    "self_contained_frame": True, "independent_cold_decode": True,
                    "bit_exact_values_and_missing_bitmap": True,
                })
            else:
                label_rows.append({"label": row.label, "status": "unsupported",
                                   "reason": "test fixture unsupported"})
        block_receipt = {
            "schema": baseline86.SCHEMA, "dataset_id": "test-dataset",
            "dataset_provenance_sha256": "d" * 64,
            "block_id": manifest["blocks"][0]["block_id"], "entity_id": "e",
            "split": "test", "column": "value", "dtype": "int64", "block_index": 0,
            "original_numeric_bytes": 8, "label_results": label_rows,
        }
        block_path = baseline_root / "blocks" / "test-dataset" / "block.json"
        write_json(block_path, block_receipt)
        baseline_summary = {
            "schema": baseline86.SCHEMA, "catalog_snapshot_sha256": catalog.snapshot_sha256,
            "registry_version": catalog.registry_version,
            "registry_sha256": catalog.registry_sha256,
            "blocks": [{
                "dataset_id": "test-dataset", "block_id": manifest["blocks"][0]["block_id"],
                "receipt_path": block_path.relative_to(baseline_root).as_posix(),
            }],
            "complete_frame_artifacts": {"enabled": True, "root": str(frames.resolve())},
        }
        write_json(baseline_root / "summary.json", baseline_summary)
        return {
            "plan": plan_path, "reference": reference_path, "training": training_path,
            "delivery_sha": delivery_sha, "baseline": baseline_root, **paths,
        }

    def test_aggregate_enforces_coverage_artifacts_and_produces_two_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            detail, summary, sources, count = aggregate.aggregate(
                campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                training_receipt=fixture["training"],
                frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                baseline86_path=fixture["baseline"],
                no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
            )
            audit_path = Path(directory) / "audit.json"
            aggregate.write_payload_audit(audit_path, sources, count)
            detail, summary, _, _ = aggregate.aggregate(
                campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                training_receipt=fixture["training"],
                frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                baseline86_path=fixture["baseline"],
                no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                no_block_reuse=fixture["no_block_reuse"], payload_audit=audit_path,
            )
        self.assertEqual(summary["status"], "passed_fail_closed_aggregation")
        self.assertEqual(len(detail["dataset_entity_column_summaries"]), 5 * 6)
        self.assertEqual(len(detail["dataset_column_summaries"]), 5 * 6)
        self.assertEqual(len(summary["win_tie_loss_by_block_weight_J"]), 4)
        self.assertEqual(summary["coverage"]["test_blocks"], 1)

    def test_missing_weight_and_changed_frame_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            document = json.loads(fixture["beam30"].read_text())
            document["records"].pop()
            document["successful_records"] = 5
            document["attempted_records"] = 5
            document.pop("result_sha256")
            document["result_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
            write_json(fixture["beam30"], document)
            with self.assertRaisesRegex(aggregate.AggregationError, "six weights"):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

            fixture = self._fixture(root / "second")
            document = json.loads(fixture["frozen_grpo"].read_text())
            artifact = document["records"][0]["final_independent_encoding"]["complete_payload_artifact"]
            Path(artifact["absolute_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(aggregate.AggregationError, "hash/size mismatch"):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_hash_only_legacy_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            summary_path = fixture["baseline"] / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary.pop("complete_frame_artifacts")
            write_json(summary_path, summary)
            with self.assertRaisesRegex(
                aggregate.AggregationError, "hash-only rows are inadmissible"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_no_target_ablation_must_disable_objective_ordering_in_records(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture["no_target_pruning_ranking"]
            document = json.loads(path.read_text())
            document["records"][0]["online_selection"][
                "objective_guided_ordering"
            ] = True
            document.pop("result_sha256")
            document["result_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
            write_json(path, document)
            with self.assertRaisesRegex(
                aggregate.AggregationError, "objective-guided ordering evidence"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=path,
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_weighted_record_must_bind_the_formal_evaluation_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture["frozen_grpo"]
            document = json.loads(path.read_text())
            document["records"][0]["online_selection"].pop("evaluation_budget")
            document.pop("result_sha256")
            document["result_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
            write_json(path, document)
            with self.assertRaisesRegex(
                aggregate.AggregationError, "evaluation_budget"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=path, beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_solution_library_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            plan = json.loads(fixture["plan"].read_text())
            Path(plan["solution_library_manifest"]).write_bytes(b"drifted-library")
            with self.assertRaisesRegex(
                aggregate.AggregationError, "solution library"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_weighted_task_request_and_candidate_count_are_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            path = fixture["beam30"]
            document = json.loads(path.read_text())
            document["records"][0]["online_selection"]["evaluated_candidates"] = []
            document.pop("result_sha256")
            document["result_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
            write_json(path, document)
            with self.assertRaisesRegex(
                aggregate.AggregationError, "evaluated-candidate evidence"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=path,
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

            fixture = self._fixture(Path(directory) / "request")
            path = fixture["frozen_grpo"]
            document = json.loads(path.read_text())
            document["records"][0]["final_task_factory_request"][
                "reuse_previous_recipe"
            ] = True
            document.pop("result_sha256")
            document["result_sha256"] = hashlib.sha256(canonical(document)).hexdigest()
            write_json(path, document)
            with self.assertRaisesRegex(
                aggregate.AggregationError, "final TaskFactoryRequest"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"], frozen_grpo=path,
                    beam30=fixture["beam30"], baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_baseline86_failed_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            summary = json.loads((fixture["baseline"] / "summary.json").read_text())
            receipt_path = fixture["baseline"] / summary["blocks"][0]["receipt_path"]
            receipt = json.loads(receipt_path.read_text())
            failed = next(row for row in receipt["label_results"]
                          if row["status"] == "unsupported")
            failed["status"] = "failed"
            failed["error"] = failed.pop("reason")
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(
                aggregate.AggregationError, "failed label result"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
                )

    def test_payload_audit_binds_baseline_receipts_and_complete_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            _, _, sources, count = aggregate.aggregate(
                campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                training_receipt=fixture["training"],
                frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                baseline86_path=fixture["baseline"],
                no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                no_block_reuse=fixture["no_block_reuse"], payload_audit=None,
            )
            audit_path = root / "payload_audit.json"
            aggregate.write_payload_audit(audit_path, sources, count)

            summary = json.loads((fixture["baseline"] / "summary.json").read_text())
            receipt_path = fixture["baseline"] / summary["blocks"][0]["receipt_path"]
            receipt = json.loads(receipt_path.read_text())
            measured = next(
                row for row in receipt["label_results"] if row["status"] == "measured"
            )
            replacement = b"mutated!"
            replacement_sha = hashlib.sha256(replacement).hexdigest()
            Path(measured["complete_frame_artifact"]["absolute_path"]).write_bytes(
                replacement
            )
            measured["complete_frame_sha256"] = replacement_sha
            measured["complete_frame_artifact"]["sha256"] = replacement_sha
            write_json(receipt_path, receipt)

            with self.assertRaisesRegex(
                aggregate.AggregationError, "payload audit is bound to different result files"
            ):
                aggregate.aggregate(
                    campaign_plan=fixture["plan"], reference_table=fixture["reference"],
                    training_receipt=fixture["training"],
                    frozen_grpo=fixture["frozen_grpo"], beam30=fixture["beam30"],
                    baseline86_path=fixture["baseline"],
                    no_target_pruning_ranking=fixture["no_target_pruning_ranking"],
                    no_block_reuse=fixture["no_block_reuse"],
                    payload_audit=audit_path,
                )

if __name__ == "__main__":
    unittest.main(verbosity=2)
