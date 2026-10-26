from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jsonschema

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_dbp_audit_payload.py"
_SPEC = importlib.util.spec_from_file_location("build_dbp_audit_payload", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


class DBPAuditPayloadBuilderTest(unittest.TestCase):
    @staticmethod
    def _json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_collection(self, compact: Path) -> None:
        collection_path = compact / "meta/collection_receipt.json"
        collection_rows = []
        for path in sorted(
            item
            for item in compact.rglob("*")
            if item.is_file() and item != collection_path
        ):
            collection_rows.append(
                {
                    "relative_path": path.relative_to(compact).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": builder.sha256_file(path),
                }
            )
        self._json(
            collection_path,
            {
                "schema": builder.COLLECTION_RECEIPT_SCHEMA,
                "status": "passed",
                "dataset_count": 1,
                "file_count": len(collection_rows),
                "files": collection_rows,
            },
        )

    def _refresh_manifest_bindings(self, arguments: argparse.Namespace) -> None:
        manifest_receipt_path = (
            arguments.control_root / "manifests/dataset_one/manifest_receipt.json"
        )
        run_receipt_path = (
            arguments.results_root / "by_dataset/dataset_one/run_receipt.json"
        )
        run_receipt = builder.load_json(run_receipt_path)
        run_receipt["source_manifest_receipt_sha256"] = builder.sha256_file(
            manifest_receipt_path
        )
        self._json(run_receipt_path, run_receipt)
        self._write_collection(arguments.control_root.parent)

    def _fixture(self, root: Path) -> argparse.Namespace:
        dataset = "dataset_one"
        aggregate_id = "dataset_one|value|temperature"
        compact = root / "compact"
        control = compact / "control"
        results = compact / "results"
        scope = root / "paper_scope.json"
        self._json(
            scope,
            {
                "schema": builder.FROZEN_SCOPE_SCHEMA,
                "scope_id": builder.FROZEN_SCOPE_ID,
                "dataset_identity_count": 1,
                "dataset_ids": [dataset],
                "source_scope_sha256": "8" * 64,
                "benchmarkhost_input_inventory_sha256": "9" * 64,
            },
        )
        scope_sha = builder.sha256_file(scope)

        inventory = root / "feeds" / "source_column_inventory.csv"
        inventory.parent.mkdir(parents=True)
        with inventory.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "dataset_id",
                    "aggregate_id",
                    "column",
                    "source_column",
                    "dtype",
                    "width",
                    "statistics_sha256",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "dataset_id": dataset,
                    "aggregate_id": aggregate_id,
                    "column": "temperature",
                    "source_column": "temperature",
                    "dtype": "float32",
                    "width": 4,
                    "statistics_sha256": "c" * 64,
                }
            )

        artifact = control / "runtime_artifacts" / "campaign_final_d40_cppf9.json"
        self._json(
            artifact,
            {
                "schema": "dbp-float-e2e-runtime-artifacts-v1",
                "status": "passed",
                "controller_source": {"sha256": "1" * 64},
                "cpp_executable": {"sha256": "2" * 64},
                "python_codec_source": {"sha256": "3" * 64},
                "python_cli_source": {"sha256": "4" * 64},
                "python_dependency_sources": [],
                "java_class_count": 2,
                "java_classes_content_manifest_sha256": "5" * 64,
            },
        )
        artifact_document = builder.load_json(artifact)
        artifact_sha = builder.sha256_file(artifact)
        artifact_binding = builder.artifact_summary(artifact_document, artifact_sha)

        environment = root / "environment.json"
        self._json(
            environment,
            {
                "schema": "dbp-float-e2e-environment-v1",
                "status": "passed",
                "hardware": "fixture server",
                "cpu": {
                    "model": "fixture CPU",
                    "passed_receipts_at_capture": 1,
                },
                "os": "fixture OS",
                "clock": "monotonic nanoseconds",
                "cpu_affinity": [32],
                "concurrent_load": "isolated fixture",
                "shared_server": True,
                "captured_at": "2026-10-09T00:00:00Z",
                "repetitions": 1,
                "warmup_blocks": 20,
                "runtime_artifact_manifest_sha256": artifact_sha,
                "runtime_versions": {
                    "cpp": "fixture",
                    "java": "fixture",
                    "python": "fixture",
                },
            },
        )

        source_commit_receipt = control / "source_commit_receipt.json"
        self._json(
            source_commit_receipt,
            {
                "branch": "fixture/source",
                "commit": "0" * 40,
                "all_standalone_sources_match_commit": True,
                "standalone_source_sha256": {
                    "scripts/fixture.py": "a" * 64,
                },
            },
        )
        source_commit_sha = builder.sha256_file(source_commit_receipt)
        remote_build_receipt = control / "build_receipts/remote_build_receipt.json"
        self._json(remote_build_receipt, {"status": "passed"})
        remote_build_sha = builder.sha256_file(remote_build_receipt)
        golden_receipt = control / "final_goldens_receipt.json"
        golden_cases = [
            {
                "case": f"case_{index}",
                "cold_decoder_pairs": 9,
                "encoders": list(builder.RUNTIMES),
                "frame_bytes": 10 + index,
                "frame_sha256": format(index + 1, "064x"),
                "input_sha256": format(index + 101, "064x"),
                "status": "passed",
                "values": index,
                "width": 4 if index % 2 else 8,
            }
            for index in range(builder.EXPECTED_GOLDEN_CASES)
        ]
        self._json(
            golden_receipt,
            {
                "schema": builder.GOLDEN_RECEIPT_SCHEMA,
                "stage": "remote_cross_runtime_goldens",
                "status": "passed",
                "case_count": builder.EXPECTED_GOLDEN_CASES,
                "cold_decoder_pairs": builder.EXPECTED_COLD_DECODER_PAIRS,
                "cases": golden_cases,
                "common_runtime_artifact_manifest": artifact_binding,
                "runtime_artifact_identities_before_after": {"status": "passed"},
            },
        )
        golden_sha = builder.sha256_file(golden_receipt)
        campaign_binding = control / "final_campaign_binding.json"
        self._json(
            campaign_binding,
            {
                "schema": builder.CAMPAIGN_BINDING_SCHEMA,
                "status": "passed",
                "scope": {
                    "dataset_count": 1,
                    "expected_float_aggregate_count": 1,
                    "expected_measured_aggregate_count": 1,
                    "expected_n_a_aggregate_count": 0,
                    "expected_native_values_bytes": 8,
                    "scope_sha256": scope_sha,
                },
                "runtime_artifacts": {
                    "status": "passed",
                    "sha256": artifact_sha,
                },
                "source_commit": {
                    "status": "passed",
                    "branch": "fixture/source",
                    "commit": "0" * 40,
                    "receipt_sha256": source_commit_sha,
                },
                "remote_build_reproduction": {
                    "status": "passed",
                    "sha256": remote_build_sha,
                },
                "frame_verification": {
                    "status": "passed",
                    "case_count": builder.EXPECTED_GOLDEN_CASES,
                    "cold_decoder_pairs": builder.EXPECTED_COLD_DECODER_PAIRS,
                    "sha256": golden_sha,
                },
            },
        )
        campaign_binding_sha = builder.sha256_file(campaign_binding)
        manifest_dir = control / "manifests" / dataset
        aggregate_path = manifest_dir / "aggregates.jsonl"
        source_row = {
            "schema": builder.AGGREGATE_MANIFEST_SCHEMA,
            "dataset_id": dataset,
            "aggregate_id": aggregate_id,
            "dtype": "float32",
            "width": 4,
            "max_point": 2,
            "blocks": 1,
            "rows": 3,
            "valid_count": 2,
            "missing_count": 1,
            "original_bytes": 8,
            "validity_bytes": 1,
            "header_bytes": 10,
            "expected_sequence_chain_sha256": "6" * 64,
            "manifest_expected_sequence_chain_sha256": "b" * 64,
            "status": "ready",
        }
        self._jsonl(aggregate_path, [source_row])
        aggregate_sha = builder.sha256_file(aggregate_path)
        blocks_sha = builder.sha256_bytes(b"fixture\n")
        manifest_receipt_path = manifest_dir / "manifest_receipt.json"
        self._json(
            manifest_receipt_path,
            {
                "schema": builder.CONTROLLER_RECEIPT_SCHEMA,
                "stage": "dataset_manifest",
                "dataset_id": dataset,
                "status": "passed",
                "scope_sha256": scope_sha,
                "source_preflight": {
                    "status": "passed",
                    "verified_block_count": 1,
                    "verified_segment_count": 2,
                    "verified_segment_bytes": 9,
                },
                "float_aggregate_count": 1,
                "float_block_records": 1,
                "aggregates_jsonl": {"sha256": aggregate_sha},
                "blocks_tsv": {"sha256": blocks_sha},
                "column_statistics_sha256": "c" * 64,
                "source_block_catalog_sha256": "7" * 64,
                "source_block_catalog_records": 3,
                "dataset_manifest_sha256": "a" * 64,
                "ready_sha256": "f" * 64,
                "input_authority": {
                    "source_scope_sha256": "8" * 64,
                    "input_inventory_sha256": "9" * 64,
                },
            },
        )
        manifest_receipt_sha = builder.sha256_file(manifest_receipt_path)
        self._json(
            control / "input_scope_receipt.json",
            {
                "schema": builder.CONTROLLER_RECEIPT_SCHEMA,
                "stage": "manifest_batch",
                "status": "passed",
                "source_preflight": "passed",
                "failures": [],
                "scope_sha256": scope_sha,
                "dataset_count": 1,
                "requested_dataset_count": 1,
                "float_aggregate_count": 1,
                "float_block_records": 1,
                "measured_aggregate_count": 1,
                "n_a_aggregate_count": 0,
                "measured_dataset_count": 1,
                "float_dataset_count": 1,
                "row_count": 3,
                "valid_count": 2,
                "missing_count": 1,
                "validity_bytes": 1,
                "original_bytes": 8,
                "input_authority": {
                    "source_scope_sha256": "8" * 64,
                    "input_inventory_sha256": "9" * 64,
                },
            },
        )

        result = {
            "status": "ok",
            "aggregate_id": aggregate_id,
            "width": 4,
            "blocks": 1,
            "rows": 3,
            "missing_count": 1,
            "validity_bytes": 1,
            "header_bytes": 10,
            "valid_count": 2,
            "original_bytes": 8,
            "compressed_bytes": 17,
            "encode_ns": 100,
            "decode_ns": 80,
            "scaled_blocks": 1,
            "raw_blocks": 0,
            "verified_values": 2,
            "frame_sequence_sha256": "a" * 64,
        }
        result_dir = results / "by_dataset" / dataset
        runtime_receipts = []
        for runtime in builder.RUNTIMES:
            result_path = result_dir / f"{runtime}.jsonl"
            self._jsonl(result_path, [result])
            runtime_receipts.append(
                {
                    "runtime": runtime,
                    "status": "passed",
                    "returncode": 0,
                    "result_sha256": builder.sha256_file(result_path),
                    "aggregate_count": 1,
                    "repeats": 1,
                    "cpu_affinity": [32],
                    "warmup_policy": "first_min_20_blocks_once_then_reset_metrics",
                    "totals": builder.runtime_totals([result]),
                }
            )
        self._json(
            result_dir / "run_receipt.json",
            {
                "schema": builder.CONTROLLER_RECEIPT_SCHEMA,
                "stage": "dataset_runtime",
                "dataset_id": dataset,
                "status": "passed",
                "source_preflight_status": "passed",
                "cross_runtime_exact_status": "passed",
                "cross_runtime_exact_aggregate_count": 1,
                "cross_runtime_exact_fields": list(builder.EXACT_CROSS_RUNTIME_FIELDS),
                "source_manifest_receipt_sha256": manifest_receipt_sha,
                "common_runtime_artifact_manifest": artifact_binding,
                "runtime_artifact_identities_before_after": {"status": "passed"},
                "source_file_identities_before_after": {"status": "passed"},
                "control_input_identities_before_after": {
                    "status": "passed",
                    "sha256": {
                        "aggregates.jsonl": aggregate_sha,
                        "blocks.tsv": blocks_sha,
                    },
                },
                "runtimes": runtime_receipts,
            },
        )
        self._json(
            results / "campaign_run_receipt.json",
            {
                "schema": builder.CONTROLLER_RECEIPT_SCHEMA,
                "stage": "campaign_runtime",
                "status": "passed",
                "scope_sha256": scope_sha,
                "runtime_artifact_manifest_sha256": artifact_sha,
                "dataset_count": 1,
                "skipped_current": [],
                "completed": [dataset],
                "failures": [],
            },
        )
        self._json(
            results / "campaign_progress.json",
            {
                "schema": builder.CONTROLLER_RECEIPT_SCHEMA,
                "stage": "campaign_progress",
                "status": "passed",
                "scope_sha256": scope_sha,
                "runtime_artifact_manifest_sha256": artifact_sha,
                "dataset_count": 1,
                "runtime_passed_count": 1,
                "source_preflight_passed_count": 1,
                "remaining_n_a_aggregate_count": 0,
                "pending_count": 0,
                "running": [],
                "failed": [],
            },
        )
        self._write_collection(compact)
        return argparse.Namespace(
            results_root=results,
            control_root=control,
            scope=scope,
            environment=environment,
            artifact_manifest=artifact,
            campaign_binding_sha256=campaign_binding_sha,
            output_dir=root / "output",
            source_column_inventory=inventory,
            allow_partial=False,
        )

    def test_complete_payload_closes_all_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self._fixture(root)
            with mock.patch.object(builder, "EXPECTED_SCOPE_DATASETS", 1):
                payload, column_csv, receipt = builder.build_payload(arguments)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["summary"]["schema_aggregate_count"], 1)
            self.assertEqual(len(payload["columns"]), 3)
            self.assertEqual(receipt["status"], "passed")
            self.assertTrue(
                all(gate["status"] == "passed" for gate in receipt["gates"].values())
            )
            self.assertNotIn(str(root), json.dumps(payload))
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[3]
                    / "deploy"
                    / "huggingface-compressionbenchmark"
                    / "dbp-audit.schema.json"
                ).read_text()
            )
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(payload, schema)
            with self.assertRaises(builder.AuditBuildError):
                builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertFalse((arguments.output_dir / "dbp-audit.json").exists())
            with mock.patch.object(builder, "validate_public_payload"):
                result = builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertEqual(result["json"], "dbp-audit.json")
            synthetic_payload = json.loads(
                (arguments.output_dir / "dbp-audit.json").read_text()
            )
            synthetic_payload["status"] = "partial"
            jsonschema.validate(synthetic_payload, schema)

    def test_invalid_source_block_catalog_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self._fixture(root)
            manifest_path = (
                arguments.control_root / "manifests/dataset_one/manifest_receipt.json"
            )
            manifest = builder.load_json(manifest_path)
            manifest["source_block_catalog_sha256"] = "not-a-sha256"
            self._json(manifest_path, manifest)
            self._refresh_manifest_bindings(arguments)
            arguments.allow_partial = True
            with mock.patch.object(builder, "EXPECTED_SCOPE_DATASETS", 1):
                payload, column_csv, receipt = builder.build_payload(arguments)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(receipt["gates"]["source_manifest"]["status"], "failed")
            result = builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertEqual(result["json"], "dbp-audit.partial.json")

    def test_self_consistent_forged_authority_differs_from_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self._fixture(root)
            forged = {
                "source_scope_sha256": "d" * 64,
                "input_inventory_sha256": "e" * 64,
            }
            manifest_path = (
                arguments.control_root / "manifests/dataset_one/manifest_receipt.json"
            )
            manifest = builder.load_json(manifest_path)
            manifest["input_authority"] = forged
            self._json(manifest_path, manifest)
            input_scope_path = arguments.control_root / "input_scope_receipt.json"
            input_scope = builder.load_json(input_scope_path)
            input_scope["input_authority"] = forged
            self._json(input_scope_path, input_scope)
            self._refresh_manifest_bindings(arguments)
            arguments.allow_partial = True
            with mock.patch.object(builder, "EXPECTED_SCOPE_DATASETS", 1):
                payload, column_csv, receipt = builder.build_payload(arguments)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(receipt["gates"]["source_authority"]["status"], "failed")
            result = builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertEqual(result["json"], "dbp-audit.partial.json")

    def test_campaign_binding_requires_explicit_matching_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self._fixture(root)
            arguments.campaign_binding_sha256 = "f" * 64
            arguments.allow_partial = True
            with mock.patch.object(builder, "EXPECTED_SCOPE_DATASETS", 1):
                payload, column_csv, receipt = builder.build_payload(arguments)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(receipt["status"], "incomplete")
            self.assertEqual(receipt["gates"]["campaign_binding"]["status"], "failed")
            result = builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertEqual(result["json"], "dbp-audit.partial.json")
            self.assertFalse((arguments.output_dir / "dbp-audit.json").exists())

    def test_environment_capture_must_cover_every_dataset_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self._fixture(root)
            environment = builder.load_json(arguments.environment)
            environment["cpu"]["passed_receipts_at_capture"] = 0
            self._json(arguments.environment, environment)
            arguments.allow_partial = True
            with mock.patch.object(builder, "EXPECTED_SCOPE_DATASETS", 1):
                payload, column_csv, receipt = builder.build_payload(arguments)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(receipt["gates"]["environment"]["status"], "failed")
            result = builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertEqual(result["json"], "dbp-audit.partial.json")
            self.assertFalse((arguments.output_dir / "dbp-audit.json").exists())

    def test_partial_mode_cannot_write_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = self._fixture(root)
            java_path = arguments.results_root / "by_dataset/dataset_one/java.jsonl"
            java_row = builder.read_jsonl(java_path)[0]
            java_row["frame_sequence_sha256"] = "b" * 64
            self._jsonl(java_path, [java_row])
            receipt_path = (
                arguments.results_root / "by_dataset/dataset_one/run_receipt.json"
            )
            run_receipt = builder.load_json(receipt_path)
            for row in run_receipt["runtimes"]:
                if row["runtime"] == "java":
                    row["result_sha256"] = builder.sha256_file(java_path)
            self._json(receipt_path, run_receipt)
            arguments.allow_partial = True
            with mock.patch.object(builder, "EXPECTED_SCOPE_DATASETS", 1):
                payload, column_csv, receipt = builder.build_payload(arguments)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(receipt["status"], "incomplete")
            self.assertEqual(receipt["gates"]["cross_runtime"]["status"], "failed")
            result = builder.write_outputs(arguments, payload, column_csv, receipt)
            self.assertEqual(result["json"], "dbp-audit.partial.json")
            self.assertFalse((arguments.output_dir / "dbp-audit.json").exists())
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[3]
                    / "deploy"
                    / "huggingface-compressionbenchmark"
                    / "dbp-audit.schema.json"
                ).read_text()
            )
            jsonschema.validate(
                json.loads(
                    (arguments.output_dir / "dbp-audit.partial.json").read_text()
                ),
                schema,
            )


if __name__ == "__main__":
    unittest.main()
