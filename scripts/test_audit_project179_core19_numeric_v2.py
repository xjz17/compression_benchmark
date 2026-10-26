#!/usr/bin/env python3
"""Fail-closed tests for the Core19 v3 audit-only closure auditor."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import audit_project179_core19_numeric_v1 as base
import audit_project179_core19_numeric_v2 as auditor
from audit_project179_core19_numeric_v2 import AuditError
from test_audit_project179_core19_numeric_v1 import (
    Core19AuditFixture,
    load_json,
    sha256,
    write_json,
)


def empty_coverage() -> dict:
    return {
        "blocking_source_numeric_failure_count": 0,
        "blocking_source_numeric_failures": [],
        "provenance_only_nonready_entry_count": 0,
        "provenance_only_nonready_entries": [],
    }


def trains_fixture() -> tuple[dict, dict, list[dict]]:
    rows = auditor._expected_provenance_rows("TrainsUK2")
    coverage = {
        "blocking_source_numeric_failure_count": 0,
        "blocking_source_numeric_failures": [],
        "provenance_only_nonready_entry_count": len(rows),
        "provenance_only_nonready_entries": copy.deepcopy(rows),
    }
    prepared_rows = []
    metadata = []
    for row in rows:
        index = int(row["catalog_index"])
        prepared_rows.append(auditor._expected_prepared_absent_row(index))
        metadata.append(
            {
                "column_index": str(index),
                "column_name": row["column_name"],
                "logical_type": "string",
            }
        )
    active = {
        "blocking_source_numeric_failures": [],
        "provenance_only_nonready_entries": copy.deepcopy(rows),
        "coverage": copy.deepcopy(coverage),
        "identity": {
            "coverage_classification": copy.deepcopy(coverage),
            "ready_raw_columns": [],
        },
        "columns": [],
        "totals": {
            "blocking_source_numeric_failures": 0,
            "provenance_only_nonready_entries": len(rows),
        },
    }
    return active, {"columns": prepared_rows}, metadata


class DatasetProvenanceValidationTest(unittest.TestCase):
    def assert_code(self, code: str, active: dict, prepared: dict, metadata: list[dict]) -> None:
        with self.assertRaises(AuditError) as caught:
            auditor.validate_dataset_provenance(
                "TrainsUK2", active=active, prepared=prepared, metadata=metadata
            )
        self.assertEqual(caught.exception.code, code)

    def test_exact_trains_inventory_is_accepted(self) -> None:
        active, prepared, metadata = trains_fixture()
        result = auditor.validate_dataset_provenance(
            "TrainsUK2", active=active, prepared=prepared, metadata=metadata
        )
        self.assertEqual(result["count"], 21)
        self.assertEqual(
            result["rows_sha256"], auditor.EXPECTED_TRAINS_PROVENANCE_SHA256
        )
        self.assertEqual(result["indices"], list(auditor.EXPECTED_PROVENANCE_INDICES["TrainsUK2"]))

    def test_coordinated_numeric_disguise_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        for inventory in (
            active["provenance_only_nonready_entries"],
            active["coverage"]["provenance_only_nonready_entries"],
            active["identity"]["coverage_classification"]["provenance_only_nonready_entries"],
        ):
            inventory[0]["catalog_logical_type"] = "integer"
        prepared["columns"][0]["catalog_logical_type"] = "integer"
        metadata[0]["logical_type"] = "integer"
        self.assert_code("provenance_only_identity", active, prepared, metadata)

    def test_actual_prepared_payload_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        prepared["columns"][0].update(
            status="ready",
            source_index=1,
            dtype="int64",
            kind="numeric",
            raw_path="/tmp/not-allowed.bin",
            raw_sha256="0" * 64,
            raw_bytes=8,
            source_rows=1,
            retained_rows=1,
        )
        self.assert_code("provenance_only_actual_present", active, prepared, metadata)

    def test_extra_prepared_path_or_hash_field_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        prepared["columns"][0]["source_path_shadow"] = "/tmp/evil.csv"
        prepared["columns"][0]["source_sha256_shadow"] = "0" * 64
        self.assert_code(
            "provenance_only_prepared_field_drift", active, prepared, metadata
        )

    def test_frozen_prepared_source_binding_drift_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        prepared["columns"][0]["source_path"] = "/tmp/evil.csv"
        prepared["columns"][0]["source_sha256"] = "0" * 64
        self.assert_code("provenance_only_actual_present", active, prepared, metadata)

    def test_active_actual_column_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        active["columns"].append(
            {"key": "TrainsUK2__catalog_1", "column_name": "column_1"}
        )
        self.assert_code("provenance_only_actual_present", active, prepared, metadata)

    def test_metadata_numeric_drift_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        metadata[0]["logical_type"] = "integer"
        self.assert_code("provenance_only_metadata", active, prepared, metadata)

    def test_repeated_inventory_field_drift_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        active["coverage"]["provenance_only_nonready_entries"][0][
            "upstream_status"
        ] = "ready"
        self.assert_code("provenance_only_inventory_drift", active, prepared, metadata)

    def test_extra_field_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        for inventory in (
            active["provenance_only_nonready_entries"],
            active["coverage"]["provenance_only_nonready_entries"],
            active["identity"]["coverage_classification"]["provenance_only_nonready_entries"],
        ):
            inventory[0]["unexpected"] = True
        self.assert_code("provenance_only_field_drift", active, prepared, metadata)

    def test_count_drift_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        active["totals"]["provenance_only_nonready_entries"] = 20
        self.assert_code("provenance_only_counts", active, prepared, metadata)

    def test_string_float_and_bool_counts_are_rejected(self) -> None:
        for value in ("21", 21.0, True):
            with self.subTest(value=value):
                active, prepared, metadata = trains_fixture()
                active["totals"]["provenance_only_nonready_entries"] = value
                self.assert_code(
                    "provenance_only_counts", active, prepared, metadata
                )

    def test_blocking_failure_list_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        active["blocking_source_numeric_failures"] = [{"column": "column_0"}]
        self.assert_code("numeric_nonready", active, prepared, metadata)

    def test_any_other_dataset_provenance_is_rejected(self) -> None:
        active, prepared, metadata = trains_fixture()
        with self.assertRaises(AuditError) as caught:
            auditor.validate_dataset_provenance(
                "UnexpectedDataset", active=active, prepared=prepared, metadata=metadata
            )
        self.assertIn(
            caught.exception.code,
            {"provenance_only_counts", "provenance_only_identity"},
        )


class ProvenanceOutputCoverageTest(unittest.TestCase):
    def make_manifest(self) -> dict:
        indices = list(auditor.EXPECTED_PROVENANCE_INDICES["TrainsUK2"])
        timestamp = set(indices[:9])
        return {
            "timestamp_receipts": [
                {
                    "source_index": index,
                    "column_name": f"column_{index}",
                    "status": "candidate_ready",
                }
                for index in indices
                if index in timestamp
            ],
            "string_receipts": [
                {
                    "source_index": index,
                    "column_name": f"column_{index}",
                    "status": "dropped_remaining_string",
                }
                for index in indices
                if index not in timestamp
            ],
        }

    def test_exact_timestamp_and_string_coverage_passes(self) -> None:
        auditor.validate_output_receipt_coverage(
            "TrainsUK2", output_manifest=self.make_manifest()
        )

    def test_missing_or_duplicate_coverage_fails(self) -> None:
        for mutation in ("missing", "duplicate"):
            with self.subTest(mutation=mutation):
                manifest = self.make_manifest()
                if mutation == "missing":
                    manifest["timestamp_receipts"].pop()
                else:
                    manifest["timestamp_receipts"].append(
                        copy.deepcopy(manifest["timestamp_receipts"][0])
                    )
                with self.assertRaises(AuditError) as caught:
                    auditor.validate_output_receipt_coverage(
                        "TrainsUK2", output_manifest=manifest
                    )
                self.assertEqual(
                    caught.exception.code, "provenance_only_output_receipts"
                )


class IsolatedBaseAuditAdapterTest(unittest.TestCase):
    def test_child_program_compiles(self) -> None:
        compile(auditor._BASE_CHILD, "<isolated-v1-audit>", "exec")

    def test_numeric_missing_waiver_requires_exact_message_and_details(self) -> None:
        specification = {
            "D": {
                "columns": (
                    {"metadata_row": {"column_name": "b"}},
                    {"metadata_row": {"column_name": "a"}},
                )
            }
        }
        with mock.patch.object(
            auditor, "EXPECTED_CATALOG_NUMERIC_ABSENCES", specification
        ):
            self.assertTrue(
                auditor._is_allowed_legacy_v1_assertion(
                    "numeric_candidate_missing",
                    auditor.CATALOG_NUMERIC_MISSING_MESSAGE,
                    {"dataset": "D", "columns": ["a", "b"]},
                )
            )
            for code, message, details in (
                (
                    "numeric_candidate_missing",
                    "wrong message",
                    {"dataset": "D", "columns": ["a", "b"]},
                ),
                (
                    "numeric_candidate_missing",
                    auditor.CATALOG_NUMERIC_MISSING_MESSAGE,
                    {"dataset": "D", "columns": ["a", "b", "c"]},
                ),
                (
                    "numeric_candidate_missing",
                    auditor.CATALOG_NUMERIC_MISSING_MESSAGE,
                    {"dataset": "Other", "columns": ["a", "b"]},
                ),
                (
                    "numeric_candidate_missing",
                    auditor.CATALOG_NUMERIC_MISSING_MESSAGE,
                    {"dataset": "D", "columns": ["b", "a"]},
                ),
            ):
                with self.subTest(code=code, message=message, details=details):
                    self.assertFalse(
                        auditor._is_allowed_legacy_v1_assertion(
                            code, message, details
                        )
                    )

    def test_command_uses_clean_environment_isolated_no_site_python(self) -> None:
        python_path = Path(sys.executable).resolve()
        before = {
            "bindings": {
                "source_contract": {"sha256": "1" * 64},
                "source_top": {"sha256": "2" * 64},
                "active:TrainsUK2": {"sha256": "3" * 64},
            }
        }
        numeric_evidence = {"evidence_sha256": "4" * 64}
        child_record = {
            "base_receipt": {"receipt": True},
            "numeric_all_null_evidence_sha256": "4" * 64,
        }
        completed = __import__("subprocess").CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(child_record) + "\n", stderr=""
        )
        with mock.patch.object(
            auditor, "EXPECTED_BASE_PYTHON_PATH", python_path
        ), mock.patch.object(auditor.subprocess, "run", return_value=completed) as run:
            result = auditor._run_base_audit_subprocess(
                base_python_path=python_path,
                base_auditor_path=Path(auditor.base.__file__).resolve(),
                base_auditor_sha256=sha256(Path(auditor.base.__file__).resolve()),
                auditor_path=Path(auditor.__file__).resolve(),
                auditor_sha256=sha256(Path(auditor.__file__).resolve()),
                numeric_all_null_evidence=numeric_evidence,
                canonical_scope_path=Path("/scope.json"),
                policy_path=Path("/policy.json"),
                producer_path=Path("/producer.py"),
                active_root=Path("/active"),
                backup_root=Path("/backup"),
                output_root=Path("/output"),
                zstd_path=Path("/usr/bin/zstd"),
                before=before,
            )
        command = run.call_args.args[0]
        self.assertEqual(
            command[:6],
            [
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE=1",
                str(python_path),
                "-I",
            ],
        )
        self.assertEqual(command[6:9], ["-S", "-B", "-c"])
        self.assertEqual(result, {"receipt": True})

    def test_structured_child_failure_is_preserved(self) -> None:
        python_path = Path(sys.executable).resolve()
        before = {
            "bindings": {
                "source_contract": {"sha256": "1" * 64},
                "source_top": {"sha256": "2" * 64},
                "active:TrainsUK2": {"sha256": "3" * 64},
            }
        }
        child_error = {
            "code": "numeric_candidate_missing",
            "message": auditor.CATALOG_NUMERIC_MISSING_MESSAGE,
            "details": {"dataset": "Other", "columns": ["x"]},
        }
        completed = __import__("subprocess").CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=json.dumps(
                {"schema": auditor.AUDIT_SCHEMA, "status": "FAIL", "error": child_error}
            )
            + "\n",
        )
        with mock.patch.object(
            auditor, "EXPECTED_BASE_PYTHON_PATH", python_path
        ), mock.patch.object(auditor.subprocess, "run", return_value=completed):
            with self.assertRaises(AuditError) as caught:
                auditor._run_base_audit_subprocess(
                    base_python_path=python_path,
                    base_auditor_path=Path(auditor.base.__file__).resolve(),
                    base_auditor_sha256=sha256(Path(auditor.base.__file__).resolve()),
                    auditor_path=Path(auditor.__file__).resolve(),
                    auditor_sha256=sha256(Path(auditor.__file__).resolve()),
                    numeric_all_null_evidence={"evidence_sha256": "4" * 64},
                    canonical_scope_path=Path("/scope.json"),
                    policy_path=Path("/policy.json"),
                    producer_path=Path("/producer.py"),
                    active_root=Path("/active"),
                    backup_root=Path("/backup"),
                    output_root=Path("/output"),
                    zstd_path=Path("/usr/bin/zstd"),
                    before=before,
                )
        self.assertEqual(caught.exception.code, "base_audit_subprocess")
        self.assertEqual(caught.exception.details["child_error"], child_error)

    def test_fresh_parent_loader_ignores_in_memory_base_audit_replacement(self) -> None:
        base_path = Path(auditor._BASE_AUDITOR_PATH).resolve()

        def fake_audit(**_kwargs):
            return {"status": "forged"}

        with mock.patch.object(auditor.base, "audit", fake_audit):
            fresh = auditor._load_fresh_pinned_base_module(
                base_auditor_path=base_path,
                base_auditor_sha256=sha256(base_path),
            )
        self.assertIsNot(fresh.audit, fake_audit)
        self.assertEqual(
            Path(fresh.audit.__code__.co_filename).resolve(), base_path
        )


class TrainsActualSourceBindingTest(unittest.TestCase):
    def test_exact_header_and_full_hash_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve() / "TrainsUK2.csv"
            original = b"actual_0,actual_3\n1,2\n"
            source.write_bytes(original)
            patches = (
                mock.patch.object(auditor, "EXPECTED_TRAINS_SOURCE_PATH", str(source)),
                mock.patch.object(
                    auditor,
                    "EXPECTED_TRAINS_SOURCE_HEADER",
                    ("actual_0", "actual_3"),
                ),
                mock.patch.object(
                    auditor, "EXPECTED_TRAINS_SOURCE_BYTES", len(original)
                ),
                mock.patch.object(
                    auditor, "EXPECTED_TRAINS_SOURCE_SHA256", sha256(source)
                ),
            )
            for patcher in patches:
                patcher.start()
            try:
                binding = auditor._validate_trains_source(source)
                self.assertEqual(binding["sha256"], sha256(source))

                source.write_bytes(b"actual_0,renamed\n1,2\n")
                with self.assertRaises(AuditError) as caught:
                    auditor._validate_trains_source(source)
                self.assertEqual(caught.exception.code, "provenance_only_source_header")

                source.write_bytes(b"actual_0,actual_3\n9,9\n")
                with self.assertRaises(AuditError) as caught:
                    auditor._validate_trains_source(source)
                self.assertEqual(caught.exception.code, "provenance_only_source_binding")
            finally:
                for patcher in reversed(patches):
                    patcher.stop()


class NumericAllNullExceptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.dataset = "D"
        self.active_root = self.root / "active"
        self.backup_root = self.root / "backup"
        self.prepared_root = self.root / "prepared"
        (self.active_root / "datasets" / self.dataset).mkdir(parents=True)
        self.backup_root.mkdir()
        self.prepared_root.mkdir()
        self.source = self.root / "D.csv"
        self.source.write_text("value\n1\n2\n", encoding="utf-8")
        self.metadata = self.backup_root / "D_column_info.csv"
        self.metadata.write_text(
            "column_index,column_name,logical_type,max_point_number,column_csv_size_bytes\n"
            "0,value,integer,0,2\n"
            "7,legacy,float,0,8\n",
            encoding="utf-8",
        )
        self.catalog_metadata = self.root / "catalog" / "D_column_info.csv"
        self.catalog_metadata.parent.mkdir()
        self.catalog_metadata.write_text(
            "column_index,column_name,logical_type,max_point_number,column_csv_size_bytes\n"
            "0,value,integer,0,2\n",
            encoding="utf-8",
        )
        self.archive = self.backup_root / "D_removed_or_timestamp_columns.csv"
        self.archive.write_text(
            "__row_number__,legacy\n1,null\n2,null\n", encoding="utf-8"
        )
        self.prepared = self.prepared_root / "manifest.json"
        write_json(
            self.prepared,
            {
                "columns": [
                    {
                        "status": "ready",
                        "source_index": 0,
                        "column_name": "value",
                        "source_path": str(self.source),
                        "source_sha256": sha256(self.source),
                        "source_rows": 2,
                        "catalog_index": "0",
                        "catalog_logical_type": "integer",
                        "metadata_path": str(self.catalog_metadata),
                        "metadata_sha256": sha256(self.catalog_metadata),
                    }
                ]
            },
        )
        self.active = self.active_root / "datasets" / self.dataset / "manifest.json"
        self._write_active()
        self.specification = self._specification()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_active(self) -> None:
        write_json(
            self.active,
            {
                "identity": {
                    "prepared_dataset_manifest_path": str(self.prepared),
                    "prepared_dataset_manifest_sha256": sha256(self.prepared),
                    "upstream_metadata_path": str(self.catalog_metadata),
                    "upstream_metadata_sha256": sha256(self.catalog_metadata),
                    "upstream_source_path": str(self.source),
                    "upstream_source_bytes": self.source.stat().st_size,
                    "upstream_source_sha256": sha256(self.source),
                }
            },
        )

    def _specification(self, *, token: str = "null") -> dict:
        return {
            "catalog_metadata_path": str(self.catalog_metadata),
            "audited_metadata_path": str(self.metadata),
            "catalog_metadata_sha256": sha256(self.catalog_metadata),
            "audited_metadata_sha256": sha256(self.metadata),
            "source_path": str(self.source),
            "source_bytes": self.source.stat().st_size,
            "source_sha256": sha256(self.source),
            "source_rows": 2,
            "source_header": ("value",),
            "archive_path": str(self.archive),
            "archive_bytes": self.archive.stat().st_size,
            "archive_sha256": sha256(self.archive),
            "archive_header": ("__row_number__", "legacy"),
            "columns": (
                {
                    "archive_index": 1,
                    "expected_token": token,
                    "metadata_row": {
                        "column_index": "7",
                        "column_name": "legacy",
                        "logical_type": "float",
                        "max_point_number": "0",
                        "column_csv_size_bytes": "8",
                    },
                },
            ),
        }

    def run_validation(self, specification: dict | None = None) -> dict:
        with mock.patch.object(
            auditor,
            "EXPECTED_CATALOG_NUMERIC_ABSENCES",
            {self.dataset: specification or self.specification},
        ):
            return auditor._validate_numeric_all_null_exceptions(
                datasets=[self.dataset],
                active_root=self.active_root,
                backup_root=self.backup_root,
                zstd_path=Path("/unused-for-plain-csv"),
            )

    def test_exact_source_metadata_and_full_archive_null_scan_pass(self) -> None:
        evidence = self.run_validation()
        self.assertEqual(evidence["total"], 1)
        row = evidence["by_dataset"][self.dataset]
        self.assertEqual(row["columns"][0]["semantic_null_rows"], 2)
        self.assertEqual(row["columns"][0]["observed_token_counts"], {"null": 2})
        self.assertEqual(row["actual_source"]["header"], ["value"])
        self.assertEqual(row["backup_archive"]["rows"], 2)

    def test_case_space_and_other_missing_tokens_are_rejected(self) -> None:
        for token in ("NULL", " null ", "NaN", ""):
            with self.subTest(token=token):
                self.archive.write_text(
                    f"__row_number__,legacy\n1,{token}\n2,{token}\n",
                    encoding="utf-8",
                )
                specification = self._specification(token="null")
                specification["archive_bytes"] = self.archive.stat().st_size
                specification["archive_sha256"] = sha256(self.archive)
                with self.assertRaises(AuditError) as caught:
                    self.run_validation(specification)
                self.assertEqual(caught.exception.code, "numeric_all_null_nonnull")

    def test_extra_missing_numeric_metadata_row_is_rejected(self) -> None:
        with self.metadata.open("a", encoding="utf-8") as handle:
            handle.write("8,extra,float,0,1\n")
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "numeric_all_null_identity")

    def test_prepared_source_binding_drift_is_rejected(self) -> None:
        prepared = load_json(self.prepared)
        prepared["columns"][0]["source_sha256"] = "f" * 64
        write_json(self.prepared, prepared)
        self._write_active()
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "numeric_all_null_prepared")

    def test_current_catalog_metadata_drift_is_rejected(self) -> None:
        self.catalog_metadata.write_text(
            "column_index,column_name,logical_type,max_point_number,column_csv_size_bytes\n"
            "0,value,float,0,2\n",
            encoding="utf-8",
        )
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "numeric_all_null_metadata_binding")

    def test_actual_source_payload_drift_is_rejected(self) -> None:
        self.source.write_text("value\n9\n9\n", encoding="utf-8")
        prepared = load_json(self.prepared)
        prepared["columns"][0]["source_sha256"] = sha256(self.source)
        write_json(self.prepared, prepared)
        self._write_active()
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "numeric_all_null_source_binding")

    def test_actual_source_header_drift_is_rejected_after_rebinding(self) -> None:
        self.source.write_text("renamed\n1\n2\n", encoding="utf-8")
        prepared = load_json(self.prepared)
        prepared["columns"][0]["source_sha256"] = sha256(self.source)
        write_json(self.prepared, prepared)
        self._write_active()
        specification = self._specification()
        specification["source_header"] = ("value",)
        with self.assertRaises(AuditError) as caught:
            self.run_validation(specification)
        self.assertEqual(caught.exception.code, "numeric_all_null_source_header")

    def test_archive_header_drift_is_rejected_after_rebinding(self) -> None:
        self.archive.write_text(
            "__row_number__,renamed\n1,null\n2,null\n", encoding="utf-8"
        )
        specification = self._specification()
        specification["archive_bytes"] = self.archive.stat().st_size
        specification["archive_sha256"] = sha256(self.archive)
        with self.assertRaises(AuditError) as caught:
            self.run_validation(specification)
        self.assertEqual(caught.exception.code, "numeric_all_null_archive_header")

    def test_unapproved_dataset_missing_numeric_is_rejected(self) -> None:
        with mock.patch.object(auditor, "EXPECTED_CATALOG_NUMERIC_ABSENCES", {}):
            with self.assertRaises(AuditError) as caught:
                auditor._validate_numeric_all_null_dataset(
                    "Other",
                    active={},
                    active_binding={},
                    prepared={"columns": []},
                    prepared_binding={},
                    metadata=[{"column_name": "x", "logical_type": "integer"}],
                    metadata_stable_binding={},
                    backup_root=self.backup_root,
                    zstd_path=Path("/unused"),
                )
        self.assertEqual(caught.exception.code, "numeric_all_null_unexpected")

    def test_hardlinked_archive_is_rejected(self) -> None:
        os_link = __import__("os").link
        os_link(self.archive, self.root / "archive-hardlink.csv")
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "hardlink_rejected")


class AggregateProvenanceValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets = [f"D{index:02d}" for index in range(18)] + ["TrainsUK2"]
        self.per_dataset = {
            name: {"count": 21 if name == "TrainsUK2" else 0}
            for name in self.datasets
        }
        totals = {
            "blocking_source_numeric_failures": 0,
            "provenance_only_nonready_entries": 21,
        }
        self.contract = {
            "dataset_count": 19,
            "datasets": self.datasets,
            "totals": copy.deepcopy(totals),
        }
        self.top = {
            "datasets": [
                {
                    "dataset": name,
                    "totals": {
                        "blocking_source_numeric_failures": 0,
                        "provenance_only_nonready_entries": 21
                        if name == "TrainsUK2"
                        else 0,
                    },
                }
                for name in self.datasets
            ],
            "totals": copy.deepcopy(totals),
        }

    def run_validation(self) -> None:
        auditor.validate_provenance_aggregates(
            datasets=self.datasets,
            source_contract=self.contract,
            source_top=self.top,
            per_dataset=self.per_dataset,
        )

    def test_exact_aggregate_passes(self) -> None:
        self.run_validation()

    def test_aggregate_count_drift_is_rejected(self) -> None:
        self.contract["totals"]["provenance_only_nonready_entries"] = 22
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "provenance_only_counts")

    def test_aggregate_count_must_be_strict_integer(self) -> None:
        for value in ("21", 21.0, True):
            with self.subTest(value=value):
                self.contract["totals"]["provenance_only_nonready_entries"] = value
                with self.assertRaises(AuditError) as caught:
                    self.run_validation()
                self.assertEqual(caught.exception.code, "provenance_only_counts")
                self.contract["totals"]["provenance_only_nonready_entries"] = 21

    def test_extra_dataset_is_rejected(self) -> None:
        self.per_dataset["ExtraDataset"] = {"count": 1}
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "provenance_only_extra_dataset")

    def test_top_dataset_count_drift_is_rejected(self) -> None:
        self.top["datasets"][0]["totals"]["provenance_only_nonready_entries"] = 1
        with self.assertRaises(AuditError) as caught:
            self.run_validation()
        self.assertEqual(caught.exception.code, "provenance_only_counts")


class Core19AuditV2IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary.name).resolve()
        self.fixture = Core19AuditFixture(temporary_root)
        self.release_scope = temporary_root / "release-scope.json"
        shutil.copyfile(self.fixture.scope, self.release_scope)
        totals = {
            "blocking_source_numeric_failures": 0,
            "chunks": 1,
            "provenance_only_nonready_entries": 0,
            "raw_bytes": 16,
            "ready_columns": 1,
            "retained_rows": 2,
        }
        for dataset in self.fixture.datasets:
            active_path = self.fixture.active / "datasets" / dataset / "manifest.json"
            active = load_json(active_path)
            coverage = empty_coverage()
            active.update(
                blocking_source_numeric_failures=[],
                provenance_only_nonready_entries=[],
                coverage=copy.deepcopy(coverage),
                columns=[],
                totals=copy.deepcopy(totals),
            )
            active["identity"]["coverage_classification"] = copy.deepcopy(coverage)
            write_json(active_path, active)
            output_path = self.fixture.output / "datasets" / dataset / "manifest.json"
            output = load_json(output_path)
            output["identity"]["active_manifest_sha256"] = sha256(active_path)
            write_json(output_path, output)
        self.fixture._build_active_top()
        self.fixture.refresh_top()
        authority_values = {
            "EXPECTED_SCOPE_SHA256": sha256(self.fixture.scope),
            "EXPECTED_POLICY_SHA256": sha256(self.fixture.policy),
            "EXPECTED_PRODUCER_SHA256": sha256(self.fixture.producer),
            "EXPECTED_ZSTD_PATH": str(self.fixture.zstd),
            "EXPECTED_ZSTD_SHA256": sha256(self.fixture.zstd),
        }
        self.patchers = [
            mock.patch.object(base, name, value)
            for name, value in authority_values.items()
        ] + [
            mock.patch.object(auditor.base, name, value)
            for name, value in authority_values.items()
        ] + [
            mock.patch.object(auditor, "EXPECTED_SCOPE_SHA256", sha256(self.fixture.scope)),
            mock.patch.object(auditor, "EXPECTED_CANONICAL_SCOPE_PATH", self.fixture.scope),
            mock.patch.object(auditor, "EXPECTED_PROVENANCE_INDICES", {}),
            mock.patch.object(auditor, "EXPECTED_CATALOG_NUMERIC_ABSENCES", {}),
            mock.patch.object(
                auditor, "EXPECTED_BASE_PYTHON_PATH", Path(sys.executable).resolve()
            ),
            mock.patch.object(
                auditor,
                "EXPECTED_BASE_PYTHON_SHA256",
                sha256(Path(sys.executable).resolve()),
            ),
            mock.patch.object(
                auditor,
                "_run_base_audit_subprocess",
                side_effect=self._run_base_in_process,
            ),
            mock.patch.object(
                auditor,
                "_run_parent_base_audit",
                side_effect=self._run_base_in_process,
            ),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def run_audit(self, *, canonical_scope: Path | None = None, release_scope: Path | None = None):
        return auditor.audit(
            canonical_scope_path=canonical_scope or self.fixture.scope,
            release_scope_path=release_scope or self.release_scope,
            policy_path=self.fixture.policy,
            producer_path=self.fixture.producer,
            active_root=self.fixture.active,
            backup_root=self.fixture.backup,
            output_root=self.fixture.output,
            zstd_path=self.fixture.zstd,
            base_python_path=Path(sys.executable).resolve(),
        )

    def _run_base_in_process(self, **kwargs):
        return auditor.base.audit(
            scope_path=kwargs["canonical_scope_path"],
            policy_path=kwargs["policy_path"],
            producer_path=kwargs["producer_path"],
            active_root=kwargs["active_root"],
            backup_root=kwargs["backup_root"],
            output_root=kwargs["output_root"],
            zstd_path=kwargs["zstd_path"],
        )

    def _authentic_base_receipt(self) -> dict:
        return self._run_base_in_process(
            canonical_scope_path=self.fixture.scope,
            policy_path=self.fixture.policy,
            producer_path=self.fixture.producer,
            active_root=self.fixture.active,
            backup_root=self.fixture.backup,
            output_root=self.fixture.output,
            zstd_path=self.fixture.zstd,
        )

    def test_zero_exception_fixture_passes_and_binds_both_scopes(self) -> None:
        receipt = self.run_audit()
        self.assertEqual(receipt["schema"], auditor.AUDIT_SCHEMA)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["provenance_only"]["total"], 0)
        self.assertEqual(receipt["numeric_all_null_exceptions"]["total"], 0)
        self.assertEqual(
            receipt["input_bindings"]["canonical_scope"]["sha256"],
            receipt["input_bindings"]["frozen_release_scope"]["sha256"],
        )
        self.assertTrue(
            receipt["release_checks"]["canonical_and_frozen_release_scope_bound"]
        )
        self.assertTrue(
            receipt["release_checks"]
            ["provenance_only_rows_catalog_string_absent_and_frozen"]
        )
        self.assertTrue(
            receipt["release_checks"]
            ["numeric_all_null_exceptions_exact_source_header_and_backup_semantic_null"]
        )
        expected_keys = set(base.audit(
            scope_path=self.fixture.scope,
            policy_path=self.fixture.policy,
            producer_path=self.fixture.producer,
            active_root=self.fixture.active,
            backup_root=self.fixture.backup,
            output_root=self.fixture.output,
            zstd_path=self.fixture.zstd,
        )["release_checks"]) | {
            "canonical_and_frozen_release_scope_bound",
            "provenance_only_rows_catalog_string_absent_and_frozen",
            "numeric_all_null_exceptions_exact_source_header_and_backup_semantic_null",
        }
        self.assertEqual(set(receipt["release_checks"]), expected_keys)

    def test_parent_semantic_exception_scan_runs_once(self) -> None:
        original = auditor._validate_numeric_all_null_exceptions
        with mock.patch.object(
            auditor,
            "_validate_numeric_all_null_exceptions",
            wraps=original,
        ) as validate:
            self.run_audit()
        self.assertEqual(validate.call_count, 1)

    def test_minimal_fake_base_pass_receipt_is_rejected(self) -> None:
        fake = {
            "schema": auditor.base.AUDIT_SCHEMA,
            "status": "PASS",
            "mode": "cold_read_only",
        }
        with mock.patch.object(
            auditor, "_run_base_audit_subprocess", return_value=fake
        ):
            with self.assertRaises(AuditError) as caught:
                self.run_audit()
        self.assertEqual(caught.exception.code, "base_receipt_shape")

    def test_complete_fake_receipt_dataset_summary_drift_is_rejected(self) -> None:
        mutations = {
            "prepared_sources": lambda row: row["prepared_sources"][0].__setitem__(
                "sha256", "f" * 64
            ),
            "output_stable_bindings": lambda row: row["output_stable_bindings"][0].__setitem__(
                "sha256", "f" * 64
            ),
            "backup": lambda row: row["backup"].__setitem__("sha256", "f" * 64),
            "stable_identity": lambda row: row["metadata_stable_binding"].__setitem__(
                "stat_ino", row["metadata_stable_binding"]["stat_ino"] + 1
            ),
            "counts": lambda row: row.__setitem__(
                "prepared_numeric_columns", row["prepared_numeric_columns"] + 1
            ),
            "selection": lambda row: row.__setitem__(
                "selected_source_indices", [0, 99]
            ),
            "chunks": lambda row: row.__setitem__("chunks", row["chunks"] + 1),
            "bytes": lambda row: row.__setitem__(
                "selected_bytes", row["selected_bytes"] + 8
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(field=name):
                fake = copy.deepcopy(self._authentic_base_receipt())
                mutate(fake["datasets"][0])
                with mock.patch.object(
                    auditor, "_run_base_audit_subprocess", return_value=fake
                ):
                    with self.assertRaises(AuditError) as caught:
                        self.run_audit()
                self.assertEqual(caught.exception.code, "base_receipt_mismatch")

    def test_payload_change_after_child_receipt_is_rejected_by_parent_reaudit(self) -> None:
        def child_then_mutate(**kwargs):
            receipt = self._run_base_in_process(**kwargs)
            payload = Path(
                receipt["datasets"][0]["output_stable_bindings"][0]["path"]
            )
            payload.write_bytes(payload.read_bytes() + b"tamper")
            return receipt

        with mock.patch.object(
            auditor, "_run_base_audit_subprocess", side_effect=child_then_mutate
        ):
            with self.assertRaises(AuditError) as caught:
                self.run_audit()
        self.assertIn(
            caught.exception.code,
            {"raw_hash", "chunk_hash", "file_changed", "dataset_counts"},
        )

    def test_payload_change_after_parent_receipt_is_rejected_by_final_gate(self) -> None:
        def parent_then_mutate(**kwargs):
            receipt = self._run_base_in_process(**kwargs)
            payload = Path(
                receipt["datasets"][0]["output_stable_bindings"][0]["path"]
            )
            payload.write_bytes(payload.read_bytes() + b"tamper")
            return receipt

        with mock.patch.object(
            auditor, "_run_parent_base_audit", side_effect=parent_then_mutate
        ):
            with self.assertRaises(AuditError) as caught:
                self.run_audit()
        self.assertEqual(caught.exception.code, "file_changed")

    def test_extra_output_file_after_parent_receipt_is_rejected(self) -> None:
        def parent_then_add(**kwargs):
            receipt = self._run_base_in_process(**kwargs)
            (self.fixture.output / "unexpected.bin").write_bytes(b"unexpected")
            return receipt

        with mock.patch.object(
            auditor, "_run_parent_base_audit", side_effect=parent_then_add
        ):
            with self.assertRaises(AuditError) as caught:
                self.run_audit()
        self.assertEqual(caught.exception.code, "output_file_set")

    def test_policy_producer_and_zstd_changes_after_parent_are_rejected(self) -> None:
        for role, path in (
            ("policy", self.fixture.policy),
            ("producer", self.fixture.producer),
            ("zstd", self.fixture.zstd),
        ):
            with self.subTest(role=role):
                original = path.read_bytes()

                def parent_then_mutate(**kwargs):
                    receipt = self._run_base_in_process(**kwargs)
                    path.write_bytes(original + b"tamper")
                    return receipt

                try:
                    with mock.patch.object(
                        auditor,
                        "_run_parent_base_audit",
                        side_effect=parent_then_mutate,
                    ):
                        with self.assertRaises(AuditError) as caught:
                            self.run_audit()
                    self.assertEqual(caught.exception.code, "file_changed")
                finally:
                    path.write_bytes(original)

    def test_in_memory_base_function_replacement_is_rejected(self) -> None:
        original = auditor.base.audit

        def fake_audit(**_kwargs):
            return {"schema": auditor.base.AUDIT_SCHEMA, "status": "PASS"}

        with mock.patch.object(auditor.base, "audit", fake_audit):
            with self.assertRaises(AuditError) as caught:
                self.run_audit()
        self.assertEqual(caught.exception.code, "authority_base_auditor_function")
        self.assertIs(auditor.base.audit, original)

    def test_release_scope_tamper_is_rejected(self) -> None:
        release = load_json(self.release_scope)
        release["scope_id"] = "tampered"
        write_json(self.release_scope, release)
        with self.assertRaises(AuditError) as caught:
            self.run_audit()
        self.assertEqual(caught.exception.code, "authority_scope_hash")

    def test_same_bytes_at_wrong_canonical_path_is_rejected(self) -> None:
        wrong = Path(self.temporary.name).resolve() / "wrong-canonical.json"
        shutil.copyfile(self.fixture.scope, wrong)
        with self.assertRaises(AuditError) as caught:
            self.run_audit(canonical_scope=wrong)
        self.assertEqual(caught.exception.code, "authority_canonical_scope_path")

    def test_scope_files_must_be_distinct(self) -> None:
        with self.assertRaises(AuditError) as caught:
            self.run_audit(release_scope=self.fixture.scope)
        self.assertEqual(caught.exception.code, "scope_alias")


if __name__ == "__main__":
    unittest.main()
