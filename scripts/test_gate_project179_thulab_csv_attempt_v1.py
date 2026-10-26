#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "attempt_gate", HERE / "gate_project179_thulab_csv_attempt_v1.py"
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")


SPECIAL = [
    "ETDataset-ett", "trending_stocks_yahoo_finance",
    "africa_senegal_mauritania_crops_spi", "bitcoin",
    "dmi-aarhus-weather-data", "forbush_decreases", "macroecon",
]


class Fixture:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "output"
        self.root.mkdir()
        self.scope = base / "scope.json"
        self.raw = base / "raw" / "DOWNLOAD_MANIFEST.json"
        self.run_id = "attempt-test-r1"
        self.scope_id = "project-dataset-scope-20260928"
        self.policy_sha = "1" * 64
        self.exception_sha = "2" * 64
        self.exporter_sha = "3" * 64
        self.validator_sha = "4" * 64
        self.coordinator_sha = "5" * 64
        self.ids = SPECIAL + [f"dataset-{index:03d}" for index in range(153)]
        assert len(self.ids) == 160
        scope_rows = [
            {"id": dataset_id, "repo_id": f"repo/{dataset_id}", "revision": "main"}
            for dataset_id in self.ids
        ]
        write_json(self.scope, {
            "scope_id": self.scope_id,
            "timestamp_dataset_count": 160,
            "timestamp_datasets": scope_rows,
        })
        self.scope_sha = sha(self.scope)
        raw_rows = []
        for row in scope_rows:
            dataset_id = row["id"]
            receipt_rel = f"_receipts/{dataset_id}.json"
            write_json(self.raw.parent / receipt_rel, {
                "schema": "thulab-timestamp-dataset-download-receipt-v1",
                "status": "complete", "dataset_id": dataset_id,
                "repo_id": row["repo_id"], "revision": row["revision"],
                "tsfile_count": 1, "tsfile_bytes": 5,
                "files": [{
                    "path": "data.tsfile", "bytes": 5, "sha256": "7" * 64,
                }],
            })
            raw_rows.append({
                "dataset_id": dataset_id, "repo_id": row["repo_id"],
                "revision": row["revision"],
                "receipt": f"/legacy/remote/thulab_real_timestamp/{receipt_rel}",
            })
        write_json(self.raw, {
            "schema": "thulab-timestamp-scope-download-v1",
            "status": "complete", "scope_id": self.scope_id,
            "scope_sha256": self.scope_sha, "dataset_count": 160,
            "expected_dataset_count": 160, "datasets": raw_rows,
        })
        self.raw_sha = sha(self.raw)
        self.results = [self._success(index, row) for index, row in enumerate(scope_rows)]
        self._write_attempt_results()

    def _report(self, dataset_id: str) -> dict:
        rescans = []
        row_audits = []
        if dataset_id in GATE.FULL_VALUE_DATASETS:
            rescans = [{"dataset_id": dataset_id, "status": "passed"}]
        if dataset_id == GATE.MACRO:
            row_audits = [{
                "dataset_id": dataset_id, "status": "passed", "source_files": 64,
                "source_rows": 1_284_546, "eligible_rows": 1_284_538,
                "dropped_rows": 8, "affected_sources": [{} for _ in range(8)],
            }]
        return {
            "schema": GATE.VALIDATION_SCHEMA, "status": "passed",
            "dataset_count": 1, "validation_workers": 1,
            "all_160_required": False,
            "datasets": [{
                "dataset_id": dataset_id, "actual_bytes": 0,
                "emitted_rows": 1, "value_columns": ["c0000"],
            }],
            "schema_metadata_pass": [{"dataset_id": dataset_id, "status": "passed"}],
            "nan_source_audits": [{"dataset_id": dataset_id, "status": "passed"}],
            "nan_prefix_audits": [], "pilot_source_rescans": rescans,
            "row_exception_audits": row_audits,
        }

    def _success(self, index: int, scope_row: dict) -> dict:
        dataset_id = scope_row["id"]
        collection_root = self.root / "collections" / dataset_id
        dataset_dir = collection_root / dataset_id
        dataset_dir.mkdir(parents=True)
        for name, body in (("data.csv", "timestamp_unix_us,series_id,c0000\n1,0,2\n"),
                           ("series_map.csv", "series_id\n0\n"),
                           ("columns.csv", "output_column\nc0000\n")):
            (dataset_dir / name).write_text(body, encoding="utf-8")
        files = []
        for name in GATE.OUTPUT_FILES[:-1]:
            path = dataset_dir / name
            files.append({
                "path": name, "bytes_decimal_20": f"{path.stat().st_size:020d}",
                "sha256": sha(path),
            })
        placeholder = {
            "status": "complete", "dataset_id": dataset_id,
            "repo_id": scope_row["repo_id"], "revision": scope_row["revision"],
            "scope_id": self.scope_id, "scope_sha256": self.scope_sha,
            "timestamp_policy_file_sha256": self.policy_sha,
            "row_exception_policy_file_sha256": self.exception_sha,
            "exporter_sha256": self.exporter_sha,
            "cap_bytes_decimal_20": f"{GATE.DATASET_CAP:020d}",
            "actual_total_bytes_decimal_20": "0" * 20,
            "emitted_rows_decimal_20": f"{1:020d}",
            "retained_value_column_count_decimal_20": f"{1:020d}",
            "files": [*files, {
                "path": "dataset_manifest.json", "bytes_decimal_20": "0" * 20,
                "sha256": None,
            }],
        }
        manifest_path = dataset_dir / "dataset_manifest.json"
        # Fixed-width decimal strings make this two-pass self-size stable.
        write_json(manifest_path, placeholder)
        manifest_bytes = manifest_path.stat().st_size
        total = sum((dataset_dir / row["path"]).stat().st_size for row in files) + manifest_bytes
        placeholder["actual_total_bytes_decimal_20"] = f"{total:020d}"
        placeholder["files"][-1]["bytes_decimal_20"] = f"{manifest_bytes:020d}"
        write_json(manifest_path, placeholder)
        assert manifest_path.stat().st_size == manifest_bytes
        collection_manifest = {
            "schema": GATE.COLLECTION_SCHEMA, "status": "complete",
            "scope_id": self.scope_id, "scope_sha256": self.scope_sha,
            "timestamp_policy_file_sha256": self.policy_sha,
            "row_exception_policy_file_sha256": self.exception_sha,
            "dataset_byte_cap": GATE.DATASET_CAP, "dataset_count": 1,
            "datasets": [{
                "dataset_id": dataset_id,
                "receipt": f"{dataset_id}/dataset_manifest.json",
                "actual_capped_bytes": total,
            }],
        }
        collection_manifest_path = collection_root / "CSV_EXPORT_MANIFEST.json"
        write_json(collection_manifest_path, collection_manifest)
        validation_path = self.root / "validations" / f"{dataset_id}.json"
        validation_report = self._report(dataset_id)
        validation_report["datasets"][0]["actual_bytes"] = total
        write_json(validation_path, validation_report)
        attempt_dir = self.root / "attempts" / dataset_id
        attempt_dir.mkdir(parents=True)
        for name in ("export.stdout.log", "export.stderr.log",
                     "validation.stdout.log", "validation.stderr.log"):
            (attempt_dir / name).write_text("", encoding="utf-8")
        evidence_names = {
            "export_stdout": "export.stdout.log",
            "export_stderr": "export.stderr.log",
            "validation_stdout": "validation.stdout.log",
            "validation_stderr": "validation.stderr.log",
        }
        log_evidence = {}
        for key, name in evidence_names.items():
            path = attempt_dir / name
            log_evidence[key] = {
                "path": str(path.relative_to(self.root)), "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
        result = {
            "schema": "project179-thulab-csv-best-effort-dataset-attempt-v1",
            "run_id": self.run_id,
            "dataset_id": dataset_id, "repo_id": scope_row["repo_id"],
            "revision": scope_row["revision"], "binding_sha256": "6" * 64,
            "source_file_count": 1, "source_bytes": 5, "reservation_bytes": 100,
            "source_files": [{"path": "data.tsfile", "sha256": "7" * 64}],
            "status": "success", "stage": "complete",
            "exactly_once_attempt_index": index,
            "logs": {
                key: f"attempts/{dataset_id}/{name}"
                for key, name in evidence_names.items()
            },
            "log_evidence": log_evidence,
            "collection": {
                "path": f"collections/{dataset_id}",
                "manifest_path": f"collections/{dataset_id}/CSV_EXPORT_MANIFEST.json",
                "manifest_sha256": sha(collection_manifest_path),
                "dataset_receipt_path": f"collections/{dataset_id}/{dataset_id}/dataset_manifest.json",
                "dataset_receipt_sha256": sha(manifest_path),
            },
            "validation": {
                "path": f"validations/{dataset_id}.json", "sha256": sha(validation_path),
            },
            "export_process": {"returncode": 0, "signal": None},
            "validation_process": {"returncode": 0, "signal": None},
            "failure": None,
        }
        attempt_path = attempt_dir / "ATTEMPT.json"
        write_json(attempt_path, result)
        result["evidence"] = {
            "attempt": {
                "path": str(attempt_path.relative_to(self.root)),
                "sha256": sha(attempt_path), "bytes": attempt_path.stat().st_size,
            },
            **log_evidence,
        }
        return result

    def fail_outcome(
        self, index: int, *, stage: str, error_type: str,
        returncode: int | None, signal: int | None,
    ) -> None:
        result = self.results[index]
        dataset_id = result["dataset_id"]
        process_name = "validation_process" if stage == "validation" else "export_process"
        result.update({
            "status": "failed", "stage": stage, "validation": None,
            "failure": {
                "stage": stage, "returncode": returncode, "signal": signal,
                "error_type": error_type, "error_message": error_type,
                "last_stdout_event": None, "last_source_event": None,
                "failed_export_stages": [],
            },
        })
        result[process_name] = {"returncode": returncode, "signal": signal}
        failure_path = self.root / "attempts" / dataset_id / "FAILURE.json"
        write_json(failure_path, {
            "schema": "project179-thulab-csv-best-effort-failure-v1",
            "dataset_id": dataset_id, "failure": result["failure"],
        })
        result["failure"]["evidence"] = [{
            "path": str(failure_path.relative_to(self.root)),
            "sha256": sha(failure_path), "bytes": failure_path.stat().st_size,
        }]
        attempt_path = self.root / "attempts" / dataset_id / "ATTEMPT.json"
        write_json(attempt_path, {
            key: value for key, value in result.items() if key != "evidence"
        })
        ref = result["evidence"]["attempt"]
        ref["sha256"], ref["bytes"] = sha(attempt_path), attempt_path.stat().st_size
        self._write_attempt_results()

    def _write_attempt_results(self) -> None:
        failures = sum(row["status"] == "failed" for row in self.results)
        attempt = {
            "schema": GATE.ATTEMPT_SCHEMA,
            "status": "attempts_complete_with_failures" if failures else "attempts_complete",
            "run_id": self.run_id, "scope_id": self.scope_id,
            "scope_sha256": self.scope_sha, "raw_manifest_sha256": self.raw_sha,
            "timestamp_policy_file_sha256": self.policy_sha,
            "row_exception_policy_file_sha256": self.exception_sha,
            "exporter_sha256": self.exporter_sha,
            "validator_sha256": self.validator_sha,
            "coordinator_sha256": self.coordinator_sha,
            "dataset_count": 160, "attempted_dataset_count": 160,
            "success_count": 160 - failures, "failure_count": failures,
            "dataset_ids": self.ids, "results": self.results,
        }
        write_json(self.root / "ATTEMPT_RESULTS.json", attempt)

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            attempt_results=self.root / "ATTEMPT_RESULTS.json", scope=self.scope,
            raw_manifest=self.raw, output_root=self.root, run_id=self.run_id,
            expected_scope_id=self.scope_id, expected_scope_sha256=self.scope_sha,
            expected_raw_manifest_sha256=self.raw_sha,
            expected_timestamp_policy_sha256=self.policy_sha,
            expected_row_exception_policy_sha256=self.exception_sha,
            expected_exporter_sha256=self.exporter_sha,
            expected_validator_sha256=self.validator_sha,
            expected_coordinator_sha256=self.coordinator_sha,
            failure_json=self.root / "FAILURE_DATASETS.json",
            failure_csv=self.root / "FAILURE_DATASETS.csv",
            pass_gate=self.root / "PASS_COMPLETION_GATE.json",
            formal_completion_gate=self.root / "FORMAL_COMPLETION_GATE.json",
        )


class AttemptGateTests(unittest.TestCase):
    def make(self) -> tuple[tempfile.TemporaryDirectory, Fixture]:
        temp = tempfile.TemporaryDirectory()
        return temp, Fixture(Path(temp.name))

    def test_all_success_publishes_formal_gate(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        self.assertEqual(GATE.main(self.argv(fixture.args())), 0)
        formal = json.loads(fixture.args().formal_completion_gate.read_text())
        self.assertEqual(formal["status"], "complete")
        self.assertEqual(formal["validated_dataset_count"], 160)

    def test_native_failure_is_preserved_and_never_counted_success(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        fixture.fail_outcome(
            9, stage="validation", error_type="native_signal",
            returncode=-11, signal=11,
        )
        self.assertEqual(GATE.main(self.argv(fixture.args())), 0)
        passed = json.loads(fixture.args().pass_gate.read_text())
        failed = json.loads(fixture.args().failure_json.read_text())
        self.assertEqual(passed["status"], "complete_with_failures")
        self.assertEqual(passed["validated_success_count"], 159)
        self.assertEqual(failed["failed_datasets"][0]["signal"], 11)
        self.assertFalse(fixture.args().formal_completion_gate.exists())

    def test_admission_failure_without_subprocess_is_preserved(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        fixture.fail_outcome(
            10, stage="admission", error_type="admission_failed",
            returncode=None, signal=None,
        )
        failure, passed, _csv, formal = GATE.audit(fixture.args())
        self.assertEqual(failure["failed_datasets"][0]["returncode"], None)
        self.assertEqual(passed["validated_success_count"], 159)
        self.assertIsNone(formal)

    def test_schema_preflight_failure_accepts_not_run_or_failed_child(self) -> None:
        for returncode in (None, 1):
            with self.subTest(returncode=returncode):
                temp, fixture = self.make()
                try:
                    fixture.fail_outcome(
                        12, stage="export", error_type="schema_preflight_failed",
                        returncode=returncode, signal=None,
                    )
                    failure, passed, _csv, formal = GATE.audit(fixture.args())
                    self.assertEqual(failure["failed_dataset_count"], 1)
                    self.assertEqual(passed["status"], "complete_with_failures")
                    self.assertIsNone(formal)
                finally:
                    temp.cleanup()

    def test_postcondition_failure_after_zero_returncode_stays_failed(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        fixture.fail_outcome(
            13, stage="validation", error_type="postcondition_failed",
            returncode=0, signal=None,
        )
        failure, passed, _csv, formal = GATE.audit(fixture.args())
        self.assertEqual(failure["failed_datasets"][0]["returncode"], 0)
        self.assertEqual(passed["failed_dataset_ids"], [fixture.ids[13]])
        self.assertIsNone(formal)

    def test_tampered_failure_reason_not_bound_by_attempt_receipt_fails(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        fixture.fail_outcome(
            14, stage="export", error_type="subprocess_nonzero",
            returncode=1, signal=None,
        )
        fixture.results[14]["failure"]["error_message"] = "tampered summary"
        fixture._write_attempt_results()
        with self.assertRaises(GATE.AttemptGateError):
            GATE.audit(fixture.args())

    def test_coverage_gap_or_duplicate_fails_closed(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        fixture.results[10]["dataset_id"] = fixture.results[9]["dataset_id"]
        fixture._write_attempt_results()
        with self.assertRaises(GATE.AttemptGateError):
            GATE.audit(fixture.args())

    def test_materialized_hash_mutation_fails_closed(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        dataset_id = fixture.ids[8]
        path = fixture.root / "collections" / dataset_id / dataset_id / "data.csv"
        path.write_text("mutated\n", encoding="utf-8")
        with self.assertRaises(GATE.AttemptGateError):
            GATE.audit(fixture.args())

    def test_success_without_checked_validation_fails_closed(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        fixture.results[11]["validation"] = None
        fixture._write_attempt_results()
        with self.assertRaises(GATE.AttemptGateError):
            GATE.audit(fixture.args())

    def test_macro_success_requires_independent_exception_audit(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        report_path = fixture.root / "validations" / "macroecon.json"
        report = json.loads(report_path.read_text())
        report["row_exception_audits"] = []
        write_json(report_path, report)
        fixture.results[6]["validation"]["sha256"] = sha(report_path)
        fixture._write_attempt_results()
        with self.assertRaises(GATE.AttemptGateError):
            GATE.audit(fixture.args())

    def test_canonical_raw_receipt_symlink_fails_closed(self) -> None:
        temp, fixture = self.make()
        self.addCleanup(temp.cleanup)
        receipt_path = fixture.raw.parent / "_receipts" / f"{fixture.ids[15]}.json"
        target = fixture.raw.parent / "outside-receipt.json"
        target.write_bytes(receipt_path.read_bytes())
        receipt_path.unlink()
        receipt_path.symlink_to(target)
        with self.assertRaises(GATE.AttemptGateError):
            GATE.audit(fixture.args())

    @staticmethod
    def argv(args: argparse.Namespace) -> list[str]:
        values = []
        for key, value in vars(args).items():
            values.extend(["--" + key.replace("_", "-"), str(value)])
        return values


if __name__ == "__main__":
    unittest.main()
