#!/usr/bin/env python3
"""Fail-closed tests for the dedicated Forbush native-read CSV gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate_project179_thulab_csv_native_v1 as gate  # noqa: E402


EXPORTER_SHA = "1" * 64
VALIDATOR_SHA = "2" * 64
SCOPE_SHA = "3" * 64
POLICY_SHA = "4" * 64


def _encode(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _fixed(value: int) -> str:
    return f"{value:020d}"


def _temporal_empty_fields() -> list[dict]:
    fields = []
    for source_name in sorted(gate.TEMPORAL_EMPTY_EXPECTED):
        expected = gate.TEMPORAL_EMPTY_EXPECTED[source_name]
        empty_count = expected["source_non_null_empty_count"]
        null_count = expected["source_null_count"]
        fields.append({
            "source_name": source_name,
            "source_type": "STRING",
            "source_category": "FIELD",
            "output_column": expected["output_column"],
            "conversion": "policy_temporal_string_to_int64_us",
            "representation": gate.TEMPORAL_EMPTY_REPRESENTATION,
            "source_non_null_empty_count": empty_count,
            "source_non_null_empty_row_ordinals": list(range(empty_count)),
            "source_null_count": null_count,
            "source_null_row_ordinals": list(
                range(empty_count, empty_count + null_count)
            ),
            "source_valid_count": expected["source_valid_count"],
        })
    return fields


def _temporal_empty_wrapper(fields: list[dict]) -> dict:
    return {
        "source_path": gate.SOURCE,
        "source_sha256": gate.SOURCE_SHA,
        "table_name": gate.ID,
        "source_schema_sha256": gate.SCHEMA_SHA,
        "source_rows": gate.SOURCE_ROWS,
        "fields": copy.deepcopy(fields),
    }


def _write_receipt(path: Path, receipt: dict) -> None:
    self_rows = [row for row in receipt["files"] if row["path"] == "dataset_manifest.json"]
    if len(self_rows) != 1:
        raise AssertionError("fixture must have exactly one manifest self-entry")
    self_row = self_rows[0]
    self_row["bytes_decimal_20"] = "0" * 20
    self_row["sha256"] = None
    payload = _encode(receipt)
    self_row["bytes_decimal_20"] = _fixed(len(payload))
    final = _encode(receipt)
    if len(final) != len(payload):
        raise AssertionError("fixed-width manifest size did not converge")
    path.write_bytes(final)


class NativeGateFixture:
    def __init__(self, root: Path) -> None:
        self.output = root / "csv-output"
        self.dataset_dir = self.output / gate.ID
        self.dataset_dir.mkdir(parents=True)

        payloads = {
            "data.csv": b"timestamp_unix_us,series_id,c0000\n1,0,\n",
            "series_map.csv": b"series_id,table_name,typed_tags_json,source_paths_json\n0,forbush_decreases,[],[\\\"forbush_decreases.tsfile\\\"]\n",
            "columns.csv": b"column,table_name,source_name,source_type,output_dtype,conversion\nc0000,forbush_decreases,value,DOUBLE,float64,identity\n",
        }
        file_rows: list[dict] = []
        for name in gate.FILES[:-1]:
            target = self.dataset_dir / name
            target.write_bytes(payloads[name])
            file_rows.append({
                "path": name,
                "bytes_decimal_20": _fixed(target.stat().st_size),
                "sha256": gate.digest(target),
            })
        file_rows.append({
            "path": "dataset_manifest.json",
            "bytes_decimal_20": "0" * 20,
            "sha256": None,
        })
        temporal_empty_fields = _temporal_empty_fields()
        self.receipt = {
            "status": "complete",
            "dataset_id": gate.ID,
            "exporter_sha256": EXPORTER_SHA,
            "scope_sha256": SCOPE_SHA,
            "timestamp_policy_file_sha256": POLICY_SHA,
            "emitted_rows_decimal_20": _fixed(8726),
            "lost_rows_due_to_byte_cap_decimal_20": _fixed(0),
            "candidate_value_column_count_decimal_20": _fixed(23),
            "retained_value_column_count_decimal_20": _fixed(23),
            "dropped_trailing_value_column_count_decimal_20": _fixed(0),
            "row_exception_filtered_rows": 0,
            "source_tsfiles": [{"path": gate.SOURCE, "sha256": gate.SOURCE_SHA}],
            "source_scans": [{
                "source_path": gate.SOURCE,
                "source_sha256": gate.SOURCE_SHA,
                "source_schema_sha256": gate.SCHEMA_SHA,
                "table_name": gate.ID,
                "source_rows": gate.SOURCE_ROWS,
                "read_mode": "rowwise_direct_source_bound",
                "dataframe_attempted": False,
                "rowwise_fallback": False,
                "temporal_empty_string_audits": copy.deepcopy(
                    temporal_empty_fields
                ),
            }],
            "files": file_rows,
        }
        _write_receipt(self.dataset_dir / "dataset_manifest.json", self.receipt)

        numeric = [f"numeric_{index:02d}" for index in range(21)]
        native_audit = {
            "source_path": gate.SOURCE,
            "source_sha256": gate.SOURCE_SHA,
            "source_sha256_before": gate.SOURCE_SHA,
            "source_sha256_after": gate.SOURCE_SHA,
            "table_name": gate.ID,
            "source_schema_sha256": gate.SCHEMA_SHA,
            "query_mode": "rowwise_direct_source_bound",
            "dataframe_attempted": False,
            "rowwise_fallback": False,
        }
        self.report = {
            "status": "passed",
            "dataset_count": 1,
            "datasets": [{"dataset_id": gate.ID}],
            "pilot_source_rescans": [{
                "dataset_id": gate.ID,
                "status": "passed",
                "source_files": 1,
                "source_rows": 8726,
                "compared_output_rows": 8726,
                "all_null_cells": 78024,
                "candidate_value_column_count": 23,
                "compared_numeric_fields": numeric,
                "compared_temporal_fields": [
                    "cme_datetime_utc",
                    "solar_source_datetime_utc",
                ],
                "temporal_empty_string_audits": [
                    _temporal_empty_wrapper(temporal_empty_fields)
                ],
                "native_query_audits": [native_audit],
                "no_rows_or_value_columns_dropped": True,
            }],
            "schema_metadata_pass": [{
                "dataset_id": gate.ID,
                "status": "passed",
                "candidate_value_column_count": 23,
                "retained_value_column_count": 23,
                "dropped_value_column_count": 0,
            }],
        }

    def check(self, report: dict | None = None) -> dict:
        return gate.check_native(
            self.report if report is None else report,
            self.output,
            EXPORTER_SHA,
            VALIDATOR_SHA,
            SCOPE_SHA,
            POLICY_SHA,
        )

    def rewrite_receipt(self) -> None:
        _write_receipt(self.dataset_dir / "dataset_manifest.json", self.receipt)


class NativeGateTests(unittest.TestCase):
    def test_valid_full_value_fixture_and_actual_hashes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NativeGateFixture(Path(temporary))
            evidence = fixture.check()
            self.assertEqual(evidence["source_rows"], 8726)
            self.assertEqual(evidence["output_rows"], 8726)
            self.assertEqual(evidence["all_null_cells"], 78024)
            self.assertEqual(evidence["value_columns"], 23)
            self.assertEqual(set(evidence["output_hashes"]), set(gate.FILES))
            for name, digest in evidence["output_hashes"].items():
                self.assertEqual(digest, gate.digest(fixture.dataset_dir / name))

    def test_independent_report_loss_mode_and_identity_drift_fail_closed(self) -> None:
        cases = {
            "dropped_row": lambda report: report["pilot_source_rescans"][0].update(
                compared_output_rows=8725
            ),
            "dropped_column": lambda report: report["pilot_source_rescans"][0].update(
                candidate_value_column_count=22
            ),
            "too_few_nulls": lambda report: report["pilot_source_rescans"][0].update(
                all_null_cells=78023
            ),
            "dataframe_mode": lambda report: report["pilot_source_rescans"][0][
                "native_query_audits"
            ][0].update(query_mode="dataframe_then_rowwise", dataframe_attempted=True),
            "rowwise_fallback": lambda report: report["pilot_source_rescans"][0][
                "native_query_audits"
            ][0].update(rowwise_fallback=True),
            "wrong_source_sha": lambda report: report["pilot_source_rescans"][0][
                "native_query_audits"
            ][0].update(source_sha256="a" * 64),
            "changed_source_after": lambda report: report["pilot_source_rescans"][0][
                "native_query_audits"
            ][0].update(source_sha256_after="b" * 64),
            "wrong_schema": lambda report: report["pilot_source_rescans"][0][
                "native_query_audits"
            ][0].update(source_schema_sha256="c" * 64),
            "wrong_empty_source": lambda report: report["pilot_source_rescans"][0][
                "temporal_empty_string_audits"
            ][0].update(source_sha256="f" * 64),
            "wrong_true_null_count": lambda report: report[
                "pilot_source_rescans"
            ][0]["temporal_empty_string_audits"][0]["fields"][0].update(
                source_null_count=6440
            ),
            "wrong_output_column": lambda report: report[
                "pilot_source_rescans"
            ][0]["temporal_empty_string_audits"][0]["fields"][0].update(
                output_column="c0020"
            ),
            "tampered_empty_position": lambda report: report[
                "pilot_source_rescans"
            ][0]["temporal_empty_string_audits"][0]["fields"][0][
                "source_non_null_empty_row_ordinals"
            ].__setitem__(-1, 8070),
            "tampered_true_null_position": lambda report: report[
                "pilot_source_rescans"
            ][0]["temporal_empty_string_audits"][0]["fields"][0][
                "source_null_row_ordinals"
            ].__setitem__(-1, 8070),
        }
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NativeGateFixture(Path(temporary))
            for name, mutate in cases.items():
                with self.subTest(name=name):
                    changed = copy.deepcopy(fixture.report)
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        fixture.check(changed)

    def test_export_receipt_row_column_source_and_mode_drift_fail_closed(self) -> None:
        cases = {
            "dropped_row": lambda receipt: receipt.update(
                emitted_rows_decimal_20=_fixed(8725)
            ),
            "dropped_column": lambda receipt: receipt.update(
                retained_value_column_count_decimal_20=_fixed(22),
                dropped_trailing_value_column_count_decimal_20=_fixed(1),
            ),
            "filtered_row": lambda receipt: receipt.update(row_exception_filtered_rows=1),
            "wrong_source_sha": lambda receipt: receipt["source_tsfiles"][0].update(
                sha256="d" * 64
            ),
            "wrong_scan_schema": lambda receipt: receipt["source_scans"][0].update(
                source_schema_sha256="e" * 64
            ),
            "dataframe_mode": lambda receipt: receipt["source_scans"][0].update(
                read_mode="dataframe_then_rowwise", dataframe_attempted=True
            ),
            "fallback": lambda receipt: receipt["source_scans"][0].update(
                rowwise_fallback=True
            ),
            "wrong_empty_source_table": lambda receipt: receipt["source_scans"][
                0
            ].update(table_name="other"),
            "wrong_empty_true_null_count": lambda receipt: receipt[
                "source_scans"
            ][0]["temporal_empty_string_audits"][0].update(source_null_count=6440),
            "tampered_empty_position": lambda receipt: receipt["source_scans"][0][
                "temporal_empty_string_audits"
            ][0]["source_non_null_empty_row_ordinals"].__setitem__(-1, 8070),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = NativeGateFixture(Path(temporary))
                mutate(fixture.receipt)
                fixture.rewrite_receipt()
                with self.assertRaises(ValueError):
                    fixture.check()

    def test_actual_output_sha_conflict_fails_even_when_size_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = NativeGateFixture(Path(temporary))
            target = fixture.dataset_dir / "data.csv"
            original = target.read_bytes()
            tampered = original.replace(b"1,0,", b"2,0,", 1)
            self.assertEqual(len(tampered), len(original))
            target.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "actual SHA mismatch"):
                fixture.check()

    def test_extra_or_symlink_output_is_rejected(self) -> None:
        for mode in ("extra", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = NativeGateFixture(Path(temporary))
                if mode == "extra":
                    (fixture.dataset_dir / "unexpected").write_bytes(b"x")
                else:
                    real = fixture.dataset_dir / "data.csv"
                    link = fixture.dataset_dir / "data.link"
                    link.symlink_to(real)
                    real.unlink()
                    link.rename(fixture.dataset_dir / "data.csv")
                with self.assertRaises(ValueError):
                    fixture.check()


if __name__ == "__main__":
    unittest.main()
