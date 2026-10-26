#!/usr/bin/env python3
"""Tests for the deterministic project179 THULab timestamp policy builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_project179_thulab_utc_policy_v1 as builder  # noqa: E402
import prepare_project179_thulab_numeric_v1 as materializer  # noqa: E402


SCOPE = REPOSITORY_ROOT / "benchmark/reference/datasets/project_dataset_scope_20260928.json"
AUDIT = (
    REPOSITORY_ROOT
    / "analysis_outputs/project179_numeric_utc_20260928/THULAB_TSFILE_SCHEMA_AUDIT.json"
)
POLICY = (
    REPOSITORY_ROOT
    / "benchmark/reference/datasets/project179_thulab_utc_policy_20260928.json"
)
TEMPORAL_SAMPLES = (
    REPOSITORY_ROOT
    / "analysis_outputs/project179_numeric_utc_20260928/temporal_samples_by_dataset"
)


class Project179ThulabUtcPolicyBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = builder.build_policy(scope_path=SCOPE, audit_path=AUDIT)
        cls.by_id = {row["dataset_id"]: row for row in cls.generated["datasets"]}

    def test_generated_artifact_is_current_and_generation_is_byte_deterministic(self):
        expected = builder.policy_bytes(self.generated)
        self.assertEqual(POLICY.read_bytes(), expected)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            builder.write_policy(first, self.generated)
            regenerated = builder.build_policy(scope_path=SCOPE, audit_path=AUDIT)
            builder.write_policy(second, regenerated)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), hashlib.sha256(second.read_bytes()).hexdigest())

    def test_complete_scope_binding_and_materializer_schema_compatibility(self):
        scope, rows, scope_sha256 = materializer.load_scope(SCOPE)
        policies, policy_sha256 = materializer.load_policies(
            POLICY,
            scope_id=scope["scope_id"],
            scope_sha256=scope_sha256,
            scope_rows=rows,
        )
        self.assertEqual(len(policies), 160)
        self.assertEqual(set(policies), {row.dataset_id for row in rows})
        self.assertEqual(self.generated["scope_sha256"], scope_sha256)
        self.assertEqual(policy_sha256, hashlib.sha256(POLICY.read_bytes()).hexdigest())
        scope_by_id = {row.dataset_id: row for row in rows}
        for dataset_id, policy in self.by_id.items():
            self.assertEqual(policy["repo_id"], scope_by_id[dataset_id].repo_id)
            self.assertEqual(policy["revision"], scope_by_id[dataset_id].revision)

    def test_decisions_follow_audit_classes_and_no_stored_coordinate_claims_event_utc(self):
        self.assertEqual(
            self.generated["decision_counts"],
            {
                "admit_exact_utc": 34,
                "admit_stored_epoch_coordinate": 123,
                "admit_dataset_adapter": 3,
                "block": 0,
            },
        )
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        audit_by_id = {row["dataset_id"]: row for row in audit["datasets"]}
        for dataset_id, policy in self.by_id.items():
            evidence_class = audit_by_id[dataset_id]["utc_evidence_class"]
            if evidence_class == builder.EXACT_UTC_CLASS:
                self.assertEqual(policy["decision"], "admit_exact_utc")
                self.assertTrue(policy["timestamp"]["utc_instant_verified"])
            elif evidence_class in builder.STORED_COORDINATE_CLASSES:
                self.assertEqual(policy["decision"], "admit_stored_epoch_coordinate")
                self.assertIs(policy["timestamp"]["event_utc_instant_claim"], False)
                self.assertNotIn("utc_instant_verified", policy["timestamp"])
            else:
                self.assertEqual(policy["decision"], "admit_dataset_adapter")
            self.assertTrue(policy["evidence"])

    def test_admitted_rows_use_single_table_int64_unix_microseconds(self):
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        audit_by_id = {row["dataset_id"]: row for row in audit["datasets"]}
        unit_map = builder.OBSERVED_UNIT_TO_POLICY_UNIT
        for dataset_id, policy in self.by_id.items():
            if policy["decision"] == "admit_dataset_adapter":
                self.assertEqual(policy["table_name"], "__single_table__")
                for rule in policy["mixed_axis_adapter"]["source_rules"]:
                    if rule["action"] != "materialize":
                        continue
                    self.assertEqual(rule["timestamp"]["output_unit"], "us")
                    self.assertEqual(
                        rule["timestamp"]["output_column"], "timestamp_unix_us"
                    )
                continue
            timestamp = policy["timestamp"]
            self.assertEqual(policy["table_name"], "__single_table__")
            self.assertEqual(timestamp["source_column"], "Time")
            self.assertEqual(
                timestamp["source_unit"],
                unit_map[audit_by_id[dataset_id]["observed_time_axis_unit"]],
            )
            self.assertEqual(timestamp["output_unit"], "us")
            self.assertEqual(timestamp["output_column"], "timestamp_unix_us")
        self.assertEqual(self.generated["default_output_type"], "int64")
        self.assertEqual(self.generated["default_output_unit"], "us")

    def test_observed_units_override_readme_prose_for_named_datasets(self):
        expected_units = {
            "qiskit_calibration_drift": "ms",
            "deep-space-probes": "s",
            "dmi-aarhus-weather-data": "s",
            "electricity_demand": "s",
            "cic-ids-2017-flowprep": "us",
        }
        for dataset_id, source_unit in expected_units.items():
            policy = self.by_id[dataset_id]
            self.assertEqual(policy["timestamp"]["source_unit"], source_unit)
        qiskit_evidence = "\n".join(self.by_id["qiskit_calibration_drift"]["evidence"])
        self.assertIn("README says UTC epoch us", qiskit_evidence)
        self.assertIn("observed_time_axis_unit=epoch_ms", qiskit_evidence)

    def test_mixed_surrogate_datasets_have_receipt_bound_adapters(self):
        self.assertEqual(
            {
                dataset_id
                for dataset_id, policy in self.by_id.items()
                if policy["decision"] == "admit_dataset_adapter"
            },
            {"finance1k", "fulus_lyd_rates_2025", "knowable"},
        )
        for dataset_id in ("finance1k", "fulus_lyd_rates_2025", "knowable"):
            self.assertNotIn("timestamp", self.by_id[dataset_id])
            self.assertIn("mixed_axis_adapter", self.by_id[dataset_id])
        finance_rules = {
            row["rule_id"]: row
            for row in self.by_id["finance1k"]["mixed_axis_adapter"]["source_rules"]
        }
        self.assertEqual(finance_rules["finance1k-panel"]["expected_source_file_count"], 446)
        self.assertEqual(finance_rules["finance1k-stats"]["expected_source_file_count"], 446)
        self.assertEqual(
            finance_rules["finance1k-tickers-exclusion"]["action"],
            "exclude_entire_source_file",
        )
        fulus_filter = self.by_id["fulus_lyd_rates_2025"]["mixed_axis_adapter"][
            "source_rules"
        ][0]["row_filter"]
        self.assertEqual(
            (
                fulus_filter["expected_input_rows"],
                fulus_filter["expected_output_rows"],
                fulus_filter["expected_drop_rows"],
            ),
            (15536, 15471, 65),
        )

    def test_scope_audit_identity_mismatch_fails_closed(self):
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        audit["datasets"][0]["dataset_id"] = "outside_scope"
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "audit.json"
            mutated.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(builder.PolicyBuildError, "identities differ"):
                builder.build_policy(scope_path=SCOPE, audit_path=mutated)

    def test_temporal_candidate_coverage_is_closed_and_format_specific(self):
        self.assertEqual(
            self.generated["temporal_string_coverage"],
            {
                "candidate_definition_count": 158,
                "candidate_dataset_count": 40,
                "candidate_rule_count": 48,
                "converted_rule_count": 45,
                "converted_definition_count": 155,
                "dropped_clock_only_rule_count": 3,
                "dropped_clock_only_definition_count": 3,
                "converted_definition_counts_by_format_class": {
                    "date_only_utc_midnight_coordinate": 2,
                    "embedded_offset": 38,
                    "embedded_offset_fractional": 30,
                    "explicit_z": 1,
                    "explicit_z_fractional": 3,
                    "integer_epoch_days": 7,
                    "naive_assume_utc_coordinate": 74,
                },
                "dropped_definition_counts_by_format_class": {
                    "clock_only_drop": 3,
                },
                "tag_definition_count": 38,
                "field_definition_count": 117,
                "ordinary_string_contract": (
                    "STRING/TEXT columns absent from temporal_string_fields are dropped; "
                    "TAG strings may partition entities but are not retained as payload"
                ),
                "heterogeneous_schema_contract": (
                    "rules are a dataset-level union and apply independently only to frozen "
                    "source-file schemas containing the declared name/type/category"
                ),
            },
        )
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        audit_by_id = {row["dataset_id"]: row for row in audit["datasets"]}
        for dataset_id, policy in self.by_id.items():
            expected = sum(audit_by_id[dataset_id]["timestamp_string_candidates"].values())
            actual = sum(
                row["candidate_definition_count"]
                for row in policy["temporal_string_fields"]
            ) + sum(
                row["candidate_definition_count"]
                for row in policy["dropped_timestamp_like_string_fields"]
            )
            self.assertEqual(actual, expected, dataset_id)

    def test_temporal_rules_preserve_exact_offsets_and_bound_coordinate_conventions(self):
        dmi = self.by_id["dmi-aarhus-weather-data"]["temporal_string_fields"][0]
        self.assertEqual(dmi["source_name"], "reference_time")
        self.assertEqual(dmi["source_category"], "TAG")
        self.assertEqual(dmi["format"], "%Y-%m-%dT%H:%M:%SZ")
        self.assertTrue(dmi["event_utc_instant_claim"])

        source_bound = self.by_id["dmi-aarhus-weather-data"][
            "source_bound_temporal_fields"
        ]
        self.assertEqual(len(source_bound), 1)
        numeric_reference = source_bound[0]
        self.assertEqual(numeric_reference["source_path"], "data/data.tsfile")
        self.assertEqual(
            numeric_reference["source_sha256"],
            "d40d9fdca20f018eb48676d9730033c439ab99e1a821d7b94cce01876b3a1ff6",
        )
        self.assertEqual(numeric_reference["table_name"], "dmi_aarhus_data")
        self.assertEqual(
            numeric_reference["source_schema_sha256"],
            "a06fde8814d9018899ce0bc948c238096cc95d80089731810192dd32bd7ca878",
        )
        self.assertEqual(
            (numeric_reference["source_name"], numeric_reference["source_type"],
             numeric_reference["source_category"]),
            ("reference_time", "INT64", "FIELD"),
        )
        self.assertEqual(
            (numeric_reference["parse_kind"], numeric_reference["source_unit"],
             numeric_reference["output_unit"]),
            ("integer_epoch", "s", "us"),
        )
        self.assertTrue(numeric_reference["event_utc_instant_claim"])
        self.assertTrue(all(
            not policy["source_bound_temporal_fields"]
            for dataset_id, policy in self.by_id.items()
            if dataset_id != "dmi-aarhus-weather-data"
        ))

        eea = {
            row["source_name"]: row
            for row in self.by_id["eea_pm25_forecasting"]["temporal_string_fields"]
        }
        self.assertEqual(eea["dt_local"]["timezone"], "embedded_offset")
        self.assertEqual(eea["dt_local"]["format"], "%Y-%m-%d %H:%M:%S%z")
        self.assertTrue(eea["dt_local"]["event_utc_instant_claim"])
        self.assertEqual(
            eea["start"]["timezone_basis"], "project_utc_coordinate_convention"
        )
        self.assertIs(eea["start"]["event_utc_instant_claim"], False)

        jovian = {
            row["source_name"]: row
            for row in self.by_id["jovian_lightning"]["temporal_string_fields"]
        }
        self.assertEqual(jovian["t_utc_end"]["format"], "%Y-%m-%dT%H:%M:%S.%fZ")
        self.assertTrue(jovian["t_utc_end"]["event_utc_instant_claim"])

        start = self.by_id["bitcoin"]["temporal_string_fields"][0]
        self.assertEqual(start["source_category"], "TAG")
        self.assertEqual(start["format_class"], "naive_assume_utc_coordinate")
        self.assertIs(start["event_utc_instant_claim"], False)

    def test_date_only_epoch_days_and_clock_only_are_not_overclaimed(self):
        knowable = {
            row["source_name"]: row
            for row in self.by_id["knowable"]["temporal_string_fields"]
        }
        disclosure = knowable["disclosure_date"]
        self.assertEqual(disclosure["parse_kind"], "integer_epoch")
        self.assertEqual(disclosure["source_unit"], "d")
        self.assertEqual(disclosure["semantic_kind"], "unix_epoch_day_coordinate")
        self.assertIs(disclosure["event_utc_instant_claim"], False)
        self.assertEqual(knowable["published_at"]["timezone"], "embedded_offset")
        self.assertIs(knowable["published_at"]["event_utc_instant_claim"], False)
        self.assertEqual(
            knowable["published_at"]["accepted_formats"],
            ["%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S.%f%z"],
        )
        self.assertEqual(knowable["retrieved_at"]["format"], "%Y-%m-%d %H:%M:%S.%f%z")

        fulus = self.by_id["fulus_lyd_rates_2025"]["temporal_string_fields"][0]
        self.assertEqual(fulus["format_class"], "date_only_utc_midnight_coordinate")
        self.assertEqual(fulus["format"], "%Y-%m-%d")
        self.assertIs(fulus["event_utc_instant_claim"], False)

        reg = self.by_id["reg_forecasting"]
        self.assertEqual(
            [row["source_name"] for row in reg["temporal_string_fields"]], ["date"]
        )
        self.assertEqual(
            [row["source_name"] for row in reg["dropped_timestamp_like_string_fields"]],
            ["time"],
        )
        traqq = self.by_id["traqq_weekly_activity"]
        self.assertEqual(traqq["temporal_string_fields"], [])
        self.assertEqual(
            {row["source_name"] for row in traqq["dropped_timestamp_like_string_fields"]},
            {"start_time", "end_time"},
        )
        for row in (
            reg["dropped_timestamp_like_string_fields"]
            + traqq["dropped_timestamp_like_string_fields"]
        ):
            self.assertEqual(row["decision"], "drop_clock_only_without_calendar_date")

    def test_rules_are_union_scoped_for_heterogeneous_source_files(self):
        expected_applicability = (
            "each frozen source file whose table schema contains the matching name, "
            "STRING type, and source_category; absence in other heterogeneous source "
            "files is allowed"
        )
        for policy in self.by_id.values():
            for row in policy["temporal_string_fields"]:
                self.assertEqual(row["applicability"], expected_applicability)
                self.assertEqual(row["output_type"], "int64")
                self.assertEqual(row["output_unit"], "us")
                self.assertEqual(
                    row["output_name"], f"{row['source_name']}__timestamp_unix_us"
                )

    def test_temporal_sample_bindings_are_complete_and_content_addressed(self):
        evidence = self.generated["temporal_string_evidence"]
        self.assertEqual(evidence["file_count"], 40)
        self.assertEqual(len(evidence["files"]), 40)
        self.assertEqual(
            {row["path"] for row in evidence["files"]},
            {path.name for path in TEMPORAL_SAMPLES.glob("*.json")},
        )
        for row in evidence["files"]:
            path = TEMPORAL_SAMPLES / row["path"]
            self.assertEqual(
                row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
