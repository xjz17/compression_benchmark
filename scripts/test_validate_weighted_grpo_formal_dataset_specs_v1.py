#!/usr/bin/env python3
"""Focused tests for the frozen weighted-GRPO formal dataset-spec gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_weighted_grpo_formal_dataset_specs_v1 import (  # noqa: E402
    CAMPAIGN_DATE,
    DEFAULT_AUDIT,
    DEFAULT_COHORT,
    DEFAULT_REVISIONS,
    ROOT,
    SpecGateError,
    validate_bundle,
    validate_calendar_audit_evidence,
    validate_helsinki_civil_audit_evidence,
    validate_portuguese_civil_audit_evidence,
    validate_formal_spec,
    validate_prepared_admissions,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class FormalDatasetSpecGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = _load(DEFAULT_AUDIT)
        cls.cohort = _load(DEFAULT_COHORT)
        cls.revisions = _load(DEFAULT_REVISIONS)
        cls.cohort_by_id = {
            item["id"]: item
            for item in cls.cohort["datasets"]
            if item["role"] == "train_validation"
        }
        cls.revision_by_id = {
            item["dataset_id"]: item["revision"]
            for item in cls.revisions["sources"]
        }

    def test_current_bundle_has_exact_ten_new_specs_and_eleven_admitted_sources(self):
        result = validate_bundle()
        self.assertEqual(result["fixed_train_validation_sources"], 16)
        self.assertEqual(result["approved"], 11)
        self.assertEqual(result["blocked"], 5)
        self.assertFalse(result["complete_cohort_trainable"])
        self.assertEqual(
            result["approved_dataset_ids"],
            sorted([
                "weatherAUS",
                "electricity_load_2011_2014",
                "eea_pm25_forecasting",
                "exchange_rate",
                "bitcoin_price_timeseries",
                "online_retail_2_uci",
                "national_illness",
                "scaphandre_cpu_usage",
                "geomagnetic_kp_index",
                "finland_traffic",
                "deep-space-probes",
            ]),
        )
        self.assertEqual(
            result["blocked_dataset_ids"],
            sorted([
                "ETDataset-ett",
                "METR-LA",
                "freshretailnet_50k",
                "gluco_tsfm_benchmark",
                "app_flow",
            ]),
        )
        specs = result["checked_formal_specs"]
        self.assertEqual(specs["eea_pm25_forecasting"]["source_files"], 8)
        self.assertEqual(
            specs["electricity_load_2011_2014"]["native_type_counts"],
            {"FLOAT": 370},
        )
        self.assertFalse(
            specs["electricity_load_2011_2014"]
            ["source_backed_portuguese_civil_audit"]["exact_event_utc"]
        )
        self.assertEqual(specs["eea_pm25_forecasting"]["numeric_columns_including_time"], 69)
        self.assertEqual(specs["bitcoin_price_timeseries"]["numeric_columns_including_time"], 6)
        self.assertEqual(specs["scaphandre_cpu_usage"]["native_type_counts"], {"FLOAT": 1})
        self.assertEqual(specs["finland_traffic"]["native_type_counts"], {"FLOAT": 1})
        self.assertFalse(
            specs["finland_traffic"]
            ["source_backed_helsinki_civil_audit"]["exact_event_utc"]
        )
        self.assertEqual(
            specs["geomagnetic_kp_index"]["native_type_counts"],
            {"DOUBLE": 1, "INT64": 2},
        )
        self.assertEqual(specs["exchange_rate"]["native_type_counts"], {"FLOAT": 8})
        self.assertEqual(specs["national_illness"]["native_type_counts"], {"FLOAT": 7})
        self.assertEqual(
            specs["online_retail_2_uci"]["native_type_counts"], {"FLOAT": 1}
        )
        self.assertEqual(
            specs["exchange_rate"]["source_backed_calendar_audit"]["row_count"],
            7588,
        )
        self.assertEqual(
            specs["national_illness"]["source_backed_calendar_audit"]["row_count"],
            966,
        )
        self.assertEqual(
            specs["online_retail_2_uci"]["source_backed_calendar_audit"]["row_count"],
            739,
        )
        self.assertEqual(
            specs["deep-space-probes"]["source_overlay"],
            "juliensimon/deep-space-probes@ed64a3ba6e24baccb33574d3b36562ba5c1b7332",
        )
        self.assertTrue(
            specs["deep-space-probes"]["invalid_benchmarkhost_mirror_rejected"]
        )

    def test_campaign_dates_and_resolved_metr_overlap_evidence_are_current(self):
        self.assertEqual(self.audit["created_date"], CAMPAIGN_DATE)
        for admission in self.audit["sources"]:
            spec_path = admission.get("spec_path")
            if spec_path:
                self.assertEqual(_load(ROOT / spec_path)["created_date"], CAMPAIGN_DATE)

        metr = next(
            item for item in self.audit["sources"] if item["dataset_id"] == "METR-LA"
        )
        self.assertEqual(
            metr["blockers"],
            ["timezone_or_epoch_basis_not_documented_for_t0_timestamp"],
        )
        self.assertIn(
            "all_7088922_adjacent_window_comparisons_pass_exact_FLOAT32_bit_agreement",
            metr["resolved_evidence"],
        )

        stale = copy.deepcopy(self.audit)
        stale["created_date"] = "2026-09-18"
        with self.assertRaisesRegex(SpecGateError, "created_date"):
            self._validate_mutated_audit(stale)

    def test_deep_space_exact_upstream_is_admitted_and_benchmarkhost_mirror_is_rejected(self):
        deep = next(
            item for item in self.audit["sources"]
            if item["dataset_id"] == "deep-space-probes"
        )
        self.assertEqual(deep["status"], "approved")
        self.assertTrue(deep["formal_gate_eligible"])
        self.assertEqual(
            deep["revision"],
            "ed64a3ba6e24baccb33574d3b36562ba5c1b7332",
        )
        rejected_mirror = {
            "dataset_id": "deep-space-probes",
            "source_group": "hf:juliensimon/deep-space-probes",
            "preparation_spec_sha256": "66187ec5cd3dacbb5fa239518899b2eb73ed6c0b6ebdf4acc1cfe1855caee4f7",
            "source": {
                "repo_id": "BenchmarkHost/deep-space-probes",
                "repo_path": "deep-space-probes.tsfile",
                "revision": "2ce6a7c5b0e22c3b6aa19459e4fa16a433f7bc8c",
                "bytes": 1,
                "sha256": "0" * 64,
                "git_oid": "0" * 40,
                "format": "tsfile",
                "upstream_locator": "https://huggingface.co/datasets/BenchmarkHost/deep-space-probes",
            },
            "canonical_rebuild": {},
            "selection": {},
            "input_contract": {"timestamp": {}},
            "totals": {},
        }
        with self.assertRaisesRegex(
            SpecGateError,
            "does not descend from the overlay",
        ):
            validate_prepared_admissions([rejected_mirror])

        admitted = validate_prepared_admissions([{"dataset_id": "weatherAUS"}])
        self.assertEqual(admitted["admitted_dataset_ids"], ["weatherAUS"])
        self.assertTrue(admitted["blocked_artifacts_rejected_before_byte_verification"])

    def _validate_mutated_audit(self, audit) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps(audit), encoding="utf-8")
            validate_bundle(path, DEFAULT_COHORT, DEFAULT_REVISIONS)

    def test_dataset_or_revision_substitution_is_rejected(self):
        audit = copy.deepcopy(self.audit)
        audit["sources"][0]["dataset_id"] = "replacement_dataset"
        with self.assertRaisesRegex(SpecGateError, "fixed source substitution"):
            self._validate_mutated_audit(audit)

        audit = copy.deepcopy(self.audit)
        target = next(item for item in audit["sources"] if item["dataset_id"] == "bitcoin_price_timeseries")
        target["revision"] = "0" * 40
        with self.assertRaisesRegex(SpecGateError, "revision substitution"):
            self._validate_mutated_audit(audit)

    def test_blocked_source_cannot_gain_a_spec_or_lose_its_reason(self):
        audit = copy.deepcopy(self.audit)
        target = next(item for item in audit["sources"] if item["dataset_id"] == "ETDataset-ett")
        target["spec_path"] = (
            "analysis_outputs/weight_conditioned_grpo_20260917_v1/specs/"
            "bitcoin_price_timeseries_tsfile_v1.json"
        )
        with self.assertRaisesRegex(SpecGateError, "blocked source"):
            self._validate_mutated_audit(audit)

        audit = copy.deepcopy(self.audit)
        target = next(item for item in audit["sources"] if item["dataset_id"] == "ETDataset-ett")
        target["blockers"] = []
        with self.assertRaisesRegex(SpecGateError, "blocked source"):
            self._validate_mutated_audit(audit)

    def _new_spec(self, dataset_id: str):
        admission = next(
            item for item in self.audit["sources"] if item["dataset_id"] == dataset_id
        )
        return _load(ROOT / admission["spec_path"])

    def _schema_fingerprint(self, dataset_id: str):
        admission = next(
            item for item in self.audit["sources"] if item["dataset_id"] == dataset_id
        )
        return admission.get("numeric_schema_sha256")

    def test_time_is_both_int64_numeric_output_and_global_ordering_key(self):
        for dataset_id in self.audit["summary"]["new_formal_spec_dataset_ids"]:
            if dataset_id == "deep-space-probes":
                continue
            spec = self._new_spec(dataset_id)
            result = validate_formal_spec(
                spec,
                cohort_entry=self.cohort_by_id[dataset_id],
                revision=self.revision_by_id[dataset_id],
                expected_numeric_schema_sha256=self._schema_fingerprint(dataset_id),
            )
            self.assertGreaterEqual(result["numeric_columns_including_time"], 2)

            mutated = copy.deepcopy(spec)
            mutated["row_contract"]["emit_ordering_key_as_numeric_column"] = False
            with self.assertRaisesRegex(SpecGateError, "row contract"):
                validate_formal_spec(
                    mutated,
                    cohort_entry=self.cohort_by_id[dataset_id],
                    revision=self.revision_by_id[dataset_id],
                    expected_numeric_schema_sha256=self._schema_fingerprint(dataset_id),
                )

    def test_source_width_and_collection_file_identity_are_not_silently_changed(self):
        spec = self._new_spec("scaphandre_cpu_usage")
        widened = copy.deepcopy(spec)
        widened["table"]["value_columns"][0]["tsfile_type"] = "DOUBLE"
        with self.assertRaisesRegex(SpecGateError, "numeric source type/schema substitution"):
            validate_formal_spec(
                widened,
                cohort_entry=self.cohort_by_id["scaphandre_cpu_usage"],
                revision=self.revision_by_id["scaphandre_cpu_usage"],
                expected_numeric_schema_sha256=self._schema_fingerprint("scaphandre_cpu_usage"),
            )

        collection = self._new_spec("eea_pm25_forecasting")
        collection["source"]["files"][0]["path"] = collection["source"]["files"][1]["path"]
        with self.assertRaisesRegex(SpecGateError, "silently substitutes"):
            validate_formal_spec(
                collection,
                cohort_entry=self.cohort_by_id["eea_pm25_forecasting"],
                revision=self.revision_by_id["eea_pm25_forecasting"],
                expected_numeric_schema_sha256=self._schema_fingerprint("eea_pm25_forecasting"),
            )

    def test_only_native_numeric_fields_are_selected(self):
        spec = self._new_spec("geomagnetic_kp_index")
        spec["table"]["value_columns"].append({
            "source_name": "storm_level",
            "output_name": "storm_level",
            "tsfile_type": "STRING",
        })
        with self.assertRaisesRegex(SpecGateError, "non-numeric"):
            validate_formal_spec(
                spec,
                cohort_entry=self.cohort_by_id["geomagnetic_kp_index"],
                revision=self.revision_by_id["geomagnetic_kp_index"],
                expected_numeric_schema_sha256=self._schema_fingerprint("geomagnetic_kp_index"),
            )

    def test_calendar_date_encoding_is_explicit_and_does_not_infer_timezone(self):
        spec = self._new_spec("exchange_rate")
        result = validate_formal_spec(
            spec,
            cohort_entry=self.cohort_by_id["exchange_rate"],
            revision=self.revision_by_id["exchange_rate"],
            expected_numeric_schema_sha256=self._schema_fingerprint("exchange_rate"),
        )
        self.assertEqual(result["numeric_columns_including_time"], 9)

        inferred = copy.deepcopy(spec)
        inferred["timestamp"]["source_event_timezone_inferred"] = True
        with self.assertRaisesRegex(SpecGateError, "must not infer"):
            validate_formal_spec(
                inferred,
                cohort_entry=self.cohort_by_id["exchange_rate"],
                revision=self.revision_by_id["exchange_rate"],
                expected_numeric_schema_sha256=self._schema_fingerprint("exchange_rate"),
            )

        audit_path = ROOT / next(
            item["audit_path"]
            for item in self.audit["sources"]
            if item["dataset_id"] == "exchange_rate"
        )
        report = _load(audit_path)
        record = next(
            item for item in report["records"] if item["dataset_id"] == "exchange_rate"
        )
        substituted = copy.deepcopy(record)
        substituted["source_csv_sha256"] = "0" * 64
        with self.assertRaisesRegex(SpecGateError, "source CSV fingerprint mismatch"):
            validate_calendar_audit_evidence(
                spec,
                substituted,
                audit_path_text=str(audit_path.relative_to(ROOT)),
            )

    def test_portuguese_civil_coordinate_cannot_be_relabelled_as_exact_utc(self):
        dataset_id = "electricity_load_2011_2014"
        spec = self._new_spec(dataset_id)
        audit_path_text = next(
            item["audit_path"]
            for item in self.audit["sources"]
            if item["dataset_id"] == dataset_id
        )
        report = _load(ROOT / audit_path_text)
        record = next(
            item for item in report["records"] if item["dataset_id"] == dataset_id
        )
        exact_utc = copy.deepcopy(record)
        exact_utc["event_utc_instant_claim"] = True
        with self.assertRaisesRegex(SpecGateError, "claims exact UTC instants"):
            validate_portuguese_civil_audit_evidence(
                spec,
                exact_utc,
                audit_path_text=audit_path_text,
            )

    def test_helsinki_civil_coordinate_cannot_be_relabelled_as_exact_utc(self):
        dataset_id = "finland_traffic"
        spec = self._new_spec(dataset_id)
        audit_path_text = next(
            item["audit_path"]
            for item in self.audit["sources"]
            if item["dataset_id"] == dataset_id
        )
        report = _load(ROOT / audit_path_text)
        record = next(
            item for item in report["records"] if item["dataset_id"] == dataset_id
        )
        exact_utc = copy.deepcopy(record)
        exact_utc["event_utc_instant_claim"] = True
        with self.assertRaisesRegex(SpecGateError, "claims exact UTC instants"):
            validate_helsinki_civil_audit_evidence(
                spec,
                exact_utc,
                audit_path_text=audit_path_text,
            )


if __name__ == "__main__":
    unittest.main()
