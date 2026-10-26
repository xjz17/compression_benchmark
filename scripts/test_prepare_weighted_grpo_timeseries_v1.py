#!/usr/bin/env python3
"""Unit tests for the weighted-GRPO time-series preparation contract."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_weighted_grpo_timeseries_v1 import (  # noqa: E402
    COHORT_ASSIGNMENT_SHA256,
    COHORT_POLICY_SHA256,
    COHORT_VERSION,
    DEFAULT_COHORT_MANIFEST,
    FORMAT_VERSION,
    PreparationError,
    TimestampContract,
    audit_local_source,
    equal_prefix_allocation,
    load_cohort_manifest,
    load_preparation_contract,
    pack_numeric_token,
    parse_timestamp_token,
    max_rows_per_block,
    prepare_dataset,
    validate_cohort_manifest,
    validate_contract_against_cohort,
    validate_prepared_cohort_manifests,
    verify_prepared_dataset,
)


class CohortManifestTest(unittest.TestCase):
    def test_versioned_manifest_has_balanced_training_and_isolated_sources(self):
        manifest = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
        summary = validate_cohort_manifest(manifest)
        self.assertEqual(summary["version"], COHORT_VERSION)
        self.assertEqual(summary["assignment_sha256"], COHORT_ASSIGNMENT_SHA256)
        self.assertEqual(summary["policy_sha256"], COHORT_POLICY_SHA256)
        self.assertEqual(summary["domain_count"], 8)
        self.assertEqual(summary["train_validation_count"], 16)
        self.assertEqual(summary["test_candidate_count"], 32)
        self.assertEqual(set(summary["train_counts_by_domain"].values()), {2})
        self.assertTrue(all(
            count >= 1 for count in summary["test_counts_by_domain"].values()
        ))

    def test_cross_role_source_group_is_rejected(self):
        manifest = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
        changed = json.loads(json.dumps(manifest))
        train_group = next(
            item["source_group"]
            for item in changed["datasets"]
            if item["role"] == "train_validation"
        )
        next(
            item for item in changed["datasets"] if item["role"] == "test_candidate"
        )["source_group"] = train_group
        with self.assertRaisesRegex(PreparationError, "cross train/test"):
            validate_cohort_manifest(changed)

    def test_frozen_assignment_digest_rejects_balanced_relabeling(self):
        manifest = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
        changed = json.loads(json.dumps(manifest))
        first = next(item for item in changed["datasets"] if item["id"] == "ETDataset-ett")
        second = next(item for item in changed["datasets"] if item["id"] == "weatherAUS")
        first["domain"], second["domain"] = second["domain"], first["domain"]
        with self.assertRaisesRegex(PreparationError, "differs from frozen v1"):
            validate_cohort_manifest(changed)

    def test_run_manifest_rejects_exact_hash_cross_role_leakage(self):
        cohort = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
        entries = {entry["id"]: entry for entry in cohort["datasets"]}

        def prepared(dataset_id: str) -> dict:
            entry = entries[dataset_id]
            return {
                "dataset_id": dataset_id,
                "domain": entry["domain"],
                "source_group": entry["source_group"],
                "cohort": {
                    "version": COHORT_VERSION,
                    "assignment_sha256": COHORT_ASSIGNMENT_SHA256,
                    "policy_sha256": COHORT_POLICY_SHA256,
                    "role": entry["role"],
                },
                "split_contract": {
                    "cohort_role": entry["role"],
                    "allowed_splits": (
                        ["train", "validation"]
                        if entry["role"] == "train_validation"
                        else ["test"]
                    ),
                    "partition": (
                        "per_entity_chronological_80_20_before_blocks"
                        if entry["role"] == "train_validation"
                        else "all_selected_rows_are_test_before_blocks"
                    ),
                    "train_boundary_rule": (
                        "floor(selected_rows * 4 / 5)"
                        if entry["role"] == "train_validation"
                        else None
                    ),
                },
                "blocks": [{
                    "split": (
                        "train" if entry["role"] == "train_validation" else "test"
                    )
                }],
                "source": {"sha256": "a" * 64},
            }

        with self.assertRaisesRegex(PreparationError, "exact source hash crosses"):
            validate_prepared_cohort_manifests(
                [prepared("ETDataset-ett"), prepared("enedis-with-holidays")],
                cohort,
                require_complete_train=False,
            )


class NumericAndTimestampContractTest(unittest.TestCase):
    def test_numeric_widths_and_nan_missing_distinction(self):
        missing = frozenset({"", "null"})
        self.assertEqual(pack_numeric_token("-128", "int8", missing), b"\x80")
        self.assertEqual(pack_numeric_token("65535", "uint16", missing), b"\xff\xff")
        self.assertEqual(pack_numeric_token("2147483647", "int32", missing), struct.pack("<i", 2147483647))
        self.assertEqual(pack_numeric_token("4294967295", "uint32", missing), struct.pack("<I", 4294967295))
        self.assertEqual(pack_numeric_token(str((1 << 63) - 1), "int64", missing), struct.pack("<q", (1 << 63) - 1))
        self.assertEqual(pack_numeric_token(str((1 << 64) - 1), "uint64", missing), struct.pack("<Q", (1 << 64) - 1))
        self.assertEqual(pack_numeric_token("bits16:0x7e01", "float16", missing), bytes.fromhex("017e"))
        self.assertEqual(pack_numeric_token("1.5", "float32", missing), struct.pack("<f", 1.5))
        self.assertEqual(pack_numeric_token("-0.0", "float64", missing), struct.pack("<d", -0.0))
        self.assertIsNone(pack_numeric_token("NULL", "float32", missing))
        nan32 = pack_numeric_token("bits32:0x7fc01234", "float32", missing)
        self.assertEqual(nan32, bytes.fromhex("3412c07f"))
        self.assertIsNotNone(pack_numeric_token("nan", "float64", missing))
        with self.assertRaises(PreparationError):
            pack_numeric_token("2147483648", "int32", missing)
        with self.assertRaises(PreparationError):
            pack_numeric_token("1.0", "int64", missing)
        with self.assertRaisesRegex(PreparationError, "outside float64"):
            pack_numeric_token("1e400", "float64", missing)

    def test_epoch_conversion_is_exact_and_no_row_fallback_exists(self):
        contract = TimestampContract(
            source_name="ts",
            output_name="timestamp_unix",
            source_kind="epoch",
            source_unit="ms",
            output_unit="us",
            timezone_name="UTC",
            formats=("iso8601",),
            ambiguous_fold=None,
            duplicate_policy="reject",
        )
        self.assertEqual(parse_timestamp_token("1234", contract), 1_234_000)
        with self.assertRaisesRegex(PreparationError, "row-index substitution is forbidden"):
            parse_timestamp_token("", contract)
        lossy = TimestampContract(
            source_name="ts",
            output_name="timestamp_unix",
            source_kind="epoch",
            source_unit="us",
            output_unit="ms",
            timezone_name="UTC",
            formats=("iso8601",),
            ambiguous_fold=None,
            duplicate_policy="reject",
        )
        with self.assertRaisesRegex(PreparationError, "cannot be represented exactly"):
            parse_timestamp_token("1234001", lossy)

    def test_datetime_timezone_and_fraction_are_explicit(self):
        contract = TimestampContract(
            source_name="ts",
            output_name="timestamp_unix",
            source_kind="datetime",
            source_unit=None,
            output_unit="us",
            timezone_name="Asia/Shanghai",
            formats=("iso8601",),
            ambiguous_fold=None,
            duplicate_policy="reject",
        )
        value = parse_timestamp_token("2026-09-17T08:00:00.123456", contract)
        expected = int(datetime(2026, 9, 17, 0, 0, 0, 123456, tzinfo=timezone.utc).timestamp() * 1_000_000)
        self.assertEqual(value, expected)
        second_contract = TimestampContract(
            **{**contract.__dict__, "output_unit": "s"}
        )
        with self.assertRaisesRegex(PreparationError, "cannot be represented exactly"):
            parse_timestamp_token("2026-09-17T08:00:00.123456", second_contract)
        utc_declared = TimestampContract(
            **{**contract.__dict__, "timezone_name": "UTC"}
        )
        with self.assertRaisesRegex(PreparationError, "does not match declared timezone"):
            parse_timestamp_token("2026-09-17T08:00:00+08:00", utc_declared)
        with self.assertRaisesRegex(PreparationError, "finer than microseconds"):
            parse_timestamp_token("2026-09-17T08:00:00.123456789", contract)

    def test_fair_prefix_allocation(self):
        counts = {("a",): 10, ("b",): 3, ("c",): 8}
        self.assertEqual(
            equal_prefix_allocation(counts, 12),
            {("a",): 5, ("b",): 3, ("c",): 4},
        )
        with self.assertRaisesRegex(PreparationError, "at least one row"):
            equal_prefix_allocation(counts, 2)

    def test_block_cap_includes_missing_bitmap(self):
        self.assertEqual(max_rows_per_block(8, 16), 1)
        self.assertEqual(max_rows_per_block(4, 16), 3)
        self.assertLessEqual(3 * 4 + (3 + 7) // 8, 16)

    def test_csv_source_audit_fingerprints_without_guessing_time_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = PreparationIntegrationTest._write_fixture(root)
            audited = audit_local_source(source)
            self.assertEqual(audited["format"], "csv")
            self.assertEqual(audited["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIsNone(audited["timestamp_unit"])
            self.assertIn("no_precision_guess", audited["timestamp_contract_status"])
            self.assertEqual(audited["columns"], ["entity", "ts", "i32", "i64", "f32", "f64"])

    def test_tsfile_audit_does_not_claim_materialization_or_timestamp_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.tsfile"
            path.write_bytes(b"not-a-tsfile")
            audited = audit_local_source(path)
            self.assertEqual(audited["format"], "tsfile")
            self.assertIsNone(audited["timestamp_unit"])
            self.assertEqual(
                audited["materialization_support"],
                "audit_only_not_materialized_by_csv_v1",
            )
            self.assertTrue(audited["schema_status"].startswith("blocked_") or
                            audited["schema_status"].startswith("no_table_"))


class PreparationIntegrationTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(
        root: Path,
        *,
        nonchronological: bool = False,
        bad_timestamp: bool = False,
        duplicate_timestamp: bool = False,
    ) -> Path:
        source = root / "fixture.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["entity", "ts", "i32", "i64", "f32", "f64"])
            for index in range(10):
                for entity, offset in (("alpha", 0), ("beta", 1000)):
                    timestamp: int | str = offset + index
                    if nonchronological and entity == "alpha" and index == 4:
                        timestamp = 1
                    if duplicate_timestamp and entity == "alpha" and index == 4:
                        timestamp = 3
                    if bad_timestamp and entity == "beta" and index == 2:
                        timestamp = "not-a-timestamp"
                    i32 = "" if entity == "alpha" and index == 1 else index
                    i64 = "null" if entity == "beta" and index == 2 else (1 << 53) + index
                    f32 = "bits32:0x7fc01234" if entity == "alpha" and index == 2 else index + 0.25
                    f64 = "-0.0" if entity == "beta" and index == 3 else index + 0.5
                    writer.writerow([entity, timestamp, i32, i64, f32, f64])
        return source

    @staticmethod
    def _write_spec(root: Path, source: Path, **overrides) -> Path:
        spec = {
            "format_version": FORMAT_VERSION,
            "dataset_id": "synthetic_fixture",
            "domain": "energy",
            "source_group": "synthetic_fixture_source",
            "source": {
                "path": source.name,
                "format": "csv",
                "encoding": "utf-8",
                "delimiter": ",",
                "upstream_revision": "synthetic-v1",
                "upstream_locator": "local:test-fixture",
                "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
            },
            "entity_columns": ["entity"],
            "timestamp": {
                "source_name": "ts",
                "output_name": "timestamp_unix",
                "source_kind": "epoch",
                "source_unit": "ms",
                "output_unit": "ms",
                "timezone": "UTC",
                "duplicate_policy": "reject"
            },
            "numeric_columns": [
                {"source_name": "i32", "dtype": "int32"},
                {"source_name": "i64", "dtype": "int64"},
                {"source_name": "f32", "dtype": "float32"},
                {"source_name": "f64", "dtype": "float64"}
            ],
            "missing_tokens": ["", "null", "none", "<na>"],
            "dataset_byte_cap": 320,
            "block_byte_cap": 16
        }
        spec.update(overrides)
        path = root / "spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path

    def test_prepare_preserves_types_missing_mask_split_blocks_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source)
            contract = load_preparation_contract(spec_path)
            output = root / "prepared"
            manifest = prepare_dataset(contract, output)

            # Five prefix rows per entity fit the 320-byte logical cap.  Four
            # are train and one is validation, all before block formation.
            self.assertEqual(manifest["selection"]["source_rows"], 20)
            self.assertEqual(manifest["selection"]["selected_rows"], 10)
            self.assertEqual(manifest["selection"]["selected_logical_slot_bytes"], 320)
            self.assertEqual(
                [(item["selected_rows"], item["train_rows"], item["validation_rows"])
                 for item in manifest["entities"]],
                [(5, 4, 1), (5, 4, 1)],
            )
            self.assertTrue(all(block["logical_slot_bytes"] <= 16 for block in manifest["blocks"]))
            self.assertTrue(all(block["prepared_block_bytes"] <= 16 for block in manifest["blocks"]))
            self.assertTrue(all(block["split"] in {"train", "validation"} for block in manifest["blocks"]))
            self.assertEqual(
                {column["dtype"] for column in manifest["input_contract"]["numeric_columns"]},
                {"int32", "int64", "float32", "float64"},
            )

            i32_missing = sum(
                block["missing_count"] for block in manifest["blocks"] if block["column"] == "i32"
            )
            i64_missing = sum(
                block["missing_count"] for block in manifest["blocks"] if block["column"] == "i64"
            )
            f32_missing = sum(
                block["missing_count"] for block in manifest["blocks"] if block["column"] == "f32"
            )
            self.assertEqual((i32_missing, i64_missing, f32_missing), (1, 1, 0))

            f32_payload = b"".join(
                (output / block["values_path"]).read_bytes()
                for block in manifest["blocks"]
                if block["column"] == "f32"
            )
            self.assertIn(bytes.fromhex("3412c07f"), f32_payload)
            self.assertTrue((output / "dataset_manifest.json").is_file())
            persisted = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["provenance_sha256"], manifest["provenance_sha256"])
            self.assertEqual(persisted["totals"]["original_numeric_bytes"], 308)
            self.assertEqual(persisted["metric_contract"]["denominator_bytes"], 308)
            self.assertEqual(
                persisted["metric_contract"]["compression_ratio"],
                "complete_compressed_bytes / B_raw",
            )
            verified = verify_prepared_dataset(output)
            self.assertEqual(verified["provenance_sha256"], manifest["provenance_sha256"])

            second = root / "prepared_again"
            repeated = prepare_dataset(contract, second)
            self.assertEqual(repeated["provenance_sha256"], manifest["provenance_sha256"])
            with self.assertRaisesRegex(PreparationError, "refusing to overwrite"):
                prepare_dataset(contract, output)

            first_values = output / manifest["blocks"][0]["values_path"]
            first_values.write_bytes(first_values.read_bytes() + b"\x00")
            with self.assertRaisesRegex(PreparationError, "file size mismatch"):
                verify_prepared_dataset(output)

    def test_test_candidate_materializes_all_selected_rows_as_test_without_five_row_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source, dataset_byte_cap=64)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec.update({
                "dataset_id": "enedis-with-holidays",
                "domain": "energy",
                "source_group": "hf:theforecastingcompany/enedis-with-holidays",
            })
            spec["source"]["upstream_locator"] = (
                "https://huggingface.co/datasets/THULab/enedis-with-holidays"
            )
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            cohort = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
            output = root / "prepared_test"
            manifest = prepare_dataset(
                load_preparation_contract(spec_path),
                output,
                cohort_manifest=cohort,
            )

            self.assertEqual(manifest["cohort"]["role"], "test_candidate")
            self.assertEqual(
                manifest["split_contract"]["allowed_splits"], ["test"]
            )
            self.assertEqual({block["split"] for block in manifest["blocks"]}, {"test"})
            self.assertEqual(
                {
                    (entity["selected_rows"], entity["train_rows"],
                     entity["validation_rows"], entity["test_rows"])
                    for entity in manifest["entities"]
                },
                {(1, 0, 0, 1)},
            )
            verified = verify_prepared_dataset(
                output, expected_role="test_candidate"
            )
            self.assertEqual(verified["cohort_role"], "test_candidate")

            manifest_path = output / "dataset_manifest.json"
            changed = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed["blocks"][0]["split"] = "train"
            manifest_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "invalid split"):
                verify_prepared_dataset(output, expected_role="test_candidate")

    def test_expected_role_rejects_role_swap_and_noncontiguous_block_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            output = root / "prepared"
            prepare_dataset(
                load_preparation_contract(self._write_spec(root, source)), output
            )
            with self.assertRaisesRegex(PreparationError, "explicit frozen cohort binding"):
                verify_prepared_dataset(output, expected_role="train_validation")

            manifest_path = output / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target = next(
                block for block in manifest["blocks"] if block["block_index"] == 0
            )
            target["block_index"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "block indexes are not contiguous"):
                verify_prepared_dataset(output)

    def test_invalid_or_nonchronological_timestamp_fails_without_output(self):
        for kwargs, pattern in (({"bad_timestamp": True}, "invalid integer epoch"),
                                ({"nonchronological": True}, "not chronological"),
                                ({"duplicate_timestamp": True}, "duplicate timestamp")):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self._write_fixture(root, **kwargs)
                contract = load_preparation_contract(self._write_spec(root, source))
                output = root / "prepared"
                with self.assertRaisesRegex(PreparationError, pattern):
                    prepare_dataset(contract, output)
                self.assertFalse(output.exists())

    def test_too_small_cap_does_not_silently_drop_an_entity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source, dataset_byte_cap=32)
            contract = load_preparation_contract(spec_path)
            with self.assertRaisesRegex(PreparationError, "at least one row"):
                prepare_dataset(contract, root / "prepared")

    def test_cap_that_selects_too_few_rows_rejects_invalid_80_20_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source, dataset_byte_cap=128)
            contract = load_preparation_contract(spec_path)
            with self.assertRaisesRegex(PreparationError, "at least five selected rows"):
                prepare_dataset(contract, root / "prepared")

    def test_epoch_timezone_must_be_utc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["timestamp"]["timezone"] = "Asia/Shanghai"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "must be UTC"):
                load_preparation_contract(spec_path)

    def test_expected_source_fingerprint_is_required_and_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["source"]["expected_sha256"] = "0" * 64
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            contract = load_preparation_contract(spec_path)
            with self.assertRaisesRegex(PreparationError, "does not match"):
                prepare_dataset(contract, root / "prepared")

    def test_cli_cohort_binding_rejects_synthetic_or_mismatched_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            contract = load_preparation_contract(self._write_spec(root, source))
            cohort = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)
            with self.assertRaisesRegex(PreparationError, "not a unique member"):
                validate_contract_against_cohort(contract, cohort)

    def test_bitmap_semantics_are_verified_not_only_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            contract = load_preparation_contract(self._write_spec(root, source))
            output = root / "prepared"
            prepare_dataset(contract, output)
            manifest_path = output / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            block = next(item for item in manifest["blocks"] if item["missing_count"] == 0)
            bitmap_path = output / block["missing_bitmap_path"]
            bitmap = bytearray(bitmap_path.read_bytes())
            bitmap[0] |= 0x80
            bitmap_path.write_bytes(bitmap)
            block["missing_bitmap_sha256"] = hashlib.sha256(bitmap).hexdigest()
            basis = {key: value for key, value in manifest.items() if key != "provenance_sha256"}
            canonical = json.dumps(
                basis,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            manifest["provenance_sha256"] = hashlib.sha256(canonical).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "bitmap missing count mismatch"):
                verify_prepared_dataset(output)

    def test_nan_cannot_be_configured_as_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            spec_path = self._write_spec(root, source, missing_tokens=["", "NaN"])
            with self.assertRaisesRegex(PreparationError, "NaN is a floating-point value"):
                load_preparation_contract(spec_path)


if __name__ == "__main__":
    unittest.main()
