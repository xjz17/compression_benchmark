#!/usr/bin/env python3
"""Focused tests for the exact-upstream deep-space canonical rebuild."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_weighted_grpo_timeseries_v1 import PreparationError  # noqa: E402
from rebuild_weighted_grpo_deep_space_v1 import (  # noqa: E402
    DeepSpaceSpec,
    SOURCE_BYTES,
    SOURCE_SHA256,
    SPEC_VERSION,
    TIMESTAMP_OUTPUT_NAME,
    UPSTREAM_REVISION,
    VALUE_COLUMNS,
    audit_source,
    formal_spec_document,
    load_spec,
    logical_spec_sha256,
    prepare_experiment_dataset,
    rebuild_canonical,
    validate_audit_receipt,
    verify_canonical,
    with_source_path,
)


class DeepSpaceRebuildTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path, *, timestamp_unit: str = "us") -> Path:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = root / "deep_space_fixture.parquet"
        entities = pa.array(
            ["probe_a"] * 5 + ["probe_b"] * 5,
            type=pa.large_string(),
        )
        timestamps = pa.array(
            [1_000_000 + index * 3_600_000_000 for index in range(5)]
            + [2_000_000 + index * 3_600_000_000 for index in range(5)],
            type=pa.timestamp(timestamp_unit),
        )
        arrays = [entities, timestamps]
        names = ["spacecraft", "datetime"]
        for column_index, name in enumerate(VALUE_COLUMNS):
            bits = np.array(
                [
                    0x8000000000000000,
                    0x7FF8000000001234,
                    0x3FF0000000000000 + column_index,
                    0x4000000000000000 + column_index,
                    0,
                    0x4010000000000000 + column_index,
                    0x4020000000000000 + column_index,
                    0x4030000000000000 + column_index,
                    0x4040000000000000 + column_index,
                    0,
                ],
                dtype=np.uint64,
            )
            # Rows four and nine are timestamp-only in every numeric column.
            mask = np.array([False, False, False, False, True, False, False, False, False, True])
            arrays.append(pa.array(bits.view(np.float64), mask=mask, type=pa.float64()))
            names.append(name)
        table = pa.Table.from_arrays(arrays, names=names)
        pq.write_table(
            table,
            path,
            compression="NONE",
            use_dictionary=False,
            row_group_size=6,
            version="2.6",
        )
        return path

    @staticmethod
    def _contract(path: Path, *, block_byte_cap: int = 41) -> DeepSpaceSpec:
        return DeepSpaceSpec(
            path=path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_bytes=path.stat().st_size,
            block_byte_cap=block_byte_cap,
            spec_sha256="f" * 64,
        )

    def test_frozen_formal_spec_has_exact_revision_and_fingerprint(self):
        path = (
            SCRIPTS.parent
            / "analysis_outputs/weight_conditioned_grpo_20260917_v1/specs/"
            "deep_space_probes_upstream_parquet_v1.json"
        )
        contract = load_spec(path)
        self.assertEqual(contract.expected_sha256, SOURCE_SHA256)
        self.assertEqual(contract.expected_bytes, SOURCE_BYTES)
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw, formal_spec_document(Path(raw["source"]["path"])))
        self.assertEqual(raw["format_version"], SPEC_VERSION)
        self.assertEqual(raw["source"]["upstream_revision"], UPSTREAM_REVISION)
        self.assertEqual(raw["timestamp"]["output_name"], TIMESTAMP_OUTPUT_NAME)
        self.assertIn("without scaling", raw["timestamp"]["conversion"])
        overridden = with_source_path(contract, Path("/tmp/exact-source.parquet"))
        self.assertEqual(overridden.expected_sha256, SOURCE_SHA256)
        self.assertEqual(overridden.path, Path("/tmp/exact-source.parquet").resolve())
        other_machine = formal_spec_document(
            Path("/srv/formal/exact/deep_space_probes.parquet")
        )
        self.assertEqual(contract.spec_sha256, logical_spec_sha256(other_machine))

        audit_path = (
            SCRIPTS.parent
            / "analysis_outputs/weight_conditioned_grpo_20260917_v1/data_audit/"
            "deep_space_upstream_parquet_audit_v1.json"
        )
        evidence = validate_audit_receipt(contract, audit_path)
        self.assertEqual(evidence["rows"], 1_183_368)
        self.assertEqual(evidence["timestamp_only_rows"], 38_167)

        with tempfile.TemporaryDirectory() as directory:
            changed = json.loads(audit_path.read_text(encoding="utf-8"))
            changed["timestamp_only_rows"] -= 1
            tampered = Path(directory) / "audit.json"
            tampered.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "timestamp_only_rows"):
                validate_audit_receipt(contract, tampered)

    def test_rebuild_retains_timestamp_only_rows_null_masks_and_float_bits(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            contract = self._contract(source)
            audit = audit_source(contract, enforce_production_shape=False)
            self.assertEqual(audit["parquet"]["rows"], 10)
            self.assertEqual(audit["timestamp_only_rows"], 2)
            self.assertTrue(audit["row_preservation"]["timestamp_only_rows_are_retained"])
            output = root / "canonical"
            manifest = rebuild_canonical(
                contract, output, enforce_production_shape=False
            )
            verified = verify_canonical(output)
            self.assertEqual(verified["rows"], 10)
            self.assertEqual(verified["timestamp_only_rows"], 2)
            self.assertEqual(manifest["row_contract"]["selection"], "none_all_source_rows_retained")

            first_column_blocks = sorted(
                (
                    block
                    for block in manifest["blocks"]
                    if block["entity_value"] == "probe_a"
                    and block["column"] == VALUE_COLUMNS[0]
                ),
                key=lambda block: block["block_index"],
            )
            payload = b"".join(
                (output / block["values_path"]).read_bytes()
                for block in first_column_blocks
            )
            bits = np.frombuffer(payload, dtype="<u8").tolist()
            self.assertEqual(bits[0], 0x8000000000000000)  # negative zero
            self.assertEqual(bits[1], 0x7FF8000000001234)  # NaN payload

            timestamp_rows = sum(
                block["row_count"]
                for block in manifest["blocks"]
                if block["column"] == TIMESTAMP_OUTPUT_NAME
            )
            self.assertEqual(timestamp_rows, 10)
            self.assertEqual(
                manifest["totals"]["logical_slot_bytes"],
                10 * (len(VALUE_COLUMNS) + 1) * 8,
            )

            prepared_output = root / "prepared"
            prepared = prepare_experiment_dataset(
                output,
                prepared_output,
                dataset_byte_cap=10 * (len(VALUE_COLUMNS) + 1) * 8,
                block_byte_cap=41,
            )
            self.assertEqual(prepared["selection"]["source_rows"], 10)
            self.assertEqual(prepared["selection"]["selected_rows"], 10)
            self.assertEqual(
                [(item["train_rows"], item["validation_rows"])
                 for item in prepared["entities"]],
                [(4, 1), (4, 1)],
            )
            self.assertTrue(
                prepared["canonical_rebuild"]["selection_occurs_after_full_row_rebuild"]
            )

    def test_wrong_timestamp_precision_and_source_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            milliseconds = self._write_fixture(root, timestamp_unit="ms")
            with self.assertRaisesRegex(PreparationError, "timestamp\[us\]"):
                audit_source(
                    self._contract(milliseconds), enforce_production_shape=False
                )

            microseconds = self._write_fixture(root, timestamp_unit="us")
            bad = DeepSpaceSpec(
                **{
                    **self._contract(microseconds).__dict__,
                    "expected_sha256": "0" * 64,
                }
            )
            with self.assertRaisesRegex(PreparationError, "SHA-256 mismatch"):
                audit_source(bad, enforce_production_shape=False)

    def test_verifier_rejects_bitmap_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_fixture(root)
            output = root / "canonical"
            manifest = rebuild_canonical(
                self._contract(source), output, enforce_production_shape=False
            )
            target = output / manifest["blocks"][0]["missing_bitmap_path"]
            payload = bytearray(target.read_bytes())
            payload[0] ^= 1
            target.write_bytes(payload)
            with self.assertRaisesRegex(PreparationError, "block hash changed"):
                verify_canonical(output)


if __name__ == "__main__":
    unittest.main()
