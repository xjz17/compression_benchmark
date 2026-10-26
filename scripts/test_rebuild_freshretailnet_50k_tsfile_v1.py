#!/usr/bin/env python3
"""Focused tests for the canonical FreshRetailNet-50K TsFile rebuild."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_rebuild_freshretailnet_50k_v1 import (  # noqa: E402
    ENTITY_COLUMNS,
    _physical_schema,
)
from prepare_weighted_grpo_timeseries_v1 import (  # noqa: E402
    DEFAULT_COHORT_MANIFEST,
    load_cohort_manifest,
)
from prepare_weighted_grpo_tsfile_v1 import (  # noqa: E402
    audit_tsfile_spec,
    load_tsfile_spec,
)
from rebuild_freshretailnet_50k_tsfile_v1 import (  # noqa: E402
    DERIVED_SUBDIRECTORY,
    TABLE_NAME,
    VALUE_FIELDS,
    FreshRetailRebuildError,
    SplitBuildResult,
    build_formal_spec,
    rebuild_frozen_source,
    validate_frozen_source_audit,
    write_rebuilt_split,
    _native_row_bytes,
    _trim_reader_boundary,
)
from validate_weighted_grpo_formal_dataset_specs_v1 import (  # noqa: E402
    SpecGateError,
    validate_formal_spec,
)


class FreshRetailRebuildTest(unittest.TestCase):
    def test_reader_overlap_requires_all_value_bits_to_match(self):
        native = {
            **{f"entity:{name}": np.asarray([index, index], dtype="<i8")
               for index, name in enumerate(ENTITY_COLUMNS)},
            "timestamp_unix_ms": np.asarray([1000, 2000], dtype="<i8"),
            **{name: np.asarray([0, 1], dtype="<i4" if kind == "INT32" else "<f8")
               for name, kind in VALUE_FIELDS},
        }
        signature = (tuple(str(index) for index, _ in enumerate(ENTITY_COLUMNS)), 1000)
        original = _native_row_bytes(native, 0)
        trimmed = _trim_reader_boundary(native, signature, original)
        self.assertEqual(trimmed["timestamp_unix_ms"].tolist(), [2000])
        native["hours_sale_00"][0] = -0.0
        with self.assertRaisesRegex(FreshRetailRebuildError, "different value bits"):
            _trim_reader_boundary(native, signature, original)
        unchanged = _trim_reader_boundary(native, None, None)
        self.assertEqual(len(unchanged["timestamp_unix_ms"]), 2)

    @staticmethod
    def _rows(entity_offset: int, start_day: int, days: int, *, bad_status=False):
        rows = []
        # Source numeric order 2,10 is deliberately the reverse of the TsFile
        # STRING-tag scan order 10,2.  The reconstruction digest must therefore
        # compare per-entity chronology rather than global physical row order.
        for entity_index, identifier in enumerate((entity_offset + 1, entity_offset + 9)):
            for day in range(days):
                sale = [
                    (-0.0 if hour == 0 else 1.0000000000000002 + identifier + day + hour)
                    for hour in range(24)
                ]
                status = [int((hour + day) % 2) for hour in range(24)]
                if bad_status and entity_index == 0 and day == 0:
                    status[7] = 2
                rows.append({
                    **{name: identifier * 10 + index for index, name in enumerate(ENTITY_COLUMNS)},
                    "dt": str(pa.scalar(start_day + day, type=pa.date32()).as_py()),
                    "sale_amount": 3.0000000000000004 + day,
                    "hours_sale": sale,
                    "stock_hour6_22_cnt": 7 + day,
                    "hours_stock_status": status,
                    "discount": 0.125,
                    "holiday_flag": day % 2,
                    "activity_flag": (day + 1) % 2,
                    "precpt": 0.25 + day,
                    "avg_temperature": -0.0 if day == 0 else 20.5,
                    "avg_humidity": 80.25,
                    "avg_wind_level": 2.75,
                })
        return rows

    @classmethod
    def _write_parquet(cls, path: Path, rows) -> None:
        schema = _physical_schema()
        arrays = []
        for field in schema:
            arrays.append(pa.array([row[field.name] for row in rows], type=field.type))
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, path, row_group_size=3, compression="zstd")

    @staticmethod
    def _write_spec(path: Path, value) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def test_width_preserving_rebuild_and_formal_collection_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_root = root / "training_raw"
            derived = campaign_root / "freshretailnet_50k" / DERIVED_SUBDIRECTORY
            derived.mkdir(parents=True)
            train_source = root / "train.parquet"
            eval_source = root / "eval.parquet"
            train_rows = self._rows(1, 19_000, 4)
            eval_rows = self._rows(1, 19_004, 2)
            self._write_parquet(train_source, train_rows)
            self._write_parquet(eval_source, eval_rows)

            train_output = derived / "freshretailnet_50k_rebuilt_series_1.tsfile"
            eval_output = derived / "freshretailnet_50k_rebuilt_series_2.tsfile"
            train_result = write_rebuilt_split(
                train_source,
                train_output,
                source_split="train",
                expected_rows=len(train_rows),
            )
            eval_result = write_rebuilt_split(
                eval_source,
                eval_output,
                source_split="eval",
                expected_rows=len(eval_rows),
            )
            spec = build_formal_spec(campaign_root, [train_result, eval_result])
            spec_path = derived / "preparation_spec.json"
            self._write_spec(spec_path, spec)

            loaded = load_tsfile_spec(spec_path)
            self.assertEqual(len(loaded.value_columns), 57)
            self.assertEqual(
                sum(column.tsfile_type == "DOUBLE" for column in loaded.value_columns),
                30,
            )
            self.assertEqual(
                sum(column.tsfile_type == "INT32" for column in loaded.value_columns),
                27,
            )
            audit = audit_tsfile_spec(
                loaded,
                cohort_manifest=load_cohort_manifest(DEFAULT_COHORT_MANIFEST),
                row_limit=32,
            )
            self.assertEqual(audit["status"], "schema_and_bounded_rows_verified")
            self.assertEqual(audit["sampled_rows"], len(train_rows) + len(eval_rows))

            cohort_entry = next(
                item
                for item in load_cohort_manifest(DEFAULT_COHORT_MANIFEST)["datasets"]
                if item["id"] == "freshretailnet_50k"
            )
            checked = validate_formal_spec(
                spec,
                cohort_entry=cohort_entry,
                revision="fd0aa204f48c7d55e55ef3be3e9947a492f16d9c",
            )
            self.assertEqual(checked["source_files"], 2)
            self.assertEqual(checked["value_columns"], 57)
            self.assertEqual(checked["native_type_counts"], {"DOUBLE": 30, "INT32": 27})
            bad_layout_spec = build_formal_spec(root / "canonical", [train_result, eval_result])
            with self.assertRaisesRegex(SpecGateError, "expected suffix.*training_raw"):
                validate_formal_spec(
                    bad_layout_spec, cohort_entry=cohort_entry,
                    revision="fd0aa204f48c7d55e55ef3be3e9947a492f16d9c",
                )

            from tsfile import TsFileReader

            reader = TsFileReader(str(train_output))
            try:
                columns = reader.get_all_table_schemas()[TABLE_NAME].get_columns()
                by_name = {
                    column.get_column_name(): getattr(column.get_data_type(), "name", "")
                    for column in columns
                }
            finally:
                reader.close()
            self.assertEqual(by_name["hours_sale_00"], "DOUBLE")
            self.assertEqual(by_name["hours_stock_status_00"], "INT32")
            self.assertEqual(by_name["stock_hour6_22_cnt"], "INT32")

    def test_non_binary_stock_status_fails_closed_and_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.parquet"
            output = root / "bad.tsfile"
            rows = self._rows(1, 19_000, 1, bad_status=True)
            self._write_parquet(source, rows)
            with self.assertRaisesRegex(FreshRetailRebuildError, "0/1 INT32"):
                write_rebuilt_split(
                    source,
                    output,
                    source_split="train",
                    expected_rows=len(rows),
                )
            self.assertFalse(output.exists())

    def test_spec_rejects_fragment_renaming_or_order_substitution(self):
        fake = SplitBuildResult(
            source_split="eval",
            output_name="freshretailnet_50k_rebuilt_series_1.tsfile",
            rows=1,
            entities=1,
            output_bytes=1,
            output_sha256=hashlib.sha256(b"x").hexdigest(),
            semantic_sha256=hashlib.sha256(b"semantic").hexdigest(),
        )
        with self.assertRaisesRegex(FreshRetailRebuildError, "train then eval"):
            build_formal_spec(Path("/tmp/training_raw"), [fake, fake])

    def test_unreviewed_source_audit_fingerprint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            audit.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FreshRetailRebuildError, "fingerprint changed"):
                validate_frozen_source_audit(audit, {})

    def test_invalid_formal_layout_fails_before_expensive_source_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("rebuild_freshretailnet_50k_tsfile_v1.audit_and_rebuild") as audit:
                with self.assertRaisesRegex(FreshRetailRebuildError, "must end in /training_raw"):
                    rebuild_frozen_source(
                        root / "source", root / "canonical", root / "formal.json",
                        source_audit_path=root / "audit.json",
                    )
                audit.assert_not_called()

    def test_failed_admission_preserves_cold_verified_tsfiles_without_formal_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            (source_root / "data").mkdir(parents=True)
            self._write_parquet(source_root / "data/train.parquet", self._rows(1, 19000, 4))
            self._write_parquet(source_root / "data/eval.parquet", self._rows(1, 19004, 2))
            campaign_root = root / "canonical" / "training_raw"
            formal_spec = root / "formal.json"
            external_receipt = root / "external_receipt.json"

            def write_small(source, output, *, source_split, expected_rows):
                return write_rebuilt_split(
                    source, output, source_split=source_split,
                    expected_rows=pq.ParquetFile(source).metadata.num_rows,
                )

            with mock.patch("rebuild_freshretailnet_50k_tsfile_v1.audit_and_rebuild", return_value={}), \
                    mock.patch("rebuild_freshretailnet_50k_tsfile_v1.validate_frozen_source_audit", 
                               return_value={"upstream_repository_inventory": []}), \
                    mock.patch("rebuild_freshretailnet_50k_tsfile_v1.write_rebuilt_split", side_effect=write_small), \
                    mock.patch("prepare_weighted_grpo_tsfile_v1.audit_tsfile_spec",
                               side_effect=RuntimeError("injected loader admission failure")):
                with self.assertRaisesRegex(RuntimeError, "injected loader admission failure"):
                    rebuild_frozen_source(
                        source_root, campaign_root, formal_spec,
                        source_audit_path=root / "audit.json", receipt_output=external_receipt,
                    )

            derived = campaign_root / "freshretailnet_50k" / DERIVED_SUBDIRECTORY
            failure = json.loads((derived / "rebuild_failure_receipt.json").read_bytes())
            self.assertEqual(failure["failed_stage"], "production_collection_loader_audit")
            self.assertFalse(failure["formal_training_eligible"])
            self.assertFalse(failure["formal_spec_published"])
            self.assertEqual(len(failure["full_cold_verified_fragments"]), 2)
            for fragment in failure["full_cold_verified_fragments"]:
                payload = (derived / fragment["output_name"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), fragment["output_sha256"])
            receipt = json.loads((derived / "rebuild_receipt.json").read_bytes())
            self.assertEqual(receipt["status"], "full_cold_verified_but_admission_failed")
            self.assertFalse(formal_spec.exists())
            self.assertFalse(external_receipt.exists())


if __name__ == "__main__":
    unittest.main()
