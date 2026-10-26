#!/usr/bin/env python3
"""Focused tests for schema-driven TsFile materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_weighted_grpo_timeseries_v1 import (  # noqa: E402
    DEFAULT_COHORT_MANIFEST,
    PreparationError,
    load_cohort_manifest,
    verify_prepared_dataset,
)
from prepare_weighted_grpo_tsfile_v1 import (  # noqa: E402
    audit_tsfile_spec,
    inspect_tsfile_source,
    load_tsfile_spec,
    prepare_tsfile_dataset,
)
import prepare_weighted_grpo_tsfile_v1 as tsfile_preparer  # noqa: E402


try:  # The production extractor emits a precise blocker when this is absent.
    import tsfile
except Exception:  # pragma: no cover - depends on the test environment
    tsfile = None


class _FakeColumn:
    def __init__(self, name: str, data_type: str, category: str) -> None:
        self._name = name
        self._data_type = data_type
        self._category = category

    def get_column_name(self):
        return self._name

    def get_data_type(self):
        return self._data_type

    def get_category(self):
        return self._category


class _FakeSchema:
    def __init__(self) -> None:
        self._columns = [
            _FakeColumn("country", "STRING", "TAG"),
            _FakeColumn("sitenumber", "INT64", "TAG"),
            _FakeColumn("i32", "INT32", "FIELD"),
            _FakeColumn("i64", "INT64", "FIELD"),
            _FakeColumn("f32", "FLOAT", "FIELD"),
            _FakeColumn("f64", "DOUBLE", "FIELD"),
        ]

    def get_columns(self):
        return self._columns


class _FakeMetadata:
    def get_column_list(self):
        return ["Time", "country", "sitenumber", "i32", "i64", "f32", "f64"]

    def get_data_type_list(self):
        return ["INT64", "STRING", "INT64", "INT32", "INT64", "FLOAT", "DOUBLE"]


class _FakeResult:
    def __init__(self, rows) -> None:
        self._rows = rows
        self._index = -1

    def get_metadata(self):
        return _FakeMetadata()

    def next(self):
        self._index += 1
        return self._index < len(self._rows)

    def get_value_by_index(self, index):
        return self._rows[self._index][index - 1]

    def is_null_by_index(self, index):
        return self.get_value_by_index(index) is None

    def close(self):
        return None


class _FakeReader:
    rows_by_path = {}

    def __init__(self, path: str) -> None:
        self.path = path

    def get_all_table_schemas(self):
        return {"eea_pm25_forecasting_series": _FakeSchema()}

    def query_table(self, table_name, columns, start, end):
        if table_name != "eea_pm25_forecasting_series":
            raise AssertionError(table_name)
        expected = ["country", "sitenumber", "i32", "i64", "f32", "f64"]
        if list(columns) != expected:
            raise AssertionError(columns)
        return _FakeResult(self.rows_by_path[self.path])

    def close(self):
        return None


class _FakeTsFileModule:
    TsFileReader = _FakeReader
    __version__ = "fake-collection-reader-v1"


class TsFileCollectionPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)

    @staticmethod
    def _write_container(path: Path, marker: bytes) -> None:
        path.write_bytes(b"TsFile" + marker + b"TsFile")

    @classmethod
    def _write_spec(cls, root: Path, rows_by_file, *, reverse_files=False) -> Path:
        files = []
        _FakeReader.rows_by_path = {}
        for index, rows in enumerate(rows_by_file, start=1):
            source = root / f"eea_pm25_forecasting_series_{index}.tsfile"
            cls._write_container(source, f"source-{index}".encode("ascii"))
            _FakeReader.rows_by_path[str(source)] = rows
            files.append({
                "repo_path": source.name,
                "path": str(source),
                "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
        if reverse_files:
            files.reverse()
        raw = {
            "format_version": "weighted-grpo-tsfile-collection-preparation-v1",
            "dataset_id": "eea_pm25_forecasting",
            "domain": "environment",
            "source_group": "hf:cosuleabianca/eea-pm25-forecasting",
            "source": {
                "format": "tsfile_collection",
                "upstream_revision": "unit-test-exact-revision",
                "upstream_locator": "https://huggingface.co/datasets/BenchmarkHost/eea_pm25_forecasting",
                "schema_policy": "identical_across_all_files",
                "files": files,
            },
            "model": "table",
            "timestamp": {
                "source_unit": "ms",
                "output_unit": "ms",
                "timezone": "UTC",
                "output_name": "timestamp_unix_ms",
                "duplicate_policy": "reject",
            },
            "query_range": {
                "start_inclusive": -(1 << 63),
                "end_inclusive": (1 << 63) - 1,
            },
            "row_contract": {
                "ordering_key": "Time",
                "emit_ordering_key_as_numeric_column": True,
                "ordering_key_output_dtype": "INT64",
                "sort_scope": "global_entity_timestamp_ascending",
                "duplicate_comparison_scope": "all_listed_source_files",
                "cross_file_entity_merge": True,
                "source_file_order": "numeric_series_suffix_ascending",
                "split_after_global_merge": "per_entity_chronological_80_20",
            },
            "table": {
                "table_name": "eea_pm25_forecasting_series",
                "entity_columns": ["country", "sitenumber"],
                "value_columns": [
                    {"source_name": "i32", "tsfile_type": "INT32"},
                    {"source_name": "i64", "tsfile_type": "INT64"},
                    {"source_name": "f32", "tsfile_type": "FLOAT"},
                    {"source_name": "f64", "tsfile_type": "DOUBLE"},
                ],
            },
            "dataset_byte_cap": 4096,
            "block_byte_cap": 256,
        }
        spec = root / "collection_spec.json"
        spec.write_text(json.dumps(raw), encoding="utf-8")
        return spec

    @staticmethod
    def _row(timestamp: int, value: int):
        return (timestamp, "DE", 7, value, (1 << 40) + value, value + 0.25, value + 0.5)

    @staticmethod
    def _timestamp_values(output: Path, manifest, split: str):
        blocks = sorted(
            (
                block for block in manifest["blocks"]
                if block["split"] == split
                and block["column"] == "timestamp_unix_ms"
            ),
            key=lambda block: block["block_index"],
        )
        return [
            value[0]
            for block in blocks
            for value in struct.iter_unpack(
                "<q", (output / block["values_path"]).read_bytes()
            )
        ]

    def test_global_merge_sorts_before_split_and_preserves_native_widths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self._write_spec(root, [
                [self._row(30, 3), self._row(10, 1), self._row(50, 5)],
                [self._row(20, 2), self._row(60, 6), self._row(40, 4)],
            ])
            spec = load_tsfile_spec(spec_path)
            output = root / "prepared"
            with mock.patch.object(
                tsfile_preparer, "_load_tsfile_module", return_value=_FakeTsFileModule
            ):
                manifest = prepare_tsfile_dataset(
                    spec, output, cohort_manifest=self.cohort
                )
                verified = verify_prepared_dataset(output)
            self.assertTrue(verified["source_verified"])
            self.assertEqual(
                [column["source_name"] for column in manifest["input_contract"]["numeric_columns"]],
                ["Time", "i32", "i64", "f32", "f64"],
            )
            self.assertEqual(
                [column["dtype"] for column in manifest["input_contract"]["numeric_columns"]],
                ["int64", "int32", "int64", "float32", "float64"],
            )
            self.assertEqual(self._timestamp_values(output, manifest, "train"), [10, 20, 30, 40])
            self.assertEqual(self._timestamp_values(output, manifest, "validation"), [50, 60])
            self.assertEqual(manifest["source"]["file_count"], 2)

    def test_duplicate_entity_time_across_files_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_tsfile_spec(self._write_spec(root, [
                [self._row(10, 1), self._row(20, 2), self._row(30, 3)],
                [self._row(10, 4), self._row(40, 5), self._row(50, 6)],
            ]))
            with mock.patch.object(
                tsfile_preparer, "_load_tsfile_module", return_value=_FakeTsFileModule
            ):
                with self.assertRaisesRegex(PreparationError, "duplicate \\(entity, Time\\)"):
                    prepare_tsfile_dataset(
                        spec, root / "prepared", cohort_manifest=self.cohort
                    )

    def test_frozen_hash_rejects_file_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self._write_spec(root, [
                [self._row(10, 1), self._row(20, 2), self._row(30, 3)],
                [self._row(40, 4), self._row(50, 5), self._row(60, 6)],
            ])
            raw = json.loads(spec_path.read_text(encoding="utf-8"))
            first_hash = raw["source"]["files"][0]["expected_sha256"]
            raw["source"]["files"][0]["expected_sha256"] = raw["source"]["files"][1]["expected_sha256"]
            raw["source"]["files"][1]["expected_sha256"] = first_hash
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            spec = load_tsfile_spec(spec_path)
            with mock.patch.object(
                tsfile_preparer, "_load_tsfile_module", return_value=_FakeTsFileModule
            ):
                with self.assertRaisesRegex(PreparationError, "does not match the frozen spec"):
                    prepare_tsfile_dataset(
                        spec, root / "prepared", cohort_manifest=self.cohort
                    )

    def test_collection_file_order_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = self._write_spec(root, [
                [self._row(10, 1)],
                [self._row(20, 2)],
            ], reverse_files=True)
            with self.assertRaisesRegex(PreparationError, "ordered by unique numeric series suffix"):
                load_tsfile_spec(spec_path)


@unittest.skipUnless(tsfile is not None, "Apache tsfile Python package unavailable")
class TsFilePreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = load_cohort_manifest(DEFAULT_COHORT_MANIFEST)

    @staticmethod
    def _write_table_fixture(root: Path) -> Path:
        from tsfile import (
            ColumnCategory,
            ColumnSchema,
            TableSchema,
            Tablet,
            TSDataType,
            TsFileTableWriter,
        )

        path = root / "table_fixture.tsfile"
        schema = TableSchema("metrics", [
            ColumnSchema("entity", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("i32", TSDataType.INT32, ColumnCategory.FIELD),
            ColumnSchema("i64", TSDataType.INT64, ColumnCategory.FIELD),
            ColumnSchema("f32", TSDataType.FLOAT, ColumnCategory.FIELD),
            ColumnSchema("f64", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ])
        tablet = Tablet(
            ["entity", "i32", "i64", "f32", "f64"],
            [
                TSDataType.STRING,
                TSDataType.INT32,
                TSDataType.INT64,
                TSDataType.FLOAT,
                TSDataType.DOUBLE,
            ],
            20,
        )
        tablet.set_table_name("metrics")
        nan32 = struct.unpack("<f", bytes.fromhex("3412c07f"))[0]
        nan64 = struct.unpack("<d", bytes.fromhex("341200000000f87f"))[0]
        for row in range(20):
            entity_row = row % 10
            tablet.add_timestamp(row, 100 + entity_row)
            tablet.add_value_by_name("entity", row, "a" if row < 10 else "b")
            tablet.add_value_by_name("i32", row, row)
            tablet.add_value_by_name("i64", row, (1 << 40) + row)
            if entity_row != 3:
                value32 = nan32 if entity_row == 2 else (-0.0 if entity_row == 1 else row + 0.25)
                tablet.add_value_by_name("f32", row, value32)
            if entity_row != 4:
                value64 = nan64 if entity_row == 2 else (-0.0 if entity_row == 1 else row + 0.5)
                tablet.add_value_by_name("f64", row, value64)
        writer = TsFileTableWriter(str(path), schema)
        writer.write_table(tablet)
        writer.close()
        return path

    @staticmethod
    def _write_tree_fixture(root: Path) -> Path:
        from tsfile import Field, RowRecord, TSDataType, TimeseriesSchema, TsFileWriter

        path = root / "tree_fixture.tsfile"
        writer = TsFileWriter(str(path))
        columns = (
            ("i32", TSDataType.INT32),
            ("i64", TSDataType.INT64),
            ("f32", TSDataType.FLOAT),
        )
        for device in ("d1", "d2"):
            for name, data_type in columns:
                writer.register_timeseries(device, TimeseriesSchema(name, data_type))
        for device in ("d1", "d2"):
            for row in range(10):
                fields = [
                    Field("i32", row, TSDataType.INT32),
                    Field("i64", (1 << 40) + row, TSDataType.INT64),
                    Field("f32", row + 0.25, TSDataType.FLOAT),
                ]
                writer.write_row_record(RowRecord(device, 100 + row, fields))
        writer.close()
        return path

    @staticmethod
    def _write_spec(root: Path, source: Path, *, model: str) -> Path:
        common = {
            "format_version": "weighted-grpo-tsfile-preparation-v1",
            "dataset_id": "ETDataset-ett",
            "domain": "energy",
            "source_group": "github:zhouhaoyi/ETDataset",
            "source": {
                "path": str(source),
                "format": "tsfile",
                "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "upstream_revision": "unit-test-exact-revision",
                "upstream_locator": "https://huggingface.co/datasets/BenchmarkHost/ETDataset-ett",
            },
            "model": model,
            "timestamp": {
                "source_unit": "ms",
                "output_unit": "us",
                "timezone": "UTC",
                "output_name": "timestamp_unix_us",
                "duplicate_policy": "reject",
            },
            "query_range": {
                "start_inclusive": -(1 << 63),
                "end_inclusive": (1 << 63) - 1,
            },
            "dataset_byte_cap": 4096,
            "block_byte_cap": 25,
        }
        if model == "table":
            common["table"] = {
                "table_name": "metrics",
                "entity_columns": ["entity"],
                "value_columns": [
                    {"source_name": "i32", "tsfile_type": "INT32"},
                    {"source_name": "i64", "tsfile_type": "INT64"},
                    {"source_name": "f32", "tsfile_type": "FLOAT"},
                    {"source_name": "f64", "tsfile_type": "DOUBLE"},
                ],
            }
        else:
            common["tree"] = {
                "devices": ["d1", "d2"],
                "value_columns": [
                    {"source_name": "i32", "tsfile_type": "INT32"},
                    {"source_name": "i64", "tsfile_type": "INT64"},
                    {"source_name": "f32", "tsfile_type": "FLOAT"},
                ],
            }
        path = root / f"{model}_spec.json"
        path.write_text(json.dumps(common), encoding="utf-8")
        return path

    def test_table_materialization_preserves_widths_bits_nulls_and_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_table_fixture(root)
            spec = load_tsfile_spec(self._write_spec(root, source, model="table"))
            audit = audit_tsfile_spec(spec, cohort_manifest=self.cohort, row_limit=3)
            self.assertEqual(audit["status"], "schema_and_bounded_rows_verified")
            self.assertFalse(audit["timestamp"]["precision_inferred"])

            output = root / "prepared"
            manifest = prepare_tsfile_dataset(spec, output, cohort_manifest=self.cohort)
            verified = verify_prepared_dataset(output)
            self.assertEqual(verified["dataset_id"], "ETDataset-ett")
            self.assertTrue(verified["source_verified"])
            self.assertEqual(
                [column["dtype"] for column in manifest["input_contract"]["numeric_columns"]],
                ["int64", "int32", "int64", "float32", "float64"],
            )
            self.assertEqual(
                {(entity["train_rows"], entity["validation_rows"]) for entity in manifest["entities"]},
                {(8, 2)},
            )
            self.assertTrue(all(
                block["prepared_block_bytes"] <= 25 for block in manifest["blocks"]
            ))

            entity_a = next(
                entity for entity in manifest["entities"]
                if entity["entity_values"] == ['STRING:"a"']
            )
            f32_first = next(
                block for block in manifest["blocks"]
                if block["entity_id"] == entity_a["entity_id"]
                and block["split"] == "train"
                and block["column"] == "f32"
                and block["block_index"] == 0
            )
            f32_values = (output / f32_first["values_path"]).read_bytes()
            f32_bitmap = (output / f32_first["missing_bitmap_path"]).read_bytes()
            self.assertEqual(f32_values[4:8], bytes.fromhex("00000080"))
            self.assertEqual(f32_values[8:12], bytes.fromhex("3412c07f"))
            self.assertEqual(f32_bitmap, b"\x08")

            f64_first = next(
                block for block in manifest["blocks"]
                if block["entity_id"] == entity_a["entity_id"]
                and block["split"] == "train"
                and block["column"] == "f64"
                and block["block_index"] == 0
            )
            f64_values = (output / f64_first["values_path"]).read_bytes()
            self.assertEqual(f64_values[8:16], bytes.fromhex("0000000000000080"))
            self.assertEqual(f64_values[16:24], bytes.fromhex("341200000000f87f"))
            self.assertEqual(manifest["metric_contract"]["denominator_name"], "B_raw")
            self.assertLessEqual(
                manifest["selection"]["selected_logical_slot_bytes"],
                manifest["selection"]["dataset_byte_cap"],
            )

    def test_tree_materialization_uses_explicit_devices_as_entities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_tree_fixture(root)
            spec = load_tsfile_spec(self._write_spec(root, source, model="tree"))
            audit = audit_tsfile_spec(spec, cohort_manifest=self.cohort, row_limit=2)
            self.assertEqual(audit["sampled_rows"], 4)
            self.assertEqual(audit["sampled_entities"], 2)

            output = root / "prepared_tree"
            manifest = prepare_tsfile_dataset(spec, output, cohort_manifest=self.cohort)
            self.assertEqual(len(manifest["entities"]), 2)
            self.assertEqual(
                {tuple(entity["entity_values"]) for entity in manifest["entities"]},
                {('DEVICE:"d1"',), ('DEVICE:"d2"',)},
            )
            verified = verify_prepared_dataset(output)
            self.assertEqual(verified["dataset_id"], "ETDataset-ett")
            self.assertTrue(verified["source_verified"])

    def test_real_tsfile_test_candidate_materializes_and_verifies_only_test_split(self):
        """No mocked reader: exercise the Apache writer, reader, materializer, and verifier."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_table_fixture(root)
            spec_path = self._write_spec(root, source, model="table")
            raw = json.loads(spec_path.read_text(encoding="utf-8"))
            raw.update({
                "dataset_id": "sleep_score_fitbit",
                "domain": "health",
                "source_group": "hf:aai530-group6/sleep-score-fitbit",
                "dataset_byte_cap": 64,
            })
            raw["source"]["upstream_locator"] = (
                "https://huggingface.co/datasets/BenchmarkHost/sleep_score_fitbit"
            )
            spec_path.write_text(json.dumps(raw), encoding="utf-8")

            output = root / "prepared_test"
            manifest = prepare_tsfile_dataset(
                load_tsfile_spec(spec_path),
                output,
                cohort_manifest=self.cohort,
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

    def test_model_mismatch_and_timestamp_ambiguity_are_precise_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table_source = self._write_table_fixture(root)
            inspected = inspect_tsfile_source(table_source)
            self.assertEqual(inspected["model"], "table")
            self.assertIsNone(inspected["timestamp_unit"])
            self.assertFalse(inspected["precision_inferred"])
            self.assertEqual(inspected["table_schemas"][0]["table_name"], "metrics")
            tree_spec_path = self._write_spec(root, table_source, model="tree")
            tree_spec = load_tsfile_spec(tree_spec_path)
            with self.assertRaisesRegex(PreparationError, "TSFILE_MODEL_BLOCKER"):
                audit_tsfile_spec(tree_spec, cohort_manifest=self.cohort)

            raw = json.loads(tree_spec_path.read_text(encoding="utf-8"))
            raw["timestamp"]["timezone"] = "Asia/Shanghai"
            tree_spec_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "requires UTC"):
                load_tsfile_spec(tree_spec_path)

    def test_audited_civil_coordinate_is_preserved_without_utc_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_tree_fixture(root)
            spec_path = self._write_spec(root, source, model="tree")
            raw = json.loads(spec_path.read_text(encoding="utf-8"))
            raw["timestamp"].update({
                "output_unit": "ms",
                "timezone": "Europe/Helsinki source civil-time labels",
                "semantic_kind": "helsinki_civil_time_coordinate_encoding",
                "source_event_timezone_inferred": False,
                "event_utc_instant_claim": False,
                "event_time_mapping_is_injective": False,
                "output_name": "timestamp_unix_ms_civil_label",
            })
            spec_path.write_text(json.dumps(raw), encoding="utf-8")

            spec = load_tsfile_spec(spec_path)
            self.assertEqual(spec.timestamp.source_kind, "civil_time_coordinate")
            self.assertEqual(
                spec.timestamp_semantic_kind,
                "helsinki_civil_time_coordinate_encoding",
            )
            self.assertEqual(
                dict(spec.timestamp_semantic_flags),
                {
                    "source_event_timezone_inferred": False,
                    "event_utc_instant_claim": False,
                    "event_time_mapping_is_injective": False,
                },
            )

            raw["timestamp"].pop("event_utc_instant_claim")
            spec_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(PreparationError, "event_utc_instant_claim=false"):
                load_tsfile_spec(spec_path)

    def test_invalid_container_is_rejected_before_native_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.tsfile"
            source.write_bytes(b"not-a-tsfile-container")
            spec = load_tsfile_spec(self._write_spec(root, source, model="tree"))
            with self.assertRaisesRegex(PreparationError, "TSFILE_CONTAINER_BLOCKER"):
                audit_tsfile_spec(spec, cohort_manifest=self.cohort)


if __name__ == "__main__":
    unittest.main()
