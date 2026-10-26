#!/usr/bin/env python3
"""Tests for the project179 THULab numeric-only materializer."""

from __future__ import annotations

import copy
import hashlib
import gzip
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

from prepare_weighted_grpo_timeseries_v1 import PreparationError  # noqa: E402
import prepare_project179_thulab_numeric_v1 as materializer  # noqa: E402


REVISION_A = "a" * 40
REVISION_B = "b" * 40


class _FakeColumn:
    def __init__(self, name: str, data_type: str, category: str) -> None:
        self.name = name
        self.data_type = data_type
        self.category = category

    def get_column_name(self):
        return self.name

    def get_data_type(self):
        return self.data_type

    def get_category(self):
        return self.category


class _FakeSchema:
    COLUMNS = (
        ("country", "STRING", "TAG"),
        ("site", "INT64", "TAG"),
        ("i32", "INT32", "FIELD"),
        ("i64", "INT64", "FIELD"),
        ("f32", "FLOAT", "FIELD"),
        ("f64", "DOUBLE", "FIELD"),
        ("description", "TEXT", "FIELD"),
        ("enabled", "BOOLEAN", "FIELD"),
        ("calendar_day", "DATE", "FIELD"),
        ("event_time", "STRING", "FIELD"),
    )

    def __init__(self, columns=None) -> None:
        self.columns = tuple(columns or self.COLUMNS)

    def get_columns(self):
        return [_FakeColumn(*column) for column in self.columns]


class _FakeMetadata:
    def __init__(self, columns, schema_columns) -> None:
        types = {name: dtype for name, dtype, _category in schema_columns}
        self.names = ["Time", *columns]
        self.types = ["INT64", *(types[name] for name in columns)]

    def get_column_list(self):
        return self.names

    def get_data_type_list(self):
        return self.types


class _FakeResult:
    def __init__(self, rows, columns, schema_columns) -> None:
        self.rows = rows
        self.columns = list(columns)
        self.index = -1
        self.closed = False
        self.schema_columns = schema_columns

    def get_metadata(self):
        return _FakeMetadata(self.columns, self.schema_columns)

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def get_value_by_index(self, index):
        name = "Time" if index == 1 else self.columns[index - 2]
        return self.rows[self.index][name]

    def is_null_by_index(self, index):
        return self.get_value_by_index(index) is None

    def close(self):
        self.closed = True


class _FakeReader:
    rows_by_path = {}
    table_by_path = {}
    schema_by_path = {}
    query_count_by_path = {}

    def __init__(self, path: str) -> None:
        self.path = path

    def get_all_table_schemas(self):
        return {
            self.table_by_path.get(self.path, "metrics"):
            _FakeSchema(self.schema_by_path.get(self.path))
        }

    def query_table(self, table_name, columns, start, end):
        if table_name != self.table_by_path.get(self.path, "metrics"):
            raise AssertionError(table_name)
        if start != -(1 << 63) or end != (1 << 63) - 1:
            raise AssertionError((start, end))
        schema_columns = self.schema_by_path.get(self.path, _FakeSchema.COLUMNS)
        allowed = {name for name, _dtype, _category in schema_columns}
        if any(column not in allowed for column in columns):
            raise AssertionError(columns)
        self.query_count_by_path[self.path] = (
            self.query_count_by_path.get(self.path, 0) + 1
        )
        return _FakeResult(self.rows_by_path[self.path], columns, schema_columns)

    def close(self):
        return None


class _FakeTsFileModule:
    TsFileReader = _FakeReader


class _SyntheticRows:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return {"Time": index + 1, "entity": f"e-{index:06d}", "value": index}


class Project179NumericMaterializerTest(unittest.TestCase):
    @staticmethod
    def _sharded_records(dataset_dir: Path):
        manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
        with gzip.open(dataset_dir / manifest["stream_index"]["path"], "rt", encoding="utf-8") as handle:
            return manifest, [json.loads(line) for line in handle]

    @staticmethod
    def _rewrite_sharded_records(output: Path, dataset_id: str, manifest, records):
        dataset_dir = output / dataset_id
        index_path = dataset_dir / manifest["stream_index"]["path"]
        raw = b"".join(materializer._canonical_json(record) + b"\n" for record in records)
        with index_path.open("wb") as file_handle:
            with gzip.GzipFile(
                filename="stream_index.jsonl", mode="wb", fileobj=file_handle, mtime=0
            ) as gzip_handle:
                gzip_handle.write(raw)
        manifest["stream_index"].update({
            "compressed_bytes": index_path.stat().st_size,
            "compressed_sha256": materializer.sha256_file(index_path),
            "uncompressed_bytes": len(raw),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        })
        manifest.pop("provenance_sha256", None)
        manifest["provenance_sha256"] = hashlib.sha256(
            materializer._canonical_json(manifest)
        ).hexdigest()
        manifest_path = dataset_dir / "dataset_manifest.json"
        materializer._atomic_json(manifest_path, manifest)
        receipt_path = output / "_receipts" / f"{dataset_id}.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["manifest_sha256"] = materializer.sha256_file(manifest_path)
        materializer._atomic_json(receipt_path, receipt)

    def _write_fixture(self, root: Path, *, output_unit=None):
        scope = {
            "schema": "web-compression-project-dataset-scope-v1",
            "scope_id": "unit-project179",
            "timestamp_dataset_count": 2,
            "timestamp_datasets": [
                {"id": "qiskit_calibration_drift", "repo_id": "THULab/qiskit_calibration_drift", "revision": REVISION_A},
                {"id": "blocked_dataset", "repo_id": "THULab/blocked_dataset", "revision": REVISION_B},
            ],
        }
        scope_path = root / "scope.json"
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        scope_hash = hashlib.sha256(scope_path.read_bytes()).hexdigest()

        mirror = root / "mirror"
        receipts = mirror / "_receipts"
        receipts.mkdir(parents=True)
        top_rows = []
        _FakeReader.rows_by_path = {}
        _FakeReader.table_by_path = {}
        _FakeReader.schema_by_path = {}
        _FakeReader.query_count_by_path = {}
        for dataset_id, revision in (
            ("qiskit_calibration_drift", REVISION_A),
            ("blocked_dataset", REVISION_B),
        ):
            dataset_dir = mirror / dataset_id
            dataset_dir.mkdir()
            source = dataset_dir / "part-1.tsfile"
            source.write_bytes(b"TsFile" + dataset_id.encode("ascii") + b"TsFile")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            files = [{"path": source.name, "bytes": source.stat().st_size, "sha256": digest}]
            receipt = {
                "schema": "thulab-timestamp-dataset-download-receipt-v1",
                "status": "complete",
                "dataset_id": dataset_id,
                "repo_id": f"THULab/{dataset_id}",
                "revision": revision,
                "files": files,
            }
            (receipts / f"{dataset_id}.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8"
            )
            top_rows.append({
                "dataset_id": dataset_id,
                "repo_id": f"THULab/{dataset_id}",
                "revision": revision,
            })
            _FakeReader.rows_by_path[str(source.resolve())] = [
                {"Time": 1_000_001, "country": "DE", "site": 7, "i32": 1, "i64": 10, "f32": 1.25, "f64": None, "event_time": "2024-01-01 00:00:00.123456"},
                {"Time": 1_000_002, "country": "US", "site": 7, "i32": 2, "i64": 20, "f32": None, "f64": 2.5, "event_time": "2024-01-01 00:00:01.123456"},
                {"Time": 1_000_003, "country": "DE", "site": 7, "i32": 3, "i64": 30, "f32": 3.25, "f64": 3.5, "event_time": "2024-01-01 00:00:02.123456"},
            ]
        top = {
            "schema": "thulab-timestamp-scope-download-v1",
            "status": "complete",
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_hash,
            "datasets": top_rows,
        }
        (mirror / "DOWNLOAD_MANIFEST.json").write_text(
            json.dumps(top, sort_keys=True), encoding="utf-8"
        )

        timestamp = {
            "source_column": "Time",
            "source_unit": "us",
            "timezone": "UTC",
            "semantic_kind": "unix_epoch_instant",
            "source_unit_verified": True,
            "utc_instant_verified": True,
            "timezone_inferred": False,
        }
        if output_unit is not None:
            timestamp["output_unit"] = output_unit
            timestamp["output_column"] = f"timestamp_unix_{output_unit}"
        policy = {
            "schema": materializer.POLICY_VERSION,
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_hash,
            "datasets": [{
                "dataset_id": "qiskit_calibration_drift",
                "repo_id": "THULab/qiskit_calibration_drift",
                "revision": REVISION_A,
                "decision": "admit_exact_utc",
                "reason": "unit-test source contract",
                "model": "table",
                "table_name": "metrics",
                "timestamp": timestamp,
                "evidence": ["unit-test frozen source metadata"],
            }],
        }
        policy_path = root / "policies.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        return scope_path, mirror, policy_path

    def _rewrite_fixture_with_admitted_datasets(
        self, scope_path: Path, mirror: Path, policy_path: Path, dataset_ids
    ):
        scope = json.loads(scope_path.read_text())
        base_rows = list(_FakeReader.rows_by_path.values())[0]
        scope["timestamp_dataset_count"] = len(dataset_ids)
        scope["timestamp_datasets"] = []
        top_rows = []
        policy_template = json.loads(policy_path.read_text())["datasets"][0]
        policy_rows = []
        for index, dataset_id in enumerate(dataset_ids, start=1):
            revision = f"{index:x}" * 40
            repo_id = f"THULab/{dataset_id}"
            scope["timestamp_datasets"].append({
                "id": dataset_id,
                "repo_id": repo_id,
                "revision": revision,
            })
            top_rows.append({
                "dataset_id": dataset_id,
                "repo_id": repo_id,
                "revision": revision,
            })
            dataset_dir = mirror / dataset_id
            dataset_dir.mkdir(exist_ok=True)
            source = dataset_dir / "part-1.tsfile"
            source.write_bytes(b"TsFile" + dataset_id.encode("ascii") + b"TsFile")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            receipt = {
                "schema": "thulab-timestamp-dataset-download-receipt-v1",
                "status": "complete",
                "dataset_id": dataset_id,
                "repo_id": repo_id,
                "revision": revision,
                "files": [{
                    "path": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": digest,
                }],
            }
            (mirror / "_receipts" / f"{dataset_id}.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8"
            )
            _FakeReader.rows_by_path[str(source.resolve())] = copy.deepcopy(base_rows)
            policy_row = copy.deepcopy(policy_template)
            policy_row.update({
                "dataset_id": dataset_id,
                "repo_id": repo_id,
                "revision": revision,
            })
            policy_rows.append(policy_row)
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        scope_hash = hashlib.sha256(scope_path.read_bytes()).hexdigest()
        (mirror / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({
            "schema": "thulab-timestamp-scope-download-v1",
            "status": "complete",
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_hash,
            "datasets": top_rows,
        }, sort_keys=True), encoding="utf-8")
        policy_path.write_text(json.dumps({
            "schema": materializer.POLICY_VERSION,
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_hash,
            "datasets": policy_rows,
        }), encoding="utf-8")

    def _bind_fixture(self, scope_path, mirror, policy_path, *, dataset_ids=None):
        scope, rows, scope_sha256 = materializer.load_scope(scope_path)
        policies, policy_file_sha256 = materializer.load_policies(
            policy_path,
            scope_id=scope["scope_id"],
            scope_sha256=scope_sha256,
            scope_rows=rows,
        )
        return materializer.bind_mirror(
            mirror,
            scope=scope,
            rows=rows,
            scope_sha256=scope_sha256,
            policies=policies,
            policy_file_sha256=policy_file_sha256,
            dataset_byte_cap=1024,
            chunk_byte_cap=16,
            metadata_compressed_byte_cap=1024,
            storage_layout="legacy_v1",
            producer_sha256="1" * 64,
            helper_timeseries_sha256="2" * 64,
            helper_tsfile_sha256="3" * 64,
            dataset_ids=dataset_ids,
        )

    def test_bind_mirror_selection_hashes_only_selected_and_preserves_scope_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            original = materializer._eligible_files
            inspected = []

            def record_inventory(dataset_dir, **kwargs):
                inspected.append(dataset_dir.name)
                return original(dataset_dir, **kwargs)

            with mock.patch.object(
                materializer, "_eligible_files", side_effect=record_inventory
            ):
                selected = self._bind_fixture(
                    scope,
                    mirror,
                    policies,
                    dataset_ids=["blocked_dataset"],
                )
            self.assertEqual(
                [binding.row.dataset_id for binding in selected],
                ["blocked_dataset"],
            )
            self.assertEqual(inspected, ["blocked_dataset"])

            full = self._bind_fixture(scope, mirror, policies)
            self.assertEqual(
                selected[0].binding_sha256,
                next(
                    binding.binding_sha256
                    for binding in full
                    if binding.row.dataset_id == "blocked_dataset"
                ),
            )

            reverse_requested = self._bind_fixture(
                scope,
                mirror,
                policies,
                dataset_ids=["blocked_dataset", "qiskit_calibration_drift"],
            )
            self.assertEqual(
                [binding.row.dataset_id for binding in reverse_requested],
                ["qiskit_calibration_drift", "blocked_dataset"],
            )

    def test_bind_mirror_selection_rejects_duplicate_unknown_and_empty_before_mirror_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope_path, _mirror, policy_path = self._write_fixture(root)
            missing_mirror = root / "missing-mirror"
            for requested, message in (
                ("blocked_dataset", "sequence of strings"),
                (["blocked_dataset", "blocked_dataset"], "contain duplicates"),
                (["unknown_dataset"], "outside scope"),
                ([], "must not be empty"),
                (["blocked_dataset", 7], "non-empty strings"),
            ):
                with self.subTest(requested=requested), self.assertRaisesRegex(
                    PreparationError, message
                ):
                    self._bind_fixture(
                        scope_path,
                        missing_mirror,
                        policy_path,
                        dataset_ids=requested,
                    )

    def test_bind_mirror_selection_does_not_inventory_unselected_source_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            unselected = mirror / "blocked_dataset/part-1.tsfile"
            unselected.write_bytes(b"TsFile tampered but unselected TsFile")
            selected = self._bind_fixture(
                scope,
                mirror,
                policies,
                dataset_ids=["qiskit_calibration_drift"],
            )
            self.assertEqual(
                [binding.row.dataset_id for binding in selected],
                ["qiskit_calibration_drift"],
            )
            with self.assertRaisesRegex(
                PreparationError, "receipt/file inventory mismatch"
            ):
                self._bind_fixture(
                    scope,
                    mirror,
                    policies,
                    dataset_ids=["blocked_dataset"],
                )

    def test_bind_mirror_selection_still_validates_all_receipt_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            unselected_receipt = mirror / "_receipts/blocked_dataset.json"
            receipt = json.loads(unselected_receipt.read_text(encoding="utf-8"))
            receipt["revision"] = REVISION_A
            unselected_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                PreparationError, "invalid source receipt identity for blocked_dataset"
            ):
                self._bind_fixture(
                    scope,
                    mirror,
                    policies,
                    dataset_ids=["qiskit_calibration_drift"],
                )

    def test_bind_mirror_selection_still_validates_all_manifest_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            manifest_path = mirror / "DOWNLOAD_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            unselected = next(
                item
                for item in manifest["datasets"]
                if item["dataset_id"] == "blocked_dataset"
            )
            unselected["revision"] = REVISION_A
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                PreparationError, "THULab mirror identity mismatch for blocked_dataset"
            ):
                self._bind_fixture(
                    scope,
                    mirror,
                    policies,
                    dataset_ids=["qiskit_calibration_drift"],
                )

    def test_numeric_only_materialization_partitions_tags_and_defaults_to_microseconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                top = materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )
            self.assertEqual(top["status_counts"], {"materialized": 1, "blocked": 1})
            blocked = json.loads((output / "_receipts/blocked_dataset.json").read_text())
            self.assertEqual(blocked["blocker"]["code"], "missing_explicit_timestamp_policy")
            self.assertFalse((output / "blocked_dataset").exists())

            manifest_path = output / "qiskit_calibration_drift/dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["timestamp_policy"]["output_unit"], "us")
            self.assertEqual(manifest["timestamp_policy"]["output_column"], "timestamp_unix_us")
            self.assertEqual(manifest["entity_count"], 2)
            self.assertEqual(len(manifest["streams"]), 8)
            self.assertEqual(
                [
                    item["source_name"]
                    for item in manifest["table"]["source_file_tables"][0]["retained_fields"]
                ],
                ["i32", "i64", "f32", "f64"],
            )
            dropped = {
                item["name"]: item["decision"]
                for item in manifest["table"]["source_file_tables"][0]["dropped_columns"]
            }
            self.assertEqual(dropped["country"], "partition_entity_then_drop_from_payload")
            self.assertEqual(dropped["description"], "drop_non_numeric_field")
            self.assertEqual(dropped["enabled"], "drop_non_numeric_field")
            self.assertEqual(dropped["calendar_day"], "drop_non_numeric_field")

            de_i32 = next(
                stream for stream in manifest["streams"]
                if stream["field"] == "i32" and 'STRING:"DE"' in stream["entity_values"]
            )
            timestamps = b"".join(
                (output / "qiskit_calibration_drift" / chunk["timestamps_path"]).read_bytes()
                for chunk in de_i32["chunks"]
            )
            values = b"".join(
                (output / "qiskit_calibration_drift" / chunk["values_path"]).read_bytes()
                for chunk in de_i32["chunks"]
            )
            self.assertEqual([item[0] for item in struct.iter_unpack("<q", timestamps)], [1_000_001, 1_000_003])
            self.assertEqual([item[0] for item in struct.iter_unpack("<i", values)], [1, 3])

            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                verified = materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    verify_only=True,
                )
            self.assertEqual(verified["mode"], "verify_only")

    def test_sharded_layout_is_equivalent_bounded_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            legacy = root / "legacy"
            sharded = root / "sharded"
            with (
                mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule),
                mock.patch.object(materializer, "SPOOL_BUCKET_STREAMS", 2),
            ):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=legacy, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="legacy_v1",
                )
                _FakeReader.query_count_by_path = {}
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=sharded, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="sharded_v1",
                )
            legacy_manifest = json.loads(
                (legacy / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            dataset_dir = sharded / "qiskit_calibration_drift"
            shard_manifest, records = self._sharded_records(dataset_dir)
            self.assertEqual(shard_manifest["storage_layout"], "sharded_v1")
            self.assertNotIn("streams", shard_manifest)
            self.assertNotIn("candidates", shard_manifest)
            self.assertNotIn("selected", shard_manifest)
            legacy_receipt = json.loads(
                (legacy / "_receipts/qiskit_calibration_drift.json").read_text()
            )
            sharded_receipt = json.loads(
                (sharded / "_receipts/qiskit_calibration_drift.json").read_text()
            )
            sharded_top = json.loads(
                (sharded / "MATERIALIZATION_MANIFEST.json").read_text()
            )
            self.assertEqual(legacy_receipt["storage_layout"], "legacy_v1")
            self.assertEqual(sharded_receipt["storage_layout"], "sharded_v1")
            self.assertNotEqual(
                legacy_receipt["binding_sha256"], sharded_receipt["binding_sha256"]
            )
            self.assertEqual(shard_manifest["totals"], legacy_manifest["totals"])
            self.assertEqual(shard_manifest["selection"], legacy_manifest["selection"])
            expected_runtime = {
                "producer_sha256": materializer.sha256_file(materializer.PRODUCER_PATH),
                "helper_timeseries_sha256": materializer.sha256_file(
                    materializer.HELPER_TIMESERIES_PATH
                ),
                "helper_tsfile_sha256": materializer.sha256_file(
                    materializer.HELPER_TSFILE_PATH
                ),
            }
            self.assertEqual(shard_manifest["runtime_code_bindings"], expected_runtime)
            self.assertEqual(sharded_receipt["runtime_code_bindings"], expected_runtime)
            self.assertEqual(sharded_top["runtime_code_bindings"], expected_runtime)
            self.assertEqual(records[0]["producer_sha256"], expected_runtime["producer_sha256"])
            self.assertEqual(
                records[0]["helper_timeseries_sha256"],
                expected_runtime["helper_timeseries_sha256"],
            )
            self.assertEqual(
                records[0]["helper_tsfile_sha256"],
                expected_runtime["helper_tsfile_sha256"],
            )
            self.assertTrue(all(
                shard["bytes"] <= 16 for shard in shard_manifest["payload_shards"]
            ))
            stream_records = [row["stream"] for row in records if row["record_type"] == "stream"]
            entity_records = [row for row in records if row["record_type"] == "entity"]
            self.assertEqual(len(stream_records), len(legacy_manifest["streams"]))
            self.assertTrue(all("entity_values" not in row for row in stream_records))
            self.assertEqual(len(entity_records), shard_manifest["selected_entity_dictionary_count"])

            def legacy_payload(stream, name):
                return b"".join(
                    (legacy / "qiskit_calibration_drift" / chunk[f"{name}_path"]).read_bytes()
                    for chunk in stream["chunks"]
                )

            def sharded_payload(stream, name):
                parts = []
                for chunk in stream["chunks"]:
                    segment = chunk[f"{name}_segment"]
                    with (dataset_dir / segment["path"]).open("rb") as handle:
                        handle.seek(segment["offset"])
                        parts.append(handle.read(segment["length"]))
                return b"".join(parts)

            for legacy_stream, sharded_stream in zip(
                legacy_manifest["streams"], stream_records, strict=True
            ):
                for key in ("stream_index", "entity_id", "field", "dtype", "row_count"):
                    self.assertEqual(legacy_stream[key], sharded_stream[key])
                for name in ("timestamps", "values", "missing_bitmap"):
                    self.assertEqual(
                        legacy_payload(legacy_stream, name),
                        sharded_payload(sharded_stream, name),
                    )

            source = str((mirror / "qiskit_calibration_drift/part-1.tsfile").resolve())
            self.assertEqual(_FakeReader.query_count_by_path[source], 2)
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                verified = materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=sharded, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )
            self.assertEqual(verified["storage_layout"], "sharded_v1")

            extra = dataset_dir / "unexpected.bin"
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(PreparationError, "unexpected or missing"):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=sharded, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )
            extra.unlink()
            symlink = dataset_dir / "unexpected-link"
            symlink.symlink_to("stream_index.jsonl.gz")
            with self.assertRaisesRegex(PreparationError, "symlink"):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=sharded, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )
            symlink.unlink()
            shard_path = dataset_dir / shard_manifest["payload_shards"][0]["path"]
            original = shard_path.read_bytes()
            shard_path.write_bytes(b"!" + original[1:])
            with self.assertRaisesRegex(PreparationError, "payload shard hash mismatch"):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=sharded, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )
            shard_path.write_bytes(original)
            index_path = dataset_dir / shard_manifest["stream_index"]["path"]
            index_bytes = index_path.read_bytes()
            index_path.write_bytes(index_bytes + b"tamper")
            with self.assertRaisesRegex(PreparationError, "compressed size mismatch"):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=sharded, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )

    def test_whole_stream_prefix_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=24,
                    chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            self.assertEqual(manifest["selection"]["selected_stream_count"], 1)
            self.assertEqual(manifest["selection"]["selected_logical_bytes"], 24)
            self.assertEqual(manifest["streams"][0]["field"], "i32")
            chunk = manifest["streams"][0]["chunks"][0]
            path = output / "qiskit_calibration_drift" / chunk["timestamps_path"]
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(PreparationError, "size mismatch"):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=24,
                    chunk_byte_cap=16,
                    verify_only=True,
                    dataset_ids=["qiskit_calibration_drift"],
                )

    def test_sharded_segment_boundary_tamper_is_rejected_after_rebinding_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "sharded"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="sharded_v1",
                )
            dataset_dir = output / "qiskit_calibration_drift"
            manifest, records = self._sharded_records(dataset_dir)
            first_stream = next(row["stream"] for row in records if row["record_type"] == "stream")
            first_stream["chunks"][0]["timestamps_segment"]["offset"] += 1
            self._rewrite_sharded_records(
                output, "qiskit_calibration_drift", manifest, records
            )
            with self.assertRaisesRegex(PreparationError, "segment boundary/order mismatch"):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )

    def test_sqlite_entity_store_bounds_python_working_set_at_100k_entities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            source = (mirror / "qiskit_calibration_drift/part-1.tsfile").resolve()
            source_key = str(source)
            _FakeReader.schema_by_path[source_key] = (
                ("entity", "STRING", "TAG"),
                ("value", "INT32", "FIELD"),
            )
            _FakeReader.rows_by_path[source_key] = _SyntheticRows(100_000)
            output = root / "stress"
            with (
                mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule),
                mock.patch.object(materializer, "ENTITY_COUNT_CACHE_LIMIT", 32),
                mock.patch.object(materializer, "ENTITY_LOOKUP_CACHE_LIMIT", 32),
            ):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=12, chunk_byte_cap=12,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="sharded_v1",
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            self.assertEqual(manifest["selection"]["candidate_stream_count"], 100_000)
            self.assertEqual(manifest["selection"]["selected_stream_count"], 1)
            self.assertEqual(manifest["entity_count"], 100_000)
            bounded = manifest["bounded_working_set"]
            self.assertEqual(bounded["entity_counts"], "sqlite_backed")
            self.assertLessEqual(bounded["peak_entity_count_cache_entries"], 32)
            self.assertLessEqual(bounded["peak_entity_lookup_cache_entries"], 32)
            self.assertEqual(_FakeReader.query_count_by_path[source_key], 2)

    def test_sharded_us_to_ns_fault_is_rejected_after_rebinding_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "us-to-ns-fault"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="sharded_v1",
                )
            dataset_dir = output / "qiskit_calibration_drift"
            manifest, records = self._sharded_records(dataset_dir)
            first_stream = next(row["stream"] for row in records if row["record_type"] == "stream")
            first_stream["chunks"][0]["timestamp_unit"] = "ns"
            self._rewrite_sharded_records(
                output, "qiskit_calibration_drift", manifest, records
            )
            with self.assertRaisesRegex(PreparationError, "not signed int64 Unix microseconds"):
                materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"], verify_only=True,
                    storage_layout="sharded_v1",
                )

    def test_subset_then_full_resume_then_full_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            raw = json.loads(policies.read_text())
            second = copy.deepcopy(raw["datasets"][0])
            second.update({
                "dataset_id": "blocked_dataset",
                "repo_id": "THULab/blocked_dataset",
                "revision": REVISION_B,
            })
            raw["datasets"].append(second)
            policies.write_text(json.dumps(raw), encoding="utf-8")
            output = root / "resume"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                subset = materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="sharded_v1",
                )
                self.assertEqual(subset["selected_dataset_count"], 1)
                full = materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                )
                self.assertEqual(full["status_counts"], {"materialized": 2, "blocked": 0})
                verified = materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    storage_layout="sharded_v1", verify_only=True,
                )
            self.assertEqual(verified["mode"], "verify_only")
            self.assertEqual(verified["selected_dataset_count"], 2)

    def test_sharded_metadata_compressed_cap_fails_closed_during_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "metadata-cap"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                top = materializer.run_collection(
                    scope_path=scope, mirror=mirror, policy_path=policies,
                    output=output, dataset_byte_cap=1024, chunk_byte_cap=16,
                    metadata_compressed_byte_cap=1,
                    dataset_ids=["qiskit_calibration_drift"],
                    storage_layout="sharded_v1",
                )
            self.assertEqual(top["status_counts"], {"materialized": 0, "blocked": 1})
            receipt = json.loads(
                (output / "_receipts/qiskit_calibration_drift.json").read_text()
            )
            self.assertIn("compressed stream index exceeds", receipt["blocker"]["detail"])
            self.assertFalse((output / "qiskit_calibration_drift").exists())

    def test_non_us_output_policy_including_us_to_ns_and_overflow_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, _mirror, policies = self._write_fixture(root, output_unit="ns")
            raw_scope, rows, scope_hash = materializer.load_scope(scope)
            with self.assertRaisesRegex(PreparationError, "output_unit must be us"):
                materializer.load_policies(
                    policies,
                    scope_id=raw_scope["scope_id"],
                    scope_sha256=scope_hash,
                    scope_rows=rows,
                )
        self.assertEqual(materializer._convert_timestamp(7, "ms", "us"), 7000)
        with self.assertRaisesRegex(PreparationError, "exceeds signed int64"):
            materializer._convert_timestamp((1 << 63) - 1, "s", "us")

    def test_explicit_temporal_string_primary_and_single_table_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            raw = json.loads(policies.read_text())
            admitted = raw["datasets"][0]
            admitted["table_name"] = "__single_table__"
            admitted["timestamp"] = {
                "source_column": "event_time",
                "output_column": "timestamp_unix_us",
                "timezone": "UTC",
                "semantic_kind": "unix_epoch_instant",
                "source_format_verified": True,
                "utc_instant_verified": True,
                "timezone_inferred": False,
            }
            admitted["temporal_string_fields"] = [{
                "source_name": "event_time",
                "source_type": "STRING",
                "source_category": "FIELD",
                "output_name": "event_time__timestamp_unix_us",
                "parse_kind": "strptime",
                "format": "%Y-%m-%d %H:%M:%S%z",
                "accepted_formats": [
                    "%Y-%m-%d %H:%M:%S%z",
                    "%Y-%m-%d %H:%M:%S.%f%z",
                ],
                "format_class": "embedded_offset_seconds_or_fractional",
                "timezone": "embedded_offset",
                "timezone_basis": "embedded_numeric_offset",
                "semantic_kind": "unix_epoch_instant",
                "event_utc_instant_claim": True,
                "output_type": "int64",
                "output_unit": "us",
                "candidate_definition_count": 1,
                "applicability": "matching source file schema only",
                "evidence": ["frozen unit-test field contract"],
            }]
            policies.write_text(json.dumps(raw), encoding="utf-8")
            source = (mirror / "qiskit_calibration_drift/part-1.tsfile").resolve()
            for row in _FakeReader.rows_by_path[str(source)]:
                row["event_time"] += "+00:00"
            _FakeReader.table_by_path[str(source)] = "qiskit_only_table"
            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=2048,
                    chunk_byte_cap=32,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            self.assertEqual(
                manifest["table"]["source_file_tables"][0]["table_name"],
                "qiskit_only_table",
            )
            self.assertEqual(manifest["timestamp_policy"]["source_column"], "event_time")
            self.assertEqual(manifest["timestamp_policy"]["output_unit"], "us")
            temporal_policy = manifest["timestamp_policy"]["temporal_string_fields"][0]
            self.assertTrue(temporal_policy["event_utc_instant_claim"])
            self.assertEqual(temporal_policy["semantic_kind"], "unix_epoch_instant")
            self.assertEqual(temporal_policy["timezone_basis"], "embedded_numeric_offset")
            self.assertEqual(
                temporal_policy["format_class"],
                "embedded_offset_seconds_or_fractional",
            )
            de_temporal = next(
                stream for stream in manifest["streams"]
                if stream["field"] == "event_time__timestamp_unix_us"
                and 'STRING:"DE"' in stream["entity_values"]
            )
            values = b"".join(
                (output / "qiskit_calibration_drift" / chunk["values_path"]).read_bytes()
                for chunk in de_temporal["chunks"]
            )
            self.assertEqual(
                [item[0] for item in struct.iter_unpack("<q", values)],
                [1704067200123456, 1704067202123456],
            )
            self.assertTrue(all(
                stream["table_name"] == "qiskit_only_table"
                for stream in manifest["streams"]
            ))

    def test_source_bound_integer_temporal_field_is_exact_and_preserves_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            source = (mirror / "qiskit_calibration_drift/part-1.tsfile").resolve()
            source_key = str(source)
            schema_columns = tuple(
                ("reference_time", "INT64", "FIELD")
                if name == "event_time"
                else (name, dtype, category)
                for name, dtype, category in _FakeSchema.COLUMNS
            )
            _FakeReader.schema_by_path[source_key] = schema_columns
            _FakeReader.rows_by_path[source_key][0]["reference_time"] = 1_704_067_200
            _FakeReader.rows_by_path[source_key][1]["reference_time"] = None
            _FakeReader.rows_by_path[source_key][2]["reference_time"] = 1_704_067_202
            for row in _FakeReader.rows_by_path[source_key]:
                row.pop("event_time", None)
            schema_basis = [
                {"name": name, "tsfile_type": dtype, "category": category}
                for name, dtype, category in schema_columns
            ]
            schema_sha256 = hashlib.sha256(
                materializer._canonical_json(schema_basis)
            ).hexdigest()

            raw = json.loads(policies.read_text())
            admitted = raw["datasets"][0]
            # The same-named dataset-level STRING rule applies to a different
            # heterogeneous source.  This numeric source may override it only
            # through the exact path/content/table/schema binding below.
            admitted["temporal_string_fields"] = [{
                "source_name": "reference_time",
                "source_type": "STRING",
                "source_category": "TAG",
                "output_name": "reference_time__timestamp_unix_us",
                "parse_kind": "strptime",
                "format": "%Y-%m-%dT%H:%M:%SZ",
                "timezone": "UTC",
                "format_class": "explicit_z",
                "timezone_basis": "explicit_z_suffix",
                "semantic_kind": "unix_epoch_instant",
                "event_utc_instant_claim": True,
                "output_type": "int64",
                "output_unit": "us",
                "candidate_definition_count": 1,
                "applicability": "matching source file schema only",
                "evidence": ["unit-test heterogeneous string representation"],
            }]
            admitted["source_bound_temporal_fields"] = [{
                "source_path": "part-1.tsfile",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "table_name": "metrics",
                "source_schema_sha256": schema_sha256,
                "source_name": "reference_time",
                "source_type": "INT64",
                "source_category": "FIELD",
                "source_unit": "s",
                "output_name": "reference_time__timestamp_unix_us",
                "output_type": "int64",
                "output_unit": "us",
                "parse_kind": "integer_epoch",
                "timezone": "UTC",
                "event_utc_instant_claim": True,
                "semantic_kind": "unix_epoch_instant",
                "timezone_basis": "upstream_encoded_epoch",
                "evidence": ["unit-test exact source/schema binding"],
            }]
            policies.write_text(json.dumps(raw), encoding="utf-8")

            output = root / "source-bound"
            with mock.patch.object(
                materializer, "_load_tsfile_module", return_value=_FakeTsFileModule
            ):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=20_000,
                    chunk_byte_cap=64,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            bound_policy = manifest["timestamp_policy"][
                "source_bound_temporal_fields"
            ][0]
            self.assertEqual(bound_policy["source_schema_sha256"], schema_sha256)
            retained = manifest["table"]["source_file_tables"][0]["retained_fields"]
            self.assertEqual(
                sum(row["source_name"] == "reference_time" for row in retained),
                1,
            )
            self.assertEqual(
                next(row for row in retained if row["source_name"] == "reference_time")[
                    "conversion"
                ],
                "source_bound_integer_epoch_to_int64_unix_us",
            )
            de_reference = next(
                stream for stream in manifest["streams"]
                if stream["field"] == "reference_time__timestamp_unix_us"
                and 'STRING:"DE"' in stream["entity_values"]
            )
            values = b"".join(
                (output / "qiskit_calibration_drift" / chunk["values_path"]).read_bytes()
                for chunk in de_reference["chunks"]
            )
            self.assertEqual(
                [item[0] for item in struct.iter_unpack("<q", values)],
                [1_704_067_200_000_000, 1_704_067_202_000_000],
            )

            spec = materializer.SourceBoundTemporalField(
                source_path="part-1.tsfile",
                source_sha256="a" * 64,
                table_name="metrics",
                source_schema_sha256="b" * 64,
                source_name="reference_time",
                source_type="INT64",
                source_category="FIELD",
                source_unit="s",
                output_name="reference_time__timestamp_unix_us",
                output_unit="us",
                parse_kind="integer_epoch",
                timezone_name="UTC",
                event_utc_instant_claim=True,
                semantic_kind="unix_epoch_instant",
                timezone_basis="upstream_encoded_epoch",
                evidence=("unit-test",),
            )
            with self.assertRaisesRegex(PreparationError, "expected integer"):
                materializer._convert_temporal_value("1704067200", spec)
            with self.assertRaisesRegex(PreparationError, "exceeds signed int64"):
                materializer._convert_temporal_value((1 << 63) - 1, spec)

    def test_temporal_parse_failure_creates_file_specific_blocked_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            raw = json.loads(policies.read_text())
            admitted = raw["datasets"][0]
            admitted["timestamp"] = {
                "source_column": "event_time",
                "output_column": "timestamp_unix_us",
                "timezone": "UTC",
                "semantic_kind": "unix_epoch_instant",
                "source_format_verified": True,
                "utc_instant_verified": True,
                "timezone_inferred": False,
            }
            admitted["temporal_string_fields"] = [{
                "source_name": "event_time",
                "source_type": "STRING",
                "source_category": "FIELD",
                "parse_kind": "strptime",
                "format": "%Y-%m-%d%z",
                "format_class": "embedded_offset",
                "timezone": "embedded_offset",
                "timezone_basis": "embedded_numeric_offset",
                "semantic_kind": "unix_epoch_instant",
                "event_utc_instant_claim": True,
                "output_type": "int64",
                "output_unit": "us",
                "candidate_definition_count": 1,
                "applicability": "matching source file schema only",
                "evidence": ["unit-test evidence"],
            }]
            policies.write_text(json.dumps(raw), encoding="utf-8")
            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                top = materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            self.assertEqual(top["status_counts"], {"materialized": 0, "blocked": 1})
            receipt = json.loads(
                (output / "_receipts/qiskit_calibration_drift.json").read_text()
            )
            self.assertIn("source_file=part-1.tsfile", receipt["blocker"]["detail"])
            self.assertIn("does not match frozen format", receipt["blocker"]["detail"])
            self.assertEqual(receipt["files"][0]["status"], "blocked_at_this_file")

    def test_stored_epoch_coordinate_is_labeled_without_event_utc_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            raw = json.loads(policies.read_text())
            admitted = raw["datasets"][0]
            admitted["decision"] = "admit_stored_epoch_coordinate"
            admitted["timestamp"] = {
                "source_column": "Time",
                "source_unit": "us",
                "output_column": "timestamp_unix_us",
                "output_unit": "us",
                "semantic_kind": "stored_epoch_coordinate",
                "source_unit_verified": True,
                "event_utc_instant_claim": False,
                "timezone_basis": "upstream_encoded_epoch",
                "timezone_inferred": False,
            }
            policies.write_text(json.dumps(raw), encoding="utf-8")
            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            policy = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )["timestamp_policy"]
            self.assertEqual(policy["admission_decision"], "admit_stored_epoch_coordinate")
            self.assertEqual(policy["semantic_kind"], "stored_epoch_coordinate")
            self.assertFalse(policy["event_utc_instant_claim"])
            self.assertIsNone(policy["timezone"])
            self.assertEqual(policy["timezone_basis"], "upstream_encoded_epoch")

    def test_heterogeneous_source_schemas_and_string_only_file_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            raw_policy = json.loads(policies.read_text())
            raw_policy["datasets"][0]["table_name"] = "__single_table__"
            raw_policy["datasets"][0]["temporal_string_fields"] = [{
                "source_name": "start_timestamp",
                "source_type": "STRING",
                "source_category": "TAG",
                "output_name": "start_timestamp__timestamp_unix_us",
                "parse_kind": "integer_epoch",
                "source_unit": "d",
                "timezone": "UTC",
                "format_class": "integer_epoch_days",
                "timezone_basis": "utc_midnight_coordinate_convention",
                "semantic_kind": "unix_epoch_day_coordinate",
                "event_utc_instant_claim": False,
                "output_type": "int64",
                "output_unit": "us",
                "candidate_definition_count": 1,
                "applicability": "matching source file schema only",
                "evidence": ["unit-test day ordinal contract"],
            }]
            raw_policy["datasets"][0]["dropped_timestamp_like_string_fields"] = [{
                "source_name": "description",
                "source_type": "TEXT",
                "source_category": "FIELD",
                "decision": "drop_clock_only_without_calendar_date",
                "format_class": "clock_only_drop",
                "reason": "unit-test clock lacks a calendar date",
                "candidate_definition_count": 1,
                "evidence": ["unit-test explicit clock-only drop"],
            }]
            policies.write_text(json.dumps(raw_policy), encoding="utf-8")

            dataset_dir = mirror / "qiskit_calibration_drift"
            second = dataset_dir / "part-2.tsfile"
            second.write_bytes(b"TsFileheterogeneous-secondTsFile")
            second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
            third = dataset_dir / "part-3.tsfile"
            third.write_bytes(b"TsFilestring-only-thirdTsFile")
            third_digest = hashlib.sha256(third.read_bytes()).hexdigest()
            receipt_path = mirror / "_receipts/qiskit_calibration_drift.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["files"].append({
                "path": second.name,
                "bytes": second.stat().st_size,
                "sha256": second_digest,
            })
            receipt["files"].append({
                "path": third.name,
                "bytes": third.stat().st_size,
                "sha256": third_digest,
            })
            receipt["files"].sort(key=lambda item: item["path"])
            receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            second_key = str(second.resolve())
            _FakeReader.table_by_path[second_key] = "different_table"
            _FakeReader.schema_by_path[second_key] = (
                ("device", "STRING", "TAG"),
                ("start_timestamp", "STRING", "TAG"),
                ("temperature", "DOUBLE", "FIELD"),
                ("note", "STRING", "FIELD"),
            )
            _FakeReader.rows_by_path[second_key] = [
                {"Time": 2_000_001, "device": "d1", "start_timestamp": "19723", "temperature": 10.5, "note": "x"},
                {"Time": 2_000_002, "device": "d1", "start_timestamp": "19724", "temperature": 11.5, "note": "y"},
            ]
            third_key = str(third.resolve())
            _FakeReader.table_by_path[third_key] = "string_lookup_table"
            _FakeReader.schema_by_path[third_key] = (
                ("asset_index", "STRING", "TAG"),
                ("ticker", "STRING", "FIELD"),
            )
            _FakeReader.rows_by_path[third_key] = [
                {"Time": 0, "asset_index": "0", "ticker": "AAA"},
                {"Time": 1, "asset_index": "1", "ticker": "BBB"},
            ]

            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=4096,
                    chunk_byte_cap=32,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            source_tables = manifest["table"]["source_file_tables"]
            self.assertEqual(
                {(item["source_path"], item["table_name"]) for item in source_tables},
                {
                    ("part-1.tsfile", "metrics"),
                    ("part-2.tsfile", "different_table"),
                    ("part-3.tsfile", "string_lookup_table"),
                },
            )
            self.assertEqual(
                {tuple(item["partition_tags"]) for item in source_tables},
                {
                    ("country", "site"),
                    ("device", "start_timestamp"),
                    ("asset_index",),
                },
            )
            third_plan = next(
                item for item in source_tables if item["source_path"] == "part-3.tsfile"
            )
            self.assertEqual(third_plan["retained_fields"], [])
            self.assertEqual(third_plan["source_rows"], 2)
            self.assertFalse(any(
                stream["source_path"] == "part-3.tsfile"
                for stream in manifest["streams"]
            ))
            self.assertEqual(
                _FakeReader.query_count_by_path[third_key],
                1,
                "a source file with no selected stream must not be decoded twice",
            )
            first_drops = {
                item["name"]: item
                for item in next(
                    item for item in source_tables
                    if item["source_path"] == "part-1.tsfile"
                )["dropped_columns"]
            }
            self.assertEqual(
                first_drops["description"]["decision"],
                "drop_clock_only_without_calendar_date",
            )
            self.assertNotIn(
                "description",
                {
                    item["name"]
                    for item in next(
                        item for item in source_tables
                        if item["source_path"] == "part-2.tsfile"
                    )["dropped_columns"]
                },
            )
            self.assertEqual(
                manifest["timestamp_policy"]["dropped_timestamp_like_string_fields"][0][
                    "format_class"
                ],
                "clock_only_drop",
            )
            second_streams = [
                stream for stream in manifest["streams"]
                if stream["source_path"] == "part-2.tsfile"
            ]
            self.assertEqual(len(second_streams), 4)
            self.assertEqual(
                {stream["field"] for stream in second_streams},
                {"temperature", "start_timestamp__timestamp_unix_us"},
            )
            self.assertTrue(all(stream["table_name"] == "different_table" for stream in second_streams))
            self.assertTrue(all(stream["row_count"] == 1 for stream in second_streams))
            temporal_streams = [
                stream for stream in second_streams
                if stream["field"] == "start_timestamp__timestamp_unix_us"
            ]
            self.assertEqual(len({stream["entity_id"] for stream in temporal_streams}), 2)
            values = b"".join(
                (output / "qiskit_calibration_drift" / chunk["values_path"]).read_bytes()
                for stream in temporal_streams
                for chunk in stream["chunks"]
            )
            self.assertEqual(
                sorted(item[0] for item in struct.iter_unpack("<q", values)),
                [19723 * 86_400 * 1_000_000, 19724 * 86_400 * 1_000_000],
            )
            temporal_policy = manifest["timestamp_policy"]["temporal_string_fields"][0]
            self.assertFalse(temporal_policy["event_utc_instant_claim"])
            self.assertEqual(temporal_policy["semantic_kind"], "unix_epoch_day_coordinate")
            self.assertEqual(
                temporal_policy["timezone_basis"],
                "utc_midnight_coordinate_convention",
            )

    def test_cap_prefix_uses_frozen_schema_order_not_alphabetical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            source = (mirror / "qiskit_calibration_drift/part-1.tsfile").resolve()
            source_key = str(source)
            _FakeReader.schema_by_path[source_key] = (
                ("country", "STRING", "TAG"),
                ("z_metric", "INT32", "FIELD"),
                ("a_metric", "INT32", "FIELD"),
            )
            _FakeReader.rows_by_path[source_key] = [
                {"Time": 1, "country": "DE", "z_metric": 9, "a_metric": 1},
                {"Time": 2, "country": "DE", "z_metric": 8, "a_metric": 2},
                {"Time": 3, "country": "US", "z_metric": 7, "a_metric": 3},
            ]
            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=24,
                    chunk_byte_cap=16,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            self.assertEqual([stream["field"] for stream in manifest["streams"]], ["z_metric"])
            self.assertIn(
                "frozen source schema column order",
                manifest["selection"]["ordering"],
            )

    def test_mixed_axis_adapter_filters_rows_and_excludes_whole_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            raw = json.loads(policies.read_text())
            admitted = raw["datasets"][0]
            admitted["decision"] = "admit_dataset_adapter"
            admitted["table_name"] = "__single_table__"
            admitted.pop("timestamp")
            admitted["temporal_string_fields"] = [{
                "source_name": "event_time",
                "source_type": "STRING",
                "source_category": "FIELD",
                "output_name": "event_time__timestamp_unix_us",
                "parse_kind": "strptime",
                "format": "%Y-%m-%d %H:%M:%S.%f%z",
                "format_class": "embedded_offset_fractional",
                "timezone": "embedded_offset",
                "timezone_basis": "embedded_numeric_offset",
                "semantic_kind": "stored_timestamp_coordinate",
                "event_utc_instant_claim": False,
                "output_type": "int64",
                "output_unit": "us",
                "candidate_definition_count": 1,
                "applicability": "matching source file schema only",
                "evidence": ["unit-test embedded-offset stored-coordinate contract"],
            }]
            admitted["mixed_axis_adapter"] = {
                "adapter_id": "unit-mixed-axis-v1",
                "source_rules": [
                    {
                        "rule_id": "primary-table",
                        "table_names": ["metrics"],
                        "action": "materialize",
                        "expected_source_file_count": 1,
                        "reason": "unit-test materialized table",
                        "evidence": ["unit-test source/table evidence"],
                        "timestamp": {
                            "source_column": "event_time",
                            "output_column": "timestamp_unix_us",
                            "output_unit": "us",
                            "source_format_verified": True,
                            "event_utc_instant_claim": False,
                            "semantic_kind": "stored_timestamp_coordinate",
                            "timezone_basis": "embedded_numeric_offset",
                            "timezone_inferred": False,
                        },
                        "row_filter": {
                            "kind": "all",
                            "conditions": [
                                {
                                    "source_name": "Time",
                                    "source_type": "INT64",
                                    "source_category": "TIME",
                                    "operator": "int64_gte",
                                    "value": 1_000_002,
                                    "drop_reason": "physical_time_below_threshold",
                                },
                                {
                                    "source_name": "f64",
                                    "source_type": "DOUBLE",
                                    "source_category": "FIELD",
                                    "operator": "not_null",
                                    "drop_reason": "required_numeric_value_is_null",
                                },
                            ],
                            "expected_input_rows": 3,
                            "expected_output_rows": 2,
                            "expected_drop_rows": 1,
                            "evidence": ["unit-test exact row-filter accounting"],
                        },
                    },
                    {
                        "rule_id": "lookup-exclusion",
                        "table_names": ["lookup"],
                        "action": "exclude_entire_source_file",
                        "expected_source_file_count": 1,
                        "reason": "lookup has no valid time-series payload",
                        "evidence": ["unit-test explicit whole-file exclusion"],
                    },
                ],
            }
            policies.write_text(json.dumps(raw), encoding="utf-8")

            primary = (mirror / "qiskit_calibration_drift/part-1.tsfile").resolve()
            for row in _FakeReader.rows_by_path[str(primary)]:
                row["event_time"] += "+00:00"
            lookup = mirror / "qiskit_calibration_drift/lookup.tsfile"
            lookup.write_bytes(b"TsFilelookup-onlyTsFile")
            receipt_path = mirror / "_receipts/qiskit_calibration_drift.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["files"].append({
                "path": lookup.name,
                "bytes": lookup.stat().st_size,
                "sha256": hashlib.sha256(lookup.read_bytes()).hexdigest(),
            })
            receipt["files"].sort(key=lambda item: item["path"])
            receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            lookup_key = str(lookup.resolve())
            _FakeReader.table_by_path[lookup_key] = "lookup"
            _FakeReader.schema_by_path[lookup_key] = (
                ("key", "STRING", "TAG"),
                ("label", "STRING", "FIELD"),
            )
            _FakeReader.rows_by_path[lookup_key] = [
                {"Time": 0, "key": "0", "label": "not-a-series"},
            ]

            output = root / "prepared"
            with mock.patch.object(materializer, "_load_tsfile_module", return_value=_FakeTsFileModule):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=4096,
                    chunk_byte_cap=32,
                    dataset_ids=["qiskit_calibration_drift"],
                )
            manifest = json.loads(
                (output / "qiskit_calibration_drift/dataset_manifest.json").read_text()
            )
            self.assertEqual(
                manifest["timestamp_policy"]["admission_decision"],
                "admit_dataset_adapter",
            )
            self.assertIsNone(manifest["timestamp_policy"]["event_utc_instant_claim"])
            self.assertFalse(
                manifest["timestamp_policy"]["source_rules"][0]["timestamp"][
                    "event_utc_instant_claim"
                ]
            )
            primary_plan = next(
                plan for plan in manifest["table"]["source_file_tables"]
                if plan["source_path"] == "part-1.tsfile"
            )
            self.assertEqual(
                primary_plan["row_filter_observed"]["input_rows"], 3
            )
            self.assertEqual(
                primary_plan["row_filter_observed"]["output_rows"], 2
            )
            self.assertEqual(
                primary_plan["row_filter_observed"]["dropped_rows"], 1
            )
            self.assertEqual(
                primary_plan["row_filter_observed"]["condition_failure_counts"],
                {
                    "physical_time_below_threshold": 1,
                    "required_numeric_value_is_null": 1,
                },
            )
            self.assertFalse(any(
                stream["source_path"] == "lookup.tsfile"
                for stream in manifest["streams"]
            ))
            receipt = json.loads(
                (output / "_receipts/qiskit_calibration_drift.json").read_text()
            )
            lookup_receipt = next(
                item for item in receipt["files"] if item["path"] == "lookup.tsfile"
            )
            self.assertEqual(
                lookup_receipt["status"],
                "excluded_entire_source_file_by_dataset_adapter",
            )

    def test_source_receipt_detects_mirror_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            (mirror / "qiskit_calibration_drift/part-1.tsfile").write_bytes(b"TsFiletamperedTsFile")
            with self.assertRaisesRegex(PreparationError, "receipt/file inventory mismatch"):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "prepared",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )

    def test_source_symlink_to_external_same_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            source = mirror / "qiskit_calibration_drift/part-1.tsfile"
            external = root / "same-source.tsfile"
            external.write_bytes(source.read_bytes())
            source.unlink()
            source.symlink_to(external)
            with self.assertRaisesRegex(
                PreparationError, "source dataset tree entry must not be a symlink"
            ):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "prepared",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )

    def test_source_receipt_file_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            receipt = mirror / "_receipts/qiskit_calibration_drift.json"
            external = root / "same-receipt.json"
            external.write_bytes(receipt.read_bytes())
            receipt.unlink()
            receipt.symlink_to(external)
            with self.assertRaisesRegex(
                PreparationError, "source receipt entry must be a regular file"
            ):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "prepared",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )

    def test_source_receipt_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            receipts = mirror / "_receipts"
            external = root / "same-receipts"
            receipts.rename(external)
            receipts.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(
                PreparationError, "source receipt directory must be a real directory"
            ):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "prepared",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )

    def test_source_dataset_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            dataset = mirror / "qiskit_calibration_drift"
            external = root / "same-dataset"
            dataset.rename(external)
            dataset.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(
                PreparationError, "source dataset directory.*must be a real directory"
            ):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "prepared",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )

    def test_requested_output_rejects_regular_and_dangling_symlinks_before_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            real_output = root / "real-output"
            real_output.mkdir()
            for name, target in (
                ("ordinary-output-link", real_output),
                ("dangling-output-link", root / "missing-output-target"),
            ):
                output = root / name
                output.symlink_to(target, target_is_directory=True)
                with self.subTest(name=name), self.assertRaisesRegex(
                    PreparationError, "requested output path must be a real directory"
                ):
                    materializer.run_collection(
                        scope_path=scope,
                        mirror=mirror,
                        policy_path=policies,
                        output=output,
                        dataset_byte_cap=1024,
                        chunk_byte_cap=16,
                    )

    def test_receipt_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "prepared"
            output.mkdir()
            outside_receipts = root / "outside-receipts"
            outside_receipts.mkdir()
            (output / "_receipts").symlink_to(
                outside_receipts, target_is_directory=True
            )
            with self.assertRaisesRegex(
                PreparationError, "receipt directory must be a real directory"
            ):
                materializer.run_collection(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                )

    def test_count_only_planning_receipt_supports_four_dataset_pilot_and_full_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            pilot_ids = [
                "finance1k",
                "freshretailnet_50k",
                "kaggle_web_traffic",
                "kaggle_web_traffic_weekly",
            ]
            all_ids = [*pilot_ids, "qiskit_calibration_drift"]
            self._rewrite_fixture_with_admitted_datasets(
                scope, mirror, policies, all_ids
            )
            output = root / "materialized-output-must-not-exist"
            first_path = root / "pilot-plan-a.json"
            second_path = root / "pilot-plan-b.json"
            full_path = root / "full-plan.json"
            with mock.patch.object(
                materializer, "_load_tsfile_module", return_value=_FakeTsFileModule
            ):
                pilot = materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    planning_receipt=first_path,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                    dataset_ids=pilot_ids,
                )
                materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    planning_receipt=second_path,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                    dataset_ids=pilot_ids,
                )
                full = materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    planning_receipt=full_path,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                )
            self.assertFalse(output.exists())
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(pilot["selected_dataset_ids"], pilot_ids)
            self.assertEqual(pilot["selected_dataset_count"], 4)
            self.assertEqual(pilot["full_scope_dataset_count"], 5)
            self.assertEqual(full["selected_dataset_count"], 5)
            self.assertEqual(full["selected_dataset_ids"], all_ids)
            self.assertEqual(pilot["storage_layout"], "sharded_v1")
            self.assertEqual(pilot["totals"]["candidate_stream_count"], 32)
            self.assertEqual(pilot["totals"]["selected_stream_count"], 32)
            self.assertEqual(pilot["totals"]["per_stream_payload_files"], 0)
            self.assertGreater(
                pilot["totals"]["expected_payload_shard_count_upper_bound"], 0
            )
            for dataset in pilot["datasets"]:
                self.assertFalse(
                    dataset["source_validation"]["payload_values_packed"]
                )
                self.assertEqual(
                    dataset["layout_assertions"]["per_stream_payload_files"], 0
                )
                self.assertGreater(
                    dataset["expected_stream_index"][
                        "uncompressed_bytes_upper_bound"
                    ],
                    0,
                )
                self.assertGreater(
                    dataset["peak_temporary_bytes_upper_bound"], 0
                )

    def test_planning_detects_source_tamper_during_count_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            receipt = root / "plan.json"
            source = mirror / "qiskit_calibration_drift/part-1.tsfile"
            original_schema = materializer._schema_for_binding
            tampered = False

            def schema_then_tamper(binding):
                nonlocal tampered
                result = original_schema(binding)
                if binding.row.dataset_id == "qiskit_calibration_drift" and not tampered:
                    source.write_bytes(b"TsFilechanged-after-bindingTsFile")
                    tampered = True
                return result

            with (
                mock.patch.object(
                    materializer, "_load_tsfile_module", return_value=_FakeTsFileModule
                ),
                mock.patch.object(
                    materializer, "_schema_for_binding", side_effect=schema_then_tamper
                ),
                self.assertRaisesRegex(PreparationError, "source changed after mirror binding"),
            ):
                materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "output",
                    planning_receipt=receipt,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                    dataset_ids=["qiskit_calibration_drift"],
                )
            self.assertFalse(receipt.exists())
            self.assertFalse((root / "output").exists())

    def test_planning_receipt_path_must_be_outside_output_and_real_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(PreparationError, "outside the materialization output"):
                materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=output,
                    planning_receipt=output / "plan.json",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                    dataset_ids=["qiskit_calibration_drift"],
                )
            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(PreparationError, "real directory"):
                materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "other-output",
                    planning_receipt=linked_parent / "plan.json",
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                    dataset_ids=["qiskit_calibration_drift"],
                )

    def test_planning_receipt_never_clobbers_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scope, mirror, policies = self._write_fixture(root)
            receipt = root / "plan.json"
            receipt.write_bytes(b"keep-this-exact-content")
            with self.assertRaisesRegex(PreparationError, "already exists"):
                materializer.run_planning_receipt(
                    scope_path=scope,
                    mirror=mirror,
                    policy_path=policies,
                    output=root / "output",
                    planning_receipt=receipt,
                    dataset_byte_cap=1024,
                    chunk_byte_cap=16,
                    storage_layout="sharded_v1",
                    dataset_ids=["qiskit_calibration_drift"],
                )
            self.assertEqual(receipt.read_bytes(), b"keep-this-exact-content")


if __name__ == "__main__":
    unittest.main()
