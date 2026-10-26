#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("validate_project179_thulab_csv_v1.py")
SPEC = importlib.util.spec_from_file_location("validate_project179_thulab_csv_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
# ProcessPoolExecutor pickles a top-level worker by its module-qualified name.
# Register this deliberately importlib-loaded test subject under that canonical
# name before executing it, exactly as a normal import would.  Without this,
# pickle imports a second module object and correctly rejects the worker as
# "not the same object"; that is a test-loader artifact, not the CLI runtime.
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(rows) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fixed(value: int) -> str:
    return f"{value:020d}"


class _Column:
    def __init__(self, name: str, dtype: str, category: str) -> None:
        self.name = name
        self.dtype = dtype
        self.category = category

    def get_column_name(self):
        return self.name

    def get_data_type(self):
        return self.dtype

    def get_category(self):
        return self.category


class _Schema:
    def get_columns(self):
        return [_Column("value", "INT64", "FIELD")]


class _Metadata:
    def get_column_list(self):
        return ["Time", "value"]

    def get_data_type_list(self):
        return ["INT64", "INT64"]


class _Result:
    def __init__(self) -> None:
        self.rows = [{"Time": 1, "value": 10}, {"Time": 2, "value": 20}]
        self.index = -1

    def get_metadata(self):
        return _Metadata()

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def is_null_by_index(self, index):
        return False

    def get_value_by_index(self, index):
        return self.rows[self.index]["Time" if index == 1 else "value"]

    def close(self):
        return None


class _Reader:
    def __init__(self, _path: str) -> None:
        pass

    def get_all_table_schemas(self):
        return {"metrics": _Schema()}

    def query_table(self, table, columns, start, end):
        if table != "metrics" or columns != ["value"]:
            raise AssertionError((table, columns))
        if start != -(1 << 63) or end != (1 << 63) - 1:
            raise AssertionError((start, end))
        return _Result()

    def close(self):
        return None


class _TsFileModule:
    TsFileReader = _Reader


class _WideSchema:
    def get_columns(self):
        return [
            _Column(f"v{index:04d}", "DOUBLE", "FIELD")
            for index in range(5_548)
        ]


class _WideReader(_Reader):
    def get_all_table_schemas(self):
        return {"metrics": _WideSchema()}

    def query_table(self, table, columns, start, end):
        if table != "metrics" or len(columns) != 5_548:
            raise AssertionError((table, len(columns)))
        return _WideResult(columns)


class _WideMetadata:
    def __init__(self, columns) -> None:
        self.columns = columns

    def get_column_list(self):
        return ["Time", *self.columns]

    def get_data_type_list(self):
        return ["INT64", *("DOUBLE" for _ in self.columns)]


class _WideResult:
    def __init__(self, columns) -> None:
        self.columns = columns
        self.index = -1

    def get_metadata(self):
        return _WideMetadata(self.columns)

    def next(self):
        self.index += 1
        return self.index < 2

    def is_null_by_index(self, index):
        return False

    def get_value_by_index(self, index):
        return self.index + 1 if index == 1 else 0.0

    def close(self):
        return None


class _WideTsFileModule:
    TsFileReader = _WideReader


class _NullTagSchema:
    def get_columns(self):
        return [
            _Column("symbol", "STRING", "TAG"),
            _Column("value", "INT64", "FIELD"),
        ]


class _NullTagMetadata:
    def get_column_list(self):
        return ["Time", "symbol", "value"]

    def get_data_type_list(self):
        return ["INT64", "STRING", "INT64"]


class _NullTagResult:
    def __init__(self) -> None:
        self.rows = [
            [1, None, 10],
            [2, "A", 20],
            [3, None, 30],
        ]
        self.index = -1

    def get_metadata(self):
        return _NullTagMetadata()

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def is_null_by_index(self, index):
        return self.rows[self.index][index - 1] is None

    def get_value_by_index(self, index):
        return self.rows[self.index][index - 1]

    def close(self):
        return None


class _NullTagReader:
    def __init__(self, _path: str) -> None:
        pass

    def get_all_table_schemas(self):
        return {"metrics": _NullTagSchema()}

    def query_table(self, table, columns, start, end):
        if table != "metrics" or columns != ["symbol", "value"]:
            raise AssertionError((table, columns))
        return _NullTagResult()

    def close(self):
        return None


class _NullTagTsFileModule:
    TsFileReader = _NullTagReader


class _MacroSchema:
    def __init__(self, dtype: str = "FLOAT") -> None:
        self.dtype = dtype

    def get_columns(self):
        return [_Column("value", self.dtype, "FIELD")]


class _MacroMetadata:
    def __init__(self, dtype: str = "FLOAT") -> None:
        self.dtype = dtype

    def get_column_list(self):
        return ["Time", "value"]

    def get_data_type_list(self):
        return ["INT64", self.dtype]


class _MacroResult:
    def __init__(self, rows, dtype: str = "FLOAT") -> None:
        self.rows = rows
        self.dtype = dtype
        self.index = -1

    def get_metadata(self):
        return _MacroMetadata(self.dtype)

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def is_null_by_index(self, index):
        return self.rows[self.index][index - 1] is None

    def get_value_by_index(self, index):
        return self.rows[self.index][index - 1]

    def close(self):
        return None


class _MacroReader:
    fixtures = {}

    def __init__(self, path: str) -> None:
        self.fixture = self.fixtures[Path(path).name]

    def get_all_table_schemas(self):
        return {
            self.fixture["table"]: _MacroSchema(self.fixture.get("dtype", "FLOAT"))
        }

    def query_table(self, table, columns, start, end):
        if table != self.fixture["table"] or columns != ["value"]:
            raise AssertionError((table, columns))
        return _MacroResult(self.fixture["rows"], self.fixture.get("dtype", "FLOAT"))

    def close(self):
        return None


class _MacroTsFileModule:
    TsFileReader = _MacroReader


class _BitcoinSchema:
    def get_columns(self):
        return [
            _Column("series_id", "STRING", "TAG"),
            _Column("series_name", "STRING", "TAG"),
            _Column("start_timestamp", "STRING", "TAG"),
            _Column("target", "FLOAT", "FIELD"),
        ]


class _BitcoinMetadata:
    def get_column_list(self):
        return ["Time", "series_id", "series_name", "start_timestamp", "target"]

    def get_data_type_list(self):
        return ["INT64", "STRING", "STRING", "STRING", "FLOAT"]


class _BitcoinResult(_NullTagResult):
    def __init__(self) -> None:
        self.rows = [
            [1, "bitcoin_0", "BTC", "2009-01-03 00:00:00", float("nan")],
            [2, "bitcoin_0", "BTC", "2009-01-03 00:00:00", 1.5],
        ]
        self.index = -1

    def get_metadata(self):
        return _BitcoinMetadata()


class _BitcoinReader:
    def __init__(self, _path: str) -> None:
        pass

    def get_all_table_schemas(self):
        return {"metrics": _BitcoinSchema()}

    def query_table(self, table, columns, start, end):
        if table != "metrics" or columns != [
            "series_id", "series_name", "start_timestamp", "target"
        ]:
            raise AssertionError((table, columns))
        return _BitcoinResult()

    def close(self):
        return None


class _BitcoinTsFileModule:
    TsFileReader = _BitcoinReader


class _DmiSchema:
    def __init__(self, string_tag: bool) -> None:
        self.string_tag = string_tag

    def get_columns(self):
        return [
            _Column(
                "reference_time",
                "STRING" if self.string_tag else "INT64",
                "TAG" if self.string_tag else "FIELD",
            ),
            _Column("lead_time_hours", "INT64", "FIELD"),
            _Column("value", "DOUBLE", "FIELD"),
        ]


class _DmiResult:
    def __init__(self, rows, columns, types) -> None:
        self.rows = rows
        self.columns = columns
        self.types = types
        self.index = -1

    def get_metadata(self):
        columns = self.columns
        types = self.types

        class Metadata:
            def get_column_list(self):
                return columns

            def get_data_type_list(self):
                return types

        return Metadata()

    def next(self):
        self.index += 1
        return self.index < len(self.rows)

    def is_null_by_index(self, index):
        return self.rows[self.index][index - 1] is None

    def get_value_by_index(self, index):
        return self.rows[self.index][index - 1]

    def close(self):
        return None


class _DmiReader:
    fixtures = {}

    def __init__(self, path: str) -> None:
        self.fixture = self.fixtures[Path(path).as_posix()]

    def get_all_table_schemas(self):
        return {
            self.fixture["table"]: _DmiSchema(self.fixture["string_tag"])
        }

    def query_table(self, table, columns, start, end):
        if table != self.fixture["table"]:
            raise AssertionError(table)
        if columns != ["reference_time", "lead_time_hours", "value"]:
            raise AssertionError(columns)
        types = [
            "INT64",
            "STRING" if self.fixture["string_tag"] else "INT64",
            "INT64",
            "DOUBLE",
        ]
        return _DmiResult(
            self.fixture["rows"],
            ["Time", *columns],
            types,
        )

    def close(self):
        return None


class _DmiTsFileModule:
    TsFileReader = _DmiReader


class CsvValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scope_path = self.root / "scope.json"
        self.policy_path = self.root / "policy.json"
        self.mirror = self.root / "mirror"
        self.output = self.root / "output"
        self.dataset_id = "alpha"
        self.revision = "a" * 40
        self.cap = 100_000
        self.policy_decision = "admit_exact_utc"
        self.candidate_columns = ["c0000"]
        self.retained_columns = ["c0000"]
        self.dropped_columns: list[str] = []
        self.next_trailing_column_total = 0
        self.query_columns = ["value"]
        self.manifest_overrides: dict = {}
        self.scan_overrides: dict = {}
        self._write_inputs()
        self._write_payload_files()
        self._publish_manifests()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_inputs(self) -> None:
        scope = {
            "schema": validator.SCOPE_SCHEMA,
            "scope_id": "pilot-scope",
            "timestamp_dataset_count": 1,
            "timestamp_datasets": [{
                "id": self.dataset_id,
                "repo_id": f"THULab/{self.dataset_id}",
                "revision": self.revision,
            }],
        }
        self.scope_path.write_text(json.dumps(scope), encoding="utf-8")
        scope_sha = sha(self.scope_path)
        policy_row = {
            "dataset_id": self.dataset_id,
            "repo_id": f"THULab/{self.dataset_id}",
            "revision": self.revision,
            "decision": self.policy_decision,
            "reason": "fixture",
            "model": "table",
            "table_name": "metrics",
            "timestamp": {
                "source_column": "Time",
                "source_unit": "us",
                "output_unit": "us",
                "output_column": "timestamp_unix_us",
                "timezone": "UTC",
                "semantic_kind": "unix_epoch_instant",
                "source_unit_verified": True,
                "utc_instant_verified": True,
                "timezone_inferred": False,
            },
            "temporal_string_fields": [],
            "dropped_timestamp_like_string_fields": [],
            "evidence": ["fixture"],
        }
        policy = {
            "schema": validator.POLICY_SCHEMA,
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_sha,
            "decision_counts": {
                "admit_exact_utc": 1,
                "admit_stored_epoch_coordinate": 0,
                "admit_dataset_adapter": 0,
                "block": 0,
            },
            "datasets": [policy_row],
        }
        self.policy_path.write_text(json.dumps(policy), encoding="utf-8")

        dataset_dir = self.mirror / self.dataset_id
        receipt_dir = self.mirror / "_receipts"
        dataset_dir.mkdir(parents=True)
        receipt_dir.mkdir()
        source = dataset_dir / "part.tsfile"
        source.write_bytes(b"TsFile-pilot-TsFile")
        files = [{"path": "part.tsfile", "bytes": source.stat().st_size, "sha256": sha(source)}]
        source_receipt = {
            "schema": validator.SOURCE_RECEIPT_SCHEMA,
            "status": "complete",
            "dataset_id": self.dataset_id,
            "repo_id": f"THULab/{self.dataset_id}",
            "revision": self.revision,
            "files": files,
        }
        source_receipt_path = receipt_dir / f"{self.dataset_id}.json"
        source_receipt_path.write_text(json.dumps(source_receipt, sort_keys=True), encoding="utf-8")
        (self.mirror / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({
            "schema": validator.SOURCE_MANIFEST_SCHEMA,
            "status": "complete",
            "scope_id": scope["scope_id"],
            "scope_sha256": scope_sha,
            "datasets": [{
                "dataset_id": self.dataset_id,
                "repo_id": f"THULab/{self.dataset_id}",
                "revision": self.revision,
            }],
        }), encoding="utf-8")

    def _write_payload_files(self) -> None:
        dataset_dir = self.output / self.dataset_id
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "data.csv").write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [1, 0, 10],
            [2, 0, 20],
        ]))
        (dataset_dir / "series_map.csv").write_bytes(csv_bytes([
            ["series_id", "table_name", "tag_names_json",
             "typed_tag_values_json", "source_paths_json"],
            [0, "metrics", "[]", '["SINGLE:__single_entity__"]', '["part.tsfile"]'],
        ]))
        logical = '["metrics","value","INT64","identity"]'
        (dataset_dir / "columns.csv").write_bytes(csv_bytes([
            ["column", "logical_key_json", "table_name", "source_name",
             "source_type", "output_dtype", "conversion"],
            ["c0000", logical, "metrics", "value", "INT64", "int64", "identity"],
        ]))

    def _manifest_base(self) -> dict:
        scope = json.loads(self.scope_path.read_text())
        policy = json.loads(self.policy_path.read_text())
        policy_row = policy["datasets"][0]
        source_receipt_path = self.mirror / "_receipts" / f"{self.dataset_id}.json"
        source_receipt = json.loads(source_receipt_path.read_text())
        scan = {
            "source_path": "part.tsfile",
            "source_sha256": source_receipt["files"][0]["sha256"],
            "table_name": "metrics",
            "source_action": "materialize",
            "source_rows": 2,
            "eligible_rows": 2,
            "filtered_rows": 0,
            "timestamp_policy_filtered_rows": 0,
            "row_exception_filtered_rows": 0,
            "row_exception": None,
            "null_tag_rows": 0,
            "null_tag_cells": 0,
            "positive_infinity_cells": 0,
            "positive_infinity_by_column": [],
            "nan_cells": 0,
            "nan_by_column": [],
            "row_coverage_exception": None,
            "dataframe_attempted": True,
            "rowwise_fallback": False,
            "query_columns": self.query_columns,
        }
        scan.update(self.scan_overrides)
        base = {
            "schema": validator.RECEIPT_SCHEMA,
            "status": "complete",
            "dataset_id": self.dataset_id,
            "repo_id": f"THULab/{self.dataset_id}",
            "revision": self.revision,
            "scope_id": scope["scope_id"],
            "scope_sha256": sha(self.scope_path),
            "timestamp_policy_file_sha256": sha(self.policy_path),
            "timestamp_policy_sha256": hashlib.sha256(
                validator._canonical_json(policy_row).encode("utf-8")
            ).hexdigest(),
            "timestamp_policy_decision": policy_row["decision"],
            "row_exception_policy_file_sha256": None,
            "row_exception_id": None,
            "source_receipt_sha256": sha(source_receipt_path),
            "source_tsfiles": source_receipt["files"],
            "exporter_sha256": "e" * 64,
            "tsfile_version": validator.REQUIRED_TSFILE_VERSION,
            "row_coverage_query": validator.ROW_COVERAGE_QUERY,
            "null_tag_encoding": validator.NULL_TAG_ENCODING,
            "null_tag_values_preserved": True,
            "positive_infinity_encoding": validator.POSITIVE_INFINITY_ENCODING,
            "positive_infinity_values_preserved": True,
            "nan_encoding": validator.NAN_ENCODING,
            "nan_values_preserved": True,
            "dataframe_batch_rows": 8,
            "sqlite_insert_batch_rows": 8,
            "source_rows": 2,
            "policy_filtered_rows": 0,
            "timestamp_policy_filtered_rows": 0,
            "row_exception_filtered_rows": 0,
            "series_count_discovered": 1,
            "null_tag_rows": 0,
            "null_tag_cells": 0,
            "positive_infinity_cells": 0,
            "source_nan_cells": 0,
            "source_nan_by_column": [],
            "emitted_nan_cells": 0,
            "emitted_nan_by_column": [],
            "source_scans": [scan],
            "ordinary_strings_emitted_to_data_csv": False,
            "series_mapping_is_payload": False,
        }
        base.update(self.manifest_overrides)
        return base

    def _render_dataset_manifest(self, cap: int) -> tuple[bytes, int]:
        dataset_dir = self.output / self.dataset_id
        base = self._manifest_base()
        manifest_size = 0
        for _ in range(10):
            file_rows = []
            other_total = 0
            for name in ("data.csv", "series_map.csv", "columns.csv"):
                path = dataset_dir / name
                other_total += path.stat().st_size
                file_rows.append({
                    "path": name,
                    "bytes_decimal_20": fixed(path.stat().st_size),
                    "sha256": sha(path),
                })
            payload = {
                **base,
                "cap_accounting": "all four files in this dataset directory",
                "cap_bytes_decimal_20": fixed(cap),
                "actual_total_bytes_decimal_20": fixed(other_total + manifest_size),
                "dataset_manifest_bytes_decimal_20": fixed(manifest_size),
                "eligible_rows_decimal_20": fixed(2),
                "emitted_rows_decimal_20": fixed(2),
                "lost_rows_due_to_byte_cap_decimal_20": fixed(0),
                "next_row_bytes_decimal_20": fixed(0),
                "next_row_would_total_bytes_decimal_20": fixed(0),
                "next_trailing_column_would_total_bytes_decimal_20": fixed(
                    self.next_trailing_column_total
                ),
                "candidate_value_column_count_decimal_20": fixed(
                    len(self.candidate_columns)
                ),
                "retained_value_column_count_decimal_20": fixed(
                    len(self.retained_columns)
                ),
                "dropped_trailing_value_column_count_decimal_20": fixed(
                    len(self.dropped_columns)
                ),
                "candidate_value_columns": self.candidate_columns,
                "retained_value_columns": self.retained_columns,
                "dropped_trailing_value_columns": self.dropped_columns,
                "last_retained_value_column": self.retained_columns[-1],
                "first_dropped_trailing_value_column": (
                    self.dropped_columns[0] if self.dropped_columns else None
                ),
                "files": [
                    *file_rows,
                    {
                        "path": "dataset_manifest.json",
                        "bytes_decimal_20": fixed(manifest_size),
                        "sha256": None,
                    },
                ],
            }
            rendered = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
            if len(rendered) == manifest_size:
                return rendered, other_total + manifest_size
            manifest_size = len(rendered)
        raise AssertionError("manifest fixed point failed")

    def _publish_manifests(self, cap: int | None = None) -> None:
        cap = self.cap if cap is None else cap
        dataset_dir = self.output / self.dataset_id
        rendered, total = self._render_dataset_manifest(cap)
        (dataset_dir / "dataset_manifest.json").write_bytes(rendered)
        scope = json.loads(self.scope_path.read_text())
        collection = {
            "schema": validator.COLLECTION_SCHEMA,
            "status": "complete",
            "scope_id": scope["scope_id"],
            "scope_sha256": sha(self.scope_path),
            "timestamp_policy_file_sha256": sha(self.policy_path),
            "row_exception_policy_file_sha256": None,
            "tsfile_version": validator.REQUIRED_TSFILE_VERSION,
            "dataset_byte_cap": cap,
            "dataset_count": 1,
            "datasets": [{
                "dataset_id": self.dataset_id,
                "receipt": f"{self.dataset_id}/dataset_manifest.json",
                "actual_capped_bytes": total,
                "emitted_rows": 2,
                "lost_rows_due_to_byte_cap": 0,
            }],
        }
        (self.output / "CSV_EXPORT_MANIFEST.json").write_text(
            json.dumps(collection, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def _validate(self, **kwargs):
        kwargs.setdefault("tsfile_module", _TsFileModule)
        return validator.validate_export(
            scope_path=self.scope_path,
            mirror=self.mirror,
            policy_path=self.policy_path,
            output=self.output,
            expected_dataset_byte_cap=kwargs.pop("expected_dataset_byte_cap", self.cap),
            require_production_policy_counts=kwargs.pop(
                "require_production_policy_counts", False
            ),
            **kwargs,
        )

    def test_accepts_structurally_complete_numeric_export(self) -> None:
        report = self._validate()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["datasets"][0]["emitted_rows"], 2)

    def test_rejects_nonnumeric_cell_even_when_hashes_are_refreshed(self) -> None:
        data = self.output / self.dataset_id / "data.csv"
        data.write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [1, 0, "ordinary-string"],
            [2, 0, 20],
        ]))
        self._publish_manifests()
        with self.assertRaisesRegex(validator.CsvValidationError, "non-canonical integer"):
            self._validate()

    def test_rejects_nonchronological_rows(self) -> None:
        data = self.output / self.dataset_id / "data.csv"
        data.write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [2, 0, 20],
            [1, 0, 10],
        ]))
        self._publish_manifests()
        with self.assertRaisesRegex(validator.CsvValidationError, "not chronological"):
            self._validate()

    def test_rejects_series_mapping_id_set_mismatch(self) -> None:
        mapping = self.output / self.dataset_id / "series_map.csv"
        mapping.write_bytes(csv_bytes([
            ["series_id", "table_name", "tag_names_json",
             "typed_tag_values_json", "source_paths_json"],
        ]))
        self._publish_manifests()
        with self.assertRaisesRegex(validator.CsvValidationError, "ID set"):
            self._validate()

    def test_rejects_extra_temp_file_and_symlink(self) -> None:
        extra = self.output / self.dataset_id / ".data-candidate.csv"
        extra.write_text("partial", encoding="utf-8")
        with self.assertRaisesRegex(validator.CsvValidationError, "temporary path"):
            self._validate()
        extra.unlink()
        link = self.output / self.dataset_id / "unexpected-link"
        link.symlink_to(self.output / self.dataset_id / "data.csv")
        with self.assertRaisesRegex(validator.CsvValidationError, "symlink"):
            self._validate()

    def test_rejects_actual_bytes_over_declared_cap(self) -> None:
        actual = sum(
            path.stat().st_size
            for path in (self.output / self.dataset_id).iterdir()
        )
        too_small = actual - 1
        self._publish_manifests(too_small)
        with self.assertRaisesRegex(validator.CsvValidationError, "byte cap violated"):
            self._validate(expected_dataset_byte_cap=too_small)

    def test_rejects_fixed_column_count_mismatch(self) -> None:
        manifest_path = self.output / self.dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        original_size = manifest_path.stat().st_size
        manifest["candidate_value_column_count_decimal_20"] = fixed(2)
        rendered = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.assertEqual(len(rendered), original_size)
        manifest_path.write_bytes(rendered)
        with self.assertRaisesRegex(validator.CsvValidationError, "counts do not match"):
            self._validate()

    def test_rejects_source_hash_drift(self) -> None:
        (self.mirror / self.dataset_id / "part.tsfile").write_bytes(b"changed")
        with self.assertRaisesRegex(validator.CsvValidationError, "source file inventory/hash"):
            self._validate()

    def test_rejects_row_exception_without_explicit_policy_file(self) -> None:
        self.manifest_overrides["row_exception_id"] = (
            validator.MACROECON_ROW_EXCEPTION_ID
        )
        self._publish_manifests()
        with self.assertRaisesRegex(
            validator.CsvValidationError,
            "unapproved row exception",
        ):
            self._validate()

    def test_frozen_macro_exception_policy_may_bind_dmi_only_subset(self) -> None:
        digest = "d" * 64
        dataset_id = validator.DMI_DATASET_ID
        receipt = {
            "row_exception_policy_file_sha256": digest,
            "timestamp_policy_filtered_rows": 0,
            "row_exception_filtered_rows": 0,
            "policy_filtered_rows": 0,
            "row_exception_id": None,
            "source_scans": [{
                "source_path": "data/data.tsfile",
                "timestamp_policy_filtered_rows": 0,
                "row_exception_filtered_rows": 0,
                "filtered_rows": 0,
                "row_exception": None,
            }],
        }
        arguments = {
            "manifest": {"row_exception_policy_file_sha256": digest},
            "dataset_ids": [dataset_id],
            "receipts": {dataset_id: receipt},
            "source_tsfiles": {dataset_id: []},
            # Strict parsing and scope/policy SHA validation occurs before this
            # binding function.  A non-None object means that exact frozen
            # policy was supplied; it is intentionally inapplicable to DMI.
            "exception_policy": {"dataset_id": validator.MACROECON_DATASET_ID},
            "exception_policy_sha256": digest,
            "exception_sources": {"unused.tsfile": {}},
        }
        validator._validate_row_exception_bindings(**arguments)

        wrong_manifest = dict(arguments)
        wrong_manifest["manifest"] = {
            "row_exception_policy_file_sha256": "e" * 64
        }
        with self.assertRaisesRegex(
            validator.CsvValidationError, "policy binding mismatch"
        ):
            validator._validate_row_exception_bindings(**wrong_manifest)

        invalid_receipt = json.loads(json.dumps(receipt))
        invalid_receipt["row_exception_filtered_rows"] = 1
        invalid_receipt["policy_filtered_rows"] = 1
        invalid_receipt["source_scans"][0]["row_exception_filtered_rows"] = 1
        invalid_receipt["source_scans"][0]["filtered_rows"] = 1
        invalid_receipt["source_scans"][0]["row_exception"] = {"unexpected": True}
        invalid_subset = dict(arguments)
        invalid_subset["receipts"] = {dataset_id: invalid_receipt}
        with self.assertRaisesRegex(
            validator.CsvValidationError, "unapproved row exception"
        ):
            validator._validate_row_exception_bindings(**invalid_subset)

    def test_macro_export_still_requires_full_exception_receipt_contract(self) -> None:
        digest = "d" * 64
        dataset_id = validator.MACROECON_DATASET_ID
        receipt = {
            "row_exception_policy_file_sha256": digest,
            "timestamp_policy_filtered_rows": 0,
            "row_exception_filtered_rows": 0,
            "policy_filtered_rows": 0,
            "row_exception_id": None,
            "source_scans": [],
        }
        with self.assertRaisesRegex(
            validator.CsvValidationError,
            "macroecon row-exception aggregate receipt mismatch",
        ):
            validator._validate_row_exception_bindings(
                manifest={"row_exception_policy_file_sha256": digest},
                dataset_ids=[dataset_id],
                receipts={dataset_id: receipt},
                source_tsfiles={dataset_id: []},
                exception_policy={"source_receipt_sha256": None},
                exception_policy_sha256=digest,
                exception_sources={},
            )

    def test_production_policy_count_gate_fails_closed(self) -> None:
        with self.assertRaisesRegex(validator.CsvValidationError, "160 datasets"):
            self._validate(require_production_policy_counts=True)

    def test_optional_all_160_gate_rejects_subset_export(self) -> None:
        with self.assertRaisesRegex(validator.CsvValidationError, "exact frozen set"):
            self._validate(require_all_160=True)

    def test_small_native_dataset_pilot_rescan(self) -> None:
        report = self._validate(
            pilot_rescan_dataset_ids=[self.dataset_id],
            tsfile_module=_TsFileModule,
        )
        self.assertEqual(report["pilot_source_rescans"][0]["status"], "passed")

    def test_real_tsfile_215_lowercase_time_metadata_pilot(self) -> None:
        try:
            from importlib.metadata import version
            from tsfile import (
                ColumnCategory,
                ColumnSchema,
                TableSchema,
                Tablet,
                TSDataType,
                TsFileReader,
                TsFileTableWriter,
            )
        except Exception as error:  # pragma: no cover - portable optional dependency
            self.skipTest(f"tsfile fixture dependency unavailable: {error}")
        if version("tsfile") != validator.REQUIRED_TSFILE_VERSION:
            self.skipTest("test is specific to the pinned tsfile 2.1.5 SDK")

        source = self.mirror / self.dataset_id / "part.tsfile"
        source.unlink()
        schema = TableSchema(
            "metrics",
            [ColumnSchema("value", TSDataType.INT64, ColumnCategory.FIELD)],
        )
        with TsFileTableWriter(str(source), schema) as writer:
            tablet = Tablet(["value"], [TSDataType.INT64], 2)
            for index, value in enumerate((10, 20)):
                tablet.add_timestamp(index, index + 1)
                tablet.add_value_by_name("value", index, value)
            writer.write_table(tablet)

        # Lock the real SDK behavior that motivated the normalization: the
        # implicit physical time column is lowercase even though the policy
        # and exporter contract call it ``Time``.
        with TsFileReader(str(source)) as reader:
            result = reader.query_table("metrics", ["value"], -(1 << 63), (1 << 63) - 1)
            try:
                names = [str(item) for item in result.get_metadata().get_column_list()]
                self.assertEqual(names, ["time", "value"])
            finally:
                result.close()

        source_receipt_path = self.mirror / "_receipts" / f"{self.dataset_id}.json"
        source_receipt = json.loads(source_receipt_path.read_text())
        source_receipt["files"] = [{
            "path": "part.tsfile",
            "bytes": source.stat().st_size,
            "sha256": sha(source),
        }]
        source_receipt_path.write_text(
            json.dumps(source_receipt, sort_keys=True), encoding="utf-8"
        )
        self._publish_manifests()
        report = self._validate(
            pilot_rescan_dataset_ids=[self.dataset_id],
            tsfile_module=validator._load_tsfile_module(),
        )
        self.assertEqual(report["pilot_source_rescans"][0]["source_rows"], 2)
        parallel = self._validate(
            validation_workers=2,
            tsfile_module=None,
        )
        self.assertEqual(parallel["validation_workers"], 2)
        self.assertEqual(parallel["nan_source_audits"][0]["dataset_id"], self.dataset_id)

    def test_independent_schema_metadata_pass(self) -> None:
        report = self._validate(
            schema_metadata_pass=True,
            tsfile_module=_TsFileModule,
        )
        schema = report["schema_metadata_pass"][0]
        self.assertEqual(schema["status"], "passed")
        self.assertEqual(schema["candidate_value_column_count"], 1)

    def test_synthetic_5548_column_cap_contract_and_schema_pass(self) -> None:
        self.cap = 900_000
        self.candidate_columns = [f"c{index:04d}" for index in range(5_548)]
        self.retained_columns = ["c0000"]
        self.dropped_columns = self.candidate_columns[1:]
        self.next_trailing_column_total = self.cap + 1
        self.query_columns = [f"v{index:04d}" for index in range(5_548)]
        logical = '["metrics","v0000","DOUBLE","identity"]'
        columns_path = self.output / self.dataset_id / "columns.csv"
        columns_path.write_bytes(csv_bytes([
            ["column", "logical_key_json", "table_name", "source_name",
             "source_type", "output_dtype", "conversion"],
            ["c0000", logical, "metrics", "v0000", "DOUBLE", "float64", "identity"],
        ]))
        self._publish_manifests()
        report = self._validate(
            schema_metadata_pass=True,
            tsfile_module=_WideTsFileModule,
        )
        dataset = report["datasets"][0]
        self.assertEqual(dataset["candidate_value_column_count"], 5_548)
        self.assertEqual(dataset["dropped_value_column_count"], 5_547)
        self.assertLessEqual(dataset["actual_bytes"], self.cap)
        self.assertEqual(
            report["schema_metadata_pass"][0]["candidate_value_column_count"],
            5_548,
        )

        manifest_path = self.output / self.dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        original_size = manifest_path.stat().st_size
        manifest["next_trailing_column_would_total_bytes_decimal_20"] = fixed(self.cap)
        rendered = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.assertEqual(len(rendered), original_size)
        manifest_path.write_bytes(rendered)
        with self.assertRaisesRegex(validator.CsvValidationError, "maximality witness"):
            self._validate()

    def test_requires_pinned_tsfile_215(self) -> None:
        with mock.patch.object(validator, "package_version", return_value="2.1.4"):
            with self.assertRaisesRegex(
                validator.CsvValidationError, "requires tsfile 2.1.5"
            ):
                validator._load_tsfile_module()

    def test_parallel_validation_workers_are_bounded_and_require_runtime_import(self) -> None:
        worker = validator._audit_nan_source_dataset_worker
        self.assertIs(sys.modules[worker.__module__], validator)
        self.assertIs(
            getattr(sys.modules[worker.__module__], worker.__name__),
            worker,
        )
        for workers in (0, 11):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(
                    validator.CsvValidationError, "workers must be in 1..10"
                ):
                    self._validate(validation_workers=workers)
        with self.assertRaisesRegex(
            validator.CsvValidationError, "pinned importable TsFile runtime"
        ):
            self._validate(validation_workers=2, tsfile_module=_TsFileModule)

    def test_rejects_missing_null_tag_preservation_contract(self) -> None:
        manifest_path = self.output / self.dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        original_size = manifest_path.stat().st_size
        manifest["null_tag_encoding"] = "TYPE:none"
        rendered = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.assertEqual(len(rendered), original_size)
        manifest_path.write_bytes(rendered)
        with self.assertRaisesRegex(validator.CsvValidationError, "NULL TAG preservation"):
            self._validate()

    def test_typed_null_tag_is_canonical_and_distinct_from_string_null(self) -> None:
        self.assertEqual(validator._typed_tag("STRING", None), "STRING:null")
        self.assertEqual(validator._typed_tag("STRING", "null"), 'STRING:"null"')

    def test_positive_infinity_is_column_typed_scoped_and_reads_back_positive(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        frozen = {
            "part.tsfile": {
                "sha256": sha(source),
                "rows": 2,
                "positive_infinity_cells": 1,
                "table_name": "metrics",
            }
        }
        self.manifest_overrides = {"positive_infinity_cells": 1}
        self.scan_overrides = {
            "positive_infinity_cells": 1,
            "positive_infinity_by_column": [{
                "source_name": "z_gpp",
                "output_column": "c0000",
                "source_type": "DOUBLE",
                "count": 1,
            }],
            "row_coverage_exception": {
                "exception_id": validator.AFRICA_COVERAGE_EXCEPTION_ID,
                "source_path": "part.tsfile",
                "source_sha256": sha(source),
                "expected_source_rows": 2,
                "verified_source_rows": 2,
                "omitted_query_columns": list(
                    validator.AFRICA_OMITTED_QUERY_COLUMNS
                ),
                "omitted_query_column_schema": (
                    validator._africa_omitted_schema_rows()
                ),
                "omitted_column_contract": (
                    "ordinary_STRING_FIELD_drop_non_numeric_field"
                ),
                "row_coverage_proof": (
                    "frozen_source_sha256_and_exact_source_row_count"
                ),
            },
        }
        dataset_dir = self.output / self.dataset_id
        (dataset_dir / "data.csv").write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [1, 0, "inf"],
            [2, 0, "1.5"],
        ]))
        (dataset_dir / "columns.csv").write_bytes(csv_bytes([
            ["column", "logical_key_json", "table_name", "source_name",
             "source_type", "output_dtype", "conversion"],
            ["c0000", '["metrics","z_gpp","DOUBLE","identity"]',
             "metrics", "z_gpp", "DOUBLE", "float64", "identity"],
        ]))
        self._publish_manifests()
        with (
            mock.patch.object(validator, "AFRICA_DATASET_ID", self.dataset_id),
            mock.patch.object(validator, "AFRICA_FROZEN_SOURCES", frozen),
            mock.patch.object(validator, "AFRICA_POSITIVE_INFINITY_TOTAL", 1),
        ):
            report = self._validate()
        self.assertEqual(report["status"], "passed")
        decoded = float("inf")
        self.assertTrue(decoded > 0 and decoded == float("inf"))

    def test_nan_is_exact_lowercase_float_token_and_other_nonfinite_is_rejected(self) -> None:
        self.assertEqual(
            validator._numeric_cell(
                "nan", "fixture", "float32", allow_nan=True
            ),
            "nan",
        )
        for dtype in ("int32", "int64"):
            with self.assertRaisesRegex(validator.CsvValidationError, "integer"):
                validator._numeric_cell("nan", "fixture", dtype, allow_nan=True)
        for token, pattern in (
            ("nan", "NaN is not approved"),
            ("-inf", "non-finite value is forbidden"),
            ("inf", "positive infinity is not approved"),
            ("NaN", "non-finite value is forbidden"),
            ("-nan", "non-finite value is forbidden"),
        ):
            with self.subTest(token=token):
                with self.assertRaisesRegex(validator.CsvValidationError, pattern):
                    validator._numeric_cell(
                        token,
                        "fixture",
                        "float32",
                        allow_positive_infinity=False,
                        allow_nan=False,
                    )
        with self.assertRaisesRegex(validator.CsvValidationError, "forbidden"):
            validator._numeric_cell(
                "-inf", "fixture", "float32", allow_positive_infinity=True
            )

    def test_true_nan_is_preserved_and_null_remains_empty(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        self.manifest_overrides.update({
            "source_nan_cells": 1,
            "source_nan_by_column": [{
                "output_column": "c0000", "output_dtype": "float32", "count": 1,
            }],
            "emitted_nan_cells": 1,
            "emitted_nan_by_column": [{
                "output_column": "c0000", "output_dtype": "float32", "count": 1,
            }],
        })
        self.scan_overrides.update({
            "nan_cells": 1,
            "nan_by_column": [{
                "source_name": "value", "source_type": "FLOAT",
                "output_column": "c0000", "count": 1,
            }],
        })
        dataset_dir = self.output / self.dataset_id
        (dataset_dir / "data.csv").write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [1, 0, "nan"],
            [2, 0, ""],
        ]))
        (dataset_dir / "columns.csv").write_bytes(csv_bytes([
            ["column", "logical_key_json", "table_name", "source_name",
             "source_type", "output_dtype", "conversion"],
            ["c0000", '["metrics","value","FLOAT","identity"]',
             "metrics", "value", "FLOAT", "float32", "identity"],
        ]))
        _MacroReader.fixtures = {
            "part.tsfile": {
                "table": "metrics",
                "rows": [[1, float("nan")], [2, None]],
            }
        }
        self._publish_manifests()
        report = self._validate(tsfile_module=_MacroTsFileModule)
        self.assertEqual(report["datasets"][0]["emitted_rows"], 2)
        audit = report["nan_source_audits"][0]
        self.assertEqual(audit["nan_cells"], 1)
        self.assertEqual(audit["null_cells"], 1)

    def test_complete_rows_reject_source_nan_converted_to_empty_csv(self) -> None:
        self.manifest_overrides.update({
            "source_nan_cells": 1,
            "source_nan_by_column": [{
                "output_column": "c0000", "output_dtype": "float32", "count": 1,
            }],
            "emitted_nan_cells": 0,
            "emitted_nan_by_column": [],
        })
        self.scan_overrides.update({
            "nan_cells": 1,
            "nan_by_column": [{
                "source_name": "value", "source_type": "FLOAT",
                "output_column": "c0000", "count": 1,
            }],
        })
        self._publish_manifests()
        with self.assertRaisesRegex(
            validator.CsvValidationError, "did not preserve every retained source NaN"
        ):
            self._validate()

    def test_nan_audit_rejects_fabricated_underreported_and_null_as_nan(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        source_row = {"path": "part.tsfile", "bytes": source.stat().st_size, "sha256": sha(source)}
        policy = json.loads(self.policy_path.read_text())["datasets"][0]

        def run(rows, declared_count):
            _MacroReader.fixtures = {"part.tsfile": {"table": "metrics", "rows": rows}}
            receipt = {
                "candidate_value_columns": ["c0000"],
                "source_scans": [{
                    "source_path": "part.tsfile",
                    "source_sha256": sha(source),
                    "source_action": "materialize",
                    "table_name": "metrics",
                    "source_rows": len(rows),
                    "nan_cells": declared_count,
                    "nan_by_column": ([{
                        "source_name": "value", "source_type": "FLOAT",
                        "output_column": "c0000", "count": declared_count,
                    }] if declared_count else []),
                }],
            }
            return validator._audit_nan_sources(
                dataset_ids=[self.dataset_id],
                mirror=self.mirror,
                policies={self.dataset_id: policy},
                receipts={self.dataset_id: receipt},
                source_tsfiles={self.dataset_id: [source_row]},
                exception_sources={},
                tsfile_module=_MacroTsFileModule,
            )

        for name, rows, declared in (
            ("fabricated", [[1, 1.0]], 1),
            ("underreported", [[1, float("nan")]], 0),
            ("null_as_nan", [[1, None]], 1),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    validator.CsvValidationError, "independent source NaN evidence differs"
                ):
                    run(rows, declared)

    def test_nan_audit_rejects_literal_string_nan_in_float_schema(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        _MacroReader.fixtures = {
            "part.tsfile": {"table": "metrics", "rows": [[1, "nan"]]}
        }
        with self.assertRaisesRegex(
            validator.CsvValidationError, "non-numeric NaN"
        ):
            validator._scan_nan_source_exact(
                module=_MacroTsFileModule,
                source_path=source,
                table_name="metrics",
                query_columns=["value"],
                float_columns={"value": "FLOAT"},
            )

    def test_forbush_nan_audit_is_exact_bound_and_never_attempts_dataframe(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        digest = sha(source)
        schema_sha = validator._schema_sha256([("value", "FLOAT", "FIELD")])

        class TrapResult(_MacroResult):
            dataframe_calls = 0

            def read_data_frame(self, _rows):
                type(self).dataframe_calls += 1
                raise AssertionError("Forbush direct-rowwise audit attempted DataFrame")

        class TrapReader(_MacroReader):
            def query_table(self, table, columns, start, end):
                if table != self.fixture["table"] or columns != ["value"]:
                    raise AssertionError((table, columns))
                return TrapResult(
                    self.fixture["rows"], self.fixture.get("dtype", "FLOAT")
                )

        class TrapModule:
            TsFileReader = TrapReader

        TrapReader.fixtures = {
            "part.tsfile": {
                "table": "metrics",
                "rows": [[1, float("nan")], [2, None]],
            }
        }
        policy = json.loads(self.policy_path.read_text())["datasets"][0]
        source_row = {
            "path": "part.tsfile", "bytes": source.stat().st_size, "sha256": digest,
        }
        receipt = {
            "candidate_value_columns": ["c0000"],
            "source_nan_cells": 1,
            "source_scans": [{
                "source_path": "part.tsfile", "source_sha256": digest,
                "source_action": "materialize", "table_name": "metrics",
                "source_rows": 2, "eligible_rows": 2,
                "nan_cells": 1,
                "nan_by_column": [{
                    "source_name": "value", "source_type": "FLOAT",
                    "output_column": "c0000", "count": 1,
                }],
            }],
        }
        with (
            mock.patch.object(validator, "FORBUSH_DATASET_ID", self.dataset_id),
            mock.patch.object(validator, "FORBUSH_SOURCE_PATH", "part.tsfile"),
            mock.patch.object(validator, "FORBUSH_SOURCE_SHA256", digest),
            mock.patch.object(validator, "FORBUSH_TABLE_NAME", "metrics"),
            mock.patch.object(validator, "FORBUSH_SCHEMA_SHA256", schema_sha),
        ):
            report = validator._audit_nan_sources(
                dataset_ids=[self.dataset_id], mirror=self.mirror,
                policies={self.dataset_id: policy},
                receipts={self.dataset_id: receipt},
                source_tsfiles={self.dataset_id: [source_row]},
                exception_sources={}, tsfile_module=TrapModule,
            )[0]
            self.assertEqual(TrapResult.dataframe_calls, 0)
            self.assertEqual(report["nan_cells"], 1)
            self.assertEqual(report["null_cells"], 1)
            self.assertEqual(report["source_query_modes"], [{
                "path": "part.tsfile", "sha256": digest,
                "table_name": "metrics", "source_schema_sha256": schema_sha,
                "query_mode": validator.FORBUSH_NATIVE_QUERY_MODE,
            }])

            wrong_source = dict(source_row, sha256="f" * 64)
            with self.assertRaisesRegex(
                validator.CsvValidationError, "direct-rowwise.*binding differs"
            ):
                validator._audit_nan_sources(
                    dataset_ids=[self.dataset_id], mirror=self.mirror,
                    policies={self.dataset_id: policy},
                    receipts={self.dataset_id: receipt},
                    source_tsfiles={self.dataset_id: [wrong_source]},
                    exception_sources={}, tsfile_module=TrapModule,
                )

    def test_nan_receipt_rows_follow_output_column_then_source_name(self) -> None:
        output_by_key = {
            validator._canonical_json(["metrics", "z", "FLOAT", "identity"]): "c0000",
            validator._canonical_json(["metrics", "a", "FLOAT", "identity"]): "c0001",
        }
        self.assertEqual(
            validator._expected_source_nan_rows(
                table_name="metrics",
                nan_by_name={"a": 2, "z": 1},
                float_columns={"a": "FLOAT", "z": "FLOAT"},
                output_by_key=output_by_key,
            ),
            [
                {"source_name": "z", "source_type": "FLOAT",
                 "output_column": "c0000", "count": 1},
                {"source_name": "a", "source_type": "FLOAT",
                 "output_column": "c0001", "count": 2},
            ],
        )

    def test_nan_audit_applies_exact_macro_first_row_exception(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        digest = sha(source)
        source_row = {"path": "part.tsfile", "bytes": source.stat().st_size,
                      "sha256": digest}
        _MacroReader.fixtures = {"part.tsfile": {
            "table": "metrics",
            "rows": [[validator.INT64_MIN, None], [0, float("nan")]],
        }}
        policy = json.loads(self.policy_path.read_text())["datasets"][0]
        receipt = {
            "candidate_value_columns": ["c0000"],
            "source_nan_cells": 1,
            "source_scans": [{
                "source_path": "part.tsfile", "source_sha256": digest,
                "source_action": "materialize", "table_name": "metrics",
                "source_rows": 2, "eligible_rows": 1,
                "nan_cells": 1,
                "nan_by_column": [{
                    "source_name": "value", "source_type": "FLOAT",
                    "output_column": "c0000", "count": 1,
                }],
            }],
        }
        exception = {
            "path": "part.tsfile", "sha256": digest, "table_name": "metrics",
            "expected_source_rows": 2, "expected_eligible_rows": 1,
            "expected_dropped_rows": 1,
        }
        with mock.patch.object(validator, "MACROECON_DATASET_ID", self.dataset_id):
            reports = validator._audit_nan_sources(
                dataset_ids=[self.dataset_id], mirror=self.mirror,
                policies={self.dataset_id: policy}, receipts={self.dataset_id: receipt},
                source_tsfiles={self.dataset_id: [source_row]},
                exception_sources={"part.tsfile": exception},
                tsfile_module=_MacroTsFileModule,
            )
        self.assertEqual(reports[0]["nan_cells"], 1)
        self.assertEqual(reports[0]["null_cells"], 0)
        self.assertEqual(reports[0]["source_rows_verified"], 2)

    def test_nan_audit_applies_frozen_adapter_filter_before_receipt_counts(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        _MacroReader.fixtures = {"part.tsfile": {
            "table": "metrics",
            "rows": [[1, float("nan")], [100, float("nan")]],
        }}
        rows, eligible, nulls, nan_by_name, positive, negative = (
            validator._scan_nan_source_exact(
                module=_MacroTsFileModule,
                source_path=source,
                table_name="metrics",
                query_columns=["value"],
                float_columns={"value": "FLOAT"},
                row_filter={
                    "kind": "all",
                    "conditions": [{
                        "source_name": "Time", "operator": "int64_gte", "value": 50,
                    }],
                },
            )
        )
        self.assertEqual((rows, eligible, nulls), (2, 1, 0))
        self.assertEqual(nan_by_name, {"value": 1})
        self.assertEqual((positive, negative), ({}, {}))

    def test_cap_truncated_nan_prefix_uses_timestamp_source_and_row_order(self) -> None:
        second = self.mirror / self.dataset_id / "part2.tsfile"
        second.write_bytes(b"TsFile-second-TsFile")
        first = self.mirror / self.dataset_id / "part.tsfile"
        sources = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in (first, second)
        ]
        _MacroReader.fixtures = {
            "part.tsfile": {"table": "metrics", "rows": [
                [1, 1.0], [5, float("nan")], [5, 2.0], [9, float("nan")],
            ]},
            "part2.tsfile": {"table": "metrics", "rows": [
                [2, float("nan")], [5, 3.0], [5, float("nan")],
                [10, float("nan")],
            ]},
        }
        (self.output / self.dataset_id / "data.csv").write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [1, 0, "1"], [2, 0, "nan"], [5, 0, "nan"],
            [5, 0, "2"], [5, 0, "3"],
        ]))
        receipt = {
            "lost_rows_due_to_byte_cap_decimal_20": fixed(3),
            "eligible_rows_decimal_20": fixed(8),
            "emitted_rows_decimal_20": fixed(5),
            "retained_value_columns": ["c0000"],
            "source_nan_by_column": [{
                "output_column": "c0000", "output_dtype": "float32", "count": 5,
            }],
            "emitted_nan_cells": 2,
            "emitted_nan_by_column": [{
                "output_column": "c0000", "output_dtype": "float32", "count": 2,
            }],
        }
        policy = json.loads(self.policy_path.read_text())["datasets"][0]
        columns = {"c0000": {
            "table_name": "metrics", "source_name": "value",
            "source_type": "FLOAT", "output_dtype": "float32",
            "conversion": "identity",
        }}
        report = validator._audit_capped_nan_prefixes(
            dataset_ids=[self.dataset_id], mirror=self.mirror, output=self.output,
            policies={self.dataset_id: policy}, receipts={self.dataset_id: receipt},
            source_tsfiles={self.dataset_id: sources},
            columns_by_dataset={self.dataset_id: columns}, exception_sources={},
            tsfile_module=_MacroTsFileModule,
        )
        self.assertEqual(report, [{
            "dataset_id": self.dataset_id,
            "status": "passed",
            "emitted_rows": 5,
            "cutoff_timestamp_unix_us": 5,
            "rows_before_cutoff": 2,
            "rows_at_cutoff": 4,
            "cutoff_tie_rows_included": 3,
            "retained_source_nan_cells": 5,
            "expected_emitted_nan_cells": 2,
            "expected_emitted_nan_by_column": [{
                "output_column": "c0000", "output_dtype": "float32", "count": 2,
            }],
        }])
        receipt["emitted_nan_cells"] = 1
        receipt["emitted_nan_by_column"][0]["count"] = 1
        with self.assertRaisesRegex(
            validator.CsvValidationError, "exact chronological NaN prefix"
        ):
            validator._audit_capped_nan_prefixes(
                dataset_ids=[self.dataset_id], mirror=self.mirror, output=self.output,
                policies={self.dataset_id: policy}, receipts={self.dataset_id: receipt},
                source_tsfiles={self.dataset_id: sources},
                columns_by_dataset={self.dataset_id: columns}, exception_sources={},
                tsfile_module=_MacroTsFileModule,
            )

    def test_cap_nan_only_in_dropped_column_needs_no_prefix_rescan(self) -> None:
        receipt = {
            "lost_rows_due_to_byte_cap_decimal_20": fixed(1),
            "retained_value_columns": ["c0000"],
            "source_nan_by_column": [{
                "output_column": "c0001", "output_dtype": "float32", "count": 1,
            }],
            "emitted_nan_cells": 0,
            "emitted_nan_by_column": [],
        }
        self.assertEqual(
            validator._audit_capped_nan_prefixes(
                dataset_ids=[self.dataset_id], mirror=self.mirror, output=self.output,
                policies={}, receipts={self.dataset_id: receipt}, source_tsfiles={},
                columns_by_dataset={}, exception_sources={}, tsfile_module=None,
            ),
            [],
        )

    def test_trending_targeted_rescan_preserves_both_null_tag_rows(self) -> None:
        data = self.output / self.dataset_id / "data.csv"
        data.write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000"],
            [1, 1, 10],
            [2, 0, 20],
            [3, 1, 30],
        ]))
        source = self.mirror / self.dataset_id / "part.tsfile"
        source_row = {
            "path": "part.tsfile",
            "bytes": source.stat().st_size,
            "sha256": sha(source),
        }
        policy_row = json.loads(self.policy_path.read_text())["datasets"][0]
        receipt = {
            "source_scans": [{
                "source_path": "part.tsfile",
                "source_sha256": sha(source),
                "source_action": "materialize",
                "source_rows": 3,
                "eligible_rows": 3,
                "filtered_rows": 0,
                "null_tag_rows": 2,
                "null_tag_cells": 2,
                "positive_infinity_cells": 0,
                "positive_infinity_by_column": [],
                "nan_cells": 0,
                "nan_by_column": [],
            }],
            "source_rows": 3,
            "eligible_rows_decimal_20": fixed(3),
            "dropped_trailing_value_columns": [],
            "null_tag_rows": 2,
            "null_tag_cells": 2,
            "positive_infinity_cells": 0,
            "source_nan_cells": 0,
        }
        mapping = {
            0: {
                "table_name": "metrics",
                "tag_names": ["symbol"],
                "typed_tags": ['STRING:"A"'],
                "source_paths": ["part.tsfile"],
            },
            1: {
                "table_name": "metrics",
                "tag_names": ["symbol"],
                "typed_tags": ["STRING:null"],
                "source_paths": ["part.tsfile"],
            },
        }
        columns = {
            "c0000": {
                "logical_key": '["metrics","value","INT64","identity"]',
                "table_name": "metrics",
                "source_name": "value",
                "source_type": "INT64",
                "output_dtype": "int64",
                "conversion": "identity",
            }
        }
        with (
            mock.patch.object(validator, "TRENDING_DATASET_ID", self.dataset_id),
            mock.patch.object(validator, "TRENDING_SOURCE_PATH", "part.tsfile"),
            mock.patch.object(validator, "TRENDING_SOURCE_SHA256", sha(source)),
            mock.patch.object(validator, "TRENDING_SOURCE_ROWS", 3),
            mock.patch.object(validator, "TRENDING_NULL_TAG_ROWS", 2),
        ):
            report = validator._pilot_rescan(
                dataset_id=self.dataset_id,
                mirror=self.mirror,
                output=self.output,
                policy_row=policy_row,
                receipt=receipt,
                source_tsfiles=[source_row],
                mapping=mapping,
                columns=columns,
                max_source_files=1,
                max_source_bytes=1024,
                tsfile_module=_NullTagTsFileModule,
            )
        self.assertEqual(report["source_rows"], 3)
        self.assertEqual(report["null_tag_rows"], 2)
        self.assertEqual(report["null_tag_cells"], 2)

    def test_bitcoin_pilot_matches_true_nan_and_temporal_value_rows(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        source_row = {
            "path": "part.tsfile", "bytes": source.stat().st_size, "sha256": sha(source)
        }
        start_us = validator._convert_temporal_string(
            "2009-01-03 00:00:00",
            {"parse_kind": "strptime", "format": "%Y-%m-%d %H:%M:%S",
             "timezone": "UTC", "output_unit": "us"},
        )
        (self.output / self.dataset_id / "data.csv").write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000", "c0001"],
            [1000, 0, "nan", start_us],
            [2000, 0, "1.5", start_us],
        ]))
        policy_row = {
            "decision": "admit_stored_epoch_coordinate",
            "model": "table",
            "table_name": "metrics",
            "timestamp": {"source_column": "Time", "source_unit": "ms", "output_unit": "us"},
            "temporal_string_fields": [{
                "source_name": "start_timestamp", "source_type": "STRING",
                "source_category": "TAG", "parse_kind": "strptime",
                "format": "%Y-%m-%d %H:%M:%S", "timezone": "UTC", "output_unit": "us",
            }],
            "dropped_timestamp_like_string_fields": [],
        }
        receipt = {
            "source_scans": [{
                "source_path": "part.tsfile", "source_sha256": sha(source),
                "source_action": "materialize", "table_name": "metrics",
                "source_rows": 2, "eligible_rows": 2, "filtered_rows": 0,
                "null_tag_rows": 0, "null_tag_cells": 0,
                "positive_infinity_cells": 0, "positive_infinity_by_column": [],
                "nan_cells": 1,
                "nan_by_column": [{
                    "source_name": "target", "source_type": "FLOAT",
                    "output_column": "c0000", "count": 1,
                }],
                "query_columns": ["series_id", "series_name", "start_timestamp", "target"],
            }],
            "source_rows": 2, "eligible_rows_decimal_20": fixed(2),
            "dropped_trailing_value_columns": [],
            "null_tag_rows": 0, "null_tag_cells": 0,
            "positive_infinity_cells": 0, "source_nan_cells": 1,
        }
        typed_tags = [
            validator._typed_tag("STRING", "bitcoin_0"),
            validator._typed_tag("STRING", "BTC"),
            validator._typed_tag("STRING", "2009-01-03 00:00:00"),
        ]
        mapping = {0: {
            "table_name": "metrics",
            "tag_names": ["series_id", "series_name", "start_timestamp"],
            "typed_tags": typed_tags,
            "source_paths": ["part.tsfile"],
        }}
        columns = {
            "c0000": {
                "logical_key": '["metrics","target","FLOAT","identity"]',
                "table_name": "metrics", "source_name": "target", "source_type": "FLOAT",
                "output_dtype": "float32", "conversion": "identity",
            },
            "c0001": {
                "logical_key": '["metrics","start_timestamp","STRING","policy_temporal_string_to_int64_us"]',
                "table_name": "metrics", "source_name": "start_timestamp",
                "source_type": "STRING", "output_dtype": "int64",
                "conversion": "policy_temporal_string_to_int64_us",
            },
        }
        with (
            mock.patch.object(validator, "BITCOIN_DATASET_ID", self.dataset_id),
            mock.patch.object(validator, "BITCOIN_SOURCE_PATH", "part.tsfile"),
            mock.patch.object(validator, "BITCOIN_SOURCE_SHA256", sha(source)),
            mock.patch.object(validator, "BITCOIN_SOURCE_ROWS", 2),
            mock.patch.object(validator, "BITCOIN_TRUE_NAN_CELLS", 1),
        ):
            report = validator._pilot_rescan(
                dataset_id=self.dataset_id,
                mirror=self.mirror,
                output=self.output,
                policy_row=policy_row,
                receipt=receipt,
                source_tsfiles=[source_row],
                mapping=mapping,
                columns=columns,
                max_source_files=1,
                max_source_bytes=1024,
                tsfile_module=_BitcoinTsFileModule,
            )
        self.assertEqual(report["nan_cells"], 1)
        self.assertEqual(report["float_null_cells"], 0)
        self.assertEqual(report["compared_output_rows"], 2)

    def test_forbush_full_rowwise_rescan_compares_numeric_and_temporal_values(self) -> None:
        source = self.mirror / self.dataset_id / "part.tsfile"
        digest = sha(source)
        schema_columns = [
            ("onset_type", "INT64", "FIELD"),
            ("flux", "DOUBLE", "FIELD"),
            ("cme_datetime_utc", "STRING", "FIELD"),
            ("solar_source_datetime_utc", "STRING", "FIELD"),
        ]
        schema_sha = validator._schema_sha256(schema_columns)
        rows = [
            [1, None, 1.5, "", "2020-01-01 01:00:00"],
            [2, 2, None, "2020-01-02 00:00:00", None],
        ]

        class ForbushSchema:
            def get_columns(self):
                return [_Column(*row) for row in schema_columns]

        class ForbushMetadata:
            def get_column_list(self):
                return ["Time", *(row[0] for row in schema_columns)]

            def get_data_type_list(self):
                return ["INT64", *(row[1] for row in schema_columns)]

        class ForbushResult:
            def __init__(self):
                self.index = -1

            def get_metadata(self):
                return ForbushMetadata()

            def next(self):
                self.index += 1
                return self.index < len(rows)

            def is_null_by_index(self, index):
                return rows[self.index][index - 1] is None

            def get_value_by_index(self, index):
                return rows[self.index][index - 1]

            def close(self):
                return None

        class ForbushReader:
            def __init__(self, _path):
                pass

            def get_all_table_schemas(self):
                return {"metrics": ForbushSchema()}

            def query_table(self, table, columns, start, end):
                self.assert_query = (table, columns, start, end)
                if table != "metrics" or columns != [row[0] for row in schema_columns]:
                    raise AssertionError((table, columns))
                return ForbushResult()

            def close(self):
                return None

        class ForbushModule:
            TsFileReader = ForbushReader

        temporal_rules = [{
            "source_name": name, "source_type": "STRING", "source_category": "FIELD",
            "parse_kind": "strptime", "format": "%Y-%m-%d %H:%M:%S",
            "timezone": "UTC", "output_unit": "us",
        } for name in validator.FORBUSH_TEMPORAL_FIELDS]
        policy = {
            "decision": "admit_stored_epoch_coordinate", "model": "table",
            "table_name": "metrics",
            "timestamp": {
                "source_column": "Time", "source_unit": "ms", "output_unit": "us",
            },
            "temporal_string_fields": temporal_rules,
            "source_bound_temporal_fields": [],
            "dropped_timestamp_like_string_fields": [],
        }
        cme1 = validator._convert_temporal_string(rows[1][3], temporal_rules[0])
        solar0 = validator._convert_temporal_string(rows[0][4], temporal_rules[1])
        (self.output / self.dataset_id / "data.csv").write_bytes(csv_bytes([
            ["timestamp_unix_us", "series_id", "c0000", "c0001", "c0002", "c0003"],
            [1000, 0, 1.5, "", "", solar0],
            [2000, 0, "", 2, cme1, ""],
        ]))
        columns = {
            "c0000": {
                "logical_key": '["metrics","flux","DOUBLE","identity"]',
                "table_name": "metrics", "source_name": "flux",
                "source_type": "DOUBLE", "output_dtype": "float64",
                "conversion": "identity",
            },
            "c0001": {
                "logical_key": '["metrics","onset_type","INT64","identity"]',
                "table_name": "metrics", "source_name": "onset_type",
                "source_type": "INT64", "output_dtype": "int64",
                "conversion": "identity",
            },
            "c0002": {
                "logical_key": '["metrics","cme_datetime_utc","STRING","policy_temporal_string_to_int64_us"]',
                "table_name": "metrics", "source_name": "cme_datetime_utc",
                "source_type": "STRING", "output_dtype": "int64",
                "conversion": "policy_temporal_string_to_int64_us",
            },
            "c0003": {
                "logical_key": '["metrics","solar_source_datetime_utc","STRING","policy_temporal_string_to_int64_us"]',
                "table_name": "metrics", "source_name": "solar_source_datetime_utc",
                "source_type": "STRING", "output_dtype": "int64",
                "conversion": "policy_temporal_string_to_int64_us",
            },
        }
        receipt = {
            "source_scans": [{
                "source_path": "part.tsfile", "source_sha256": digest,
                "source_schema_sha256": schema_sha,
                "source_action": "materialize", "table_name": "metrics",
                "read_mode": validator.FORBUSH_NATIVE_QUERY_MODE,
                "dataframe_attempted": False, "rowwise_fallback": False,
                "source_rows": 2, "eligible_rows": 2, "filtered_rows": 0,
                "null_tag_rows": 0, "null_tag_cells": 0,
                "positive_infinity_cells": 0, "positive_infinity_by_column": [],
                "nan_cells": 0, "nan_by_column": [],
                "temporal_empty_string_audits": [{
                    "source_name": "cme_datetime_utc",
                    "source_type": "STRING",
                    "source_category": "FIELD",
                    "output_column": "c0002",
                    "conversion": "policy_temporal_string_to_int64_us",
                    "representation": (
                        validator.FORBUSH_TEMPORAL_EMPTY_REPRESENTATION
                    ),
                    "source_non_null_empty_count": 1,
                    "source_non_null_empty_row_ordinals": [0],
                    "source_null_count": 0,
                    "source_null_row_ordinals": [],
                    "source_valid_count": 1,
                }, {
                    "source_name": "solar_source_datetime_utc",
                    "source_type": "STRING",
                    "source_category": "FIELD",
                    "output_column": "c0003",
                    "conversion": "policy_temporal_string_to_int64_us",
                    "representation": (
                        validator.FORBUSH_TEMPORAL_EMPTY_REPRESENTATION
                    ),
                    "source_non_null_empty_count": 0,
                    "source_non_null_empty_row_ordinals": [],
                    "source_null_count": 1,
                    "source_null_row_ordinals": [1],
                    "source_valid_count": 1,
                }],
            }],
            "source_rows": 2, "eligible_rows_decimal_20": fixed(2),
            "dropped_trailing_value_columns": [],
            "null_tag_rows": 0, "null_tag_cells": 0,
            "positive_infinity_cells": 0, "source_nan_cells": 0,
        }
        source_row = {
            "path": "part.tsfile", "bytes": source.stat().st_size, "sha256": digest,
        }
        mapping = {0: {
            "table_name": "metrics", "tag_names": [],
            "typed_tags": ["SINGLE:__single_entity__"],
            "source_paths": ["part.tsfile"],
        }}
        with (
            mock.patch.object(validator, "FORBUSH_DATASET_ID", self.dataset_id),
            mock.patch.object(validator, "FORBUSH_SOURCE_PATH", "part.tsfile"),
            mock.patch.object(validator, "FORBUSH_SOURCE_SHA256", digest),
            mock.patch.object(validator, "FORBUSH_TABLE_NAME", "metrics"),
            mock.patch.object(validator, "FORBUSH_SCHEMA_SHA256", schema_sha),
            mock.patch.object(validator, "FORBUSH_SOURCE_ROWS", 2),
            mock.patch.object(validator, "FORBUSH_ALL_NULL_CELLS", 3),
            mock.patch.object(validator, "FORBUSH_CANDIDATE_VALUE_COLUMNS", 4),
            mock.patch.object(validator, "FORBUSH_IDENTITY_NUMERIC_COLUMNS", 2),
            mock.patch.object(validator, "FORBUSH_DOUBLE_COLUMNS", 1),
            mock.patch.object(validator, "FORBUSH_TEMPORAL_VALUE_COUNTS", {
                "cme_datetime_utc": {
                    "source_non_null_empty_count": 1,
                    "source_null_count": 0,
                    "source_valid_count": 1,
                },
                "solar_source_datetime_utc": {
                    "source_non_null_empty_count": 0,
                    "source_null_count": 1,
                    "source_valid_count": 1,
                },
            }),
        ):
            report = validator._pilot_rescan(
                dataset_id=self.dataset_id, mirror=self.mirror, output=self.output,
                policy_row=policy, receipt=receipt, source_tsfiles=[source_row],
                mapping=mapping, columns=columns, max_source_files=1,
                max_source_bytes=1024, tsfile_module=ForbushModule,
            )
            for key, value in (
                ("source_non_null_empty_row_ordinals", [1]),
                ("source_non_null_empty_count", 2),
            ):
                tampered = json.loads(json.dumps(receipt))
                tampered["source_scans"][0]["temporal_empty_string_audits"][0][
                    key
                ] = value
                with self.subTest(tampered=key), self.assertRaisesRegex(
                    validator.CsvValidationError,
                    "producer temporal empty-string audit differs",
                ):
                    validator._pilot_rescan(
                        dataset_id=self.dataset_id,
                        mirror=self.mirror,
                        output=self.output,
                        policy_row=policy,
                        receipt=tampered,
                        source_tsfiles=[source_row],
                        mapping=mapping,
                        columns=columns,
                        max_source_files=1,
                        max_source_bytes=1024,
                        tsfile_module=ForbushModule,
                    )
            for invalid, message in (
                (" ", "source temporal string is empty"),
                ("not-a-time", "source temporal string parse failed"),
            ):
                rows[0][3] = invalid
                with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    validator.CsvValidationError, message
                ):
                    validator._pilot_rescan(
                        dataset_id=self.dataset_id,
                        mirror=self.mirror,
                        output=self.output,
                        policy_row=policy,
                        receipt=receipt,
                        source_tsfiles=[source_row],
                        mapping=mapping,
                        columns=columns,
                        max_source_files=1,
                        max_source_bytes=1024,
                        tsfile_module=ForbushModule,
                    )
        self.assertEqual(report["source_rows"], 2)
        self.assertEqual(report["compared_output_rows"], 2)
        self.assertEqual(report["all_null_cells"], 3)
        self.assertEqual(report["candidate_value_column_count"], 4)
        self.assertEqual(report["compared_numeric_fields"], ["flux", "onset_type"])
        self.assertEqual(
            report["compared_temporal_fields"],
            list(validator.FORBUSH_TEMPORAL_FIELDS),
        )
        self.assertTrue(report["no_rows_or_value_columns_dropped"])
        self.assertEqual(
            report["native_query_audits"][0]["query_mode"],
            validator.FORBUSH_NATIVE_QUERY_MODE,
        )
        self.assertEqual(
            report["temporal_empty_string_audits"],
            [{
                "source_path": "part.tsfile",
                "source_sha256": digest,
                "table_name": "metrics",
                "source_schema_sha256": schema_sha,
                "source_rows": 2,
                "fields": receipt["source_scans"][0][
                    "temporal_empty_string_audits"
                ],
            }],
        )

    def test_dmi_source_bound_integer_and_string_temporal_full_rescan(self) -> None:
        source_rows = [
            (
                "data/data.tsfile",
                "dmi_aarhus_data",
                False,
                [[3_700, 100, 1, 1.5]],
            ),
            (
                "data/training-matrix.tsfile",
                "dmi_aarhus_training_matrix",
                True,
                [[3_800, "1970-01-01T00:03:20Z", 1, 2.5]],
            ),
        ]
        source_tsfiles = []
        frozen = {}
        _DmiReader.fixtures = {}
        for relative, table, string_tag, rows in source_rows:
            path = self.mirror / self.dataset_id / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))
            schema_columns = validator._schema_columns(_DmiSchema(string_tag))
            source = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
            source_tsfiles.append(source)
            reference_seconds = (
                200 if string_tag else 100
            )
            frozen[relative] = {
                "sha256": source["sha256"],
                "bytes": source["bytes"],
                "table_name": table,
                "schema_sha256": validator._schema_sha256(schema_columns),
                "source_type": "STRING" if string_tag else "INT64",
                "source_category": "TAG" if string_tag else "FIELD",
                "rows": 1,
                "reference_epoch_s_min": reference_seconds,
                "reference_epoch_s_max": reference_seconds,
                "time_epoch_s_min": rows[0][0],
                "time_epoch_s_max": rows[0][0],
                "conversion": (
                    "policy_temporal_string_to_int64_us"
                    if string_tag else validator.DMI_SOURCE_BOUND_CONVERSION
                ),
            }
            _DmiReader.fixtures[path.as_posix()] = {
                "table": table,
                "string_tag": string_tag,
                "rows": rows,
            }

        first = source_tsfiles[0]
        policy_row = {
            "decision": "admit_exact_utc",
            "model": "table",
            "table_name": "__single_table__",
            "timestamp": {
                "source_column": "Time", "source_unit": "s", "output_unit": "us",
            },
            "source_bound_temporal_fields": [{
                "source_path": first["path"],
                "source_sha256": first["sha256"],
                "table_name": "dmi_aarhus_data",
                "source_schema_sha256": frozen[first["path"]]["schema_sha256"],
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
                "evidence": ["fixture"],
            }],
            "temporal_string_fields": [{
                "source_name": "reference_time",
                "source_type": "STRING",
                "source_category": "TAG",
                "parse_kind": "strptime",
                "format": "%Y-%m-%dT%H:%M:%SZ",
                "timezone": "UTC",
                "output_unit": "us",
            }],
            "dropped_timestamp_like_string_fields": [],
        }

        logical = {}
        for table, source_type, conversion in (
            ("dmi_aarhus_data", "INT64", validator.DMI_SOURCE_BOUND_CONVERSION),
            ("dmi_aarhus_training_matrix", "STRING", "policy_temporal_string_to_int64_us"),
        ):
            for name, dtype, item_conversion in (
                ("lead_time_hours", "INT64", "identity"),
                ("value", "DOUBLE", "identity"),
                ("reference_time", source_type, conversion),
            ):
                key = validator._canonical_json([table, name, dtype, item_conversion])
                logical[key] = (table, name, dtype, item_conversion)
        ordered = sorted(
            logical.items(), key=lambda item: (item[1][3] != "identity", item[0])
        )
        columns = {}
        for index, (key, (table, name, dtype, conversion)) in enumerate(ordered):
            columns[f"c{index:04d}"] = {
                "logical_key": key,
                "table_name": table,
                "source_name": name,
                "source_type": dtype,
                "output_dtype": (
                    "float64" if dtype == "DOUBLE" and conversion == "identity" else "int64"
                ),
                "conversion": conversion,
            }

        tag_value = "1970-01-01T00:03:20Z"
        entity_keys = {
            "dmi_aarhus_data": validator._canonical_json([
                "dmi_aarhus_data", ["SINGLE:__single_entity__"]
            ]),
            "dmi_aarhus_training_matrix": validator._canonical_json([
                "dmi_aarhus_training_matrix",
                [validator._typed_tag("STRING", tag_value)],
            ]),
        }
        stable_ids = {
            key: index for index, key in enumerate(sorted(entity_keys.values()))
        }
        mapping = {
            stable_ids[entity_keys["dmi_aarhus_data"]]: {
                "table_name": "dmi_aarhus_data",
                "tag_names": [],
                "typed_tags": ["SINGLE:__single_entity__"],
                "source_paths": ["data/data.tsfile"],
            },
            stable_ids[entity_keys["dmi_aarhus_training_matrix"]]: {
                "table_name": "dmi_aarhus_training_matrix",
                "tag_names": ["reference_time"],
                "typed_tags": [validator._typed_tag("STRING", tag_value)],
                "source_paths": ["data/training-matrix.tsfile"],
            },
        }

        data_rows = [["timestamp_unix_us", "series_id", *columns]]
        for table, time_s, reference, lead, value in (
            ("dmi_aarhus_data", 3_700, 100, 1, 1.5),
            ("dmi_aarhus_training_matrix", 3_800, tag_value, 1, 2.5),
        ):
            cells = []
            for meta in columns.values():
                if meta["table_name"] != table:
                    cells.append("")
                elif meta["source_name"] == "lead_time_hours":
                    cells.append(str(lead))
                elif meta["source_name"] == "value":
                    cells.append(str(value))
                elif meta["conversion"] == validator.DMI_SOURCE_BOUND_CONVERSION:
                    cells.append(str(reference * 1_000_000))
                else:
                    cells.append(str(200 * 1_000_000))
            data_rows.append([
                str(time_s * 1_000_000),
                str(stable_ids[entity_keys[table]]),
                *cells,
            ])
        (self.output / self.dataset_id / "data.csv").write_bytes(csv_bytes(data_rows))

        scans = []
        for source, (_relative, table, _string_tag, _rows) in zip(
            source_tsfiles, source_rows
        ):
            scans.append({
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "source_action": "materialize",
                "table_name": table,
                "source_rows": 1,
                "eligible_rows": 1,
                "filtered_rows": 0,
                "null_tag_rows": 0,
                "null_tag_cells": 0,
                "positive_infinity_cells": 0,
                "positive_infinity_by_column": [],
                "nan_cells": 0,
                "nan_by_column": [],
            })
        receipt = {
            "source_scans": scans,
            "source_rows": 2,
            "eligible_rows_decimal_20": fixed(2),
            "dropped_trailing_value_columns": [],
            "null_tag_rows": 0,
            "null_tag_cells": 0,
            "positive_infinity_cells": 0,
            "source_nan_cells": 0,
        }
        with (
            mock.patch.object(validator, "DMI_DATASET_ID", self.dataset_id),
            mock.patch.object(validator, "DMI_FROZEN_SOURCES", frozen),
            mock.patch.object(validator, "DMI_SOURCE_ROWS", 2),
        ):
            report = validator._pilot_rescan(
                dataset_id=self.dataset_id,
                mirror=self.mirror,
                output=self.output,
                policy_row=policy_row,
                receipt=receipt,
                source_tsfiles=source_tsfiles,
                mapping=mapping,
                columns=columns,
                max_source_files=2,
                max_source_bytes=1024,
                tsfile_module=_DmiTsFileModule,
            )
        self.assertEqual(report["source_rows"], 2)
        self.assertEqual(report["compared_output_rows"], 2)
        self.assertEqual(
            [row["conversion"] for row in report["temporal_unit_audits"]],
            [
                validator.DMI_SOURCE_BOUND_CONVERSION,
                "policy_temporal_string_to_int64_us",
            ],
        )
        self.assertTrue(all(
            row["reference_time_nulls"] == 0
            and row["int64_us_overflow_count"] == 0
            and row["lead_relation_mismatch_count"] == 0
            for row in report["temporal_unit_audits"]
        ))

    def test_source_bound_temporal_conversion_rejects_overflow_and_strings(self) -> None:
        rule = {
            "source_type": "INT64", "source_category": "FIELD",
            "parse_kind": "integer_epoch", "source_unit": "s",
            "output_type": "int64", "output_unit": "us", "timezone": "UTC",
        }
        self.assertEqual(
            validator._convert_source_bound_temporal(123, rule), 123_000_000
        )
        for invalid in ("123", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(validator.CsvValidationError, "not an INT64"):
                    validator._convert_source_bound_temporal(invalid, rule)
        with self.assertRaisesRegex(validator.CsvValidationError, "overflows int64"):
            validator._convert_source_bound_temporal(validator.INT64_MAX, rule)

    def test_temporal_schema_mismatch_requires_exact_source_binding(self) -> None:
        schema_columns = [
            ("reference_time", "INT64", "FIELD"),
            ("value", "DOUBLE", "FIELD"),
        ]
        source = {"path": "data/data.tsfile", "sha256": "a" * 64}
        generic = {
            "temporal_string_fields": [{
                "source_name": "reference_time",
                "source_type": "STRING",
                "source_category": "TAG",
            }],
            "source_bound_temporal_fields": [],
        }
        with self.assertRaisesRegex(
            validator.CsvValidationError, "schema mismatch is not source-bound"
        ):
            validator._applicable_temporal_rules(
                dataset_id="fixture",
                policy_row=generic,
                source=source,
                table_name="dmi_aarhus_data",
                schema_columns=schema_columns,
            )

        bound_rule = {
            "source_path": source["path"],
            "source_sha256": "b" * 64,
            "table_name": "dmi_aarhus_data",
            "source_schema_sha256": validator._schema_sha256(schema_columns),
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
            "evidence": ["fixture"],
        }
        with self.assertRaisesRegex(
            validator.CsvValidationError, "source-bound temporal identity differs"
        ):
            validator._applicable_temporal_rules(
                dataset_id="fixture",
                policy_row={
                    **generic,
                    "source_bound_temporal_fields": [bound_rule],
                },
                source=source,
                table_name="dmi_aarhus_data",
                schema_columns=schema_columns,
            )

    def test_africa_projection_exception_is_exact_and_schema_typed(self) -> None:
        ordinary = [
            (name, "STRING", "FIELD")
            for name in validator.AFRICA_OMITTED_QUERY_COLUMNS
        ]
        schema = [
            ("country", "STRING", "TAG"),
            ("value", "DOUBLE", "FIELD"),
            *ordinary,
            ("ascii_note", "STRING", "FIELD"),
        ]
        source = {
            "path": "test/test.tsfile",
            "sha256": validator.AFRICA_FROZEN_SOURCES["test/test.tsfile"]["sha256"],
        }
        query = validator._coverage_query_columns(
            dataset_id=validator.AFRICA_DATASET_ID,
            source=source,
            schema_columns=schema,
        )
        self.assertEqual(query, ["country", "value", "ascii_note"])
        changed = list(schema)
        changed[2] = (validator.AFRICA_OMITTED_QUERY_COLUMNS[0], "TEXT", "FIELD")
        with self.assertRaisesRegex(validator.CsvValidationError, "metadata contract"):
            validator._coverage_query_columns(
                dataset_id=validator.AFRICA_DATASET_ID,
                source=source,
                schema_columns=changed,
            )

    def test_schema_pass_rejects_any_unapproved_omitted_column(self) -> None:
        self.query_columns = []
        self._publish_manifests()
        with self.assertRaisesRegex(validator.CsvValidationError, "full schema"):
            self._validate(
                schema_metadata_pass=True,
                tsfile_module=_TsFileModule,
            )

    def test_pilot_rescan_rejects_adapter(self) -> None:
        policy = json.loads(self.policy_path.read_text())
        policy["datasets"][0]["decision"] = "admit_dataset_adapter"
        policy["decision_counts"] = {
            "admit_exact_utc": 0,
            "admit_stored_epoch_coordinate": 0,
            "admit_dataset_adapter": 1,
            "block": 0,
        }
        self.policy_path.write_text(json.dumps(policy), encoding="utf-8")
        self._publish_manifests()
        with self.assertRaisesRegex(
            validator.CsvValidationError, "adapter policy is missing|unsupported for adapter"
        ):
            self._validate(
                pilot_rescan_dataset_ids=[self.dataset_id],
                tsfile_module=_TsFileModule,
            )

    def _macro_exception_fixture(self):
        macro_dir = self.mirror / validator.MACROECON_DATASET_ID
        macro_dir.mkdir(exist_ok=True)
        sources = []
        exception_sources = {}
        scans = []
        fixtures = {}
        for index, name in enumerate(("macro_a.tsfile", "macro_b.tsfile")):
            path = macro_dir / name
            path.write_bytes(f"macro-{index}".encode())
            digest = sha(path)
            table_name = Path(name).stem
            sources.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})
            exception_sources[name] = {
                "path": name,
                "table_name": table_name,
                "sha256": digest,
                "expected_source_rows": 2,
                "expected_eligible_rows": 1,
                "expected_dropped_rows": 1,
            }
            scans.append({
                "source_path": name,
                "table_name": table_name,
                "source_rows": 2,
                "eligible_rows": 1,
                "filtered_rows": 1,
                "timestamp_policy_filtered_rows": 0,
                "row_exception_filtered_rows": 1,
                "query_columns": ["value"],
                "row_exception": {},
            })
            fixtures[name] = {
                "table": table_name,
                "rows": [[validator.INT64_MIN, None], [0, float(index + 1)]],
            }
        exception_policy = {
            "expected_source_file_count": 2,
            "expected_candidate_value_column_count": 2,
            "expected_source_rows": 4,
            "expected_eligible_rows": 2,
            "expected_dropped_rows": 2,
        }
        policy_row = {
            "decision": "admit_stored_epoch_coordinate",
            "model": "table",
            "table_name": "__single_table__",
            "timestamp": {
                "source_column": "Time",
                "source_unit": "ms",
                "output_unit": "us",
            },
        }
        receipt = {"source_scans": scans}
        return (
            sources,
            exception_sources,
            exception_policy,
            policy_row,
            receipt,
            fixtures,
        )

    def test_macroecon_exception_audit_rescans_all_rows(self) -> None:
        (
            sources,
            exception_sources,
            exception_policy,
            policy_row,
            receipt,
            fixtures,
        ) = self._macro_exception_fixture()
        _MacroReader.fixtures = fixtures
        report = validator._audit_macroecon_row_exception(
            mirror=self.mirror,
            policy_row=policy_row,
            receipt=receipt,
            source_tsfiles=sources,
            exception_policy=exception_policy,
            exception_sources=exception_sources,
            tsfile_module=_MacroTsFileModule,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["source_files"], 2)
        self.assertEqual(report["source_rows"], 4)
        self.assertEqual(report["eligible_rows"], 2)
        self.assertEqual(report["dropped_rows"], 2)
        self.assertEqual(len(report["affected_sources"]), 2)

    def test_macroecon_exception_audit_rejects_all_drift_modes(self) -> None:
        cases = {
            "extra_drop": (
                lambda sources, declared, receipt, fixtures: fixtures["macro_a.tsfile"].update({
                    "rows": [
                        [validator.INT64_MIN, None],
                        [validator.INT64_MIN, None],
                        [0, 1.0],
                    ]
                }),
                "unapproved macroecon INT64_MIN row",
            ),
            "payload": (
                lambda sources, declared, receipt, fixtures: fixtures["macro_a.tsfile"]["rows"][0].__setitem__(1, 9.0),
                "unapproved macroecon INT64_MIN row",
            ),
            "schema": (
                lambda sources, declared, receipt, fixtures: fixtures["macro_a.tsfile"].update({"dtype": "DOUBLE"}),
                "schema is not all FLOAT FIELD",
            ),
            "ordinal": (
                lambda sources, declared, receipt, fixtures: fixtures["macro_a.tsfile"].update({
                    "rows": [[0, 1.0], [validator.INT64_MIN, None]]
                }),
                "unapproved macroecon INT64_MIN row",
            ),
            "hash": (
                lambda sources, declared, receipt, fixtures: sources[0].update({"sha256": "f" * 64}),
                "affected source differs from policy",
            ),
            "count": (
                lambda sources, declared, receipt, fixtures: receipt["source_scans"][0].update({"source_rows": 3}),
                "independent row counts differ",
            ),
        }
        for name, (mutate, pattern) in cases.items():
            with self.subTest(name=name):
                (
                    sources,
                    exception_sources,
                    exception_policy,
                    policy_row,
                    receipt,
                    fixtures,
                ) = self._macro_exception_fixture()
                mutate(sources, exception_sources, receipt, fixtures)
                _MacroReader.fixtures = fixtures
                with self.assertRaisesRegex(validator.CsvValidationError, pattern):
                    validator._audit_macroecon_row_exception(
                        mirror=self.mirror,
                        policy_row=policy_row,
                        receipt=receipt,
                        source_tsfiles=sources,
                        exception_policy=exception_policy,
                        exception_sources=exception_sources,
                        tsfile_module=_MacroTsFileModule,
                    )


if __name__ == "__main__":
    unittest.main()
