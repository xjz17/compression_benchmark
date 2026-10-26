#!/usr/bin/env python3
"""Unit tests for the non-destructive Core19 numeric/UTC-us materializer."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import tempfile
import unittest

from prepare_project179_core19_numeric_v1 import (
    _json,
    parse_timestamp_us,
    prepare_dataset,
    prepare_scope,
    sha256_file,
)


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark/reference/datasets/project179_core19_utc_policy_20260928.json"
)


class Project179Core19PreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.active = self.root / "active"
        self.backup = self.root / "backup"
        self.output = self.root / "output"
        self.backup.mkdir()
        self.policy = _json(POLICY_PATH)
        self.policy_sha = sha256_file(POLICY_PATH)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_dataset(self, dataset: str, columns: list[tuple[str, str, list[int | float]]], metadata):
        prepared_dir = self.root / "prepared" / dataset
        prepared_dir.mkdir(parents=True)
        prepared_columns = []
        frozen = []
        for source_index, (name, dtype, values) in enumerate(columns):
            path = prepared_dir / f"column_{source_index}.{dtype}.bin"
            pack = struct.Struct("<q" if dtype == "int64" else "<d")
            path.write_bytes(b"".join(pack.pack(value) for value in values))
            digest = sha256_file(path)
            prepared_columns.append(
                {
                    "key": f"{dataset}__col{source_index}",
                    "column_name": name,
                    "source_index": source_index,
                    "dtype": dtype,
                    "status": "ready",
                    "raw_path": str(path),
                    "raw_bytes": path.stat().st_size,
                    "raw_sha256": digest,
                    "retained_rows": len(values),
                    "removed_nan_count": 0,
                }
            )
            frozen.append(
                {
                    "key": f"{dataset}__col{source_index}",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        prepared_manifest = prepared_dir / "manifest.json"
        prepared_manifest.write_text(
            json.dumps({"dataset": dataset, "columns": prepared_columns}), encoding="utf-8"
        )
        metadata_path = self.backup / f"{dataset}_column_info.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["column_index", "column_name", "logical_type", "max_point_number"])
            writer.writerows(metadata)
        active_path = self.active / "datasets" / dataset / "manifest.json"
        active_path.parent.mkdir(parents=True)
        active_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "identity": {
                        "prepared_dataset_manifest_path": str(prepared_manifest),
                        "prepared_dataset_manifest_sha256": sha256_file(prepared_manifest),
                        "ready_raw_columns": frozen,
                        "upstream_metadata_path": str(metadata_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        return active_path

    def backup_csv(self, dataset: str, header, rows) -> Path:
        path = self.backup / f"{dataset}_removed_or_timestamp_columns.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_strict_timestamp_parsers_preserve_microseconds_and_offsets(self) -> None:
        exact = {
            "source_columns": ["x"],
            "format": "%Y-%m-%dT%H:%M:%S.%f%z",
            "timezone_rule": "exact_offset",
        }
        value = parse_timestamp_us(exact, ["2021-06-29T18:54:25.123456-07:00"])
        expected = int(
            (datetime(2021, 6, 30, 1, 54, 25, 123456, tzinfo=timezone.utc)
             - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
            * 1_000_000
        )
        self.assertEqual(value, expected)
        coordinate = {
            "source_columns": ["x"],
            "format": "%Y-%m-%d %H:%M:%S.%f",
            "timezone_rule": "assume_utc_coordinate",
        }
        self.assertEqual(
            parse_timestamp_us(coordinate, ["2013-08-05 21:57:33.963476"]),
            1375739853963476,
        )
        mixed_coordinate = {
            "source_columns": ["x"],
            "formats": ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"],
            "timezone_rule": "assume_utc_coordinate",
        }
        self.assertEqual(
            parse_timestamp_us(mixed_coordinate, ["2013-08-05 21:57:33.963476"]),
            1375739853963476,
        )
        self.assertEqual(
            parse_timestamp_us(mixed_coordinate, ["2013-08-01 17:14:30"]),
            1375377270000000,
        )
        with self.assertRaises(ValueError):
            parse_timestamp_us(mixed_coordinate, ["2013/08/01 17:14:30"])

    def test_timestamp_conversion_string_drop_source_order_and_cap(self) -> None:
        self.make_dataset(
            "Comments",
            [("id", "int64", [1, 2]), ("rate", "float64", [3.5, 4.0]),
             ("date", "int64", [0, 0])],
            [[0, "id", "integer", 0], [2, "rate", "float", 1],
             [3, "date", "string", 0], [4, "comment", "string", 0]],
        )
        self.backup_csv(
            "Comments", ["__row_number__", "date", "comment", "", ""],
            [[1, "2021-06-29T18:54:25-07:00", "a", "", ""],
             [2, "2021-06-29T18:54:26-07:00", "b", "", ""]],
        )
        result = prepare_dataset(
            "Comments", self.active, self.backup, self.output,
            self.policy["datasets"]["Comments"], self.policy_sha,
            limit_bytes=48, chunk_bytes=8, resume=False, verify_only=False,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual([row["source_index"] for row in result["columns"]], [0, 2, 3])
        self.assertTrue(all(row["dtype"] in {"int64", "float64"} for row in result["columns"]))
        self.assertEqual(result["counts"]["timestamp_admitted"], 1)
        self.assertEqual(result["counts"]["timestamp_selected"], 1)
        self.assertEqual(result["counts"]["remaining_strings_dropped"], 1)
        date = next(row for row in result["columns"] if row["column_name"] == "date")
        values = struct.unpack("<qq", Path(date["raw_path"]).read_bytes())
        self.assertEqual(values[1] - values[0], 1_000_000)
        self.assertTrue(all(chunk["bytes"] <= 8 for row in result["columns"] for chunk in row["chunks"]))
        self.assertTrue(all(sum(row["split_rows"].values()) == row["retained_rows"] for row in result["columns"]))

        capped_output = self.root / "capped"
        capped = prepare_dataset(
            "Comments", self.active, self.backup, capped_output,
            self.policy["datasets"]["Comments"], self.policy_sha,
            limit_bytes=40, chunk_bytes=8, resume=False, verify_only=False,
        )
        self.assertEqual([row["source_index"] for row in capped["columns"]], [0, 2])
        self.assertEqual(capped["selection"]["selected_bytes"], 32)
        self.assertEqual(capped["counts"]["timestamp_admitted"], 1)
        self.assertEqual(capped["counts"]["timestamp_selected"], 0)

    def test_prepared_numeric_unix_units_become_exact_int64_microseconds(self) -> None:
        self.make_dataset(
            "Bar-Crawl", [("time", "int64", [0, 1493733882409]), ("x", "float64", [1.0, 2.0])],
            [[0, "time", "integer", 0], [1, "x", "float", 1]],
        )
        result = prepare_dataset(
            "Bar-Crawl", self.active, self.backup, self.output,
            self.policy["datasets"]["Bar-Crawl"], self.policy_sha,
            limit_bytes=64, resume=False, verify_only=False,
        )
        timestamp = result["columns"][0]
        self.assertEqual(timestamp["dtype"], "int64")
        self.assertTrue(timestamp["event_utc_instant_claim"])
        self.assertEqual(
            struct.unpack("<qq", Path(timestamp["raw_path"]).read_bytes()),
            (0, 1493733882409000),
        )

    def test_missing_telco_backup_is_explicitly_blocked_but_numeric_survives(self) -> None:
        self.make_dataset(
            "Telco", [("column_1", "float64", [1.25, 2.5])],
            [[0, "column_0", "string", 0], [1, "column_1", "float", 2],
             [69, "column_69", "string", 0], [70, "column_70", "string", 0],
             [104, "column_104", "string", 0]],
        )
        result = prepare_dataset(
            "Telco", self.active, self.backup, self.output,
            self.policy["datasets"]["Telco"], self.policy_sha,
            limit_bytes=64, resume=False, verify_only=False,
        )
        self.assertEqual(result["status"], "complete_with_blocked_timestamps")
        self.assertEqual(result["counts"]["timestamp_blocked"], 4)
        self.assertEqual([row["column_name"] for row in result["columns"]], ["column_1"])
        self.assertTrue(
            all(row["status"] == "blocked_timestamp_source_missing"
                for row in result["timestamp_receipts"])
        )

    def test_telco_generic_names_preserve_original_indices_after_string_removal(self) -> None:
        # The production Telco numeric metadata is densely renumbered after
        # string removal (column_1 -> metadata index 0, column_73 -> 69).
        # ``column_N`` remains the authoritative original source position.
        self.make_dataset(
            "Telco",
            [
                ("column_1", "float64", [1.25, 2.5]),
                ("column_73", "int64", [7, 8]),
            ],
            [
                [0, "column_1", "float", 2],
                [69, "column_73", "integer", 0],
            ],
        )
        self.backup_csv(
            "Telco",
            ["__row_number__", "column_0", "column_69", "column_70", "column_104"],
            [
                [1, "2020-01-01 00:00:00.000001", "2020-01-02 00:00:00.000002",
                 "2020-01-03 00:00:00.000003", "2020-01-04 00:00:00.000004"],
                [2, "2021-01-01 00:00:00.000001", "2021-01-02 00:00:00.000002",
                 "2021-01-03 00:00:00.000003", "2021-01-04 00:00:00.000004"],
            ],
        )
        result = prepare_dataset(
            "Telco", self.active, self.backup, self.output,
            self.policy["datasets"]["Telco"], self.policy_sha,
            limit_bytes=128, resume=False, verify_only=False,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(
            [row["source_index"] for row in result["columns"]],
            [0, 1, 69, 70, 73, 104],
        )
        self.assertEqual(result["counts"]["timestamp_selected"], 4)
        self.assertEqual(
            {row["column_name"]: row["source_index"] for row in result["columns"]
             if row["source_kind"] == "prepared_numeric"},
            {"column_1": 1, "column_73": 73},
        )

    def test_resume_and_verify_only_fail_closed_on_output_tamper(self) -> None:
        self.make_dataset(
            "Comments", [("id", "int64", [1]), ("date", "int64", [0])],
            [[0, "id", "integer", 0], [3, "date", "string", 0]],
        )
        self.backup_csv(
            "Comments", ["__row_number__", "date"],
            [[1, "2021-06-29T18:54:25-07:00"]],
        )
        first = prepare_scope(
            self.active, self.backup, self.output, POLICY_PATH,
            datasets=["Comments"], limit_bytes=64,
        )
        self.assertEqual(first["totals"]["timestamp_selected"], 1)
        resumed = prepare_scope(
            self.active, self.backup, self.output, POLICY_PATH,
            datasets=["Comments"], limit_bytes=64, resume=True,
        )
        self.assertEqual(resumed["totals"], first["totals"])
        verified = prepare_scope(
            self.active, self.backup, self.output, POLICY_PATH,
            datasets=["Comments"], limit_bytes=64, verify_only=True,
        )
        self.assertEqual(verified["verification_status"], "verified_existing")
        dataset_manifest = _json(self.output / "datasets/Comments/manifest.json")
        victim = Path(dataset_manifest["columns"][0]["raw_path"])
        victim.write_bytes(b"x" * victim.stat().st_size)
        with self.assertRaises(RuntimeError):
            prepare_scope(
                self.active, self.backup, self.output, POLICY_PATH,
                datasets=["Comments"], limit_bytes=64, verify_only=True,
            )
        repaired = prepare_scope(
            self.active, self.backup, self.output, POLICY_PATH,
            datasets=["Comments"], limit_bytes=64, resume=True,
        )
        self.assertEqual(repaired["status"], "complete")


if __name__ == "__main__":
    unittest.main()
