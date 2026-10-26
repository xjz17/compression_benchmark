#!/usr/bin/env python3
"""Small fixtures for actual data-integrity hazards in streaming preparation."""
import csv
import math
from pathlib import Path
import struct
import tempfile
import unittest

from prepare_exhaustive_graph_columns_v1 import prepare_columns, sha256_file


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.output = Path(self.temp.name) / "out"
        (self.root / "orignial_datasets").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, dataset, header, rows):
        path = self.root / "orignial_datasets" / f"{dataset}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def metadata(self, dataset, rows):
        path = self.root / "backend/dataset" / dataset / f"{dataset}_column_info.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["column_index", "column_name", "logical_type", "max_point_number"])
            writer.writerows(rows)

    def run_dataset(self, dataset):
        return prepare_columns(self.root, self.output, [dataset])

    def test_name_alignment_int64_and_special_float_bits(self):
        source = self.source("Air-Quality", ["timestamp", "b", "a"], [
            ["1078941600000", "-0.0", "9007199254740993"],
            ["1078941600001", "inf", ""], ["1078941600002", "-inf", "-9223372036854775808"],
            ["1078941600003", "nan", "9223372036854775807"],
        ])
        self.metadata("Air-Quality", [[0, "a", "integer", 0], [1, "b", "float", 1], [2, "Date", "string", 0]])
        manifest = self.run_dataset("Air-Quality")
        columns = {column["column_name"]: column for column in manifest["columns"]}
        self.assertEqual(columns["a"]["source_index"], 2)
        self.assertEqual(columns["a"]["catalog_index"], "0")
        self.assertEqual(Path(columns["a"]["raw_path"]).read_bytes(), struct.pack("<qqq", 9007199254740993, -(1 << 63), (1 << 63) - 1))
        self.assertEqual(Path(columns["b"]["raw_path"]).read_bytes(), struct.pack("<ddd", -0.0, math.inf, -math.inf))
        self.assertEqual(columns["a"]["removed_nan_count"], 1)
        self.assertEqual(columns["b"]["raw_bytes"], 24)
        self.assertEqual(columns["timestamp"]["dtype"], "int64")
        self.assertEqual(columns["Date"]["status"], "catalog_column_absent_from_source")
        self.assertEqual(columns["a"]["source_sha256"], sha256_file(source))

    def test_unknown_types_scan_all_rows_and_reject_precision_loss(self):
        self.source("Comments", ["id", "rate", "text", "mixed", "huge"], [
            ["9007199254740993", "1.0", "1", "9007199254740993", "9223372036854775808"],
            ["2", "2.5", "words\nwith newline", "1.5", "3"],
        ])
        columns = {column["column_name"]: column for column in self.run_dataset("Comments")["columns"]}
        self.assertEqual(columns["id"]["kind"], "int64")
        self.assertEqual(columns["rate"]["kind"], "float64")
        self.assertEqual(columns["text"]["status"], "nonnumeric")
        self.assertEqual(columns["mixed"]["status"], "unsupported_mixed_numeric_precision")
        self.assertEqual(columns["huge"]["status"], "unsupported_integer_range")

    def test_strict_invalids_missing_source_and_timestamp_text(self):
        self.source("medium_posts", ["published", "number", "good"], [["2026-09-10", "oops", "1"], ["", "2", "2"]])
        self.metadata("medium_posts", [[0, "published", "timestamp", 0], [1, "number", "integer", 0], [2, "good", "integer", 0]])
        columns = {column["column_name"]: column for column in self.run_dataset("medium_posts")["columns"]}
        self.assertEqual(columns["published"]["status"], "unsupported_timestamp_text")
        self.assertEqual(columns["number"]["status"], "invalid_numeric_value")
        self.assertIsNone(columns["number"]["raw_path"])
        self.assertEqual(columns["good"]["status"], "ready")
        self.metadata("Telco", [[0, "x", "float", 0]])
        self.assertEqual(self.run_dataset("Telco")["columns"][0]["status"], "missing_source")

    def test_resume_validates_raw_and_source_hashes(self):
        source = self.source("Comments", ["id"], [["9007199254740993"], ["2"]])
        first = self.run_dataset("Comments")
        path = Path(first["columns"][0]["raw_path"])
        stamp = path.stat().st_mtime_ns
        self.run_dataset("Comments")
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        path.write_bytes(b"x" * 16)
        self.run_dataset("Comments")
        self.assertEqual(path.read_bytes(), struct.pack("<qq", 9007199254740993, 2))
        self.source("Comments", ["id"], [["3"]])
        changed = self.run_dataset("Comments")["columns"][0]
        self.assertEqual(changed["raw_bytes"], 8)
        self.assertEqual(changed["source_sha256"], sha256_file(source))
        self.assertEqual(Path(changed["raw_path"]).read_bytes(), struct.pack("<q", 3))

    def test_malformed_csv_preserves_failed_column_inventory(self):
        self.source("Comments", ["a", "b"], [["1", "2"], ["3", "4", "extra"]])
        result = self.run_dataset("Comments")
        self.assertEqual(result["status"], "complete_with_issues")
        self.assertEqual([column["column_name"] for column in result["columns"]], ["a", "b"])
        self.assertTrue(all(column["status"] == "preparation_failed" for column in result["columns"]))

    def test_stale_string_metadata_infers_numeric_source_without_float_cast(self):
        self.source("Comments", ["date", "comment"], [["9007199254740993", "123"], ["1625018065000", "text"]])
        self.metadata("Comments", [[0, "date", "string", 0], [1, "comment", "string", 0]])
        columns = {column["column_name"]: column for column in self.run_dataset("Comments")["columns"]}
        self.assertEqual(columns["date"]["status"], "ready")
        self.assertEqual(columns["date"]["kind"], "int64")
        self.assertEqual(columns["date"]["catalog_logical_type"], "string")
        self.assertEqual(columns["date"]["type_source"], "full_source_inference")
        self.assertEqual(Path(columns["date"]["raw_path"]).read_bytes(), struct.pack("<qq", 9007199254740993, 1625018065000))
        self.assertEqual(columns["comment"]["status"], "nonnumeric")


if __name__ == "__main__":
    unittest.main()
