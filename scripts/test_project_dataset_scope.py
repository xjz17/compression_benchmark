#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from project_dataset_scope import (  # noqa: E402
    CORE_CSV_DATASETS,
    DEFAULT_SCOPE_PATH,
    EXPECTED_SCOPE_SHA256,
    PROJECT_DATASETS,
    PROJECT_DATASET_SCOPE,
    TIMESTAMP_DATASETS,
    load_project_dataset_scope,
    sha256_file,
)


class ProjectDatasetScopeTest(unittest.TestCase):
    def test_default_scope_file_is_frozen_by_hash(self) -> None:
        self.assertEqual(sha256_file(DEFAULT_SCOPE_PATH), EXPECTED_SCOPE_SHA256)

    def test_current_scope_is_exactly_179_unique_datasets(self) -> None:
        self.assertEqual(len(CORE_CSV_DATASETS), 19)
        self.assertEqual(len(TIMESTAMP_DATASETS), 160)
        self.assertEqual(len(PROJECT_DATASETS), 179)
        self.assertEqual(len(set(PROJECT_DATASETS)), 179)
        self.assertNotIn("poisson_timestamps", PROJECT_DATASETS)
        self.assertEqual(PROJECT_DATASET_SCOPE["total_dataset_count"], 179)

    def test_every_timestamp_source_is_frozen_to_benchmarkhost_revision(self) -> None:
        rows = PROJECT_DATASET_SCOPE["timestamp_datasets"]
        self.assertEqual(
            {row["repo_id"] for row in rows},
            {f"BenchmarkHost/{dataset}" for dataset in TIMESTAMP_DATASETS},
        )
        self.assertTrue(all(len(row["revision"]) == 40 for row in rows))

    def test_count_or_overlap_tampering_is_rejected(self) -> None:
        value = json.loads(DEFAULT_SCOPE_PATH.read_text(encoding="utf-8"))
        value["timestamp_datasets"][0]["id"] = value["core_csv_datasets"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_project_dataset_scope(path)

    def test_unfrozen_timestamp_revision_is_rejected(self) -> None:
        value = json.loads(DEFAULT_SCOPE_PATH.read_text(encoding="utf-8"))
        value["timestamp_datasets"][0]["revision"] = "main"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid frozen revision"):
                load_project_dataset_scope(path)


if __name__ == "__main__":
    unittest.main()
