#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from derive_core19_1g_scope import derive, sha256_file


class DeriveCoreScopeTest(unittest.TestCase):
    def test_filters_retired_dataset_and_preserves_dataset_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            datasets = [f"d{i:02d}" for i in range(19)]
            selections: dict[str, object] = {}
            entries: list[dict[str, object]] = []
            for index, dataset in enumerate([*datasets, "poisson_timestamps"]):
                manifest_path = source / "datasets" / dataset / "manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "dataset": dataset,
                    "columns": [{"source_index": 0, "raw_bytes": index + 1, "chunks": []}],
                    "totals": {},
                }
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                digest = sha256_file(manifest_path)
                entries.append(
                    {
                        "dataset": dataset,
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": digest,
                        "totals": {
                            "blocking_source_numeric_failures": 0,
                            "chunks": 1,
                            "provenance_only_nonready_entries": 0,
                            "raw_bytes": index + 1,
                            "ready_columns": 1,
                            "retained_rows": 1,
                        },
                    }
                )
                selections[dataset] = {
                    "selected_bytes": index + 1,
                    "selected_columns": 1,
                    "selected_source_indices": [0],
                    "source_columns": 2 if dataset == "d00" else 1,
                }
            top = {
                "version": "paper20-typed-split-v1",
                "identity": {
                    "datasets": [*datasets, "poisson_timestamps"],
                    "preview_selection": {"datasets": selections},
                },
                "preview_protocol": {
                    "limit_bytes": 1 << 30,
                    "selection": selections,
                },
                "datasets": entries,
                "totals": {},
            }
            source_top = source / "manifest.json"
            source_top.write_text(json.dumps(top), encoding="utf-8")
            scope = root / "scope.json"
            scope.write_text(
                json.dumps(
                    {
                        "scope_id": "test-scope",
                        "core_csv_datasets": datasets,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            receipt = derive(
                source_top,
                scope,
                output,
                expected_source_top_sha256=hashlib.sha256(
                    source_top.read_bytes()
                ).hexdigest(),
                verify_all_chunks=False,
            )

            result = json.loads((output / "manifest.json").read_text())
            self.assertEqual(result["identity"]["datasets"], datasets)
            self.assertNotIn(
                "poisson_timestamps",
                result["identity"]["preview_selection"]["datasets"],
            )
            self.assertEqual(result["totals"]["chunks"], 19)
            self.assertEqual(result["derived_scope"]["trimmed_datasets"], ["d00"])
            self.assertEqual(receipt["dataset_count"], 19)
            for dataset in datasets:
                copied = output / "datasets" / dataset / "manifest.json"
                self.assertEqual(
                    sha256_file(copied),
                    result["datasets"][datasets.index(dataset)]["manifest_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
