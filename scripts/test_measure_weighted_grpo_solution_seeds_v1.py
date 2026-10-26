#!/usr/bin/env python3
"""Focused unit tests for training-only solution-seed measurement helpers."""
from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import tracemalloc
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import measure_weighted_grpo_solution_seeds_v1 as seeds  # noqa: E402
from weight_conditioned_search_v1 import CandidateSummary  # noqa: E402


class SeedMeasurementHelpersTest(unittest.TestCase):
    def _dataset(self, root, dataset_id, records):
        root.mkdir(parents=True, exist_ok=True)
        blocks = []
        for index, record in enumerate(records):
            values = record.get("payload", struct.pack(
                "<II", record.get("marker", index), record.get("bucket", 0)
            ))
            bitmap = b"\x00"
            values_path = f"{index}.values"
            bitmap_path = f"{index}.bitmap"
            (root / values_path).write_bytes(values)
            (root / bitmap_path).write_bytes(bitmap)
            blocks.append({
                "split": record.get("split", "train"),
                "dtype": record.get("dtype", "uint32"),
                "entity_id": record.get("entity_id", "e"),
                "column": record.get("column", "value"),
                "block_index": record.get("block_index", index),
                "values_path": values_path,
                "missing_bitmap_path": bitmap_path,
                "values_sha256": seeds._sha256(values),
                "missing_bitmap_sha256": seeds._sha256(bitmap),
                "original_numeric_bytes": len(values),
                "prepared_block_bytes": len(values) + len(bitmap),
            })
        return SimpleNamespace(
            root=root, manifest_path=root / "dataset_manifest.json",
            manifest={
                "dataset_id": dataset_id,
                "cohort": {"role": "train_validation"},
                "provenance_sha256": seeds._sha256(dataset_id.encode()),
                "blocks": blocks,
            },
        )

    @staticmethod
    def _bucket(dtype, values, bitmap):
        bucket = struct.unpack_from("<I", values, 4)[0]
        return f"bucket-{bucket:03d}", (bucket,)

    @staticmethod
    def _former_selection(datasets, limit):
        rows = []
        for dataset in datasets:
            for block in dataset.manifest["blocks"]:
                if block["split"] != "train":
                    continue
                values = (dataset.root / block["values_path"]).read_bytes()
                bitmap = (dataset.root / block["missing_bitmap_path"]).read_bytes()
                bucket, features = SeedMeasurementHelpersTest._bucket(
                    block["dtype"], values, bitmap
                )
                rows.append({
                    "dataset": dataset, "block": block, "dtype": block["dtype"],
                    "values": values, "bitmap": bitmap,
                    "feature_bucket_id": bucket, "feature_vector": list(features),
                })
        rows.sort(key=lambda row: (
            row["dtype"], row["feature_bucket_id"],
            str(row["dataset"].manifest["dataset_id"]),
            str(row["block"].get("entity_id", "")),
            str(row["block"].get("column", "")),
            int(row["block"].get("block_index", -1)),
        ))
        selected = []
        seen = set()
        counts = {}
        for row in rows:
            dtype = row["dtype"]
            key = (dtype, row["feature_bucket_id"])
            if key not in seen and counts.get(dtype, 0) < limit:
                selected.append(row)
                seen.add(key)
                counts[dtype] = counts.get(dtype, 0) + 1
        return selected

    def _select(self, datasets, limit):
        by_path = {dataset.manifest_path: dataset for dataset in datasets}
        with mock.patch.object(
            seeds.experiment, "_load_prepared_dataset",
            side_effect=lambda path, **kwargs: by_path[path],
        ) as loader, mock.patch.object(
            seeds.factory, "solution_library_feature_bucket", new=self._bucket
        ):
            # Keep the loader assertion separate from source-verification tests:
            # these fixtures isolate representative selection, not manifest I/O.
            result = seeds._candidate_blocks(
                list(by_path), max_blocks_per_dtype=limit
            )
            loader.assert_has_calls([
                mock.call(path, verify_source=True, expected_role="train_validation")
                for path in by_path
            ])
        return result

    def test_streaming_selection_matches_full_sort_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = [
                self._dataset(root / "z", "z", [
                    {"bucket": 9}, {"bucket": 5}, {"bucket": 3},
                    {"bucket": 2, "entity_id": "z"},
                    {"bucket": 2, "entity_id": "a", "block_index": 8},
                    {"bucket": 2, "entity_id": "a", "block_index": 1},
                    {"bucket": 0, "split": "validation"},
                    {"bucket": 8, "dtype": "int32"},
                    {"bucket": 7, "dtype": "int32"},
                ]),
                self._dataset(root / "a", "a", [
                    {"bucket": 2, "marker": 90},
                    {"bucket": 1}, {"bucket": 9},
                    {"bucket": 1, "dtype": "int32"},
                ]),
            ]
            for limit in (1, 2, 4, 20):
                with self.subTest(limit=limit):
                    selected, provenance = self._select(datasets, limit)
                    self.assertEqual(selected, self._former_selection(datasets, limit))
                    self.assertEqual([row["dataset_id"] for row in provenance], ["a", "z"])
                    self.assertEqual([row["verified_train_blocks"] for row in provenance], [4, 8])
                    for row, dataset in zip(provenance, reversed(datasets)):
                        self.assertEqual(row["manifest_path"], str(dataset.manifest_path.resolve()))
                        self.assertEqual(row["provenance_sha256"], dataset.manifest["provenance_sha256"])

    def test_exact_tie_retains_first_payload_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._dataset(root / "first", "same", [
                {"bucket": 0, "block_index": 1, "marker": 100},
                {"bucket": 0, "block_index": 1, "marker": 200},
            ])
            second = self._dataset(root / "second", "same", [
                {"bucket": 0, "block_index": 1, "marker": 300},
            ])
            for datasets, marker in (([first, second], 100), ([second, first], 300)):
                selected, _ = self._select(datasets, 1)
                self.assertEqual(selected, self._former_selection(datasets, 1))
                self.assertEqual(struct.unpack_from("<I", selected[0]["values"])[0], marker)

    def test_discarded_blocks_still_require_intact_payload_and_sizes(self):
        cases = (
            ("values_sha256", "wrong", "fingerprint mismatch"),
            ("missing_bitmap_sha256", "wrong", "fingerprint mismatch"),
            ("original_numeric_bytes", 1, "B_raw"),
            ("prepared_block_bytes", 1, "prepared_block_bytes"),
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), "d", [
                {"bucket": 0}, {"bucket": 9},
            ])
            discarded = dataset.manifest["blocks"][1]
            for field, bad_value, error in cases:
                with self.subTest(field=field), mock.patch.dict(discarded, {field: bad_value}):
                    with self.assertRaisesRegex(seeds.SeedMeasurementError, error):
                        self._select([dataset], 1)

    def test_validation_payload_is_not_used_for_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), "d", [
                {"bucket": 9}, {"bucket": 0, "split": "validation"},
            ])
            dataset.manifest["blocks"][1]["values_path"] = "not-read.values"
            selected, provenance = self._select([dataset], 1)
            self.assertEqual(selected[0]["feature_bucket_id"], "bucket-009")
            self.assertEqual(provenance[0]["verified_train_blocks"], 1)

    def test_candidate_payload_memory_is_bounded_by_selected_representatives(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), "d", [
                {"payload": struct.pack("<II", index, index % 12) + bytes(128 * 1024 - 8)}
                for index in range(72)
            ])
            tracemalloc.start()
            try:
                selected, _ = self._select([dataset], 2)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual(len(selected), 2)
            # The former full-cohort list retained >9 MiB of values here.
            # Allow ample overhead beyond two retained blocks plus one current.
            self.assertLess(peak, 2 * 1024 * 1024)

    def test_best_nonempty_recipe_excludes_stop_only_and_is_deterministic(self):
        rows = (
            CandidateSummary((), 0.01, 0.01, 1.0, 1.0, 1),
            CandidateSummary((8,), 0.3, 0.2, 2.0, 2.0, 20),
            CandidateSummary((7,), 0.3, 0.2, 2.0, 2.0, 19),
            CandidateSummary((6,), 0.4, 0.1, 1.0, 1.0, 10),
        )
        self.assertEqual(seeds._best_nonempty_recipe(rows), (7,))

    def test_no_nonempty_recipe_fails_closed(self):
        with self.assertRaisesRegex(seeds.SeedMeasurementError, "non-empty"):
            seeds._best_nonempty_recipe((
                CandidateSummary((), 0.1, 0.1, 1.0, 1.0, 1),
            ))

    def test_measurement_digest_excludes_only_its_own_field(self):
        row = {"dataset_id": "d", "recipe_action_ids": [1, 2], "value": 3}
        first = seeds._measurement_digest(row)
        row["measurement_sha256"] = "stale"
        self.assertEqual(first, seeds._measurement_digest(row))
        row["value"] = 4
        self.assertNotEqual(first, seeds._measurement_digest(row))

    def test_seed_search_separates_path_depth_from_total_graph_nodes(self):
        self.assertEqual(seeds.SEARCH_DEPTH, 10)
        self.assertEqual(
            seeds.SEARCH_MAX_GRAPH_ACTIONS,
            seeds.graph_task.MAX_GRAPH_NODES,
        )
        self.assertGreater(seeds.SEARCH_MAX_GRAPH_ACTIONS, seeds.SEARCH_DEPTH)
        self.assertEqual(seeds.SEARCH_TIMING_WARMUPS, 1)
        self.assertEqual(seeds.SEARCH_TIMING_REPETITIONS, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
