#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_combo_teacher_v6 as target
import online_matched_train_v6 as fit3


def action(action_id: int, runtime: str, slot: int | None) -> target.Action:
    return target.Action(action_id, runtime, "A001", "bytes", slot, True,
                         f"{runtime}@{slot}")


def block(split: str = "train", roots: int = 2) -> fit3.TrainingBlock:
    values = bytes(range(64))
    rows = [
        {"kind": "numeric", "width": 8, "data_hex": values.hex(),
         "logical_view": "int64", "original_numeric_payload": True},
    ]
    if roots == 2:
        rows.append({"kind": "serial", "width": 1, "data_hex": "00",
                     "logical_view": "bytes", "original_numeric_payload": False})
    return fit3.TrainingBlock("sample", "dataset", "energy", split, "cal/sample",
                              fit3.digest(b"root"), tuple(rows), 64, 2.0, 3.0)


class FakeBackend:
    model_sha256 = "1" * 64

    def __init__(self, mode: str):
        self.mode = mode
        self.recipe: list[int] = []
        self.open = 0

    def _snapshot(self, terminal: bool = False) -> fit3.Snapshot:
        if terminal:
            return fit3.Snapshot(True, None, (), {
                "terminal": True, "recipe_action_ids": list(self.recipe)})
        if self.mode == "lz4":
            expected = [173, 161, 165]
            ordinary = expected[min(sum(value < fit3.FLOW_STOP_BEGIN
                                        for value in self.recipe), 2)]
            legal = (ordinary,) + tuple(fit3.FLOW_STOP_BEGIN + slot
                                        for slot in range(self.open))
        else:
            if 139 not in self.recipe:
                ordinary = (139,)
            else:
                ordinary = (346 if self.open == 2 else 151,)
            legal = ordinary + tuple(fit3.FLOW_STOP_BEGIN + slot
                                     for slot in range(self.open))
        return fit3.Snapshot(False, np.zeros(fit3.OBSERVATION_DIM),
                             tuple(sorted(legal)), {
                                 "terminal": False,
                                 "open_flow_count": self.open,
                                 "recipe_action_ids": list(self.recipe)})

    def reset(self, item, weight, contract):
        self.recipe = []
        self.open = len(item.roots)
        return self._snapshot()

    def step(self, action_id):
        require_legal = action_id in self._snapshot().legal_action_ids
        if not require_legal:
            raise ValueError(f"illegal fake action {action_id}")
        self.recipe.append(action_id)
        if action_id >= fit3.FLOW_STOP_BEGIN:
            self.open -= 1
        elif self.mode == "multi" and action_id == 139:
            self.open = 2
        if self.open == 0:
            return self._snapshot(True)
        return self._snapshot()


class OfflineComboTeacherV4Test(unittest.TestCase):
    def test_catalog_uses_exact_runtime_action_not_substring_alias(self):
        catalog = target.ExactCatalog.__new__(target.ExactCatalog)
        catalog.actions = {
            170: action(170, "op.entropy_fse.zstd_rans.cpp_reference_v1", 0),
            173: action(173,
                        "op.block_state_control.lz4_matchfinder_residual_core.cpp_v1", 0),
        }
        catalog.excluded_implementation_ids = frozenset()
        self.assertEqual(catalog.exact(
            (170,), "op.entropy_fse.zstd_rans.cpp_reference_v1", target_slot=0), 170)
        with self.assertRaisesRegex(ValueError, "exact stage"):
            catalog.exact((170,), "zstd", target_slot=0)
        with self.assertRaisesRegex(ValueError, "exact stage"):
            catalog.exact((173,), "lz4", target_slot=0)

    def test_frozen_space_names_atomic_stages_truthfully(self):
        plans = target.frozen_plans()
        names = {plan.name for plan in plans}
        self.assertIn("lz4_stage_triplet", names)
        self.assertIn("rans_reference", names)
        self.assertNotIn("zstd", names)
        shuffle = next(plan for plan in plans if plan.name == "shuffle8+rle")
        self.assertEqual(shuffle.prefix[0],
                         "op.block_transform.blosc_shuffle.typesize8.cpp_reference_v1")
        self.assertLessEqual(len(plans), 128)

    def test_lz4_triplet_tracks_primary_after_auxiliary_stop(self):
        catalog = target.ExactCatalog.__new__(target.ExactCatalog)
        catalog.actions = {
            173: action(173, target.LZ4_STAGES[0], 0),
            161: action(161, target.LZ4_STAGES[1], 0),
            165: action(165, target.LZ4_STAGES[2], 0),
        }
        catalog.excluded_implementation_ids = frozenset()
        recipe = target.materialize_plan(
            FakeBackend("lz4"), block(), fit3.OnlineContract(), catalog,
            target.Plan("lz4_stage_triplet", target.LZ4_STAGES))
        self.assertEqual(recipe, (fit3.FLOW_STOP_BEGIN + 1, 173, 161, 165,
                                  fit3.FLOW_STOP_BEGIN))

    def test_multiflow_plan_applies_backend_and_stop_to_every_leaf(self):
        catalog = target.ExactCatalog.__new__(target.ExactCatalog)
        catalog.actions = {
            139: action(139, "op.dictionary.project_words64.cpp_v1", 0),
            346: action(346, "op.run_length.byte_optimal.cpp_v1", 1),
            151: action(151, "op.run_length.byte_optimal.cpp_v1", 0),
        }
        catalog.excluded_implementation_ids = frozenset()
        recipe = target.materialize_plan(
            FakeBackend("multi"), block(roots=1), fit3.OnlineContract(), catalog,
            target.Plan("dictionary_words64+each_rle",
                        ("op.dictionary.project_words64.cpp_v1",),
                        ("op.run_length.byte_optimal.cpp_v1",), True))
        self.assertEqual(recipe, (139, 346, fit3.FLOW_STOP_BEGIN + 1,
                                  151, fit3.FLOW_STOP_BEGIN))

    def test_quality_gate_requires_beating_raw_lz4_and_zstd(self):
        candidate_rows = []
        baselines = []
        for sample in ("a", "b"):
            candidate_rows.append({"sample_id": sample, "candidate": "strong",
                                   "status": "ok", "weight": [1.0, 0.0, 0.0],
                                   "frame_bytes": 40, "original_numeric_bytes": 100,
                                   "compression_ratio": 0.4})
            for method, frame in (("RawNGF1", 110), ("LZ4", 70), ("Zstd", 60)):
                baselines.append({"sample_id": sample, "method": method,
                                  "status": "ok", "frame_bytes": frame,
                                  "original_numeric_bytes": 100,
                                  "compression_ratio": frame / 100})
        proof = target.quality_proof(candidate_rows, baselines)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["common_samples"], 2)
        worse = [dict(row) for row in candidate_rows]
        worse[0]["frame_bytes"] = 100
        worse[0]["compression_ratio"] = 1.0
        self.assertFalse(target.quality_proof(worse, baselines)["passed"])

    def test_reused_weight_timing_is_explicitly_approximate(self):
        item = block()
        measured = fit3.RecipeMeasurement(
            (1, fit3.FLOW_STOP_BEGIN), 32, 64, 100, 80, "1" * 64,
            "2" * 64, "key", True)
        catalog = target.ExactCatalog.__new__(target.ExactCatalog)
        catalog.actions = {1: action(1, "op.run_length.byte_optimal.cpp_v1", 0)}
        catalog.excluded_implementation_ids = frozenset()
        row = target.measurement_row(
            target.Plan("rle", ("op.run_length.byte_optimal.cpp_v1",)),
            measured.recipe, catalog, measured, item, (0.4, 0.3, 0.3),
            measured_timing_weight=(1.0, 0.0, 0.0))
        self.assertTrue(row["time_reused_across_weights"])
        self.assertTrue(row["approximate_teacher_timing"])
        self.assertEqual(row["measured_timing_weight"], [1.0, 0.0, 0.0])
        self.assertTrue(row["in_process_roundtrip"])
        self.assertFalse(row["independent_decode"])


if __name__ == "__main__":
    unittest.main()
