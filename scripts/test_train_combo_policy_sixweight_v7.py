#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import train_combo_policy_sixweight_v7 as trainer
from test_train_combo_policy_sixweight_v7_soft_targets import (
    candidate_rows, teacher_records)


class SixWeightV7IntegrationTest(unittest.TestCase):
    def test_admitted_102_template_teacher_reaches_soft_sft_seam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate_measurements.jsonl"
            candidate_path.write_text("".join(
                json.dumps(row) + "\n" for row in candidate_rows()),
                encoding="utf-8")
            search_path = root / "search_space.json"
            search_path.write_text(json.dumps({
                "schema": "native-online-offline-combo-space-v6",
                "train_only_teacher": True,
                "validation_evaluation_labels_written": False,
                "online_candidate_search": False,
                "online_action_space_unchanged": True,
                "strong_path_action_registry_changed": False,
                "reuse_candidate_time_weights": False,
                "approximate_teacher_timing": False,
                "measured_candidate_timing_weights": [list(weight)
                    for weight in trainer.soft.EXPECTED_WEIGHTS],
                "candidate_limit": 102,
                "materialized_candidate_count": 102,
                "plans": [{} for _ in range(102)],
            }), encoding="utf-8")
            data = trainer.admit_soft_teacher(
                teacher_records(), candidate_path, search_path,
                min_teacher_templates=102, top_k=2, regret_temperature=0.05)
            self.assertEqual(data.receipt["materialized_teacher_templates"], 102)
            self.assertEqual(data.receipt["top_k"], 2)
            self.assertEqual(data.targets.shape[1], 6373)
            self.assertFalse(data.receipt["architecture_changed"])
            with self.assertRaisesRegex(ValueError, "102"):
                trainer.admit_soft_teacher(
                    teacher_records(), candidate_path, search_path,
                    min_teacher_templates=103)


if __name__ == "__main__":
    unittest.main()
