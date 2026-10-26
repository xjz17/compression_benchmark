#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from numeric_first_teacher_accuracy_v1 import numeric_first_accuracy


class FakePolicy:
    def forward(self, observations):
        logits = np.zeros((len(observations), 6343), dtype=np.float64)
        for index, observation in enumerate(observations):
            logits[index, 10] = observation[0]
            logits[index, 6341] = -observation[0]
        return logits, ()


def trajectory(sample, weight, feature, numeric_action):
    common = {"sample_id": sample, "weight": weight}
    return [
        {**common, "step": 0, "action_id": 6342,
         "legal_action_ids": [10, 6341, 6342], "observation": [feature]},
        {**common, "step": 1, "action_id": numeric_action,
         "legal_action_ids": [10, 6341], "observation": [feature]},
    ]


class NumericFirstAccuracyTest(unittest.TestCase):
    def test_excludes_bitmap_prefix_and_reports_weighted_accuracy(self):
        records = (trajectory("a", [1.0, 0.0, 0.0], 1.0, 10) +
                   trajectory("b", [0.0, 1.0, 0.0], 1.0, 6341))
        result = numeric_first_accuracy(
            FakePolicy(), records, sample_weights=[1.0, 3.0, 1.0, 1.0])
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["numeric_first_step"], 1)
        self.assertTrue(result["historical_step0_excluded"])
        self.assertEqual(result["action_accuracy"], 0.5)
        self.assertEqual(result["weighted_action_accuracy"], 0.75)
        self.assertEqual(result["operator_action_accuracy"], 1.0)
        self.assertEqual(result["stop_action_accuracy"], 0.0)
        self.assertEqual(len(result["by_weight"]), 2)

    def test_rejects_non_bitmap_step_zero(self):
        records = trajectory("a", [1.0, 0.0, 0.0], 1.0, 10)
        records[0]["action_id"] = 6341
        with self.assertRaisesRegex(ValueError, "bitmap-flow STOP"):
            numeric_first_accuracy(FakePolicy(), records)


if __name__ == "__main__":
    unittest.main()
