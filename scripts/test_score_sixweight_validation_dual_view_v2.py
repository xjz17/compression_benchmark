#!/usr/bin/env python3
from __future__ import annotations

import copy
import random
import unittest

import score_sixweight_validation_dual_view_v2 as scorer


def inputs():
    def metrics(ratio):
        return {"compression_ratio": ratio,
                "compression_ns_per_byte": scorer.E0,
                "decompression_ns_per_byte": scorer.D0}

    specs = (
        ("large-a", 65536, "energy", "A", 0.60, 0.30),
        ("large-b", 131072, "finance", "B", 0.55, 0.40),
        ("small", 1024, "health", "C", 0.65, 0.80),
    )
    baseline_rows, model_rows = [], []
    for sample, raw, domain, dataset, zstd_ratio, model_ratio in specs:
        base = {"sample_id": sample, "B_raw": raw, **metrics(zstd_ratio),
                "independent_cold_decode": True}
        base["J_by_weight"] = {name: scorer.objective(base, weight)
                               for name, weight in scorer.WEIGHTS.items()}
        baseline_rows.append(base)
        for weight in scorer.WEIGHTS.values():
            row = {"sample_id": sample, "B_raw": raw, "weight": list(weight),
                   "domain": domain, "dataset_id": dataset,
                   "size_bucket": "ge64k" if raw >= 65536 else "lt64k",
                   "status": "ok", "error": None, **metrics(model_ratio)}
            row["J_total"] = scorer.objective(row, weight)
            model_rows.append(row)
    return ({"schema": "sixweight-validation-zstd-v1", "test_used": False,
             "cpu": 25, "objective": "a*R+b*E/(E+E0)+c*D/(D+D0)",
             "validation_manifest_sha256": "frozen", "rows": baseline_rows},
            {"schema": "native-combo-greedy-validation-v6",
             "greedy_only_checkpoint_selection": True,
             "validation_candidate_evaluations": 0, "rows": model_rows})


class DeterministicDualViewTest(unittest.TestCase):
    def test_row_order_cannot_change_any_score_leaf(self):
        baseline, validation = inputs()
        expected = scorer.score(baseline, validation, "frozen")
        for seed in range(20):
            shuffled_baseline = copy.deepcopy(baseline)
            shuffled_validation = copy.deepcopy(validation)
            random.Random(seed).shuffle(shuffled_baseline["rows"])
            random.Random(seed + 100).shuffle(shuffled_validation["rows"])
            self.assertEqual(
                scorer.score(shuffled_baseline, shuffled_validation, "frozen"),
                expected)

    def test_postflight_tolerates_only_numerically_harmless_float_drift(self):
        baseline, validation = inputs()
        expected = scorer.score(baseline, validation, "frozen")
        tiny = copy.deepcopy(expected)
        tiny["views"]["all"]["by_weight"]["505"]["relative_J"] += 4.4e-16
        scorer.require_recomputed_summary(tiny, expected)
        material = copy.deepcopy(expected)
        material["views"]["all"]["by_weight"]["505"]["relative_J"] += 1e-7
        with self.assertRaisesRegex(ValueError, "differs materially"):
            scorer.require_recomputed_summary(material, expected)
        identity = copy.deepcopy(expected)
        identity["selection_rule"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs materially"):
            scorer.require_recomputed_summary(identity, expected)
        count = copy.deepcopy(expected)
        count["validation_row_count"] = float(count["validation_row_count"])
        with self.assertRaisesRegex(ValueError, "differs materially"):
            scorer.require_recomputed_summary(count, expected)


if __name__ == "__main__":
    unittest.main()
