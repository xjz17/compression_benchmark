#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import train_combo_policy_v4 as target
import online_matched_train_v3_fit3 as core


def block(sample, bucket, dataset="dataset", domain="energy", raw=65536):
    return SimpleNamespace(sample_id=sample, bucket=bucket, dataset_id=dataset,
                           domain=domain, split="train",
                           original_numeric_bytes=raw)


def validation_rows(*, failed_pair=None):
    rows = []
    specs = (("large-a", 65536), ("large-b", 131072), ("small", 1024))
    for sample, raw in specs:
        for weight in core.FIXED_WEIGHTS:
            key = (sample, tuple(weight))
            value = 1.0
            if tuple(weight) == (1.0, 0.0, 0.0):
                value = {"large-a": 0.2, "large-b": 0.4, "small": 99.0}[sample]
            elif tuple(weight) == (0.4, 0.3, 0.3):
                value = {"large-a": 0.4, "large-b": 0.6, "small": 99.0}[sample]
            rows.append({"sample_id": sample, "B_raw": raw, "weight": list(weight),
                         "status": "failed" if key == failed_pair else "ok",
                         "J_total": value, "operator_count": 0,
                         "compression_ratio": 0.5,
                         "compression_ns_per_byte": 2.0,
                         "decompression_ns_per_byte": 1.0})
    return rows


def teacher_records(samples):
    rows = []
    for sample in samples:
        for weight in core.FIXED_WEIGHTS:
            recipe = (170, core.FLOW_STOP_BEGIN)
            for step, action in enumerate(recipe):
                rows.append({"sample_id": sample, "weight": list(weight),
                             "offline_candidate": "rans_reference",
                             "teacher_model_sha256": "a" * 64,
                             "recipe_action_ids": list(recipe), "step": step,
                             "action_id": action,
                             "teacher_metrics": {"compression_ratio": 0.5,
                                                 "compression_ns_per_byte": 2.0,
                                                 "decompression_ns_per_byte": 1.0}})
    return rows


class TrainComboPolicyV4Test(unittest.TestCase):
    def test_supervised_weights_balance_trajectories_not_tokens(self):
        blocks = (block("a", "small", "A"), block("b", "small", "A"),
                  block("c", "large", "B"))
        records = ([{"sample_id": "a", "weight": [1.0, 0.0, 0.0]}] +
                   [{"sample_id": "b", "weight": [0.4, 0.3, 0.3]}] * 3 +
                   [{"sample_id": "c", "weight": [1.0, 0.0, 0.0]}] * 2)
        weights, receipt = target.supervised_weights(records, blocks)
        self.assertAlmostEqual(float(weights.sum()), 6.0)
        self.assertAlmostEqual(float(weights[0]), 1.5)
        self.assertAlmostEqual(float(weights[1:4].sum()), 1.5)
        self.assertAlmostEqual(float(weights[4:].sum()), 3.0)
        self.assertEqual(receipt["trajectories"], 3)

    def test_size_first_sampler_gives_occupied_buckets_equal_mass(self):
        blocks = (block("s1", "small", "A", "d1"),
                  block("s2", "small", "A", "d1"),
                  block("l1", "large", "B", "d2"))
        sampler = target.SizeFirstSampler(blocks)
        rng = np.random.default_rng(17)
        counts = {"small": 0, "large": 0}
        for _ in range(10000):
            counts[sampler.sample(rng).bucket] += 1
        self.assertLess(abs(counts["small"] / 10000 - 0.5), 0.025)
        self.assertEqual(set(sampler.receipt()), {"large", "small"})

    def test_summary_uses_only_large_100_and_433_with_equal_weight(self):
        result = target.summarize(validation_rows())
        self.assertAlmostEqual(result["selection_score"], 0.4)
        self.assertEqual(result["primary_samples"], 2)
        self.assertEqual(result["primary_rows"], 4)
        self.assertEqual(set(result["by_weight"]), set(target.WEIGHT_NAMES.values()))
        self.assertTrue(result["selection_eligible"])

    def test_summary_rejects_missing_matrix_and_marks_any_failure_ineligible(self):
        rows = validation_rows()
        with self.assertRaisesRegex(ValueError, "six-weight matrix"):
            target.summarize(rows[:-1])
        failed = target.summarize(validation_rows(
            failed_pair=("small", (0.0, 0.0, 1.0))))
        self.assertEqual(failed["failed_rows"], 1)
        self.assertFalse(failed["selection_eligible"])

    def test_strong_teacher_requires_complete_consistent_six_weight_trajectories(self):
        blocks = (block("a", "small"), block("b", "large"))
        records = teacher_records(("a", "b"))
        receipt = target.validate_strong_teacher(records, blocks)
        self.assertEqual(receipt["trajectories"], 12)
        self.assertEqual(receipt["ordinary_length_histogram"], {"1": 12})
        broken = [dict(row) for row in records]
        broken[1] = {**broken[1], "action_id": core.FLOW_STOP_BEGIN + 1}
        with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
            target.validate_strong_teacher(broken, blocks)
        old = [dict(row) for row in records]
        old[0].pop("offline_candidate")
        with self.assertRaisesRegex(ValueError, "not a strong v4"):
            target.validate_strong_teacher(old, blocks)

    def test_strong_teacher_evidence_binds_quality_and_train_only_search(self):
        contract = core.OnlineContract()
        blocks = (block("a", "large"),)
        records = teacher_records(("a",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher.jsonl"
            teacher.write_text("{}\n", encoding="utf-8")
            search = {"schema": "native-online-offline-combo-space-v4",
                      "train_only_teacher": True,
                      "validation_evaluation_labels_written": False,
                      "online_candidate_search": False}
            target.write(root / "search_space.json", search)
            summary = {"schema": "native-online-offline-combo-teacher-v4",
                       "completed": True, "validation_evaluated": False,
                       "teacher_records": len(records),
                       "quality_gate": {"status": "complete", "passed": True,
                                        "common_samples": 1}}
            target.write(root / "summary.json", summary)
            frame = root / "winner.ngf1"
            frame.write_bytes(b"test frame")
            cold = {"schema": "native-online-winner-cold-decode-v4",
                    "sample_id": "a", "recipe_action_ids": [170, core.FLOW_STOP_BEGIN],
                    "independent_process": True, "bit_exact_roots": True,
                    "original_numeric_bytes": 65536, "frame_path": str(frame),
                    "frame_sha256": core.digest(frame.read_bytes()),
                    "frame_bytes": frame.stat().st_size}
            cold_path = root / "winner_cold_decode_receipts.jsonl"
            cold_path.write_text(json.dumps(cold) + "\n", encoding="utf-8")
            receipt = {"schema": "native-online-offline-combo-teacher-receipt-v4",
                       "teacher_sha256": core.digest(teacher.read_bytes()),
                       "records": len(records), "train_samples": 1,
                       "validation_records": 0,
                       "unique_winner_frames_cold_decoded": 1,
                       "winner_cold_decode_receipts_sha256": core.digest(cold_path.read_bytes()),
                       "contract_sha256": contract.fingerprint,
                       "search_space_sha256": core.digest(
                           (root / "search_space.json").read_bytes())}
            target.write(root / "teacher.receipt.json", receipt)
            evidence = target.load_strong_teacher_evidence(
                teacher, records, blocks, contract)
            self.assertEqual(evidence["search_space_sha256"],
                             receipt["search_space_sha256"])
            summary["quality_gate"]["passed"] = False
            target.write(root / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "quality/coverage"):
                target.load_strong_teacher_evidence(teacher, records, blocks, contract)
            summary["quality_gate"]["passed"] = True
            target.write(root / "summary.json", summary)
            frame.write_bytes(b"corrupted frame")
            with self.assertRaisesRegex(ValueError, "frame changed"):
                target.load_strong_teacher_evidence(teacher, records, blocks, contract)


if __name__ == "__main__":
    unittest.main()
