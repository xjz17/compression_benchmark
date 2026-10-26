#!/usr/bin/env python3
"""Fail-closed tests for the v7 six-weight pre-GPU oracle gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parent


def load_controller():
    helper = types.ModuleType("run_sixweight_campaign_helpers_v2")

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    helper.require = require
    helper.sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sys.modules[helper.__name__] = helper
    spec = importlib.util.spec_from_file_location(
        "run_sixweight_pretrain_v7_under_test",
        ROOT / "run_sixweight_pretrain_v7.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TARGET = load_controller()


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class SixWeightOracleGateTest(unittest.TestCase):
    def make_measurements(self, teacher: Path, *, fail_505=False, reused=False):
        candidates = []
        baselines = []
        for index, raw in enumerate((1000, 3000)):
            sample = f"train-{index}"
            baseline = {
                "method": "Zstd", "status": "ok", "sample_id": sample,
                "original_numeric_bytes": raw, "compression_ratio": 0.5,
                "compression_ns_per_byte": 20.0,
                "decompression_ns_per_byte": 3.0,
                "independent_decode": True,
                "timing_scope": "direct_native_complete_method_no_mlp",
            }
            baselines.append(baseline)
            for weight in TARGET.SIX_WEIGHTS:
                bad = fail_505 and weight == [0.5, 0.0, 0.5]
                row = {
                    "status": "ok", "split": "train", "sample_id": sample,
                    "candidate": "synthetic_registered_path",
                    "original_numeric_bytes": raw, "weight": weight,
                    "measured_timing_weight":
                        TARGET.SIX_WEIGHTS[0] if reused else weight,
                    "time_reused_across_weights": reused,
                    "approximate_teacher_timing": reused,
                    "compression_ratio": 0.9 if bad else 0.4,
                    "compression_ns_per_byte": 10.0,
                    "decompression_ns_per_byte": 8.0 if bad else 2.0,
                }
                row["objective"] = TARGET.objective(row, weight)
                candidates.append(row)
        write_jsonl(teacher / "candidate_measurements.jsonl", candidates)
        write_jsonl(teacher / "direct_baselines.jsonl", baselines)
        write_jsonl(teacher / "teacher.jsonl", [{"status": "ok"}])
        (teacher / "search_space.json").write_text(json.dumps({"plans": 102}))
        manifest = teacher / "train_blocks_manifest.json"
        manifest.write_text(json.dumps({
            "split": "train",
            "samples": [{"sample_id": "train-0", "B_raw": 1000},
                        {"sample_id": "train-1", "B_raw": 3000}],
        }))
        return manifest

    def test_accepts_only_when_all_six_weighted_oracles_win(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory)
            manifest = self.make_measurements(teacher)
            receipt = TARGET.validate_sixweight_oracle_vs_zstd(teacher, manifest)
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["sample_count"], 2)
            self.assertEqual(receipt["original_numeric_bytes"], 4000)
            self.assertEqual(receipt["candidate_measurements_sha256"],
                             TARGET.base.sha(teacher / "candidate_measurements.jsonl"))
            self.assertEqual(receipt["train_manifest_sha256"],
                             TARGET.base.sha(manifest))
            self.assertEqual(set(receipt["comparisons"]),
                             {"100", "010", "001", "550", "505", "433"})
            self.assertTrue(all(row["strictly_better"] for row in
                                receipt["comparisons"].values()))

    def test_reports_505_failure_for_controller_to_refuse_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory)
            manifest = self.make_measurements(teacher, fail_505=True)
            receipt = TARGET.validate_sixweight_oracle_vs_zstd(teacher, manifest)
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["comparisons"]["505"]["strictly_better"])
            self.assertTrue(all(row["strictly_better"] for key, row in
                                receipt["comparisons"].items() if key != "505"))

    def test_rejects_reused_weight100_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory)
            manifest = self.make_measurements(teacher, reused=True)
            with self.assertRaisesRegex(RuntimeError, "reused or approximated"):
                TARGET.validate_sixweight_oracle_vs_zstd(teacher, manifest)

    def test_rejects_manifest_sample_or_B_raw_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher = Path(directory)
            manifest = self.make_measurements(teacher)
            document = json.loads(manifest.read_text())
            document["samples"][1]["B_raw"] += 1
            manifest.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "differ from train manifest"):
                TARGET.validate_sixweight_oracle_vs_zstd(teacher, manifest)


class StrongTeacherV2BindingTest(unittest.TestCase):
    def test_accepts_exact_102_template_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"paths": ["frozen"]}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()
            union_path = root / "candidate_union.json"
            union_path.write_text(json.dumps({
                "schema": "operator86-pdf102-candidate-union-v1",
                "payload": payload,
                "space_sha256": hashlib.sha256(canonical).hexdigest(),
            }))
            union = TARGET.union_identity(union_path)
            extension = {
                "union_space_sha256": union["space_sha256"],
                "union_catalog": str(union["path"]),
                "union_catalog_sha256": union["file_sha256"],
                "coverage_status": TARGET.COVERAGE_STATUS,
                "full_union_training_ready": False,
                "schema": "native-online-offline-combo-sixweight-typed-multiflow-v2",
                "candidate_counts": {
                    "all_76_v6_templates": 76,
                    "registered_specialist_seeds": 7,
                    "typed_multiflow_registered_connections": 9,
                    "total": 92,
                },
            }
            shared = {
                "union_space_sha256": union["space_sha256"],
                "coverage_status": TARGET.COVERAGE_STATUS,
                "full_union_coverage_claimed": False,
                "schema_extension": extension,
                "strong_path_schema": TARGET.STRONG_PATH_SCHEMA,
                "strong_path_candidates": TARGET.STRONG_PATH_CANDIDATES,
                "strong_path_candidate_count": 10,
                "new_reference_chain_candidate_count": 8,
                "strong_path_total_template_count": 102,
                "strong_reference_chain_candidates":
                    TARGET.STRONG_REFERENCE_CHAIN_CANDIDATES,
                "strong_path_action_registry_changed": False,
                "standard_bzip2_claimed": False,
                "standard_zstd_claimed": False,
                "standard_brotli_claimed": False,
                "upstream_wire_equivalence_claimed": False,
            }
            teacher = root / "teacher"
            teacher.mkdir()
            search = {**shared, "candidate_limit": 102,
                      "materialized_candidate_count": 102,
                      "plans": [{} for _ in range(102)],
                      "reuse_candidate_time_weights": False,
                      "approximate_teacher_timing": False,
                      "measured_candidate_timing_weights": TARGET.SIX_WEIGHTS}
            for filename, document in (("search_space.json", search),
                                       ("summary.json", shared),
                                       ("teacher.receipt.json", shared)):
                (teacher / filename).write_text(json.dumps(document))
            row = {"union_space_sha256": union["space_sha256"],
                   "coverage_status": TARGET.COVERAGE_STATUS,
                   "strong_path_schema": TARGET.STRONG_PATH_SCHEMA}
            for filename in ("candidate_measurements.jsonl", "teacher.jsonl"):
                write_jsonl(teacher / filename, [row])
            receipt = TARGET.validate_union_teacher_binding(teacher, union)
            self.assertEqual(receipt["teacher_template_count"], 102)


if __name__ == "__main__":
    unittest.main()
