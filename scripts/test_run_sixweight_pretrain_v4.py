#!/usr/bin/env python3
"""Fail-closed unit tests for the v4 controller admission adapter."""
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
        "run_sixweight_pretrain_v4_under_test",
        ROOT / "run_sixweight_pretrain_v4.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TARGET = load_controller()


class StrongTeacherBindingTest(unittest.TestCase):
    def make_fixture(self, root: Path, *, approximate: bool = False):
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
            "strong_path_action_registry_changed": False,
            "standard_zstd_claimed": False,
            "standard_brotli_claimed": False,
        }
        teacher = root / "teacher"
        teacher.mkdir()
        search = {**shared,
            "candidate_limit": TARGET.TEACHER_TEMPLATE_COUNT,
            "materialized_candidate_count": TARGET.TEACHER_TEMPLATE_COUNT,
            "plans": [{} for _ in range(TARGET.TEACHER_TEMPLATE_COUNT)],
            "reuse_candidate_time_weights": approximate,
            "approximate_teacher_timing": approximate,
            "measured_candidate_timing_weights":
                [TARGET.SIX_WEIGHTS[0]] if approximate else TARGET.SIX_WEIGHTS,
        }
        for filename, document in (("search_space.json", search),
                                   ("summary.json", shared),
                                   ("teacher.receipt.json", shared)):
            (teacher / filename).write_text(json.dumps(document))
        row = {"union_space_sha256": union["space_sha256"],
               "coverage_status": TARGET.COVERAGE_STATUS,
               "strong_path_schema": TARGET.STRONG_PATH_SCHEMA}
        for filename in ("candidate_measurements.jsonl", "teacher.jsonl"):
            (teacher / filename).write_text(json.dumps(row) + "\n")
        return teacher, union

    def test_accepts_exact_sixweight_94_template_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher, union = self.make_fixture(Path(directory))
            receipt = TARGET.validate_union_teacher_binding(teacher, union)
            self.assertEqual(receipt["teacher_template_count"], 94)
            self.assertFalse(receipt["action_registry_changed"])
            self.assertEqual(receipt["row_counts"], {
                "candidate_measurements.jsonl": 1, "teacher.jsonl": 1})

    def test_rejects_weight100_timing_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher, union = self.make_fixture(Path(directory), approximate=True)
            with self.assertRaisesRegex(RuntimeError, "exact six-weight timing"):
                TARGET.validate_union_teacher_binding(teacher, union)

    def test_rejects_registry_relabel(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher, union = self.make_fixture(Path(directory))
            path = teacher / "summary.json"
            document = json.loads(path.read_text())
            document["strong_path_action_registry_changed"] = True
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(RuntimeError, "teacher union binding"):
                TARGET.validate_union_teacher_binding(teacher, union)


if __name__ == "__main__":
    unittest.main()
