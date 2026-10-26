#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import offline_combo_teacher_strong_paths_v1 as v1
import offline_combo_teacher_strong_paths_v2 as target


ROOT = Path(__file__).resolve().parents[1]


class StrongPathsV2Test(unittest.TestCase):
    def test_preserves_94_and_adds_eight_unique_plans(self):
        old = v1.frozen_plans()
        plans = target.frozen_plans()
        self.assertEqual(len(old), 94)
        self.assertEqual(plans[:94], old)
        self.assertEqual(len(plans), 102)
        self.assertEqual(len({plan.name for plan in plans}), 102)
        self.assertTrue(all(
            plan.coverage_origin == "strong_registered_reference_chains_v2"
            for plan in plans[-8:]))

    def test_every_new_descriptor_exists_in_frozen_registry(self):
        catalog = json.loads((
            ROOT / "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json"
        ).read_text())
        actions = catalog["registry_payload"]["actions"]
        self.assertEqual(catalog["registry_fingerprint"],
                         "50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd")
        for plan in target.frozen_plans()[-8:]:
            for descriptor in (*plan.prefix, *plan.leaf_backend):
                matches = [row for row in actions
                           if row["runtime_action"] == descriptor.runtime_action
                           and row.get("implementation_id") == descriptor.implementation_id
                           and (descriptor.io_variant is None or
                                row.get("io_variant") == descriptor.io_variant)]
                self.assertTrue(matches, (plan.name, descriptor))
                self.assertTrue(any(row.get("target_slot") == 0 for row in matches))

    def test_receipt_extension_disclaims_upstream_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("search_space.json", "summary.json",
                         "teacher.receipt.json"):
                (output / name).write_text("{}\n")
            for name in ("candidate_measurements.jsonl", "teacher.jsonl"):
                (output / name).write_text('{"status":"ok"}\n')
            target._augment(output)
            for name in ("search_space.json", "summary.json",
                         "teacher.receipt.json"):
                document = json.loads((output / name).read_text())
                self.assertEqual(document["strong_path_total_template_count"], 102)
                self.assertEqual(document["new_reference_chain_candidate_count"], 8)
                self.assertFalse(document["strong_path_action_registry_changed"])
                self.assertFalse(document["standard_bzip2_claimed"])
                self.assertFalse(document["standard_brotli_claimed"])
                self.assertFalse(document["standard_zstd_claimed"])
                self.assertFalse(document["upstream_wire_equivalence_claimed"])


if __name__ == "__main__":
    unittest.main()
