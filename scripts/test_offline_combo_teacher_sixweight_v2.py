#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import offline_combo_teacher_sixweight_v2 as target


CATALOG = ROOT / "cpp/native_pipeline_v1/frozen_complete_action_catalog_v1.json"
PROFILE = ROOT / "cpp/native_pipeline_v1/experiment_no_cmix_nncp_v1.json"


class SixWeightTeacherV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = target.ExactCatalog(CATALOG, PROFILE)

    def test_real_catalog_is_exact_frozen_v7(self):
        self.assertEqual(target.base.sha256(CATALOG), target.CATALOG_SHA256)
        self.assertEqual(len(self.catalog.actions), target.fit3.FLOW_STOP_BEGIN)
        self.assertTrue({"A109", "A110"}.issubset(
            self.catalog.excluded_implementation_ids))

    def test_all_83_inherited_plans_and_nine_connections_are_present(self):
        plans = target.frozen_plans()
        self.assertEqual(len(plans), 92)
        self.assertEqual(sum(
            plan.coverage_origin == "all_76_v6_templates" for plan in plans), 76)
        self.assertEqual(sum(
            plan.coverage_origin == "registered_specialist_seed" for plan in plans), 7)
        typed = [plan for plan in plans if
                 plan.coverage_origin == "typed_multiflow_registered_connection"]
        self.assertEqual(len(typed), 9)
        self.assertEqual(len({plan.name for plan in plans}), 92)
        self.assertIn("typed_multiflow_xor_semantic_ports+each_rans_reference",
                      {plan.name for plan in typed})
        self.assertNotIn("zstd", " ".join(plan.name.lower() for plan in typed))
        self.assertEqual(target.COVERAGE_STATUS, "partial_current_runtime_only")

    def test_two_root_policy_horizon_bound_is_explicit(self):
        typed = [plan for plan in target.frozen_plans() if
                 plan.coverage_origin == "typed_multiflow_registered_connection"]
        estimates = {plan.name: target.estimated_two_root_steps(plan)
                     for plan in typed}
        self.assertLessEqual(max(estimates.values()), 10)
        self.assertEqual(
            estimates["typed_multiflow_xor_semantic_ports+each_"
                      "max_value_bitpack+snappy_stages"], 10)

    def test_stage_dispatch_policy_views_are_tracked(self):
        self.assertEqual(target.output_views(target.FOR_MINIMUM, 2),
                         ["bytes", "bytes"])
        self.assertEqual(target.output_views(target.DICTIONARY_WORDS64, 2),
                         ["bytes", "bytes"])
        self.assertEqual(target.output_views(target.XOR_SEMANTIC_PORTS, 2),
                         ["uint64", "xor_residual64"])
        with self.assertRaisesRegex(ValueError, "two typed ports"):
            target.output_views(target.XOR_SEMANTIC_PORTS, 1)

    def test_full_descriptors_resolve_uniquely_on_real_catalog(self):
        legal = tuple(self.catalog.actions)
        for slot in (0, 1):
            for current_view, descriptor, expected_view in (
                    ("int64", target.FOR_MINIMUM, "int64"),
                    ("float64", target.FOR_MINIMUM, "bytes"),
                    ("int64", target.DICTIONARY_WORDS64, "int64"),
                    ("float64", target.DICTIONARY_WORDS64, "float64"),
                    ("int64", target.XOR_SEMANTIC_PORTS, "int64"),
                    ("float64", target.XOR_SEMANTIC_PORTS, "float64"),
                    ("uint64", target.MAX_VALUE_BITPACK[0], "bytes"),
                    ("xor_residual64", target.MAX_VALUE_BITPACK[0], "bytes"),
                    ("uint64", target.LEAF_BACKENDS[1][1][0], "bytes"),
                    ("xor_residual64", target.LEAF_BACKENDS[2][1][0], "bytes"),
                    ("xor_residual64", target.LEAF_BACKENDS[3][1][0], "bytes")):
                action_id = self.catalog.exact(
                    legal, descriptor, target_slot=slot,
                    current_input_view=current_view)
                action = self.catalog.actions[action_id]
                self.assertEqual(action.target_slot, slot)
                self.assertEqual(action.input_view, expected_view)
                self.assertEqual(action.implementation_id,
                                 descriptor.implementation_id)

    def test_snappy_pair_is_descriptor_exact_and_not_an_alias(self):
        legal = tuple(self.catalog.actions)
        chosen = [self.catalog.actions[self.catalog.exact(
            legal, descriptor, target_slot=0, current_input_view="bytes")]
                  for descriptor in target.SNAPPY_STAGE_PAIR]
        self.assertEqual([row.implementation_id for row in chosen],
                         ["A082", "A068"])
        self.assertEqual([row.runtime_action for row in chosen], [
            "op.block_state_control.snappy_matchfinder_residual_core.cpp_v1",
            "op.match_token_format.snappy.cpp_v1"])


if __name__ == "__main__":
    unittest.main()
