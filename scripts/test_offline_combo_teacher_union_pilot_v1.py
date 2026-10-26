#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_combo_teacher_union_pilot_v1 as target


def action(action_id, view, *, max_point=None):
    return target.CatalogAction(
        action_id, "stage", "A001", view, 0, True,
        f"stage[{view}]({max_point})", "stage", "none", None,
        max_point, ())


class UnionPilotTest(unittest.TestCase):
    def catalog(self, actions):
        value = target.ExactCatalog.__new__(target.ExactCatalog)
        value.actions = {row.action_id: row for row in actions}
        value.excluded_implementation_ids = frozenset({"A109", "A110"})
        return value

    def test_all_historical_templates_and_registered_seeds_are_present(self):
        plans = target.frozen_plans()
        self.assertEqual(sum(plan.coverage_origin == "all_76_v6_templates"
                             for plan in plans), 76)
        self.assertIn("union_seed_gorilla_registered",
                      {plan.name for plan in plans})
        self.assertIn("union_seed_svb_registered",
                      {plan.name for plan in plans})
        self.assertLessEqual(len(plans), 160)

    def test_current_view_selects_typed_row_not_lowest_id(self):
        catalog = self.catalog((action(2, "bytes"), action(9, "int64")))
        chosen = catalog.exact((2, 9), target.ActionDescriptor("stage"),
                               target_slot=0, current_input_view="int64")
        self.assertEqual(chosen, 9)

    def test_parameter_ambiguity_fails_closed(self):
        catalog = self.catalog((action(4, "float64", max_point=1),
                                action(5, "float64", max_point=2)))
        with self.assertRaisesRegex(ValueError, "2 legal matches"):
            catalog.exact((4, 5), target.ActionDescriptor("stage"),
                          target_slot=0, current_input_view="float64")
        chosen = catalog.exact(
            (4, 5), target.ActionDescriptor("stage", max_point=2),
            target_slot=0, current_input_view="float64")
        self.assertEqual(chosen, 5)


if __name__ == "__main__":
    unittest.main()
