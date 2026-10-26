"""Check the cross-version accounting boundary, not performance claims."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import unittest

import operator_implementation_catalog as catalog
from operator_catalog_layers import validate_layers

ROOT = Path(__file__).resolve().parents[1]


class LayeredRevisionTests(unittest.TestCase):
    def test_frozen_source_crosswalk_and_ids_are_preserved(self):
        path = ROOT / "benchmark/reference/actions/operator_implementations_102.json"
        original = json.loads(path.read_bytes())
        current = catalog.inventory()
        self.assertEqual(current["previous_catalog_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        before = {x["implementation_id"]: x for x in original["implementations"]}
        self.assertEqual(set(before), set(catalog.BY_ID))
        for key, item in before.items():
            self.assertEqual(item["source_entry_ids"], catalog.BY_ID[key]["source_entry_ids"])
            self.assertEqual(item["source_action_names"], catalog.BY_ID[key]["source_action_names"])
        self.assertEqual(original["retired"], current["retired"])
        self.assertEqual(original["runtime_action_recipes"], current["runtime_action_recipes"])

    def test_layers_partition_entries_without_counting_profiles_twice(self):
        self.assertEqual(catalog.TRACE_ENTRY_COUNT, 102)
        self.assertEqual(catalog.MECHANISM_COUNT, 67)
        self.assertEqual(catalog.LAYER_COUNTS, {"mechanism": 67, "profile": 7, "recipe": 14, "adapter": 14})
        by_layer = Counter(item["accounting"]["layer"] for item in catalog.IMPLEMENTATIONS)
        self.assertEqual(dict(by_layer), catalog.LAYER_COUNTS)
        mechanisms = {item["implementation_id"] for item in catalog.MECHANISMS}
        for item in catalog.IMPLEMENTATIONS:
            acc = item["accounting"]
            self.assertLessEqual(set(acc["mechanism_ids"]), mechanisms)
            if acc["layer"] == "profile":
                self.assertNotIn(item["implementation_id"], mechanisms)
                self.assertEqual(len(acc["mechanism_ids"]), 1)
        self.assertEqual({f for item in catalog.IMPLEMENTATIONS for f in item["family_numbers"]}, set(range(1, 31)))

    def test_opaque_backends_are_not_declared_fully_expanded(self):
        opaque = [item for item in catalog.IMPLEMENTATIONS
                  if item["accounting"].get("kind") == "opaque_composite_backend"]
        self.assertEqual(len(opaque), 10)
        for item in opaque:
            self.assertEqual(item["accounting"]["mechanism_ids"], [])
        self.assertIn("not_registered_in_graph_policy", catalog.BY_ID["A111"]["execution_status"])
        self.assertIn("not", catalog.inventory()["scope"])

    def test_wrong_profile_target_is_rejected_even_when_counts_match(self):
        entries = catalog.inventory()["implementations"]
        next(item for item in entries if item["implementation_id"] == "A051")["accounting"]["mechanism_ids"] = ["A049"]
        with self.assertRaises(AssertionError):
            validate_layers(entries)


if __name__ == "__main__":
    unittest.main(verbosity=2)
