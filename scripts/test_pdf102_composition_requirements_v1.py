#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import unittest

import pdf102_composition_requirements_v1 as subject


class Pdf102CompositionRequirementsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = subject.build_pdf_requirements()

    def test_all_102_trace_entries_are_accounted(self) -> None:
        operators = self.result["operators"]
        self.assertEqual(102, len(operators))
        self.assertEqual(102, len({row["id"] for row in operators}))
        self.assertEqual(
            {"A109", "A110"},
            {row["id"] for row in operators if not row["eligible"]},
        )

    def test_every_operator_and_edge_has_pdf_pages(self) -> None:
        for operator in self.result["operators"]:
            self.assertGreater(operator["source"]["review_pdf_operator_page"], 0)
            self.assertGreater(operator["source"]["review_pdf_compatibility_page"], 0)
            self.assertGreater(operator["source"]["handbook_pdf_operator_page"], 0)
        for edge in self.result["edges"]:
            source = edge["source"]
            self.assertGreater(source["review_pdf_compatibility_page"], 0)
            self.assertGreater(source["consumer_pdf_operator_page"], 0)
            self.assertGreater(source["consumer_pdf_compatibility_page"], 0)
            self.assertTrue(source["consumer_contract"])
            self.assertTrue(source["consumer_evidence"])

    def test_edges_are_typed_candidates_not_runtime_mask(self) -> None:
        known = {row["id"] for row in self.result["operators"]}
        ids = set()
        kinds = set()
        for edge in self.result["edges"]:
            self.assertIn(edge["from"], known)
            self.assertIn(edge["to"], known)
            self.assertEqual("forward", edge["direction"])
            self.assertFalse(edge["runtime_legal"])
            self.assertEqual("documented_candidate_unverified", edge["runtime_status"])
            self.assertTrue(edge["condition"])
            self.assertNotIn(edge["edge_id"], ids)
            ids.add(edge["edge_id"])
            kinds.add(edge["kind"])
            if edge["from"] in {"A109", "A110"} or edge["to"] in {"A109", "A110"}:
                self.assertFalse(edge["allowed_by_scope"])
        self.assertEqual({"direct", "adapted", "hint", "view"}, kinds)
        self.assertEqual(0, self.result["catalog"]["edge_kind_counts"]["state"])
        self.assertEqual(11, len(self.result["unbound_state_outputs"]))
        self.assertTrue(all(not row["runtime_legal"] for row in self.result["unbound_state_outputs"]))

    def test_specific_a111_to_a020_requirement_is_preserved(self) -> None:
        matches = [
            edge
            for edge in self.result["edges"]
            if edge["from"] == "A111" and edge["to"] == "A020" and edge["kind"] == "hint"
        ]
        self.assertEqual(3, len(matches))
        self.assertEqual({"low", "pos", "high"}, {edge["producer_port"] for edge in matches})
        self.assertTrue(all(edge["condition"] for edge in matches))

    def test_dependency_appendix_directions_and_state_rule_are_preserved(self) -> None:
        requirements = self.result["dependency_requirements"]
        self.assertEqual(10, len(requirements))
        by_id = {row["dependency_id"]: row for row in requirements}
        self.assertIn("probability-predict-code-observe-order", by_id)
        pairs = {
            (edge["from"], edge["to"])
            for edge in self.result["edges"]
            if edge["relation"] == "documented_order_dependency"
        }
        self.assertTrue(
            {
                ("A013", "A048"),
                ("A065", "A057"),
                ("A004", "A031"),
                ("A004", "A028"),
                ("A031", "A076"),
                ("A028", "A076"),
            }.issubset(pairs)
        )

    def test_json_serializable_and_deterministic(self) -> None:
        first = json.dumps(self.result, ensure_ascii=False, sort_keys=True)
        second = json.dumps(subject.build_pdf_requirements(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(first, second)

    def test_full_union_build_when_method_inventory_is_present(self) -> None:
        method_module = Path(__file__).with_name("operator86_required_paths_v1.py")
        if not method_module.is_file():
            self.skipTest("operator86_required_paths_v1.py is being produced by the sibling task")
        import operator_space_union_v1

        document = operator_space_union_v1.build_union()
        payload = document["payload"]
        self.assertEqual(86, len(payload["original_method_paths"]))
        self.assertEqual(102, len(payload["pdf102"]["operators"]))
        self.assertEqual(len(self.result["edges"]), len(payload["pdf102"]["edges"]))
        expected_hashes = {operator_space_union_v1.digest(edge) for edge in payload["pdf102"]["edges"]}
        obligation_hashes = {
            row["edge_sha256"]
            for row in payload["obligations"]
            if row["kind"] == "pdf_edge"
        }
        self.assertEqual(expected_hashes, obligation_hashes)


if __name__ == "__main__":
    unittest.main()
