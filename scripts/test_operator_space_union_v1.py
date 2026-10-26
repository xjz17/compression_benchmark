from __future__ import annotations

import copy
import unittest

import operator_space_union_v1 as target


def fixture() -> dict:
    methods = [{"id": f"method:method{n}", "kind": "method_path", "label": f"method{n}",
                "excluded": False} for n in range(86)]
    ids = [f"A{n:03d}" for n in range(1, 101)] + ["A109", "A110"]
    operators = [{"id": f"implementation:{identity}", "kind": "implementation",
                  "implementation_id": identity, "excluded": identity in target.EXCLUDED}
                 for identity in ids]
    edges = [{"id": f"edge:{n}", "kind": "pdf_edge", "from": "A028", "to": "A028",
              "excluded": False} for n in range(201)]
    return target.seal({"obligations": methods + operators + edges, "online_search": False})


class CoverageUnionTest(unittest.TestCase):
    def test_resume_covers_tail_beyond_64(self):
        document = fixture()
        cursor = 0
        seen = []
        for _ in range(7):
            schedule = target.coverage_schedule(document, cursor=cursor, budget=64)
            seen.extend(row["id"] for row in schedule["obligations"])
            cursor = schedule["next_cursor"]
        expected = {row["id"] for row in document["payload"]["obligations"]
                    if not row["excluded"]}
        self.assertEqual(expected, set(seen))
        self.assertIn("method:method85", seen)
        self.assertNotIn("implementation:A109", seen)
        self.assertNotIn("implementation:A110", seen)

    def test_queue_interleaves_required_paths_and_pdf_edges(self):
        document = fixture()
        batch = target.coverage_schedule(document, cursor=0, budget=6)
        self.assertEqual([row["kind"] for row in batch["obligations"]],
                         ["method_path", "implementation", "pdf_edge"] * 2)
        self.assertEqual(batch["queue_kind"], "coverage_requirements_not_executable_recipes")

    def test_missing_method_cannot_shrink_coverage_denominator(self):
        document = fixture()
        payload = copy.deepcopy(document["payload"])
        payload["obligations"] = payload["obligations"][1:]
        with self.assertRaisesRegex(ValueError, "86"):
            target.validate_document(target.seal(payload))

    def test_fingerprint_rejects_changed_conditions(self):
        document = fixture()
        document["payload"]["obligations"][-1]["condition"] = "pretend always legal"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            target.validate_document(document)

    def test_no_online_search_or_exclusion_drift(self):
        payload = fixture()["payload"]
        payload["online_search"] = True
        with self.assertRaisesRegex(ValueError, "online search"):
            target.validate_document(target.seal(payload))
        payload["online_search"] = False
        next(row for row in payload["obligations"] if row.get("implementation_id") == "A109")["excluded"] = False
        with self.assertRaisesRegex(ValueError, "exclusion"):
            target.validate_document(target.seal(payload))

    def test_all_view_variants_preserved_and_profiles_not_guessed(self):
        catalog = {"registry_payload": {"actions": [
            {"action_id": 100, "implementation_id": "A020", "category": "mechanism",
             "input_view": "bytes", "target_slot": 0},
            {"action_id": 101, "implementation_id": "A020", "category": "mechanism",
             "input_view": "int64", "target_slot": 0},
            {"action_id": 295, "implementation_id": "A020", "category": "mechanism",
             "input_view": "bytes", "target_slot": 1},
        ]}}
        layered = {"implementations": [
            {"implementation_id": "A020", "accounting": {"layer": "mechanism", "mechanism_ids": ["A020"]}},
            {"implementation_id": "A041", "accounting": {"layer": "profile", "mechanism_ids": ["A020"]}},
            {"implementation_id": "A043", "accounting": {"layer": "recipe", "mechanism_ids": ["A020"]}},
            {"implementation_id": "A096", "accounting": {"layer": "adapter", "mechanism_ids": []}},
        ]}
        bindings = target.binding_inventory(layered, catalog)
        self.assertEqual([row["action_id"] for row in bindings[0]["direct_slot0_configurations"]], [100, 101])
        self.assertFalse(any(row["selectable_proven"] for row in bindings))
        self.assertIn("parameter_binding", bindings[1]["status"])
        self.assertIn("branch_and_state", bindings[2]["status"])
        self.assertIn("packet_binding", bindings[3]["status"])

    def test_budget_does_not_duplicate_a_cycle(self):
        batch = target.coverage_schedule(fixture(), cursor=0, budget=100000)
        self.assertEqual(batch["selected_count"], batch["cycle_size"])
        ids = [row["id"] for row in batch["obligations"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_duplicate_id_rejected(self):
        payload = fixture()["payload"]
        payload["obligations"].append(payload["obligations"][-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            target.validate_document(target.seal(payload))


class CurrentSourcesIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = target.build_union()

    def test_full_current_sources_form_requested_union(self):
        p = self.document["payload"]
        self.assertEqual(len(p["original_method_paths"]), 86)
        self.assertEqual(len(p["pdf102"]["operators"]), 102)
        self.assertEqual(len(p["pdf102"]["edges"]), 10954)
        self.assertFalse(p["execution_admission"]["ready_for_full_union_training"])
        self.assertIn("Zstd", {row["label"] for row in p["original_method_paths"]})
        self.assertIn("Huff0", {row["label"] for row in p["original_method_paths"]})

    def test_documented_self_loops_and_state_reviews_retained(self):
        p = self.document["payload"]
        self.assertTrue(any(row["from"] == row["to"] for row in p["pdf102"]["edges"]))
        self.assertEqual(len(p["pdf102"]["unbound_state_outputs"]),
                         sum(row["kind"] == "state_contract_review" for row in p["obligations"]))

    def test_cannot_silently_drop_port_edge_even_with_recomputed_hash(self):
        p = copy.deepcopy(self.document["payload"])
        for index, row in enumerate(p["obligations"]):
            if row["kind"] == "pdf_edge":
                p["obligations"].pop(index)
                break
        with self.assertRaisesRegex(ValueError, "port edges"):
            target.validate_document(target.seal(p))

    def test_queue_complete_for_current_scope(self):
        batch = target.coverage_schedule(self.document, cursor=0, budget=100000)
        required = {row["id"] for row in self.document["payload"]["obligations"] if not row["excluded"]}
        self.assertEqual(required, {row["id"] for row in batch["obligations"]})

    def test_resealing_cannot_make_a_required_method_disappear(self):
        p = copy.deepcopy(self.document["payload"])
        row = next(row for row in p["obligations"] if row["id"] == "method:Chimp")
        row["excluded"] = True
        row["required"] = False
        with self.assertRaisesRegex(ValueError, "cannot be excluded"):
            target.coverage_schedule(target.seal(p), cursor=0, budget=100000)

    def test_review_and_edge_projection_cannot_be_relabelled(self):
        for kind, key, value in (
            ("state_contract_review", "excluded", True),
            ("dependency_review", "dependency_sha256", "0" * 64),
            ("pdf_edge", "required", False),
            ("pdf_edge", "from", "A999"),
        ):
            with self.subTest(kind=kind, key=key):
                p = copy.deepcopy(self.document["payload"])
                row = next(row for row in p["obligations"] if row["kind"] == kind)
                row[key] = value
                with self.assertRaisesRegex(ValueError, "canonical source projection"):
                    target.validate_document(target.seal(p))


if __name__ == "__main__":
    unittest.main()
