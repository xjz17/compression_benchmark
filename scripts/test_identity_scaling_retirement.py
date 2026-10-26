"""A005 retirement: active selection rejects it while old frames still decode."""
from __future__ import annotations

from pathlib import Path
import unittest

import generate_air_operator_pair_heatmap as heatmap
import operator_taxonomy_v30_v6 as taxonomy
import primitive_operator_action_runtime as primitive
import typed_graph_runtime as graph
import unified_action_runtime as unified


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "analysis_outputs/identity_scaling_retirement_20260915/fixtures"
IDENTITY = "op.scaling.reference_v1"


def _legacy_source_and_wire() -> tuple[unified.ActionData, unified.ActionData]:
    raw = (FIXTURES / "input_i64.bin").read_bytes()
    source = unified.ActionData("int64", raw, len(raw) // 8)
    opa1 = (FIXTURES / "identity_wire.bin").read_bytes()
    wire = unified._pack_result((IDENTITY,), source, opa1, None)
    return source, wire


class IdentityScalingRetirementTests(unittest.TestCase):
    def test_counts_and_reserved_ids_are_separate(self):
        self.assertEqual(primitive.implementation_catalog.IMPLEMENTATION_COUNT, 106)
        self.assertEqual(primitive.ACTION_REGISTRY_VERSION, "cpp-canonical-127-v3")
        self.assertEqual(primitive.PREVIOUS_ACTION_REGISTRY_VERSION, "cpp-canonical-127-v2")
        self.assertEqual(primitive.LEGACY_ACTION_REGISTRY_VERSION, "cpp-canonical-127-v1")
        self.assertEqual(len(primitive.LEGACY_PRIMITIVE_ACTIONS), 29)
        self.assertEqual(len(primitive.PRIMITIVE_ACTIONS), 27)
        self.assertEqual(len(primitive.FROZEN_POLICY_ACTIONS), 127)
        self.assertEqual(len(primitive.POLICY_ACTIONS), 122)
        self.assertEqual(primitive.FROZEN_POLICY_ACTION_BY_ID[5].name, IDENTITY)
        self.assertNotIn(5, primitive.POLICY_ACTION_BY_ID)
        self.assertEqual(
            sorted(set(primitive.FROZEN_POLICY_ACTION_BY_ID) - set(primitive.POLICY_ACTION_BY_ID)),
            [5, 18, 35, 87, 90],
        )
        self.assertEqual(primitive.STOP_ACTION_ID, 127)

        from canonical206_search_env import Env
        env = Env({
            "kind": "int64", "raw": unified.ActionData.ints([1, 2]).payload,
            "max_point": 0,
        }, sample_rows=0)
        base = len(env._base_features)
        self.assertEqual(env.action_count, 128)
        self.assertEqual(env.feature_dim, base + 127 + 12)
        self.assertLess(base + 126, env.feature_dim - 12)

    def test_current_enumeration_and_reference_recipes_omit_identity(self):
        self.assertNotIn(IDENTITY, primitive.ACTION_BY_NAME)
        self.assertNotIn(IDENTITY, unified.CANONICAL_ACTION_BY_NAME)
        self.assertNotIn("scaling", {item.operator_id for item in heatmap.OPERATORS})
        for pipeline in primitive.benchmark_86_action_pipelines().values():
            self.assertNotIn(IDENTITY, str(pipeline))
        self.assertIn(
            "op.scaling.ts_2diff.cpp_v1",
            primitive.benchmark_86_exact_action_pipelines()["2DIFF"]["float"],
        )
        self.assertIn(
            "op.scaling.alp.cpp_v1",
            primitive.benchmark_86_exact_action_pipelines()["ALP"]["float"],
        )

    def test_current_forward_paths_reject_identity(self):
        source = unified.ActionData.ints([1, -2, 3])
        with self.assertRaisesRegex(unified.ActionNotExecutableError, "retired"):
            unified.run_action(IDENTITY, source)
        with self.assertRaisesRegex(unified.ActionNotExecutableError, "retired"):
            unified.run_policy_action(
                5, source, registry_version=primitive.ACTION_REGISTRY_VERSION,
            )
        state = graph.GraphState.from_streams([
            graph.Stream("numeric", 8, source.payload),
        ])
        with self.assertRaisesRegex(unified.ActionNotExecutableError, "retired"):
            unified.run_graph_action(
                state, 5, ["s0"], registry_version=taxonomy.REGISTRY_VERSION,
                input_view="int64",
            )

    def test_old_uar_and_graph_frames_still_decode(self):
        source, wire = _legacy_source_and_wire()
        self.assertEqual(unified.run_action(IDENTITY, wire, reverse=True), source)

        header, payload = unified._unpack_result(wire)
        header.update(
            policy_registry_version=primitive.LEGACY_ACTION_REGISTRY_VERSION,
            policy_algorithm=IDENTITY,
            policy_action_id=5,
            io_variant=IDENTITY,
            io_input_kind="int64",
            io_input_magic=None,
        )
        policy_wire = unified._pack_existing_result(header, payload)
        with self.assertRaisesRegex(unified.ActionNotExecutableError, "retired"):
            unified.run_policy_action(
                5, policy_wire, registry_version=primitive.ACTION_REGISTRY_VERSION,
                reverse=True,
            )
        self.assertEqual(
            unified.run_policy_action(
                5, policy_wire, registry_version=primitive.LEGACY_ACTION_REGISTRY_VERSION,
                reverse=True,
            ),
            source,
        )

        original = graph.Stream("numeric", 8, source.payload)
        leaf = graph.Stream("serial", 1, wire.payload)
        record = {
            "action": IDENTITY,
            "inputs": ["s0"],
            "input_specs": [original.spec()],
            "kind": "legacy",
            "outputs": ["s1"],
            "output_specs": [leaf.spec()],
            "input_view": "int64",
        }
        old_state = graph.GraphState(
            {"s1": leaf}, {"s0": original.spec()}, (record,), 2,
            "operator-taxonomy-30-v4",
        )
        self.assertEqual(graph.restore_frame(old_state.stop()), {"s0": original})

    def test_v1_identity_migrates_to_an_explicit_empty_recipe(self):
        migrated = primitive.migrate_policy_action(
            5, primitive.LEGACY_ACTION_REGISTRY_VERSION,
        )
        self.assertEqual(migrated["migration_kind"], "retired_identity_bypass")
        self.assertEqual(migrated["replacement_policy_action_ids"], [])
        self.assertNotIn("policy_action_id", migrated)

        from canonical182_policy_paths import path_from_record
        old = primitive.LEGACY_ACTION_REGISTRY_VERSION
        only_identity = path_from_record({
            "ids": [5], "io_variants": [IDENTITY], "parameters": [{}],
            "policy_registry_version": old,
        })
        self.assertEqual(tuple(only_identity), ())
        mixed = path_from_record({
            "ids": [5, 0],
            "io_variants": [IDENTITY, "op.eraser.reference_v1"],
            "parameters": [{}, {}], "policy_registry_version": old,
        })
        self.assertEqual(tuple(mixed), (0,))

    def test_v6_exposes_retirement_without_counting_an_active_codec(self):
        descriptor = taxonomy.describe_action(5)
        self.assertEqual((descriptor.name, descriptor.kind), (IDENTITY, "retired"))
        self.assertFalse(taxonomy.availability(5).selectable)
        report = taxonomy.registry()
        self.assertEqual(report["counts"]["retired_descriptor_slots"], 5)
        self.assertEqual(report["implementation_directory"]["implementation_count"], 106)
        with self.assertRaisesRegex(ValueError, "retired current graph slot"):
            taxonomy.migrate_action(5, taxonomy.REGISTRY_VERSION)

    def test_all_graph_and_policy_predecessors_migrate_identity_without_key_errors(self):
        import operator_taxonomy_v30 as v1
        import operator_taxonomy_v30_v2 as v2
        import operator_taxonomy_v30_v3 as v3
        import operator_taxonomy_v30_v4 as v4

        for module in (v1, v2, v3, v4):
            with self.subTest(graph_version=module.REGISTRY_VERSION):
                migrated = taxonomy.migrate_action(5, module.REGISTRY_VERSION)
                self.assertEqual(migrated["kind"], "retired_recipe")
                self.assertEqual(migrated["replacement_action_ids"], [])
        # V4 remains a frozen historical view and still describes its old slot.
        self.assertEqual(v4.migrate_action(5, v4.REGISTRY_VERSION)["action_id"], 5)

        for version, action_id in (
            (primitive.LEGACY_ACTION_REGISTRY_VERSION, 5),
            ("cpp-canonical-129-v1", 5),
            ("cpp-canonical-137-v1", 5),
            ("cpp-canonical-140-v1", 5),
            ("cpp-canonical-143-v1", 5),
            ("cpp-canonical-156-v1", 5),
            ("cpp-canonical-182-v1", 5),
            ("cpp-canonical-206-v1", 1),
            ("legacy-225-v1", 1),
        ):
            with self.subTest(policy_version=version):
                migrated = taxonomy.migrate_action(action_id, version)
                self.assertEqual(migrated["kind"], "retired_recipe")
                self.assertEqual(migrated["replacement_action_ids"], [])


if __name__ == "__main__":
    unittest.main()
