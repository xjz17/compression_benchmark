"""Current ZigZag decomposition contracts and frozen-wire compatibility."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import action_semantic_ports as semantic
import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified


class ZigZagDecompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = unified.ActionData.ints(
            [100, -10, 5, 5, 200, -300, 8, 9, 10, -(1 << 40), 1 << 40]
        )

    def test_current_registry_reserves_old_slots_but_excludes_fused_actions(self):
        self.assertEqual(primitive.ACTION_REGISTRY_VERSION, "cpp-canonical-127-v4")
        self.assertEqual(len(primitive.FROZEN_POLICY_ACTIONS), 127)
        self.assertEqual(len(primitive.POLICY_ACTIONS), 122)
        self.assertEqual(primitive.STOP_ACTION_ID, 127)
        self.assertEqual(
            sorted(set(primitive.FROZEN_POLICY_ACTION_BY_ID) - set(primitive.POLICY_ACTION_BY_ID)),
            [5, 18, 35, 87, 90],
        )
        self.assertEqual(len(primitive.LEGACY_PRIMITIVE_ACTIONS), 29)
        self.assertEqual(len(primitive.PRIMITIVE_ACTIONS), 27)
        self.assertNotIn("op.integer_code.reference_v1", primitive.ACTION_BY_NAME)

    def test_v2_composites_migrate_to_atomic_current_recipes(self):
        expected = {
            17: ([17, 55], list(primitive.LECO_ACTIONS[:2])),
            18: ([30, 55], list(primitive.REGER_ACTIONS[1:3])),
            25: ([25], [primitive.ULEB128_DBP_ACTION.name]),
            35: ([30, 55], list(primitive.BOS64_INT_ACTIONS[:2])),
            87: ([55, 25], list(primitive.REFERENCE_INTEGER_ACTIONS)),
            90: ([25], [primitive.ULEB128_SNAPPY_ACTION.name]),
        }
        for old_id, (ids, variants) in expected.items():
            with self.subTest(old_id=old_id):
                migrated = primitive.migrate_policy_action(
                    old_id, "cpp-canonical-127-v2"
                )
                self.assertEqual(migrated["migration_kind"], "replacement_recipe")
                self.assertEqual(migrated["replacement_policy_action_ids"], ids)
                self.assertEqual(migrated["replacement_io_variants"], variants)

    def test_retired_fused_forward_selectors_are_rejected(self):
        for name in sorted(primitive.RETIRED_COMPOSITE_ACTION_NAMES):
            with self.subTest(name=name):
                with self.assertRaisesRegex(unified.ActionNotExecutableError, "retired"):
                    unified.run_action(name, self.source)

    def test_split_chains_round_trip(self):
        for label, actions in (
            ("raw ZigZag then ULEB128", primitive.REFERENCE_INTEGER_ACTIONS),
            ("LECO", primitive.LECO_ACTIONS),
            ("REGER", primitive.REGER_ACTIONS),
            ("BOS64", primitive.BOS64_INT_ACTIONS),
            ("DBP", primitive.DBP_INT_ACTIONS),
        ):
            with self.subTest(label=label):
                encoded = unified.run_actions(actions, self.source)
                self.assertEqual(
                    unified.run_actions(actions, encoded, reverse=True), self.source
                )

        byte_source = unified.ActionData.bytes(b"aaaaabaaaaabaaaaab0123456789")
        encoded = unified.run_actions(primitive.SNAPPY_ACTIONS, byte_source)
        self.assertEqual(
            unified.run_actions(primitive.SNAPPY_ACTIONS, encoded, reverse=True),
            byte_source,
        )

    def test_current_policy_recipe_runs_with_reserved_ids(self):
        current = self.source
        steps = ((55, primitive.ZIGZAG_RAW_ACTION.name),
                 (25, primitive.ULEB128_RAW_ACTION.name))
        for action_id, variant in steps:
            current = unified.run_policy_action(
                action_id, current, registry_version=primitive.ACTION_REGISTRY_VERSION,
                io_variant=variant,
            )
        for action_id, variant in reversed(steps):
            current = unified.run_policy_action(
                action_id, current, registry_version=primitive.ACTION_REGISTRY_VERSION,
                io_variant=variant, reverse=True,
            )
        self.assertEqual(current, self.source)

    def test_split_chains_preserve_public_wire(self):
        raw = self.source.payload
        old_leco = primitive._run_learning_chain(
            raw, ("leco-autoregression-forward", "leco-bitpack-forward")
        )
        self.assertEqual(primitive.encode_exact_leco_action_pipeline(raw), old_leco)

        old_reger = primitive._run_learning_chain(
            raw,
            (
                "reger-reorder-forward",
                "reger-autoregression-forward",
                "reger-for-forward",
                "reger-bitpack-forward",
                "reger-residual-core-forward",
            ),
        )
        self.assertEqual(primitive.encode_exact_reger_action_pipeline(raw), old_reger)

        old_bos = primitive._run_bos_stage_cli("adapter-delta-forward", raw)
        old_bos = primitive._run_bos_stage_cli("adapter-wire-forward-int", old_bos)
        self.assertEqual(
            primitive.encode_exact_bos64_action_pipeline(raw, "int"), old_bos
        )

    def test_semantic_ports_expose_each_atomic_boundary(self):
        cases = []

        leco_signed = unified.run_action(primitive.LECO_ACTIONS[0], self.source)
        leco_unsigned = unified.run_action(primitive.LECO_ACTIONS[1], leco_signed)
        cases.extend((
            (leco_signed, "LRD1", (("model_parameters", "float64"), ("residuals", "int64"))),
            (leco_unsigned, "LAR1", (("model_parameters", "float64"), ("residuals", "uint64"))),
        ))

        reger_reordered = unified.run_action(primitive.REGER_ACTIONS[0], self.source)
        reger_signed = unified.run_action(primitive.REGER_ACTIONS[1], reger_reordered)
        reger_unsigned = unified.run_action(primitive.REGER_ACTIONS[2], reger_signed)
        cases.extend((
            (reger_signed, "RSD1", (("first", "int64"), ("original_index", "uint32"), ("residuals", "int64"))),
            (reger_unsigned, "RAR1", (("first", "int64"), ("original_index", "uint32"), ("residuals", "uint64"))),
        ))

        bos_signed = unified.run_action(primitive.BOS64_INT_ACTIONS[0], self.source)
        bos_unsigned = unified.run_action(primitive.BOS64_INT_ACTIONS[1], bos_signed)
        cases.extend((
            (bos_signed, "B6S1", (("first", "int64"), ("residuals", "int64"))),
            (bos_unsigned, "B6D1", (("first", "int64"), ("residuals", "uint64"))),
        ))

        for wire, schema, names_and_kinds in cases:
            with self.subTest(schema=schema):
                ports = semantic.to_ports(wire)
                self.assertEqual(ports.metadata["schema"], schema)
                self.assertEqual(
                    tuple((stream.name, stream.kind) for stream in ports.streams),
                    names_and_kinds,
                )
                self.assertEqual(semantic.from_ports(ports), wire)

    def test_uleb_profiles_do_not_hide_bypass_streams(self):
        dbp = unified.run_actions(primitive.DBP_INT_ACTIONS, self.source)
        dbp_ports = semantic.to_ports(dbp)
        self.assertEqual(dbp_ports.metadata["schema"], "dbp_uleb128")
        self.assertEqual(
            tuple((s.name, s.kind) for s in dbp_ports.streams),
            (("first", "int64"), ("encoded_residuals", "bytes")),
        )
        self.assertEqual(semantic.from_ports(dbp_ports), dbp)

        source = unified.ActionData.bytes(b"aaaaabaaaaabaaaaab0123456789")
        snappy = unified.run_actions(primitive.SNAPPY_ACTIONS, source)
        snappy_ports = semantic.to_ports(snappy)
        self.assertEqual(snappy_ports.metadata["schema"], "snappy_uleb128")
        self.assertEqual(
            tuple((s.name, s.kind) for s in snappy_ports.streams),
            (("encoded_length", "bytes"), ("token_body", "bytes")),
        )
        self.assertEqual(semantic.from_ports(snappy_ports), snappy)

    def test_benchmark_exact_chains_use_explicit_zigzag(self):
        exact = primitive.benchmark_86_exact_action_pipelines()
        self.assertEqual(exact["LECO"]["int"], list(primitive.LECO_ACTIONS))
        self.assertEqual(exact["REGER"]["int"], list(primitive.REGER_ACTIONS))
        self.assertEqual(exact["BOS"]["int"], list(primitive.BOS64_INT_ACTIONS))
        self.assertEqual(exact["DBP"]["int"], list(primitive.DBP_INT_ACTIONS))
        self.assertEqual(exact["Snappy"], list(primitive.SNAPPY_ACTIONS))
        for method in ("LECO", "REGER", "BOS", "DBP"):
            self.assertIn("op.zigzag.", "\n".join(exact[method]["int"]))


if __name__ == "__main__":
    unittest.main()
