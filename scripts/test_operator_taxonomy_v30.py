"""Registry-only tests; no build, binary probing, model loading, or data writes."""
import json
import unittest

import canonical143_registry as legacy
import operator_taxonomy_v30 as taxonomy


class TaxonomyTests(unittest.TestCase):
    def test_exact_families_and_dense_namespace(self):
        self.assertEqual(tuple(f.number for f in taxonomy.FAMILIES), tuple(range(1, 31)))
        self.assertEqual(taxonomy.FAMILY_BY_NUMBER[29].operator_id, "split_columns")
        self.assertEqual(taxonomy.FAMILY_BY_NUMBER[30].operator_id, "prediction_probability")
        self.assertEqual(tuple(a.action_id for a in taxonomy.ACTIONS), tuple(range(154)))
        self.assertEqual(len(taxonomy.CODEC_ACTIONS), 146)
        self.assertEqual(len(taxonomy.MODEL_PROVIDERS), 5)
        self.assertTrue(all(a.operator_id is None and a.family_number is None for a in taxonomy.SPECIAL_ACTIONS))

    def test_all_legacy_ids_names_and_variants_are_preserved(self):
        for row in legacy.GROUPS:
            action = taxonomy.describe_action(row["policy_action_id"])
            self.assertEqual(action.name, row["name"])
            self.assertEqual(action.legacy_policy_action_id, row["policy_action_id"])
            self.assertEqual(action.io_variants, tuple(v["name"] for v in row["io_variants"]))
            self.assertEqual(action.legacy_operator_id, row["operator_id"])
            self.assertEqual(action.operator_id, "split_columns" if action.action_id == 65 else row["operator_id"])

    def test_columnize_reclassifies_without_duplicate(self):
        action = taxonomy.describe_action("op.split_columns.columnize.cpp_v1")
        self.assertEqual(action.action_id, 65)
        self.assertEqual(action.name, "op.block_transform.byte_shuffle.cpp_v2")
        columns = [a for a in taxonomy.CODEC_ACTIONS if a.operator_id == "split_columns"]
        self.assertEqual(tuple(a.mechanism for a in columns), taxonomy.COLUMN_MECHANISMS)
        self.assertEqual(len(columns), 4)
        self.assertFalse(taxonomy.availability(65, mode="split-struct").available)
        self.assertTrue(taxonomy.availability(65, {action.name: True}, mode="split-struct").available)

    def test_snapshot_is_detached(self):
        snapshot = taxonomy.legacy_snapshot()
        snapshot[0]["name"] = "not-real"
        self.assertNotEqual(taxonomy.legacy_snapshot()[0]["name"], "not-real")
        self.assertEqual(legacy.GROUPS[65]["operator_id"], "block_transform")

    def test_provider_descriptors_are_not_automatic_codec_implementations(self):
        self.assertEqual(tuple(a.mechanism for a in taxonomy.MODEL_PROVIDERS), ("PPM", "DMC", "PAQ", "cmix", "NNCP"))
        for action in taxonomy.MODEL_PROVIDERS:
            self.assertEqual(action.kind, "model_provider")
            self.assertFalse(taxonomy.availability(action.action_id).available)
            self.assertNotIn(action, taxonomy.selectable_actions())
        self.assertFalse(taxonomy.availability("op.prediction_probability.nncp.provider_v1").selectable)

    def test_runtime_overrides_and_large_memory_gate(self):
        ppm = taxonomy.MODEL_PROVIDERS[0].name
        cmix = taxonomy.MODEL_PROVIDERS[3].name
        caps = {ppm: {"available": True, "status": "connected", "reason": "adapter test"}, cmix: True}
        self.assertTrue(taxonomy.availability(ppm, caps).selectable)
        self.assertTrue(taxonomy.availability(cmix, caps).available)
        self.assertFalse(taxonomy.availability(cmix, caps).selectable)
        self.assertTrue(taxonomy.availability(cmix, caps, allow_large_memory=True).selectable)
        self.assertFalse(taxonomy.availability(ppm, {ppm: {"available": False, "selectable": True}}).selectable)
        self.assertFalse(taxonomy.availability(cmix, {cmix: {"available": True, "requires_opt_in": []}}).selectable)

    def test_special_split_variants_are_one_action_kind(self):
        self.assertEqual(len(taxonomy.SPECIAL_ACTIONS), 3)
        split = taxonomy.describe_action("split_by_range")
        self.assertEqual(split.name, "split_segments")
        self.assertEqual(split.modes, ("continuous", "split_by_range"))
        self.assertEqual(taxonomy.describe_action("concat").action_id, 152)
        self.assertEqual(taxonomy.describe_action("stop").action_id, 151)

    def test_legacy_stop_collision_is_versioned(self):
        old = taxonomy.migrate_action(143, taxonomy.LEGACY_REGISTRY_VERSION)
        new = taxonomy.migrate_action(143, taxonomy.REGISTRY_VERSION)
        self.assertEqual((old["action_id"], old["name"], old["kind"]), (151, "stop", "special"))
        self.assertEqual(new["name"], "op.split_columns.float_deconstruct.cpp_v1")
        self.assertEqual(new["kind"], "codec")

    def test_every_legacy_step_maps_without_renumbering(self):
        for row in legacy.GROUPS:
            variant = row["io_variants"][0]["name"]
            output = taxonomy.migrate_action(row["policy_action_id"], taxonomy.LEGACY_REGISTRY_VERSION,
                                             io_variant=variant, parameters={"retained": 1})
            self.assertEqual(output["action_id"], row["policy_action_id"])
            self.assertEqual(output["io_variant"], variant)
            self.assertEqual(output["parameters"], {"retained": 1})

    def test_invalid_identity_and_capability_inputs_fail(self):
        for value in (True, -1, 154, 1.5, None):
            with self.assertRaises(ValueError):
                taxonomy.describe_action(value)
        with self.assertRaises(ValueError):
            taxonomy.migrate_action(143, "cpp-canonical-206-v1")
        with self.assertRaises(ValueError):
            taxonomy.migrate_action(144, taxonomy.LEGACY_REGISTRY_VERSION)
        with self.assertRaises(ValueError):
            taxonomy.migrate_action(143, taxonomy.LEGACY_REGISTRY_VERSION, parameters={"width": 4})
        with self.assertRaises(ValueError):
            taxonomy.migrate_action(0, taxonomy.LEGACY_REGISTRY_VERSION, io_variant="fake")
        with self.assertRaises(ValueError):
            taxonomy.availability(0, {"invented-model": True})
        with self.assertRaises(ValueError):
            taxonomy.availability(0, {taxonomy.ACTION_BY_ID[0].name: {"available": "yes"}})
        with self.assertRaises(ValueError):
            taxonomy.availability(65, mode="fake")
        with self.assertRaises(ValueError):
            taxonomy.availability(65, {taxonomy.ACTION_BY_ID[65].name: True, "op.split_columns.columnize.cpp_v1": False})

    def test_export_is_serializable_and_counts_do_not_claim_availability(self):
        report = taxonomy.registry()
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["counts"]["operator_families"], 30)
        self.assertEqual(report["counts"]["all_action_descriptors"], 154)
        self.assertEqual(report["counts"]["special_action_kinds"], 3)
        self.assertEqual(report["counts"]["selectable_actions"], 144)  # legacy143 + STOP, not new adapters
        self.assertFalse(report["columnize_reclassification"]["adds_algorithm"])
        self.assertEqual(report["legacy_stop_migration"], {"source_id": 143, "target_id": 151})


if __name__ == "__main__":
    unittest.main()
