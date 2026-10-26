"""Local-only contract tests for the additive v1 operator harness registry."""
from __future__ import annotations

import json
import unittest
from unittest import mock

import operator_harness_registry_v1 as registry
import operator_implementation_catalog as catalog


class HarnessRegistryTests(unittest.TestCase):
    def test_all_layered_trace_ids_are_present_once(self):
        document = registry.load_registry()
        self.assertEqual(document["schema"], registry.SCHEMA)
        self.assertEqual(document["catalog_sha256"], catalog.MANIFEST_SHA256)
        self.assertEqual(document["operator_count"], 102)
        self.assertEqual(document["layer_counts"], {
            "mechanism": 67, "profile": 7, "recipe": 14, "adapter": 14,
        })
        self.assertEqual(
            {item["trace_id"] for item in document["operators"]}, set(catalog.BY_ID)
        )
        self.assertEqual(len(registry.registry_by_id(document)), 102)
        field_sets = {frozenset(item) for item in document["operators"]}
        self.assertEqual(len(field_sets), 1, "all 102 entries must use one OpSpec shape")
        required = {
            "trace_id", "layer", "mechanism_refs", "source_actions", "source_refs",
            "typed_port_status", "state", "resources", "gate", "versions", "hashes",
        }
        self.assertLessEqual(required, next(iter(field_sets)))

    def test_registry_is_deterministic_and_json_compatible(self):
        first = registry.load_registry()
        second = registry.load_registry()
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )
        self.assertEqual(json.loads(json.dumps(first, sort_keys=True)), first)
        self.assertRegex(first["registry_sha256"], r"^[0-9a-f]{64}$")
        for item in first["operators"]:
            self.assertEqual(item["versions"]["harness_schema"], registry.SCHEMA)
            self.assertEqual(item["hashes"]["catalog_sha256"], catalog.MANIFEST_SHA256)
            for key in (
                "catalog_entry_sha256", "source_refs_sha256",
                "resolved_source_files_sha256",
            ):
                self.assertRegex(item["hashes"][key], r"^[0-9a-f]{64}$")

    def test_only_explicit_representatives_are_bound(self):
        by_id = registry.registry_by_id()
        bound = tuple(
            item["trace_id"] for item in by_id.values() if item["action_bindings"]
        )
        self.assertEqual(bound, registry.REPRESENTATIVE_IDS)
        expected_actions = {
            "A013": "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            "A020": "op.bitpack_int.max_value_width.cpp_v1",
            "A023": "op.integer_code.uleb128.reference_values.cpp_v2",
            "A031": "op.xor.previous_word.semantic_ports.cpp_v1",
            "A046": "op.zigzag.signed64.raw.cpp_v2",
            "A048": "harness.a048.byte_planes.v1",
            "A077": "op.entropy_huffman.project_bytes.cpp_v1",
            "A080": "op.entropy_fse.zstd_rans.cpp_reference_v1",
            "A111": "op.outlier_separation.high_bits.uint32.cpp_v1",
        }
        for trace_id, action in expected_actions.items():
            item = by_id[trace_id]
            self.assertEqual(item["gate"]["status"], "bound_unmeasured")
            self.assertEqual(item["action_bindings"][0]["action"], action)
            self.assertEqual(
                item["action_bindings"][0]["evidence_status"], "registered_unmeasured"
            )
        self.assertEqual(by_id["A084"]["typed_port_status"], "unqualified")
        self.assertFalse(by_id["A084"]["gate"]["selectable"])
        self.assertEqual(
            by_id["A048"]["action_bindings"][0]["output_ports"][0]["kind"],
            "serial",
        )
        self.assertEqual(
            by_id["A048"]["action_bindings"][0]["output_port_multiplicity"],
            {
                "kind": "parameterized",
                "physical_expression": "max(1,ceil(bit_width/8))",
                "semantic_expression": "ceil(bit_width/8)",
                "zero_width_carrier": True,
            },
        )
        self.assertEqual(
            by_id["A111"]["action_bindings"][0]["output_port_multiplicity"],
            {"kind": "fixed", "count": 3},
        )
        for trace_id in ("A077", "A080"):
            binding = by_id[trace_id]["action_bindings"][0]
            self.assertEqual(binding["profiles"], ["*"])
            self.assertEqual(
                binding["input_ports"][0]["logical_view"],
                "byte_payload_any_physical_shape",
            )

    def test_nncp_is_non_overridable_and_never_representative(self):
        nncp = registry.registry_by_id()[registry.NNCP_ID]
        self.assertNotIn(registry.NNCP_ID, registry.REPRESENTATIVE_IDS)
        self.assertEqual(nncp["gate"]["status"], "disabled_by_policy")
        self.assertFalse(nncp["gate"]["selectable"])
        self.assertFalse(nncp["gate"]["overridable"])
        case = self._case(
            [registry.NNCP_ID],
            "op.prediction_probability.nncp.provider_v1",
            enable_disabled=True,
        )
        with self.assertRaisesRegex(ValueError, "disabled_by_policy"):
            registry.validate_case(case)

    def test_validate_case_accepts_bound_action_and_preserves_receipt_fields(self):
        case = self._case(
            ["A111"],
            "op.outlier_separation.high_bits.uint32.cpp_v1",
            parameters={"width": "12"},
            block_id="fixture-block",
        )
        validated = registry.validate_case(case)
        self.assertEqual(validated, case)
        self.assertIsNot(validated, case)
        self.assertEqual(validated["block_id"], "fixture-block")

    def test_validate_case_uses_frozen_registry_without_reloading_sources(self):
        frozen = registry.load_registry()
        case = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", profile="int64",
            input_root=self._root(8, "int64"),
        )
        with mock.patch.object(registry, "load_registry", side_effect=AssertionError("reload")):
            self.assertEqual(registry.validate_case(case, frozen), case)

    def test_validate_case_requires_exact_claim_action_accounting(self):
        unqualified = self._case(["A084"], "opaque.recipe")
        with self.assertRaisesRegex(ValueError, "unqualified"):
            registry.validate_case(unqualified)

        wrong_owner = self._case(
            ["A013"], "op.zigzag.signed64.raw.cpp_v2"
        )
        with self.assertRaisesRegex(ValueError, "not bound"):
            registry.validate_case(wrong_owner)

        unused = self._case(
            ["A013", "A046"],
            "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            profile="int64",
        )
        with self.assertRaisesRegex(ValueError, "unused"):
            registry.validate_case(unused)

        unknown_parameter = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", parameters={"width": "4"}
        )
        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            registry.validate_case(unknown_parameter)

        bad_profile = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", profile="float64"
        )
        with self.assertRaisesRegex(ValueError, "profile.*not declared"):
            registry.validate_case(bad_profile)

        for action in ("stop", "concat", "split_segments", "split"):
            with self.subTest(action=action):
                builtin_as_stage = self._case(["A046"], action, profile=None)
                with self.assertRaisesRegex(
                    ValueError, "must be represented by its recipe step kind"
                ):
                    registry.validate_case(builtin_as_stage)

        entropy_any_shape = self._case(
            ["A077"], "op.entropy_huffman.project_bytes.cpp_v1", profile="uint32"
        )
        self.assertEqual(registry.validate_case(entropy_any_shape), entropy_any_shape)

    def test_validate_case_accepts_a_multi_operator_reversible_recipe(self):
        case = self._case(
            ["A013", "A020"],
            "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            profile="int64",
            input_root=self._root(8, "int64", count=4),
        )
        case["recipe"]["steps"].insert(1, {
            "kind": "stage", "inputs": [2],
            "stage": {
                "action": "op.bitpack_int.max_value_width.cpp_v1",
                "profile": "uint64",
                "parameters": {"bit_order": "msb", "alignment_bytes": "1"},
            },
        })
        self.assertEqual(registry.validate_case(case), case)

    def test_typed_topology_rejects_invalid_or_reused_flow_ids(self):
        invalid = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", profile="int64",
            input_root=self._root(8, "int64"),
        )
        invalid["recipe"]["steps"][0]["inputs"] = [99]
        with self.assertRaisesRegex(ValueError, "input flow 99 is not active"):
            registry.validate_case(invalid)

        reused = self._case(
            ["A046", "A023"], "op.zigzag.signed64.raw.cpp_v2", profile="int64",
            input_root=self._root(8, "int64"),
        )
        reused["recipe"]["steps"].insert(1, {
            "kind": "stage", "inputs": [1],
            "stage": {
                "action": "op.integer_code.uleb128.reference_values.cpp_v2",
                "profile": "uint64", "parameters": {},
            },
        })
        reused["recipe"]["steps"].insert(2, {
            "kind": "stage", "inputs": [1],
            "stage": {
                "action": "op.integer_code.uleb128.reference_values.cpp_v2",
                "profile": "uint64", "parameters": {},
            },
        })
        with self.assertRaisesRegex(ValueError, "input flow 1 is not active"):
            registry.validate_case(reused)

    def test_typed_topology_rejects_width_signedness_and_missing_stop(self):
        wrong_width = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", profile="int64",
            input_root=self._root(4, "int64"),
        )
        with self.assertRaisesRegex(ValueError, "expects numeric width 8"):
            registry.validate_case(wrong_width)

        unsigned = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", profile="int64",
            input_root=self._root(8, "uint64"),
        )
        with self.assertRaisesRegex(ValueError, "expects logical view 'int64'"):
            registry.validate_case(unsigned)

        for_unsigned = self._case(
            ["A013"], "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            profile="int64", input_root=self._root(8, "uint64"),
        )
        with self.assertRaisesRegex(ValueError, "expects logical view 'int64'"):
            registry.validate_case(for_unsigned)

        missing_stop = self._case(
            ["A046"], "op.zigzag.signed64.raw.cpp_v2", profile="int64",
            input_root=self._root(8, "int64"),
        )
        missing_stop["recipe"]["steps"].pop()
        with self.assertRaisesRegex(ValueError, "terminal stop"):
            registry.validate_case(missing_stop)

    def test_typed_parameter_domains_are_checked_before_native_execution(self):
        for action, trace_id, profile, root, name, invalid in (
            ("op.outlier_separation.high_bits.uint32.cpp_v1", "A111", "uint32",
             self._root(4, "uint32"), "width", "33"),
            ("harness.a048.byte_planes.v1", "A048", "uint64",
             self._root(8, "uint64"), "bit_width", "65"),
            ("op.bitpack_int.max_value_width.cpp_v1", "A020", "uint64",
             self._root(8, "uint64"), "minimum_width", "65"),
        ):
            with self.subTest(action=action, parameter=name):
                case = self._case(
                    [trace_id], action, profile=profile, input_root=root,
                    parameters={name: invalid},
                )
                with self.assertRaisesRegex(ValueError, f"parameter '{name}'"):
                    registry.validate_case(case)

        for parameters, message in (
            ({"bit_order": "sideways"}, "bit_order"),
            ({"alignment_bytes": "2"}, "alignment_bytes"),
        ):
            case = self._case(
                ["A020"], "op.bitpack_int.max_value_width.cpp_v1",
                profile="uint64", input_root=self._root(8, "uint64"),
                parameters=parameters,
            )
            with self.assertRaisesRegex(ValueError, message):
                registry.validate_case(case)

    def test_split_and_concat_transfer_preserves_a_valid_entropy_input(self):
        case = self._case(
            ["A077"], "op.entropy_huffman.project_bytes.cpp_v1", profile="bytes",
            input_root=self._root(1, "bytes", kind="serial", count=4),
        )
        case["recipe"]["steps"] = [
            {"kind": "split_segments", "inputs": [0], "segment_sizes": [2, 2]},
            {"kind": "concat", "inputs": [1, 2]},
            case["recipe"]["steps"][0] | {"inputs": [3]},
            {"kind": "stop"},
        ]
        self.assertEqual(registry.validate_case(case), case)

    @staticmethod
    def _case(
        implementation_ids, action, parameters=None, profile="uint32", input_root=None, **extra
    ):
        result = {
            "case_id": "fixture",
            "implementation_ids": implementation_ids,
            "roots": [input_root or {
                "kind": "numeric", "width": 4, "logical_view": "uint32",
                "data_hex": "01000000",
            }],
            "recipe": {
                "policy_registry": registry.POLICY_REGISTRY,
                "steps": [
                    {
                        "kind": "stage", "inputs": [0],
                        "stage": {
                            "action": action, "profile": profile,
                            "parameters": parameters or {},
                        },
                    },
                    {"kind": "stop"},
                ],
            },
        }
        result.update(extra)
        return result

    @staticmethod
    def _root(width, logical_view, *, kind="numeric", count=1):
        return {
            "kind": kind, "width": width, "logical_view": logical_view,
            "data_hex": (b"\x00" * width * count).hex(),
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
