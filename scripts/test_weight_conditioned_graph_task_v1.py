#!/usr/bin/env python3
"""Focused executable tests for the GRPO typed-graph task adapter."""
from __future__ import annotations

import struct
import copy
import unittest
from unittest import mock

import numpy as np

import exception_separation
import nncp_entropy_codec
import operator_implementation_catalog
import prediction_entropy_codec
import residual_window_pack
import typed_graph_runtime as graph_runtime
import weight_conditioned_graph_task_v1 as graph_task_module
from weight_conditioned_graph_task_v1 import (
    ACTION_CATALOG,
    ACTION_COUNT,
    CONTINUATION_CANDIDATE_BUDGET,
    CandidateBudget,
    EvaluationBudgetExceeded,
    GraphTaskConfig,
    GraphTaskError,
    MAX_ACTIVE_STREAMS,
    EDIT_POSITION_ACTIONS,
    EDIT_TYPE_ACTIONS,
    REGISTRY_FINGERPRINT,
    RecipeEdit,
    STATE_CURRENT_OBJECTIVE_INDEX,
    STATE_FEATURE_DIM,
    STATE_FEATURE_SCHEMA,
    STOP_ACTION,
    TypedGraphCompressionTask,
    assert_formal_training_ready,
    catalog_report,
    find_action,
    mechanism_probe_report,
)
from weight_conditioned_grpo_v1 import FrozenPolicy, initialize_policy, propose_frozen
from weight_conditioned_search_v1 import _maybe_certified_prune
from weighted_objective_pruning_v1 import objective_lower_bound


def int_stream(values):
    return graph_runtime.Stream("numeric", 8, struct.pack(f"<{len(values)}q", *values))


class WeightConditionedGraphTaskTests(unittest.TestCase):
    def task(self, values=tuple(range(32)), **options):
        config = options.pop("config", GraphTaskConfig(
            max_policy_edits=6, independent_cold_validation=False
        ))
        return TypedGraphCompressionTask(
            options.pop("task_id", "ints"), [int_stream(values)], root_views=["int64"],
            compression_reference_ns_per_byte=10.0,
            decompression_reference_ns_per_byte=12.0,
            config=config, **options,
        )

    def test_global_catalog_is_fixed_and_unsupported_inventory_is_explicit(self):
        other = self.task(task_id="other", values=(7, 7, 9, 9))
        first = self.task()
        self.assertEqual(first.action_count, ACTION_COUNT)
        self.assertEqual(other.action_count, ACTION_COUNT)
        self.assertEqual(first.registry_fingerprint, REGISTRY_FINGERPRINT)
        self.assertEqual(other.registry_fingerprint, REGISTRY_FINGERPRINT)
        self.assertEqual(STOP_ACTION, first.stop_action)
        self.assertEqual(tuple(row.action_id for row in ACTION_CATALOG), tuple(range(ACTION_COUNT)))
        report = catalog_report()
        unsupported = {row["name"]: row["reason"]
                       for row in report["unsupported_v7_descriptors"]}
        self.assertNotIn("op.prediction_probability.ppm.provider_v1", unsupported)
        self.assertNotIn("op.prediction_probability.cmix.provider_v1", unsupported)
        self.assertNotIn("op.prediction_probability.nncp.provider_v1", unsupported)
        self.assertIn("cmix", report["prediction_entropy_capability"]["providers"])
        self.assertEqual(report["fixed_policy_action_count"], ACTION_COUNT)
        edit_space = report["continuation_edit_action_space"]
        self.assertEqual(edit_space["type_action_ids"], EDIT_TYPE_ACTIONS)
        self.assertEqual(edit_space["position_range"], [0, 64])
        self.assertEqual(edit_space["first_block_max_completed_edits"], 64)
        self.assertEqual(edit_space["continuation_max_completed_edits"], 2)
        self.assertIn("original root data flows", edit_space["replay_contract"])
        self.assertIn("different counts", report["claim_boundary"])
        self.assertIn("adjacent-XOR semantic-port", report["direct_bridge_constraints"]["A035"])
        self.assertFalse(any(
            row.category.endswith("bridge") for row in ACTION_CATALOG
        ))
        self.assertEqual(report["state_feature_schema"], STATE_FEATURE_SCHEMA)
        self.assertEqual(report["state_feature_dim"], STATE_FEATURE_DIM)
        internal = {
            row["adapter_runtime_action"]
            for row in report["internal_non_policy_adapters"]
        }
        self.assertEqual(internal, {
            "op.match_window_model.native_lz.window4096",
            "op.lz_match.blosclz.greedy.cpp_reference_v1",
            "op.block_transform.spdp.cpp_v1",
        })
        cmix = find_action("op.prediction_probability.cmix.provider_v1")
        self.assertNotIn(cmix, first.legal_actions(first.initial_state()))

    def test_formal_training_gate_requires_exhaustive_probe_receipt(self):
        report = catalog_report()
        coverage = report["mechanism_coverage"]
        self.assertEqual(coverage["trace_entry_count"], 102)
        self.assertEqual(coverage["mechanism_count"], 67)
        self.assertFalse(report["formal_training_ready"])
        self.assertEqual(coverage["omitted_selectable_mechanisms"], [])
        self.assertEqual(coverage["cataloged_mechanism_count"], 67)
        cataloged_ids = {
            row.implementation_id for row in ACTION_CATALOG
            if row.implementation_id is not None
        }
        self.assertEqual(
            cataloged_ids,
            {row["implementation_id"]
             for row in operator_implementation_catalog.MECHANISMS},
        )
        non_atomic_ids = {
            row["implementation_id"]
            for row in operator_implementation_catalog.IMPLEMENTATIONS
            if row["accounting"]["layer"] != "mechanism"
        }
        self.assertTrue(cataloged_ids.isdisjoint(non_atomic_ids))
        probes = mechanism_probe_report()
        self.assertEqual(probes["mechanism_count"], 67)
        for row in probes["mechanisms"]:
            if row["passed"]:
                self.assertTrue(any(probe["passed"] for probe in row["action_probes"]))
        failed = {
            row["implementation_id"]
            for row in coverage["probe_failed_mechanisms"]
        }
        self.assertTrue({"A006", "A007", "A008"}.isdisjoint(failed))
        self.assertEqual(
            "A109" not in failed,
            prediction_entropy_codec.cmix_receipt_status().get("current") is True,
        )
        if not nncp_entropy_codec.availability().get("graph_eligible"):
            self.assertIn("A110", failed)
        self.assertGreaterEqual(coverage["probe_passed_mechanism_count"], 65)
        reachability = {
            row["implementation_id"]
            for row in coverage["policy_reachability_blockers"]
        }
        self.assertTrue({"A035", "A060", "A067", "A070", "A081"}.isdisjoint(reachability))
        self.assertIn("adapter_exhaustive_probe", {
            row["kind"] for row in report["formal_training_blockers"]
        })
        upstream_blocked = {
            row["implementation_id"]
            for row in coverage["upstream_blocked_mechanisms"]
        }
        self.assertEqual(
            "A109" not in upstream_blocked,
            prediction_entropy_codec.availability()["providers"]["cmix"][
                "graph_eligible"
            ],
        )
        self.assertEqual(
            "A110" not in upstream_blocked,
            nncp_entropy_codec.availability().get("graph_eligible") is True,
        )
        with self.assertRaisesRegex(GraphTaskError, "readiness gate failed"):
            assert_formal_training_ready()
        with self.assertRaisesRegex(GraphTaskError, "readiness gate failed"):
            self.task(config=GraphTaskConfig(
                max_policy_edits=2,
                independent_cold_validation=False,
                require_formal_training_ready=True,
            ))

    def test_profile_and_recipe_bridges_are_internal_to_replayable_actions(self):
        task = self.task(values=tuple(range(64)))
        action = find_action(
            "op.lz_match.native_lz.aligned8.min3.max18", input_view="bytes"
        )
        self.assertIn(action, task.legal_actions(task.initial_state()))
        state = task.transition(task.initial_state(), action)
        assert state is not None
        self.assertEqual(state.recipe, (action,))
        self.assertEqual(
            [step["action"] for step in state.graph.steps],
            [
                "op.match_window_model.native_lz.window4096",
                "op.lz_match.native_lz.aligned8.min3.max18",
            ],
        )
        outcome = task.evaluate_terminal(state)
        self.assertEqual(outcome.operator_graph["recipe_action_ids"], [action])
        self.assertEqual(
            graph_runtime.restore_frame(outcome.compressed_payload)["s0"].data,
            int_stream(tuple(range(64))).data,
        )

        shuffle = find_action(
            "op.block_transform.blosc_shuffle.typesize1.cpp_reference_v1",
            input_view="bytes",
        )
        blosclz_tokens = find_action(
            "op.match_token_format.blosclz.ctrl8.cpp_reference_v1",
            input_view="bytes",
        )
        self.assertNotIn(blosclz_tokens, task.legal_actions(task.initial_state()))
        shuffled = task.transition(task.initial_state(), shuffle)
        assert shuffled is not None
        self.assertIn(blosclz_tokens, task.legal_actions(shuffled))
        tokenized = task.transition(shuffled, blosclz_tokens)
        assert tokenized is not None
        self.assertEqual(tokenized.recipe, (shuffle, blosclz_tokens))
        self.assertIn(
            "op.lz_match.blosclz.greedy.cpp_reference_v1",
            [step["action"] for step in tokenized.graph.steps],
        )

        spdp_delta = find_action("op.delta.spdp.cpp_v1", input_view="bytes")
        spdp_entropy = find_action(
            "op.entropy_special.spdp.cpp_v1", input_view="bytes"
        )
        delta_state = task.transition(task.initial_state(), spdp_delta)
        assert delta_state is not None
        self.assertIn(spdp_entropy, task.legal_actions(delta_state))
        entropy_state = task.transition(delta_state, spdp_entropy)
        assert entropy_state is not None
        self.assertEqual(entropy_state.recipe, (spdp_delta, spdp_entropy))
        self.assertIn(
            "op.block_transform.spdp.cpp_v1",
            [step["action"] for step in entropy_state.graph.steps],
        )

    def test_current_receipts_make_the_five_raw_policy_paths_formally_reachable(self):
        gate = copy.deepcopy(graph_task_module.operator_gate.build_report())
        for row in gate["entries"]:
            if row["layer"] == "mechanism":
                row["selectable"] = True
                row["status"] = "validated"
        by_id = {}
        preferred = {
            "A035": residual_window_pack.ELF_PLUS_ACTION,
            "A060": "op.lz_match.native_lz.aligned8.min3.max18",
            "A067": "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3",
            "A070": "op.match_token_format.blosclz.ctrl8.cpp_reference_v1",
            "A081": "op.entropy_special.spdp.cpp_v1",
        }
        for item in operator_implementation_catalog.MECHANISMS:
            iid = item["implementation_id"]
            candidates = [row for row in ACTION_CATALOG if row.implementation_id == iid]
            selected = next((row for row in candidates
                             if row.runtime_action == preferred.get(iid)), candidates[0])
            by_id[iid] = {
                "implementation_id": iid, "passed": True,
                "action_probes": [{
                    "runtime_action": selected.runtime_action,
                    "passed": True, "frame_bytes": 1,
                }],
            }
        probes = {
            "schema": "weight-conditioned-mechanism-probes-v1",
            "registry_fingerprint": REGISTRY_FINGERPRINT,
            "mechanisms": list(by_id.values()),
        }
        with mock.patch.object(graph_task_module.operator_gate, "build_report", return_value=gate), \
             mock.patch.object(graph_task_module, "mechanism_probe_report", return_value=probes):
            coverage = graph_task_module._mechanism_coverage()
        self.assertEqual(coverage["policy_reachability_blockers"], [])
        rows = {row["implementation_id"]: row for row in coverage["mechanisms"]}
        for iid in preferred:
            self.assertTrue(rows[iid]["raw_column_policy_reachability"])

    def test_a109_probe_uses_only_current_pinned_large_memory_receipt(self):
        spec = next(
            row for row in ACTION_CATALOG
            if row.implementation_id == "A109" and row.target_slot == 0
        )
        stale = {
            "current": False,
            "action": spec.runtime_action,
            "provider": "cmix",
            "frame": "PEC1",
            "frame_size": 66,
            "cold_decode_independent_process": True,
            "reason": "test stale receipt",
        }
        with mock.patch.object(
            prediction_entropy_codec, "cmix_receipt_status", return_value=stale
        ):
            record = graph_task_module._run_action_probe(spec)
        self.assertFalse(record["passed"])
        self.assertIn("stale", record["error"])

        current = dict(stale, current=True, reason="test current receipt")
        with mock.patch.object(
            prediction_entropy_codec, "cmix_receipt_status", return_value=current
        ):
            record = graph_task_module._run_action_probe(spec)
        self.assertTrue(record["passed"])
        self.assertEqual(
            record["evidence_kind"],
            "current_external_full_cmix_large_memory_receipt",
        )
        self.assertEqual(record["frame_bytes"], 66)

    def test_a110_probe_still_requires_current_nnp1_graph_receipt(self):
        spec = next(
            row for row in ACTION_CATALOG
            if row.implementation_id == "A110" and row.target_slot == 0
        )
        with mock.patch.object(
            nncp_entropy_codec, "availability",
            return_value={"graph_eligible": False, "reason": "test stale NNP1 receipt"},
        ):
            record = graph_task_module._run_action_probe(spec)
        self.assertFalse(record["passed"])
        self.assertIn("stale NNP1", record["error"])

    def test_validated_prediction_entropy_action_is_in_policy_catalog(self):
        task = self.task(config=GraphTaskConfig(
            max_policy_edits=2, independent_cold_validation=True
        ))
        action = find_action("op.prediction_probability.ppm.provider_v1")
        eligible = prediction_entropy_codec.availability()["providers"]["ppmd"][
            "graph_eligible"
        ]
        if not eligible:
            self.assertNotIn(action, task.legal_actions(task.initial_state()))
            return
        self.assertIn(action, task.legal_actions(task.initial_state()))
        state = task.transition(task.initial_state(), action)
        assert state is not None
        outcome = task.evaluate_terminal(state)
        self.assertEqual(graph_runtime.restore_frame(outcome.compressed_payload)["s0"],
                         int_stream(tuple(range(32))))

    def test_a110_complete_nnp1_action_is_cataloged_and_receipt_gated(self):
        task = self.task(config=GraphTaskConfig(
            max_policy_edits=2, independent_cold_validation=False
        ))
        action = find_action(nncp_entropy_codec.ACTION)
        eligible = nncp_entropy_codec.availability().get("graph_eligible") is True
        self.assertEqual(action in task.legal_actions(task.initial_state()), eligible)
        descriptor = task.action_record(action)
        self.assertEqual(descriptor.category, "mechanism")
        self.assertEqual(descriptor.implementation_id, "A110")
        self.assertEqual(descriptor.runtime_action, nncp_entropy_codec.ACTION)
        oversized = self.task(
            values=tuple(range(513)),
            config=GraphTaskConfig(max_policy_edits=2, independent_cold_validation=False),
        )
        self.assertNotIn(action, oversized.legal_actions(oversized.initial_state()))

    def test_real_integer_composition_replays_complete_frame_and_cold_restores(self):
        task = self.task(config=GraphTaskConfig(
            max_policy_edits=4, timing_repetitions=2,
            independent_cold_validation=True,
        ))
        state = task.initial_state()
        delta = find_action("op.delta.reference_v1", input_view="int64")
        huffman = find_action(
            "op.entropy_huffman.project_bytes.cpp_v1", input_view="bytes"
        )
        self.assertIn(delta, task.legal_actions(state))
        state = task.transition(state, delta)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertIn(huffman, task.legal_actions(state))
        state = task.transition(state, huffman)
        assert state is not None
        outcome = task.evaluate_terminal(state)
        self.assertTrue(outcome.compressed_payload.startswith(b"TGF1"))
        self.assertEqual(graph_runtime.restore_frame(outcome.compressed_payload)["s0"],
                         int_stream(tuple(range(32))))
        self.assertEqual(outcome.metrics.original_numeric_bytes, 32 * 8)
        self.assertGreaterEqual(outcome.metrics.compression_time_ns, 0)
        self.assertGreaterEqual(outcome.metrics.decompression_time_ns, 0)
        self.assertEqual(outcome.operator_graph["recipe_action_ids"], [delta, huffman])
        self.assertTrue(outcome.operator_graph["independent_cold_validation"])
        self.assertEqual(outcome.operator_graph["timing_warmups"], 1)
        self.assertEqual(outcome.operator_graph["timing_repetitions"], 2)
        self.assertEqual(outcome.operator_graph["timing_aggregation"], "median")
        self.assertEqual(len(outcome.operator_graph["compression_repeat_time_ns"]), 2)
        self.assertEqual(len(outcome.operator_graph["decompression_repeat_time_ns"]), 2)
        self.assertFalse(outcome.operator_graph[
            "cold_worker_launch_in_decompression_time"
        ])
        self.assertIn("excludes source-file reads", outcome.operator_graph[
            "compression_timing_boundary"])

    def test_decimal_scaling_max_point_is_cataloged_current_data_checked_and_cold(self):
        values = (0.0, 1.25, 2.5, 10.75, -3.5, 100.0)
        root = graph_runtime.Stream(
            "numeric", 8, struct.pack(f"<{len(values)}d", *values)
        )
        cases = (
            ("op.scaling.tsfile_rle.cpp_v1", None),
            ("op.scaling.exact_decimal.cpp_v2", "op.scaling.ts_2diff.cpp_v1"),
            ("op.scaling.delta_binary_packed.cpp_v1", None),
        )
        for index, (runtime_action, io_variant) in enumerate(cases):
            with self.subTest(runtime_action=runtime_action):
                task = TypedGraphCompressionTask(
                    f"scaling-{index}", [root], root_views=["float64"],
                    compression_reference_ns_per_byte=1.0,
                    decompression_reference_ns_per_byte=1.0,
                    config=GraphTaskConfig(
                        max_policy_edits=1, independent_cold_validation=True
                    ),
                )
                action = find_action(
                    runtime_action, input_view="float64", max_point=2,
                    io_variant=io_variant,
                )
                self.assertIn(action, task.legal_actions(task.initial_state()))
                state = task.transition(task.initial_state(), action)
                assert state is not None
                outcome = task.evaluate_terminal(state)
                self.assertEqual(
                    graph_runtime.restore_frame(outcome.compressed_payload),
                    {"s0": root},
                )
                self.assertEqual(
                    outcome.operator_graph["recipe"][0]["max_point"], 2
                )
                self.assertTrue(outcome.operator_graph["independent_cold_validation"])

        fractional = graph_runtime.Stream("numeric", 8, struct.pack("<d", 0.1))
        rejected = TypedGraphCompressionTask(
            "nonexact-scale", [fractional], root_views=["float64"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=1, independent_cold_validation=False),
        )
        fixed_zero = find_action(
            "op.scaling.exact_decimal.cpp_v2", input_view="float64",
            max_point=0, io_variant="op.scaling.ts_2diff.cpp_v1",
        )
        self.assertNotIn(fixed_zero, rejected.legal_actions(rejected.initial_state()))

    def test_explicit_valid_value_denominator_excludes_missing_bitmap(self):
        values = int_stream((11, 22, 33))
        missing = graph_runtime.Stream("serial", 1, b"\x04")
        task = TypedGraphCompressionTask(
            "values+missing", [values, missing], root_views=["int64", "bytes"],
            original_numeric_bytes=len(values.data),
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(
                max_policy_edits=2, independent_cold_validation=True
            ),
        )
        outcome = task.evaluate_terminal(task.initial_state())
        restored = graph_runtime.restore_frame(outcome.compressed_payload)
        self.assertEqual(restored, {"s0": values, "s1": missing})
        self.assertEqual(outcome.metrics.original_numeric_bytes, len(values.data))
        self.assertEqual(
            outcome.operator_graph["denominator_source"],
            "explicit_valid_numeric_value_bytes",
        )
        self.assertTrue(outcome.operator_graph[
            "complete_frame_includes_all_root_flows"
        ])
        with self.assertRaisesRegex(ValueError, "original_numeric_bytes"):
            TypedGraphCompressionTask(
                "bad-denominator", [values, missing], root_views=["int64", "bytes"],
                original_numeric_bytes=len(values.data) + 1,
                compression_reference_ns_per_byte=1.0,
                decompression_reference_ns_per_byte=1.0,
            )

    def test_uint64_lane_actions_mask_unaligned_missing_bitmap_but_keep_numeric_input(self):
        actions = (
            "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            "op.bitpack_int.max_value_width.cpp_v1",
        )
        values = int_stream(tuple(range(386)))
        for bitmap_bytes in (1, 49, 8):
            missing = graph_runtime.Stream("serial", 1, bytes(bitmap_bytes))
            task = TypedGraphCompressionTask(
                f"uint64-lanes+bitmap-{bitmap_bytes}", [values, missing],
                root_views=["int64", "bytes"], original_numeric_bytes=len(values.data),
                compression_reference_ns_per_byte=1.0,
                decompression_reference_ns_per_byte=1.0,
                config=GraphTaskConfig(
                    max_policy_edits=1, timing_warmups=0, timing_repetitions=1,
                    independent_cold_validation=True,
                ),
            )
            initial = task.initial_state()
            legal = task.legal_actions(initial)
            for name in actions:
                numeric_action = find_action(name, input_view="int64", target_slot=0)
                byte_action = find_action(name, input_view="bytes", target_slot=1)
                self.assertIn(numeric_action, legal)
                self.assertEqual(graph_task_module.ACTION_CATALOG[byte_action].input_byte_multiple, 8)
                if bitmap_bytes % 8:
                    self.assertNotIn(byte_action, legal)
                    selected = numeric_action
                else:
                    self.assertIn(byte_action, legal)
                    selected = byte_action
                state = task.transition(initial, selected)
                assert state is not None
                outcome = task.evaluate_terminal(state)
                self.assertEqual(graph_runtime.restore_frame(outcome.compressed_payload),
                                 {"s0": values, "s1": missing})
                self.assertTrue(outcome.operator_graph["independent_cold_validation"])

    def test_minimum_for_masks_float32_physical_width_but_preserves_byte_packing(self):
        values = graph_runtime.Stream(
            "numeric", 4, np.arange(772, dtype="<f4").tobytes()
        )
        missing = graph_runtime.Stream("serial", 1, bytes(97))
        task = TypedGraphCompressionTask(
            "float32+bitmap-97", [values, missing],
            root_views=["float32", "bytes"], original_numeric_bytes=len(values.data),
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(
                max_policy_edits=2, timing_warmups=0, timing_repetitions=1,
                independent_cold_validation=True,
            ),
        )
        initial = task.initial_state()
        legal = task.legal_actions(initial)
        minimum_for = find_action(
            "op.frame_of_reference.minimum.int64_uint64_le.cpp_v1",
            input_view="bytes", target_slot=0,
        )
        packing = find_action(
            "op.bitpack_int.max_value_width.cpp_v1",
            input_view="bytes", target_slot=0,
        )
        self.assertEqual(len(values.data) % 8, 0)
        self.assertNotIn(minimum_for, legal)
        self.assertEqual(ACTION_CATALOG[minimum_for].json_record()[
            "required_nonserial_width"
        ], 8)
        self.assertIsNone(ACTION_CATALOG[packing].required_nonserial_width)
        self.assertIn(packing, legal)
        state = task.transition(initial, packing)
        assert state is not None
        outcome = task.evaluate_terminal(state)
        self.assertEqual(graph_runtime.restore_frame(outcome.compressed_payload),
                         {"s0": values, "s1": missing})
        self.assertTrue(outcome.operator_graph["independent_cold_validation"])

    def test_bitsplit_rejects_float_deconstruct_struct3_but_keeps_native_widths(self):
        values = graph_runtime.Stream("numeric", 4, np.arange(8, dtype="<f4").tobytes())
        missing = graph_runtime.Stream("serial", 1, b"\x00")
        config = GraphTaskConfig(
            max_policy_edits=2, timing_warmups=0, timing_repetitions=1,
            independent_cold_validation=True,
        )
        task = TypedGraphCompressionTask(
            "float-deconstruct-struct3", [values, missing],
            root_views=["float32", "bytes"], original_numeric_bytes=len(values.data),
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0, config=config,
        )
        deconstruct = find_action("float_deconstruct", input_view="float32", target_slot=0)
        state = task.transition(task.initial_state(), deconstruct)
        assert state is not None
        self.assertEqual(tuple(state.graph.streams.values())[1].kind, "struct")
        self.assertEqual(tuple(state.graph.streams.values())[1].width, 3)
        rejected = find_action("bitsplit", input_view="bytes", target_slot=1)
        self.assertNotIn(rejected, task.legal_actions(state))
        self.assertEqual(ACTION_CATALOG[rejected].allowed_physical_widths, (1, 2, 4, 8))
        self.assertEqual(ACTION_CATALOG[rejected].json_record()[
            "allowed_physical_widths"
        ], (1, 2, 4, 8))
        self.assertEqual(graph_runtime.restore_frame(state.graph.stop()),
                         {"s0": values, "s1": missing})
        for kind in ("numeric", "struct"):
            for width in (2, 4, 8):
                root = graph_runtime.Stream(kind, width, bytes(range(width * 4)))
                legal_task = TypedGraphCompressionTask(
                    f"bitsplit-{kind}-{width}", [root], root_views=["bytes"],
                    compression_reference_ns_per_byte=1.0,
                    decompression_reference_ns_per_byte=1.0, config=config,
                )
                initial = legal_task.initial_state()
                action = find_action("bitsplit", input_view="bytes", target_slot=0)
                self.assertIn(action, legal_task.legal_actions(initial))
                following = legal_task.transition(initial, action)
                assert following is not None
                outcome = legal_task.evaluate_terminal(following)
                self.assertEqual(graph_runtime.restore_frame(outcome.compressed_payload),
                                 {"s0": root})
                self.assertTrue(outcome.operator_graph["independent_cold_validation"])

    def test_max_value_pack_masks_negative_signed_values_but_keeps_unsigned_byte_view(self):
        task = self.task(values=(-1, 0, 1, 2))
        initial = task.initial_state()
        name = "op.bitpack_int.max_value_width.cpp_v1"
        self.assertNotIn(find_action(name, input_view="int64"), task.legal_actions(initial))
        unsigned = find_action(name, input_view="bytes")
        self.assertIn(unsigned, task.legal_actions(initial))
        state = task.transition(initial, unsigned)
        assert state is not None
        outcome = task.evaluate_terminal(state)
        self.assertEqual(graph_runtime.restore_frame(outcome.compressed_payload)["s0"],
                         int_stream((-1, 0, 1, 2)))

    def test_pongo_trailing_pack_requires_actual_pongo_xor_parent(self):
        values = tuple(float(index) / 8.0 for index in range(32))
        root = graph_runtime.Stream("numeric", 8, struct.pack("<32d", *values))
        task = TypedGraphCompressionTask(
            "pongo", [root], root_views=["float64"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=4, independent_cold_validation=False),
        )
        eraser = find_action("op.eraser.pongo.cpp_v1", input_view="float64")
        xor = find_action("op.xor.pongo.cpp_v1", input_view="bytes")
        trailing = find_action(
            "op.trailing_zero_pack.pongo.cpp_v1", input_view="bytes"
        )
        state = task.initial_state()
        self.assertNotIn(xor, task.legal_actions(state))
        self.assertNotIn(trailing, task.legal_actions(state))
        state = task.transition(state, eraser)
        assert state is not None
        self.assertIn(xor, task.legal_actions(state))
        self.assertNotIn(trailing, task.legal_actions(state))
        state = task.transition(state, xor)
        assert state is not None
        self.assertIn(trailing, task.legal_actions(state))
        state = task.transition(state, trailing)
        assert state is not None
        self.assertEqual(
            graph_runtime.restore_frame(task.evaluate_terminal(state).compressed_payload),
            {"s0": root},
        )

    def test_direct_a111_and_a035_are_typed_multi_stream_actions(self):
        words32 = (1, 2, 3, 1025, 7, 0xffffffff)
        task32 = TypedGraphCompressionTask(
            "uint32", [graph_runtime.Stream(
                "numeric", 4, struct.pack(f"<{len(words32)}I", *words32)
            )], root_views=["uint32"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=2, independent_cold_validation=False),
        )
        a111 = find_action(exception_separation.ALGORITHM_NAME, parameter_value=8)
        state32 = task32.initial_state()
        self.assertIn(a111, task32.legal_actions(state32))
        state32 = task32.transition(state32, a111)
        assert state32 is not None
        self.assertEqual(len(state32.graph.streams), 3)
        self.assertEqual(set(state32.views().values()), {"uint32"})
        restored32 = graph_runtime.restore_frame(task32.evaluate_terminal(state32).compressed_payload)
        self.assertEqual(restored32["s0"].data, struct.pack(f"<{len(words32)}I", *words32))

        words64 = (0, 1, 0x100, 0x10000, 0xffffffffffffffff)
        raw_task64 = TypedGraphCompressionTask(
            "words64", [graph_runtime.Stream(
                "numeric", 8, struct.pack(f"<{len(words64)}Q", *words64)
            )], root_views=["bytes"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=2, independent_cold_validation=False),
        )
        a035 = find_action(residual_window_pack.ELF_PLUS_ACTION)
        self.assertNotIn(a035, raw_task64.legal_actions(raw_task64.initial_state()))
        task64 = TypedGraphCompressionTask(
            "residual64", [graph_runtime.Stream(
                "numeric", 8, struct.pack(f"<{len(words64)}Q", *words64)
            )], root_views=["xor_residual64"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=2, independent_cold_validation=False),
        )
        self.assertIn(a035, task64.legal_actions(task64.initial_state()))
        state64 = task64.transition(task64.initial_state(), a035)
        assert state64 is not None
        self.assertEqual(next(iter(state64.graph.streams.values())).kind, "serial")
        restored64 = graph_runtime.restore_frame(task64.evaluate_terminal(state64).compressed_payload)
        self.assertEqual(restored64["s0"].data, struct.pack(f"<{len(words64)}Q", *words64))

    def test_limits_are_enforced_before_execution(self):
        words32 = graph_runtime.Stream("numeric", 4, struct.pack("<8I", *range(8)))
        limited = TypedGraphCompressionTask(
            "limited", [words32], root_views=["uint32"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_active_streams=2, max_policy_edits=2,
                                   independent_cold_validation=False),
        )
        a111 = find_action(exception_separation.ALGORITHM_NAME, parameter_value=4)
        self.assertNotIn(a111, limited.legal_actions(limited.initial_state()))
        self.assertIsNone(limited.transition(limited.initial_state(), a111))

        depth_one = self.task(config=GraphTaskConfig(
            max_ordinary_depth=1, max_policy_edits=3, independent_cold_validation=False
        ))
        delta = find_action("op.delta.reference_v1", input_view="int64")
        state = depth_one.transition(depth_one.initial_state(), delta)
        assert state is not None
        huffman = find_action(
            "op.entropy_huffman.project_bytes.cpp_v1", input_view="bytes"
        )
        self.assertNotIn(huffman, depth_one.legal_actions(state))
        self.assertIn(STOP_ACTION, depth_one.legal_actions(state))

        self.assertEqual(MAX_ACTIVE_STREAMS, 32)

    def test_certified_frame_bound_excludes_live_payload_and_is_suffix_safe(self):
        task = self.task()
        initial = task.initial_state()
        lower0 = task.certified_bound_state(initial)
        frame0 = task.evaluate_terminal(initial).compressed_payload
        self.assertLessEqual(lower0.unavoidable_bytes, len(frame0))
        self.assertEqual(lower0.fixed_frame_bytes, 44)
        self.assertEqual(lower0.exhausted_leaf_logical_bits, ())
        delta = find_action("op.delta.reference_v1", input_view="int64")
        following = task.transition(initial, delta)
        assert following is not None
        lower1 = task.certified_bound_state(following)
        frame1 = task.evaluate_terminal(following).compressed_payload
        self.assertGreaterEqual(
            lower1.unavoidable_bytes, lower0.unavoidable_bytes
        )
        self.assertLessEqual(lower1.unavoidable_bytes, len(frame1))

    def test_edit_bound_keeps_better_deletion_completion_below_incumbent(self):
        zigzag = find_action("op.zigzag.signed64.raw.cpp_v2", input_view="int64")
        huffman = find_action(
            "op.entropy_huffman.project_bytes.cpp_v1", input_view="bytes"
        )
        task = self.task(
            values=(1, 2, 3, 4), previous_recipe=(zigzag, huffman),
            continuation_edit_mode=True,
            config=GraphTaskConfig(
                max_policy_edits=2, timing_warmups=0, timing_repetitions=1,
                independent_cold_validation=False,
            ),
        )
        initial = task.initial_state()
        pending = task.transition(initial, EDIT_TYPE_ACTIONS["delete"])
        assert pending is not None
        sibling = task.transition(pending, EDIT_POSITION_ACTIONS[0])
        assert sibling is not None
        incumbent = task.evaluate_terminal(sibling).compression_ratio

        branch = task.transition(pending, EDIT_POSITION_ACTIONS[1])
        assert branch is not None
        pending_second = task.transition(branch, EDIT_TYPE_ACTIONS["delete"])
        assert pending_second is not None
        completed = task.transition(pending_second, EDIT_POSITION_ACTIONS[0])
        assert completed is not None
        candidate = task.evaluate_terminal(completed)
        self.assertEqual(completed.recipe, ())
        self.assertLess(candidate.compression_ratio, incumbent)
        for state in (initial, pending, branch, pending_second, completed):
            bound = task.certified_bound_state(state)
            self.assertLessEqual(bound.unavoidable_bytes, len(candidate.compressed_payload))
            self.assertLess(objective_lower_bound(bound, (1, 0, 0)), incumbent)
            # Beam and policy-guided search use this same certified-prune hook.
            self.assertFalse(_maybe_certified_prune(
                task, state, (1, 0, 0), incumbent,
                lambda current_task, current_state: current_task.certified_bound_state(current_state),
                True,
            ))
        replacing = task.transition(initial, EDIT_TYPE_ACTIONS["replace"])
        assert replacing is not None
        self.assertEqual(task.certified_bound_state(replacing),
                         task.certified_bound_state(initial))

    def test_candidate_budget_and_previous_recipe_amortization(self):
        delta = find_action("op.delta.reference_v1", input_view="int64")
        budget = CandidateBudget(1)
        first = self.task(candidate_budget=budget)
        state = first.transition(first.initial_state(), delta)
        assert state is not None
        first.evaluate_terminal(state)
        with self.assertRaises(EvaluationBudgetExceeded):
            first.evaluate_terminal(state)
        first.reset_evaluation_budget()
        first.evaluate_terminal(state)

        continuation = first.continuation_task(
            "next", [int_stream(tuple(range(100, 132)))], root_views=["int64"],
            previous_recipe=(delta,), max_local_edits=2,
        )
        root = continuation.initial_state()
        self.assertEqual(root.recipe, (delta,))
        self.assertEqual(root.replayed_nodes, 1)
        self.assertEqual(root.policy_edits, 0)
        self.assertEqual(continuation.max_steps, 7)
        self.assertEqual(continuation.candidate_budget.snapshot(),
                         (0, CONTINUATION_CANDIDATE_BUDGET))
        self.assertEqual(graph_runtime.restore_frame(
            continuation.evaluate_terminal(root).compressed_payload
        )["s0"].data, int_stream(tuple(range(100, 132))).data)
        huffman = find_action(
            "op.entropy_huffman.project_bytes.cpp_v1", input_view="bytes"
        )
        self.assertEqual(
            TypedGraphCompressionTask.edit_recipe(
                (delta,), (RecipeEdit("insert", 1, huffman),)
            ),
            (delta, huffman),
        )
        self.assertEqual(
            TypedGraphCompressionTask.edit_recipe(
                (delta, huffman), (RecipeEdit("delete", 1),)
            ),
            (delta,),
        )
        zigzag = find_action("op.zigzag.signed64.raw.cpp_v2", input_view="int64")
        self.assertEqual(
            TypedGraphCompressionTask.edit_recipe(
                (delta,), (RecipeEdit("replace", 0, zigzag),)
            ),
            (zigzag,),
        )

        words32 = graph_runtime.Stream("numeric", 4, struct.pack("<8I", *range(8)))
        base = TypedGraphCompressionTask(
            "uint32-base", [words32], root_views=["uint32"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=2, independent_cold_validation=False),
        )
        width4 = find_action(exception_separation.ALGORITHM_NAME, parameter_value=4)
        width8 = find_action(exception_separation.ALGORITHM_NAME, parameter_value=8)
        tuned = base.continuation_task(
            "uint32-next", [words32], root_views=["uint32"],
            previous_recipe=(width4,), max_local_edits=2,
            previous_edits=(RecipeEdit("parameter_change", 0, width8),),
        )
        self.assertEqual(tuned.previous_recipe, (width8,))
        self.assertEqual(tuned.initial_state().replayed_nodes, 1)
        self.assertEqual(
            graph_runtime.restore_frame(
                tuned.evaluate_terminal(tuned.initial_state()).compressed_payload
            )["s0"], words32,
        )

    def test_continuation_edits_are_real_hierarchical_policy_actions(self):
        delta = find_action("op.delta.reference_v1", input_view="int64")
        huffman = find_action(
            "op.entropy_huffman.project_bytes.cpp_v1", input_view="bytes"
        )
        zigzag = find_action("op.zigzag.signed64.raw.cpp_v2", input_view="int64")
        parent = self.task(config=GraphTaskConfig(
            max_policy_edits=3, independent_cold_validation=False
        ))

        insert_task = parent.continuation_task(
            "insert", [int_stream(tuple(range(32)))], root_views=["int64"],
            previous_recipe=(delta,), max_local_edits=2,
        )
        state = insert_task.initial_state()
        self.assertIn(EDIT_TYPE_ACTIONS["insert"], insert_task.legal_actions(state))
        self.assertNotIn(huffman, insert_task.legal_actions(state))
        state = insert_task.transition(state, EDIT_TYPE_ACTIONS["insert"])
        assert state is not None
        self.assertIn(EDIT_POSITION_ACTIONS[1], insert_task.legal_actions(state))
        state = insert_task.transition(state, EDIT_POSITION_ACTIONS[1])
        assert state is not None
        self.assertIn(huffman, insert_task.legal_actions(state))
        state = insert_task.transition(state, huffman)
        assert state is not None
        self.assertEqual(state.recipe, (delta, huffman))
        self.assertEqual(state.edit_history, (("insert", 1, huffman),))

        state = insert_task.transition(state, EDIT_TYPE_ACTIONS["delete"])
        assert state is not None
        state = insert_task.transition(state, EDIT_POSITION_ACTIONS[1])
        assert state is not None
        self.assertEqual(state.recipe, (delta,))
        self.assertEqual(state.edit_count, 2)
        self.assertEqual(insert_task.legal_actions(state), (STOP_ACTION,))
        outcome = insert_task.evaluate_terminal(state)
        self.assertEqual(outcome.operator_graph["starting_recipe_action_ids"], [delta])
        self.assertEqual(outcome.operator_graph["completed_recipe_edits"], 2)
        self.assertEqual(len(outcome.operator_graph["edit_history"]), 2)

        replace_task = parent.continuation_task(
            "replace", [int_stream(tuple(range(32)))], root_views=["int64"],
            previous_recipe=(delta,), max_local_edits=2,
        )
        state = replace_task.transition(
            replace_task.initial_state(), EDIT_TYPE_ACTIONS["replace"]
        )
        assert state is not None
        state = replace_task.transition(state, EDIT_POSITION_ACTIONS[0])
        assert state is not None
        self.assertIn(zigzag, replace_task.legal_actions(state))
        state = replace_task.transition(state, zigzag)
        assert state is not None
        self.assertEqual(state.recipe, (zigzag,))

        words32 = graph_runtime.Stream("numeric", 4, struct.pack("<8I", *range(8)))
        base32 = TypedGraphCompressionTask(
            "parameter-parent", [words32], root_views=["uint32"],
            compression_reference_ns_per_byte=1.0,
            decompression_reference_ns_per_byte=1.0,
            config=GraphTaskConfig(max_policy_edits=2, independent_cold_validation=False),
        )
        width4 = find_action(exception_separation.ALGORITHM_NAME, parameter_value=4)
        width8 = find_action(exception_separation.ALGORITHM_NAME, parameter_value=8)
        parameter_task = base32.continuation_task(
            "parameter", [words32], root_views=["uint32"],
            previous_recipe=(width4,), max_local_edits=2,
        )
        state = parameter_task.transition(
            parameter_task.initial_state(), EDIT_TYPE_ACTIONS["parameter_change"]
        )
        assert state is not None
        state = parameter_task.transition(state, EDIT_POSITION_ACTIONS[0])
        assert state is not None
        self.assertIn(width8, parameter_task.legal_actions(state))
        self.assertNotIn(width4, parameter_task.legal_actions(state))
        state = parameter_task.transition(state, width8)
        assert state is not None
        self.assertEqual(state.recipe, (width8,))
        self.assertEqual(parameter_task.state_features(state).shape, (STATE_FEATURE_DIM,))
        policy = initialize_policy([parameter_task], seed=11, hidden_sizes=(8, 8))
        self.assertEqual(policy.identity.action_count, ACTION_COUNT)

    def test_features_are_numeric_and_weights_only_enter_in_grpo_core(self):
        task = self.task()
        static = task.task_features()
        dynamic = task.state_features(task.initial_state())
        self.assertEqual(static.shape, (18,))
        self.assertEqual(dynamic.shape, (STATE_FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(static)))
        self.assertTrue(np.all(np.isfinite(dynamic)))
        policy = initialize_policy([task], seed=3, hidden_sizes=(8, 8))
        self.assertEqual(policy.identity.input_dim, static.size + dynamic.size + 3)
        frozen = FrozenPolicy(policy)
        proposal = propose_frozen(frozen, task, (0.4, 0.3, 0.3), greedy=True)
        self.assertEqual(proposal.weight, (0.4, 0.3, 0.3))
        self.assertIsInstance(proposal.compressed_payload, bytes)

    def test_state_features_include_measured_j_and_every_flow_parent_byte_hash(self):
        task = self.task(objective_weight=(1.0, 0.0, 0.0))
        state = task.initial_state()
        before = task.state_features(state)
        self.assertEqual(before[STATE_CURRENT_OBJECTIVE_INDEX], -1.0)
        outcome = task.evaluate_terminal(state)
        after = task.state_features(state)
        expected_j = len(outcome.compressed_payload) / (
            task.original_numeric_bytes
            + len(outcome.compressed_payload)
        )
        self.assertAlmostEqual(after[STATE_CURRENT_OBJECTIVE_INDEX], expected_j)
        # Twelve trailing values are six matching parent/byte hash buckets.
        self.assertGreater(float(np.sum(after[-12:-6])), 0.0)
        self.assertGreater(float(np.sum(after[-6:])), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
