#!/usr/bin/env python3
"""Parameterized native-LZ execution, exact historical replay and metadata checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

import native_lz_parameters as lp
import primitive_operator_action_runtime as p
import unified_action_runtime as u
from action_semantic_ports import from_ports


class NativeLzParametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = u.ActionData.bytes(b"ABCDEFGHabcdefgh" * 80 + bytes(range(256)) * 2)
        cls.window_name = "op.match_window_model.native_lz.window32768"
        cls.parse_name = "op.match_parse_policy.native_lz.aligned8_greedy"
        cls.pack_names = tuple(f"op.match_token_format.native_lz.ctrl8_offset16_len8.bias{x}" for x in (3, 4, 5))
        cls.window = u.run_action(cls.window_name, cls.raw)
        cls.version = p.ACTION_REGISTRY_VERSION
        cls.action_id = p.POLICY_ACTION_BY_NAME[lp.ALGORITHM_NAME].action_id
        cls.old_manifest = json.loads((Path(__file__).resolve().parents[1] / "benchmark/reference/actions/canonical182.json").read_text())

    def encode(self, parameters=None, **kwargs):
        return u.run_policy_action(self.action_id, self.window,
            registry_version=self.version, parameters=parameters, **kwargs)

    def decode(self, data, parameters=None, **kwargs):
        return u.run_policy_action(self.action_id, data, registry_version=self.version,
            parameters=parameters, reverse=True, **kwargs)

    def test_one_algorithm_one_io_profile(self):
        matches = [item for item in p.POLICY_ACTIONS if item.name == lp.ALGORITHM_NAME]
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(p.POLICY_ACTIONS), 122)
        self.assertEqual(len(p.FROZEN_POLICY_ACTIONS), 127)
        self.assertEqual(p.STOP_ACTION_ID, 127)
        self.assertEqual(p.POLICY_IO_VARIANTS[lp.ALGORITHM_NAME], (lp.ALGORITHM_NAME,))
        self.assertEqual(len(u.IO_VARIANT_ACTIONS), 170)
        self.assertFalse(any(lp.parameters_from_name(item.name) for item in p.POLICY_ACTIONS))

    def test_dynamic_bounds_and_nls_packet_round_trip(self):
        for minimum, maximum in ((3, 8), (4, 110), (5, 111), (8, 256), (16, 40),
                                  (24, 258), (32, 104), (64, 256), (256, 258)):
            params = {"min_match": minimum, "max_match": maximum}
            with self.subTest(params=params):
                encoded = self.encode(params)
                header, payload = u._unpack_result(encoded)
                self.assertEqual(header["actions"], [lp.ALGORITHM_NAME])
                self.assertEqual(header["action_parameters"], params)
                context = p.native_lz_stage_packet_context(payload)
                self.assertEqual((context["min_match"], context["max_match"]), (minimum, maximum))
                self.assertEqual(self.decode(encoded), self.window)
                self.assertEqual(self.decode(encoded, params), self.window)

    def test_search_min_and_wire_bias_are_independent(self):
        for minimum in (8, 16, 32, 64):
            for bias_name in self.pack_names:
                with self.subTest(minimum=minimum, bias=bias_name):
                    matched = self.encode({"min_match": minimum, "max_match": 256})
                    parsed = u.run_action(self.parse_name, matched)
                    packed = u.run_action(bias_name, parsed)
                    header, wire = u._unpack_result(packed)
                    self.assertEqual(header["native_lz_stage_context"]["min_match"], minimum)
                    restored_tokens = u.run_action(bias_name, packed, reverse=True)
                    self.assertEqual(restored_tokens, parsed)
                    restored_match = u.run_action(self.parse_name, restored_tokens, reverse=True)
                    self.assertEqual(restored_match, matched)
                    self.assertEqual(self.decode(restored_match), self.window)
                    self.assertEqual(p._expand_native_tokens(
                        p._unpack_native_wire(wire, int(bias_name[-1]), self.raw.count), self.raw.count), self.raw.payload)

    def test_parallel_parameter_list_full_pipeline(self):
        actions = (self.window_name, lp.ALGORITHM_NAME, self.parse_name, self.pack_names[0])
        parameters = (None, {"min_match": 64, "max_match": 256}, None, None)
        packed = u.run_actions(actions, self.raw, parameters=parameters)
        self.assertEqual(u.run_actions(actions, packed, parameters=parameters, reverse=True), self.raw)
        single = u.run_actions((lp.ALGORITHM_NAME,), self.window, parameters=parameters[1])
        self.assertEqual(self.decode(single), self.window)
        with self.assertRaises(u.ActionSequenceError):
            u.run_actions(actions, self.raw, parameters=parameters[1])
        with self.assertRaises(u.ActionSequenceError):
            u.run_actions(actions, self.raw, parameters=(None,))

    def test_historical_names_preserve_exact_unrounded_bounds(self):
        old_names = [item.name for item in p.LEGACY_225_POLICY_ACTIONS
                     if lp.parameters_from_name(item.name) is not None]
        self.assertGreaterEqual(len(old_names), 27)
        _, window_wire = u._unpack_result(self.window)
        for name in old_names:
            with self.subTest(name=name):
                encoded = u.run_action(name, self.window)
                header, wire = u._unpack_result(encoded)
                params = lp.parameters_from_name(name)
                self.assertEqual(header["action_parameters"], params)
                self.assertEqual(wire, p.encode_exact_native_lz_stage(name, window_wire))
                self.assertEqual(u.run_action(name, encoded, reverse=True), self.window)
        alias = "op.lz_match.native_lz.aligned8.min3.max111"
        encoded = u.run_action(alias, self.window)
        self.assertEqual(u._unpack_result(encoded)[0]["action_parameters"]["max_match"], 111)
        with self.assertRaises(u.ActionSequenceError):
            u.run_action(alias, self.window, parameters={"min_match": 3, "max_match": 110})

    def test_old182_policy_packet_inverse(self):
        _, window_wire = u._unpack_result(self.window)
        old_groups = [g for g in self.old_manifest["actions"]
                      if lp.parameters_from_name(g["name"]) is not None]
        self.assertEqual(len(old_groups), 27)
        for group in old_groups:
            name = group["name"]
            with self.subTest(name=name):
                raw = p.encode_exact_native_lz_stage(name, window_wire)
                encoded = u._pack_result((name,), self.window, raw, None,
                    upstream_header=u._unpack_result(self.window)[0], stage_input_payload_count=len(window_wire),
                    extra_metadata={"policy_registry_version": "cpp-canonical-182-v1",
                                    "policy_algorithm": name, "policy_action_id": group["policy_action_id"],
                                    "io_variant": name})
                self.assertEqual(self.decode(encoded), self.window)

    def test_historical_uar_without_policy_metadata(self):
        name = "op.lz_match.native_lz.aligned8.min4.max110"
        _, window_wire = u._unpack_result(self.window)
        raw = p.encode_exact_native_lz_stage(name, window_wire)
        encoded = u._pack_result((name,), self.window, raw, None,
            upstream_header=u._unpack_result(self.window)[0], stage_input_payload_count=len(window_wire))
        self.assertEqual(self.decode(encoded), self.window)

    def test_typed_streams_rebuild_parameterized_stage(self):
        params = {"min_match": 16, "max_match": 256}
        ports = self.encode(params, output="streams")
        self.assertEqual(ports.header["action_parameters"], params)
        self.assertEqual(self.decode(ports), self.window)
        self.assertEqual(self.decode(from_ports(ports)), self.window)

    def test_invalid_parameters_and_wrong_algorithm(self):
        invalid = ({}, {"min_match": 8}, {"min_match": 8, "max_match": 256, "extra": 1},
                   {"min_match": True, "max_match": 256}, {"min_match": 8, "max_match": 256.0},
                   {"min_match": 2, "max_match": 256}, {"min_match": 8, "max_match": 259},
                   {"min_match": 17, "max_match": 23}, {"min_match": 64, "max_match": 32},
                   [], "8,256")
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(u.ActionRuntimeError):
                self.encode(params)
        with self.assertRaises(u.ActionTypeError):
            u.run_policy_action(self.window_name, self.raw, registry_version=self.version,
                                parameters={"min_match": 8, "max_match": 256})

    def test_header_parameter_integrity(self):
        encoded = self.encode({"min_match": 8, "max_match": 256})
        original, wire = u._unpack_result(encoded)
        for bad in (None, {}, [], "8,256", {"min_match": True, "max_match": 256},
                    {"min_match": 16, "max_match": 256}, {"min_match": 8, "max_match": 248}):
            header = deepcopy(original)
            header["action_parameters"] = bad
            with self.subTest(bad=bad), self.assertRaises(u.ActionRuntimeError):
                self.decode(u._pack_existing_result(header, wire))
        header = deepcopy(original); del header["action_parameters"]
        with self.assertRaises(u.ActionRuntimeError):
            self.decode(u._pack_existing_result(header, wire))
        header["actions"] = ["op.lz_match.native_lz.aligned8.min8.max256"]
        with self.assertRaises(u.ActionRuntimeError):
            self.decode(u._pack_existing_result(header, wire))
        with self.assertRaises(u.ActionRuntimeError):
            self.decode(encoded, {"min_match": 16, "max_match": 256})
        for field, value in (("policy_action_id", self.action_id + 1),
                             ("policy_action_id", True), ("policy_registry_version", None),
                             ("policy_registry_version", "cpp-canonical-unknown")):
            header = deepcopy(original); header[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(u.ActionRuntimeError):
                self.decode(u._pack_existing_result(header, wire))

    def test_old182_identity_and_parameter_metadata_agree(self):
        groups = [g for g in self.old_manifest["actions"]
                  if lp.parameters_from_name(g["name"]) is not None]
        first, other = groups[:2]
        _, window_wire = u._unpack_result(self.window)
        raw = p.encode_exact_native_lz_stage(first["name"], window_wire)
        encoded = u._pack_result((first["name"],), self.window, raw, None,
            upstream_header=u._unpack_result(self.window)[0],
            extra_metadata={"policy_registry_version": "cpp-canonical-182-v1",
                            "policy_algorithm": first["name"],
                            "policy_action_id": first["policy_action_id"],
                            "io_variant": first["name"]})
        original, wire = u._unpack_result(encoded)
        for field in ("policy_algorithm", "policy_action_id"):
            header = deepcopy(original)
            header[field] = other["name"] if field == "policy_algorithm" else other["policy_action_id"]
            with self.subTest(field=field), self.assertRaises(u.ActionRuntimeError):
                self.decode(u._pack_existing_result(header, wire))

    def test_old182_shifted_non_lz_id_still_decodes(self):
        group = next(g for g in self.old_manifest["actions"]
                     if g["name"] == "op.match_window_model.native_lz.window12288")
        current_id = p.POLICY_ACTION_BY_NAME[p.ACTION_CANONICAL_NAMES[group["name"]]].action_id
        self.assertNotEqual(current_id, group["policy_action_id"])
        encoded = u.run_policy_action(current_id, self.raw, registry_version=self.version,
            io_variant=group["io_variants"][0]["name"], parameters=None)
        header, wire = u._unpack_result(encoded)
        header.update(policy_registry_version="cpp-canonical-182-v1",
                      policy_action_id=group["policy_action_id"],
                      policy_algorithm=group["name"], actions=[group["name"]], io_variant=group["name"])
        header.pop("action_parameters")
        old = u._pack_existing_result(header, wire)
        self.assertEqual(u.run_policy_action(current_id, old, registry_version=self.version,
            reverse=True, parameters=None), self.raw)

    def test_default_is_frozen_calibration_choice(self):
        if not lp.DEFAULTS_PATH.is_file():
            self.skipTest("calibration is not yet available")
        encoded = self.encode()
        self.assertEqual(u._unpack_result(encoded)[0]["action_parameters"], lp.default_parameters())
        self.assertEqual(self.decode(encoded), self.window)


if __name__ == "__main__":
    unittest.main(verbosity=2)
