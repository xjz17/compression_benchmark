#!/usr/bin/env python3
"""Current-registry regression retaining all 39 merged historical interfaces.

The immutable inputs and UAR1 outputs were captured with the 206 registry before
its replacement. Historical replay retains all 39 payloads; replacement RLE
profiles validate their new format separately. This never regenerates fixtures.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import unittest

import primitive_operator_action_runtime as primitive
import unified_action_runtime as unified


FIXTURES = Path(__file__).resolve().parent / "fixtures/canonical206_io_compat"
VERSION = primitive.ACTION_REGISTRY_VERSION
OLD_VERSION = "cpp-canonical-206-v1"
RLE_REPLACEMENTS = {
    "op.run_length.reference_v1": "op.run_length.byte_optimal.words64.cpp_v1",
    "op.run_length.bzip2_uint8.cpp_reference_v1": "op.run_length.byte_optimal.uint8.cpp_v1",
    "op.run_length.tsfile_int64.v1": "op.run_length.byte_optimal.words64.cpp_v1",
    "op.run_length.rlbe.cpp_v1": "op.run_length.byte_optimal.rlbe_lengths.cpp_v1",
}


def load_data(record: dict) -> unified.ActionData:
    raw = (FIXTURES / record["file"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["sha256"]:
        raise AssertionError(f"frozen fixture changed: {record['file']}")
    return unified.ActionData(record["kind"], raw, record["count"])


class Canonical182CompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((FIXTURES / "manifest.json").read_text())
        cls.expected = json.loads((FIXTURES / "migration_expectations.json").read_text())
        current = json.loads((FIXTURES.parents[2] / "benchmark/reference/actions/canonical127.json").read_text())
        cls.from137 = {old_id: group['policy_action_id'] for group in current['actions']
                       for old_id in group['previous_137_policy_ids']}
        cls.from137[137] = 127
        # Keep the immutable 182 oracle; apply only the explicitly audited
        # contiguous 75..101 parameter-configuration collapse in this test.
        for source, targets in (("from206", primitive.LEGACY_206_POLICY_ACTIONS),
                                ("from225", primitive.LEGACY_225_POLICY_ACTIONS)):
            for old_id, target in cls.expected[source].items():
                previous_id = target["policy_action_id"]
                target["policy_action_id"] = (previous_id if previous_id < 75 else
                                                75 if previous_id <= 101 else previous_id - 26)
                if target["policy_action_id"] >= 87:
                    target["policy_action_id"] = (87 if target["policy_action_id"] <= 100 else target["policy_action_id"] - 13)
                bitpack_id = target["policy_action_id"]
                target["policy_action_id"] = 25 if bitpack_id in (32,33,34) else bitpack_id-3 if bitpack_id>34 else bitpack_id
                for_id = target["policy_action_id"]
                target["policy_action_id"] = 14 if 14 <= for_id <= 17 else for_id-3 if for_id>17 else for_id
                target["parameters"] = {}
                if 75 <= previous_id <= 101:
                    import re
                    match = re.fullmatch(r"op\.lz_match\.native_lz\.aligned8\.min(\d+)\.max(\d+)",
                                         targets[int(old_id)].name)
                    assert match is not None
                    target["parameters"] = dict(zip(("min_match", "max_match"), map(int, match.groups())))
                    target["io_variant"] = "op.lz_match.native_lz.aligned8.cpp_v2"
                if 113 <= previous_id <= 126:
                    import re
                    match = re.fullmatch(r"op\.match_window_model\.native_lz\.window(\d+)", targets[int(old_id)].name)
                    assert match is not None
                    target["parameters"] = {"window_bytes": int(match.group(1))}
                    target["io_variant"] = "op.match_window_model.native_lz.window.cpp_v2"
                # Apply the published 137 crosswalk, keeping the frozen oracle
                # and its source packet fixtures unchanged.
                old137_id = target['policy_action_id']
                target['policy_action_id'] = cls.from137[old137_id]
                if 115 <= old137_id <= 120:
                    engine = 1 if old137_id <= 117 else 2
                    target['io_variant'] = f'op.block_state_control.blosc{engine}_residual_core.cpp_v2'
                    target['parameters'] = {'typesize': (1, 4, 8)[(old137_id - 115) % 3]}
                elif old137_id in (122, 123, 127):
                    mode, level = {122: ('gzip_prefix', 6), 123: ('zlib_prefix', 9),
                                   127: ('corad_ctk2', 9)}[old137_id]
                    target['io_variant'] = f'op.block_state_control.deflate_residual_core.{mode}.cpp_v2'
                    target['parameters'] = {'level': level}
                if target['io_variant'] in RLE_REPLACEMENTS:
                    target['io_variant'] = RLE_REPLACEMENTS[target['io_variant']]
                    target.update(migration_kind='implementation_replacement',
                                  source_registry_version=OLD_VERSION if source == 'from206' else 'cpp-canonical-225-v1',
                                  source_action_id=int(old_id), requires_remeasurement=True)
                elif target['io_variant'] == 'op.bitpack_int.sprintz.cpp_v1':
                    target['io_variant'] = 'op.bitpack_int.sprintz.unified_rle_tail.cpp_v2'
                    target.update(migration_kind='implementation_replacement',
                                  source_registry_version=OLD_VERSION if source == 'from206' else 'cpp-canonical-225-v1',
                                  source_action_id=int(old_id), requires_remeasurement=True)
        cls.records = cls.manifest["records"]
        cls.by_id = {item["old_policy_id"]: item for item in cls.records}
        cls.siblings = {}
        for group in cls.manifest["groups"]:
            members = group["member_ids"]
            for index, old_id in enumerate(members):
                cls.siblings[old_id] = members[(index + 1) % len(members)]

    def target(self, record: dict) -> tuple[int, str]:
        new_id = self.expected["from206"][str(record["old_policy_id"])]["policy_action_id"]
        return new_id, primitive.POLICY_ACTION_BY_ID[new_id].name

    def forward(self, record: dict, *, by_name: bool = False) -> unified.ActionData:
        new_id, name = self.target(record)
        return unified.run_policy_action(
            name if by_name else new_id,
            load_data(record["input"]),
            registry_version=VERSION,
            io_variant=RLE_REPLACEMENTS.get(record["old_canonical_name"], record["old_canonical_name"]),
            max_point=record["max_point"],
            output="wire",
        )

    def call_old_interface(self, old_id: int, source: unified.ActionData, **kwargs):
        migrated = primitive.migrate_policy_action(old_id, OLD_VERSION)
        if migrated.get("migration_kind") == "replacement_recipe":
            reverse = kwargs.pop("reverse", False)
            output = kwargs.pop("output", "wire")
            if kwargs:
                raise AssertionError(f"unexpected composite-recipe options: {kwargs}")
            current = source
            steps = tuple(zip(
                migrated["replacement_policy_action_ids"],
                migrated["replacement_io_variants"],
                migrated["replacement_parameters"],
            ))
            for action_id, variant, parameters in reversed(steps) if reverse else steps:
                current = unified.run_policy_action(
                    action_id, current, registry_version=VERSION,
                    io_variant=variant, parameters=parameters, reverse=reverse,
                )
            if output == "streams":
                from action_semantic_ports import to_ports
                return to_ports(current)
            return current
        target = self.expected["from206"][str(old_id)]
        return unified.run_policy_action(
            target["policy_action_id"], source, registry_version=VERSION,
            io_variant=target["io_variant"], **kwargs,
        )

    def test_immutable_fixture_coverage(self) -> None:
        self.assertEqual(self.manifest["registry_version"], OLD_VERSION)
        self.assertEqual(len(self.records), 39)
        self.assertEqual(len(self.by_id), 39)
        self.assertEqual(len(self.manifest["groups"]), 15)
        self.assertEqual(set(self.siblings), set(self.by_id))
        self.assertEqual(self.manifest["validation"]["old_forward_inverse_exact"], 39)
        self.assertTrue(self.manifest["validation"]["source_hashes_unchanged_during_capture"])
        for record in self.records:
            with self.subTest(old_id=record["old_policy_id"]):
                for field in ("root", "input", "encoded", "inverse", "codec_wire"):
                    load_data(record[field])
                self.assertEqual(load_data(record["input"]), load_data(record["inverse"]))
                header, wire = unified._unpack_result(load_data(record["encoded"]))
                self.assertEqual(header, record["uar1_header"])
                self.assertEqual(wire, load_data(record["codec_wire"]).payload)
                self.assertNotIn("policy_registry_version", header)

    def test_current_registry_preserves_retired_slot_and_stop(self) -> None:
        policy = primitive.POLICY_ACTIONS
        self.assertEqual(primitive.ACTION_REGISTRY_VERSION, VERSION)
        self.assertEqual(len(policy), 122)
        self.assertEqual(len(unified.CANONICAL_ACTIONS), 122)
        self.assertEqual(
            [item.action_id for item in policy],
            [i for i in range(127) if i not in {5, 18, 35, 87, 90}],
        )
        self.assertEqual(len({item.name for item in policy}), 122)
        self.assertEqual(primitive.FROZEN_POLICY_ACTION_BY_ID[5].name,
                         "op.scaling.reference_v1")
        self.assertEqual(len({item.operator_id for item in policy}), 28)
        self.assertEqual(primitive.STOP_ACTION_ID, 127)
        self.assertNotIn(129, primitive.POLICY_ACTION_BY_ID)
        for group in self.manifest["groups"]:
            expected_id = int(group["document_id"][1:]) - 1
            expected_id = expected_id if expected_id < 75 else expected_id - 26
            expected_id = expected_id if expected_id < 87 else expected_id - 13
            expected_id = 25 if expected_id in (32,33,34) else expected_id-3 if expected_id>34 else expected_id
            expected_id = 14 if 14 <= expected_id <= 17 else expected_id-3 if expected_id>17 else expected_id
            expected_id = self.from137[expected_id]
            old_names = {self.by_id[old_id]["old_canonical_name"] for old_id in group["member_ids"]}
            self.assertNotIn(primitive.POLICY_ACTION_BY_ID[expected_id].name, old_names)
            for old_id in group["member_ids"]:
                self.assertEqual(self.expected["from206"][str(old_id)]["policy_action_id"], expected_id)

    def test_complete_historical_id_maps(self) -> None:
        old206 = primitive.LEGACY_206_POLICY_ID_TO_CURRENT
        old225 = primitive.LEGACY_225_POLICY_ID_TO_CURRENT
        self.assertEqual(set(old206), set(range(207)))
        self.assertEqual(set(old225), set(range(226)))
        for old_id, target in self.expected["from206"].items():
            self.assertEqual(old206[int(old_id)], target["policy_action_id"], ("206", old_id))
        for old_id, target in self.expected["from225"].items():
            self.assertEqual(old225[int(old_id)], target["policy_action_id"], ("225", old_id))
        self.assertEqual(old206[206], 127)
        self.assertEqual(old225[225], 127)

    def test_explicit_206_migration_preserves_variant(self) -> None:
        for old_id, expected in self.expected["from206"].items():
            with self.subTest(old_id=old_id):
                if expected["policy_action_id"] == 5:
                    migrated = primitive.migrate_policy_action(int(old_id), registry_version=OLD_VERSION)
                    self.assertEqual(migrated["migration_kind"], "retired_identity_bypass")
                    self.assertEqual(migrated["replacement_policy_action_ids"], [])
                    continue
                migrated = primitive.migrate_policy_action(int(old_id), registry_version=OLD_VERSION)
                if int(old_id) == 206:
                    self.assertEqual(migrated, expected)
                    continue
                canonical = primitive.ACTION_CANONICAL_NAMES.get(
                    primitive.LEGACY_206_POLICY_ACTIONS[int(old_id)].name,
                    primitive.LEGACY_206_POLICY_ACTIONS[int(old_id)].name,
                )
                if canonical in primitive.COMPOSITE_POLICY_REPLACEMENTS:
                    replacement = primitive.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                    self.assertEqual(migrated["migration_kind"], "replacement_recipe")
                    self.assertEqual(migrated["replacement_policy_action_ids"],
                                     [item[0] for item in replacement])
                    self.assertEqual(migrated["replacement_io_variants"],
                                     [item[1] for item in replacement])
                    self.assertEqual(migrated["source_registry_version"], OLD_VERSION)
                else:
                    self.assertEqual(migrated, expected)

    def test_explicit_225_and_current_id_migration(self) -> None:
        for version in ("cpp-canonical-225-v1", "legacy-225-v1"):
            for old_id, expected in self.expected["from225"].items():
                with self.subTest(version=version, old_id=old_id):
                    if expected["policy_action_id"] == 5:
                        migrated = primitive.migrate_policy_action(int(old_id), registry_version=version)
                        self.assertEqual(migrated["migration_kind"], "retired_identity_bypass")
                        self.assertEqual(migrated["replacement_policy_action_ids"], [])
                        continue
                    migrated = primitive.migrate_policy_action(int(old_id), registry_version=version)
                    if int(old_id) == 225:
                        self.assertEqual(migrated, expected)
                        continue
                    source = (primitive.LEGACY_225_POLICY_ACTIONS[int(old_id)].name)
                    canonical = primitive.ACTION_CANONICAL_NAMES.get(source, source)
                    if canonical in primitive.COMPOSITE_POLICY_REPLACEMENTS:
                        replacement = primitive.COMPOSITE_POLICY_REPLACEMENTS[canonical]
                        self.assertEqual(migrated["migration_kind"], "replacement_recipe")
                        self.assertEqual(migrated["replacement_policy_action_ids"],
                                         [item[0] for item in replacement])
                        self.assertEqual(migrated["replacement_io_variants"],
                                         [item[1] for item in replacement])
                        self.assertEqual(migrated["source_registry_version"], version)
                    else:
                        self.assertEqual(
                            migrated,
                            dict(expected, source_registry_version=version)
                            if expected.get('requires_remeasurement') else expected,
                        )
        for current_id in range(128):
            if current_id in {5, 18, 35, 87, 90}:
                with self.assertRaises(ValueError):
                    primitive.migrate_policy_action(current_id, registry_version=VERSION)
                continue
            if current_id == 115:
                with self.assertRaises(ValueError):
                    primitive.migrate_policy_action(current_id, registry_version=VERSION)
                for profile, level in (('gzip_prefix', 6), ('zlib_prefix', 9), ('corad_ctk2', 9)):
                    migrated = primitive.migrate_policy_action(current_id, registry_version=VERSION,
                        io_variant=f'op.block_state_control.deflate_residual_core.{profile}.cpp_v2')
                    self.assertEqual(migrated['policy_action_id'], current_id)
                    self.assertEqual(migrated['parameters'], {'level': level})
                continue
            migrated = primitive.migrate_policy_action(current_id, registry_version=VERSION)
            self.assertEqual(migrated["policy_action_id"], current_id)
            self.assertEqual(migrated["parameters"],
                             {67: {"min_match": 8, "max_match": 256},
                              79: {"window_bytes": 65535},
                              112: {"typesize": 1}, 113: {"typesize": 1}}.get(current_id, {}))
        for invalid_id, version in ((True, OLD_VERSION), (-1, OLD_VERSION),
                                    (207, OLD_VERSION), (226, "legacy-225-v1"),
                                    (183, VERSION), (0, "unknown")):
            with self.subTest(invalid_id=invalid_id, version=version), self.assertRaises(ValueError):
                primitive.migrate_policy_action(invalid_id, registry_version=version)

    def test_all_39_historical_replays_keep_frozen_codec_payloads(self) -> None:
        for record in self.records:
            with self.subTest(old_id=record['old_policy_id']):
                source = load_data(record['input'])
                encoded = unified.run_policy_action(
                    record['old_policy_id'], source, registry_version=OLD_VERSION,
                    io_variant=record['old_canonical_name'], max_point=record['max_point'])
                header, wire = unified._unpack_result(encoded)
                self.assertEqual(wire, load_data(record['codec_wire']).payload)
                restored = unified.run_policy_action(
                    record['old_policy_id'], encoded, registry_version=OLD_VERSION, reverse=True)
                self.assertEqual(restored, source)

    def test_all_39_current_profiles_parity_or_explicit_replacement(self) -> None:
        for record in self.records:
            with self.subTest(old_id=record["old_policy_id"]):
                encoded = self.forward(record)
                self.assertEqual(encoded.kind, "bytes")
                self.assertEqual(self.forward(record, by_name=True), encoded)
                header, wire = unified._unpack_result(encoded)
                old_variant = record['old_canonical_name']
                variant = RLE_REPLACEMENTS.get(old_variant, old_variant)
                if old_variant in RLE_REPLACEMENTS:
                    self.assertEqual(wire[:4], b'URL1')
                    self.assertEqual(wire[4], 8 if old_variant == 'op.run_length.reference_v1' else 1)
                    self.assertNotEqual(wire, load_data(record['codec_wire']).payload)
                else:
                    self.assertEqual(wire, load_data(record["codec_wire"]).payload)
                _new_id, name = self.target(record)
                self.assertEqual(header["policy_algorithm"], name)
                self.assertEqual(header["io_variant"], variant)
                self.assertEqual(header["policy_registry_version"], VERSION)
                self.assertEqual(header["actions"], [variant] if old_variant in RLE_REPLACEMENTS
                                 else record["uar1_header"]["actions"])
                if "upstream_result_header" in record["uar1_header"]:
                    self.assertEqual(header["upstream_result_header"], record["uar1_header"]["upstream_result_header"])

    def test_all_39_new_wires_reverse_from_stored_variant(self) -> None:
        for record in self.records:
            with self.subTest(old_id=record["old_policy_id"]):
                encoded = self.forward(record)
                new_id, name = self.target(record)
                # Deliberately omit io_variant and max_point.  Inverse must use
                # the stored interface, even when all siblings accept bytes.
                for target in (new_id, name):
                    restored = unified.run_policy_action(
                        target, encoded, registry_version=VERSION, reverse=True,
                    )
                    self.assertEqual(restored, load_data(record["input"]))

    def test_129_to_127_crosswalk_marks_all_rle_replacements(self) -> None:
        expected_map = {old: old if old < 65 else 65 if old <= 67 else old - 2
                        for old in range(130)}
        self.assertEqual(primitive.LEGACY_129_POLICY_ID_TO_CURRENT, expected_map)
        cases = ((65, 'op.run_length.reference_v1'),
                 (65, 'op.run_length.bzip2_uint8.cpp_reference_v1'),
                 (66, 'op.run_length.tsfile_int64.v1'),
                 (67, 'op.run_length.rlbe.cpp_v1'))
        for old_id, variant in cases:
            with self.subTest(old_id=old_id, variant=variant):
                migrated = primitive.migrate_policy_action(
                    old_id, registry_version='cpp-canonical-129-v1', io_variant=variant)
                self.assertEqual(migrated, {
                    'policy_action_id': 65, 'io_variant': RLE_REPLACEMENTS[variant],
                    'parameters': {}, 'migration_kind': 'implementation_replacement',
                    'source_registry_version': 'cpp-canonical-129-v1',
                    'source_action_id': old_id, 'requires_remeasurement': True})

    def test_new_rle_all_four_profiles_and_float_raw_bits(self) -> None:
        integers = struct.pack('<6q', -350, -350, 0, 125, 125, 900)
        rls = unified.ActionData.bytes(b'RLS1' + struct.pack('<IQ', 2, 6) + integers)
        deltas = struct.pack('<6q', -9, 0, 6, 3, 1, 0)
        rld = unified.ActionData.bytes(b'RLD1' + struct.pack('<Q', 6) + deltas)
        ieee = (0, 1 << 63, 0x7ff0000000000000, 0xfff0000000000000,
                1, 0x7ff8000000000001, 0x7ff8000000001234, 0x7ff0000000000001)
        floats = unified.ActionData('float64', struct.pack('<8Q', *ieee), 8)
        cases = (
            ('words64', unified.ActionData('int64', integers, 6), b'URL1', 8),
            ('words64', floats, b'URL1', 8),
            ('uint8', unified.ActionData.bytes(b'aaabbb\0\0\0\xff'), b'URL1', 1),
            ('tsfile_scaled', rls, b'URS1', 8),
            ('rlbe_lengths', rld, b'URB1', 1),
        )
        for profile, source, magic, width in cases:
            variant = f'op.run_length.byte_optimal.{profile}.cpp_v1'
            with self.subTest(profile=profile, kind=source.kind):
                encoded = unified.run_policy_action(
                    65, source, registry_version=VERSION, io_variant=variant)
                header, wire = unified._unpack_result(encoded)
                self.assertEqual(wire[:4], magic)
                self.assertEqual(header['policy_action_id'], 65)
                self.assertEqual(header['policy_algorithm'], 'op.run_length.byte_optimal.cpp_v1')
                self.assertEqual(header['policy_registry_version'], VERSION)
                self.assertEqual(header['io_variant'], variant)
                self.assertEqual(header['actions'], [variant])
                core_offset = 0 if magic == b'URL1' else 8 if magic == b'URS1' else 28
                self.assertEqual(wire[core_offset:core_offset+4], b'URL1')
                self.assertEqual(wire[core_offset+4], width)
                self.assertEqual(unified.run_policy_action(
                    65, encoded, registry_version=VERSION, reverse=True), source)
                ports = unified.run_policy_action(
                    65, source, registry_version=VERSION, io_variant=variant, output='streams')
                self.assertEqual(unified.run_policy_action(
                    65, ports, registry_version=VERSION, reverse=True), source)

    def test_all_39_frozen_old_uar1_packets_remain_readable(self) -> None:
        for record in self.records:
            with self.subTest(old_id=record["old_policy_id"]):
                old_wire = load_data(record["encoded"])
                expected_input = load_data(record["input"])
                restored = unified.run_action(record["old_canonical_name"], old_wire, reverse=True)
                self.assertEqual(restored, expected_input)
                # The original packet intentionally has no registry metadata.
                # Its frozen version and ID disambiguate old RLE from URL1.
                restored = unified.run_policy_action(
                    record['old_policy_id'], old_wire, registry_version=OLD_VERSION, reverse=True)
                self.assertEqual(restored, expected_input)
                new_id, _name = self.target(record)
                if record['old_canonical_name'] in RLE_REPLACEMENTS:
                    with self.assertRaises(unified.ActionSequenceError):
                        unified.run_policy_action(new_id, old_wire, registry_version=VERSION, reverse=True)
                else:
                    restored = unified.run_policy_action(
                        new_id, old_wire, registry_version=VERSION, reverse=True)
                    self.assertEqual(restored, expected_input)

    def test_all_39_structured_ports_reverse_through_public_policy_api(self) -> None:
        from action_semantic_ports import ActionPorts

        for record in self.records:
            with self.subTest(old_id=record["old_policy_id"]):
                new_id, _name = self.target(record)
                source = load_data(record["input"])
                ports = unified.run_policy_action(
                    new_id, source, registry_version=VERSION,
                    io_variant=RLE_REPLACEMENTS.get(record["old_canonical_name"], record["old_canonical_name"]),
                    max_point=record["max_point"], output="streams",
                )
                self.assertIsInstance(ports, ActionPorts)
                self.assertTrue(ports.streams)
                # Pass the public result directly; inverse must recover the
                # selected legacy interface from the ports' stored metadata.
                restored = unified.run_policy_action(
                    new_id, ports, registry_version=VERSION, reverse=True,
                )
                self.assertEqual(restored, source)

    def test_reverse_rejects_wrong_sibling_variant(self) -> None:
        for record in self.records:
            with self.subTest(old_id=record["old_policy_id"]):
                wrong = self.by_id[self.siblings[record["old_policy_id"]]]["old_canonical_name"]
                new_id, _name = self.target(record)
                for encoded in (self.forward(record), load_data(record["encoded"])):
                    with self.assertRaises(unified.ActionRuntimeError):
                        unified.run_policy_action(
                            new_id, encoded, registry_version=VERSION,
                            io_variant=wrong, reverse=True,
                        )

    def test_version_unknown_variant_and_stop_rejections(self) -> None:
        record = self.records[0]
        source = load_data(record["input"])
        new_id, _name = self.target(record)
        for version in (OLD_VERSION, "old-225", "cpp-canonical-182-unknown"):
            with self.subTest(version=version), self.assertRaises(unified.ActionRuntimeError):
                # The old version itself is supported. Pairing a current ID
                # with an interface from a different old ID must be rejected.
                unified.run_policy_action(new_id, source, registry_version=version,
                                         io_variant=record['old_canonical_name'])
        for invalid_id in (-1, 129, 182, 206, 999):
            with self.subTest(invalid_id=invalid_id), self.assertRaises(unified.ActionRuntimeError):
                unified.run_policy_action(invalid_id, source, registry_version=VERSION)
        for variant in ("not-an-interface", self.by_id[133]["old_canonical_name"]):
            with self.subTest(variant=variant), self.assertRaises(unified.ActionRuntimeError):
                unified.run_policy_action(new_id, source, registry_version=VERSION, io_variant=variant)

    def test_bare_float_xor_packets_pack_and_reverse_without_upstream_header(self) -> None:
        source = unified.ActionData.floats([-19.25, -1.0, -0.0, 0.0, 0.125, 3.5, 17.25])
        for xor_id, pack_id, magic in ((97, 98, b"GXF1"), (99, 100, b"CXF1")):
            with self.subTest(xor_id=xor_id, pack_id=pack_id):
                xors = self.call_old_interface(xor_id, source)
                _, packet = unified._unpack_result(xors)
                self.assertEqual(packet[:4], magic)
                bare = unified.ActionData.bytes(packet)
                packed = self.call_old_interface(pack_id, bare)
                header, _wire = unified._unpack_result(packed)
                self.assertNotIn("upstream_result_header", header)
                new_id = self.expected["from206"][str(pack_id)]["policy_action_id"]
                recovered = unified.run_policy_action(new_id, packed, registry_version=VERSION, reverse=True)
                self.assertEqual(recovered, bare)
                # The same floating XOR payload also remains reversible in a
                # nested public chain carrying the original float64 kind.
                nested = self.call_old_interface(pack_id, xors)
                self.assertEqual(self.call_old_interface(pack_id, nested, reverse=True), xors)

    def test_scaling_supported_points_and_rejected_numeric_domain(self) -> None:
        source = unified.ActionData.floats([-1.0, 0.0, 1.0])
        for old_id in (80, 86):
            for point in range(19):
                with self.subTest(old_id=old_id, point=point):
                    encoded = self.call_old_interface(old_id, source, max_point=point)
                    self.assertEqual(self.call_old_interface(old_id, encoded, reverse=True), source)
            limit_source = unified.ActionData.floats(
                [-float(1 << 63), math.nextafter(float(1 << 63), -math.inf)]
            )
            with self.subTest(old_id=old_id, accepted_int64_limits=True):
                encoded = self.call_old_interface(old_id, limit_source, max_point=0)
                self.assertEqual(self.call_old_interface(old_id, encoded, reverse=True), limit_source)
            for point in (-1, 19):
                with self.subTest(old_id=old_id, bad_point=point), self.assertRaises(ValueError):
                    self.call_old_interface(old_id, source, max_point=point)
            for value, point in ((0.1 + 0.2, 2), (float("nan"), 2),
                                 (float("inf"), 2), (float("-inf"), 2),
                                 (float(1 << 63), 0),
                                 (math.nextafter(-float(1 << 63), -math.inf), 0),
                                 (-0.0, 2)):
                with self.subTest(old_id=old_id, value=value), self.assertRaises(ValueError):
                    self.call_old_interface(old_id, unified.ActionData.floats([value]), max_point=point)

    def test_signed_and_unsigned_for_do_not_cross_numeric_domains(self) -> None:
        from action_semantic_ports import from_ports

        def values(ports, name, kind):
            stream = next(s for s in ports.streams if s.name == name)
            self.assertEqual(stream.kind, kind)
            fmt = "<Q" if kind == "uint64" else "<q"
            return tuple(v[0] for v in struct.iter_unpack(fmt, stream.payload))

        edges = unified.ActionData.ints([-(1 << 63), 0, (1 << 63) - 1])
        reference = self.call_old_interface(2, edges, output="streams")
        self.assertEqual(values(reference, "main", "uint64"), (0, 1 << 63, (1 << 64) - 1))
        self.assertEqual(self.call_old_interface(2, reference, reverse=True), edges)

        bos_source = unified.ActionData.ints([0, -(1 << 63), -1])
        bos_delta = self.call_old_interface(111, bos_source)
        bos = self.call_old_interface(112, bos_delta, output="streams")
        self.assertEqual(values(bos, "base", "int64"), (-(1 << 63),))
        self.assertEqual(values(bos, "offsets", "uint64"), (0, (1 << 64) - 1))
        self.assertEqual(self.call_old_interface(112, from_ports(bos), reverse=True), bos_delta)

        sorted_values = self.call_old_interface(144, edges)
        reger_delta = self.call_old_interface(145, sorted_values)
        reger = self.call_old_interface(146, reger_delta, output="streams")
        self.assertEqual(values(reger, "base", "uint64"), ((1 << 64) - 2,))
        self.assertEqual(values(reger, "offsets", "uint64"), (1, 0))
        self.assertEqual(self.call_old_interface(146, reger, reverse=True), reger_delta)

    def test_auto_dispatch_rejects_same_kind_ambiguity(self) -> None:
        for old_id, data in ((80, unified.ActionData.floats([1.0])),
                             (158, unified.ActionData.bytes(bytes(range(17)))),
                             (163, unified.ActionData.bytes(b"payload")),
                             (170, unified.ActionData.bytes(b"payload")),
                             (7, unified.ActionData.ints([1, 2, 3])),
                             (8, unified.ActionData.ints([1, 2, 3]))):
            new_id = self.expected["from206"][str(old_id)]["policy_action_id"]
            with self.subTest(old_id=old_id), self.assertRaises(unified.ActionTypeError):
                unified.run_policy_action(new_id, data, registry_version=VERSION, max_point=2)

    def test_explicit_packet_profiles_reject_wrong_magic(self) -> None:
        source = unified.ActionData.bytes(b"BAD!" + bytes(128))
        for old_id in (112, 146, 91, 95, 102, 98, 100, 103, 108, 110,
                       59, 188, 133, 134, 135, 205):
            with self.subTest(old_id=old_id), self.assertRaises(ValueError):
                self.call_old_interface(old_id, source)

    def test_reverse_rejects_tampered_policy_identity(self) -> None:
        record = self.by_id[159]
        encoded = self.forward(record)
        header, wire = unified._unpack_result(encoded)
        new_id, _name = self.target(record)
        sibling = self.by_id[160]["old_canonical_name"]
        for field, value in (("policy_action_id", new_id + 1),
                             ("policy_action_id", str(new_id)),
                             ("policy_action_id", float(new_id)),
                             ("policy_action_id", True),
                             ("policy_action_id", None),
                             ("policy_algorithm", "not-a-policy-algorithm"),
                             ("policy_algorithm", None),
                             ("policy_registry_version", OLD_VERSION),
                             ("policy_registry_version", None),
                             ("io_variant", sibling)):
            tampered = dict(header)
            tampered[field] = value
            bad = unified._pack_existing_result(tampered, wire)
            with self.subTest(field=field, value=value), self.assertRaises(unified.ActionRuntimeError):
                unified.run_policy_action(new_id, bad, registry_version=VERSION, reverse=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
