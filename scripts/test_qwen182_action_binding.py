"""Tiny deterministic execution gates, not a repeat of the Qwen benchmark."""
import copy
import json
import struct
import unittest
import zlib
from unittest.mock import patch

import primitive_operator_action_runtime as p
import unified_action_runtime as u
import native_lz_parameters as lz
from canonical182_policy_paths import catalog_fingerprint, path_from_names, path_from_record, path_record
from canonical206_search_env import HEADER, decode_frame
from compact_pipeline_frame import WIRE_FORMAT, expand_frame
from qwen182_action_binding import (
    BINDING_VERSION, FLOAT_SCALE_ROOT_VARIANTS, LEGACY_VERSION, PREVIOUS_VERSION,
    ROOT_SEMANTICS, QwenActionBinding,
)


def old(ids):
    return {'ids': ids, 'policy_registry_version': LEGACY_VERSION}


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.binding = QwenActionBinding()
        self.raw = struct.pack('<16q', *range(-8, 8))

    def test_dynamic_root_inventory(self):
        roots = self.binding.candidate_steps()
        self.assertEqual(len(self.binding.algorithms), 122)
        active_variants = {
            variant for action in self.binding.algorithms.values()
            for variant in p.POLICY_IO_VARIANTS[action.name]
        }
        self.assertEqual(len(roots), len(active_variants))
        self.assertEqual(
            {row['ids'][0] for row in roots},
            set(range(127)) - {5, 18, 35, 87, 90},
        )
        self.assertEqual(len({row['io_variants'][0] for row in roots}),
                         len(active_variants))
        self.assertEqual({row['io_variants'][0] for row in roots}, active_variants)
        self.assertTrue(all(row['policy_registry_version'] == p.ACTION_REGISTRY_VERSION
                            for row in roots))
        self.assertTrue(all(row['binding_version'] == BINDING_VERSION for row in roots))
        lz_steps = [row for row in roots if row['io_variants'] == [lz.ALGORITHM_NAME]]
        self.assertEqual(len(lz_steps), 1)
        self.assertEqual(lz_steps[0]['parameters'], [lz.default_parameters()])

    def test_old_winner_migrates_and_current_frame_round_trips(self):
        recipe = self.binding.normalize_record(old([165, 169, 156, 75, 184]))
        self.assertEqual(recipe['ids'], [60, 72, 77, 97, 116])
        self.assertEqual(recipe['io_variants'],
                         [p.LEGACY_206_POLICY_ACTIONS[i].name for i in [165, 169, 156, 75, 184]])
        frame = self.binding.encode_recipe(self.raw, recipe, wire_format='legacy')
        self.assertEqual(decode_frame(frame), self.raw)
        self.assertEqual(HEADER.unpack_from(frame)[2], 5)
        meta, _ = u._unpack_result(u.ActionData.bytes(frame[HEADER.size:]))
        self.assertEqual(meta['policy_registry_version'], p.ACTION_REGISTRY_VERSION)
        self.assertEqual(meta['policy_action_id'], 116)
        self.assertEqual(meta['io_variant'], recipe['io_variants'][-1])

    def test_old137_paths_preserve_block_parameters_and_input_profiles(self):
        old_version = 'cpp-canonical-137-v1'
        blosc_records = []
        for old_id, current_id, typesize in (
                (115, 112, 1), (116, 112, 4), (117, 112, 8),
                (118, 113, 1), (119, 113, 4), (120, 113, 8)):
            with self.subTest(old_id=old_id):
                recipe = self.binding.normalize_record({
                    'ids': [old_id], 'policy_registry_version': old_version})
                self.assertEqual(recipe['ids'], [current_id])
                self.assertEqual(recipe['parameters'], [{'typesize': typesize}])
                self.assertEqual(recipe['policy_registry_version'], p.ACTION_REGISTRY_VERSION)
                self.assertEqual(self.binding.normalize_record(json.loads(json.dumps(recipe))), recipe)
                blosc_records.append(recipe)
        self.assertEqual(len({self.binding.cache_key(r) for r in blosc_records}), 6)
        for old_id, input_profile, level in (
                (122, 'gzip_prefix', 6), (123, 'zlib_prefix', 9), (127, 'corad_ctk2', 9)):
            with self.subTest(old_id=old_id):
                recipe = self.binding.normalize_record({
                    'ids': [old_id], 'policy_registry_version': old_version})
                self.assertEqual(recipe['ids'], [115])
                self.assertEqual(recipe['parameters'], [{'level': level}])
                self.assertEqual(recipe['io_variants'], [
                    f'op.block_state_control.deflate_residual_core.{input_profile}.cpp_v2'])
        shared = self.binding.normalize_record({
            'ids': [107], 'policy_registry_version': old_version})
        self.assertEqual(shared['ids'], [78])
        self.assertEqual(shared['io_variants'], ['op.block_state_control.reference_v1'])

    def test_old_reference_pair_groups_by_variant_not_current_id(self):
        recipe = self.binding.normalize_record(old([7, 11]))
        self.assertEqual(recipe['ids'], [30, 55])
        self.assertEqual(self.binding.reference_count(recipe), 2)
        frame = self.binding.encode_recipe(self.raw, recipe, wire_format='legacy')
        self.assertEqual(HEADER.unpack_from(frame)[2], 1)
        self.assertEqual(decode_frame(frame), self.raw)
        source = self.binding.root_data(self.raw, recipe)
        exact_compat = u.run_actions(recipe['io_variants'], source, max_point=None)
        expected_frame = HEADER.pack(b'C206', 1, 1, len(self.raw) // 8,
                                     zlib.crc32(self.raw)) + exact_compat.payload
        self.assertEqual(frame, expected_frame)
        new_id_7 = self.binding.normalize_record(old([80]))
        self.assertEqual(new_id_7['ids'], [7])
        self.assertEqual(self.binding.reference_count(new_id_7), 0)

    def test_reference_prefix_followed_by_nonreference_preserves_layers(self):
        recipe = self.binding.normalize_record(old([7, 11, 74]))
        self.assertEqual(self.binding.reference_count(recipe), 2)
        frame = self.binding.encode_recipe(self.raw, recipe, wire_format='legacy')
        self.assertEqual(HEADER.unpack_from(frame)[2], 2)
        self.assertEqual(decode_frame(frame), self.raw)

    def test_cache_identity_preserves_merged_variants(self):
        recipes = [self.binding.normalize_record(old([i])) for i in (158, 159, 160)]
        self.assertEqual(len({tuple(r['ids']) for r in recipes}), 1)
        self.assertEqual(len({self.binding.cache_key(r) for r in recipes}), 3)
        for recipe in recipes:
            self.assertEqual(self.binding.normalize_record(json.loads(json.dumps(recipe))), recipe)
            frame = self.binding.encode_recipe(self.raw, recipe, wire_format='legacy')
            self.assertEqual(decode_frame(frame), self.raw)
        path = self.binding.path(recipes[0])
        self.assertEqual(path_from_record(path_record(path)), path)

    def test_lz_parameters_reach_forward_inverse_payload_and_replay(self):
        raw = b'ABCDEFGH' * 65
        names = (
            'op.match_window_model.native_lz.window32768', lz.ALGORITHM_NAME,
            'op.match_parse_policy.native_lz.aligned8_greedy',
            'op.match_token_format.native_lz.ctrl8_offset16_len8.bias3',
        )
        records, frames, payloads = [], [], []
        for maximum in (16, 256):
            parameters = {'min_match': 8, 'max_match': maximum}
            recipe = self.binding._record(path_from_names(
                names, parameters=({}, parameters, {}, {})))
            original = u.run_policy_action
            with patch.object(u, 'run_policy_action', wraps=original) as tracked:
                frame = self.binding.encode_recipe(raw, recipe, wire_format='legacy')
            lz_calls = [call for call in tracked.call_args_list
                        if call.args[0] == p.POLICY_ACTION_BY_NAME[lz.ALGORITHM_NAME].action_id
                        and call.kwargs.get('parameters') is not None]
            self.assertEqual(len(lz_calls), 2)
            self.assertEqual([call.kwargs['parameters'] for call in lz_calls],
                             [parameters, parameters])
            self.assertTrue(lz_calls[1].kwargs['reverse'])
            _, payload = u._unpack_result(u.ActionData.bytes(frame[HEADER.size:]))
            self.assertEqual(self.binding.decode_recipe(frame), raw)
            persisted = json.loads(json.dumps(self.binding.input_record(raw, recipe)))
            self.assertEqual(self.binding.encode_recipe(raw, persisted, wire_format='legacy'), frame)
            records.append(persisted); frames.append(frame); payloads.append(payload)
        self.assertEqual(records[0]['ids'], records[1]['ids'])
        self.assertEqual(records[0]['io_variants'], records[1]['io_variants'])
        self.assertNotEqual(self.binding.cache_key(records[0]), self.binding.cache_key(records[1]))
        self.assertNotEqual(payloads[0], payloads[1])
        self.assertGreater(len(payloads[0]), len(payloads[1]))

    def test_every_old182_and_206_lz_choice_preserves_its_parameters(self):
        for version, actions in ((PREVIOUS_VERSION, p.LEGACY_182_POLICY_ACTIONS),
                                 (LEGACY_VERSION, p.LEGACY_206_POLICY_ACTIONS)):
            count = 0
            for spec in actions:
                parameters = lz.parameters_from_name(spec.name)
                if parameters is None:
                    continue
                with self.subTest(version=version, action=spec.action_id):
                    recipe = self.binding.normalize_record({
                        'ids': [spec.action_id], 'policy_registry_version': version})
                    self.assertEqual(recipe['ids'], [67])
                    self.assertEqual(recipe['io_variants'], [lz.ALGORITHM_NAME])
                    self.assertEqual(recipe['parameters'], [parameters])
                count += 1
            self.assertEqual(count, 27)

    def test_old182_lz_recipe_matches_historical_token_payload(self):
        raw = b'ABCDEFGH' * 65
        names = (
            'op.match_window_model.native_lz.window32768',
            'op.lz_match.native_lz.aligned8.min3.max18',
            'op.match_parse_policy.native_lz.aligned8_greedy',
            'op.match_token_format.native_lz.ctrl8_offset16_len8.bias3',
        )
        old_ids = {spec.name: spec.action_id for spec in p.LEGACY_182_POLICY_ACTIONS}
        recipe = self.binding.normalize_record({
            'ids': [old_ids[name] for name in names], 'policy_registry_version': PREVIOUS_VERSION})
        self.assertEqual(recipe['parameters'][1], {'min_match': 3, 'max_match': 18})
        frame = self.binding.encode_recipe(raw, recipe, wire_format='legacy')
        _, payload = u._unpack_result(u.ActionData.bytes(frame[HEADER.size:]))
        historical = raw
        for name in names:
            historical = p.encode_exact_native_lz_stage(name, historical)
        self.assertEqual(payload, historical)
        self.assertEqual(self.binding.decode_recipe(frame), raw)
        self.assertEqual(self.binding.encode_recipe(raw, json.loads(json.dumps(recipe)),
                                                   wire_format='legacy'), frame)

    def test_root_views_preserve_bytes_and_prior_priority(self):
        # Arbitrary words, including NaN and negative-zero bits. There is no
        # BF16 numeric expansion: these payload bytes must never be changed.
        raw = struct.pack('<4Q', 0x8000000000000000, 0x7ff8000000000001,
                          0xffffffffffffffff, 0x3f803f803f803f80)
        for old_id, expected in ((7, 'int64'), (79, 'float64'), (165, 'bytes')):
            root = self.binding.root_data(raw, old([old_id]))
            self.assertEqual(root.kind, expected)
            self.assertEqual(root.payload, raw)
            self.assertEqual(root.count, len(raw) if expected == 'bytes' else len(raw) // 8)
        # old BOS delta accepts bytes and int64; root must pick int64.
        both = next(step for step in self.binding.steps
                    if step.io_variants[0] == p.BOS64_INT_ACTIONS[0])
        self.assertEqual(self.binding.root_data(raw, self.binding._record(both)).kind, 'int64')

    def test_float_scaling_selectors_use_exact_legacy_variant_names(self):
        self.assertEqual(FLOAT_SCALE_ROOT_VARIANTS,
                         {p.LEGACY_206_POLICY_ACTIONS[i].name for i in (79, 80, 86, 93)})
        raw = struct.pack('<4d', 1., 2., 3., 4.)
        original = u.run_policy_action
        for old_id in (79, 80, 86, 93):
            with patch.object(u, 'run_policy_action', wraps=original) as tracked:
                frame = self.binding.encode_recipe(raw, old([old_id]), wire_format='legacy')
            forward = next(call for call in tracked.call_args_list
                           if not call.kwargs.get('reverse', False))
            self.assertEqual(forward.kwargs['max_point'], 0)
            self.assertIn(forward.kwargs['io_variant'], FLOAT_SCALE_ROOT_VARIANTS)
            self.assertEqual(decode_frame(frame), raw)

    def test_candidate_rule_is_broad_and_variant_based(self):
        reference = self.binding.normalize_record(old([7, 11]))
        following = self.binding.candidate_steps(reference)
        names = {r['io_variants'][0] for r in following}
        self.assertTrue(self.binding.reference_variants <= names)
        self.assertEqual(names, {
            variant for variant, desc in self.binding.descriptors.items()
            if 'bytes' in desc.accepted_input_kinds or variant in self.binding.reference_variants
        })
        after_nonref = {r['io_variants'][0] for r in self.binding.candidate_steps(old([165]))}
        self.assertEqual(after_nonref, {
            variant for variant, desc in self.binding.descriptors.items()
            if 'bytes' in desc.accepted_input_kinds
        })
        self.assertFalse(self.binding.reference_variants & after_nonref)

    def test_depth_limit_and_stop_rejection(self):
        self.assertEqual(self.binding.candidate_steps(old([74] * 10)), ())
        for bad in (0, 11, True, 1.5):
            with self.assertRaises(ValueError): QwenActionBinding(bad)
        for bad in (old([]), old([74] * 11), old([206]),
                    {'ids': [156], 'io_variants': ['STOP'],
                     'policy_registry_version': p.ACTION_REGISTRY_VERSION}):
            with self.assertRaises(ValueError): self.binding.normalize_record(bad)
        with self.assertRaises(ValueError):
            QwenActionBinding(2).normalize_record(old([7, 11, 74]))

    def test_unversioned_ids_and_ambiguous_or_wrong_variants_rejected(self):
        for bad in ([165], {'ids': [165]}, {'ids': [65], 'policy_registry_version': p.ACTION_REGISTRY_VERSION},
                    old([True]), old([1.5]), old([-1])):
            with self.assertRaises(ValueError): self.binding.normalize_record(bad)
        current = self.binding.normalize_record(old([158]))
        current['io_variants'] = self.binding.normalize_record(old([7]))['io_variants']
        with self.assertRaises(ValueError): self.binding.normalize_record(current)
        conflict = dict(old([7]), registry_version=p.ACTION_REGISTRY_VERSION)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.binding.normalize_record(conflict)

    def test_input_record_and_root_metadata_validation(self):
        recipe = self.binding.input_record(self.raw, old([165]))
        self.assertEqual(recipe['root_semantics'], ROOT_SEMANTICS)
        self.assertFalse(recipe['root_semantics']['numeric_cast'])
        self.assertEqual(recipe['input_binding']['root_kind'], 'bytes')
        self.assertEqual(recipe['input_binding']['original_bytes'], len(self.raw))
        self.assertEqual(self.binding.normalize_record(recipe), recipe)
        other_input = self.binding.input_record(bytes(len(self.raw)), old([165]))
        self.assertNotEqual(self.binding.cache_key(recipe), self.binding.cache_key(other_input))
        self.assertEqual(self.binding.decode_recipe(self.binding.encode_recipe(self.raw, recipe)), self.raw)
        with self.assertRaisesRegex(ValueError, 'input_binding'):
            self.binding.encode_recipe(bytes(len(self.raw)), recipe)
        changed = copy.deepcopy(recipe)
        changed['root_semantics']['numeric_cast'] = True
        with self.assertRaises(ValueError): self.binding.normalize_record(changed)
        with self.assertRaisesRegex(ValueError, 'max_depth'):
            QwenActionBinding(9).normalize_record(recipe)
        changed = copy.deepcopy(recipe)
        changed['binding_version'] = 'future-unknown'
        with self.assertRaises(ValueError): self.binding.normalize_record(changed)

    def test_input_alignment_and_frame_corruption_rejected(self):
        with self.assertRaises(ValueError): self.binding.root_data(b'123', old([7]))
        with self.assertRaises(ValueError): self.binding.encode_recipe(b'123', old([165]))
        frame = self.binding.encode_recipe(self.raw, old([74]), wire_format='legacy')
        with self.assertRaises(Exception): decode_frame(frame[:-1])
        corrupted = bytearray(frame)
        corrupted[14] ^= 1  # original CRC32 field, not codec payload
        with self.assertRaises(ValueError): decode_frame(bytes(corrupted))

    def test_default_compact_never_larger_and_expands_byte_identically(self):
        for recipe in (old([7, 11]), old([74]), old([165, 169, 156, 75, 184])):
            with self.subTest(recipe=recipe):
                legacy = self.binding.encode_recipe(self.raw, recipe, wire_format='legacy')
                compact = self.binding.encode_recipe(self.raw, recipe)
                self.assertLessEqual(len(compact), len(legacy))
                self.assertEqual(expand_frame(compact), legacy)
                self.assertEqual(self.binding.decode_recipe(compact), self.raw)
                self.assertEqual(self.binding.decode_recipe(legacy), self.raw)

    def test_new_storage_contract_changes_fingerprint_and_rejects_stale_records(self):
        recipe = self.binding.normalize_record(old([74]))
        self.assertEqual(recipe['storage_wire_format'], WIRE_FORMAT)
        self.assertEqual(recipe['binding_version'], BINDING_VERSION)
        previous_fingerprint = catalog_fingerprint({
            'binding_version': 'qwen-canonical182-bitviews-v1',
            'root_semantics': ROOT_SEMANTICS,
            'max_depth': self.binding.max_depth,
            'candidate_rule': 'all-roots-then-bytes-or-reference-prefix-v1',
        })
        self.assertNotEqual(self.binding.fingerprint, previous_fingerprint)
        for old_binding in ('qwen-canonical182-bitviews-v1', 'qwen-canonical182-bitviews-v2',
                            'qwen-canonical137-bitviews-parameters-v6'):
            stale_binding = dict(recipe, binding_version=old_binding)
            with self.assertRaisesRegex(ValueError, 'binding version'):
                self.binding.normalize_record(stale_binding)
        stale_storage = dict(recipe, storage_wire_format='c206-uncompressed-json-v1')
        with self.assertRaisesRegex(ValueError, 'storage wire format'):
            self.binding.normalize_record(stale_storage)
        # Old algorithm IDs remain migratable when no old binding/storage
        # contract is asserted; that does not retain their old size scores.
        self.assertEqual(self.binding.normalize_record(old([74]))['ids'], recipe['ids'])

    def test_unknown_wire_format_rejected_before_encoding(self):
        for invalid in ('gzip', '', None, 0):
            with self.subTest(wire_format=invalid):
                with self.assertRaisesRegex(ValueError, 'wire_format'):
                    self.binding.encode_recipe(self.raw, old([74]), wire_format=invalid)


if __name__ == '__main__':
    unittest.main()
