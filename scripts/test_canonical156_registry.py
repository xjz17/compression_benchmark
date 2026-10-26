"""Independent migration and exact-pipeline checks for parameterized LZ."""
import json
from pathlib import Path
import unittest

import native_lz_parameters as lz
import native_action_parameters as action_parameters
import primitive_operator_action_runtime as p


class RegistryParametersTest(unittest.TestCase):
    def test_every_previous_algorithm_and_io_profile_migrates(self):
        previous = json.loads((Path(__file__).parents[1] /
            'benchmark/reference/actions/canonical182.json').read_text())
        profiles = 0
        for old in previous['actions']:
            old_id = old['policy_action_id']
            expected_id = old_id if old_id < 75 else (75 if old_id <= 101 else old_id - 26)
            expected_id = expected_id if expected_id < 87 else (87 if expected_id <= 100 else expected_id - 13)
            expected_id = 25 if expected_id in (32,33,34) else expected_id-3 if expected_id>34 else expected_id
            expected_id = 14 if 14 <= expected_id <= 17 else expected_id-3 if expected_id>17 else expected_id
            for variant in old['io_variants']:
                with self.subTest(old_id=old_id, variant=variant['name']):
                    actual = p.migrate_policy_action(old_id, 'cpp-canonical-182-v1',
                                                     io_variant=variant['name'])
                    module = action_parameters.module_for_name(old['name'])
                    params = action_parameters.parameters_from_name(old['name'])
                    self.assertEqual(actual, {
                        'policy_action_id': expected_id,
                        'io_variant': module.ALGORITHM_NAME if module else variant['name'],
                        'parameters': params or {},
                    })
                    profiles += 1
        self.assertEqual(profiles, 206)
        self.assertEqual(p.migrate_policy_action(182, 'cpp-canonical-182-v1'),
                         {'policy_action_id': 137, 'io_variant': None, 'parameters': {}})

    def test_learning_grid_has_no_alignment_duplicates(self):
        choices = lz.parameter_candidates()
        self.assertEqual(len(choices), 147)
        effective = {(max(8, (c['min_match'] + 7) // 8 * 8),
                       c['max_match'] // 8 * 8) for c in choices}
        self.assertEqual(len(effective), len(choices))
        for choice in choices:
            self.assertEqual(lz.normalize_parameters(choice), choice)
        self.assertIn(lz.default_parameters(), choices)

    def test_direct_pipeline_uses_supplied_parameters(self):
        raw = b'ABCDEFGH' * 65
        actions = ('op.match_window_model.native_lz.window32768', lz.ALGORITHM_NAME,
                   'op.match_parse_policy.native_lz.aligned8_greedy',
                   'op.match_token_format.native_lz.ctrl8_offset16_len8.bias3')
        sizes = []
        for maximum in (16, 256):
            wire = p.encode_registered_exact_action_pipeline(raw, actions,
                parameters=(None, {'min_match': 8, 'max_match': maximum}, None, None))
            self.assertEqual(p._expand_native_tokens(p._unpack_native_wire(wire, 3, len(raw)),
                                                     len(raw)), raw)
            sizes.append(len(wire))
        # Eight initial literals followed by 32 short or 2 long matches.
        self.assertEqual(sizes, [109, 16])

    def test_benchmark_recipes_keep_historical_parameters(self):
        old = p.benchmark_86_legacy_action_pipelines()
        names = p.benchmark_86_canonical_action_pipelines()
        variants = p.benchmark_86_io_variant_pipelines()
        params = p.benchmark_86_action_parameters()
        def compare(a, b, c, d):
            if isinstance(a, dict):
                self.assertEqual(set(a), set(b)); self.assertEqual(set(a), set(c))
                self.assertEqual(set(a), set(d))
                for key in a: compare(a[key], b[key], c[key], d[key])
                return
            self.assertEqual(len(a), len(b)); self.assertEqual(len(a), len(c))
            self.assertEqual(len(a), len(d))
            for old_name, name, variant, settings in zip(a, b, c, d):
                module = action_parameters.module_for_name(old_name)
                expected = action_parameters.parameters_from_name(old_name)
                self.assertEqual(settings, expected or {})
                if expected:
                    self.assertEqual(name, module.ALGORITHM_NAME)
                    self.assertEqual(variant, module.ALGORITHM_NAME)
        compare(old, names, variants, params)


if __name__ == '__main__':
    unittest.main()
