#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

import native_experiment_scope_v1 as scope


class ScopeTests(unittest.TestCase):
    def test_exact_user_exclusions_keep_original_ids(self):
        value = scope.load()
        self.assertEqual(value['enabled_action_count'], 6277)
        self.assertEqual(value['enabled_fixed_configuration_count'], 191)
        self.assertEqual(len(value['excluded_action_ids']), 64)
        scope.verify_recipe([0, 1], value)
        with self.assertRaisesRegex(ValueError, 'user-excluded'):
            scope.verify_recipe([value['excluded_action_ids'][0]], value)

    def test_native_must_acknowledge_identical_profile(self):
        value = scope.load()
        scope.verify_native({'experiment_scope': value}, value)
        for altered in (None, {**value, 'profile_sha256': '0'*64},
                        {**value, 'enabled_action_count': 6341}):
            with self.assertRaisesRegex(ValueError, 'differs'):
                scope.verify_native({'experiment_scope': altered}, value)

    def test_profile_cannot_drop_an_exclusion_or_add_another(self):
        original = json.loads(scope.DEFAULT_PROFILE.read_bytes())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'profile.json'
            for ids in (original['excluded_action_ids'][:-1],
                        [0, *original['excluded_action_ids']]):
                p.write_text(json.dumps({**original, 'excluded_action_ids': ids}))
                with self.assertRaisesRegex(ValueError, 'IDs drift'):
                    scope.load(profile=p)
            p.write_text(json.dumps({**original, 'catalog_sha256': 'f'*64}))
            with self.assertRaisesRegex(ValueError, 'identity drift'):
                scope.load(profile=p)


if __name__ == '__main__':
    unittest.main()
