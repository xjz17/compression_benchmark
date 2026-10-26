"""Current RLE recipes, full frames and policy identity (bounded integration)."""
import copy
import struct
import subprocess
import sys
import unittest
from pathlib import Path

import optimal_rle_actions as r
import primitive_operator_action_runtime as p
import unified_action_runtime as u
from canonical182_policy_paths import path_from_names, path_from_record, path_record
from canonical182_policy_checkpoint import validate
from canonical206_search_env import Env, REF_VARIANTS, encode_cold, decode_frame, make_frame
from canonical206_recipe_grpo import initialize_policy
from qwen182_action_binding import QwenActionBinding

ROOT = Path(__file__).resolve().parents[1]


class CurrentRlePolicyTests(unittest.TestCase):
    def test_contextual_recipe_upgrade_and_explicit_current_contract(self):
        names = ('op.scaling.tsfile_rle.cpp_v1', 'op.run_length.tsfile_int64.v1')
        expected = path_from_names((names[0], r.SCALED_VARIANT))
        self.assertEqual(path_from_names(names), expected)
        self.assertEqual(path_from_record({'ids': [6, 66], 'policy_registry_version':
                                          'cpp-canonical-129-v1'}), expected)
        self.assertEqual(path_from_record(path_record(expected)), expected)
        # Explicit raw words is never guessed to mean a scaling packet.
        explicit = path_from_names((names[0], r.WORDS_VARIANT))
        self.assertEqual(explicit.io_variants[-1], r.WORDS_VARIANT)
        self.assertEqual(path_from_names((names[-1],)).io_variants, (r.WORDS_VARIANT,))
        self.assertNotIn(r.WORDS_VARIANT, REF_VARIANTS)
        self.assertNotIn('op.run_length.reference_v1', REF_VARIANTS)

    def test_new_rle_with_entropy_complete_frame_and_independent_decoder(self):
        raw = struct.pack('<128q', *([5]*48 + list(range(32)) + [17]*48))
        column = dict(kind='int64', raw=raw, max_point=0, name='rle', dataset='integration')
        for variant in (r.WORDS_VARIANT, r.BYTES_VARIANT):
            env = Env(column, 0)
            path = path_from_names((variant, 'op.entropy_huffman.project_bytes.cpp_v1'))
            prefix = ()
            for action, io in zip(path, path.io_variants):
                self.assertIn(io, env.legal_io_variants(prefix, action))
                prefix = env.transition(prefix, action, io_variant=io)
                self.assertIsNotNone(prefix, env.records[-1])
            frame = encode_cold(column, path)
            self.assertEqual(frame, make_frame(column, env.nodes[prefix]))
            self.assertEqual(len(frame), env.nodes[prefix].frame_bytes)
            self.assertEqual(decode_frame(frame), raw)
            result = subprocess.run([sys.executable, '-c',
                'import sys;from canonical206_search_env import decode_frame;'
                'sys.stdout.buffer.write(decode_frame(sys.stdin.buffer.read()))'],
                cwd=ROOT/'scripts', input=frame, capture_output=True, check=True)
            self.assertEqual(result.stdout, raw)

    def test_current_86_rle_and_rlbe_paths_have_native_replacements(self):
        maps = p.benchmark_86_io_variant_pipelines()
        self.assertEqual(maps['RLE']['int'][-1], r.WORDS_VARIANT)
        self.assertEqual(maps['RLE']['float_max_point_0_to_18'][-1], r.SCALED_VARIANT)
        self.assertTrue(all(path[-1] == r.RLBE_VARIANT for path in maps['RLBE'].values()))
        for kind, raw in [('int64', struct.pack('<8q', *([7]*8))),
                          ('float64', struct.pack('<8d', *([1.25]*8)))]:
            for method in ('RLE', 'RLBE'):
                key = 'int' if kind == 'int64' else 'float_max_point_0_to_18' if method == 'RLE' else 'float'
                variants = maps[method][key]
                source = u.ActionData(kind, raw, 8)
                current = source
                for name in variants:
                    current = u.run_policy_action(p.IO_VARIANT_TO_POLICY_NAME[name], current,
                        registry_version=p.ACTION_REGISTRY_VERSION, io_variant=name, max_point=2)
                final = current
                for name in reversed(variants):
                    current = u.run_policy_action(p.IO_VARIANT_TO_POLICY_NAME[name], current,
                        registry_version=p.ACTION_REGISTRY_VERSION, reverse=True)
                self.assertEqual(current, source)
                self.assertTrue(u._unpack_result(final)[1].startswith(
                    b'URB1' if method == 'RLBE' else b'URS1' if kind == 'float64' else b'URL1'))

    def test_previous_binding_and_actor_identity_cannot_be_relabelled(self):
        binding = QwenActionBinding()
        record = path_record(path_from_names((r.WORDS_VARIANT,)))
        record['binding_version'] = 'qwen-canonical129-bitviews-parameters-v7'
        with self.assertRaisesRegex(ValueError, 'binding version'):
            binding.normalize_record(record)
        env = Env(dict(kind='int64', raw=struct.pack('<4q', 1,1,2,2),
                       max_point=0, name='rle', dataset='integration'), 0)
        actor = initialize_policy(env, 3, 4)
        validate(actor, env)
        actor.policy_identity = copy.deepcopy(actor.policy_identity)
        actor.policy_identity['policy_registry_version'] = 'cpp-canonical-129-v1'
        with self.assertRaisesRegex(ValueError, 'catalog/registry'):
            validate(actor, env)


if __name__ == '__main__':
    unittest.main()
