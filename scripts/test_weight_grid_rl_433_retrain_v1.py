"""Unit tests with fake metric vectors only: no codec or training execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fresh433', Path(__file__).with_name('run_weight_grid_rl_433_retrain_v1.py'))
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def vector(index=0, enc=1.0, dec=2.0):
    terms = list(fresh.grid.TAILS[index])
    return dict(specs=[], terminals=terms, sequence=[index], bytes=4+index, raw_bytes=128,
        frame_sha256='frame'+str(index), input_sha256='input', kind='int64',
        encode_seconds_median=enc, decode_seconds_median=dec,
        lossless=True, deterministic=True, status='PASS')


class Oracle:
    def __init__(self):
        self.calls = []
        self.verified = []
    def measure(self, raw, kind, specs, terminals, **kwargs):
        index = fresh.grid.TAILS.index(tuple(terminals))
        self.calls.append((raw, kind, specs, tuple(terminals), kwargs))
        row = vector(index)
        row.update(input_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw), kind=kind,
                   cache_key='key'+str(index), phase=kwargs['phase'])
        return row, None
    def verify_record(self, row):
        self.verified.append(row)


class Fresh433Tests(unittest.TestCase):
    def test_reference_exactness_and_framed_validation(self):
        column = dict(input_sha256='input', raw_bytes=128, kind='int64')
        row = {**vector(), 'frame_path': '/fake/frame'}
        with patch.object(fresh.grid.prior, 'verify_frame') as check:
            fresh.check_reference(column, row)
            check.assert_called_once_with(row)
            legacy_without_kind = {k:v for k,v in row.items() if k != 'kind'}
            fresh.check_reference(column, legacy_without_kind)
            self.assertNotIn('kind', legacy_without_kind)
            for changed in ({'input_sha256':'other'}, {'raw_bytes':127}, {'kind':'float64'},
                            {'specs':[{'op':'delta'}]}, {'terminals':['bzip2']},
                            {'encode_seconds_median':0}, {'encode_seconds_median':float('nan')}):
                with self.assertRaises(AssertionError):
                    fresh.check_reference(column, {**row, **changed})

    def test_baselines_new_reference_never_recalibrated_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            raw = bytes(range(128))
            input_path = out/'input.bin'
            input_path.write_bytes(raw)
            column = dict(key='C', input_path=str(input_path), input_sha256=hashlib.sha256(raw).hexdigest(),
                          raw_bytes=128, kind='int64')
            reference = {**vector(enc=.123, dec=.456), 'input_sha256':column['input_sha256']}
            references = {'C':reference}
            oracle = Oracle()
            first = fresh.prepare_shared([column], references, out, oracle, 'identity', False)
            self.assertEqual(len(oracle.calls), 38)
            self.assertTrue(all(call[-1]==dict(phase='selection',warmups=1,repeats=7) for call in oracle.calls))
            self.assertEqual(first['C']['reference'], reference)
            self.assertEqual(first['C']['reference']['encode_seconds_median'], .123)
            with patch.object(fresh, 'check_reference'):
                again = fresh.prepare_shared([column], references, out, oracle, 'identity', True)
            self.assertEqual(again, first)
            self.assertEqual(len(oracle.calls), 38)
            self.assertEqual(len(oracle.verified), 38)
            with self.assertRaises(AssertionError):
                fresh.prepare_shared([column], references, out, oracle, 'wrong_identity', True)
            changed = copy.deepcopy(references)
            changed['C']['decode_seconds_median'] = .457
            with self.assertRaises(AssertionError):
                fresh.prepare_shared([column], changed, out, oracle, 'identity', True)

    def test_baseline_resume_rejects_catalog_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            raw = bytes(128)
            input_path = out/'input.bin'
            input_path.write_bytes(raw)
            column = dict(key='C', input_path=str(input_path), input_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=128, kind='int64')
            refs = {'C': {**vector(), 'input_sha256':column['input_sha256']}}
            fresh.prepare_shared([column], refs, out, Oracle(), 'identity', False)
            saved = fresh.grid.read_json(out/'REFERENCES_AND_BASELINES.json')
            saved['columns']['C']['baselines'][0]['sequence'] = [1]
            fresh.grid.dump(out/'REFERENCES_AND_BASELINES.json', saved)
            with patch.object(fresh, 'check_reference'), self.assertRaises(AssertionError):
                fresh.prepare_shared([column], refs, out, Oracle(), 'identity', True)
            saved['columns']['C']['baselines'][0]['sequence'] = [0]
            saved['columns']['C']['reference']['encode_seconds_median'] = 9.
            fresh.grid.dump(out/'REFERENCES_AND_BASELINES.json', saved)
            with patch.object(fresh, 'check_reference'), self.assertRaises(AssertionError):
                fresh.prepare_shared([column], refs, out, Oracle(), 'identity', True)

    def test_final_replays_cannot_change_learner_winners(self):
        ref = vector(enc=2., dec=3.)
        baseline = [vector(i) for i in range(38)]
        runs = []
        for method in ('ppo', 'grpo'):
            for seed, score in ((1,-.3),(2,-.2),(3,-.4)):
                row = fresh.grid.score(vector(seed), ref, fresh.WEIGHT)
                row['selection_origin'] = 'own_training_plus_common_safety'
                runs.append(dict(method=method, seed=seed, selection_score=score, selected=row))
        historical = {'methods':[{**vector(), 'method':'historical_'+str(i)} for i in range(14)]}
        beam = [dict(method=method, experiment_identity='beam', selected={**vector(4),'cost':-999.})
                for method in fresh.BEAM_METHODS]
        selected = fresh.build_selections({}, runs, dict(reference=ref,baselines=baseline), historical, beam)
        self.assertEqual(len(selected), 21)
        by = {r['method']:r for r in selected}
        for method in ('ppo','grpo'):
            self.assertEqual(by[method]['source_seed'], 2)
            self.assertEqual(by[method]['recipe'], next(r['selected'] for r in runs if r['method']==method and r['seed']==2))
        for old in historical['methods']:
            for key in ('specs','terminals','frame_sha256','bytes','raw_bytes'):
                self.assertEqual(by[old['method']]['recipe'][key], old[key])
        self.assertEqual(by['beam30_open']['recipe']['frame_sha256'], beam[1]['selected']['frame_sha256'])
        for name, limit in [('fresh_fixed5_J_best',5),('fresh_complete30_J_best',30),('common38_J_best',38)]:
            expected = min((fresh.grid.score(x,ref,fresh.WEIGHT) for x in baseline[:limit]), key=fresh.grid.prior.score_rank)
            self.assertEqual(by[name]['recipe'],expected)

    def test_method_collision_rejected(self):
        ref = vector()
        row = {**fresh.grid.score(vector(),ref,fresh.WEIGHT),'selection_origin':'own'}
        runs = [dict(method=m,seed=1,selection_score=-.1,selected=row) for m in ('ppo','grpo')]
        with self.assertRaises(AssertionError):
            fresh.build_selections({}, runs, dict(reference=ref,baselines=[vector(i) for i in range(38)]),
                                   {'methods':[{**vector(),'method':'ppo'}]}, [])

    def test_source_identity_includes_wrapper_test_and_extra_reference_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp)/'reference.json'
            reference.write_text('{"E0":1}')
            def snapshot(*_):
                return dict(file_sha256={},dependencies={'test':'mock'})
            with patch.object(fresh.grid,'source_snapshot',side_effect=snapshot):
                first = fresh.source_snapshot([],Path(tmp)/'gate',[reference])
                reference.write_text('{"E0":2}')
                second = fresh.source_snapshot([],Path(tmp)/'gate',[reference])
            for filename in (Path(fresh.__file__),Path(__file__),reference):
                self.assertIn(str(filename.resolve()),first['file_sha256'])
            self.assertNotEqual(fresh.grid.fingerprint(first),fresh.grid.fingerprint(second))

    def test_load_sources_checks_beam_constants_and_selection_binding(self):
        column = dict(key='C', input_sha256='input', raw_bytes=128, kind='int64')
        reference = {**{k:v for k,v in vector().items() if k != 'kind'}, 'frame_path':'/fake/reference'}
        selected = {**fresh.grid.score(vector(1), reference, fresh.WEIGHT),
                    'mixed_coarse_depth':4, 'frame_path':'/fake/selected'}
        summary = dict(status='COMPLETE',experiment_identity='beam',config=dict(width=30,reference_source='historical'))
        def read(path):
            path = Path(path)
            if path == fresh.grid.HISTORICAL_BENCHMARK:
                return dict(columns=[dict(key='C',kind='int64',reference=copy.deepcopy(reference))])
            if path.name=='SUMMARY.json':return summary
            if path.name=='REFERENCES.json':return dict(columns={'C':copy.deepcopy(reference)})
            if path.name=='FROZEN_SELECTION.json':return dict(experiment_identity='beam',selected=copy.deepcopy(selected))
            if path.name=='RESULT.json':
                method='beam30_'+path.parent.name
                return dict(method=method,weight_tenths=[4,3,3],column=column,reference=reference,
                            budget=dict(width=30),experiment_identity='beam',selected=copy.deepcopy(selected),receipt_sha256={})
            raise AssertionError(path)
        with patch.object(fresh.grid,'read_json',side_effect=read), patch.object(fresh.grid.prior,'verify_frame'):
            _,refs,rows,paths = fresh.load_sources([column],Path('/fake/beam'))
            self.assertEqual(refs['C'],reference)
            self.assertEqual(len(rows['C']),2)
            self.assertTrue(any(path.name=='FROZEN_SELECTION.json' for path in paths))
            summary['config']['reference_source']='formal'
            with self.assertRaises(AssertionError):fresh.load_sources([column],Path('/fake/beam'))


if __name__ == '__main__':
    unittest.main()
