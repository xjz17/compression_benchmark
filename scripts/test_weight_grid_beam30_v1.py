"""Small deterministic tests; fake scalar vectors only, no codec calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_weight_grid_beam30_v1 as beam


class FakeEnv:
    stop=99
    def __init__(self,root_count=38,children=3,depth=2,invalid=None):
        self.root_count,self.children,self.depth=root_count,children,depth
        self.invalid=invalid;self.requests=[];self.rows={}
    def legal(self,seq):
        if not seq:return tuple(range(self.root_count))
        return tuple(range(40,40+self.children))+(self.stop,) if len(seq)<=self.depth else (self.stop,)
    def transition(self,seq,action):
        if self.invalid and self.invalid(seq,action):return None
        return seq+(action,)
    def evaluate(self,seq,**kwargs):
        self.requests.append(seq)
        # Smaller root bytes conflict with the J objective on purpose.
        cost=(self.root_count-seq[0])*.01 + len(seq)*.00001
        row={'sequence':list(seq),'cost':cost,'score':-cost,'bytes':100+seq[0],
             'specs':[{}]*(len(seq)-1),'terminals':['fake']}
        self.rows[seq]=row;return row
    @staticmethod
    def rank(row):return beam.prior.score_rank(row)


class Tests(unittest.TestCase):
    def test_root_J_ranking_width30_not_bytes(self):
        env=FakeEnv(depth=0)
        result=beam.beam_search(env)
        self.assertEqual(result['layers'][0]['survivors'][0],[37])
        self.assertEqual(result['best_sampled']['sequence'],[37])
        self.assertEqual(len(result['layers'][0]['survivors']),30)
        self.assertEqual(result['candidate_attempts'],38)

    def test_child_incumbent_rank_uses_J(self):
        class ChildWins(FakeEnv):
            def evaluate(self,seq,**kw):
                row=super().evaluate(seq,**kw)
                row['score']=len(seq)*100+seq[0]*.01;row['cost']=-row['score']
                return row
        env=ChildWins(depth=2)
        result=beam.beam_search(env)
        self.assertEqual(len(result['best_sampled']['sequence']),3)
        self.assertTrue(all(len(l['survivors'])<=30 for l in result['layers']))

    def test_budget_counts_invalid_and_partial_layer(self):
        env=FakeEnv(invalid=lambda seq,a: bool(seq) and a==40)
        result=beam.beam_search(env,budget=45)
        self.assertEqual(result['candidate_attempts'],45)
        self.assertEqual(result['valid_terminal_requests']+result['invalid_attempts'],45)
        self.assertFalse(result['last_layer_complete'])
        self.assertEqual(len(env.requests),result['valid_terminal_requests'])
        self.assertEqual(result['stop_reason'],'candidate_cap_reached')

    def test_fixed_root_scope_and_implicit_stop(self):
        env=FakeEnv()
        result=beam.beam_search(env,roots=(1,))
        self.assertTrue(all(seq[0]==1 and 99 not in seq for seq in env.requests))
        self.assertEqual(result['root_count'],1)
        self.assertTrue(result['implicit_stop_at_every_measured_state'])

    def test_determinism_and_illegal_roots(self):
        a=beam.beam_search(FakeEnv());b=beam.beam_search(FakeEnv())
        self.assertEqual(a,b)
        with self.assertRaises(ValueError):beam.beam_search(FakeEnv(),roots=(1,1))
        with self.assertRaises(ValueError):beam.beam_search(FakeEnv(),roots=(50,))

    def test_exact_weight_rescore_uses_raw_vectors(self):
        ref={'encode_seconds_median':2.,'decode_seconds_median':3.}
        raw={'bytes':50,'raw_bytes':100,'encode_seconds_median':6.,'decode_seconds_median':1.,
             'cost':99,'score':-99,'reward':-99}
        one=beam.grid.score(raw,ref,(4,3,3));two=beam.grid.score(raw,ref,(10,0,0))
        self.assertAlmostEqual(one['cost'],.5)
        self.assertAlmostEqual(two['cost'],.5)
        third=beam.grid.score(raw,ref,(0,10,0))
        self.assertAlmostEqual(third['cost'],.75)
        self.assertEqual(third['reward'],-third['cost'])
        self.assertEqual(raw['cost'],99)

    def test_replay_retains_historical_method_identity(self):
        recipe={'specs':[],'terminals':['zlib6'],'bytes':25,'raw_bytes':100,'frame_sha256':'abc'}
        hist={'methods':[dict(recipe,method='old_b4_beam',selection_criterion='historical size')]}
        rows=beam.replay_selections(hist,[],[(4,3,3),(10,0,0)])
        self.assertEqual([r['method'] for r in rows],['old_b4_beam']*2)
        self.assertTrue(all('NOT_retrained' in r['selection'] for r in rows))
        self.assertEqual(rows[0]['recipe']['specs'],[])
        self.assertNotIn('score',rows[0]['recipe'])

    def test_completed_selection_fingerprint_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);frame=folder/'selected.acf3';frame.write_bytes(b'fake')
            selected={'frame_path':str(frame),'specs':[],'terminals':['bzip2'],'sequence':[1],
                      'frame_sha256':beam.grid.digest(b'fake'),'bytes':4,'raw_bytes':8,
                      'cost':.5,'weight_tenths':[4,3,3],'mixed_coarse_depth':4}
            frozen=folder/'FROZEN_SELECTION.json'
            beam.grid.dump(frozen,{'experiment_identity':'id','selected':selected})
            row={'schema':beam.SCHEMA,'experiment_identity':'id','weight_tenths':[4,3,3],
                 'tail_scope':'fixed_b4','selected':selected,'budget':{'width':30},
                 'receipt_sha256':{str(frozen):beam.grid.digest(frozen.read_bytes())}}
            beam.verify_run(row,'id',(4,3,3),'fixed_b4')
            corrupt=copy.deepcopy(row);corrupt['selected']['cost']=.1
            with self.assertRaisesRegex(AssertionError,'frozen selection'):
                beam.verify_run(corrupt,'id',(4,3,3),'fixed_b4')


if __name__=='__main__':unittest.main()
