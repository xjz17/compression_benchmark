import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('weight_grid_under_test',Path(__file__).with_name('run_weight_grid_rl_v1.py'))
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)


class GridTests(unittest.TestCase):
    def test_grid(self):
        values=grid.weight_grid()
        self.assertEqual(len(values),66)
        for w in [(10,0,0),(0,10,0),(0,0,10),(4,3,3)]:self.assertIn(w,values)
        self.assertEqual(len({grid.weight_id(w) for w in values}),66)

    def test_raw_rate_and_zero_weights(self):
        row=dict(bytes=600,raw_bytes=100,encode_seconds_median=2.,decode_seconds_median=9.)
        ref=dict(encode_seconds_median=2.,decode_seconds_median=3.)
        self.assertEqual(grid.score(row,ref,(10,0,0))['reward'],-6.)
        self.assertEqual(grid.score(row,ref,(0,10,0))['cost'],.5)
        self.assertEqual(grid.score(row,ref,(0,0,10))['cost'],.75)
        self.assertAlmostEqual(grid.score(row,ref,(4,3,3))['cost'],2.775)
        self.assertNotIn('reward',row)

    def test_weight_validation(self):
        for w in [(1,2,3),(-1,1,10),(0.1,0.2,9.7),(True,0,9)]:
            with self.assertRaises(ValueError):grid.validate_weight(w)

    def test_identity_ignores_snapshot_time_not_inputs(self):
        before={'at':'one','file_sha256':{'a':'hash'},'dependencies':{'python':'same'}}
        after={**before,'at':'two'}
        self.assertEqual(grid.experiment_identity({'w':[4,3,3]},['col'],before),
                         grid.experiment_identity({'w':[4,3,3]},['col'],after))
        self.assertNotEqual(grid.experiment_identity({'w':[4,3,3]},['col'],before),
                            grid.experiment_identity({'w':[10,0,0]},['col'],after))

    def test_cache_vector_and_protocol_identity(self):
        calls=[]
        def measure(raw,kind,specs,terms,**kw):
            calls.append(kw)
            frame=b'frame'
            return dict(bytes=5,frame_bytes=5,raw_bytes=len(raw),frame_sha256=grid.digest(frame),
                        specs=specs,terminals=list(terms),lossless=True,deterministic=True,
                        encode_seconds_median=1.,decode_seconds_median=2.),frame
        with tempfile.TemporaryDirectory() as temp:
            oracle=grid.MeasurementOracle(Path(temp)/'cache.sqlite','runtime',floor_gib=0,measure=measure)
            args=(b'12345678','int64',[],('zlib6',))
            first,_=oracle.measure(*args,phase='training',warmups=0,repeats=1)
            again,frame=oracle.measure(*args,phase='training',warmups=0,repeats=1)
            self.assertIsNone(frame);self.assertEqual(len(calls),1)
            self.assertEqual(again['request_encoder_calls'],0)
            self.assertNotIn('reward',again)
            oracle.verify_record(first)
            with self.assertRaises(AssertionError):oracle.verify_record({**first,'encode_seconds_median':100.})
            self.assertEqual(oracle.totals()['persistent_unique_vectors'],1)
            oracle.measure(*args,phase='selection',warmups=1,repeats=7)
            self.assertEqual(len(calls),2)
            with self.assertRaises(AssertionError):oracle.measure(*args,phase='training',warmups=0,repeats=1,expected='bad')
            oracle.close()

    def test_gzip_real_policy_audit(self):
        import numpy as np
        class Env:
            action_count=3;stop=2;max_steps=2
            def __init__(self):self.counters={}
            def features(self,p):return np.array([1.,len(p),float(p[0]) if p else 0.])
            def legal(self,p):return (2,) if p else (0,1)
            def transition(self,p,a):return p+(a,)
            def reward(self,p):return -.1 if p[0]==0 else -.9
        with tempfile.TemporaryDirectory() as temp:
            initials=[]
            for method in ('ppo','grpo'):
                with grid.GzipTrajectoryAudit(grid.POLICY,Path(temp)/(method+'.jsonl'),method=method,seed=17,group_size=8) as audit:
                    actor,logs=grid.POLICY.train(method,Env(),17,iterations=2,episodes=8,epochs=2)
                receipt=audit.receipt()
                self.assertEqual(receipt['episodes'],16)
                self.assertGreater(receipt['nonzero_advantage_decisions'],0)
                self.assertGreater(sum(r['actor_parameter_change_l2'] for r in logs),0)
                initials.append(logs[0]['initial_actor_sha256'])
                self.assertEqual(sum(r['critic_optimizer_steps'] for r in logs),4 if method=='ppo' else 0)
            self.assertEqual(initials[0],initials[1])

    def test_env_reweights_shared_vector_without_candidate_exposure(self):
        def measure(raw,kind,specs,terms,**kw):
            frame=b'frame'
            return dict(bytes=5,raw_bytes=len(raw),frame_sha256=grid.digest(frame),specs=specs,
                        terminals=list(terms),lossless=True,deterministic=True,
                        encode_seconds_median=2.,decode_seconds_median=3.,chain='zlib6'),frame
        reference=dict(bytes=10,encode_seconds_median=2.,decode_seconds_median=3.)
        with tempfile.TemporaryDirectory() as temp:
            oracle=grid.MeasurementOracle(Path(temp)/'cache.sqlite','runtime',floor_gib=0,measure=measure)
            first=grid.GridEnv(bytes(128),'int64',reference,Path(temp)/'first.jsonl',weight=(10,0,0),oracle=oracle)
            second=grid.GridEnv(bytes(128),'int64',reference,Path(temp)/'second.jsonl',weight=(0,10,0),oracle=oracle)
            self.assertEqual(first.reward((0,)),-5/128)
            self.assertEqual(second.measured_results,{})
            self.assertEqual(second.reward((0,)),-.5)
            self.assertEqual(second.measured_results[(0,)]['training_score'],-.5)
            self.assertEqual(oracle.stats['training_misses'],1)
            self.assertEqual(oracle.stats['training_hits'],1)
            first.close();second.close();oracle.close()

    def test_raw_final_recipe_drops_alias_scores(self):
        row=dict(specs=[],terminals=['zlib6'],bytes=5,raw_bytes=128,frame_sha256='hash',
                 selection_score=-99,training_score=-88,weight_tenths=[10,0,0],cost=99)
        clean=grid.raw_recipe(row)
        for key in ('selection_score','training_score','weight_tenths','cost'):self.assertNotIn(key,clean)

    def test_completed_benchmark_still_checks_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            grid.dump(folder/'FROZEN_SELECTION.json',dict(experiment_identity='id',selections=['old']))
            grid.dump(folder/'RESULT.json',dict(experiment_identity='id',recipes={}))
            with self.assertRaises(AssertionError):
                grid.benchmark_column({},['new'],[],{},folder,'id',True)


if __name__=='__main__':unittest.main()
