"""Mock-only remaining65/scheduler tests; no codec, training or child process."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_weight_grid_rl_remaining65_v1 as remaining
import schedule_weight_grid_remaining65_v1 as scheduler

grid=remaining.grid


def vector(index,weight,enc=1.,dec=2.):
    row=dict(bytes=10+index,raw_bytes=128,specs=[],terminals=list(grid.TAILS[index]),sequence=[index],
        frame_sha256='frame'+str(index),encode_seconds_median=enc,decode_seconds_median=dec)
    return {**grid.score(row,dict(encode_seconds_median=2.,decode_seconds_median=3.),weight),
            'selection_origin':'own_training_plus_common_baseline'}


class Remaining65Tests(unittest.TestCase):
    def test_exact_weight_partition(self):
        values=remaining.remaining_weights()
        self.assertEqual(len(values),65)
        self.assertEqual(len(set(values)),65)
        self.assertNotIn((4,3,3),values)
        self.assertEqual(set(values)|{(4,3,3)},set(grid.weight_grid()))
        self.assertEqual(65*20*2*3,7800)
        self.assertEqual(65*20,1300)

    def test_commands_are_serial_phase_inputs_with_exact65(self):
        commands=scheduler.phase_commands(Path('/fake/output'))
        self.assertEqual(list(commands),['beam65','rl65'])
        beam=commands['beam65'];start=beam.index('--weights')+1;end=beam.index('--tail-scopes')
        self.assertEqual([tuple(map(int,w.split(','))) for w in beam[start:end]],remaining.remaining_weights())
        self.assertEqual(beam[end+1],'open')
        self.assertEqual(beam[beam.index('--reference-source')+1],'historical')
        self.assertNotIn('--resume',beam)
        self.assertNotIn('--iterations',commands['rl65'])
        self.assertEqual(commands['rl65'][-1],'/fake/output/beam65')

    def test_per_weight_own_winners_and_final_replays(self):
        weights=[(0,0,10),(10,0,0)];runs=[]
        for weight in weights:
            for method in ('ppo','grpo'):
                for seed in (1,2,3):
                    row=vector(seed,weight)
                    runs.append(dict(method=method,weight_tenths=list(weight),seed=seed,
                        selection_score=-(seed if weight[0] else 4-seed),selected=row))
        baseline=[vector(i,(4,3,3)) for i in range(38)]
        refs=dict(encode_seconds_median=2.,decode_seconds_median=3.)
        history={'methods':[{**vector(0,(4,3,3)),'method':'old'+str(i)} for i in range(14)]}
        beam={grid.weight_id(w):dict(selected=vector(4,w),experiment_identity='beam') for w in weights}
        output=remaining.build_selections(runs,dict(reference=refs,baselines=baseline),history,beam,weights)
        self.assertEqual(len(output),40)
        for weight in weights:
            rows={r['method']:r for r in output if r['weight_tenths']==list(weight)}
            self.assertEqual(len(rows),20)
            for method in ('ppo','grpo'):
                self.assertEqual(rows[method]['source_seed'],1 if weight[0] else 3)
            self.assertEqual(rows['beam30_open']['recipe']['frame_sha256'],'frame4')
            for name,count in [('fresh_fixed5_J_best',5),('fresh_complete30_J_best',30),('common38_J_best',38)]:
                expected=min((grid.score(row,refs,weight) for row in baseline[:count]),key=grid.prior.score_rank)
                self.assertEqual(rows[name]['recipe']['cost'],expected['cost'])
        self.assertTrue(all('recipe' not in row for row in runs))

    def test_completed_phase_requires_coverage(self):
        summary=dict(status='COMPLETE',config=dict(weights_tenths=[list(w) for w in remaining.remaining_weights()],
            tail_scopes=['open'],width=30,reference_source='historical'),completed_searches=1300,columns=[{}]*20)
        def read(path):return {'status':'COMPLETE'} if Path(path).name=='STATUS.json' else summary
        with patch.object(Path,'exists',return_value=True),patch.object(grid,'read_json',side_effect=read):
            self.assertTrue(scheduler.phase_complete(Path('/fake'),'beam65'))
            summary['completed_searches']=1299
            self.assertFalse(scheduler.phase_complete(Path('/fake'),'beam65'))
            summary['config']['weights_tenths'].append([4,3,3])
            with self.assertRaises(AssertionError):scheduler.phase_complete(Path('/fake'),'beam65')

    def test_main433_must_be_genuinely_complete(self):
        summary=dict(status='COMPLETE',main_433_protocol_complete=True,completed_runs=120,
            config=dict(weights_tenths=[[4,3,3]]),columns=[{}]*20,benchmark_result_paths=['p']*20)
        def read(path):return {'status':'COMPLETE'} if Path(path).name=='STATUS.json' else summary
        with patch.object(grid,'read_json',side_effect=read):
            self.assertEqual(scheduler.require_main_433(Path('/fake')),summary)
            summary['completed_runs']=119
            with self.assertRaises(AssertionError):scheduler.require_main_433(Path('/fake'))

    def test_complete_phase_is_not_reexecuted_or_rewritten(self):
        with patch.object(scheduler,'phase_complete',return_value=True),patch.object(scheduler.subprocess,'Popen') as process:
            result=scheduler.run_phase('beam65',['fake'],Path('/fake'),'identity',True)
            self.assertEqual(result['status'],'PREVIOUSLY_COMPLETE_NOT_REEXECUTED')
            process.assert_not_called()

    def test_failed_process_stops_without_launching_another(self):
        # Validate the failure branch without a real process, sleep or filesystem writes.
        class Stream:
            def __enter__(self):return self
            def __exit__(self,*args):pass
        child=type('Child',(),{'pid':123,'returncode':7,'poll':lambda self:7})()
        with patch.object(scheduler,'phase_complete',return_value=False),patch.object(Path,'exists',return_value=False),\
             patch.object(Path,'open',return_value=Stream()),patch.object(scheduler.subprocess,'Popen',return_value=child) as process,\
             patch.object(grid,'dump'):
            with self.assertRaisesRegex(RuntimeError,'no automatic retry'):
                scheduler.run_phase('beam65',['fake'],Path('/fake'),'identity',False)
            self.assertEqual(process.call_count,1)

    def test_source_benchmark_is_bound_to_frozen_identity_and_exact_recipe(self):
        weight=(4,3,3);recipe=vector(0,weight)
        selection=dict(method='ppo',weight_id=grid.weight_id(weight),weight_tenths=list(weight),
                       selection='own',recipe=recipe)
        row={**{k:v for k,v in selection.items() if k!='recipe'},**recipe}
        bench=dict(experiment_identity='source',rows=[row])
        frozen=dict(experiment_identity='source',selections=[selection])
        summary=dict(experiment_identity='source')
        scheduler.verify_source_benchmark(bench,frozen,summary,{weight},{'ppo'})
        wrong=copy.deepcopy(bench);wrong['experiment_identity']='replacement'
        with self.assertRaises(AssertionError):scheduler.verify_source_benchmark(wrong,frozen,summary,{weight},{'ppo'})
        wrong=copy.deepcopy(bench);wrong['rows'][0]['frame_sha256']='other-frame'
        with self.assertRaises(AssertionError):scheduler.verify_source_benchmark(wrong,frozen,summary,{weight},{'ppo'})
        with self.assertRaises(AssertionError):scheduler.verify_source_benchmark(bench,frozen,summary,{weight},{'ppo','grpo'})

    def test_live_ppt_edits_do_not_change_immutable_delivery_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'main.pptx';original.write_bytes(b'original-delivered-ppt')
            output=Path(tmp)/'run';output.mkdir()
            snapshot,receipt_path,receipt=scheduler.prepare_ppt_snapshot(output,original,False)
            original.write_bytes(b'user-edit-after-delivery')
            again,_,same=scheduler.prepare_ppt_snapshot(output,original,True)
            self.assertEqual(again,snapshot)
            self.assertEqual(same,receipt)
            self.assertEqual(snapshot.read_bytes(),b'original-delivered-ppt')
            self.assertNotEqual(grid.digest(original.read_bytes()),same['snapshot_sha256'])
            original.unlink()
            scheduler.prepare_ppt_snapshot(output,original,True)
            snapshot.write_bytes(b'corrupted-snapshot')
            with self.assertRaises(AssertionError):scheduler.prepare_ppt_snapshot(output,original,True)


if __name__=='__main__':unittest.main()
