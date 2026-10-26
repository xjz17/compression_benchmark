#!/usr/bin/env python3
"""Protocol contract tests. The fake transport supplies no performance evidence."""
import copy
import unittest
import native_grpo_task_v1 as task


class Block:
    fingerprint='test-block'
    record={'original_numeric_bytes':8}
    def roots(self): return [{'kind':'numeric','width':8,'logical_view':'int64',
        'original_numeric_payload':True,'data_hex':'0100000000000000'}]


class Client:
    def __init__(self):
        self.calls=[]
        self.reset=dict(ok=True,task_handle=1,state_handle=2,max_steps=64,
            catalog=dict(action_count=task.ACTION_COUNT,fingerprint=task.REGISTRY,complete_native_graph_space=True),
            task_features=[0]*18,state_features=[0]*28,legal_action_ids=[0,100],terminal_evaluable=True)
        raw=b'opaque-test-frame'
        self.evaluation=dict(ok=True,original_numeric_bytes=8,frame_bytes=len(raw),frame_hex=raw.hex(),
            frame_sha256=task.data.digest(raw),candidate_budget=64,candidate_ordinal=1,
            encode_samples_ns=[8,16,24],decode_samples_ns=[16,24,32],recipe_action_ids=[0],J_selection=len(raw)/8)
    def call(self,request):
        self.calls.append(request)
        if request['cmd'] in ('reset','reset_from_library'): return copy.deepcopy(self.reset)
        if request['cmd']=='evaluate_terminal': return copy.deepcopy(self.evaluation)
        if request['cmd']=='step': return {**copy.deepcopy(self.reset),'state_handle':3}
        return {'ok':True}


class BridgeTests(unittest.TestCase):
    def test_timeout_cleanup_preserves_original_exception(self):
        c=Client()
        class Process:
            def poll(self): return -15
        with self.assertRaisesRegex(TimeoutError, 'original deadline'):
            with self.make(c):
                c.process=Process()
                raise TimeoutError('original deadline')
        self.assertFalse(any(r['cmd']=='release_task' for r in c.calls))

    def test_release_error_does_not_mask_active_failure(self):
        c=Client(); call=c.call; closed=[]
        def broken_release(request):
            if request['cmd']=='release_task': raise RuntimeError('release failed')
            return call(request)
        c.call=broken_release;c.close=lambda:closed.append(True)
        with self.assertRaisesRegex(ValueError,'original failure'):
            with self.make(c) as t:
                raise ValueError('original failure')
        self.assertTrue(closed)
        self.assertIn('release failed',t.cleanup_error)
        with self.assertRaisesRegex(RuntimeError,'release failed'):
            with self.make(c): pass

    def test_failed_candidate_has_only_constraint_cost(self):
        c=Client();c.evaluation=dict(terminal_status='failed',original_numeric_bytes=8,
            max_frame_bytes=1<<30,candidate_budget=64,candidate_ordinal=2,
            recipe_action_ids=[100],error='current block is not representable')
        with self.make(c) as t:
            outcome=t.evaluate_terminal(t.initial_state())
            self.assertIsInstance(outcome,task.grpo.TerminalFailure)
            self.assertFalse(hasattr(outcome,'compressed_payload'))
            self.assertEqual(task.grpo.objective_value(outcome,(0,1,0)),2)
        c=Client();c.evaluation.update(terminal_status='failed',max_frame_bytes=1<<30,error='bad')
        with self.make(c) as t:
            with self.assertRaisesRegex(ValueError,'fabricate'):t.evaluate_terminal(t.initial_state())

    def test_library_selection_stays_in_same_native_task(self):
        c=Client();c.reset.update(candidate_used=3,library_selection={'evaluations_charged':3})
        with task.NativeGraphTask(c,Block(),weights=(1,0,0),E0=1,D0=1,start_from_library=True) as t:
            self.assertEqual(c.calls[0]['cmd'],'reset_from_library')
            self.assertEqual(t.initial_state().payload['candidate_used'],3)
            self.assertEqual(t.library_selection['evaluations_charged'],3)

    def make(self,c):
        return task.NativeGraphTask(c,Block(),weights=(1,0,0),E0=1,D0=1,repeats=3)

    def test_terminal_uses_cpp_samples_original_denominator(self):
        c=Client()
        with self.make(c) as t:
            outcome=t.evaluate_terminal(t.initial_state())
            self.assertEqual(outcome.metrics.compression_time_ns,16)
            self.assertEqual(outcome.metrics.decompression_ns_per_byte,3)
        self.assertEqual(c.calls[-1]['cmd'],'release_task')

    def test_complete_catalog_is_mandatory_by_default(self):
        c=Client();c.reset['catalog']['complete_native_graph_space']=False
        with self.assertRaisesRegex(ValueError,'incomplete native'): self.make(c)

    def test_raw_denominator_and_complete_frame_verified(self):
        for field,value,message in [('original_numeric_bytes',9,'denominator'),
                                    ('frame_sha256','corrupt','SHA'),
                                    ('candidate_ordinal',65,'budget')]:
            c=Client();c.evaluation[field]=value
            with self.make(c) as t:
                with self.assertRaisesRegex(ValueError,message): t.evaluate_terminal(t.initial_state())

    def test_bad_mask_rejected(self):
        for ids in ([0,6341],[0,100,100],[False,100]):
            c=Client();c.reset['legal_action_ids']=ids
            with self.assertRaisesRegex(ValueError,'action mask'): self.make(c)

    def test_cpp_state_handles_are_immutable(self):
        c=Client()
        with self.make(c) as t:
            initial=t.initial_state()
            after=t.transition(initial,100)
            self.assertEqual(initial.handle,2)
            self.assertEqual(after.handle,3)
            with self.assertRaisesRegex(ValueError,'not legal'): t.transition(after,200)


if __name__=='__main__': unittest.main()
