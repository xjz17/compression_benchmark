#!/usr/bin/env python3
"""Explicitly non-formal real-block validation of Python-to-native training API."""
import argparse
import json
from pathlib import Path
import native_prepared_cohort_v1 as data
from native_grpo_task_v1 import NativeClient,NativeGraphTask
from calibrate_native_smallest32_v1 import native,write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','cli','catalog','cold-cli','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    m=json.loads(a.prepared.read_bytes())
    data.require(m['cohort']['role']=='train_validation','only training-role development input allowed')
    r=next(b for b in m['blocks'] if b['split']=='train' and b['dtype']=='int64' and b['original_numeric_bytes']>0)
    b=data.Block(m['dataset_id'],m['domain'],a.prepared,data.digest(a.prepared.read_bytes()),m['provenance_sha256'],'train_validation',r)
    cases=[]
    with NativeClient(a.cli,a.catalog) as c:
        catalog=c.call({'cmd':'catalog'})
        with NativeGraphTask(c,b,weights=(1,0,0),E0=1,D0=1,require_complete=False,warmups=1,repeats=3) as t:
            initial=t.initial_state()
            data.require(92 in t.legal_actions(initial),'A013 int64 unexpectedly masked')
            after=t.transition(initial,92)
            data.require(initial.handle!=after.handle and initial.payload['state']['recipe_action_ids']==[], 'native states mutated in place')
            for state in (initial,after):
                outcome=t.evaluate_terminal(state)
                decoded=native(a.cold_cli.resolve(),'decode',{'frame_hex':outcome.compressed_payload.hex()})
                data.require(decoded['roots']==b.roots(),'independent cold decode changed roots')
                cases.append({'recipe':outcome.operator_graph['recipe_action_ids'],
                    'frame_sha256':data.digest(outcome.compressed_payload),
                    'cold_decode_verified':True,'task_features':len(t.task_features()),
                    'state_features':len(t.state_features(state))})
    result={'schema':'native-training-bridge-development-validation-v1',
        'scope':'one real train block, STOP and A013; no policy training or final benchmark',
        'passed':True,'block':b.identity(),'cases':cases,'catalog':catalog,
        'native_server_sha256':data.digest(a.cli.read_bytes()),
        'native_cold_cli_sha256':data.digest(a.cold_cli.read_bytes()),
        'formal_training_admitted':False}
    write_new(a.output,result)
    print(json.dumps({'passed':True,'cases':len(cases),'output':str(a.output),'formal_training_admitted':False}))


if __name__=='__main__':main()
