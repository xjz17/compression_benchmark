#!/usr/bin/env python3
"""Real training-block library/budget bridge checks, without any policy update."""
import argparse
import json
from pathlib import Path

import native_prepared_cohort_v1 as data
import native_training_corpus_v1 as binding
import calibrate_native_smallest32_v1 as calibration
import formal_weighted_grpo_training_v1 as engine
from native_training_factory_v1 import Factory
from native_grpo_task_v1 import REGISTRY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cohort','materialized','reference','cli','catalog','cold-cli','library','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cpu',type=int,required=True)
    a=p.parse_args()
    corpus=data.load(a.cohort,a.materialized,role='train_validation')
    reference=binding.load_reference(a.reference,corpus);bound=binding.bind(corpus,reference)
    sequence=next(s for s in engine._ordered_sequences(bound.train_blocks) if s.blocks[0].dtype=='int64')
    first=sequence.blocks[0]
    with Factory(corpus,cli=a.cli,catalog=a.catalog,library=a.library,cpus=(a.cpu,)) as factory:
        request=engine._factory_request(first,stage='train',weight=(1,0,0),seed=0,iteration=0,group_index=0,
            episode_index=0,registry=REGISTRY,series_block_ordinal=0)
        with factory(request) as task:
            charged=task.initial_state().payload['candidate_used']
            data.require(0<charged<64 and task.library_selection['evaluations_charged']==charged,'library budget was reset')
            result=task.evaluate_terminal(task.initial_state())
            data.require(result.operator_graph['native_receipt']['candidate_ordinal']==charged+1,'terminal not charged after library')
            cold=calibration.native(a.cold_cli.resolve(),'decode',{'frame_hex':result.compressed_payload.hex()})
            data.require(cold['roots']==task.block.roots(),'library bridge cold decode differs')
            selected=result.operator_graph['recipe_action_ids']
        # Reuse this same source block only as an explicit contract fixture;
        # this is not a chronological training episode or measurement result.
        request=engine._factory_request(first,stage='train',weight=(1,0,0),seed=0,iteration=0,group_index=0,
            episode_index=0,registry=REGISTRY,series_block_ordinal=1,previous_recipe=selected,
            previous_recipe_source_block_id=first.block_id)
        with factory(request) as task:
            data.require(task.initial_state().payload['candidate_used']==1 and task.budget==16,
                         'previous recipe was not evaluated exactly once before continuation')
            result=task.evaluate_terminal(task.initial_state())
            data.require(result.operator_graph['native_receipt']['candidate_ordinal']==2,'continuation budget reset')
    receipt={'schema':'native-library-training-bridge-v1','passed':True,'formal_training_admitted':False,
        'scope':'real train block as first/continuation API fixture; no gradient update and no formal performance claim',
        'corpus_fingerprint':corpus.fingerprint,'block_fingerprint':first.block_fingerprint,
        'library_initial_charged':charged,'library_after_stop':charged+1,
        'continuation_initial_charged':1,'continuation_after_stop':2,'independent_cold_decode':True,
        'cli_sha256':data.digest(a.cli.read_bytes()),'library_sha256':data.digest(a.library.read_bytes())}
    calibration.write_new(a.output,receipt)
    print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__':main()
