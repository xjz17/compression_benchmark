#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
import native_prepared_cohort_v1 as data
from summarize_native_comparison_smallest32_v1 import model_columns,aggregate_model_rows,validate_pairing,amortization_rows,win_tie_loss

class SummaryTests(unittest.TestCase):
    def summarize(self,rows):
        return aggregate_model_rows(rows)[0]
    def row(self,raw=8,**kw):
        return dict(dataset_id='fixture',column='v',weight_index=0,weight=[1,0,0],entity_id='e',status='ok',
                    original_numeric_bytes=raw,frame_bytes=4,compression_time_ns=80,decompression_time_ns=40,**kw)
    def test_byte_weighted_column_and_zero_block(self):
        first=self.row();second=self.row(raw=24);second['frame_bytes']=12
        result=self.summarize([first,second])
        self.assertEqual(result['compression_ratio'],.5)
        self.assertEqual(result['compression_ns_per_byte'],5)
        self.assertEqual(result['original_numeric_bytes'],32)
        self.assertEqual(result['block_count'],2)
        mixed=self.summarize([first,second,{**self.row(raw=0),'status':'N/A'}])
        self.assertEqual(mixed['status'],'failed_or_partial');self.assertIsNone(mixed['compression_ratio'])
    def test_one_failed_block_does_not_produce_partial_speed_claim(self):
        failed={**self.row(),'status':'failed'};del failed['compression_time_ns'];del failed['frame_bytes']
        result=self.summarize([self.row(),failed])
        self.assertEqual(result['status'],'failed_or_partial')
        self.assertIsNone(result['compression_ratio'])
        self.assertIsNone(result['compression_ns_per_byte'])
    def test_entire_empty_column_is_N_A(self):
        result=self.summarize([{**self.row(raw=0),'status':'N/A'}])
        self.assertEqual(result['status'],'N/A');self.assertIsNone(result['decompression_ns_per_byte'])

    def measured(self,index,variant='model',raw=8):
        t=10*(index+1)
        return self.row(raw=raw,inference_variant=variant,block_index=index,block_id=str(index),
            recipe_action_ids=[91],representative_sample_index=0,
            samples=[dict(evaluations=index+1,certified_pruned_states=1,budget_unexplored_actions=2,
                          timing={'total_prediction_ns':t,'total_compression_ns':80,'final_encode_ns':80-t})])

    def test_prediction_quantiles_use_whole_selected_observations(self):
        value=self.summarize([self.measured(0),self.measured(1,raw=24)])
        self.assertEqual(value['prediction_time_ns'],30)
        self.assertEqual(value['prediction_ns_per_byte'],30/32)
        self.assertEqual(value['prediction_block_mean_ns'],15)
        self.assertEqual(value['prediction_block_p95_ns'],20)
        self.assertEqual(value['evaluation_count'],3)
        self.assertEqual(value['recipe_usage'],[{'recipe_action_ids':[91],'block_count':2}])

    def test_amortization_never_combines_variants_or_partial_prefixes(self):
        values=[self.measured(i,variant=v) for v in ('model','no_reuse') for i in (0,1)]
        rows=amortization_rows(values)
        self.assertEqual([r['processed_blocks'] for r in rows],[1,2,1,2])
        self.assertEqual([r['cumulative_original_numeric_bytes'] for r in rows],[8,16,8,16])
        values[1]['status']='failed'
        row=amortization_rows(values)[1]
        self.assertEqual(row['status'],'failed_or_partial')
        self.assertIsNone(row['compression_ns_per_byte'])

    def test_win_tie_loss_retains_unavailable_baselines(self):
        model=[self.summarize([self.row()])]
        baseline=[dict(dataset_id='fixture',column='v',method='direct',primary_comparison=True,
            status='ok',compression_ratio=.5,compression_ns_per_byte=20,decompression_ns_per_byte=1),
            dict(dataset_id='fixture',column='v',method='absent',primary_comparison=False,status='unsupported')]
        counts={(r['baseline'],r['metric']):r for r in win_tie_loss(model,baseline)}
        self.assertEqual(counts['direct','compression_ratio']['tie'],1)
        self.assertEqual(counts['direct','compression_ns_per_byte']['win'],1)
        self.assertEqual(counts['direct','decompression_ns_per_byte']['loss'],1)
        self.assertTrue(all(r['not_comparable']==1 for (label,_),r in counts.items() if label=='absent'))

    def test_comparison_rejects_incomplete_model_receipt(self):
        from test_native_frozen_evaluation_v1 import _Corpus,_artifacts,_identity,_publish_completed,_rows
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);identity=_identity(_Corpus(),_artifacts(p),{'calibration_sha256':'r'*64})
            rows=[{**r,'column':'v','entity_id':'e','compression_time_ns':8,'decompression_time_ns':8} for r in _rows()]
            receipt=_publish_completed(p,identity,rows)
            self.assertEqual(len(model_columns(p)),6)
            # Re-hashing partial output cannot turn missing input pairs into a
            # complete column comparison.
            worker=p/'worker_0.jsonl';worker.write_bytes(b''.join(data.canonical(r)+b'\n' for r in rows[:-1]))
            receipt['worker_files'][0]['sha256']=data.digest(worker.read_bytes())
            receipt['receipt_sha256']=data.digest(data.canonical({k:v for k,v in receipt.items() if k!='receipt_sha256'}))
            (p/'receipt.json').write_bytes(data.canonical(receipt))
            with self.assertRaisesRegex(ValueError,'pair|coverage'):model_columns(p)

class PairingTests(unittest.TestCase):
    def fixture(self,root):
        model_dir=root/'model';base_dir=root/'baseline';model_dir.mkdir();base_dir.mkdir()
        labels=['fixture_method_'+str(i) for i in range(86)]
        inventory={'methods':[{'label':label} for label in labels]}
        inv=base_dir/'inventory.json';inv.write_bytes(data.canonical(inventory))
        model=[];baseline=[];manifests=[]
        for i in range(16):
            shared=dict(dataset_id='dataset_'+str(i),column='v',original_numeric_bytes=8,block_count=1,entity_count=1)
            model.extend({**shared,'weight_index':wi} for wi in range(6))
            baseline.extend({**shared,'method':label} for label in labels)
            raw=base_dir/('raw_'+str(i)+'.jsonl');raw.write_text(''.join(json.dumps({'method':label})+'\n' for label in labels))
            manifests.append(dict(dataset_id=shared['dataset_id'],raw_jsonl=str(raw),raw_jsonl_sha256=data.digest(raw.read_bytes()),rows=86))
        columns=base_dir/'columns.json';columns.write_bytes(data.canonical(baseline))
        frames=base_dir/'frames.json';frames.write_bytes(data.canonical({'frames':[],'frame_count':0,'frame_bytes':0}))
        m=dict(corpus_fingerprint='fixture-cohort',cpus=[25],warmups=1,repeats=5,pair_count=96,
               run_identity={'expected_pair_count':96})
        b=dict(status='complete',manifest_count=16,corpus={'role':'test_candidate','corpus_fingerprint':'fixture-cohort'},
               timing_contract={'available_cpu_affinity':[25],'manifest_execution':'serial','threads':1,'warmups':1,'repeats':5},
               inventory=str(inv),inventory_sha256=data.digest(data.canonical(inventory)),
               outputs={'columns':{'path':str(columns),'sha256':data.digest(columns.read_bytes())},
                        'frames_json':{'path':str(frames),'sha256':data.digest(frames.read_bytes())}},
               complete_frame_count=0,complete_frame_bytes_on_disk=0,
               manifest_receipts=manifests,raw_row_count=16*86)
        (model_dir/'receipt.json').write_bytes(data.canonical(m));(base_dir/'receipt.json').write_bytes(data.canonical(b))
        return model_dir,base_dir,model,baseline,b
    def test_complete_pairing_and_missing_method(self):
        with tempfile.TemporaryDirectory() as d:
            md,bd,m,b,receipt=self.fixture(Path(d))
            self.assertTrue(validate_pairing(md,bd,m,b)['passed'])
            with self.assertRaisesRegex(ValueError,'missing weight or method'):validate_pairing(md,bd,m,b[:-1])
            b[0]['original_numeric_bytes']=16
            with self.assertRaisesRegex(ValueError,'original bytes'):validate_pairing(md,bd,m,b)
    def test_changed_raw_receipt_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as d:
            md,bd,m,b,receipt=self.fixture(Path(d))
            Path(receipt['manifest_receipts'][0]['raw_jsonl']).write_text('tampered')
            with self.assertRaisesRegex(ValueError,'raw measurements changed'):validate_pairing(md,bd,m,b)

    def test_baseline_frame_is_rehashed_before_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            md,bd,m,b,receipt=self.fixture(Path(d))
            payload=b'NGF1 fixture';sha=data.digest(payload);relative=sha[:2]+'/'+sha+'.ngf1'
            path=bd/'frames'/relative;path.parent.mkdir(parents=True);path.write_bytes(payload)
            frames={'frames':[{'path':'frames/'+relative,'sha256':sha,'bytes':len(payload)}],
                    'frame_count':1,'frame_bytes':len(payload)}
            (bd/'frames.json').write_bytes(data.canonical(frames))
            receipt['outputs']['frames_json']['sha256']=data.digest((bd/'frames.json').read_bytes())
            receipt['complete_frame_count']=1;receipt['complete_frame_bytes_on_disk']=len(payload)
            raw=Path(receipt['manifest_receipts'][0]['raw_jsonl'])
            lines=[json.loads(line) for line in raw.read_text().splitlines()]
            lines[0].update(status='ok',complete_frame_relative_path=relative,
                            complete_frame_sha256=sha,complete_frame_bytes=len(payload))
            raw.write_bytes(b''.join(data.canonical(line)+b'\n' for line in lines))
            receipt['manifest_receipts'][0]['raw_jsonl_sha256']=data.digest(raw.read_bytes())
            (bd/'receipt.json').write_bytes(data.canonical(receipt))
            self.assertTrue(validate_pairing(md,bd,m,b)['passed'])
            path.write_bytes(payload+b'corrupt')
            with self.assertRaisesRegex(ValueError,'frame checksum or length'):
                validate_pairing(md,bd,m,b)

if __name__=='__main__':unittest.main()
