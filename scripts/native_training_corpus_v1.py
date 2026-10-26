#!/usr/bin/env python3
"""Bind the new native calibration to reusable, cohort-neutral GRPO data types.

The old cohort loader is never invoked or monkeypatched. This module alone does
not admit formal training; action coverage and native cold-decode remain gates.
"""
from __future__ import annotations
import json
from pathlib import Path

import native_prepared_cohort_v1 as data
import calibrate_native_smallest32_v1 as calibration
import formal_weighted_grpo_training_v1 as engine


def load_reference(path: Path, corpus: data.Corpus) -> dict:
    value=json.loads(path.read_bytes())
    unsigned={k:v for k,v in value.items() if k!='calibration_sha256'}
    data.require(value.get('calibration_sha256')==data.digest(data.canonical(unsigned)),'native reference content checksum mismatch')
    data.require(value.get('schema')==calibration.SCHEMA and value.get('cohort_version')==data.gate.VERSION,'native reference identity mismatch')
    data.require(corpus.role=='train_validation' and value['corpus_fingerprint']==corpus.fingerprint,'reference corpus mismatch')
    data.require(value['test_used_for_training_statistics'] is False and
                 value['validation_used_for_time_statistics'] is False,'reference statistics contamination')
    data.require(value['lookup_rule']==calibration.LOOKUP and value['recipe']==calibration.RECIPE,'reference method/lookup changed')
    blocks={b.fingerprint:b for b in corpus.split('train') if b.record['original_numeric_bytes']>0}
    measured=value['blocks']
    data.require(len(blocks)==value['training_block_count']==len(measured),'reference training coverage changed')
    data.require({r['block_fingerprint'] for r in measured}==set(blocks),'reference contains missing/substituted train blocks')
    for row in measured:
        b=blocks[row['block_fingerprint']]
        data.require(all(row[k]==v for k,v in b.identity().items()),'reference block identity drift')
        data.require(row['cold_decode_verified'] is True,'reference cold decode absent')
        native=row['native_receipt']
        raw=b.record['original_numeric_bytes']
        data.require(native['original_numeric_bytes']==raw,'reference denominator drift')
        for field,destination in [('encode_ns','compression_ns_per_original_byte'),('decode_ns','decompression_ns_per_original_byte')]:
            samples=native[field]
            data.require(len(samples)==value['repeats'] and all(type(v) is int and v>0 for v in samples),'reference time samples invalid')
            data.require(row[destination]==calibration.statistics.median(samples)/raw,'reference normalized time drift')
    data.require(calibration.references(measured)==value['references'],'reference group aggregation drift')
    data.require(value['original_numeric_bytes']==sum(b.record['original_numeric_bytes'] for b in blocks.values()),'reference byte coverage mismatch')
    return value


def bind(corpus: data.Corpus, reference: dict) -> engine.TrainingCorpus:
    data.require(corpus.role=='train_validation','test-role corpus cannot train')
    rows=[]
    for b in corpus.blocks:
        r=b.record
        if r['original_numeric_bytes']==0:
            continue # Corpus keeps these as N/A; GRPO has no defined loss at B_raw=0.
        group=calibration.reference_for(reference,r['dtype'],r['prepared_block_bytes'])
        root=b.manifest_path.parent
        rows.append(engine.TrainingBlock(dataset_id=b.dataset_id,dataset_manifest_path=str(b.manifest_path),
            prepared_root=str(root),provenance_sha256=b.provenance_sha256,
            **{k:r[k] for k in ('block_id','entity_id','split','column','dtype','block_index',
                               'values_sha256','missing_bitmap_sha256','original_numeric_bytes','prepared_block_bytes')},
            values_path=str(data.gate.safe_child(root,r['values_path'])),
            missing_bitmap_path=str(data.gate.safe_child(root,r['missing_bitmap_path'])),
            compression_reference_ns_per_byte=group['E0'],decompression_reference_ns_per_byte=group['D0'],
            block_fingerprint=b.fingerprint))
    identity={'schema':'native-smallest32-training-corpus-v1','corpus_fingerprint':corpus.fingerprint,
              'native_reference_sha256':reference['calibration_sha256'],
              'role':'train_validation','reference_lookup':calibration.LOOKUP}
    return engine.TrainingCorpus(tuple(r for r in rows if r.split=='train'),
        tuple(r for r in rows if r.split=='validation'),
        tuple(sorted({(b.dataset_id,b.provenance_sha256) for b in corpus.blocks})),
        reference['calibration_sha256'],data.digest(data.canonical(identity)))
