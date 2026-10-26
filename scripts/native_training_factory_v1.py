#!/usr/bin/env python3
"""Bounded pool of C++ graph workers for the admitted smallest32 training run."""
from __future__ import annotations
import json
from pathlib import Path
import threading

import native_prepared_cohort_v1 as data
from native_grpo_task_v1 import NativeClient,NativeGraphTask,REGISTRY
import native_experiment_scope_v1 as experiment_scope


class Factory:
    def __init__(self, corpus: data.Corpus, *, cli: Path, catalog: Path, library: Path,
                 cpus: tuple[int,...], allow_large_memory: bool=True):
        data.require(corpus.role=='train_validation','test corpus cannot bind training factory')
        data.require(1<=len(cpus)<=4 and len(set(cpus))==len(cpus),'requires 1..4 distinct worker CPUs')
        self.blocks={b.fingerprint:b for b in corpus.blocks}
        self.by_identity={(b.dataset_id,b.record['block_id']):b for b in corpus.blocks}
        data.require(len(self.by_identity)==len(corpus.blocks),'ambiguous block identifiers')
        self.cli,self.catalog,self.library=map(Path,(cli,catalog,library))
        value=json.loads(self.library.read_bytes())
        data.require(value['source_scope']=='training_only' and value['library_format']=='native-ngf1-smallest32-v1', 'old or nontraining library')
        data.require(value['corpus_fingerprint']==corpus.fingerprint and value['action_registry_fingerprint']==REGISTRY,'library identity mismatch')
        data.require(value['native_catalog_sha256']==data.digest(self.catalog.read_bytes()),'library catalog changed')
        self.experiment_scope=experiment_scope.load(self.catalog)
        for recipe in value['recipes']:
            experiment_scope.verify_recipe(recipe['recipe_action_ids'],self.experiment_scope)
        self.library_sha=data.digest(self.library.read_bytes())
        self.cpus=cpus
        self.clients=[]
        self.lock=threading.Lock();self.cursor=0
        self.allow_large_memory=allow_large_memory
        self.reopen()

    def reopen(self):
        """Recreate the training pool only after the validation pool has closed."""
        data.require(not self.clients,'training workers are already open')
        try:
            for cpu in self.cpus:
                # The owning seed supervisor enforces the total 16-hour
                # budget and kills this process group at that deadline.
                client=NativeClient(self.cli,self.catalog,library=self.library,cpu=cpu,timeout=57600)
                self.clients.append(client)
                cap=client.call({'cmd':'catalog'})
                data.require(cap['complete_native_graph_space'] and cap['action_count']==6341 and
                    cap['registry_fingerprint']==REGISTRY,'incomplete or changed native action space')
        except BaseException:
            self.close();raise

    def solution_library_identity(self):
        return dict(path=str(self.library.resolve()),file_sha256=self.library_sha,action_registry_fingerprint=REGISTRY,
                    experiment_scope=self.experiment_scope)

    def __call__(self, request):
        data.require(request.expected_action_registry_fingerprint==REGISTRY,'request registry drift')
        block=self.by_identity[request.dataset_id,request.block_id]
        r=block.record
        data.require(request.split==block.split and request.dtype==r['dtype'] and
            request.provenance_sha256==block.provenance_sha256 and
            request.values_sha256==r['values_sha256'] and request.missing_bitmap_sha256==r['missing_bitmap_sha256'] and
            request.original_numeric_bytes==r['original_numeric_bytes'],'request block identity drift')
        data.require(request.stage!='train' or block.split=='train','validation block entered training')
        with self.lock:
            client=self.clients[self.cursor%len(self.clients)];self.cursor+=1
        task=NativeGraphTask(client,block,weights=request.weight,
            E0=request.compression_reference_ns_per_byte,D0=request.decompression_reference_ns_per_byte,
            ordinal=request.series_block_ordinal,previous_recipe=request.previous_recipe,
            allow_large_memory=self.allow_large_memory,
            start_from_library=request.stage!='identity' and request.series_block_ordinal==0)
        if request.series_block_ordinal>0:
            try:
                task.previous_recipe_evaluation=task.evaluate_terminal(task.initial_state())
                task.refresh_initial()
            except BaseException:
                task.close();raise
        return task

    def close(self):
        for client in self.clients:client.close()
        self.clients=[]
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
