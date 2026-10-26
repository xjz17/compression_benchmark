#!/usr/bin/env python3
"""GPU-training adapter to immutable C++ graph states via an untimed JSONL pipe.

The policy-update host may be Python/PyTorch. Features, masks, edits and terminal
graph encode/decode come from C++. No Python wall-clock or IPC time becomes an
E/D sample. Frozen serving uses the C++ search command, including policy time.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import sys
import tempfile
import threading
from typing import Any

import numpy as np
import native_prepared_cohort_v1 as data
import native_experiment_scope_v1 as experiment_scope
import weight_conditioned_grpo_v1 as grpo

REGISTRY='50d9be2dbc4e77ee8c7fd8a621e68bc869a55f0630daf6b08d6a5def86daf2fd'
ACTION_COUNT=6341
MAX_PROTOCOL_RESPONSE_BYTES=2*(1<<30)+(16<<20) # 1 GiB NGF1 in hex plus metadata


class NativeClient:
    """One serialized native worker. CPU affinity is inherited by all children."""
    def __init__(self, cli: Path, catalog: Path, *, cpu: int | None=None,
                 model: Path | None=None, library: Path | None=None, timeout: float=120,
                 profile: Path=experiment_scope.DEFAULT_PROFILE):
        self.experiment_scope=experiment_scope.load(catalog,profile)
        self.timeout=timeout
        self.counter=0
        self.lock=threading.RLock()
        self.stderr=tempfile.TemporaryFile()
        cmd=[str(cli.resolve()),'serve','--catalog',str(catalog.resolve()),
             '--experiment-profile',self.experiment_scope['profile_path']]
        if model is not None: cmd+=['--model',str(model.resolve())]
        if library is not None:
            cmd+=['--solution-library',str(library.resolve()),
                  '--solution-library-sha256',data.digest(library.read_bytes())]
        if cpu is not None:
            data.require(hasattr(os,'sched_getaffinity') and cpu in os.sched_getaffinity(0),'worker CPU unavailable')
            cmd=['taskset','-c',str(cpu),*cmd]
        if sys.platform.startswith('linux'):
            # An owned native worker cannot exhaust shared host memory even
            # when an upstream probability model requests excessive storage.
            cmd=['prlimit','--as='+str(64<<30),'--',*cmd]
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=self.stderr,bufsize=0)
        self.buffer=bytearray()

    def call(self, command: dict) -> dict:
        import time
        with self.lock:
            data.require(self.process.poll() is None,'native worker already exited')
            self.counter+=1
            request={**command,'request_id':self.counter}
            payload=data.canonical(request)+b'\n'
            offset=0
            while offset<len(payload):
                count=self.process.stdin.write(payload[offset:])
                if not count: raise RuntimeError('native worker input pipe closed')
                offset+=count
            self.process.stdin.flush()
            deadline=time.monotonic()+self.timeout
            newline=self.buffer.find(b'\n')
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout,selectors.EVENT_READ)
                while newline<0:
                    events=selector.select(max(0,deadline-time.monotonic()))
                    if not events:
                        self.close()
                        raise TimeoutError('native graph request exceeded deadline')
                    piece=os.read(self.process.stdout.fileno(),65536)
                    if not piece:
                        self.stderr.seek(0)
                        message=self.stderr.read(4096).decode(errors='replace')
                        raise RuntimeError('native worker closed pipe: '+message)
                    previous_size=len(self.buffer)
                    self.buffer.extend(piece)
                    data.require(len(self.buffer)<=MAX_PROTOCOL_RESPONSE_BYTES,'oversized native protocol response')
                    newline=self.buffer.find(b'\n',previous_size)
            line=bytes(self.buffer[:newline]);del self.buffer[:newline+1]
            answer=json.loads(line)
            data.require(answer.get('request_id')==self.counter,'native request/response mismatch')
            if answer.get('ok') is not True:
                raise RuntimeError('native request rejected: '+str(answer.get('error','unknown error')))
            if command.get('cmd')=='catalog':
                experiment_scope.verify_native(answer,self.experiment_scope)
            elif 'catalog' in answer:
                experiment_scope.verify_native(answer['catalog'],self.experiment_scope)
            for field in ('recipe_action_ids','best_recipe_action_ids'):
                if field in answer:experiment_scope.verify_recipe(answer[field],self.experiment_scope)
            for candidate in answer.get('candidates',[]):
                experiment_scope.verify_recipe(candidate['recipe_action_ids'],self.experiment_scope)
            return answer

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for f in (self.process.stdin,self.process.stdout,self.stderr):
            if f and not f.closed: f.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


@dataclass(frozen=True)
class State:
    handle: str
    payload: dict[str,Any]


class NativeGraphTask:
    action_count=ACTION_COUNT
    stop_action=0
    registry_fingerprint=REGISTRY

    def __init__(self, client: NativeClient, block: data.Block, *, weights,
                 E0: float,D0: float,ordinal: int=0,previous_recipe=(),
                 starting_recipe=(),require_complete: bool=True,
                 warmups: int=1,repeats: int=5,
                 allow_large_memory: bool=False,start_from_library: bool=False):
        self.client=client
        self._closed=True
        self.block=block
        self.weights=grpo.validate_weight(weights)
        self.E0,self.D0=float(E0),float(D0)
        data.require(all(math.isfinite(v) and v>0 for v in (self.E0,self.D0)),'invalid frozen time references')
        data.require(block.record['original_numeric_bytes']>0,'zero B_raw must be recorded N/A, not a GRPO task')
        data.require(ordinal>=0 and (ordinal>0 or not previous_recipe), 'previous recipe/ordinal mismatch')
        data.require(not start_from_library or (ordinal==0 and not starting_recipe), 'library selection is first-block only')
        scope=getattr(client,'experiment_scope',None)
        if scope is not None:
            experiment_scope.verify_recipe(previous_recipe,scope)
            experiment_scope.verify_recipe(starting_recipe,scope)
        self.task_id=block.fingerprint
        self.data_fingerprint=block.fingerprint
        self.root_fingerprint=data.digest(data.canonical({'block':block.fingerprint,'previous_recipe':list(previous_recipe),
            'starting_recipe':list(starting_recipe),'ordinal':ordinal,'E0':E0,'D0':D0}))
        self._request=dict(cmd='reset_from_library' if start_from_library else 'reset',task_id=self.task_id,roots=block.roots(),weights=self.weights.tolist(),
            compression_reference_ns_per_byte=E0,decompression_reference_ns_per_byte=D0,
            previous_recipe_action_ids=list(previous_recipe),starting_recipe_action_ids=list(starting_recipe),
            block_index=ordinal,continuation_edit_mode=ordinal>0,max_recipe_edits=2,
            original_numeric_bytes=block.record['original_numeric_bytes'],
            timing_warmups=warmups,timing_repetitions=repeats,require_complete_native_catalog=require_complete,
            allow_large_memory=allow_large_memory)
        self.repeats=repeats
        self.budget=64 if ordinal==0 else 16
        response=client.call(self._request)
        self.task_handle=response['task_handle']
        self._closed=False
        catalog=response['catalog']
        data.require(catalog['action_count']==ACTION_COUNT and catalog['fingerprint']==REGISTRY,'native catalog identity drift')
        data.require(not require_complete or catalog['complete_native_graph_space'] is True,'incomplete native catalog cannot train')
        self.max_steps=response['max_steps']
        self._initial=self._state(response)
        self._task_features=self._vector(response['task_features'],18)
        self.library_selection=response.get('library_selection')

    @staticmethod
    def _vector(value, size):
        out=np.asarray(value,dtype=np.float64)
        data.require(out.shape==(size,) and np.isfinite(out).all(),'invalid native feature vector')
        return out

    def _state(self, response):
        data.require(response['task_handle']==self.task_handle,'cross-task native state handle')
        self._vector(response['state_features'],28)
        ids=response['legal_action_ids']
        data.require(len(set(ids))==len(ids) and all(type(a) is int and 0<=a<ACTION_COUNT for a in ids),'invalid native action mask')
        scope=getattr(self.client,'experiment_scope',None)
        if scope is not None:experiment_scope.verify_recipe(ids,scope)
        return State(response['state_handle'],response)

    def task_features(self): return self._task_features.copy()
    def initial_state(self): return self._initial
    def refresh_initial(self):
        response=self.client.call(dict(cmd='inspect',task_handle=self.task_handle,state_handle=self._initial.handle))
        self._initial=self._state(response)
        self._task_features=self._vector(response['task_features'],18)
    def state_features(self,state):
        data.require(state.payload['task_handle']==self.task_handle,'cross-task native state handle')
        return self._vector(state.payload['state_features'],28)
    def legal_actions(self,state):
        data.require(state.payload['task_handle']==self.task_handle,'cross-task native state handle')
        return tuple(state.payload['legal_action_ids'])

    def transition(self,state,action):
        data.require(type(action) is int and action in self.legal_actions(state),'action is not legal')
        answer=self.client.call(dict(cmd='step',state_handle=state.handle,action_id=action))
        return self._state(answer)

    def evaluate_terminal(self,state):
        data.require(state.payload['task_handle']==self.task_handle,'cross-task native state handle')
        data.require(state.payload['terminal_evaluable'],'incomplete native graph cannot STOP')
        result=self.client.call(dict(cmd='evaluate_terminal',state_handle=state.handle,include_frame_hex=True))
        raw=self.block.record['original_numeric_bytes']
        data.require(result['original_numeric_bytes']==raw,'native output denominator mismatch')
        data.require(result['candidate_budget']==self.budget and 1<=result['candidate_ordinal']<=self.budget,'native candidate budget exceeded')
        if result.get('terminal_status')=='failed':
            data.require(not any(k in result for k in ('frame_hex','frame_bytes','compression_ratio','encode_samples_ns','decode_samples_ns')),
                         'failed candidate must not fabricate compression measurements')
            data.require(result['max_frame_bytes']==1<<30,'native failure-bound limit drift')
            return grpo.TerminalFailure({'recipe_action_ids':result['recipe_action_ids'],
                'terminal_status':'failed','native_receipt':result},raw,result['max_frame_bytes'],result['error'])
        frame=bytes.fromhex(result['frame_hex'])
        data.require(result['original_numeric_bytes']==raw and result['frame_bytes']==len(frame),'native output size/denominator mismatch')
        data.require(result['frame_sha256']==data.digest(frame),'native full frame SHA mismatch')
        data.require(result['candidate_budget']==self.budget and 1<=result['candidate_ordinal']<=self.budget,'native candidate budget exceeded')
        for field in ('encode_samples_ns','decode_samples_ns'):
            data.require(len(result[field])==self.repeats and all(type(v) is int and v>=0 for v in result[field]),'invalid native timing samples')
        metrics=grpo.CompressionMetrics(raw,int(statistics.median(result['encode_samples_ns'])),
            int(statistics.median(result['decode_samples_ns'])),self.E0,self.D0)
        outcome=grpo.TerminalOutcome({'recipe_action_ids':result['recipe_action_ids'],
            'native_receipt':{k:v for k,v in result.items() if k!='frame_hex'},
            'timing_scope':'C++ whole fixed-recipe graph and complete NGF1 encode/decode; excludes GPU training and IPC'},frame,metrics)
        data.require(math.isclose(grpo.objective_value(outcome,self.weights),result['J_selection'],rel_tol=1e-12,abs_tol=1e-12),'C++/training objective mismatch')
        return outcome

    def close(self):
        if not self._closed:
            # A request timeout already closes its worker. There is no native
            # task left to release, and a second request would mask that error.
            process = getattr(self.client, 'process', None)
            if process is not None and process.poll() is not None:
                self._closed=True
                return
            self.client.call(dict(cmd='release_task',task_handle=self.task_handle))
            self._closed=True

    def __enter__(self): return self
    def __exit__(self,exc_type,exc_value,traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_type is None:
                raise
            # Keep the triggering failure visible. A broken release protocol
            # must not leave the owned worker and its task memory resident.
            self.cleanup_error=repr(cleanup_error)
            try:
                self.client.close()
            except Exception:
                pass
            self._closed=True
    def __del__(self):
        # The reusable training engine drops one task after each chronological
        # block. Preserve bounded native session lifetime even on exceptions.
        try: self.close()
        except Exception: pass
