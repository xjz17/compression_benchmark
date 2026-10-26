"""Versioned algorithm composition environment (legacy module filename).

The search grammar uses registered stage transitions, up to two reference
transforms, and one additional byte-codec segment after a completed segment.
Every successful edge must pass an exact inverse check. Nothing in this module
uses the old normalized-int64 35-action measurement tables.
"""
from __future__ import annotations

import hashlib
import math
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import primitive_operator_action_runtime as p
import unified_action_runtime as u
import native_lz_parameters as lz
import current_action_parameters as action_parameters
import block_state_parameters as block_parameters
import native_window_parameters as window_parameters

from canonical182_policy_paths import (
    PolicyPath, concrete_path, path_from_names, variants_for, catalog_fingerprint, same_stages,
)

STOP = p.STOP_ACTION_ID
HEADER = struct.Struct('<4sBBQI')
SPECS = {s.action_id: s for s in p.POLICY_ACTIONS}
VARIANT_PATHS = {v: path_from_names((v,)) for i in SPECS for v in variants_for(i)}
BY_NAME = {s.name: s.action_id for s in p.POLICY_ACTIONS}
BY_NAME.update({v: path[0] for v, path in VARIANT_PATHS.items()})
DESC = {i: (u.describe_policy_action(i) if hasattr(u, 'describe_policy_action')
            else u.describe_action(s.name)) for i, s in SPECS.items()}
VARIANT_DESC = {v: u.describe_policy_action(path[0], io_variant=v) for v, path in VARIANT_PATHS.items()}
EXCLUDED = {path_from_names((n,))[0] for n in p.CAMEL_ACTIONS}
# This range belongs to the frozen 206 catalog, never the renumbered policy IDs.
_legacy_specs = getattr(p, 'LEGACY_206_POLICY_ACTIONS', p.POLICY_ACTIONS)
REF_VARIANTS = frozenset(s.name for s in _legacy_specs[:29] if s.name in VARIANT_PATHS)
REFS = frozenset(VARIANT_PATHS[v][0] for v in REF_VARIANTS)
_SHUFFLE_WIDTHS = {p.LEGACY_206_POLICY_ACTIONS[i].name: width
                   for i, width in ((158,1),(159,4),(160,8))}


def blosc_prefix_width(path):
    """Width selected by the registered upstream SHF1 adapter, never by dtype."""
    for variant in reversed(concrete_path(path).io_variants):
        if variant in _SHUFFLE_WIDTHS:
            return _SHUFFLE_WIDTHS[variant]
    return None


def _seqs():
    out = set()
    for names in u._INDEPENDENT_STAGEWISE_SEQUENCES:
        path = path_from_names(names)
        if not EXCLUDED.intersection(path):
            out.add(path)
    return tuple(sorted(out))


SEQUENCES = _seqs()
ROOT_VARIANTS = frozenset({s.io_variants[0] for s in SEQUENCES} |
    {v for v, d in VARIANT_DESC.items() if d.execution_mode == 'standalone'})
ROOTS = frozenset(VARIANT_PATHS[v][0] for v in ROOT_VARIANTS) - EXCLUDED
BYTE_ROOT_VARIANTS = frozenset(v for v in ROOT_VARIANTS
                               if 'bytes' in VARIANT_DESC[v].accepted_input_kinds)
BYTE_ROOTS = frozenset(VARIANT_PATHS[v][0] for v in BYTE_ROOT_VARIANTS) - EXCLUDED
ENTROPY_VARIANTS = frozenset((
    'op.entropy_huffman.project_bytes.cpp_v1',
    'op.entropy_range.project_bytes.cpp_v1',
    'op.entropy_fse.zstd_rans.cpp_reference_v1',
))
ENTROPY = frozenset(BY_NAME[n] for n in ENTROPY_VARIANTS)
NEXT = defaultdict(set)
VARIANT_NEXT = defaultdict(set)
for seq in SEQUENCES:
    for a, b, va, vb in zip(seq, seq[1:], seq.io_variants, seq.io_variants[1:]):
        NEXT[a].add(b)
        VARIANT_NEXT[va].add(vb)


def is_reference(path):
    return bool(path) and all(v in REF_VARIANTS for v in concrete_path(path).io_variants)


def is_standalone(path):
    return bool(path) and VARIANT_DESC[concrete_path(path).io_variants[-1]].execution_mode == 'standalone'


@dataclass
class Node:
    data: u.ActionData
    layers: int
    frame_bytes: int
    payload_bytes: int


def make_frame(column, node):
    code = {'int64': 1, 'float64': 2}[column['kind']]
    return HEADER.pack(b'C206', code, node.layers, len(column['raw']) // 8,
                       zlib.crc32(column['raw'])) + node.data.payload


def decode_frame(frame):
    magic, kind_code, layers, count, checksum = HEADER.unpack_from(frame)
    if magic != b'C206' or kind_code not in (1, 2):
        raise ValueError('invalid experiment frame')
    current = u.ActionData.bytes(frame[HEADER.size:])
    for _ in range(layers):
        meta, _payload = u._unpack_result(current)
        if meta.get('io_variant') is not None:
            current = u.run_policy_action(meta.get('policy_action_id', meta['actions'][0]), current,
                                          registry_version=meta.get('policy_registry_version', p.ACTION_REGISTRY_VERSION),
                                          reverse=True)
        else:
            current = u.run_actions(meta['actions'], current, reverse=True)
    if len(current.payload) != count * 8 or zlib.crc32(current.payload) != checksum:
        raise ValueError('decoded count/checksum mismatch')
    return current.payload


class Env:
    def __init__(self, column, sample_rows=2048, max_steps=7):
        self.column = dict(column)
        raw = column['raw'] if not sample_rows else column['raw'][:sample_rows * 8]
        self.column.update(raw=raw, count=len(raw)//8)
        self.source = u.ActionData(column['kind'], raw, len(raw)//8)
        self.max_steps = max_steps
        self.stop = STOP
        self.stop_id = STOP
        self.action_count = STOP + 1
        self.policy_registry_version = p.ACTION_REGISTRY_VERSION
        self.catalog_fingerprint = catalog_fingerprint()
        self.nodes = {(): Node(self.source, 0, HEADER.size + len(raw), len(raw))}
        self.invalid = {}
        self.best_prefix = ()
        self.records = []
        self.forward_sec = 0.0
        self.inverse_sec = 0.0
        self.cache_hits = 0
        self.features_cache = {}
        self._base_features = self._summarize(raw)
        # Policy IDs remain sparse in the frozen 0..126 slot range after A005
        # retirement.  Size this segment by the preserved STOP boundary rather
        # than by the 126 active records, so ID 126 cannot overwrite state data.
        self.feature_dim = len(self._base_features) + STOP + 12

    def _summarize(self, raw):
        buf = np.frombuffer(raw, dtype=np.uint8)
        counts = np.bincount(buf[:16384], minlength=256).astype(float)
        probs = counts[counts > 0] / max(1, counts.sum())
        entropy = float(-np.sum(probs*np.log2(probs))) / 8
        vals = np.frombuffer(raw, dtype='<i8' if self.column['kind']=='int64' else '<f8')[:2048]
        finite = vals[np.isfinite(vals)].astype(float)
        if not len(finite): finite=np.array([0.])
        def slog(x): return float(np.sign(x)*np.log1p(abs(x)))/50
        return np.array([self.column['kind']=='float64', math.log1p(len(vals))/20,
                         entropy, len(np.unique(vals))/max(1,len(vals)),
                         np.mean(vals[1:]==vals[:-1]) if len(vals)>1 else 1,
                         slog(np.median(finite)), slog(np.std(finite)),
                         slog(np.median(np.abs(np.diff(finite)))) if len(finite)>1 else 0], dtype=float)

    def features(self, prefix):
        if prefix not in self.features_cache:
            x = np.zeros(self.feature_dim)
            x[:len(self._base_features)] = self._base_features
            if prefix: x[len(self._base_features)+prefix[-1]] = 1
            n = self.nodes[prefix]
            b = self.segments(prefix)
            params = {key:value for q in getattr(prefix, 'parameters', ()) for key,value in q.items()}
            by_algorithm = {SPECS[i].name: q for i, q in zip(prefix, getattr(prefix, 'parameters', ()))}
            x[-12:-9] = [
                by_algorithm.get('op.block_state_control.blosc1_residual_core.cpp_v2', {}).get('typesize', 0)/8,
                by_algorithm.get('op.block_state_control.blosc2_residual_core.cpp_v2', {}).get('typesize', 0)/8,
                by_algorithm.get('op.block_state_control.deflate_residual_core.cpp_v2', {}).get('level', 0)/9,
            ]
            x[-9] = params.get('window_bytes', 0)/65535
            x[-8:-6] = [params.get('min_match', params.get('min', 0))/64, params.get('max_match', params.get('max', 0))/256]
            x[-6:] = [len(prefix)/self.max_steps, np.log1p(n.frame_bytes)/20,
                      np.clip(math.log(max(1,n.frame_bytes)/max(1,len(self.source.payload))),-10,10)/10,
                      len(b)/2, prefix in self.completed_prefixes(prefix), self.source.kind=='int64']
            self.features_cache[prefix] = x
        return self.features_cache[prefix]

    def segments(self, prefix):
        """Parse completed registered segments; retain an unfinished final one."""
        if not prefix: return []
        # Longest completed registered prefix, including paired reference stages.
        candidates = [s for s in SEQUENCES if same_stages(prefix[:len(s)], s)]
        if is_reference(prefix[:1]):
            k = 1
            if len(prefix)>1 and is_reference(prefix[:2]): k=2
            candidates.append(prefix[:k])
        if is_standalone(prefix[:1]) or prefix[0] in ENTROPY:
            candidates.append(prefix[:1])
        if not candidates: return [prefix]
        first = max(candidates, key=lambda s:(len(s),s))
        return [first] + ([prefix[len(first):]] if len(first)<len(prefix) else [])

    def completed_prefixes(self, prefix):
        out=set()
        for seq in SEQUENCES:
            if same_stages(prefix[-len(seq):], seq): out.add(prefix)
        if prefix and (is_reference(prefix) or is_standalone(prefix) or prefix[-1] in ENTROPY):
            out.add(prefix)
        return out

    def legal_io_variants(self, prefix, action=None):
        """Context options below one algorithm ID; no extra actor logits."""
        if len(prefix) >= self.max_steps:
            return ()
        if not prefix:
            choices = {v for v in ROOT_VARIANTS
                       if self.source.kind in VARIANT_DESC[v].accepted_input_kinds
                       or 'bytes' in VARIANT_DESC[v].accepted_input_kinds} | set(ENTROPY_VARIANTS)
        else:
            path = concrete_path(prefix)
            choices = set(VARIANT_NEXT.get(path.io_variants[-1], ()))
            if len(path) == 1 and is_reference(path):
                choices |= REF_VARIANTS
            if path in self.completed_prefixes(path) and len(self.segments(path)) < 2:
                choices |= BYTE_ROOT_VARIANTS | ENTROPY_VARIANTS
        choices = {v for v in choices if VARIANT_PATHS[v][0] not in EXCLUDED
                   and VARIANT_PATHS[v][0] not in prefix}
        if action is not None:
            choices = {v for v in choices if VARIANT_PATHS[v][0] == action}
        return tuple(sorted(choices))

    def legal(self, prefix):
        return tuple(sorted({VARIANT_PATHS[v][0] for v in self.legal_io_variants(prefix)} | {STOP}))

    def parameter_choices(self, prefix, action):
        if action not in SPECS:
            return ()
        name = SPECS[action].name
        choices = action_parameters.choices_for_names((name,))
        if name in (block_parameters.BLOSC1, block_parameters.BLOSC2):
            width = blosc_prefix_width(prefix)
            key = action_parameters.BY_NAME[name].option_key
            return tuple(q for q in choices if q[key] == width)
        return choices

    def transition(self, prefix, action, *, io_variant=None, parameters=None):
        if action == STOP:
            return prefix
        if parameters is None and action in SPECS and SPECS[action].name in (block_parameters.BLOSC1, block_parameters.BLOSC2):
            compatible = self.parameter_choices(prefix, action)
            if len(compatible) == 1:
                parameters = compatible[0]
        if parameters and action in SPECS and SPECS[action].name in action_parameters.BY_NAME:
            # The conditional head uses option-scoped keys; stage APIs expose
            # local keys (typesize/level). Retain support for direct stage calls.
            module = action_parameters.BY_NAME[SPECS[action].name]
            local_keys = set(module.parameter_candidates()[0])
            if set(parameters) != local_keys:
                parameters = action_parameters.normalize_option((SPECS[action].name,), parameters)
                parameters = action_parameters.for_stage(SPECS[action].name, parameters)
        prefix = concrete_path(prefix)
        if io_variant is None:
            choices = self.legal_io_variants(prefix, action)
            if len(choices) == 1:
                io_variant = choices[0]
            else:
                try:
                    source = self.nodes[prefix].data
                    if not prefix and not any(self.source.kind in VARIANT_DESC[v].accepted_input_kinds for v in choices):
                        source = u.ActionData.bytes(self.source.payload)
                    io_variant = u.resolve_policy_io_variant(action, source)
                    if io_variant not in choices:
                        raise ValueError('resolved variant is outside the registered path context')
                except Exception as exc:
                    error = f'policy ID {action} requires explicit io_variant: {exc}'
                    child = prefix + PolicyPath((action,))
                    if child in self.invalid:
                        self.cache_hits += 1
                        return None
                    self.invalid[child] = error
                    self.records.append(dict(prefix=list(child), io_variants=list(child.io_variants),parameters=list(child.parameters),
                        policy_registry_version=p.ACTION_REGISTRY_VERSION, status='rejected', error=error,
                        forward_sec=0.0, inverse_sec=0.0, frame_bytes=None))
                    return None
        child = prefix + concrete_path((int(action),), (io_variant,), (parameters or {},))
        if child in self.nodes:
            self.cache_hits += 1
            return child
        if child in self.invalid:
            self.cache_hits += 1
            return None
        parent=self.nodes[prefix]
        before=parent.data
        ref_pair = len(child)<=2 and is_reference(child)
        if not prefix:
            before=self.source if self.source.kind in VARIANT_DESC[io_variant].accepted_input_kinds else u.ActionData.bytes(self.source.payload)
        t=time.perf_counter()
        forward_end=t
        try:
            if ref_pair:
                out=u.run_actions(list(child.io_variants), self.source,
                                  max_point=self.column['max_point'])
            else:
                out=u.run_policy_action(action,before,registry_version=p.ACTION_REGISTRY_VERSION,
                                        io_variant=io_variant, parameters=child.parameters[-1], max_point=self.column['max_point'])
            forward_end=time.perf_counter()
            if len(out.payload) > max(1_048_576,64*len(self.source.payload)):
                raise ValueError('intermediate representation exceeds declared expansion bound')
            if ref_pair:
                inv=u.run_actions(list(child.io_variants),out,reverse=True)
                expected=self.source
            else:
                inv=u.run_policy_action(action,out,registry_version=p.ACTION_REGISTRY_VERSION,io_variant=io_variant,reverse=True)
                expected=before
            if inv != expected: raise ValueError('bitwise inverse mismatch')
            end=time.perf_counter()
            payload_size=len(u._unpack_result(out)[1])
            node=Node(out,1 if ref_pair else parent.layers+1,HEADER.size+len(out.payload),payload_size)
            self.nodes[child]=node
            if (node.frame_bytes,child)<(self.nodes[self.best_prefix].frame_bytes,self.best_prefix): self.best_prefix=child
            status='ok'; error=''
        except Exception as exc:
            end=time.perf_counter()
            self.invalid[child]=str(exc)[:400]
            status='rejected';error=f'{type(exc).__name__}: {exc}'[:400]
        self.forward_sec+=forward_end-t if forward_end>t else end-t
        self.inverse_sec+=end-forward_end if forward_end>t else 0
        self.records.append(dict(prefix=list(child),io_variants=list(child.io_variants),parameters=list(child.parameters),
                                 policy_registry_version=p.ACTION_REGISTRY_VERSION,status=status,error=error,
                                 forward_sec=forward_end-t if forward_end>t else end-t,
                                 inverse_sec=end-forward_end if forward_end>t else 0,
                                 frame_bytes=self.nodes[child].frame_bytes if child in self.nodes else None))
        return child if child in self.nodes else None

    def reward(self, prefix):
        return float(np.clip(-math.log(self.nodes[prefix].frame_bytes/max(1,len(self.source.payload))),-10,10))

    def complete(self,prefix):
        """One fixed shortest compatible recipe suffix, never a suffix oracle."""
        suffixes=[]
        for k in range(1,len(prefix)+1):
            tail=prefix[-k:]
            for seq in SEQUENCES:
                if len(seq)>k and same_stages(seq[:k], tail) and len(prefix)+len(seq)-k<=self.max_steps:
                    suffixes.append(seq[k:])
        if not suffixes: return prefix
        suffix=min(suffixes,key=lambda s:(len(s),s))
        current=prefix
        for a, variant, params in zip(suffix, suffix.io_variants, suffix.parameters):
            if a in current: return prefix
            nxt=self.transition(current,a,io_variant=variant,parameters=params)
            if nxt is None: return prefix
            current=nxt
        return current

    def summary(self):
        return dict(unique_edges=len(self.records),valid_edges=len(self.nodes)-1,rejected_edges=len(self.invalid),
                    cache_hits=self.cache_hits,forward_sec=self.forward_sec,inverse_sec=self.inverse_sec,
                    best_frame_bytes=self.nodes[self.best_prefix].frame_bytes,
                    policy_registry_version=p.ACTION_REGISTRY_VERSION,
                    best_ids=list(self.best_prefix),
                    best_io_variants=list(getattr(self.best_prefix, "io_variants", ())),
                    best_parameters=list(getattr(self.best_prefix, "parameters", ())))


def beam29(env,width=29,progress=False):
    frontier=[()]; levels=[]
    for depth in range(env.max_steps):
        scored=[]
        for prefix in frontier:
            for a in env.legal(prefix):
                if a==STOP: continue
                for variant in env.legal_io_variants(prefix, a):
                    for params in env.parameter_choices(prefix, a) or ({},):
                        nxt=env.transition(prefix,a,io_variant=variant,parameters=params)
                        if nxt is None: continue
                        completion=env.complete(nxt)
                        scored.append((min(env.nodes[nxt].frame_bytes,env.nodes[completion].frame_bytes),nxt))
        scored.sort(key=lambda t:(t[0],t[1]))
        frontier=[prefix for _,prefix in scored[:width]]
        levels.append(dict(depth=depth+1,expanded=len(scored),retained=len(frontier)))
        if progress:
            print(f"Beam29 {env.column['dataset']}:{env.column['name']} depth={depth+1} edges={len(env.records)} best_bytes={env.nodes[env.best_prefix].frame_bytes}",flush=True)
        if not frontier: break
    return env.best_prefix,levels


def encode_cold(column,prefix,io_variants=None,parameters=None):
    """Fresh encoding, deliberately excludes inverse validation from the timer."""
    prefix=concrete_path(prefix,io_variants,parameters)
    source=u.ActionData(column['kind'],column['raw'],len(column['raw'])//8)
    if not prefix:
        node=Node(source,0,HEADER.size+len(source.payload),len(source.payload))
        return make_frame(column,node)
    current=source if source.kind in VARIANT_DESC[prefix.io_variants[0]].accepted_input_kinds else u.ActionData.bytes(source.payload)
    offset=0;layers=0
    if is_reference(prefix[:1]):
        offset=1
        if len(prefix)>1 and is_reference(prefix[:2]):offset=2
        current=u.run_actions(list(prefix.io_variants[:offset]),source,max_point=column['max_point'])
        layers=1
    for action, variant, params in zip(prefix[offset:], prefix.io_variants[offset:], prefix.parameters[offset:]):
        current=u.run_policy_action(action,current,registry_version=p.ACTION_REGISTRY_VERSION,io_variant=variant,parameters=params,max_point=column['max_point'])
        layers+=1
    return make_frame(column,Node(current,layers,HEADER.size+len(current.payload),len(u._unpack_result(current)[1])))


def grammar_manifest():
    covered=set(ROOTS)|set(ENTROPY)|{i for seq in SEQUENCES for i in seq}
    return dict(version=p.ACTION_REGISTRY_VERSION,action_count=STOP+1,
                active_action_count=len(SPECS),reserved_action_slots=STOP,
                selectable_count=len(SPECS)-len(EXCLUDED),stop=STOP,
                excluded_camel=sorted(EXCLUDED),grammar_covered=sorted(covered),
                missing=sorted(set(SPECS)-covered-EXCLUDED),sequences=[list(s) for s in SEQUENCES],
                sequence_io_variants=[list(s.io_variants) for s in SEQUENCES],
                sequence_parameters=[list(s.parameters) for s in SEQUENCES],
                lz_parameter_candidates=lz.parameter_candidates(), lz_default_parameters=lz.default_parameters(),
                window_parameter_candidates=window_parameters.parameter_candidates(), window_default_parameters=window_parameters.default_parameters(),
                action_parameter_schemas=action_parameters.schemas(),
                conditional_parameter_candidates=action_parameters.all_candidates(),
                catalog_fingerprint=catalog_fingerprint(),
                variant_selection="context-unique for actor; explicit variants enumerated by beam/recipes",
                comparison_scope="atomic actor and beam have different variant reachability when context is ambiguous; not a matched search-space comparison",
                root_actions=sorted(ROOTS),root_io_variants=sorted(ROOT_VARIANTS),
                reference_io_variants=sorted(REF_VARIANTS),byte_segment_starters=sorted(BYTE_ROOTS),
                entropy_actions=sorted(ENTROPY),max_reference_prefix=2,max_segments=2,
                restriction='registered stage adjacency plus at most one outer byte segment; repeated IDs forbidden; real inverse check required')
