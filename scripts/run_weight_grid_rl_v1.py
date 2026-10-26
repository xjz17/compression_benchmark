#!/usr/bin/env python3
"""Fresh paired PPO/GRPO training on all 66 integer-tenth reward weights.

This version keeps the historical 17-family / 38-prefix / 38-tail grammar.
It shares only newly measured raw metric vectors, never another run's candidates,
policy, rewards, or selected winner. All writes are in a new output directory.
"""
from __future__ import annotations

import os
for _thread_name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_name] = '1'

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'analysis_outputs/raw_rate_multiobjective_rl_20260907_v1'
sys.path.insert(0, str(PRIOR))
import reward_protocol as rp
import run_multiobjective as prior
from trajectory_audit import TrajectoryAudit

SCHEMA = 'integer-tenth-weight-grid-rl-v1'
SEEDS = (20260906, 20260907, 20260908)
COMPONENTS = ('compression_rate', 'compression_time', 'decompression_time')
HISTORICAL_BENCHMARK = PRIOR / 'benchmark/SUMMARY.json'
POLICY = rp.legacy_tail.policy
TAILS = rp.legacy_tail.TAILS
BASE_CONFIG = dict(iterations=64, episodes=32, group_size=8, hidden=48, epochs=4, lr=.001,
                   refine_top_k=8, clip_ratio=.2, entropy_coefficient=.01, gae_lambda=.95,
                   max_policy_decisions=7, max_prefix=6, max_mixed_coarse_depth=10)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    prior.dump(Path(path), value)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return digest(canonical(value).encode())


def experiment_identity(config, columns, provenance):
    # Snapshot wall time is descriptive only; it must not defeat safe resume.
    return fingerprint(dict(schema=SCHEMA, config=config, columns=columns,
                            file_sha256=provenance['file_sha256'], dependencies=provenance['dependencies']))


def weight_grid():
    rows = [(a, b, 10-a-b) for a in range(11) for b in range(11-a)]
    assert len(rows) == len(set(rows)) == 66
    return rows


def weight_id(weight):
    return 'a%02d_b%02d_c%02d' % tuple(weight)


def validate_weight(weight):
    if (len(weight) != 3 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in weight)
            or sum(weight) != 10):
        raise ValueError('weights must be three nonnegative integer tenths summing to 10')
    return tuple(weight)


def score(record, reference, weight):
    weight = validate_weight(weight)
    rho = record['bytes'] / record['raw_bytes']
    enc, dec = record['encode_seconds_median'], record['decode_seconds_median']
    enc0, dec0 = reference['encode_seconds_median'], reference['decode_seconds_median']
    if not all(math.isfinite(v) and v > 0 for v in (rho, enc, dec, enc0, dec0)):
        raise ValueError('positive finite complete-frame rate and times required')
    parts = (rho, enc / (enc + enc0), dec / (dec + dec0))
    cost = sum(w * part for w, part in zip(weight, parts)) / 10
    return {**copy.deepcopy(record), 'compression_rate': rho,
            'encode_time_normalized': parts[1], 'decode_time_normalized': parts[2],
            'component_scores': dict(zip(COMPONENTS, parts)),
            'weights': dict(zip(COMPONENTS, (w/10 for w in weight))),
            'weight_tenths': list(weight), 'weight_id': weight_id(weight),
            'cost': cost, 'score': -cost, 'reward': -cost,
            'component_direction': 'lower_is_better'}


def check_space(output, floor_gib):
    available = shutil.disk_usage(output).free
    if available < floor_gib * 1024**3:
        raise RuntimeError(f'free space {available/1024**3:.2f} GiB below safety floor {floor_gib} GiB')
    return available


class MeasurementOracle:
    """Append-only, run-local SQLite vector cache; no reward or candidate transfer."""
    def __init__(self, path, runtime_identity, *, floor_gib=15, measure=rp.measure_recipe):
        self.path = Path(path)
        self.runtime_identity = runtime_identity
        self.measure_function = measure
        self.floor_gib = floor_gib
        self.stats = Counter()
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (cache_key TEXT PRIMARY KEY, identity TEXT NOT NULL, record TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS phase_totals (phase TEXT PRIMARY KEY, n INTEGER NOT NULL, wall REAL NOT NULL, enc INTEGER NOT NULL, dec INTEGER NOT NULL)')
        if not self.db.execute('SELECT 1 FROM phase_totals LIMIT 1').fetchone():
            self.db.execute("INSERT INTO phase_totals SELECT json_extract(record,'$.measurement_phase'),count(*),"
                "sum(json_extract(record,'$.measurement_wall_seconds')),"
                "coalesce(sum(json_extract(record,'$.complete_encoder_calls')),0),"
                "coalesce(sum(json_extract(record,'$.complete_decoder_calls')),0) FROM vectors GROUP BY 1")
        self.db.commit()

    def measure(self, raw, kind, specs, terminals, *, phase, warmups, repeats, expected=None):
        identity = dict(schema=SCHEMA, runtime=self.runtime_identity, input_sha256=digest(raw),
                        raw_bytes=len(raw), kind=kind, specs=specs, terminals=list(terminals),
                        phase=phase, warmups=warmups, repeats=repeats,
                        timing_scope=rp.PROTOCOL['timing_scope'])
        key = fingerprint(identity)
        found = self.db.execute('SELECT identity,record FROM vectors WHERE cache_key=?', (key,)).fetchone()
        self.stats[phase+'_requests'] += 1
        if found:
            if found[0] != canonical(identity):
                raise AssertionError('cache identity corruption')
            row = json.loads(found[1])
            if (not row.get('lossless') or not row.get('deterministic') or row.get('status') != 'PASS'
                    or row['input_sha256'] != digest(raw) or row['runtime_sha256'] != self.runtime_identity
                    or row['specs'] != specs or row['terminals'] != list(terminals)
                    or row['raw_bytes'] != len(raw) or row['kind'] != kind):
                raise AssertionError('cache certificate does not match request')
            if expected and row['frame_sha256'] != expected:
                raise AssertionError('cache frame does not match sampled frame')
            self.stats[phase+'_hits'] += 1
            return {**row, 'measurement_cache_hit': True, 'request_encoder_calls': 0,
                    'request_decoder_calls': 0}, None
        if self.stats['all_misses'] % 1000 == 0:
            check_space(self.path.parent, self.floor_gib)
        started = time.perf_counter()
        row, frame = self.measure_function(raw, kind, specs, terminals, warmups=warmups, repeats=repeats,
                                           expected_frame_sha256=expected)
        elapsed = time.perf_counter() - started
        if not row['lossless'] or not row['deterministic'] or row['frame_sha256'] != digest(frame):
            raise AssertionError('uncertified fresh measurement')
        row.update(status='PASS', cache_key=key, measurement_phase=phase,
                   measurement_recorded_at=now(), input_sha256=digest(raw), kind=kind,
                   runtime_sha256=self.runtime_identity, measurement_wall_seconds=elapsed)
        # Only vectors and frame certificates: no raw payload or compressed frame.
        self.db.execute('INSERT INTO vectors VALUES (?,?,?)', (key, canonical(identity), canonical(row)))
        self.db.execute('INSERT INTO phase_totals VALUES (?,?,?,?,?) ON CONFLICT(phase) DO UPDATE SET '
                        'n=n+1,wall=wall+excluded.wall,enc=enc+excluded.enc,dec=dec+excluded.dec',
                        (phase,1,elapsed,warmups+repeats,warmups+repeats))
        self.db.commit()
        self.stats['all_misses'] += 1
        self.stats[phase+'_misses'] += 1
        self.stats[phase+'_encoder_calls'] += warmups + repeats
        self.stats[phase+'_decoder_calls'] += warmups + repeats
        self.stats[phase+'_measurement_wall_seconds'] += elapsed
        if self.stats['all_misses'] % 1000 == 0:
            self.db.execute('PRAGMA wal_checkpoint(PASSIVE)')
        return {**row, 'measurement_cache_hit': False,
                'request_encoder_calls': warmups+repeats, 'request_decoder_calls': warmups+repeats}, frame

    def totals(self):
        # O(number of phases), not a scan of millions of vectors at every run.
        persistent = self.db.execute('SELECT phase,n,wall,enc,dec FROM phase_totals').fetchall()
        counts = sum(row[1] for row in persistent)
        disk = sum(p.stat().st_size for p in (self.path, Path(str(self.path)+'-wal'), Path(str(self.path)+'-shm')) if p.exists())
        return {'persistent_unique_vectors': counts, 'invocation_requests': dict(self.stats),
                'sqlite_bytes_including_wal_shm': disk,
                'persistent_by_phase': {phase:dict(unique_vectors=n,measurement_wall_seconds=seconds,
                                                   complete_encoder_calls=encodes,complete_decoder_calls=decodes)
                                        for phase,n,seconds,encodes,decodes in persistent}}

    def verify_record(self, record):
        """Read-only certificate check; missing cache entries must not remeasure."""
        item = self.db.execute('SELECT record FROM vectors WHERE cache_key=?', (record.get('cache_key'),)).fetchone()
        if item is None:
            raise AssertionError('frozen vector absent from existing oracle')
        cached = json.loads(item[0])
        if any(record.get(key) != value for key, value in cached.items()):
            raise AssertionError('frozen vector differs from original oracle certificate')

    def close(self):
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()


class GridEnv(rp.legacy_tail.TailEnv):
    def __init__(self, raw, kind, reference, trace_path, *, weight, oracle, **context):
        self.reference, self.weight, self.oracle = copy.deepcopy(reference), validate_weight(weight), oracle
        super().__init__(raw, kind, int(reference['bytes']), trace_path, **context)
        # The inherited initializer has written only its header. Continue as gzip.
        self._trace.close()
        header = self.trace_path.read_text(encoding='utf-8')
        original = self.trace_path
        self.trace_path = Path(str(original)+'.gz')
        self._trace = gzip.open(self.trace_path, 'xt', encoding='utf-8')
        self._trace.write(header)
        original.unlink()

    def _emit(self, record):
        if record.get('event') == 'environment':
            record = {**record, 'schema': SCHEMA, 'measurement_cache_shared': True,
                      'cache_scope': 'new raw metric vectors only; no candidate or policy exposure',
                      'injected_measurement_cache_entries': 0}
        super()._emit({**record, 'weight_tenths': list(self.weight), 'weight_id': weight_id(self.weight)})

    def _measure(self, sequence, origin):
        prefix, terminals = self.parts(sequence)
        try:
            row, frame = self.oracle.measure(self.raw, self.kind, self.spec_for(sequence), terminals,
                                             phase='training', warmups=0, repeats=1)
        except Exception as exc:
            self._fail(sequence, origin, 'shared_oracle_complete_encode_decode', exc)
        row = score(row, self.reference, self.weight)
        self.counters['shared_training_cache_hits' if row['measurement_cache_hit'] else 'shared_training_cache_misses'] += 1
        self.counters['complete_encoder_calls'] += row['request_encoder_calls']
        self.counters['complete_decoder_calls'] += row['request_decoder_calls']
        self.counters['native_codec_calls'] += row['request_encoder_calls'] * len(terminals)
        row.update(sequence=list(sequence), prefix=list(prefix), origin=origin,
                   training_score=row['score'],
                   compression_seconds=row['encode_seconds_median'], decompression_seconds=row['decode_seconds_median'],
                   prefix_depth=len(prefix), mixed_coarse_depth=len(prefix)+rp.legacy_tail.tail_cost(terminals),
                   algorithm_level_depth=len(prefix)+len(terminals))
        return row, frame

    def reward(self, sequence):
        value = super().reward(sequence)
        if value <= POLICY.INVALID_REWARD:
            self.counters['valid_reward_calls_at_or_below_invalid_penalty'] += 1
        return value

    def snapshot(self):
        return {**super().snapshot(),
                'valid_unique_candidates_at_or_below_invalid_penalty': sum(r['score'] <= POLICY.INVALID_REWARD for r in self.measured_results.values()),
                'valid_reward_calls_at_or_below_invalid_penalty': self.counters['valid_reward_calls_at_or_below_invalid_penalty']}

    @staticmethod
    def rank(row):
        return prior.score_rank(row)


class _GzipAuditPath:
    def __init__(self, path): self.path = Path(path)
    @property
    def parent(self): return self.path.parent
    def open(self, mode, encoding=None): return gzip.open(self.path, mode+'t', encoding=encoding)
    def read_bytes(self): return self.path.read_bytes()
    def stat(self): return self.path.stat()
    def __str__(self): return str(self.path)


class GzipTrajectoryAudit(TrajectoryAudit):
    def __init__(self, policy, path, **kwargs):
        super().__init__(policy, path, **kwargs)
        self.path = _GzipAuditPath(str(path)+'.gz')


@contextmanager
def isolated_runner_bindings(weight, oracle):
    """Only replace callables in this process-local imported runner, then restore.

    The policy/optimizer implementation and all existing source files stay intact.
    """
    names = ('add_score', 'MultiobjectiveEnv', 'measure_saved', 'TrajectoryAudit')
    old = {name: getattr(prior, name) for name in names}
    def measure_saved(raw, kind, specs, terminals, folder, *, reference=None, warmups=1, repeats=7,
                      expected=None, sequence=None, origin=None):
        folder.mkdir(parents=True, exist_ok=False)
        row, _ = oracle.measure(raw, kind, specs, terminals, phase='selection', warmups=warmups,
                                repeats=repeats, expected=expected)
        if reference is not None: row = score(row, reference, weight)
        if sequence is not None: row = prior.decorate(row, sequence, origin)
        # Retain selection vectors and hash certificates, not every pool frame.
        row['frame_retention'] = 'vector_only_until_selected'
        dump(folder/'EVIDENCE.json', row)
        return row
    prior.add_score = lambda row, ref: score(row, ref, weight)
    prior.MultiobjectiveEnv = lambda *args, **kwargs: GridEnv(*args, **kwargs, weight=weight, oracle=oracle)
    prior.measure_saved = measure_saved
    prior.TrajectoryAudit = GzipTrajectoryAudit
    try:
        yield
    finally:
        for name, value in old.items(): setattr(prior, name, value)


def persist_selected(raw, kind, row, path):
    """Artifact retention validation; explicitly outside E/D benchmark timers."""
    started = time.perf_counter()
    frame = rp.native_encode(raw, kind, row['specs'], row['terminals'])
    if len(frame) != row['bytes'] or digest(frame) != row['frame_sha256'] or rp.native_decode(frame) != raw:
        raise AssertionError('selected frame retention parity failure')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(frame)
    return {**copy.deepcopy(row), 'frame_path': str(path), 'frame_retention': 'selected_complete_frame',
            'retention_validation_seconds': time.perf_counter()-started}


def verify_completed(row, identity, weight):
    if row['experiment_identity'] != identity or row['weight_tenths'] != list(weight):
        raise AssertionError('completed run identity mismatch')
    receipt = row['trajectory_audit']
    path = Path(receipt['path'])
    if digest(path.read_bytes()) != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
        raise AssertionError('compressed trajectory receipt drift')
    with gzip.open(path, 'rt') as trace:
        if sum(1 for _ in trace) != receipt['event_count']:
            raise AssertionError('trajectory count drift')
    prior.verify_frame(row['selected'])
    if row['budget']['initial_actor_sha256'] == row['budget']['final_actor_sha256']:
        raise AssertionError('completed actor did not change')
    if POLICY.parameter_sha256(POLICY.load_policy(path.parent/'actor.npz')) != row['budget']['final_actor_sha256']:
        raise AssertionError('actor checkpoint differs from training receipt')


def source_snapshot(columns, gate):
    # Force lazy native codec imports before taking the immutable dependency set.
    extension_paths = []
    for name in ('_bz2', '_lzma', '_brotli', 'zstandard.backend_c'):
        mod = importlib.import_module(name)
        extension_paths.append(Path(mod.__file__).resolve())
    paths = prior.source_paths(columns, gate) + extension_paths + [Path(__file__), HISTORICAL_BENCHMARK,
        Path(__file__).with_name('test_weight_grid_rl_v1.py')]
    return prior.provenance_snapshot(sorted(set(paths)))


def prepare_shared(columns, folder, oracle, identity, resume):
    """Freeze all references and fixed tails before the first learner starts."""
    frozen = folder/'REFERENCES_AND_BASELINES.json'
    if frozen.exists():
        obj = read_json(frozen)
        if not resume or obj['experiment_identity'] != identity:
            raise AssertionError('shared baselines identity mismatch')
        if set(obj['columns']) != {c['key'] for c in columns}:
            raise AssertionError('frozen reference column set drift')
        for col in columns:
            item = obj['columns'][col['key']]
            reference = item['reference']
            prior.verify_frame(reference)
            oracle.verify_record(reference)
            if (reference['input_sha256'] != col['input_sha256'] or reference['specs'] != []
                    or reference['terminals'] != ['zlib6'] or reference['kind'] != col['kind']
                    or reference['raw_bytes'] != col['raw_bytes']):
                raise AssertionError('frozen reference is not this column zlib6')
            if len(item['baselines']) != len(TAILS): raise AssertionError('baseline catalog count drift')
            for index, row in enumerate(item['baselines']):
                oracle.verify_record(row)
                if (row['sequence'] != [index] or row['specs'] != [] or row['terminals'] != list(TAILS[index])
                        or row['input_sha256'] != col['input_sha256'] or row['kind'] != col['kind']):
                    raise AssertionError('baseline catalog or input identity drift')
        return obj['columns']
    result = {}
    for col in columns:
        raw = Path(col['input_path']).read_bytes()
        if digest(raw) != col['input_sha256']: raise AssertionError('input drift')
        reference, _ = oracle.measure(raw, col['kind'], [], ('zlib6',), phase='reference', warmups=3, repeats=15)
        reference = persist_selected(raw, col['kind'], reference, folder/'references'/col['key']/'zlib6.acf3')
        reference.update(frozen_before_any_training=True, frozen_at=now())
        baseline_rows = []
        for index, terminals in enumerate(TAILS):
            measured, _ = oracle.measure(raw, col['kind'], [], terminals, phase='selection', warmups=1, repeats=7)
            baseline_rows.append(prior.decorate(measured, [index], 'common_safety_baseline'))
        result[col['key']] = {'reference': reference, 'baselines': baseline_rows}
    dump(frozen, {'experiment_identity': identity, 'frozen_at': now(), 'columns': result})
    return result


def run_training(column, raw, reference, baselines, method, seed, weight, config, folder, oracle, identity, resume):
    output = folder/'RESULT.json'
    if output.exists():
        if not resume: raise RuntimeError('completed output exists; use --resume')
        row = read_json(output)
        if row.get('schema') == SCHEMA:
            verify_completed(row, identity, weight)
            return row
        # The underlying runner may finish just before retention/receipt upgrade
        # is interrupted. Preserve that partial attempt, then train afresh.
    check_space(folder.parent, oracle.floor_gib)
    if folder.exists(): prior.prepare_incomplete(folder, resume); folder.rmdir()
    fallback = min((score(r, reference, weight) for r in baselines), key=prior.score_rank)
    before = oracle.stats.copy()
    with isolated_runner_bindings(weight, oracle):
        row = prior.run_method(column, raw, reference, fallback, method, seed, config, folder, identity, False)
    row['selected'] = persist_selected(raw, column['kind'], row['selected'], folder/'selected.acf3')
    for key in ('sampled_only', 'sampled_best', 'greedy', 'safety_baseline'):
        if row.get(key) and row[key]['frame_sha256'] == row['selected']['frame_sha256']:
            row[key]['frame_path'] = row['selected']['frame_path']
    delta = dict(oracle.stats-before)
    row.update(schema=SCHEMA, weight_tenths=list(weight), weight_id=weight_id(weight),
               trace_path=str(folder/'candidates.jsonl.gz'), oracle_run_delta=delta,
               cache_hit_is_not_new_measurement=True, historical_winner_injection=False,
               shared_cache_does_not_expand_candidate_pool=True)
    budget = row['budget']
    budget['logical_refinement_encoder_calls'] = budget.pop('refinement_encoder_calls')
    budget['logical_refinement_decoder_calls'] = budget.pop('refinement_decoder_calls')
    budget['refinement_encoder_calls'] = delta.get('selection_encoder_calls', 0)
    budget['refinement_decoder_calls'] = delta.get('selection_decoder_calls', 0)
    budget['actual_new_training_encoder_calls'] = delta.get('training_encoder_calls', 0)
    budget['actual_new_training_decoder_calls'] = delta.get('training_decoder_calls', 0)
    row['optimizer_seconds'] = sum(x['update_seconds'] for x in read_json(folder/'TRAINING.json'))
    dump(output, row)
    verify_completed(row, identity, weight)
    return row


def cold_decode(frame):
    command = [sys.executable, '-c',
               'import sys;sys.path.insert(0,'+repr(str(ROOT/'scripts'))+');'
               'import atomic_composition_frame_v3 as f;'
               'sys.stdout.buffer.write(f.decode_frame(sys.stdin.buffer.read()))']
    return subprocess.run(command, input=frame, capture_output=True, check=True).stdout


def recipe_key(row):
    return fingerprint({'specs': row['specs'], 'terminals': row['terminals']})


def raw_recipe(row):
    """Remove every weight-specific field before final recipe de-duplication."""
    result = {k:copy.deepcopy(row[k]) for k in ('specs','terminals','bytes','raw_bytes','frame_sha256')}
    result['chain'] = rp.legacy_tail.chain_label(result['specs'],result['terminals'])
    return result


def benchmark_column(column, selections, baselines, reference, folder, identity, resume):
    result_path = folder/'RESULT.json'
    folder.mkdir(parents=True, exist_ok=True)
    frozen_path = folder/'FROZEN_SELECTION.json'
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        if not resume or frozen['experiment_identity'] != identity or frozen['selections'] != selections:
            raise AssertionError('frozen selected recipes drift')
    else:
        if result_path.exists(): raise AssertionError('benchmark result has no frozen selection receipt')
        dump(frozen_path, {'experiment_identity': identity, 'frozen_at': now(), 'selections': selections})
    if result_path.exists():
        old = read_json(result_path)
        if not resume or old['experiment_identity'] != identity: raise AssertionError('benchmark resume mismatch')
        for row in old['recipes'].values(): prior.verify_frame(row)
        return old
    raw = Path(column['input_path']).read_bytes()
    pool = {recipe_key(r): raw_recipe(r) for r in [*baselines, *[x['recipe'] for x in selections]]}
    encoded_times = {k: [] for k in pool}; decoded_times = {k: [] for k in pool}
    frames = {}; round_orders = []
    rng = random.Random(20260909+column['numeric_index'])
    start = time.perf_counter()
    for repeat in range(13):
        order = sorted(pool); rng.shuffle(order); round_orders.append(order)
        for key in order:
            row = pool[key]
            t = time.perf_counter_ns(); frame = rp.native_encode(raw, column['kind'], row['specs'], row['terminals']); enc = time.perf_counter_ns()-t
            t = time.perf_counter_ns(); decoded = rp.native_decode(frame); dec = time.perf_counter_ns()-t
            if decoded != raw or digest(frame) != row['frame_sha256'] or len(frame) != row['bytes']:
                raise AssertionError('final complete frame or lossless mismatch')
            if repeat == 0: frames[key] = frame
            elif frames[key] != frame: raise AssertionError('non-deterministic final encoding')
            if repeat >= 2: encoded_times[key].append(enc/1e9); decoded_times[key].append(dec/1e9)
    measured = {}
    for key, row in pool.items():
        frame = frames[key]
        cold_started = time.perf_counter()
        if cold_decode(frame) != raw: raise AssertionError('isolated cold decode mismatch')
        cold_seconds = time.perf_counter()-cold_started
        frame_path = folder/'frames'/(key+'.acf3'); frame_path.parent.mkdir(parents=True, exist_ok=True); frame_path.write_bytes(frame)
        measured[key] = {**row, 'frame_path': str(frame_path), 'encode_seconds_repeats': encoded_times[key],
                         'decode_seconds_repeats': decoded_times[key], 'encode_seconds_median': statistics.median(encoded_times[key]),
                         'decode_seconds_median': statistics.median(decoded_times[key]), 'warmups': 2, 'repeats': 11,
                         'lossless': True, 'deterministic': True, 'cold_decode_pass': True,
                         'complete_encoder_calls': 13, 'complete_decoder_calls': 13,
                         'measurement_cache_hit': False, 'final_measurement_shared_between_aliases': True,
                         'measurement_phase': 'independent_final_benchmark',
                         'input_sha256': column['input_sha256'], 'kind': column['kind'],
                         'cold_decode_validation_seconds_excluded_from_E_D': cold_seconds}
    rows = []
    for selected in selections:
        row = score(measured[recipe_key(selected['recipe'])], reference, selected['weight_tenths'])
        rows.append({**{k:v for k,v in selected.items() if k != 'recipe'}, **row})
    result = {'experiment_identity': identity, 'column': column, 'rows': rows, 'recipes': measured,
              'frozen_selection_path': str(frozen_path), 'selection_not_changed_by_final_timing': True,
              'protocol': '2 warmups + 11 randomized interleaved full-frame repeats; same-recipe aliases share the same measurement',
              'round_orders': round_orders, 'wall_seconds': time.perf_counter()-start,
              'complete_encoder_calls': len(pool)*13, 'complete_decoder_calls': len(pool)*13,
              'cold_decoder_process_calls': len(pool)}
    dump(result_path, result)
    return result


def build_selections(column, rows, shared, weights):
    selections = []
    historical = next(c for c in read_json(HISTORICAL_BENCHMARK)['columns'] if c['key'] == column['key'])
    for weight in weights:
        for method in ('ppo', 'grpo'):
            candidates = [r for r in rows if r['method']==method and r['weight_tenths']==list(weight)]
            winner = prior.select_across_seeds(candidates)
            selections.append(dict(method=method, weight_tenths=list(weight), weight_id=weight_id(weight),
                                   selection='best_of_requested_seeds_before_final_benchmark', source_seed=winner['seed'],
                                   before_benchmark_selection_cost=winner['selected']['cost'],
                                   source_selection_origin=winner['selected']['selection_origin'],
                                   seed_metrics=[dict(seed=r['seed'], cost=r['selected']['cost'],
                                       rho=r['selected']['compression_rate'], E=r['selected']['encode_seconds_median'],
                                       D=r['selected']['decode_seconds_median']) for r in candidates],
                                   recipe=winner['selected']))
        baseline = [score(r, shared['reference'], weight) for r in shared['baselines']]
        for name, subset, rank in [('fixed5_rho_best', baseline[:5], prior.cr_rank),
                                   ('fixed5_J_best', baseline[:5], prior.score_rank),
                                   ('complete30_rho_best', baseline[:30], prior.cr_rank),
                                   ('complete30_J_best', baseline[:30], prior.score_rank),
                                   ('all38_J_best', baseline, prior.score_rank)]:
            chosen = min(subset, key=rank)
            selections.append(dict(method=name, weight_tenths=list(weight), weight_id=weight_id(weight),
                                   selection='fixed_catalog_selection_not_RL_training',
                                   before_benchmark_selection_cost=chosen['cost'],recipe=chosen))
        for name in ('old_b4_beam', 'old_open_beam'):
            item = next(r for r in historical['methods'] if r['method']==name)
            # Historical recipe only; no historical timing is reused.
            recipe = {k: copy.deepcopy(item[k]) for k in ('specs','terminals','frame_sha256','bytes','raw_bytes')}
            selections.append(dict(method=name, weight_tenths=list(weight), weight_id=weight_id(weight),
                                   selection='historical_size_selected_Beam_recipe_remeasured_NOT_retrained_for_weight', recipe=recipe))
    return selections


def aggregate_results(benchmarks):
    grouped = {}
    for benchmark in benchmarks:
        col = benchmark['column']
        for row in benchmark['rows']:
            for ds in (col['dataset'], 'ALL'):
                grouped.setdefault((ds,row['weight_id'],row['method']), []).append(row)
    return [dict(dataset=ds,weight_id=w,method=m,columns=len(rows),weight_tenths=rows[0]['weight_tenths'],
                 raw_bytes=sum(r['raw_bytes'] for r in rows), compressed_bytes=sum(r['bytes'] for r in rows),
                 compression_rate=sum(r['bytes'] for r in rows)/sum(r['raw_bytes'] for r in rows),
                 encode_seconds_sum_of_column_medians=sum(r['encode_seconds_median'] for r in rows),
                 decode_seconds_sum_of_column_medians=sum(r['decode_seconds_median'] for r in rows),
                 mean_normalized_encode_time=statistics.mean(r['encode_time_normalized'] for r in rows),
                 mean_normalized_decode_time=statistics.mean(r['decode_time_normalized'] for r in rows),
                 mean_J=statistics.mean(r['cost'] for r in rows))
            for (ds,w,m),rows in sorted(grouped.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--weights', nargs='+', help='integer tenths, e.g. 4,3,3; default all 66')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--skip-benchmark', action='store_true')
    parser.add_argument('--free-space-floor-gib', type=float, default=15)
    args = parser.parse_args(argv)
    weights = [validate_weight(tuple(map(int,w.split(',')))) for w in args.weights] if args.weights else weight_grid()
    if len(weights)!=len(set(weights)) or len(args.seeds)!=len(set(args.seeds)) or not args.seeds or min(args.seeds)<0:
        parser.error('weights and seeds must be distinct; seeds nonnegative')
    if args.iterations<1 or args.episodes<8 or args.episodes%8 or args.free_space_floor_gib<0:
        parser.error('positive iterations, episodes multiple of 8, nonnegative disk floor required')
    columns, gate = prior.legacy_runner.input_columns(args.keys)
    config = {**BASE_CONFIG, 'iterations':args.iterations,'episodes':args.episodes,'seeds':args.seeds,
              'weights_tenths':[list(w) for w in weights], 'workers':1,
              'action_catalog':rp.legacy_tail.catalog(), 'tail_catalog':TAILS,
              'grammar':'17 families; 38 prefix choices; 38 tails; tail-first policy, prefix-then-tail execution',
              'not_30_family_graph_policy':True, 'not_exhaustive_composition_search':True,
              'normalization':'rho raw; E/(E+E0), D/(D+D0); one new frozen zlib6 reference per column',
              'cache':'new raw measurement vectors shared across weights/methods/seeds; no candidate exposure',
              'invalid_reward':-5.0,'invalid_caveat':'not a lower bound on valid rewards; count all valid rewards <= -5',
              'training_per_weight':'fresh actor, fresh PPO critic, fresh optimizer; no warm start',
              'J_cross_weight_warning':'J values under different weights are different objectives; do not rank weights by absolute J',
              'no_heldout_generalization_claim':True}
    if args.plan_only:
        print(json.dumps({'schema':SCHEMA,'columns':len(columns),'weights':len(weights),
                          'training_runs':len(columns)*len(weights)*2*len(args.seeds),
                          'trajectories':len(columns)*len(weights)*2*len(args.seeds)*args.iterations*args.episodes,
                          'actor_updates':len(columns)*len(weights)*2*len(args.seeds)*args.iterations*4,
                          'config':config},ensure_ascii=False,indent=2)); return 0
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output/'RUN.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_space(output,args.free_space_floor_gib)
    before = source_snapshot(columns,gate)
    identity = experiment_identity(config,columns,before)
    manifest_path = output/'MANIFEST.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not args.resume or manifest['experiment_identity']!=identity: raise AssertionError('resume source/input/config mismatch')
    else:
        unexpected=[p.name for p in output.iterdir() if p.name!='RUN.lock']
        if unexpected: raise RuntimeError('new experiment output must be empty: '+str(unexpected))
        dump(manifest_path,dict(schema=SCHEMA,experiment_identity=identity,config=config,columns=columns,
                                provenance_before=before,created_at=now(),command=sys.argv))
    oracle = MeasurementOracle(output/'measurements.sqlite3', identity, floor_gib=args.free_space_floor_gib)
    began=time.perf_counter(); status='RUNNING'; completed_paths=[]; all_benchmarks=[]; error=None
    completed_costs = Counter()
    try:
        shared=prepare_shared(columns,output,oracle,identity,args.resume)
        for column in columns:
            raw=Path(column['input_path']).read_bytes()
            if digest(raw)!=column['input_sha256']:raise AssertionError('input bytes drift')
            column_runs=[]
            for wi,weight in enumerate(weights):
                for si,seed in enumerate(args.seeds):
                    methods=('ppo','grpo') if (wi+si+column['numeric_index'])%2==0 else ('grpo','ppo')
                    init_hash=None
                    for method in methods:
                        runfolder=output/'runs'/column['key']/weight_id(weight)/f'{method}_{seed}'
                        runfolder.parent.mkdir(parents=True,exist_ok=True)
                        row=run_training(column,raw,shared[column['key']]['reference'],shared[column['key']]['baselines'],
                                         method,seed,weight,config,runfolder,oracle,identity,args.resume)
                        initial=row['budget']['initial_actor_sha256']
                        if init_hash is not None and initial!=init_hash:raise AssertionError('PPO/GRPO initial actor mismatch')
                        init_hash=initial;column_runs.append(row);completed_paths.append(str(runfolder/'RESULT.json'))
                        for name in ('train_seconds','search_seconds','optimizer_seconds','refinement_seconds','run_seconds'):
                            completed_costs[name] += row[name]
                        dump(output/'PROGRESS.json',dict(status='RUNNING',completed_runs=len(completed_paths),
                             expected_runs=len(columns)*len(weights)*len(args.seeds)*2,column=column['key'],
                             weight=weight_id(weight),method=method,seed=seed,oracle=oracle.totals(),
                             elapsed_seconds=time.perf_counter()-began,free_bytes=shutil.disk_usage(output).free))
            selections=build_selections(column,column_runs,shared[column['key']],weights)
            dump(output/'selections'/f'{column["key"]}.json',{'experiment_identity':identity,'selections':selections,'frozen_at':now()})
            if not args.skip_benchmark:
                all_benchmarks.append(benchmark_column(column,selections,shared[column['key']]['baselines'],
                    shared[column['key']]['reference'],output/'benchmark'/column['key'],identity,args.resume))
        after=source_snapshot(columns,gate)
        prior.legacy_runner.verify_unchanged(before,after)
        formal=(len(columns)==20 and weights==weight_grid() and tuple(args.seeds)==SEEDS
                and args.iterations==64 and args.episodes==32 and not args.skip_benchmark)
        status='COMPLETE' if formal else 'PARTIAL_PROTOCOL_COMPLETE'
        previous_invocations = []
        if (output/'INVOCATIONS.jsonl').exists():
            previous_invocations = [json.loads(line) for line in (output/'INVOCATIONS.jsonl').read_text().splitlines() if line]
        dump(output/'SUMMARY.json',dict(schema=SCHEMA,status=status,formal_66_weight_grid_complete=formal,
             experiment_identity=identity,config=config,run_result_paths=completed_paths,
             benchmark_result_paths=[str(output/'benchmark'/c['key']/'RESULT.json') for c in columns] if all_benchmarks else [],
             aggregates=aggregate_results(all_benchmarks),oracle=oracle.totals(),
             invocation_wall_seconds=time.perf_counter()-began,provenance_after=after,
             completed_run_cost_sums=dict(completed_costs),
             all_recorded_invocation_wall_seconds=sum(r['invocation_wall_seconds'] for r in previous_invocations)+time.perf_counter()-began,
             training_cost_note='per-run train/search/optimizer/retention times separate; shared oracle charged once globally; invocation receipts include resumed attempts',
             J_cross_weight_warning=config['J_cross_weight_warning']))
    except BaseException as exc:
        status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(output/'FAILURE.json',dict(status=status,error=error,at=now()))
        raise
    finally:
        receipt=dict(status=status,ended_at=now(),invocation_wall_seconds=time.perf_counter()-began,
                     completed_runs_this_scan=len(completed_paths),oracle=oracle.totals(),error=error)
        with (output/'INVOCATIONS.jsonl').open('a') as stream:stream.write(canonical(receipt)+'\n')
        oracle.close();fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
