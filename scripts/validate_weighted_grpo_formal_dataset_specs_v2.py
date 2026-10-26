#!/usr/bin/env python3
"""Versioned data admission: preserve v1 and independently verify smallest32.

A successful receipt admits data only. It never authorizes native timing,
model updates, GPU training, or test-derived training statistics.
"""
from __future__ import annotations

import argparse
import calendar
import collections
import datetime
import hashlib
import json
from pathlib import Path
import struct
import sys

VERSION = 'weighted-grpo-smallest32-cohort-20260918-v1'
LEGACY_VERSION = 'weighted-grpo-timeseries-cohort-v1'
FROZEN_FILES = {
    'cohort_smallest32_v1.json': '3e48506f9a189dec63a4f3939018d5325cf5bcc6cb2aa1c694c55516c6c74469',
    'timestamp_contracts.json': '62ddc2fefe88b8b824b6f5e72520fffa23a242b0c067740944a1537771e28609',
    'source_content_manifest.json': '7273f6bfffbb2858a6e6fce7ebc251926c9859c289d8febcfe80cabeff092293',
}
PRODUCER_SHA256 = '621eeca531fba90e159d85507c68ed33b4f6231c43e0f17b3334c6fef7e3d43a'
DTYPES = {'INT32': ('int32', 4), 'INT64': ('int64', 8), 'FLOAT': ('float32', 4), 'DOUBLE': ('float64', 8)}


class SpecGateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SpecGateError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def safe_child(root, name):
    candidate = (root / name).resolve()
    require(candidate.is_relative_to(root.resolve()), f'evidence path leaves contract root: {name}')
    return candidate


def load_contracts(cohort_path):
    root = cohort_path.parent
    require(sha256(cohort_path) == FROZEN_FILES['cohort_smallest32_v1.json'], 'frozen cohort substituted')
    for name, digest in FROZEN_FILES.items():
        if name != 'cohort_smallest32_v1.json':
            require(sha256(root / name) == digest, f'frozen contract substituted: {name}')
    cohort = load(cohort_path)
    pins = load(root / 'source_content_manifest.json')['datasets']
    times = load(root / 'timestamp_contracts.json')['datasets']
    ids = [e['id'] for e in cohort['datasets']]
    require(len(ids) == len(set(ids)) == 32, 'cohort requires exactly 32 unique sources')
    require(set(ids) == set(pins) == set(times), 'source/time contracts have missing or extra sources')
    for n, contract in times.items():
        for ref in contract['evidence']:
            require(sha256(safe_child(root, ref['path'])) == ref['sha256'], f'{n}: timestamp evidence changed')
        exact = contract.get('exact_source_comparison')
        if exact and exact.get('source_file_sha256'):
            require(sha256(safe_child(root, exact['source_file'])) == exact['source_file_sha256'], f'{n}: upstream comparison source changed')
    require(sha256(root / 'smallest32_data_adapter_materialization_v1.py') == PRODUCER_SHA256, 'historical producer code is missing or changed')
    return cohort, pins, times


def normalized_prepared_timestamp(stamp):
    """Full prepared contract, including defaults omitted by epoch source specs."""
    semantic_kind = stamp.get('semantic_kind')
    source_kinds = {
        None: 'epoch',
        'calendar_date_utc_midnight_encoding': 'calendar_date_coordinate',
        'helsinki_civil_time_coordinate_encoding': 'civil_time_coordinate',
        'source_naive_civil_time_coordinate': 'civil_time_coordinate',
    }
    require(semantic_kind in source_kinds, 'unsupported frozen timestamp semantic kind')
    return {
        **stamp,
        'source_name': 'time',
        'source_kind': source_kinds[semantic_kind],
        'semantic_kind': semantic_kind,
        'formats': ['integer_epoch_from_tsfile' if semantic_kind is None else 'integer_coordinate_from_tsfile'],
        'ambiguous_fold': None,
        'invalid_policy': 'fail_no_unit_or_timezone_inference',
    }


def validate_record(entry, pin, time_contract, spec, manifest, schema, cohort):
    """Bind prepared provenance to current frozen spec/time/pins, not self-report."""
    n = entry['id']
    role = entry['role']
    for obj in (spec, manifest):
        require(obj.get('dataset_id') == n, f'{n}: dataset substitution')
        require(obj.get('domain') == entry['domain'] and obj.get('source_group') == entry['source_group'], f'{n}: source identity substitution')
    require(manifest['cohort'] == {**{k: cohort[k] for k in ('version', 'assignment_sha256', 'policy_sha256')}, 'role': role}, f'{n}: cohort or role substitution')
    allowed = {'train', 'validation'} if role == 'train_validation' else {'test'}
    require({b['split'] for b in manifest['blocks']} == allowed, f'{n}: role/split leakage')
    require(manifest['split_contract']['cohort_role'] == role, f'{n}: split role substitution')
    stamp = dict(time_contract['timestamp'])
    if n == 'github_top_developers':
        stamp['duplicate_policy'] = 'allow_preserve_order'
    require(spec['timestamp'] == stamp, f'{n}: timestamp spec unit/semantics/evidence mismatch')
    require(manifest.get('timestamp_evidence') == time_contract, f'{n}: stale or altered timestamp evidence')
    require(manifest['input_contract']['timestamp'] == normalized_prepared_timestamp(stamp), f'{n}: prepared timestamp unit/semantics/defaults mismatch')
    require(manifest.get('source_content_pin') == pin, f'{n}: source pin substitution')
    require(manifest.get('preparation_spec_sha256') == hashlib.sha256(canonical(spec)).hexdigest(), f'{n}: stale preparation spec SHA256')
    source = spec['source']
    require(source.get('expected_sha256') == pin['sha256'] and source.get('upstream_revision') == pin['revision'] == entry['revision'], f'{n}: spec source hash/revision mismatch')
    require(Path(source['path']).name == pin['repo_path'], f'{n}: source filename mismatch')
    require(source.get('upstream_locator') == entry['locator'] and source.get('format') == 'tsfile', f'{n}: source locator/format mismatch')
    observed = manifest['source']
    require(observed['path'] == source['path'] and observed['sha256'] == pin['sha256'] and observed['bytes'] == pin['size'], f'{n}: prepared source content mismatch')
    require(observed['upstream_revision'] == pin['revision'] and observed['upstream_locator'] == entry['locator'], f'{n}: prepared source revision mismatch')
    require(spec['dataset_byte_cap'] == 268435456 and spec['block_byte_cap'] == 1048576, f'{n}: byte caps changed')
    require(spec['model'] == 'table', f'{n}: non-table spec')
    tags = [c['name'] for c in schema['columns'] if c['category'] == 'TAG']
    fields = [c for c in schema['columns'] if c['category'] == 'FIELD' and c['tsfile_type'] in DTYPES]
    require(fields, f'{n}: no numeric FIELD columns')
    columns = [{'source_name': c['name'], 'output_name': c['name'], 'tsfile_type': c['tsfile_type']} for c in fields]
    require(spec['table'] == {'table_name': schema['table_name'], 'entity_columns': tags, 'value_columns': columns}, f'{n}: missing/altered source numeric columns or tags')
    expected_columns = [{'source_name': 'time', 'output_name': stamp['output_name'], 'dtype': 'int64', 'width_bytes': 8, 'role': 'timestamp'}]
    expected_columns += [{'source_name': c['name'], 'output_name': c['name'], 'dtype': DTYPES[c['tsfile_type']][0], 'width_bytes': DTYPES[c['tsfile_type']][1], 'role': 'value'} for c in fields]
    require(manifest['input_contract']['numeric_columns'] == expected_columns and manifest['input_contract']['entity_columns'] == tags, f'{n}: omitted or widened prepared numeric columns')
    admission = manifest['data_admission']
    require(admission.get('cohort_adapter_sha256') == PRODUCER_SHA256, f'{n}: unrecognized producer; issue a new explicit version')
    require(all(admission.get(k) is v for k, v in [('no_timezone_inference', True), ('no_row_index_timestamp_fallback', True), ('test_used_for_training_statistics', False)]), f'{n}: admission safeguards changed')
    return fields, tags


def expected_coverage(entry, manifest, schema, fields, tags):
    n, role = entry['id'], entry['role']
    rows = []
    for col in [{'name': 'time', 'tsfile_type': 'INT64', 'category': 'TIME'}, *schema['columns']]:
        numeric = col['category'] == 'TIME' or col in fields
        rows.append({'dataset_id': n, 'role': role, 'table_name': schema['table_name'], 'column': col['name'], 'type': col['tsfile_type'], 'category': col['category'], 'status': 'prepared_numeric' if numeric else 'non_numeric', 'entity_columns': tags, 'output_column': manifest['input_contract']['timestamp']['output_name'] if col['category'] == 'TIME' else col['name'], 'reason': None if numeric else 'TAG identity or STRING value is retained as entity/context metadata, outside numeric compression scope'})
    entities = {e['entity_id']: e for e in manifest['entities']}
    for b in manifest['blocks']:
        rows.append({'dataset_id': n, 'role': role, 'entity_id': b['entity_id'], 'entity': entities[b['entity_id']], 'column': b['column'], 'source_column': b['source_column'], 'dtype': b['dtype'], 'split': b['split'], 'block_id': b['block_id'], 'rows': b['row_count'], 'B_raw': b['original_numeric_bytes'], 'status': 'prepared_block'})
    return rows


def semantic_grid(manifest, directory):
    time = manifest['input_contract']['timestamp']
    n = manifest['dataset_id']
    entities = {e['entity_id']: e for e in manifest['entities']}
    values = collections.defaultdict(list)
    for b in manifest['blocks']:
        if b['column'] == time['output_name']:
            values[b['entity_id']].extend(x[0] for x in struct.iter_unpack('<q', safe_child(directory, b['values_path']).read_bytes()))
    for eid, v in values.items():
        v.sort()
        if time['semantic_kind'] == 'calendar_date_utc_midnight_encoding':
            require(time['source_unit'] == 'ms' and all(t % 86400000 == 0 for t in v), f'{n}/{eid}: invalid calendar-date coordinate')
        if n in {'solar_weekly', 'traffic_weekly', 'saugeenday-monash', 'sunspot', 'tourism_yearly', 'us_births-monash', 'vehicle_trips'}:
            tags = {k: json.loads(x.split(':', 1)[1]) for k, x in zip(manifest['input_contract']['entity_columns'], entities[eid]['entity_values'])}
            start = datetime.datetime.fromisoformat(tags['start_timestamp'])
            base = calendar.timegm(start.timetuple()) * 1000
            if n in {'solar_weekly', 'traffic_weekly'}:
                require(v[0] == base + ((6-start.weekday()) % 7)*86400000 and all(b-a == 7*86400000 for a,b in zip(v,v[1:])), f'{n}/{eid}: source weekly grid mismatch')
            elif n == 'tourism_yearly':
                require(v == [calendar.timegm(start.replace(year=start.year+i).timetuple())*1000 for i in range(len(v))], f'{n}/{eid}: source yearly grid mismatch')
            else:
                require(v[0] == base and all(b-a == 86400000 for a,b in zip(v,v[1:])), f'{n}/{eid}: source daily grid mismatch')
    zero = [b['block_id'] for b in manifest['blocks'] if b['original_numeric_bytes'] == 0]
    return {'dataset_id': n, 'entities': len(values), 'time_semantic_kind': time['semantic_kind'], 'unit': time['source_unit'], 'all_calendar_values_at_midnight': True if time['semantic_kind'] == 'calendar_date_utc_midnight_encoding' else None, 'zero_B_raw_block_count': len(zero), 'zero_B_raw_blocks': zero, 'maximum_prepared_block_bytes': max(b['prepared_block_bytes'] for b in manifest['blocks']), 'original_numeric_bytes': manifest['totals']['original_numeric_bytes'], 'all_source_numeric_columns_and_time_listed': True}


def validate_smallest32(cohort_path, materialized):
    import prepare_weighted_grpo_timeseries_v1 as preparation
    import prepare_weighted_grpo_tsfile_v1 as tsfile
    cohort, pins, times = load_contracts(cohort_path)
    require(cohort['version'] == VERSION, 'unsupported smallest32 version')
    gate = load(materialized / 'data_gate.json')
    require(gate.get('passed') is True and gate.get('failures') == [], 'materialization gate did not pass')
    records = gate['records']
    require(len(records) == 32 and len({r['dataset_id'] for r in records}) == 32, 'missing/duplicate prepared sources')
    by_id = {r['dataset_id']: r for r in records}
    require(set(by_id) == {e['id'] for e in cohort['datasets']}, 'missing/extra materialized sources')
    reports, coverage, grids = [], [], []
    split_totals = collections.defaultdict(collections.Counter)
    groups, hashes, paths = {}, {}, collections.defaultdict(list)
    for entry in cohort['datasets']:
        n, role = entry['id'], entry['role']
        require(groups.setdefault(entry['source_group'], role) == role, f'{n}: source group leaks across roles')
        require(hashes.setdefault(pins[n]['sha256'], role) == role, f'{n}: identical source bytes cross roles')
        record = by_id[n]
        require(record['role'] == role and record['status'] == 'prepared_verified', f'{n}: materialized role/status mismatch')
        directory = materialized / ('prepared_train' if role == 'train_validation' else 'prepared_test') / n
        manifest_path = directory / 'dataset_manifest.json'
        require(Path(record['manifest']).resolve() == manifest_path.resolve() and record['manifest_sha256'] == sha256(manifest_path), f'{n}: stale materialization receipt')
        manifest = load(manifest_path)
        spec_path = materialized / 'specs' / (n + '.json')
        spec = load(spec_path)
        pin = pins[n]
        source_path = Path(spec['source']['path'])
        require(source_path.stat().st_size == pin['size'] and sha256(source_path) == pin['sha256'], f'{n}: original source SHA256/size mismatch')
        if pin.get('lfs_sha256'):
            require(pin['lfs_sha256'] == pin['sha256'], f'{n}: LFS pin mismatch')
        else:
            payload = source_path.read_bytes()
            require(hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest() == pin['blob_id'], f'{n}: Git blob pin mismatch')
        observed = tsfile.inspect_tsfile_source(source_path)
        require(observed['tsfile_python_version'] == '2.1.4' and len(observed['table_schemas']) == 1, f'{n}: reader version/schema mismatch')
        schema = observed['table_schemas'][0]
        fields, tags = validate_record(entry, pin, times[n], spec, manifest, schema, cohort)
        verification = preparation.verify_prepared_dataset(directory, verify_source=True, expected_role=role)
        coverage.extend(expected_coverage(entry, manifest, schema, fields, tags))
        grids.append(semantic_grid(manifest, directory))
        for b in manifest['blocks']:
            split_totals[b['split']]['B_raw'] += b['original_numeric_bytes']
            split_totals[b['split']]['blocks'] += 1
            split_totals[b['split']]['zero_B_raw_blocks'] += b['original_numeric_bytes'] == 0
        paths[role].append(str(manifest_path))
        reports.append({'dataset_id': n, 'role': role, 'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path), 'spec_sha256': sha256(spec_path), 'provenance_sha256': manifest['provenance_sha256'], 'source_sha256': pin['sha256'], 'verification': verification})
    require(load(materialized / 'column_coverage.json') == coverage, 'column/entity/block coverage is missing, stale, or substituted')
    expected_grid = {'version': 'smallest32-semantic-grid-audit-v1', 'passed': True, 'errors': [], 'datasets': grids, 'source_rows_and_columns_are_not_training_statistics': True}
    require(load(materialized / 'semantic_grid_audit.json') == expected_grid, 'semantic grid audit is stale or substituted')
    for role, filename in [('train_validation','train_manifest_paths.json'), ('test_candidate','test_manifest_paths.json')]:
        require(load(materialized / filename) == paths[role], f'{role}: manifest list has missing/extra/substituted sources')
    return {'schema': 'weighted-grpo-formal-dataset-spec-gate-result-v2', 'cohort_version': VERSION, 'scope': 'data_admission_only', 'passed': True, 'native_full_graph_timing_admitted': False, 'formal_training_admitted': False, 'test_used_for_training_statistics': False, 'frozen_contract_sha256': FROZEN_FILES, 'historical_producer_sha256': PRODUCER_SHA256, 'validator_sha256': sha256(Path(__file__)), 'materialized_evidence_sha256': {n: sha256(materialized/n) for n in ['data_gate.json','column_coverage.json','semantic_grid_audit.json','train_manifest_paths.json','test_manifest_paths.json']}, 'prepared_count': len(reports), 'role_counts': dict(collections.Counter(r['role'] for r in reports)), 'split_totals': dict(split_totals), 'prepared': reports}


def validate_bundle(*, cohort_path, materialized=None, audit_path=None, revisions_path=None, prepared_manifests=()):
    version = load(cohort_path).get('version')
    if version == LEGACY_VERSION:
        import validate_weighted_grpo_formal_dataset_specs_v1 as legacy
        result = legacy.validate_bundle(audit_path or legacy.DEFAULT_AUDIT, cohort_path, revisions_path or legacy.DEFAULT_REVISIONS)
        if prepared_manifests:
            result['prepared_admission'] = legacy.validate_prepared_admissions([legacy._load_prepared_manifest(p) for p in prepared_manifests], audit_path=audit_path or legacy.DEFAULT_AUDIT)
        # Return the old receipt unchanged; no schema upgrade or relabeling.
        return result
    require(version == VERSION, f'unknown cohort version: {version}')
    require(materialized is not None, 'smallest32 requires --materialized')
    require(not prepared_manifests and audit_path is None and revisions_path is None, 'smallest32 uses its complete frozen contract bundle; legacy overrides are not accepted')
    return validate_smallest32(cohort_path.resolve(), materialized.resolve())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, required=True)
    ap.add_argument('--materialized', type=Path)
    ap.add_argument('--audit', type=Path)
    ap.add_argument('--revisions', type=Path)
    ap.add_argument('--prepared-manifest', type=Path, action='append', default=[])
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    try:
        receipt = validate_bundle(cohort_path=args.cohort, materialized=args.materialized, audit_path=args.audit, revisions_path=args.revisions, prepared_manifests=args.prepared_manifest)
        if args.output:
            require(not args.output.exists(), 'receipt output already exists; old receipts cannot be overwritten')
            args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'FORMAL_DATASET_SPEC_GATE_V2_BLOCKER: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(json.dumps({k:v for k,v in receipt.items() if k != 'prepared'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
