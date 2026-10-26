"""Build a compact UI payload from the independently hashed full result snapshot."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def build() -> None:
    source = ROOT / 'data.json'
    data = json.loads(source.read_text())
    rows = data['dataset_results']
    fields = {'datasets': 'dataset', 'scopes': 'scope', 'runtimes': 'runtime',
              'comparison_profiles': 'comparison_profile', 'families': 'family',
              'configurations': 'algorithm_variant'}
    tables = {key: sorted({r[field] for r in rows}) for key, field in fields.items()}
    tables['timing_aggregations'] = sorted({r['timing_aggregation'] for r in rows})
    tables['support_sets'] = sorted({r.get('eligible_column_set_sha256', 'legacy_snapshot_unbound') for r in rows})
    tables['measurement_statuses'] = sorted({r.get('measurement_status', 'legacy_published_measurement') for r in rows})
    tables['timing_statuses'] = sorted({r.get('timing_profile_status', 'source_snapshot_timing') for r in rows})
    tables['binary_hashes'] = sorted({r.get('binary_sha256') or 'unbound' for r in rows})
    indices = {key: {v:i for i,v in enumerate(values)} for key,values in tables.items()}
    compact = []
    for r in rows:
        compact.append([*(indices[key][r[field]] for key,field in fields.items()),
                        r['compression_rate'], r['compression_time_ns_per_point'],
                        r['decompression_time_ns_per_point'], r['original_size_bytes'],
                        r['compressed_size_bytes'], r.get('timing_column_count', 0),
                        indices['timing_aggregations'][r['timing_aggregation']],
                        indices['support_sets'][r.get('eligible_column_set_sha256', 'legacy_snapshot_unbound')],
                        indices['measurement_statuses'][r.get('measurement_status', 'legacy_published_measurement')],
                        indices['timing_statuses'][r.get('timing_profile_status', 'source_snapshot_timing')],
                        indices['binary_hashes'][r.get('binary_sha256') or 'unbound']])
    payload = {key: data[key] for key in ('schema_version','benchmark','coverage','datasets','methods','scopes','source')}
    if 'dataset_summary' in data:
        payload['dataset_summary'] = data['dataset_summary']
    payload['exported_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload['source_data_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload['record_tables'] = tables
    payload['compact_records'] = compact
    output = ROOT / 'ui-data.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps({'file':output.name,'records':len(rows),'bytes':output.stat().st_size,
                      'source_data_sha256':payload['source_data_sha256'],
                      'ui_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    build()
