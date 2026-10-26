#!/usr/bin/env python3
"""Bind the Forbush native-read repair to a complete independent CSV comparison."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ID = 'forbush_decreases'
SOURCE = 'forbush_decreases.tsfile'
SOURCE_SHA = '35f8df03af8e5fb1c46a6a69f39baad746cfb9575b613a2c91497a8e47b14f46'
SCHEMA_SHA = '8c32538e20c5df65737767e6c8695fbd9a3c880f092c792c77c2752ee198d6f1'
FILES = ['data.csv', 'series_map.csv', 'columns.csv', 'dataset_manifest.json']
SOURCE_ROWS = 8_726
TEMPORAL_EMPTY_REPRESENTATION = (
    'csv_empty_source_non_null_empty_distinct_from_null'
)
TEMPORAL_EMPTY_EXPECTED = {
    'cme_datetime_utc': {
        'output_column': 'c0021',
        'source_non_null_empty_count': 1_629,
        'source_null_count': 6_441,
        'source_valid_count': 656,
    },
    'solar_source_datetime_utc': {
        'output_column': 'c0022',
        'source_non_null_empty_count': 1_284,
        'source_null_count': 6_441,
        'source_valid_count': 1_001,
    },
}
TEMPORAL_EMPTY_FIELD_KEYS = {
    'source_name', 'source_type', 'source_category', 'output_column',
    'conversion', 'representation', 'source_non_null_empty_count',
    'source_non_null_empty_row_ordinals', 'source_null_count',
    'source_null_row_ordinals', 'source_valid_count',
}
TEMPORAL_EMPTY_WRAPPER_KEYS = {
    'source_path', 'source_sha256', 'table_name', 'source_schema_sha256',
    'source_rows', 'fields',
}

def require(ok, message):
    if not ok:
        raise ValueError('Forbush native gate failed: ' + message)

def digest(path):
    require(path.is_file() and not path.is_symlink(), f'not a regular file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    digest(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(obj, dict), 'not an object')
    return obj

def _require_ordinal_list(value, count, label):
    require(isinstance(value, list), f'{label} ordinals are not a list')
    require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f'{label} ordinals are not integers',
    )
    require(len(value) == count, f'{label} ordinal count differs')
    require(value == sorted(set(value)), f'{label} ordinals are not sorted unique')
    require(
        all(0 <= item < SOURCE_ROWS for item in value),
        f'{label} ordinal outside source range',
    )

def _check_temporal_empty_fields(fields):
    require(isinstance(fields, list) and len(fields) == 2,
            'temporal empty-string field audits missing')
    require(all(isinstance(item, dict) for item in fields),
            'temporal empty-string field audits are not objects')
    require(
        [item.get('source_name') for item in fields]
        == sorted(TEMPORAL_EMPTY_EXPECTED),
        'temporal empty-string fields are not the exact sorted set',
    )
    output_columns = []
    for item in fields:
        require(isinstance(item, dict) and set(item) == TEMPORAL_EMPTY_FIELD_KEYS,
                'temporal empty-string field audit keys differ')
        source_name = item['source_name']
        expected = TEMPORAL_EMPTY_EXPECTED[source_name]
        output_column = item['output_column']
        require(
            item['source_type'] == 'STRING'
            and item['source_category'] == 'FIELD'
            and isinstance(output_column, str)
            and len(output_column) == 5
            and output_column.startswith('c')
            and output_column[1:].isdigit()
            and output_column == expected['output_column']
            and item['conversion'] == 'policy_temporal_string_to_int64_us'
            and item['representation'] == TEMPORAL_EMPTY_REPRESENTATION,
            f'temporal empty-string metadata differs for {source_name}',
        )
        for count_name in (
            'source_non_null_empty_count', 'source_null_count',
            'source_valid_count',
        ):
            expected_count = expected[count_name]
            require(item[count_name] == expected_count,
                    f'{count_name} differs for {source_name}')
        empty = item['source_non_null_empty_row_ordinals']
        nulls = item['source_null_row_ordinals']
        _require_ordinal_list(
            empty, expected['source_non_null_empty_count'],
            f'{source_name} non-NULL empty-string',
        )
        _require_ordinal_list(
            nulls, expected['source_null_count'], f'{source_name} true-NULL',
        )
        require(set(empty).isdisjoint(nulls),
                f'{source_name} empty and true-NULL positions overlap')
        require(
            SOURCE_ROWS - len(empty) - len(nulls)
            == expected['source_valid_count'],
            f'{source_name} valid-position complement differs',
        )
        output_columns.append(output_column)
    require(len(set(output_columns)) == 2,
            'temporal empty-string fields share an output column')

def _check_temporal_empty_wrapper(wrapper):
    require(isinstance(wrapper, dict) and set(wrapper) == TEMPORAL_EMPTY_WRAPPER_KEYS,
            'temporal empty-string source audit keys differ')
    require(
        wrapper['source_path'] == SOURCE
        and wrapper['source_sha256'] == SOURCE_SHA
        and wrapper['table_name'] == ID
        and wrapper['source_schema_sha256'] == SCHEMA_SHA
        and wrapper['source_rows'] == SOURCE_ROWS,
        'temporal empty-string source binding differs',
    )
    _check_temporal_empty_fields(wrapper['fields'])
    return wrapper['fields']

def check_native(report, output, exporter_sha, validator_sha, scope_sha, policy_sha):
    require(report.get('status') == 'passed' and report.get('dataset_count') == 1,
            'independent validation missing')
    require([r.get('dataset_id') for r in report.get('datasets', [])] == [ID],
            'independent dataset set differs')
    scans = report.get('pilot_source_rescans', [])
    require(len(scans) == 1 and scans[0].get('dataset_id') == ID, 'full-value scan missing')
    scan = scans[0]
    require(scan.get('status') == 'passed' and scan.get('source_files') == 1
            and scan.get('source_rows') == 8726 and scan.get('compared_output_rows') == 8726
            and scan.get('all_null_cells') == 78024
            and scan.get('candidate_value_column_count') == 23
            and scan.get('no_rows_or_value_columns_dropped') is True,
            'full rows, columns or NULL coverage differs')
    numeric = scan.get('compared_numeric_fields', [])
    require(len(numeric) == 21 and len(set(numeric)) == 21,
            'numeric full-value comparison missing')
    require(scan.get('compared_temporal_fields') ==
            ['cme_datetime_utc', 'solar_source_datetime_utc'], 'temporal value comparison missing')
    empty_audits = scan.get('temporal_empty_string_audits')
    require(isinstance(empty_audits, list) and len(empty_audits) == 1,
            'independent temporal empty-string source audit missing')
    independent_empty_fields = _check_temporal_empty_wrapper(empty_audits[0])
    require(scan.get('native_query_audits') == [{
        'source_path': SOURCE, 'source_sha256': SOURCE_SHA, 'table_name': ID,
        'source_sha256_before': SOURCE_SHA, 'source_sha256_after': SOURCE_SHA,
        'source_schema_sha256': SCHEMA_SHA,
        'query_mode': 'rowwise_direct_source_bound', 'dataframe_attempted': False,
        'rowwise_fallback': False,
    }], 'independent native query binding differs')
    schema = report.get('schema_metadata_pass', [])
    require(len(schema) == 1 and schema[0].get('dataset_id') == ID
            and schema[0].get('status') == 'passed'
            and schema[0].get('candidate_value_column_count') == 23
            and schema[0].get('retained_value_column_count') == 23
            and schema[0].get('dropped_value_column_count') == 0,
            'schema did not preserve 23 columns')
    require(output.is_dir() and not output.is_symlink(), 'output not a real directory')
    path = output / ID
    require(path.is_dir() and not path.is_symlink()
            and sorted(p.name for p in path.iterdir()) == sorted(FILES), 'output file set differs')
    receipt = load(path / 'dataset_manifest.json')
    require(receipt.get('status') == 'complete' and receipt.get('dataset_id') == ID
            and receipt.get('exporter_sha256') == exporter_sha
            and receipt.get('scope_sha256') == scope_sha
            and receipt.get('timestamp_policy_file_sha256') == policy_sha,
            'receipt payload/policy binding differs')
    require(int(receipt['emitted_rows_decimal_20']) == SOURCE_ROWS
            and int(receipt['lost_rows_due_to_byte_cap_decimal_20']) == 0
            and int(receipt['candidate_value_column_count_decimal_20']) == 23
            and int(receipt['retained_value_column_count_decimal_20']) == 23
            and int(receipt['dropped_trailing_value_column_count_decimal_20']) == 0
            and receipt.get('row_exception_filtered_rows') == 0, 'export dropped rows/columns')
    ss = receipt.get('source_tsfiles', [])
    require(len(ss) == 1 and ss[0].get('path') == SOURCE and ss[0].get('sha256') == SOURCE_SHA,
            'export source binding differs')
    reads = receipt.get('source_scans', [])
    require(len(reads) == 1 and reads[0].get('source_path') == SOURCE
            and reads[0].get('source_sha256') == SOURCE_SHA
            and reads[0].get('source_schema_sha256') == SCHEMA_SHA
            and reads[0].get('table_name') == ID
            and reads[0].get('source_rows') == SOURCE_ROWS
            and reads[0].get('read_mode') == 'rowwise_direct_source_bound'
            and reads[0].get('dataframe_attempted') is False
            and reads[0].get('rowwise_fallback') is False, 'export native read mode not proven')
    producer_empty_fields = reads[0].get('temporal_empty_string_audits')
    _check_temporal_empty_fields(producer_empty_fields)
    require(producer_empty_fields == independent_empty_fields,
            'producer and independent empty/NULL row positions differ')
    files = receipt.get('files', [])
    require([r.get('path') for r in files] == FILES, 'receipt file set differs')
    hashes = {}
    total = 0
    for f in files:
        actual = digest(path / f['path'])
        size = (path / f['path']).stat().st_size
        require(size == int(f['bytes_decimal_20']), 'actual size mismatch')
        require(f['path'] == 'dataset_manifest.json' or actual == f['sha256'], 'actual SHA mismatch')
        hashes[f['path']] = actual
        total += size
    require(total <= 1_000_000_000, 'decimal 1GB exceeded')
    return {'source_rows': SOURCE_ROWS, 'output_rows': SOURCE_ROWS,
            'all_null_cells': 78024,
            'value_columns': 23, 'source_sha256': SOURCE_SHA, 'source_schema_sha256': SCHEMA_SHA,
            'actual_capped_bytes': total, 'output_hashes': hashes,
            'temporal_empty_string_audits': empty_audits,
            'full_value_scan': scan}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('validation-report', 'csv-output', 'scope', 'timestamp-policies', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--expected-exporter-sha256', required=True)
    p.add_argument('--expected-validator-sha256', required=True)
    a = p.parse_args()
    scope_sha, policy_sha = digest(a.scope), digest(a.timestamp_policies)
    evidence = check_native(load(a.validation_report), a.csv_output,
                            a.expected_exporter_sha256, a.expected_validator_sha256,
                            scope_sha, policy_sha)
    gate = {'schema': 'project179-thulab-csv-forbush-native-gate-v1', 'status': 'complete',
            'gate': 'forbush_direct_rowwise_full_values_verified',
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'scope_sha256': scope_sha, 'timestamp_policy_file_sha256': policy_sha,
            'exporter_sha256': a.expected_exporter_sha256,
            'validator_sha256': a.expected_validator_sha256,
            'validation_report_sha256': digest(a.validation_report), 'evidence': evidence}
    require(not a.output.exists() and not a.output.is_symlink(), 'gate already exists')
    fd, name = tempfile.mkstemp(prefix='.forbush-native-gate-', dir=a.output.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(json.dumps(gate, sort_keys=True, separators=(',', ':')) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, a.output)
        dfd = os.open(a.output.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'status': 'complete', 'gate': str(a.output)}))

if __name__ == '__main__':
    main()
