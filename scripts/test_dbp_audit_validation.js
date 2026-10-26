#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const {
  auditValidationErrors,
  normalizeColumn,
  normalizeDatasetReceipt,
} = require('./dbp_measurement_validation.js');

const emptyComplete = {
  schema_version: 'dbp-native-float-audit-v1',
  status: 'complete',
  summary: {
    dataset_receipt_count: 0,
    measured_dataset_count: 0,
    n_a_dataset_count: 0,
    schema_aggregate_count: 0,
    nonempty_aggregate_count: 0,
    n_a_aggregate_count: 0,
  },
  measurement: {repeats: 1, warmup_blocks: 20, header_bytes_per_block: 10},
  environment: {shared_server: true, cpu: {passed_receipts_at_capture: 169}},
  runtimes: [{runtime: 'cpp', status: 'complete'}],
  datasets: [],
  columns: [],
  evidence: {},
};
const emptyErrors = auditValidationErrors(emptyComplete, [], []);
assert(emptyErrors.some(message => message.includes('169 unique datasets')));
assert(emptyErrors.some(message => message.includes('4,190 unique aggregates')));
assert(emptyErrors.some(message => message.includes('exactly C++, Java, and Python')));
assert(emptyErrors.some(message => message.includes('evidence.scope_sha256')));

const pendingComplete = {
  ...emptyComplete,
  summary: {
    dataset_receipt_count: 169,
    measured_dataset_count: 162,
    n_a_dataset_count: 7,
    schema_aggregate_count: 4190,
    nonempty_aggregate_count: 4096,
    n_a_aggregate_count: 94,
  },
  runtimes: [
    {runtime: 'cpp', status: 'complete', result_sha256: 'a'.repeat(64)},
    {runtime: 'java', status: 'complete', result_sha256: 'b'.repeat(64)},
    {runtime: 'python', status: 'complete', result_sha256: 'c'.repeat(64)},
  ],
};
const pendingReceipts = [normalizeDatasetReceipt({
  dataset: 'pending_dataset',
  runtime: 'cpp',
  status: 'pending',
  receipt_sha256: 'd'.repeat(64),
})];
const pendingColumns = [normalizeColumn({
  status: 'pending',
  runtime: 'cpp',
  dataset: 'pending_dataset',
  column: 'value',
  aggregate_id: 'pending_dataset|value|value',
  dtype: 'float32',
  width: 4,
  max_point: 18,
  rows: 1,
  valid_count: 1,
  missing_count: 0,
  native_bytes: 4,
  frame_bytes: 11,
  validity_bytes: 1,
  header_bytes: 10,
  blocks: 1,
  scaled_blocks: 1,
  raw_blocks: 0,
  verified_values: 1,
})];
const pendingErrors = auditValidationErrors(
  pendingComplete,
  pendingColumns,
  pendingReceipts,
);
assert(pendingErrors.some(message => message.includes('nonfinal status')));
assert(pendingErrors.some(message => message.includes('status does not match valid_count')));
assert(pendingErrors.some(message => message.includes('result rows are not exactly')));

console.log(JSON.stringify({status: 'passed', cases: ['empty-complete', 'pending-complete']}));
