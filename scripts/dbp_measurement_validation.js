"use strict";
// Frozen measurement validation retained independently of the retired website.
const finiteNumber = (value) => value !== null && value !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
const integer = (value) => finiteNumber(value) === null ? null : Math.trunc(Number(value));
const runtimeLabel = (runtime) => ({cpp:'C++',java:'Java',python:'Python'}[runtime] || String(runtime || 'Unknown'));
const successfulStatus = (value) => ['ok','complete','passed'].includes(normalizeStatus(value));
const EXPECTED_RUNTIMES = ['cpp','java','python'];
const EXPECTED_COMPLETE_SUMMARY = {dataset_receipt_count:169,measured_dataset_count:162,n_a_dataset_count:7,schema_aggregate_count:4190,nonempty_aggregate_count:4096,n_a_aggregate_count:94};
const REQUIRED_EVIDENCE_SHA_FIELDS = ['scope_sha256','source_column_inventory_sha256','runtime_artifact_manifest_sha256','campaign_binding_sha256','frame_verification_receipt_sha256','compact_collection_receipt_sha256','input_scope_receipt_sha256','campaign_run_receipt_sha256','campaign_progress_sha256','environment_receipt_sha256','source_evidence_chain_sha256'];
const REQUIRED_COLUMN_SHA_FIELDS = ['frame_sequence_sha256','expected_sequence_chain_sha256','result_sha256','receipt_sha256','source_manifest_receipt_sha256','source_aggregates_sha256','source_blocks_sha256','source_block_catalog_sha256','runtime_artifact_manifest_sha256','source_column_inventory_sha256','scope_sha256'];
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
function normalizeStatus(value) {
  const text=String(value??'').trim().toLowerCase().replace(/[\s/-]+/g,'_');
  return text||'unknown';
}
function normalizeDtype(value,width) {
  const text=String(value??'').toLowerCase();
  if(text.includes('32')||Number(width)===4)return 'float32';
  if(text.includes('64')||Number(width)===8)return 'float64';
  return text||'unknown';
}
function normalizeColumn(row) {
  const valid=integer(row.valid_count??row.verified_values),nativeBytes=integer(row.native_bytes??row.original_bytes),frameBytes=integer(row.frame_bytes??row.compressed_bytes??row.codec_wire_bytes);
  const encodeTotal=finiteNumber(row.encode_ns),decodeTotal=finiteNumber(row.decode_ns);
  return {...row,
    status:normalizeStatus(row.status),runtime:String(row.runtime??row.implementation_language??''),dataset:String(row.dataset??row.dataset_id??''),column:String(row.column??row.source_column??row.aggregate_id??''),aggregate_id:String(row.aggregate_id??''),dtype:normalizeDtype(row.dtype,row.width),
    rows:integer(row.rows??row.row_count),valid_count:valid,missing_count:integer(row.missing_count),native_bytes:nativeBytes,frame_bytes:frameBytes,
    compression_rate:finiteNumber(row.compression_rate??(nativeBytes>0&&frameBytes!==null?frameBytes/nativeBytes:null)),
    encode_ns_per_value:finiteNumber(row.encode_ns_per_value??row.compression_time_ns_per_point??(valid>0&&encodeTotal!==null?encodeTotal/valid:null)),
    decode_ns_per_value:finiteNumber(row.decode_ns_per_value??row.decompression_time_ns_per_point??(valid>0&&decodeTotal!==null?decodeTotal/valid:null)),
    blocks:integer(row.blocks),scaled_blocks:integer(row.scaled_blocks),raw_blocks:integer(row.raw_blocks),validity_bytes:integer(row.validity_bytes),header_bytes:integer(row.header_bytes),verified_values:integer(row.verified_values),
    frame_sequence_sha256:String(row.frame_sequence_sha256??''),receipt_sha256:String(row.receipt_sha256??''),message:String(row.message??'')};
}
function normalizeDatasetReceipt(row) {
  return {...row,dataset:String(row.dataset??row.dataset_id??''),runtime:String(row.runtime??''),status:normalizeStatus(row.status),reason:String(row.reason??row.message??''),receipt_sha256:String(row.receipt_sha256??'')};
}
function uniqueAggregateKey(row) {
  return row.aggregate_id||`${row.dataset}\0${row.column}\0${row.dtype}`;
}
function groupBy(items,keyFn) {
  const groups=new Map();
  for(const item of items){const key=keyFn(item);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item);}
  return groups;
}
function summaryObservation(columnRows=[],receiptRows=[]) {
  const datasetGroups=groupBy(receiptRows.filter(row=>row.dataset),row=>row.dataset);
  const aggregateGroups=groupBy(columnRows.filter(row=>uniqueAggregateKey(row)),uniqueAggregateKey);
  return {
    dataset_receipt_count:datasetGroups.size,
    measured_dataset_count:[...datasetGroups.values()].filter(rows=>rows.some(row=>successfulStatus(row.status))).length,
    n_a_dataset_count:[...datasetGroups.values()].filter(rows=>rows.length&&rows.every(row=>row.status==='n_a')).length,
    schema_aggregate_count:aggregateGroups.size,
    nonempty_aggregate_count:[...aggregateGroups.values()].filter(rows=>rows.some(row=>(row.valid_count??0)>0&&successfulStatus(row.status))).length,
    n_a_aggregate_count:[...aggregateGroups.values()].filter(rows=>rows.length&&rows.every(row=>row.status==='n_a')).length,
  };
}
function auditValidationErrors(auditData,columnRows,receiptRows) {
  const errors=[],add=(message)=>{if(errors.length<500)errors.push(message);},declared=auditData.summary||{},observed=summaryObservation(columnRows,receiptRows),complete=normalizeStatus(auditData.status)==='complete';
  if(auditData.schema_version!=='dbp-native-float-audit-v1')add('schema_version is not dbp-native-float-audit-v1');
  for(const key of Object.keys(EXPECTED_COMPLETE_SUMMARY)){const expected=integer(declared[key]),actual=integer(observed[key]);if(expected!==null&&actual!==null&&expected!==actual)add(`${key}: summary ${expected}, detail ${actual}`);}
  if(!complete)return errors;
  for(const [key,value] of Object.entries(EXPECTED_COMPLETE_SUMMARY)){if(integer(declared[key])!==value)add(`${key}: complete scope requires ${value}`);}
  const runtimeRows=Array.isArray(auditData.runtimes)?auditData.runtimes:[],runtimeIds=runtimeRows.map(row=>String(row.runtime??''));
  if(runtimeRows.length!==3||new Set(runtimeIds).size!==3||!EXPECTED_RUNTIMES.every(runtime=>runtimeIds.includes(runtime)))add('complete audit requires exactly C++, Java, and Python runtime receipts');
  for(const row of runtimeRows){if(normalizeStatus(row.status)!=='complete')add(`${runtimeLabel(row.runtime)} runtime status is not complete`);if(!SHA256_PATTERN.test(String(row.result_sha256??'')))add(`${runtimeLabel(row.runtime)} runtime result chain SHA is absent`);}
  const byDataset=groupBy(receiptRows.filter(row=>row.dataset),row=>row.dataset);
  if(receiptRows.length!==507||byDataset.size!==169)add('complete audit requires 507 runtime receipts for 169 unique datasets');
  for(const [dataset,rows] of byDataset){const ids=rows.map(row=>row.runtime);if(rows.length!==3||new Set(ids).size!==3||!EXPECTED_RUNTIMES.every(runtime=>ids.includes(runtime)))add(`${dataset}: dataset receipts are not exactly C++, Java, and Python`);if(rows.some(row=>!['ok','n_a'].includes(row.status)))add(`${dataset}: dataset receipt has a nonfinal status`);if(new Set(rows.map(row=>row.status)).size!==1)add(`${dataset}: runtime dataset statuses differ`);if(rows.some(row=>!SHA256_PATTERN.test(row.receipt_sha256)))add(`${dataset}: dataset run receipt SHA is absent`);}
  const aggregateGroups=groupBy(columnRows.filter(row=>row.aggregate_id),row=>row.aggregate_id);
  if(columnRows.length!==12570||aggregateGroups.size!==4190)add('complete audit requires 12,570 runtime rows for 4,190 unique aggregates');
  const exactFields=['status','dataset','column','dtype','width','max_point','rows','valid_count','missing_count','native_bytes','frame_bytes','validity_bytes','header_bytes','blocks','scaled_blocks','raw_blocks','verified_values','frame_sequence_sha256','expected_sequence_chain_sha256','receipt_sha256','source_manifest_receipt_sha256','source_aggregates_sha256','source_blocks_sha256','source_block_catalog_sha256','runtime_artifact_manifest_sha256','source_column_inventory_sha256','scope_sha256'];
  for(const row of columnRows){
    const id=row.aggregate_id||'(missing aggregate_id)',width=row.dtype==='float32'?4:row.dtype==='float64'?8:null,expectedStatus=(row.valid_count??0)>0?'ok':'n_a';
    if(!row.aggregate_id)add('a complete column row is missing aggregate_id');
    if(!byDataset.has(row.dataset))add(`${id}: column dataset is absent from dataset receipts`);
    for(const field of ['width','max_point','rows','valid_count','missing_count','native_bytes','frame_bytes','validity_bytes','header_bytes','encode_ns','decode_ns','blocks','scaled_blocks','raw_blocks','verified_values'])if(finiteNumber(row[field])===null)add(`${id} ${runtimeLabel(row.runtime)}: ${field} is absent or invalid`);
    if(row.status!==expectedStatus)add(`${id} ${runtimeLabel(row.runtime)}: status does not match valid_count`);
    if(!width||row.native_bytes!==width*row.valid_count)add(`${id} ${runtimeLabel(row.runtime)}: native-byte accounting differs`);
    if(row.rows!==row.valid_count+row.missing_count)add(`${id} ${runtimeLabel(row.runtime)}: row accounting differs`);
    if(row.header_bytes!==10*row.blocks)add(`${id} ${runtimeLabel(row.runtime)}: header-byte accounting differs`);
    if(row.verified_values!==row.valid_count)add(`${id} ${runtimeLabel(row.runtime)}: verified value count differs`);
    if(row.scaled_blocks+row.raw_blocks!==row.blocks)add(`${id} ${runtimeLabel(row.runtime)}: mode counts differ`);
    if(row.valid_count>0&&(finiteNumber(row.compression_rate)===null||finiteNumber(row.encode_ns_per_value)===null||finiteNumber(row.decode_ns_per_value)===null))add(`${id} ${runtimeLabel(row.runtime)}: measured rate or timing is absent`);
    for(const field of REQUIRED_COLUMN_SHA_FIELDS)if(!SHA256_PATTERN.test(String(row[field]??'')))add(`${id} ${runtimeLabel(row.runtime)}: ${field} is absent or invalid`);
  }
  for(const [aggregateId,rows] of aggregateGroups){const ids=rows.map(row=>row.runtime);if(rows.length!==3||new Set(ids).size!==3||!EXPECTED_RUNTIMES.every(runtime=>ids.includes(runtime)))add(`${aggregateId}: result rows are not exactly C++, Java, and Python`);for(const field of exactFields)if(new Set(rows.map(row=>JSON.stringify(row[field]??null))).size!==1)add(`${aggregateId}: ${field} differs across runtimes`);}
  const evidence=auditData.evidence||{};
  for(const field of REQUIRED_EVIDENCE_SHA_FIELDS)if(!SHA256_PATTERN.test(String(evidence[field]??'')))add(`evidence.${field} is absent or invalid`);
  const runtimeChains=evidence.runtime_result_chain_sha256||{};for(const runtime of EXPECTED_RUNTIMES)if(!SHA256_PATTERN.test(String(runtimeChains[runtime]??'')))add(`evidence.runtime_result_chain_sha256.${runtime} is absent or invalid`);
  if(integer(auditData.measurement?.repeats)!==1||integer(auditData.measurement?.warmup_blocks)!==20||integer(auditData.measurement?.header_bytes_per_block)!==10)add('complete audit measurement contract differs');
  if(auditData.environment?.shared_server!==true||integer(auditData.environment?.cpu?.passed_receipts_at_capture)!==169)add('complete audit environment receipt does not cover all 169 dataset receipts');
  return errors;
}
module.exports={auditValidationErrors,normalizeColumn,normalizeDatasetReceipt,summaryObservation};
