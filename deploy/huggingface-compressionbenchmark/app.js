import {WEIGHT_TENTHS,parseWeights,addRankingMetrics} from './ranking.js?v=20261011-support-matched-v1';
const $ = (id) => document.getElementById(id);
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (v) => v != null && Number.isFinite(Number(v)) ? Number(v) : null;
const mean = (values) => { const a = values.map(number).filter((n) => n !== null); return a.length ? a.reduce((s,n) => s+n,0)/a.length : null; };
function timingMode(input) {
  if(!input.length||input.some(r=>number(r.compression_time)===null||number(r.decompression_time)===null||!Number.isInteger(r.timing_column_count)||r.timing_column_count<0||!((r.timing_aggregation==='column_mean'&&r.timing_column_count>0)||(r.timing_aggregation==='dataset_aggregate'&&r.timing_column_count===0))))return 'unavailable';
  return input.some(r=>r.timing_aggregation==='dataset_aggregate')?'dataset_mean':'column_mean';
}
function aggregateTime(input,key) {
  const mode=timingMode(input);
  if(mode==='unavailable')return null;
  if(mode==='dataset_mean')return mean(input.map(r=>r[key]));
  return input.reduce((sum,r)=>sum+r[key]*r.timing_column_count,0)/input.reduce((sum,r)=>sum+r.timing_column_count,0);
}
const timingLabel=mode=>({column_mean:'Equal-column arithmetic mean',dataset_mean:'Equal-dataset arithmetic mean',dataset_aggregate:'Dataset aggregate',unavailable:'Not reported'}[mode]??'Not reported');
const fmt = (v, digits=3) => number(v) === null ? 'Not reported' : Number(v).toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits});
const countFmt = (v) => Number(v).toLocaleString('en-US');
const runtimeName = (v) => snapshot?.benchmark?.runtime_labels?.[v] || ({cpp:'C++',java:'Java',python:'Python','cpp-native166':'Native C++ · server 166','cpp-native192':'Native C++ · server 192'}[v] || v);
const runtimeHost = (v) => snapshot?.benchmark?.runtime_hosts?.[v] || ({'cpp-native166':'166','cpp-native192':'192'}[v] || 'source snapshot');
const THULAB_DATASET_PREFIX='https://huggingface.co/datasets/THULab/';
const requestedDataset=new URLSearchParams(location.search).get('dataset')?.trim()||'';
let requestedDatasetApplied=false;
const cols = [
  {key:'average_rank',label:'Average Rank ↓',digits:2,visible:true,overallOnly:true},
  {key:'rank_coverage',label:'Rank coverage',digits:0,visible:false,overallOnly:true},
  {key:'composite_score',label:'Composite score ↓',digits:4,visible:true,overallOnly:true},
  {key:'compression_rate',label:'Average rate ↓',digits:4,visible:true},
  {key:'overall_rate',label:'Overall rate ↓',digits:4,visible:false},
  {key:'compression_time',label:'Compress ns/point ↓',digits:2,visible:true},
  {key:'decompression_time',label:'Decompress ns/point ↓',digits:2,visible:true},
  {key:'normalized_compression_time',label:'Normalized compress ↓',digits:4,visible:false,overallOnly:true},
  {key:'normalized_decompression_time',label:'Normalized decompress ↓',digits:4,visible:false,overallOnly:true},
  {key:'timing_column_count',label:'Measured columns',digits:0,visible:false},
  {key:'compressed_bytes',label:'Compressed bytes',digits:0,visible:false},
  {key:'original_bytes',label:'Original bytes',digits:0,visible:false},
];
let snapshot, records=[], methods=new Map(), datasets=new Map(), currentRows=[], activeTab='overall';
let referenceCatalog=null,referenceCatalogError=null,datasetCatalog=null,datasetCatalogError=null;
let sort={key:'compression_rate',direction:1};
let selectedMethod=null;
const state = () => ({runtime:$('runtime').value,dataset:$('dataset').value,dtype:$('dtype').value,precision:$('precision').value,family:$('family').value,mode:$('row-mode').value,search:$('search').value.trim().toLowerCase(),complete:$('complete-only').checked});
function scopeKey() { const s=state(); return s.dtype==='overall'?'overall':s.dtype==='int'?'integer':s.precision==='fixed'?'fixed_float':s.precision==='non_fixed'?'nonfixed_float':'float'; }
function normalizeRecord(r) {
  const sourceRuntime=r.runtime ?? r.implementation_language ?? snapshot.benchmark?.implementation_language ?? 'cpp';
  return { ...r,configuration:r.algorithm_variant??r.algorithm??r.method,method:r.algorithm_variant??r.algorithm??r.method,source_runtime:sourceRuntime,runtime:r.comparison_profile??sourceRuntime,
    family:r.family ?? r.method_family ?? r.method ?? methods.get(r.algorithm)?.family ?? r.algorithm,
    compression_rate:number(r.compression_rate ?? r.average_compression_rate),compression_time:number(r.compression_time_ns_per_point ?? r.compression_time),decompression_time:number(r.decompression_time_ns_per_point ?? r.decompression_time),
    original_bytes:number(r.original_size_bytes ?? r.original_bytes ?? r.originalSize),compressed_bytes:number(r.compressed_size_bytes ?? r.compressed_bytes ?? r.compressedSize),
    eligible_column_set_sha256:r.eligible_column_set_sha256??'legacy_snapshot_unbound',measurement_status:r.measurement_status??'legacy_published_measurement',timing_profile_status:r.timing_profile_status??'source_snapshot_timing',binary_sha256:r.binary_sha256??'unbound' };
}
function filteredRecords(ignoreMethod=false) {
  const s=state(), scope=scopeKey();
  let result=records.filter((r) => r.runtime===s.runtime && r.scope===scope && (s.dataset==='all'||r.dataset===s.dataset));
  if(s.mode==='family') {
    const choices=new Map(),best=new Map();
    for(const r of result){const key=`${r.dataset}\0${r.family}`;if(!choices.has(key))choices.set(key,[]);choices.get(key).push(r);}
    for(const [key,rows] of choices){if(new Set(rows.map(r=>r.eligible_column_set_sha256)).size!==1)continue;for(const r of rows){const prev=best.get(key);if(!prev||r.compression_rate<prev.compression_rate||(r.compression_rate===prev.compression_rate&&r.configuration.localeCompare(prev.configuration)<0))best.set(key,r);}}
    result=[...best.values()].map(r=>({...r,method:r.family}));
  }
  return ignoreMethod?result:result.filter(r=>(s.family==='all'||r.family===s.family)&&(!s.search||(s.mode==='family'?r.family:`${r.method} ${r.family}`).toLowerCase().includes(s.search)));
}
function coverageDatasetCount() { return new Set(filteredRecords(true).map(r=>r.dataset)).size; }
function aggregate(input) {
  const groups=new Map();
  const exact=state().mode==='configuration',familiesByConfiguration=new Map();
  if(exact)for(const r of input){if(!familiesByConfiguration.has(r.method))familiesByConfiguration.set(r.method,new Set());familiesByConfiguration.get(r.method).add(r.family);}
  for(const r of input) { const key=exact?`${r.family}\0${r.method}`:r.method;if(!groups.has(key)) groups.set(key,[]); groups.get(key).push(r); }
  const totalDatasets=coverageDatasetCount();
  const rows=[];
  for(const [,source] of groups) {
    const configuration=source[0].method,method=exact&&familiesByConfiguration.get(configuration).size>1?`${source[0].family} · ${configuration}`:configuration;
    const hasBytes=source.every(r=>r.original_bytes!==null && r.original_bytes>0 && r.compressed_bytes!==null && r.compressed_bytes>0);
    const original=hasBytes?source.reduce((s,r)=>s+r.original_bytes,0):null, compressed=hasBytes?source.reduce((s,r)=>s+r.compressed_bytes,0):null;
    rows.push({method,family:source[0].family,runtime:source[0].runtime,compression_rate:mean(source.map(r=>r.compression_rate)),compression_time:aggregateTime(source,'compression_time'),decompression_time:aggregateTime(source,'decompression_time'),timing_column_count:source.reduce((sum,r)=>sum+r.timing_column_count,0),timing_aggregation:timingMode(source),overall_rate:hasBytes?compressed/original:null,original_bytes:original,compressed_bytes:compressed,coverage:new Set(source.map(r=>r.dataset)).size,totalDatasets,source});
  }
  const s=state();
  return addRankingMetrics(rows,parseWeights($('weights').value)).filter(r=>(s.family==='all'||r.family===s.family)&&(!s.search||(s.mode==='family'?r.family:`${r.method} ${r.family}`).toLowerCase().includes(s.search))&&(!s.complete||r.coverage===totalDatasets)).sort((a,b)=>{
    const av=number(a[sort.key]),bv=number(b[sort.key]);
    if(av===null&&bv===null)return a.method.localeCompare(b.method);
    if(av===null)return 1;if(bv===null)return -1;
    return sort.direction*(av-bv)||a.method.localeCompare(b.method);
  });
}
function renderTable() {
  const visible=cols.filter(c=>c.visible&&(!c.overallOnly||activeTab==='overall'));
  $('leaderboard').querySelector('thead').innerHTML=`<tr><th>Position</th><th>Method / configuration</th>${visible.map(c=>`<th aria-sort="${sort.key===c.key?(sort.direction===1?'ascending':'descending'):'none'}"><button data-sort="${c.key}" title="Sort ${escape(c.label)}">${escape(c.label)}${sort.key===c.key?(sort.direction===1?' ▴':' ▾'):''}</button></th>`).join('')}<th>Dataset coverage</th></tr>`;
  const minima=Object.fromEntries(visible.map(c=>[c.key,Math.min(...currentRows.map(r=>number(r[c.key])).filter(n=>n!==null))]));
  $('leaderboard').querySelector('tbody').innerHTML=currentRows.length?currentRows.map((r,i)=>`<tr><td class="rank numeric">${i+1}</td><td><button class="method-button" data-method="${escape(r.method)}">${escape(r.method)}</button><span class="method-family">${escape(r.family)} · ${escape(runtimeName(r.runtime))}</span></td>${visible.map(c=>`<td class="numeric ${!['compressed_bytes','original_bytes','rank_coverage','timing_column_count'].includes(c.key)&&number(r[c.key])!==null&&r[c.key]===minima[c.key]?'best':''}">${c.key==='rank_coverage'?`${r.rank_coverage} / ${r.totalDatasets}`:fmt(r[c.key],c.digits)}</td>`).join('')}<td><span class="coverage ${r.coverage===r.totalDatasets?'full':''}">${r.coverage} / ${r.totalDatasets}</span></td></tr>`).join(''):`<tr><td colspan="${visible.length+3}" class="empty-cell">No measurements match this selection. Try another runtime, data type, or dataset.</td></tr>`;
  for(const button of $('leaderboard').querySelectorAll('[data-sort]'))button.addEventListener('click',()=>{const key=button.dataset.sort;sort={key,direction:sort.key===key?-sort.direction:1};refresh();});
  for(const button of $('leaderboard').querySelectorAll('[data-method]'))button.addEventListener('click',()=>showMethod(button.dataset.method,true));
  $('row-count').textContent=`${currentRows.length} ${state().mode==='family'?'method families':'method configurations'} · sorted by ${cols.find(c=>c.key===sort.key)?.label??sort.key}`;
  $('download').disabled=!currentRows.length;
}
function refresh() {
  if(!snapshot)return;
  $('precision').disabled=$('dtype').value!=='float';
  const selectedDataset=state().dataset,selectedHasMeasurements=selectedDataset==='all'||records.some(r=>r.dataset===selectedDataset),columnCoverageNote='Compress and Decompress times use equal-column arithmetic means when every selected dataset has column timings. If a method has aggregate-only timings in the selection, its dataset ns/point values are averaged with equal weight per dataset. Timing basis is shown in method details and CSV. Column coverage may differ by method.';
  if(!selectedHasMeasurements) {
    $('results-note').textContent=`No published measurements are available for this catalog dataset. It is excluded from rankings rather than treated as zero. ${columnCoverageNote}`;
  } else if(scopeKey()==='float'&&!records.some(r=>r.scope==='float')) {
    // The exporter must publish an explicit all-float scope; never combine
    // preaggregated fixed/non-fixed metrics and call it an all-float result.
    $('results-note').textContent=`All-float measurements are not available in this snapshot. Select a precision category. ${columnCoverageNote}`;
  } else $('results-note').textContent=`${state().mode==='family'?'Each family uses its lowest-rate measured configuration only when that dataset/family has one exact support set; ambiguous support sets are omitted. Configurations may differ across datasets; use Exact configuration for fixed parameters.':'Each row retains one exact configuration and its eligible support-set hash. Lower compression rates and lower ns/point are better.'} ${columnCoverageNote}`;
  currentRows=aggregate(filteredRecords(true));
  const weights=parseWeights($('weights').value);
  const comparable=currentRows.length&&currentRows.every(r=>r.workload_signature_count<=1);
  $('score-note').textContent=`Score = ${weights[0].toFixed(1)} × average rate + ${weights[1].toFixed(1)} × normalized compress + ${weights[2].toFixed(1)} × normalized decompress. Lower is better. ${comparable?'Composite scores use one shared dataset/support signature.':'Composite scores are withheld because the displayed methods do not share one exact dataset/support signature.'} Average Rank is computed only among rows with the same dataset and eligible aggregate-id support hash; ties share their average position. Missing, N/A, failed, and unlike-support results receive no rank. Compare rank coverage and inspect method details.`;
  const s=state(),scope=scopeKey(),measuredCount=coverageDatasetCount(),ds=s.dataset==='all'?`all ${measuredCount} measured datasets`:datasetDisplayName(s.dataset);
  $('dataset-count').textContent=countFmt(measuredCount);$('dataset-count-label').textContent='measured datasets';
  const names={overall:'available numeric measurements',integer:'integer columns',float:'floating-point columns',fixed_float:'fixed-precision float columns',nonfixed_float:'non-fixed-precision float columns'};
  $('selection-summary').textContent=`${runtimeName(s.runtime)} · ${names[scope]||scope} · ${ds} · ${currentRows.length} methods`;
  $('view-title').textContent=activeTab==='dataset'?`${s.dataset==='all'?'Dataset':datasetDisplayName(s.dataset)} leaderboard`:activeTab==='tradeoffs'?'Storage and execution trade-offs':'Overall leaderboard';
  const selectedEntry=s.dataset==='all'?null:datasets.get(s.dataset),selectedDatasetUrl=thulabDatasetUrl(selectedEntry),selectedBadge=$('selected-dataset-badge');
  selectedBadge.hidden=!selectedDatasetUrl;
  if(selectedDatasetUrl){selectedBadge.href=selectedDatasetUrl;selectedBadge.setAttribute('aria-label',`Open ${datasetDisplayName(s.dataset)} in the THULab Hugging Face organization`);selectedBadge.title='Open this dataset in the THULab Hugging Face organization.';}
  renderTable();
  if(activeTab==='tradeoffs')renderChart();
  if(selectedMethod)showMethod(selectedMethod,false);
}
function setTab(tab) {
  activeTab=['overall','dataset','tradeoffs','methods','operators','about'].includes(tab)?tab:'overall';
  for(const b of document.querySelectorAll('[data-tab]')){b.classList.toggle('active',b.dataset.tab===activeTab);if(b.dataset.tab===activeTab)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');}
  $('results-view').hidden=['methods','operators','about'].includes(activeTab);$('methods-view').hidden=activeTab!=='methods';$('operators-view').hidden=activeTab!=='operators';$('about-view').hidden=activeTab!=='about';$('chart-view').hidden=activeTab!=='tradeoffs';$('dataset-catalog-panel').hidden=activeTab!=='dataset';
  $('score-controls').hidden=activeTab!=='overall';
  for(const input of $('column-options').querySelectorAll('input'))input.parentElement.hidden=Boolean(cols.find(c=>c.key===input.dataset.column)?.overallOnly&&activeTab!=='overall');
  if(activeTab!=='overall'&&cols.find(c=>c.key===sort.key)?.overallOnly)sort={key:'compression_rate',direction:1};
  if(activeTab==='dataset'&&$('dataset').value==='all'){const first=[...datasets.keys()].find(k=>records.some(r=>r.dataset===k&&r.runtime===$('runtime').value));if(first)$('dataset').value=first;}
  if(activeTab==='overall')$('dataset').value='all';
  if(activeTab==='dataset')renderDatasetCatalog();
  updateUrlState();
  refresh();
}
function showMethod(method,focus=false) {
  const row=currentRows.find(r=>r.method===method), detail=$('method-detail');
  if(!row){selectedMethod=null;detail.hidden=true;return;}selectedMethod=method;
  const meta=methods.get(method)??methods.get(row.source[0].configuration), pipeline=meta?.operator_pipeline;
  const source=row.source.slice().sort((a,b)=>a.dataset.localeCompare(b.dataset));
  detail.innerHTML=`<div class="detail-head"><div><h2 id="detail-title">${escape(method)}</h2><p class="muted">${escape(row.family)} · ${escape(runtimeName(row.runtime))} · ${row.coverage} measured datasets</p></div><button id="close-detail" class="quiet-button">Close details</button></div><div class="detail-grid"><div><strong>${fmt(row.compression_rate,4)}</strong><span>average compression rate</span></div><div><strong>${fmt(row.compression_time,2)}</strong><span>Compress ns/point</span></div><div><strong>${fmt(row.decompression_time,2)}</strong><span>Decompress ns/point</span></div><div><strong>${row.coverage} / ${row.totalDatasets}</strong><span>dataset coverage</span></div></div>${pipeline?`<p class="source-code">Interpretive operator mapping: ${escape(typeof pipeline==='string'?pipeline:JSON.stringify(pipeline))}</p>`:''}<p class="muted">Timing average: ${escape(timingLabel(row.timing_aggregation))}. Dataset metrics below come from the published result files. Measurement boundaries, matched support-set hashes, source hashes, and available verification evidence are recorded in the manifest.</p><div class="detail-table"><table><thead><tr><th>Dataset</th><th>Configuration</th><th>Rate ↓</th><th>Compress ns/point ↓</th><th>Decompress ns/point ↓</th><th>Timing basis</th><th>Measurement evidence</th><th>Timing profile</th><th>Source profile / binary</th><th>Support set</th><th>Original bytes</th><th>Compressed bytes</th><th>Dataset score ↓</th><th>Dataset Rank ↓</th><th>Rank candidates</th></tr></thead><tbody>${source.map(r=>`<tr><td>${escape(r.dataset)}</td><td>${escape(r.configuration)}</td><td class="numeric">${fmt(r.compression_rate,4)}</td><td class="numeric">${fmt(r.compression_time,2)}</td><td class="numeric">${fmt(r.decompression_time,2)}</td><td>${escape(timingLabel(r.timing_aggregation))}</td><td>${escape(r.measurement_status)}</td><td>${escape(r.timing_profile_status)}</td><td class="source-code">${escape(r.source_runtime)} / ${escape(String(r.binary_sha256).slice(0,16))}</td><td class="source-code">${escape(String(r.eligible_column_set_sha256).slice(0,16))}</td><td class="numeric">${fmt(r.original_bytes,0)}</td><td class="numeric">${fmt(r.compressed_bytes,0)}</td><td class="numeric">${fmt(r.dataset_score,4)}</td><td class="numeric">${fmt(r.dataset_rank,2)}</td><td class="numeric">${fmt(r.dataset_candidate_count,0)}</td></tr>`).join('')}</tbody></table></div>`;
  detail.hidden=false;$('close-detail').addEventListener('click',()=>{detail.hidden=true;selectedMethod=null;});
  if(focus){detail.scrollIntoView({block:'nearest',behavior:'auto'});detail.focus({preventScroll:true});}
}
function renderChart() {
  const metric=$('chart-metric').value;
  const valid=currentRows.filter(r=>r.compression_rate>0&&r[metric]>0);
  if(!valid.length){$('chart').innerHTML='<p class="chart-empty muted">No positive measurements to plot for this selection.</p>';return;}
  const width=1100,height=410,pad={left:85,right:35,top:30,bottom:65};
  const maxRate=Math.max(...valid.map(r=>r.compression_rate))*1.08;
  const logs=valid.map(r=>Math.log10(r[metric]));let minLog=Math.floor(Math.min(...logs)),maxLog=Math.ceil(Math.max(...logs));if(maxLog===minLog)maxLog=minLog+1;
  const x=r=>pad.left+r/maxRate*(width-pad.left-pad.right),y=n=>height-pad.bottom-(Math.log10(n)-minLog)/(maxLog-minLog)*(height-pad.top-pad.bottom);
  let graph='';
  for(let i=0;i<=5;i++){const rate=maxRate*i/5,px=x(rate);graph+=`<line class="gridline" x1="${px}" y1="${pad.top}" x2="${px}" y2="${height-pad.bottom}"/><text class="axis-label" x="${px}" y="${height-pad.bottom+25}" text-anchor="middle">${fmt(rate,2)}</text>`;}
  for(let i=minLog;i<=maxLog;i++){const n=10**i,py=y(n);graph+=`<line class="gridline" x1="${pad.left}" y1="${py}" x2="${width-pad.right}" y2="${py}"/><text class="axis-label" x="${pad.left-12}" y="${py+4}" text-anchor="end">${n.toLocaleString('en-US',{maximumFractionDigits:3})}</text>`;}
  for(const r of valid)graph+=`<circle class="point" tabindex="0" role="button" aria-label="${escape(r.method)}: compression rate ${fmt(r.compression_rate,4)}, ${metric==='compression_time'?'compression':'decompression'} ${fmt(r[metric],2)} ns per point" data-method="${escape(r.method)}" cx="${x(r.compression_rate)}" cy="${y(r[metric])}" r="6"><title>${escape(r.method)} | rate ${fmt(r.compression_rate,4)} | ${fmt(r[metric],2)} ns/point | coverage ${r.coverage}/${r.totalDatasets}</title></circle>`;
  graph+=`<text class="axis-label" x="${width/2}" y="${height-13}" text-anchor="middle">Average compression rate (compressed / original) ↓</text><text class="axis-label" transform="translate(20 ${height/2}) rotate(-90)" text-anchor="middle">${metric==='compression_time'?'Compress':'Decompress'} ns/point (log scale) ↓</text>`;
  $('chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Measured compression rates versus ${metric==='compression_time'?'compression':'decompression'} cost">${graph}</svg>`;
  for(const p of $('chart').querySelectorAll('[data-method]')){p.addEventListener('click',()=>showMethod(p.dataset.method,true));p.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();showMethod(p.dataset.method,true);}});}
}
function csv() {
  const quote=v=>{let text=String(v??'');if(typeof v==='string'&&/^[=+@-]/.test(text))text="'"+text;return `"${text.replace(/"/g,'""')}"`;};
  const weights=parseWeights($('weights').value);
  const columns=['position','method','comparison_profile','execution_host','scope','row_mode','selected_dataset','sort_metric','sort_direction','average_compression_rate','overall_compression_rate','compress_ns_per_point','decompress_ns_per_point','weight_a','weight_b','weight_c','normalized_compress','normalized_decompress','composite_score','average_rank','rank_dataset_coverage','normalization_reference_rows','compress_min','compress_max','decompress_min','decompress_max','normalization','rank_definition','dataset_coverage','datasets_in_scope','selected_configuration_by_dataset','source_profile_by_dataset','binary_sha256_by_dataset','eligible_column_set_sha256_by_dataset','measurement_status_by_dataset','timing_profile_status_by_dataset','timing_column_count','timing_aggregation'];
  const body=currentRows.map((r,i)=>[i+1,r.method,r.runtime,runtimeHost(r.runtime),scopeKey(),state().mode,state().dataset,sort.key,sort.direction===1?'ascending':'descending',r.compression_rate,r.overall_rate,r.compression_time,r.decompression_time,...weights.map(w=>w.toFixed(1)),r.normalized_compression_time,r.normalized_decompression_time,r.composite_score,r.average_rank,r.rank_coverage,r.normalization_reference_rows,r.encode_min,r.encode_max,r.decode_min,r.decode_max,'min-max over all measured rows in selected comparison profile/scope/row-mode before search and visibility filters','mean per-dataset composite-score fractional rank; per-dataset min-max times',r.coverage,r.totalDatasets,JSON.stringify(Object.fromEntries(r.source.map(s=>[s.dataset,s.configuration]))),JSON.stringify(Object.fromEntries(r.source.map(s=>[s.dataset,s.source_runtime]))),JSON.stringify(Object.fromEntries(r.source.map(s=>[s.dataset,s.binary_sha256]))),JSON.stringify(Object.fromEntries(r.source.map(s=>[s.dataset,s.eligible_column_set_sha256]))),JSON.stringify(Object.fromEntries(r.source.map(s=>[s.dataset,s.measurement_status]))),JSON.stringify(Object.fromEntries(r.source.map(s=>[s.dataset,s.timing_profile_status]))),r.timing_column_count,r.timing_aggregation].map(quote).join(','));
  const url=URL.createObjectURL(new Blob(['\ufeff'+[columns.join(','),...body].join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`THULab-compression-${state().runtime}-${scopeKey()}-${state().dataset}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function safeHttpUrl(value) {
  try { const url=new URL(String(value));return ['http:','https:'].includes(url.protocol)?url.href:null; } catch { return null; }
}
function thulabDatasetUrl(entry) {
  const url=safeHttpUrl(entry?.dataset_url);
  return url?.startsWith(THULAB_DATASET_PREFIX)?url:null;
}
function thulabDatasetBadge(entry,key=datasetKey(entry)) {
  const url=thulabDatasetUrl(entry);if(!url)return '';
  return `<a class="thulab-dataset-badge" href="${escape(url)}" target="_blank" rel="noopener noreferrer" aria-label="Open ${escape(key)} in the THULab Hugging Face organization" title="Open this dataset in the THULab Hugging Face organization."><span aria-hidden="true">🏆</span> THULab hosted ↗</a>`;
}
function publicNote(value) {
  const note=typeof value==='string'?value.trim():'';
  return /(?:\/Users\/|[A-Za-z]:\\|file:\/\/|codex:\/\/threads\/|\.pptx\b)/i.test(note)?'':note;
}
function publicText(value,fallback='') { const text=publicNote(value);return text||fallback; }
function referenceMap() {
  return new Map((referenceCatalog?.references||[]).map(reference=>[String(reference.id),reference]));
}
function resolvedReferences(ids) {
  const byId=referenceMap();
  return [...new Set(Array.isArray(ids)?ids.map(String):[])].map(id=>byId.get(id)).filter(Boolean);
}
function referenceSearchText(ids) {
  return resolvedReferences(ids).map(reference=>`${reference.id??''} ${reference.citation??''} ${reference.kind??''}`).join(' ');
}
function renderReferences(ids) {
  const references=resolvedReferences(ids);
  if(!references.length)return '<p class="reference-empty">No external reference is listed for this entry.</p>';
  return `<div class="documentation-references"><strong>References</strong><ul>${references.map(reference=>{const url=safeHttpUrl(reference.url),citation=escape(reference.citation||reference.id||'Reference');const meta=[reference.id,reference.source_slide?`Presentation Notes · slide ${reference.source_slide}`:'Additional official source'].filter(Boolean).map(escape).join(' · ');return `<li>${url?`<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${citation}</a>`:`<span>${citation}</span>`}${meta?`<small>${meta}</small>`:''}</li>`;}).join('')}</ul></div>`;
}
function familyConfigurations() {
  const grouped=new Map();
  for(const record of records){if(!grouped.has(record.family))grouped.set(record.family,new Set());grouped.get(record.family).add(record.configuration);}
  return new Map([...grouped].map(([family,names])=>[family,[...names].sort((a,b)=>a.localeCompare(b))]));
}
function catalogMethod(family) {
  const entries=referenceCatalog?.methods||[];
  return entries.find(entry=>entry.family===family)||entries.find(entry=>String(entry.family).toLowerCase()===family.toLowerCase())||null;
}
function renderMethodDocumentation() {
  const container=$('method-catalog'),status=$('method-doc-status');
  if(!container||!status)return;
  if(referenceCatalogError){status.textContent='Method documentation could not be loaded.';container.innerHTML=`<p class="documentation-error">${escape(referenceCatalogError.message)}</p>`;return;}
  if(!referenceCatalog){status.textContent='Loading method documentation…';container.innerHTML='';return;}
  const configurations=familyConfigurations(),query=$('method-doc-search').value.trim().toLowerCase();
  const families=[...configurations.keys()].sort((a,b)=>a.localeCompare(b));
  const rows=families.map(family=>{const entry=catalogMethod(family)||{},names=configurations.get(family),note=publicNote(entry.notes),search=[family,entry.display_name,entry.description,note,...names,referenceSearchText(entry.reference_ids)].join(' ').toLowerCase();return {family,entry,names,note,search};}).filter(row=>!query||row.search.includes(query));
  const described=families.filter(family=>catalogMethod(family)?.description).length;
  status.textContent=`${rows.length} of ${families.length} measured method families shown · ${described} descriptions available`;
  container.innerHTML=rows.length?rows.map(({family,entry,names,note})=>{const display=entry.display_name||family;return `<article class="documentation-card"><div class="documentation-card-head"><div><h3>${escape(display)}</h3>${display!==family?`<span class="documentation-code">${escape(family)}</span>`:''}</div><span class="documentation-count">${names.length} configuration${names.length===1?'':'s'}</span></div><p class="documentation-description">${escape(entry.description||'A reference description is not available for this measured family.')}</p>${note?`<p class="documentation-note">${escape(note)}</p>`:''}${renderReferences(entry.reference_ids)}<details class="configuration-details"><summary>Published configurations (${names.length})</summary><ul>${names.map(name=>`<li>${escape(name)}</li>`).join('')}</ul></details></article>`;}).join(''):'<p class="documentation-empty">No method documentation matches this search.</p>';
}
function renderOperatorDocumentation() {
  const container=$('operator-catalog'),status=$('operator-doc-status');
  if(!container||!status)return;
  if(referenceCatalogError){status.textContent='Operator documentation could not be loaded.';container.innerHTML=`<p class="documentation-error">${escape(referenceCatalogError.message)}</p>`;return;}
  if(!referenceCatalog){status.textContent='Loading operator documentation…';container.innerHTML='';return;}
  const query=$('operator-doc-search').value.trim().toLowerCase(),all=referenceCatalog.operators||[];
  const rows=all.map(operator=>{const implementations=Array.isArray(operator.implementations)?operator.implementations:[];const familySearch=[operator.id,operator.name,operator.name_en,operator.name_zh,operator.group,operator.description,referenceSearchText(operator.reference_ids)].join(' ').toLowerCase();const matchingImplementations=query&&!familySearch.includes(query)?implementations.filter(entry=>operatorImplementationSearch(entry).includes(query)):implementations;return {operator,implementations:matchingImplementations,familyMatch:!query||familySearch.includes(query)};}).filter(row=>row.familyMatch||row.implementations.length);
  const groups=new Map();
  for(const row of rows){const group=row.operator.group||'Other';if(!groups.has(group))groups.set(group,[]);groups.get(group).push(row);}
  const shownEntries=new Set(rows.flatMap(row=>row.implementations.map(entry=>entry.id||`${row.operator.id}:${entry.name||entry.name_en||''}`))).size,totalEntries=new Set(all.flatMap(operator=>(operator.implementations||[]).map(entry=>entry.id||`${operator.id}:${entry.name||entry.name_en||''}`))).size;
  status.textContent=`${rows.length} of ${all.length} operator families shown · ${shownEntries} of ${totalEntries} directory entries shown`;
  renderOperatorSummary();
  container.innerHTML=rows.length?[...groups].map(([group,groupRows])=>`<section class="operator-group"><h3>${escape(publicText(group,'Other'))}</h3><div class="documentation-grid">${groupRows.map(({operator,implementations})=>renderOperatorCard(operator,implementations)).join('')}</div></section>`).join(''):'<p class="documentation-empty">No operator family or implementation entry matches this search.</p>';
}
function operatorImplementationSearch(entry) {
  return [entry.id,entry.name,entry.name_en,entry.name_zh,entry.layer,entry.description,entry.inputs,entry.outputs,entry.example,entry.status,referenceSearchText(entry.reference_ids)].join(' ').toLowerCase();
}
function renderOperatorSummary() {
  const target=$('operator-summary');if(!target||!referenceCatalog)return;
  const summary=referenceCatalog.operator_summary||{},families=number(summary.family_count??summary.operator_family_count)??(referenceCatalog.operators||[]).length;
  const entries=number(summary.entry_count??summary.directory_entry_count??summary.implementation_count)??(referenceCatalog.operators||[]).reduce((sum,item)=>sum+(item.implementations?.length||0),0);
  const parts=[['Families',families],['Directory entries',entries],['Mechanisms',summary.mechanism_count],['Configurations',summary.configuration_count],['Compositions',summary.composition_count],['Adapters',summary.adapter_count]];
  target.innerHTML=parts.filter(([,value])=>number(value)!==null).map(([label,value])=>`<div><strong>${countFmt(value)}</strong><span>${escape(label)}</span></div>`).join('');
}
function renderOperatorCard(operator,implementations) {
  const allEntries=Array.isArray(operator.implementations)?operator.implementations:[],name=publicText(operator.name_en??operator.name,operator.id||'Operator family'),nativeName=publicText(operator.name_zh),description=publicText(operator.description,'Description not available.');
  return `<article class="documentation-card operator-card"><div class="documentation-card-head"><div><h4>${escape(name)}</h4>${nativeName&&nativeName!==name?`<span class="documentation-native-name">${escape(nativeName)}</span>`:''}${operator.id&&operator.id!==name?`<span class="documentation-code">${escape(publicText(operator.id))}</span>`:''}</div><span class="documentation-count">${implementations.length}${implementations.length!==allEntries.length?` / ${allEntries.length}`:''} entries</span></div><p class="documentation-description">${escape(description)}</p>${renderReferences(operator.reference_ids)}${implementations.length?`<details class="implementation-details"><summary>Implementation directory (${implementations.length}${implementations.length!==allEntries.length?` matching of ${allEntries.length}`:''})</summary><div class="implementation-list">${implementations.map(renderOperatorImplementation).join('')}</div></details>`:'<p class="reference-empty">No implementation entry matches this search.</p>'}</article>`;
}
function renderOperatorImplementation(entry) {
  const name=publicText(entry.name_en??entry.name,entry.id||'Implementation entry'),nativeName=publicText(entry.name_zh),layer=publicText(entry.layer,'entry'),fields=[['Principle',entry.description],['Input',entry.inputs],['Output',entry.outputs],['Example',entry.example],['Status',entry.status]].filter(([,value])=>publicText(value));
  return `<article class="implementation-entry"><div class="implementation-head"><div><strong>${escape(name)}</strong>${nativeName&&nativeName!==name?`<span class="documentation-native-name">${escape(nativeName)}</span>`:''}${entry.id?`<span class="documentation-code">${escape(publicText(entry.id))}</span>`:''}</div><span class="implementation-layer">${escape(layer)}</span></div>${fields.length?`<dl>${fields.map(([label,value])=>`<div><dt>${escape(label)}</dt><dd>${escape(publicText(value))}</dd></div>`).join('')}</dl>`:''}${entry.reference_ids?.length?renderReferences(entry.reference_ids):''}</article>`;
}
function renderDocumentationViews() { renderMethodDocumentation();renderOperatorDocumentation(); }
async function loadReferenceCatalog() {
  try {
    const response=await fetch('./reference-catalog.json?v=20261011-native-v6',{cache:'no-cache'});
    if(!response.ok)throw new Error(`Reference catalog request returned ${response.status}.`);
    const catalog=await response.json();
    if(!catalog||!Array.isArray(catalog.methods)||!Array.isArray(catalog.operators)||!Array.isArray(catalog.references))throw new Error('Reference catalog has an invalid schema.');
    referenceCatalog=catalog;
  } catch(error) { referenceCatalogError=error instanceof Error?error:new Error('Reference catalog is unavailable.'); }
  if(snapshot)renderDocumentationViews();
}
function datasetKey(entry) { return String(entry?.key??entry?.dataset??''); }
function datasetDisplayName(key) { const entry=datasets.get(key);return entry?.label&&entry.label!==key?`${entry.label} (${key})`:key; }
function datasetEntries() {
  const source=Array.isArray(datasetCatalog?.datasets)&&datasetCatalog.datasets.length?datasetCatalog.datasets:[...(snapshot?.datasets||[])];
  const unique=new Map();for(const entry of source){const key=datasetKey(entry);if(key)unique.set(key,entry);}return [...unique.values()];
}
function datasetCategory(entry) { return String(entry.category_label??entry.category??'Other'); }
function datasetSearchText(entry) { return [datasetKey(entry),entry.label,entry.category,entry.category_label,entry.source,entry.dataset_url,thulabDatasetUrl(entry)?'THULab Hugging Face hosted':''].join(' ').toLowerCase(); }
function humanBytes(value) { const amount=number(value);if(amount===null)return '';const units=['B','KB','MB','GB','TB'];let scaled=amount,index=0;while(scaled>=1000&&index<units.length-1){scaled/=1000;index++;}return `${scaled.toLocaleString('en-US',{maximumFractionDigits:index?1:0})} ${units[index]}`; }
function populateDatasetControls() {
  if(!snapshot)return;
  const entries=datasetEntries(),selected=$('dataset').value||'all',measured=new Set(records.map(record=>record.dataset)),groups=new Map();
  for(const entry of entries){const group=datasetCategory(entry);if(!groups.has(group))groups.set(group,[]);groups.get(group).push(entry);datasets.set(datasetKey(entry),entry);}
  $('dataset').innerHTML='<option value="all">All measured datasets</option>'+[...groups].map(([group,items])=>`<optgroup label="${escape(group)}">${items.sort((a,b)=>datasetKey(a).localeCompare(datasetKey(b))).map(entry=>{const key=datasetKey(entry),label=entry.label&&entry.label!==key?`${entry.label} (${key})`:key;return `<option value="${escape(key)}">${thulabDatasetUrl(entry)?'🏆 ':''}${escape(label)}${measured.has(key)?'':' · No published measurements'}</option>`;}).join('')}</optgroup>`).join('');
  const requestedIsValid=!requestedDatasetApplied&&requestedDataset&&[...$('dataset').options].some(option=>option.value===requestedDataset);
  $('dataset').value=requestedIsValid?requestedDataset:[...$('dataset').options].some(option=>option.value===selected)?selected:'all';
  if(requestedIsValid)requestedDatasetApplied=true;
  const groupSelect=$('dataset-group-filter'),previousGroup=groupSelect.value||'all',categories=[...groups.keys()];
  groupSelect.innerHTML='<option value="all">All groups</option>'+categories.map(group=>`<option value="${escape(group)}">${escape(group)} (${groups.get(group).length})</option>`).join('');
  groupSelect.value=categories.includes(previousGroup)?previousGroup:'all';
}
function renderDatasetCatalog() {
  const container=$('dataset-catalog'),status=$('dataset-catalog-status');if(!container||!status||!snapshot)return;
  const all=datasetEntries(),query=$('dataset-catalog-search').value.trim().toLowerCase(),group=$('dataset-group-filter').value,measured=new Set(records.map(record=>record.dataset));
  const visible=all.filter(entry=>(group==='all'||datasetCategory(entry)===group)&&(!query||datasetSearchText(entry).includes(query))),groups=new Map();
  for(const entry of visible){const category=datasetCategory(entry);if(!groups.has(category))groups.set(category,[]);groups.get(category).push(entry);}
  const measuredVisible=visible.filter(entry=>measured.has(datasetKey(entry))).length,thulabVisible=visible.filter(entry=>thulabDatasetUrl(entry)).length;
  status.textContent=`${visible.length} of ${all.length} datasets shown · ${thulabVisible} THULab-hosted · ${measuredVisible} shown entries have published measurements · rankings use ${measured.size} measured datasets`;
  if(datasetCatalogError&&!all.length){container.innerHTML=`<p class="documentation-error">${escape(datasetCatalogError.message)}</p>`;return;}
  container.innerHTML=visible.length?[...groups].map(([category,entries])=>`<section class="dataset-category"><h3>${escape(category)} <span>${entries.length}</span></h3><div class="dataset-list">${entries.sort((a,b)=>datasetKey(a).localeCompare(datasetKey(b))).map(entry=>renderDatasetEntry(entry,measured.has(datasetKey(entry)))).join('')}</div></section>`).join(''):'<p class="documentation-empty">No dataset matches this search and group filter.</p>';
  for(const button of container.querySelectorAll('[data-dataset-key]'))button.addEventListener('click',()=>{$('dataset').value=button.dataset.datasetKey;updateUrlState();refresh();renderDatasetCatalog();$('leaderboard-wrap').scrollIntoView({block:'nearest'});});
}
function renderDatasetEntry(entry,isMeasured) {
  const key=datasetKey(entry),label=publicText(entry.label,key),rows=number(entry.row_count),payload=humanBytes(entry.size_bytes),metadata=[rows===0?'Empty dataset':rows!==null?`${countFmt(rows)} rows`:'',number(entry.column_count)!==null?`${countFmt(entry.column_count)} columns`:'',number(entry.numeric_column_count)!==null?`${countFmt(entry.numeric_column_count)} numeric`:'',payload?`${payload} numeric payload`:'' ].filter(Boolean),sourceUrl=safeHttpUrl(entry.source_url),datasetUrl=safeHttpUrl(entry.dataset_url),thulabUrl=thulabDatasetUrl(entry),source=publicText(entry.source,'Source'),snapshotDiff=entry.published_result_row_count_matches_catalog===false;
  return `<article class="dataset-entry ${isMeasured?'measured':'pending'} ${state().dataset===key?'selected':''}"><button type="button" data-dataset-key="${escape(key)}"><span><strong>${escape(label)}</strong>${label!==key?`<small>${escape(key)}</small>`:''}</span><span class="dataset-state">${isMeasured?'Measured':'No published measurements'}</span></button>${thulabUrl?`<div class="dataset-affiliations">${thulabDatasetBadge(entry,key)}</div>`:''}${metadata.length?`<p>${metadata.map(escape).join(' · ')}</p>`:''}${snapshotDiff?'<p class="dataset-snapshot-note">Published measurements use an earlier prepared snapshot.</p>':''}<div class="dataset-links">${sourceUrl?`<a href="${escape(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escape(source)} ↗</a>`:entry.source?`<span class="dataset-source">${escape(source)}</span>`:''}${datasetUrl&&datasetUrl!==sourceUrl&&!thulabUrl?`<a href="${escape(datasetUrl)}" target="_blank" rel="noopener noreferrer">Dataset files ↗</a>`:''}</div></article>`;
}
async function loadDatasetCatalog() {
  try {
    const response=await fetch('./dataset-catalog.json?v=20261011-native-v6',{cache:'no-cache'});if(!response.ok)throw new Error(`Dataset catalog request returned ${response.status}.`);
    const catalog=await response.json();if(!catalog||!Array.isArray(catalog.datasets)||!catalog.summary)throw new Error('Dataset catalog has an invalid schema.');datasetCatalog=catalog;
  } catch(error) { datasetCatalogError=error instanceof Error?error:new Error('Dataset catalog is unavailable.'); }
  if(snapshot){populateDatasetControls();renderDatasetCatalog();renderAbout();}
}
function renderAbout() {
  const b=snapshot.benchmark||{},s=snapshot.source||{},c=snapshot.coverage||{};
  const notes=[b.scope_note,b.aggregation_note,b.selection_note,...(b.notes||[]),...(b.measurement_boundaries||[])].filter(Boolean);
  $('methodology').innerHTML=notes.map(n=>`<p class="notice">${escape(n)}</p>`).join('');
  const definitions=b.metric_definitions||{};if(Object.keys(definitions).length)$('methodology').innerHTML+=`<dl class="definitions">${Object.entries(definitions).map(([k,v])=>`<div><dt>${escape(k.replaceAll('_',' '))}</dt><dd>${escape(typeof v==='string'?v:JSON.stringify(v))}</dd></div>`).join('')}</dl>`;
  const exclusions=c.exclusions||c.excluded_datasets||snapshot.exclusions||[];
  const catalogCount=number(datasetCatalog?.summary?.dataset_count??c.catalog_dataset_count)??datasetEntries().length,measuredCount=number(datasetCatalog?.summary?.published_result_dataset_count??datasetCatalog?.summary?.published_measurement_dataset_count??datasetCatalog?.summary?.measured_dataset_count??c.measured_dataset_count)??new Set(records.map(r=>r.dataset)).size;
  const terminal=c.native86_terminal_coverage||{};
  const info={ 'Snapshot exported':snapshot.exported_at||s.exported_at||'Not reported','Dataset catalog':`${catalogCount} datasets; ${measuredCount} currently have published measurements and participate in rankings`,'Native86 server-166 terminal matrix':terminal.expected_pair_count?`${terminal.observed_pair_count} / ${terminal.expected_pair_count} dataset-method pairs; ${terminal.numeric_pair_count??0} contain published numeric aggregates, ${terminal.not_applicable_pair_count??0} are explicit N/A, ${terminal.problem_pair_count??0} ended with a failure status`:'See manifest and terminal coverage download','Dataset scope generated':s.generated_at||'Not reported', 'Source checkout':s.git_commit||'Not reported','Source files':`${s.source_files?.length??0} metadata inputs and ${s.result_csv_files_used?.length??0} result CSVs (hashes in manifest)`,'Result entries across scopes':`${records.length} (scope slices can reuse a measurement)`,'Worktree source state':s.worktree_dirty_inputs?.length?'Includes current local changes; input file hashes identify the published snapshot.':'See manifest for source file hashes.','Runtime environment':b.hardware||'Hardware details are not recorded in this exported result snapshot.','Lossless evidence':b.roundtrip_evidence||'Lossless-design method filter from the source project; per-run round-trip verification is not recorded here.'};
  $('provenance').innerHTML=`<table class="provenance-table"><tbody>${Object.entries(info).map(([k,v])=>`<tr><th>${escape(k)}</th><td class="source-code">${escape(v)}</td></tr>`).join('')}</tbody></table>${exclusions.length?`<h3>Excluded or unavailable results</h3><p class="source-code">${escape(typeof exclusions==='string'?exclusions:JSON.stringify(exclusions))}</p>`:''}`;
}
function init() {
  if(snapshot.compact_records){const t=snapshot.record_tables;snapshot.dataset_results=snapshot.compact_records.map(r=>({dataset:t.datasets[r[0]],scope:t.scopes[r[1]],runtime:t.runtimes[r[2]],comparison_profile:t.comparison_profiles[r[3]],family:t.families[r[4]],method:t.configurations[r[5]],algorithm_variant:t.configurations[r[5]],compression_rate:r[6],compression_time_ns_per_point:r[7],decompression_time_ns_per_point:r[8],original_size_bytes:r[9],compressed_size_bytes:r[10],timing_column_count:r[11],timing_aggregation:t.timing_aggregations[r[12]],eligible_column_set_sha256:t.support_sets[r[13]],measurement_status:t.measurement_statuses[r[14]],timing_profile_status:t.timing_statuses[r[15]],binary_sha256:t.binary_hashes[r[16]]}));delete snapshot.compact_records;}
  for(const m of snapshot.methods||[]) { const method=typeof m==='string'?{name:m}:m;methods.set(method.name??method.method,method);for(const c of method.configurations||[])methods.set(c.algorithm_variant,{...method,name:c.algorithm_variant,family:method.name}); }
  for(const d of snapshot.datasets||[])datasets.set(d.key??d.dataset,d);
  records=(snapshot.dataset_results||snapshot.records||[]).map(normalizeRecord).filter(r=>r.method&&r.dataset&&r.compression_rate>0);
  for(const r of records)if(!methods.has(r.method))methods.set(r.method,{name:r.method,family:r.family});
  const runtimeOrder=['cpp','java','python'];
  const runtimes=[...new Set(records.map(r=>r.runtime))].sort((a,b)=>{const ai=runtimeOrder.indexOf(a),bi=runtimeOrder.indexOf(b);return (ai<0?runtimeOrder.length:ai)-(bi<0?runtimeOrder.length:bi)||a.localeCompare(b);});
  $('runtime').innerHTML=runtimes.map(r=>`<option value="${escape(r)}">${escape(runtimeName(r))}</option>`).join('');
  populateDatasetControls();
  $('family').innerHTML='<option value="all">All method families</option>'+[...new Set(records.map(r=>r.family))].sort().map(f=>`<option value="${escape(f)}">${escape(f)}</option>`).join('');
  $('method-count').textContent=countFmt(new Set(records.map(r=>r.family)).size);$('runtime-count').textContent=runtimes.length;$('record-count').textContent=countFmt(snapshot.coverage?.runtime_configuration_count??new Set(records.map(r=>`${r.runtime}\0${r.method}`)).size);
  const exported=snapshot.exported_at||snapshot.source?.exported_at||'not reported';$('snapshot').textContent=`Result snapshot exported · ${String(exported).replace('T',' ').slice(0,19)}${exported==='not reported'?'':' UTC'}`;$('footer-snapshot').textContent=`Source inputs documented in the manifest`;
  $('column-options').innerHTML=cols.map(c=>`<label><input type="checkbox" data-column="${c.key}" ${c.visible?'checked':''}>${escape(c.label)}</label>`).join('');
  $('weights').innerHTML=WEIGHT_TENTHS.map(w=>`<option value="${w.join(',')}" ${w.join(',')==='4,3,3'?'selected':''}>a = ${(w[0]/10).toFixed(1)} · b = ${(w[1]/10).toFixed(1)} · c = ${(w[2]/10).toFixed(1)}</option>`).join('');
  $('weights').addEventListener('change',refresh);
  for(const input of $('column-options').querySelectorAll('input'))input.addEventListener('change',()=>{cols.find(c=>c.key===input.dataset.column).visible=input.checked;renderTable();});
  for(const id of ['runtime','dtype','precision','family','row-mode','complete-only'])$(id).addEventListener('change',refresh);$('dataset').addEventListener('change',()=>{updateUrlState();refresh();if(activeTab==='dataset')renderDatasetCatalog();});$('search').addEventListener('input',refresh);$('chart-metric').addEventListener('change',renderChart);$('download').addEventListener('click',csv);
  $('reset').addEventListener('click',()=>{$('runtime').value=runtimes.includes('cpp')?'cpp':runtimes[0];$('dtype').value='overall';$('precision').value='all';$('dataset').value='all';$('row-mode').value='family';$('search').value='';$('family').value='all';$('weights').value='4,3,3';$('complete-only').checked=false;selectedMethod=null;$('method-detail').hidden=true;sort={key:'compression_rate',direction:1};setTab(activeTab);});
  for(const tab of document.querySelectorAll('[data-tab]'))tab.addEventListener('click',()=>setTab(tab.dataset.tab));
  $('method-doc-search').addEventListener('input',renderMethodDocumentation);$('operator-doc-search').addEventListener('input',renderOperatorDocumentation);$('dataset-catalog-search').addEventListener('input',renderDatasetCatalog);$('dataset-group-filter').addEventListener('change',renderDatasetCatalog);
  renderAbout();renderDocumentationViews();renderDatasetCatalog();setTab(location.hash.slice(1)||(requestedDataset&&$('dataset').value===requestedDataset?'dataset':'overall'));
  document.documentElement.dataset.ready='true';
}
function updateUrlState() {
  const url=new URL(location.href);url.hash=activeTab;
  const selected=$('dataset')?.value;
  if(selected&&selected!=='all')url.searchParams.set('dataset',selected);else url.searchParams.delete('dataset');
  history.replaceState(null,'',`${url.pathname}${url.search}${url.hash}`);
}
function setupTheme() {
  let stored;try{stored=localStorage.getItem('thulab-theme');}catch{}
  const query=new URLSearchParams(location.search).get('__theme');
  const theme=stored||(['dark','light'].includes(query)?query:matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  document.documentElement.dataset.theme=theme;$('theme').textContent=theme==='dark'?'Light mode':'Dark mode';
  $('theme').addEventListener('click',()=>{const next=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=next;$('theme').textContent=next==='dark'?'Light mode':'Dark mode';try{localStorage.setItem('thulab-theme',next);}catch{}});
}
setupTheme();
void loadReferenceCatalog();
void loadDatasetCatalog();
try { const response=await fetch('./ui-data.json?v=20261011-native-v6',{cache:'no-cache'});if(!response.ok)throw new Error(`Snapshot request returned ${response.status}`);snapshot=await response.json();init(); }
catch(e){$('load-error').hidden=false;$('load-error').textContent=`The result snapshot could not be loaded. Please reload the page or download the JSON from the repository. ${e.message}`;$('snapshot').textContent='Snapshot unavailable';$('leaderboard').querySelector('tbody').innerHTML='<tr><td class="empty-cell">Results unavailable.</td></tr>';}
