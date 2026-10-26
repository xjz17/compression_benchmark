// Ranking references include all measured methods in the selected runtime,
// scope and row mode. Search, family and coverage filters only hide rows.
export const WEIGHT_TENTHS = [];
for (let a=0;a<=10;a++) for (let b=0;b<=10-a;b++) WEIGHT_TENTHS.push([a,b,10-a-b]);

export function parseWeights(value) {
  const tokens=String(value).split(','),parts=tokens.map(Number);
  if(parts.length!==3||tokens.some(t=>t.trim()==='')||parts.some(n=>!Number.isInteger(n)||n<0)||parts.reduce((s,n)=>s+n,0)!==10) throw new Error('Weights must be non-negative tenths summing to 1.0.');
  return parts.map(n=>n/10);
}

const finite=v=>v!==null&&v!==undefined&&Number.isFinite(v);
const timingAdmitted=r=>Array.isArray(r.source)?r.source.every(timingAdmitted):/^timing_admitted_/i.test(String(r.timing_profile_status??''));
const valid=r=>finite(r.compression_rate)&&r.compression_rate>0&&finite(r.compression_time)&&r.compression_time>=0&&finite(r.decompression_time)&&r.decompression_time>=0&&timingAdmitted(r);
const average=values=>values.length?values.reduce((s,v)=>s+v,0)/values.length:null;

function bounds(rows,key) {
  const values=rows.map(r=>r[key]).filter(finite);
  return values.length?[Math.min(...values),Math.max(...values)]:null;
}

export function minMax(value,range) {
  if(!finite(value)||!range)return null;
  return range[0]===range[1]?0:(value-range[0])/(range[1]-range[0]);
}

function score(row,weights,encodeRange,decodeRange) {
  const encode=minMax(row.compression_time,encodeRange),decode=minMax(row.decompression_time,decodeRange);
  return {normalized_compression_time:encode,normalized_decompression_time:decode,composite_score:valid(row)&&encode!==null&&decode!==null?weights[0]*row.compression_rate+weights[1]*encode+weights[2]*decode:null};
}

export function addRankingMetrics(rows,weights) {
  if(weights.length!==3||weights.some(w=>!finite(w)||w<0)||Math.abs(weights.reduce((s,w)=>s+w,0)-1)>1e-12)throw new Error('Invalid scoring weights.');
  const candidates=rows.filter(r=>valid(r)&&r.source.every(valid));
  const workloadSignature=row=>row.source.map(r=>`${r.dataset}:${r.eligible_column_set_sha256??'legacy_snapshot_unbound'}`).sort().join('|');
  const workloadSignatures=new Set(candidates.map(workloadSignature));
  const globallyComparable=workloadSignatures.size<=1;
  const encodeRange=globallyComparable?bounds(candidates,'compression_time'):null,decodeRange=globallyComparable?bounds(candidates,'decompression_time'):null;
  const byDataset=new Map(),ranks=new Map(),details=new Map();
  for(const row of rows) {
    ranks.set(row.method,[]);
    for(const record of row.source.filter(valid)) {
      const cohort=`${record.dataset}\0${record.eligible_column_set_sha256??'legacy_snapshot_unbound'}`;
      if(!byDataset.has(cohort))byDataset.set(cohort,[]);
      byDataset.get(cohort).push({...record,method:row.method});
    }
  }
  for(const records of byDataset.values()) {
    const encode=bounds(records,'compression_time'),decode=bounds(records,'decompression_time');
    const ordered=records.map(r=>({...r,...score(r,weights,encode,decode)})).sort((a,b)=>a.composite_score-b.composite_score||a.method.localeCompare(b.method));
    for(let start=0;start<ordered.length;) {
      let end=start+1;
      while(end<ordered.length&&ordered[end].composite_score===ordered[start].composite_score)end++;
      const rank=(start+1+end)/2;
      for(let i=start;i<end;i++) {
        ranks.get(ordered[i].method).push(rank);
        details.set(`${ordered[i].method}\0${ordered[i].dataset}\0${ordered[i].eligible_column_set_sha256??'legacy_snapshot_unbound'}`,{dataset_score:ordered[i].composite_score,dataset_rank:rank,dataset_candidate_count:ordered.length});
      }
      start=end;
    }
  }
  return rows.map(row=>({...row,...(globallyComparable&&candidates.includes(row)?score(row,weights,encodeRange,decodeRange):{normalized_compression_time:null,normalized_decompression_time:null,composite_score:null}),average_rank:average(ranks.get(row.method)),rank_coverage:ranks.get(row.method).length,normalization_reference_rows:globallyComparable?candidates.length:0,workload_signature_count:workloadSignatures.size,encode_min:encodeRange?.[0]??null,encode_max:encodeRange?.[1]??null,decode_min:decodeRange?.[0]??null,decode_max:decodeRange?.[1]??null,source:row.source.map(r=>({...r,...details.get(`${row.method}\0${r.dataset}\0${r.eligible_column_set_sha256??'legacy_snapshot_unbound'}`)}))}));
}
