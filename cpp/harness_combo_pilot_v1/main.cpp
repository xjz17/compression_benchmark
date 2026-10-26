#include "runtime.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace p = harness_combo_pilot_v1;
using J = nlohmann::json;
using B = native_graph_v1::Bytes;
using P = std::filesystem::path;
static P executable;
static void need(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
static J readj(const P& p) { std::ifstream f(p); need(bool(f), "open json: " + p.string()); J j; f >> j; return j; }
static void writej(const P& p, const J& j) {
  std::filesystem::create_directories(p.parent_path()); auto t = p; t += ".tmp";
  { std::ofstream f(t); need(bool(f), "write json"); f << j.dump(2) << '\n'; need(bool(f), "json write failed"); }
  std::filesystem::rename(t,p);
}
static B readb(const P& p, std::uint64_t off=0, std::uint64_t n=UINT64_MAX) {
  std::ifstream f(p, std::ios::binary | std::ios::ate); need(bool(f), "open binary: " + p.string());
  auto end=f.tellg(); need(end>=0 && off<=std::uint64_t(end), "binary offset");
  if(n==UINT64_MAX)n=std::uint64_t(end)-off;
  need(n<=128ULL*1024*1024 && n<=std::uint64_t(end)-off, "binary length bound");
  B b(n); f.seekg(off); if(n)f.read(reinterpret_cast<char*>(b.data()),n); need(bool(f), "short binary");return b;
}
static void writeb(const P& p,const B& b) {
  std::filesystem::create_directories(p.parent_path());auto t=p;t+=".tmp";
  {std::ofstream f(t,std::ios::binary);if(!b.empty())f.write(reinterpret_cast<const char*>(b.data()),b.size());need(bool(f),"frame write");}
  std::filesystem::rename(t,p);
}
static std::string sha(std::span<const std::uint8_t> b) {
  std::array<unsigned char,32> h{};unsigned n=0;
  need(EVP_Digest(b.data(),b.size(),h.data(),&n,EVP_sha256(),nullptr)==1 && n==32,"sha256");
  std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto x:h)s<<std::setw(2)<<unsigned(x);return s.str();
}
static std::string filesha(const P& x){return sha(readb(x));}
struct T { std::uint64_t cpu,wall; };
static T now(){timespec c{};clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&c);return {std::uint64_t(c.tv_sec)*1000000000ULL+c.tv_nsec,std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count())};}
static T elapsed(T t){auto a=now();return {a.cpu-t.cpu,a.wall-t.wall};}
static J tj(T t){return {{"cpu_ns",t.cpu},{"wall_ns",t.wall}};}
static std::uint64_t childcpu(){rusage r{};getrusage(RUSAGE_CHILDREN,&r);return std::uint64_t(r.ru_utime.tv_sec+r.ru_stime.tv_sec)*1000000000ULL+std::uint64_t(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1000;}
static std::uint64_t median(std::vector<std::uint64_t> v){need(!v.empty(),"empty median");std::sort(v.begin(),v.end());return v[v.size()/2];}
static unsigned width(const std::string& d){need(d=="int32"||d=="int64"||d=="float32"||d=="float64","dtype");return d=="int32"||d=="float32"?4:8;}
static bool missing(const p::Block& b,std::uint64_t i){return (b.missing.at(i/8)>>(i%8))&1;}
static void validate(const p::Block& b){
  need(b.missing.size()==(b.row_count+7)/8,"bitmap size");std::uint64_t n=0;
  for(auto x:b.missing)n+=std::popcount(x);
  if(b.row_count%8 && !b.missing.empty())need((b.missing.back()>>(b.row_count%8))==0,"bitmap padding");
  need(n<=b.row_count && (b.row_count-n)*width(b.dtype)==b.values.size(),"compact values");
}
static p::Block load(const J& c){
  need(c.at("split")=="development","confirm data prohibited");
  const P path=c.at("manifest_path").get<std::string>();need(filesha(path)==c.at("manifest_sha256").get<std::string>(),"manifest changed");auto m=readj(path);
  need(m.at("schema")=="native-baseline86-scaled-column-segment-manifest-v1" && m.at("dataset_id")==c.at("dataset_id"),"manifest identity/schema");
  need(path.stem()==c.at("column_id").get<std::string>() && m.at("column_identity").at("column")==c.at("column") && m.at("column_identity").at("dtype")==c.at("dtype"),"column identity");
  p::Block out;out.dtype=c.at("dtype");std::uint64_t nr=0;
  for(const auto& b:m.at("blocks")){need(b.at("dtype")==out.dtype,"mixed dtype");nr+=b.at("row_count").get<std::uint64_t>();}
  need(nr<=32ULL*1024*1024,"row limit");out.row_count=nr;out.missing.assign((nr+7)/8,0);std::uint64_t row=0;
  for(const auto& b:m.at("blocks")){
    P vp=b.at("values_path").get<std::string>(),mp=b.at("missing_bitmap_path").get<std::string>();if(vp.is_relative())vp=path.parent_path()/vp;if(mp.is_relative())mp=path.parent_path()/mp;
    auto v=readb(vp,b.at("values_offset").get<std::uint64_t>(),b.at("values_length").get<std::uint64_t>());
    auto mask=readb(mp,b.at("missing_bitmap_offset").get<std::uint64_t>(),b.at("missing_bitmap_length").get<std::uint64_t>());
    need(sha(v)==b.at("values_sha256").get<std::string>() && sha(mask)==b.at("missing_bitmap_sha256").get<std::string>(),"source region sha");
    const auto rows=b.at("row_count").get<std::uint64_t>();need(mask.size()==(rows+7)/8,"source bitmap");
    p::Block part{out.dtype,rows,v,mask};validate(part);const auto valid=v.size()/width(out.dtype);
    need(valid==b.at("valid_count").get<std::uint64_t>() && rows-valid==b.at("missing_count").get<std::uint64_t>(),"source block counts");
    for(std::uint64_t i=0;i<rows;i++)if((mask[i/8]>>(i%8))&1)out.missing[(row+i)/8]|=std::uint8_t(1U<<((row+i)%8));
    row+=rows;out.values.insert(out.values.end(),v.begin(),v.end());
  }
  need(out.values.size()==c.at("original_numeric_bytes") && out.values.size()/width(out.dtype)==c.at("valid_value_count").get<std::uint64_t>(),"column payload length/count");validate(out);return out;
}
static std::vector<std::uint64_t> offsets(const p::Block& b){std::vector<std::uint64_t> a(b.row_count+1);for(std::uint64_t i=0;i<b.row_count;i++)a[i+1]=a[i]+(!missing(b,i));return a;}
static p::Block slice(const p::Block& b,const std::vector<std::uint64_t>& off,std::uint64_t start,std::uint64_t count){
  need(start+count<=b.row_count,"sample range");p::Block r;r.dtype=b.dtype;r.row_count=count;r.missing.assign((count+7)/8,0);
  for(std::uint64_t i=0;i<count;i++)if(missing(b,start+i))r.missing[i/8]|=std::uint8_t(1U<<(i%8));
  r.values.assign(b.values.begin()+off[start]*width(b.dtype),b.values.begin()+off[start+count]*width(b.dtype));validate(r);return r;
}
static std::string key(const J& c){return c.at("dataset_id").get<std::string>()+"__"+c.at("column_id").get<std::string>().substr(0,16);}
static J decoded_info(const B& f){auto b=p::decode(f);validate(b);return {{"decode_pid",getpid()},{"frame_sha256",sha(f)},{"dtype",b.dtype},{"row_count",b.row_count},{"values_sha256",sha(b.values)},{"missing_sha256",sha(b.missing)},{"frame_only_decode",true}};}
static J independent(const P& frame,const p::Block& b){
  int fd[2];need(pipe(fd)==0,"verify pipe");auto pid=fork();need(pid>=0,"verify fork");
  if(pid==0){close(fd[0]);dup2(fd[1],STDOUT_FILENO);close(fd[1]);rlimit lim{2ULL<<30,2ULL<<30},core{0,0},cpu{29,30};setrlimit(RLIMIT_AS,&lim);setrlimit(RLIMIT_CORE,&core);setrlimit(RLIMIT_CPU,&cpu);execl(executable.c_str(),executable.c_str(),"--decode",frame.c_str(),nullptr);_exit(127);}
  close(fd[1]);std::string out;std::array<char,4096> buf{};ssize_t n;while((n=read(fd[0],buf.data(),buf.size()))>0){out.append(buf.data(),n);need(out.size()<65536,"verify output bound");}close(fd[0]);int status=0;need(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,"fresh decoder failed");
  auto r=J::parse(out);need(r.at("decode_pid")==pid && pid!=getpid() && r.at("dtype")==b.dtype && r.at("row_count")==b.row_count && r.at("values_sha256")==sha(b.values) && r.at("missing_sha256")==sha(b.missing),"independent restoration mismatch");r["verified"]=true;return r;
}
struct M { B frame;J result; };
static M measure(const std::string& id,const p::Block& b,unsigned warmups,unsigned repeats){
  M m;std::vector<std::uint64_t> ec,ew,dc,dw;J audit;T all=now();
  for(unsigned i=0;i<warmups+repeats;i++){
    auto t=now();auto e=p::encode(id,b);auto et=elapsed(t);t=now();auto restored=p::decode(e.frame);auto dt=elapsed(t);
    need(restored==b,"same-process bit mismatch");
    if(!m.frame.empty())need(m.frame==e.frame,"nondeterministic frame");
    m.frame=std::move(e.frame);audit=e.audit;
    if(i>=warmups){ec.push_back(et.cpu);ew.push_back(et.wall);dc.push_back(dt.cpu);dw.push_back(dt.wall);}
  }
  m.result={{"candidate_id",id},{"status","ok"},{"strict",audit.at("strict")},{"audit",audit},{"bit_exact",true},{"compressed_bytes",m.frame.size()},{"original_numeric_bytes",b.values.size()},{"compression_ratio",b.values.empty()?J(nullptr):J(double(m.frame.size())/b.values.size())},{"encode",{{"cpu_ns",median(ec)},{"wall_ns",median(ew)},{"cpu_samples_ns",ec},{"wall_samples_ns",ew}}},{"decode",{{"cpu_ns",median(dc)},{"wall_ns",median(dw)},{"cpu_samples_ns",dc},{"wall_samples_ns",dw}}},{"warmups",warmups},{"repeats",repeats},{"measurement_all_iterations",tj(elapsed(all))},{"frame_sha256",sha(m.frame)}};
  return m;
}
static J commit(M& m,const p::Block& b,const P& dir){
  writeb(dir/"frame.ngf",m.frame);auto verify=independent(dir/"frame.ngf",b);need(verify.at("frame_sha256")==m.result.at("frame_sha256"),"committed frame sha mismatch");
  writej(dir/"DECODE.json",verify);m.result["independent_decode"]=verify;m.result["frame_path"]=(dir/"frame.ngf").string();writej(dir/"RESULT.json",m.result);return m.result;
}
static bool stage_matches(const J& c,const std::string& stage){return stage=="all" || c.at("stage")==stage;}
static void validate_plan(const J& plan){
  need(plan.at("selected_count")==plan.at("selections").size(),"plan count");std::set<std::string> keys;std::map<std::string,std::set<std::string>> groups;
  for(const auto& c:plan.at("selections")){
    need(c.at("split")=="development","confirm forbidden");auto f=c.at("dev_fold").get<int>();const std::string stage=c.at("stage");
    need(f>=0 && f<=4 && stage==(f<3?"discovery":f==3?"tuning":"locked_validation"),"fold/stage");
    need(keys.insert(key(c)).second,"duplicate column");groups[stage].insert(c.at("source_group"));width(c.at("dtype"));
  }
  for(const auto& g:groups["discovery"])need(!groups["tuning"].contains(g)&&!groups["locked_validation"].contains(g),"source leakage");
  for(const auto& g:groups["tuning"])need(!groups["locked_validation"].contains(g),"source leakage");
}
static void matrix(const J& plan,const P& out,const std::string& stage){
  const auto catalog=p::catalog();writej(out/"CATALOG.json",catalog);std::uint64_t cases=0,failures=0,bypasses=0;
  for(const auto& c:plan.at("selections"))if(stage_matches(c,stage)){
    auto lt=now();auto b=load(c);auto loadtime=elapsed(lt);const auto dir=out/"matrix"/key(c);need(!std::filesystem::exists(dir/"COLUMN.json"),"matrix column already committed");
    J result={{"column",c},{"source_values_sha256",sha(b.values)},{"source_missing_sha256",sha(b.missing)},{"row_count",b.row_count},{"load",tj(loadtime)},{"results",J::array()},{"ineligible",J::array()}};
    auto ids=p::eligible(b);std::set<std::string> allowed(ids.begin(),ids.end());
    for(const auto& entry:catalog.at("candidates")){
      auto id=entry.at("id").get<std::string>();if(!allowed.contains(id)){result["ineligible"].push_back(id);continue;}
      try{auto m=measure(id,b,1,3);auto r=commit(m,b,dir/id);result["results"].push_back(r);if(!r.at("strict").get<bool>())bypasses++;}
      catch(const std::exception& e){J r={{"candidate_id",id},{"status","error"},{"error",e.what()},{"rank_eligible",false}};writej(dir/id/"FAILED.json",r);result["results"].push_back(r);failures++;}
      cases++;
    }
    writej(dir/"COLUMN.json",result);std::cout<<J({{"event","matrix_column_complete"},{"key",key(c)},{"cases",cases},{"failures",failures}}).dump()<<'\n'<<std::flush;
  }
  writej(out/("MATRIX_"+stage+".json"),{{"status",failures?"complete_with_rejections":"complete"},{"cases",cases},{"failures",failures},{"bypass_cases",bypasses},{"stage",stage}});
}
static void rank(const J& plan,const P& out){
  std::map<std::string,std::map<std::string,std::map<std::string,std::vector<double>>>> values;std::map<std::string,std::size_t> expected;J sources=J::array();
  for(const auto& c:plan.at("selections"))if(c.at("stage")=="discovery"){
    auto a=readj(out/"matrix"/key(c)/"COLUMN.json");double base=0;
    for(const auto& r:a.at("results"))if(r.at("candidate_id")=="zstd3" && r.at("status")=="ok" && r.at("strict")==true)base=r.at("compressed_bytes");
    need(base>0,"discovery baseline missing");
    expected[c.at("dtype").get<std::string>()]++;
    for(const auto& r:a.at("results"))if(r.at("status")=="ok" && r.at("strict")==true)values[c.at("dtype").get<std::string>()][r.at("candidate_id").get<std::string>()][c.at("source_group").get<std::string>()].push_back(double(r.at("compressed_bytes").get<std::uint64_t>())/base);
    sources.push_back({{"key",key(c)},{"dev_fold",c.at("dev_fold")},{"source_group",c.at("source_group")},{"matrix_sha256",filesha(out/"matrix"/key(c)/"COLUMN.json")}});
  }
  J j={{"schema","harness-combo-discovery-prior-v1"},{"training_stage","discovery"},{"confirm_labels_in_fit",false},{"ranking_rule","dtype-specific median of source-group median complete-frame bytes / zstd3; require all discovery columns; deterministic id tie break"},{"sources",sources},{"dtypes",J::object()}};
  for(const auto& [dtype,ids]:values){std::vector<std::pair<double,std::string>> order;
    for(const auto& [id,groups]:ids){std::size_t n=0;std::vector<double> med;for(const auto& [group,v]:groups){(void)group;auto a=v;n+=a.size();std::sort(a.begin(),a.end());med.push_back(a[a.size()/2]);}if(n!=expected.at(dtype))continue;std::sort(med.begin(),med.end());order.push_back({med[med.size()/2],id});}
    std::sort(order.begin(),order.end());J list=J::array();for(const auto& [score,id]:order)list.push_back({{"id",id},{"score",score},{"training_columns",expected.at(dtype)},{"training_source_groups",ids.at(id).size()}});j["dtypes"][dtype]=list;
  }
  need(!std::filesystem::exists(out/"PRIOR.json"),"prior already frozen");writej(out/"PRIOR.json",j);std::cout<<J({{"event","prior_frozen"},{"sha256",filesha(out/"PRIOR.json")}}).dump()<<'\n';
}
static J policy_spec(){return J::array({{{"id","prefix_k8"},{"windows",1},{"valid_values_per_window",1024},{"max_candidates",8}},{{"id","multi3_k4"},{"windows",3},{"valid_values_per_window",1024},{"max_candidates",4}},{{"id","multi3_k8"},{"windows",3},{"valid_values_per_window",1024},{"max_candidates",8}},{{"id","multi3_k16"},{"windows",3},{"valid_values_per_window",1024},{"max_candidates",16}},{{"id","prefix_k8_n32768"},{"windows",1},{"valid_values_per_window",32768},{"max_candidates",8}},{{"id","multi3_k8_n8192"},{"windows",3},{"valid_values_per_window",8192},{"max_candidates",8}},{{"id","multi3_k8_n32768"},{"windows",3},{"valid_values_per_window",32768},{"max_candidates",8}},{{"id","multi3_k16_n32768"},{"windows",3},{"valid_values_per_window",32768},{"max_candidates",16}}});}
static std::string minimum_feasible(const J& stats){
  std::string chosen;std::uint64_t best=UINT64_MAX,best_cpu=UINT64_MAX;
  for(auto it=stats.begin();it!=stats.end();++it){const auto& r=it.value();if(!r.at("feasible").get<bool>())continue;auto bytes=r.at("bytes").get<std::uint64_t>(),cpu=r.at("online_total_cpu_ns").get<std::uint64_t>();
    if(bytes<best || (bytes==best && (cpu<best_cpu || (cpu==best_cpu && (chosen.empty()||it.key()<chosen))))){chosen=it.key();best=bytes;best_cpu=cpu;}}
  return chosen.empty()?"prior_top1":chosen;
}
static void select_policy(const J& plan,const P& out){
  std::vector<std::string> names{"fixed_zstd3","fixed_lz4","prior_top1"};for(const auto& s:policy_spec())names.push_back(s.at("id"));
  J stats=J::object(),inputs=J::array();for(const auto& name:names){std::uint64_t bytes=0,cpu=0;std::map<std::string,std::pair<double,double>> groups;
    for(const auto& c:plan.at("selections"))if(c.at("stage")=="tuning"){
      auto path=out/"online"/key(c)/name/"RESULT.json";auto r=readj(path);auto base=readj(out/"online"/key(c)/"fixed_zstd3"/"RESULT.json");need(r.at("selected").at("strict")==true,"non-strict tune");
      bytes+=r.at("selected").at("compressed_bytes").get<std::uint64_t>();cpu+=r.at("online_total_cpu_ns").get<std::uint64_t>();auto& g=groups[c.at("source_group").get<std::string>()];g.first+=r.at("selected").at("compressed_bytes").get<double>();g.second+=base.at("selected").at("compressed_bytes").get<double>();inputs.push_back({{"path",path.string()},{"sha256",filesha(path)}});
    }
    double worst=0;for(const auto& [g,v]:groups){(void)g;worst=std::max(worst,v.first/v.second);}stats[name]={{"bytes",bytes},{"online_total_cpu_ns",cpu},{"worst_source_byte_ratio_to_zstd3",worst}};
  }
  const double cpu_cap=3.0*stats.at("fixed_zstd3").at("online_total_cpu_ns").get<double>();const double risk_cap=stats.at("prior_top1").at("worst_source_byte_ratio_to_zstd3").get<double>();
  for(const auto& name:names){auto& r=stats[name];r["feasible"]=r.at("online_total_cpu_ns").get<double>()<=cpu_cap && r.at("worst_source_byte_ratio_to_zstd3").get<double>()<=risk_cap;}
  const auto chosen=minimum_feasible(stats);
  need(!std::filesystem::exists(out/"POLICY_LOCK.json"),"policy already frozen");writej(out/"POLICY_LOCK.json",{{"selected_policy",chosen},{"selection_stage","tuning fold3 only"},{"criterion","minimum total complete-frame bytes, then CPU, under CPU and worst-source guards; prior_top1 is explicit fallback if no feasible strategy"},{"cpu_cap_relative_to_fixed_zstd3",3.0},{"worst_source_cap_relative_to_prior_top1",risk_cap},{"stats",stats},{"inputs",inputs},{"prior_sha256",filesha(out/"PRIOR.json")},{"confirm_used",false}});
}
static void online(const J& plan,const P& out,const std::string& stage){
  auto prior=readj(out/"PRIOR.json");auto prior_sha=filesha(out/"PRIOR.json");writej(out/"POLICIES.json",{{"policies",policy_spec()},{"selection","sum complete sample frame bytes / sum valid numeric sample bytes across nonoverlapping valid-value windows; candidate id tie break"},{"budget","hard candidate count, each candidate evaluated on all prescribed sample windows; includes two mandatory baselines"},{"timing","report actual probe encode/decode, final encode/decode and end-to-end total; no hard wall-time budget claim"},{"fit_scope","discovery folds 0-2; policy selected on fold3 only"},{"no_confirm",true}});
  if(std::filesystem::exists(out.parent_path()/"PROTOCOL.json")){auto policy=readj(out/"POLICIES.json");policy["design_history"]=readj(out.parent_path()/"PROTOCOL.json");writej(out/"POLICIES.json",policy);}
  auto specs=policy_spec();if(stage=="locked_validation"){auto lock=readj(out/"POLICY_LOCK.json");need(lock.at("prior_sha256")==prior_sha,"prior changed after policy lock");J chosen=J::array();for(const auto& s:specs)if(s.at("id")==lock.at("selected_policy"))chosen.push_back(s);specs=std::move(chosen);}
  for(const auto& c:plan.at("selections"))if(stage_matches(c,stage)){
    auto b=load(c);auto outer=out/"online"/key(c);need(!std::filesystem::exists(outer/"COLUMN.json"),"online column already committed");J records=J::array();
    auto ids=p::eligible(b);std::set<std::string> eligible(ids.begin(),ids.end());std::vector<std::string> order;
    need(prior.at("dtypes").contains(b.dtype),"missing dtype prior");for(const auto& row:prior.at("dtypes").at(b.dtype))if(eligible.contains(row.at("id").get<std::string>()))order.push_back(row.at("id"));need(!order.empty(),"empty prior");
    auto baselines=[&](){for(const auto& baseline:std::array<std::string,3>{"fixed_zstd3","fixed_lz4","prior_top1"}){
      const std::string id=baseline=="fixed_zstd3"?"zstd3":baseline=="fixed_lz4"?"lz4":order.front();
      auto begin=now();auto cc=childcpu();auto m=measure(id,b,0,1);auto r=commit(m,b,outer/baseline/"selected");auto total=elapsed(begin);
      auto path=outer/baseline/"RESULT.json";writej(path,{{"column",c},{"selected_id",id},{"selected",r},{"prior_sha256",prior_sha},{"online_total_cpu_ns",total.cpu+childcpu()-cc},{"online_total_wall_ns",total.wall},{"probe_count",0},{"baseline",baseline=="prior_top1"?"discovery-only dtype-specific fixed recipe, not historical PreparedModel":"fixed codec"}});records.push_back({{"policy_id",baseline},{"path",path.string()},{"sha256",filesha(path)}});
    }};
    const bool baseline_first=c.at("selection_index").get<unsigned>()%2==0;if(baseline_first)baselines();
    for(const auto& spec:specs){
      const auto pid=spec.at("id").get<std::string>();auto begin=now();auto child_begin=childcpu();auto sample_begin=now();auto off=offsets(b);
      const auto windows=spec.at("windows").get<std::uint64_t>();const auto n=std::min<std::uint64_t>(spec.at("valid_values_per_window").get<std::uint64_t>(),off.back()/windows);need(n>0,"not enough valid probe values");std::vector<std::uint64_t> valid_starts{0};
      if(windows==3){valid_starts.push_back((off.back()-n)/2);valid_starts.push_back(off.back()-n);}std::vector<std::uint64_t> starts;std::vector<p::Block> probes;
      for(auto vs:valid_starts){auto s=std::uint64_t(std::upper_bound(off.begin(),off.end(),vs)-off.begin()-1);auto e=std::uint64_t(std::lower_bound(off.begin(),off.end(),vs+n)-off.begin());starts.push_back(s);probes.push_back(slice(b,off,s,e-s));}auto sample_time=elapsed(sample_begin);
      std::vector<std::string> proposed{"zstd3","lz4"};for(const auto& id:order)if(std::find(proposed.begin(),proposed.end(),id)==proposed.end() && proposed.size()<spec.at("max_candidates").get<std::size_t>())proposed.push_back(id);
      J trials=J::array();std::string winner="zstd3";double best=1e300;std::uint64_t strict_trials=0,probe_encode_cpu=0,probe_decode_cpu=0;auto probe_begin=now();bool any_feasible=false;
      for(const auto& id:proposed){double score=0;bool feasible=true;J trials_for_id=J::array();
        for(std::size_t w=0;w<probes.size();w++){
          if(probes[w].values.empty()){trials_for_id.push_back({{"window",w},{"status","empty_sample"}});feasible=false;continue;}
          try{auto m=measure(id,probes[w],0,1);auto r=m.result;r["window"]=w;r["start_row"]=starts[w];trials_for_id.push_back(r);if(!r.at("strict").get<bool>())feasible=false;else strict_trials++;score+=double(m.frame.size());probe_encode_cpu+=r.at("encode").at("cpu_ns").get<std::uint64_t>();probe_decode_cpu+=r.at("decode").at("cpu_ns").get<std::uint64_t>();}
          catch(const std::exception& e){feasible=false;trials_for_id.push_back({{"window",w},{"status","error"},{"error",e.what()}});}
        }
        std::uint64_t sample_bytes=0;for(const auto& pb:probes)sample_bytes+=pb.values.size();score/=sample_bytes;trials.push_back({{"candidate_id",id},{"feasible",feasible},{"score",feasible?J(score):J(nullptr)},{"windows",trials_for_id}});
        if(feasible && (score<best || (score==best && id<winner))){best=score;winner=id;any_feasible=true;}
      }
      auto probe_time=elapsed(probe_begin);auto final_begin=now();bool fallback=!any_feasible;std::string selected=winner;J r;
      try{auto m=measure(winner,b,0,1);need(m.result.at("strict")==true,"full candidate used bypass");r=commit(m,b,outer/pid/"selected");}
      catch(const std::exception& e){fallback=true;writej(outer/pid/"REJECTED.json",{{"candidate_id",winner},{"error",e.what()}});selected="zstd3";auto m=measure(selected,b,0,1);need(m.result.at("strict")==true,"baseline not strict");r=commit(m,b,outer/pid/"selected");}
      auto final_time=elapsed(final_begin);auto total=elapsed(begin);auto child=childcpu()-child_begin;
      J result={{"policy",spec},{"column",c},{"prior_sha256",prior_sha},{"proposed_ids",proposed},{"proposed_candidate_count",proposed.size()},{"window_start_rows",starts},{"window_start_valid_values",valid_starts},{"valid_values_per_window",n},{"strict_trial_count",strict_trials},{"trials",trials},{"sample_prepare",tj(sample_time)},{"probe_all",tj(probe_time)},{"probe_encode_cpu_ns",probe_encode_cpu},{"probe_decode_verification_cpu_ns",probe_decode_cpu},{"chosen_from_probes",winner},{"selected_id",selected},{"fallback",fallback},{"no_feasible_probe_candidate",!any_feasible},{"selected",r},{"final_encode_decode_commit_verify",tj(final_time)},{"online_total_parent",tj(total)},{"independent_decoder_child_cpu_ns",child},{"online_total_cpu_ns",total.cpu+child},{"online_total_wall_ns",total.wall},{"source_file_io_excluded",true},{"parent_startup_excluded",true},{"verifier_startup_included",true},{"baseline_first_for_column",baseline_first}};
      writej(outer/pid/"RESULT.json",result);records.push_back({{"policy_id",pid},{"path",(outer/pid/"RESULT.json").string()},{"sha256",filesha(outer/pid/"RESULT.json")}});
    }
    if(!baseline_first)baselines();
    writej(outer/"COLUMN.json",{{"column",c},{"records",records},{"prior_sha256",prior_sha},{"status","complete"}});std::cout<<J({{"event","online_column_complete"},{"key",key(c)},{"stage",stage}}).dump()<<'\n'<<std::flush;
  }
  writej(out/("ONLINE_"+stage+".json"),{{"status","complete"},{"stage",stage},{"prior_sha256",prior_sha}});
}
static int selftest(const P& out){
  J selection_fixture={{"prior_top1",{{"bytes",1},{"online_total_cpu_ns",1000},{"feasible",false}}},{"feasible_larger",{{"bytes",10},{"online_total_cpu_ns",5},{"feasible",true}}}};
  need(minimum_feasible(selection_fixture)=="feasible_larger","infeasible low-byte prior must not beat feasible larger result");selection_fixture["feasible_larger"]["feasible"]=false;need(minimum_feasible(selection_fixture)=="prior_top1","explicit no-feasible fallback");
  // Exercise the production loader across a non-byte-aligned source boundary.
  auto fixture=out/"loader_fixture";J blocks=J::array();p::Block expected;expected.dtype="int64";expected.row_count=12;expected.missing={0x92,0x00};
  for(unsigned i=0;i<2;i++){
    B v;unsigned count=i?6:3;for(unsigned k=0;k<count;k++){std::uint64_t x=10*i+k;for(unsigned z=0;z<8;z++)v.push_back(std::uint8_t(x>>(8*z)));}B mask{std::uint8_t(i?0x04:0x12)};
    auto vn="values"+std::to_string(i)+".bin",mn="mask"+std::to_string(i)+".bin";writeb(fixture/vn,v);writeb(fixture/mn,mask);expected.values.insert(expected.values.end(),v.begin(),v.end());
    blocks.push_back({{"dtype","int64"},{"row_count",i?7:5},{"valid_count",count},{"missing_count",i?1:2},{"values_path",vn},{"values_offset",0},{"values_length",v.size()},{"values_sha256",sha(v)},{"missing_bitmap_path",mn},{"missing_bitmap_offset",0},{"missing_bitmap_length",mask.size()},{"missing_bitmap_sha256",sha(mask)}});
  }
  auto mp=fixture/"loader_fixture.json";writej(mp,{{"schema","native-baseline86-scaled-column-segment-manifest-v1"},{"dataset_id","synthetic_loader_fixture"},{"column_identity",{{"column","loader_fixture"},{"dtype","int64"}}},{"blocks",blocks}});
  J fc={{"split","development"},{"dtype","int64"},{"dataset_id","synthetic_loader_fixture"},{"column","loader_fixture"},{"column_id","loader_fixture"},{"manifest_path",mp.string()},{"manifest_sha256",filesha(mp)},{"original_numeric_bytes",expected.values.size()},{"valid_value_count",9}};need(load(fc)==expected,"cross-block loader fixture failed");
  J rows=J::array();unsigned cases=0,failures=0;for(const auto& dtype:std::array<std::string,4>{"int32","int64","float32","float64"}){
    for(unsigned shape=0;shape<5;shape++){
      p::Block b;b.dtype=dtype;b.row_count=shape==0?0:shape==1?1:shape==2?17:shape==3?33:1025;b.missing.assign((b.row_count+7)/8,0);
      for(std::uint64_t i=0;i<b.row_count;i++){
        bool miss=shape==2 || (shape==3 && i%3==0);if(miss){b.missing[i/8]|=std::uint8_t(1U<<(i%8));continue;}
        std::uint64_t bits=0;
        if(dtype=="float32"){const std::array<std::uint32_t,9>a{0,0x80000000U,0x7fc01234U,0x7f800000U,0xff800000U,1,0x007fffffU,0x3f800000U,0x7f7fffffU};bits=a[i%a.size()];}
        else if(dtype=="float64"){const std::array<std::uint64_t,9>a{0,0x8000000000000000ULL,0x7ff8000000001234ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL,1,0xfffffffffffffULL,0x3ff0000000000000ULL,0x7fefffffffffffffULL};bits=a[i%a.size()];}
        else{const std::array<std::uint64_t,8>a{0,1,UINT64_MAX,0x7fffffffffffffffULL,0x8000000000000000ULL,127,128,255};bits=a[i%a.size()];}
        for(unsigned k=0;k<width(dtype);k++)b.values.push_back(std::uint8_t(bits>>(k*8)));
      }
      validate(b);for(const auto& id:p::eligible(b)){
        cases++;try{auto m=measure(id,b,0,1);auto r=commit(m,b,out/(dtype+"_"+std::to_string(shape))/id);rows.push_back({{"dtype",dtype},{"shape",shape},{"id",id},{"status","ok"},{"strict",r.at("strict")}});}
        catch(const std::exception& e){rows.push_back({{"dtype",dtype},{"shape",shape},{"id",id},{"status","error"},{"error",e.what()}});failures++;}
      }
    }
  }
  writej(out/"SELFTEST.json",{{"cases",cases},{"failures",failures},{"rows",rows},{"status",failures?"failed":"passed"}});std::cout<<J({{"cases",cases},{"failures",failures}}).dump()<<'\n';return failures?1:0;
}
int main(int argc,char** argv){
  try{executable=std::filesystem::canonical("/proc/self/exe");need(argc>=2,"mode required");std::string mode=argv[1];
    if(mode=="--decode"){need(argc==3,"decode args");std::cout<<decoded_info(readb(argv[2])).dump()<<'\n';return 0;}
    if(mode=="--catalog"){std::cout<<p::catalog().dump(2)<<'\n';return 0;}
    if(mode=="--selftest"){need(argc==3,"selftest args");return selftest(argv[2]);}
    need(argc>=4,"mode PLAN OUTPUT [stage]");auto plan=readj(argv[2]);validate_plan(plan);P out=std::filesystem::absolute(argv[3]);std::string stage=argc>4?argv[4]:"all";
    if(mode=="--matrix" || mode=="--online"){
      need(stage=="discovery"||stage=="tuning"||stage=="locked_validation","explicit valid stage required");std::size_t n=0;for(const auto& c:plan.at("selections"))if(stage_matches(c,stage))n++;need(n>0,"empty stage");
      if(mode=="--online")need(stage!="discovery","discovery is fit scope only");
      if(mode=="--matrix" && stage!="discovery")need(std::filesystem::exists(out/("ONLINE_"+stage+".json")),"online choices must precede same-pool oracle");
      if(stage=="locked_validation")need(std::filesystem::exists(out/"POLICY_LOCK.json"),"policy lock required before validation");
    }
    if(mode=="--matrix")matrix(plan,out,stage);else if(mode=="--rank")rank(plan,out);else if(mode=="--select-policy")select_policy(plan,out);else if(mode=="--online")online(plan,out,stage);else need(false,"unknown mode");return 0;
  }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
