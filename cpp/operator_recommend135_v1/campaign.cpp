#include "campaign.hpp"
#include "common.hpp"
#include "model.hpp"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <iostream>
#include <numeric>
#include <random>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>

namespace operator_recommend135_v1 {
namespace {
namespace d=detail;
using Json=nlohmann::json;
using Path=std::filesystem::path;
std::uint64_t median(std::vector<std::uint64_t> v) {
  d::require(!v.empty(),"empty timing samples"); std::sort(v.begin(),v.end()); return v[v.size()/2];
}
Json child(const Path& exe,const std::vector<std::string>& args,const Path& log,int wall_seconds=180) {
  const auto start=d::wall_ns(); pid_t pid=fork(); d::require(pid>=0,"fork failed");
  if(pid==0) {
    rlimit memory{2ULL<<30,2ULL<<30}, core{0,0}, cpu{89,90};
    if(setpgid(0,0)!=0 || setrlimit(RLIMIT_AS,&memory)!=0 || setrlimit(RLIMIT_CORE,&core)!=0 || setrlimit(RLIMIT_CPU,&cpu)!=0)_exit(125);
    int fd=::open(log.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)_exit(124); dup2(fd,STDOUT_FILENO); dup2(fd,STDERR_FILENO); close(fd);
    std::vector<std::string> owned{exe.string()}; owned.insert(owned.end(),args.begin(),args.end());
    std::vector<char*> argv; for(auto& s:owned)argv.push_back(s.data()); argv.push_back(nullptr);
    execv(exe.c_str(),argv.data()); _exit(127);
  }
  setpgid(pid,pid); int status=0; bool timeout=false;
  while(true) {
    auto result=waitpid(pid,&status,WNOHANG); if(result==pid)break;
    if(result<0) { if(errno==EINTR)continue;throw std::runtime_error("waitpid failed"); }
    if(d::wall_ns()-start>std::uint64_t(wall_seconds)*1000000000ULL) {
      timeout=true; d::require(kill(-pid,SIGKILL)==0 || errno==ESRCH,"kill own process group failed");
      while(waitpid(pid,&status,0)<0) { if(errno!=EINTR)throw std::runtime_error("reap timed-out child failed"); }break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return {{"pid",pid},{"parent_pid",getpid()},{"exit_code",WIFEXITED(status)?WEXITSTATUS(status):-1},
          {"signal",WIFSIGNALED(status)?WTERMSIG(status):0},{"timeout",timeout},
          {"cpu_limit_signal",WIFSIGNALED(status)&&WTERMSIG(status)==SIGXCPU},
          {"wall_ns",d::wall_ns()-start},{"argv",args},{"log_path",log.string()}};
}
bool success(const Json& process) { return process.at("exit_code")==0 && !process.at("timeout").get<bool>(); }
void phase(const Path& output,const Json& value) {
  // Diagnostic progress only; final results use durable atomic writes.
  std::ofstream f(output/"PHASE.json",std::ios::trunc);f<<value.dump()<<'\n';
}
}

Block load_input_part(const Json& sample,const std::string& part) {
  const auto& p=sample.at(part); Block b;
  b.dtype=sample.at("dtype").get<std::string>(); b.row_count=p.at("row_count").get<std::uint64_t>();
  d::require(b.row_count<=100000000,"source row count limit");
  b.values=d::read_bytes(p.at("values_path").get<std::string>()); b.missing=d::read_bytes(p.at("missing_path").get<std::string>());
  d::require(d::sha(b.values)==p.at("values_sha256").get<std::string>(),"values source SHA mismatch");
  d::require(d::sha(b.missing)==p.at("missing_sha256").get<std::string>(),"missing source SHA mismatch");
  const std::size_t width=(b.dtype=="int64"||b.dtype=="float64")?8:
                          (b.dtype=="int32"||b.dtype=="float32")?4:0;
  d::require(width!=0 && b.values.size()%width==0,"invalid source dtype/width");
  d::require(b.missing.size()==b.row_count/8+(b.row_count%8!=0),"source mask length mismatch");
  std::uint64_t nulls=0;
  for(std::uint64_t i=0;i<b.row_count;++i)nulls+=(b.missing[i/8]>>(i%8))&1;
  d::require(b.row_count-nulls==b.values.size()/width,"source row/null/valid count mismatch");
  if(p.contains("valid_count"))d::require(p.at("valid_count").get<std::uint64_t>()==b.values.size()/width,"source valid count declaration mismatch");
  if(sample.contains("element_count"))d::require(sample.at("element_count").get<std::uint64_t>()==b.values.size()/width,"sample element count mismatch");
  if(b.row_count%8 && !b.missing.empty())d::require((b.missing.back()>>(b.row_count%8))==0,"nonzero unused source mask bits");
  return b;
}

Json measure_sample(const Json& sample,const Path& output,int repeats) {
  d::require(repeats>=5 && repeats<=25,"requires 5..25 interleaved repeats");
  std::filesystem::create_directories(output); auto head=load_input_part(sample,"head"),tail=load_input_part(sample,"tail");
  auto eligibility_cpu_start=d::cpu_ns(),eligibility_wall_start=d::wall_ns();
  const auto catalog=candidate_catalog();std::vector<std::string> eligible;
  for(const auto& item:catalog.at("candidates")) {
    auto id=item.at("candidate_id").get<std::string>();if(candidate_eligible(id,head))eligible.push_back(id);
  }
  auto eligibility_wall=d::wall_ns()-eligibility_wall_start,eligibility_cpu=d::cpu_ns()-eligibility_cpu_start;
  auto c0=d::cpu_ns(),w0=d::wall_ns();
  auto features=extract_features(head.values,head.dtype,head.missing,head.row_count);
  auto feature_wall=d::wall_ns()-w0,feature_cpu=d::cpu_ns()-c0;
  Json result{{"schema","operator-recommend135-measurements-v1"},{"sample_id",sample.at("sample_id")},
    {"measurement_pid",getpid()},{"catalog_sha256",d::sha(catalog.dump())},{"feature_names",feature_names()},
    {"features",features},{"feature_cpu_ns",feature_cpu},{"feature_wall_ns",feature_wall},
    {"eligibility_cpu_ns",eligibility_cpu},{"eligibility_wall_ns",eligibility_wall},
    {"eligibility_provenance","frozen runtime dtype/shape rule on head, before any tail encode"},
    {"input_bytes",tail.values.size()+tail.missing.size()},{"eligible_candidates",eligible},
    {"candidates",Json::object()},{"timing_contract",{{"warmups",1},{"repeats",repeats},
      {"order","seeded permutation per repetition, all candidates on identical tail"},
      {"cpu_clock","CLOCK_THREAD_CPUTIME_ID"},{"wall_clock","steady_clock"},
      {"timed_boundary","encode_candidate / decode_candidate; input loading, comparisons, fsync, audit excluded"}}}};
  std::vector<std::string> active;
  std::map<std::string,native_graph_v1::Bytes> frames;
  for(const auto& item:catalog.at("candidates")) {
    const auto id=item.at("candidate_id").get<std::string>();
    Json row{{"candidate_id",id},{"kind",item.at("kind")},{"catalog_entry",item},
      {"status","domain_rejected"},{"raw_timings",Json::array()}};
    if(std::find(eligible.begin(),eligible.end(),id)!=eligible.end()) {
      phase(output,{{"phase","warmup"},{"candidate_id",id}});
      try {
        auto audit=inspect_candidate_run(id,tail,true); row["audit"]=audit;
        if(audit.at("execution_status")!="strict_ready") {
          row["status"]=audit.at("execution_status")=="domain_rejected"?"domain_rejected":"failed";
          row["error"]="strict audit refused: "+audit.dump();result["candidates"][id]=std::move(row);continue;
        }
        d::require(!audit.at("fallback_used").get<bool>(),"unexpected harness raw fallback");
        auto frame=encode_candidate(id,tail); auto decoded=decode_candidate(frame);
        d::require(decoded==tail,"warmup bit-exact mismatch");
        row["status"]="measured"; frames[id]=std::move(frame); active.push_back(id);
      } catch(const std::exception& e) { row["status"]="failed";row["error"]=e.what(); }
    } else row["error"]="head dtype/shape is outside frozen candidate domain";
    result["candidates"][id]=std::move(row);
  }
  std::mt19937_64 rng(std::stoull(d::sha(sample.at("sample_id").get<std::string>()).substr(0,16),nullptr,16));
  for(int repeat=0;repeat<repeats;++repeat) {
    std::shuffle(active.begin(),active.end(),rng);
    for(const auto& id:active) {
      auto& row=result["candidates"][id]; if(row.at("status")!="measured")continue;
      phase(output,{{"phase","measure"},{"candidate_id",id},{"repeat",repeat}});
      try {
        auto ec=d::cpu_ns(),ew=d::wall_ns(); auto encoded=encode_candidate(id,tail);
        auto enc_wall=d::wall_ns()-ew,enc_cpu=d::cpu_ns()-ec;
        auto dc=d::cpu_ns(),dw=d::wall_ns(); auto decoded=decode_candidate(encoded);
        auto dec_wall=d::wall_ns()-dw,dec_cpu=d::cpu_ns()-dc;
        d::require(decoded==tail,"timed decode bit-exact mismatch");
        d::require(encoded==frames.at(id),"candidate output changed across repetitions");
        row["raw_timings"].push_back({{"repeat",repeat},{"encode_cpu_ns",enc_cpu},{"decode_cpu_ns",dec_cpu},
          {"encode_wall_ns",enc_wall},{"decode_wall_ns",dec_wall}});
      } catch(const std::exception& e) { row["status"]="failed";row["error"]=e.what(); }
    }
  }
  for(auto& [id,row]:result["candidates"].items()) {
    if(row.at("status")!="measured")continue;
    const auto path=std::filesystem::absolute(output/id/"frame.ngf"); d::write_bytes(path,frames.at(id));
    row["frame_path"]=path.string(); row["frame_sha256"]=d::sha(frames.at(id));row["frame_bytes"]=frames.at(id).size();
    for(const auto* key:{"encode_cpu_ns","decode_cpu_ns","encode_wall_ns","decode_wall_ns"}) {
      std::vector<std::uint64_t> times;for(const auto& t:row["raw_timings"])times.push_back(t.at(key).get<std::uint64_t>());
      row[key]=median(times);
    }
  }
  d::write_json(output/"MEASUREMENTS.json",result); return result;
}

Json decode_to_directory(const Path& frame,const Path& output) {
  auto encoded=d::read_bytes(frame); auto b=decode_candidate(encoded);
  d::write_bytes(output/"values.bin",b.values); d::write_bytes(output/"missing.bin",b.missing);
  Json receipt{{"schema","operator-recommend135-independent-decode-v1"},{"decode_pid",getpid()},
    {"frame_sha256",d::sha(encoded)},{"dtype",b.dtype},{"row_count",b.row_count},
    {"values_sha256",d::sha(b.values)},{"missing_sha256",d::sha(b.missing)},
    {"input_contract","frame bytes only; no source input, candidate ID or inverse sidecar"}};
  d::write_json(output/"DECODE.json",receipt); return receipt;
}

Json verify_decoded(const Json& sample,const Path& decoded) {
  const auto receipt=d::read_json(decoded/"DECODE.json"); const auto& tail=sample.at("tail");
  Json checks{{"values_bit_exact",d::file_sha(decoded/"values.bin")==tail.at("values_sha256").get<std::string>()},
    {"missing_bit_exact",d::file_sha(decoded/"missing.bin")==tail.at("missing_sha256").get<std::string>()},
    {"dtype_exact",receipt.at("dtype")==sample.at("dtype")},{"row_count_exact",receipt.at("row_count")==tail.at("row_count")},
    {"receipt_values_bound",receipt.at("values_sha256").get<std::string>()==d::file_sha(decoded/"values.bin")},
    {"receipt_missing_bound",receipt.at("missing_sha256").get<std::string>()==d::file_sha(decoded/"missing.bin")}};
  bool ok=true; for(const auto& c:checks.items())ok=ok && c.value().get<bool>();
  return {{"verified",ok},{"checks",checks},{"decode",receipt}};
}

Json run_campaign(const Path& input_path,const Path& groups_path,const Path& output,const Path& executable,int repeats) {
  const auto inputs=d::read_json(input_path),groups=d::read_json(groups_path),catalog=candidate_catalog();
  const auto absolute=std::filesystem::absolute(output),exe=std::filesystem::absolute(executable);
  std::filesystem::create_directories(absolute);
  d::require(!std::filesystem::exists(absolute/"CASE_INDEX.json"),"campaign output already has an index; use a new root");
  Json bindings{{"input_sha256",d::file_sha(input_path)},{"groups_sha256",d::file_sha(groups_path)},
    {"executable_sha256",d::file_sha(exe)},{"candidate_catalog_sha256",d::sha(catalog.dump())}};
  d::write_json(absolute/"BINDINGS.json",bindings);d::write_json(absolute/"CANDIDATES.json",catalog);
  Json index=Json::array(),rows=Json::array();std::size_t sample_number=0;
  std::uint64_t failures=0,strict=0,domain=0;
  for(const auto& sample:inputs.at("samples")) {
    const auto id=sample.at("sample_id").get<std::string>(),dataset=sample.at("dataset_id").get<std::string>();
    d::require(groups.at("dataset_to_group").contains(dataset),"missing dataset group: "+dataset);
    const auto sample_root=absolute/"samples"/id; std::filesystem::create_directories(sample_root);
    d::write_json(sample_root/"SAMPLE.json",sample);
    auto proc=child(exe,{"measure-sample",(sample_root/"SAMPLE.json").string(),sample_root.string(),std::to_string(repeats)},sample_root/"measure.log");
    d::write_json(sample_root/"MEASURE_PROCESS.json",proc);
    Json measured;
    if(success(proc)) {
      measured=d::read_json(sample_root/"MEASUREMENTS.json");
      d::require(measured.at("measurement_pid")==proc.at("pid"),"measurement process receipt PID mismatch");
      d::require(measured.at("catalog_sha256")==bindings.at("candidate_catalog_sha256"),"measurement catalog mismatch");
    }
    Json row{{"sample_id",id},{"dataset_id",dataset},{"group_id",groups.at("dataset_to_group").at(dataset)},
      {"scale",sample.at("scale")},{"dtype",sample.at("dtype")},{"metrics",Json::object()}};
    if(!measured.is_null())for(const auto* key:{"features","feature_cpu_ns","feature_wall_ns","eligibility_cpu_ns","eligibility_wall_ns","eligibility_provenance","catalog_sha256","input_bytes","eligible_candidates"})row[key]=measured.at(key);
    for(const auto& candidate:catalog.at("candidates")) {
      const auto cid=candidate.at("candidate_id").get<std::string>(); auto case_root=sample_root/cid;
      std::filesystem::create_directories(case_root);
      Json receipt{{"schema","operator-recommend135-result-v1"},{"sample_id",id},{"dataset_id",dataset},
        {"candidate_id",cid},{"kind",candidate.at("kind")},{"bindings",bindings},{"source",sample.at("tail")},
        {"sample_sha256",d::file_sha(sample_root/"SAMPLE.json")},{"measurement_process",proc},{"status","failed"}};
      Json metric{{"kind",candidate.at("kind")},{"status","failed"}};
      if(!measured.is_null()) {
        const auto& m=measured.at("candidates").at(cid); receipt["measurement"]=m;
        if(m.at("status")=="domain_rejected") { receipt["status"]="domain_rejected";++domain; }
        else if(m.at("status")=="measured") {
          auto decode_process=child(exe,{"decode",m.at("frame_path").get<std::string>(),(case_root/"decoded").string()},case_root/"decode.log",90);
          receipt["independent_decode_process"]=decode_process;
          if(success(decode_process)) {
            auto verified=verify_decoded(sample,case_root/"decoded"); receipt["independent_verification"]=verified;
            if(verified.at("verified").get<bool>() && decode_process.at("pid")!=proc.at("pid") &&
               verified.at("decode").at("decode_pid")==decode_process.at("pid") &&
               verified.at("decode").at("frame_sha256")==m.at("frame_sha256")) {
              d::require(d::file_sha(m.at("frame_path").get<std::string>())==m.at("frame_sha256").get<std::string>(),"frame changed after measurement");
              receipt["status"]="strict_verified"; ++strict;
              for(const auto* key:{"frame_bytes","encode_cpu_ns","decode_cpu_ns","encode_wall_ns","decode_wall_ns"})metric[key]=m.at(key);
            }
          }
        }
      }
      metric["status"]=receipt["status"]; if(receipt["status"]=="failed")++failures;
      row["metrics"][cid]=metric;
      d::write_json(case_root/"RESULT.json",receipt);
      auto hash=d::file_sha(case_root/"RESULT.json");d::write_text(case_root/"RESULT.sha256",hash+"\n");
      index.push_back({{"sample_id",id},{"candidate_id",cid},{"status",receipt.at("status")},
        {"result_path",(case_root/"RESULT.json").string()},{"sha256",hash}});
    }
    if(!measured.is_null())rows.push_back(std::move(row));
    ++sample_number;d::write_json(absolute/"CASE_INDEX.json",index);d::write_json(absolute/"ROWS.json",rows);
    Json progress{{"samples_done",sample_number},{"samples_total",inputs.at("samples").size()},{"strict_verified",strict},
      {"domain_rejected",domain},{"failed",failures},{"last_dataset",dataset},{"last_sample",id}};
    d::write_json(absolute/"PROGRESS.json",progress);std::cout<<progress.dump()<<std::endl;
  }
  Json done{{"schema","operator-recommend135-campaign-complete-v1"},{"status",failures?"completed_with_failures":"completed"},
    {"bindings",bindings},{"samples",sample_number},{"cases",index.size()},{"strict_verified",strict},{"lossless_bypass",0},
    {"domain_rejected",domain},{"failed",failures},{"index_sha256",d::file_sha(absolute/"CASE_INDEX.json")},
    {"rows_sha256",d::file_sha(absolute/"ROWS.json")}};
  d::write_json(absolute/"CAMPAIGN_COMPLETE.json",done); return done;
}
}
