#include "common.hpp"
#include "model.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace ors = operator_recommend135_v1;
namespace d = ors::detail;
using Json = nlohmann::json;
using Path = std::filesystem::path;
using Bytes = native_graph_v1::Bytes;
constexpr std::array<const char*, 4> targets{"space", "encode", "decode", "balanced"};
constexpr const char* oracle_sha256="6adef4003a23ee14253d2c04b844978ec340c056863ebabf61c11ecd632b35ba";

std::uint64_t median(std::vector<std::uint64_t> a) {
  d::require(!a.empty(), "empty timing samples");
  std::sort(a.begin(), a.end()); return a[a.size()/2];
}
struct Time { std::uint64_t cpu=0, wall=0; };
Time now() { return {d::cpu_ns(), d::wall_ns()}; }
Time elapsed(Time t) { return {d::cpu_ns()-t.cpu, d::wall_ns()-t.wall}; }
Json timing(Time t) { return {{"cpu_ns",t.cpu},{"wall_ns",t.wall}}; }
Time middle(const std::vector<Time>& a) {
  std::vector<std::uint64_t> c,w;
  for(auto t:a){c.push_back(t.cpu);w.push_back(t.wall);} return {median(c),median(w)};
}
std::vector<std::string> nonraw(std::vector<std::string> ids) {
  ids.erase(std::remove(ids.begin(),ids.end(),"raw"),ids.end()); return ids;
}
std::vector<std::string> catalog_ids(const Json& c) {
  std::vector<std::string> a; for(const auto& x:c.at("candidates"))a.push_back(x.at("candidate_id"));return a;
}
Json choices(const Json& prediction) {
  Json r=Json::object(); for(auto t:targets) {
    const auto& a=prediction.at("targets").at(t).at("top1");
    d::require(a.size()==1,"target has no scored compression candidate");r[t]=a[0];
  } return r;
}
void compare_predictions(const Json& a,const Json& b) {
  for(auto t:targets)d::require(a.at("targets").at(t)==b.at("targets").at(t),"prepared/reference full ranking mismatch");
  d::require(a.at("serving")==b.at("serving"),"prepared/reference serving mismatch");
}
struct Selection {
  Json prediction, ids; std::vector<double> features; std::vector<std::string> eligible;
  Time feature, eligibility, model, total;
};
Selection select(const ors::Block& b,const ors::PreparedRegistry& registry,const ors::PreparedModel& model) {
  Selection r;auto begin=now(),t=now();
  r.features=ors::extract_features(b.values,b.dtype,b.missing,b.row_count);r.feature=elapsed(t);
  t=now(); auto validated=registry.prepare(b);
  d::require(validated.validation_passes==1,"selector did not validate exactly once");
  r.eligible=nonraw(std::move(validated.candidate_ids));r.eligibility=elapsed(t);
  t=now();r.prediction=model.predict(r.features,r.eligible);r.ids=choices(r.prediction);r.model=elapsed(t);
  r.total=elapsed(begin);return r;
}
Json selection_receipt(const Selection& s) {
  return {{"selected_ids",s.ids},{"feature",timing(s.feature)},{"eligibility",timing(s.eligibility)},
    {"model_and_choices",timing(s.model)},{"total",timing(s.total)},{"validation_passes",1}};
}

// Each fitted model is sealed before any formal column predictions. The source
// label file contains only the existing representative-sample measurements;
// neither the full oracle nor confirm labels are passed to fit_model.
void fit(const Path& rows_path,const Path& split_path,const Path& output) {
  d::require(!std::filesystem::exists(output/"MODELS.json"),"models already frozen; use another output");
  const auto rows=d::read_json(rows_path),split=d::read_json(split_path);
  const auto& assignments=split.at("dataset_assignments");
  std::set<std::string> confirm_legacy_groups;
  for(const auto& row:rows) if(assignments.at(row.at("dataset_id").get<std::string>()).at("split")=="confirm")
    confirm_legacy_groups.insert(row.at("group_id").get<std::string>());
  ors::PreparedRegistry registry;const auto catalog=registry.catalog();
  const Json options{{"candidate_ids",catalog_ids(catalog)},{"catalog_sha256",d::sha(catalog.dump())}};
  Json index{{"schema","operator-prepared135-models-v1"},{"split_sha256",d::file_sha(split_path)},
    {"source_rows_sha256",d::file_sha(rows_path)},{"catalog_sha256",d::sha(catalog.dump())},
    {"confirm_labels_in_fit",false},{"hyperparameters_changed",false},{"training_label_scope","development subset of historical 360 representative head/tail rows, not all 3519 oracle labels"},
    {"confirm_legacy_group_guard",confirm_legacy_groups},{"models",Json::object()}};
  for(int fold=0;fold<6;++fold) {
    const std::string key=fold==5?"confirm":"dev"+std::to_string(fold);
    std::set<std::string> excluded_legacy_groups=confirm_legacy_groups;
    if(fold<5) for(const auto& row:rows) {
      const auto& a=assignments.at(row.at("dataset_id").get<std::string>());
      if(a.at("split")=="development" && a.at("dev_fold")==fold)
        excluded_legacy_groups.insert(row.at("group_id").get<std::string>());
    }
    Json training=Json::array(),training_ids=Json::array();std::set<std::string> source_groups,datasets;
    for(const auto& row:rows) {
      const auto dataset=row.at("dataset_id").get<std::string>();const auto& a=assignments.at(dataset);
      if(a.at("split")!="development" || (fold<5 && a.at("dev_fold")==fold) ||
          excluded_legacy_groups.contains(row.at("group_id").get<std::string>()))continue;
      Json clean=row;clean["group_id"]=a.at("source_identity_sha256");training.push_back(clean);
      training_ids.push_back(row.at("sample_id"));source_groups.insert(a.at("source_identity_sha256"));datasets.insert(dataset);
    }
    d::require(!training.empty(),"no development training rows after source guard");
    auto model=ors::fit_model(training,options);
    model["formal_protocol"]={{"model_key",key},{"split_sha256",index.at("split_sha256")},
      {"confirm_labels_in_fit",false},{"training_sample_ids",training_ids},{"training_dataset_ids",datasets},
      {"training_source_groups",source_groups},{"excluded_legacy_groups",excluded_legacy_groups}};
    const Path file=std::filesystem::absolute(output/("MODEL_"+key+".json"));
    d::write_json(file,model);ors::PreparedModel checked(model);
    index["models"][key]={{"path",file.string()},{"sha256",d::file_sha(file)},
      {"model_json_bytes",std::filesystem::file_size(file)},{"training_row_count",training.size()},
      {"training_sample_ids",training_ids},{"training_dataset_ids",datasets},{"training_source_groups",source_groups},
      {"excluded_legacy_groups",excluded_legacy_groups}};
  }
  d::write_json(output/"MODELS.json",index);
}

Bytes region(const Json& block,const char* path_key,const char* offset_key,const char* length_key,const char* sha_key) {
  auto n=block.at(length_key).get<std::uint64_t>(),offset=block.at(offset_key).get<std::uint64_t>();
  d::require(n<=256ULL*1024*1024,"source block exceeds frozen 256MiB stream bound");
  std::ifstream f(block.at(path_key).get<std::string>(),std::ios::binary);
  d::require(bool(f),"source payload cannot be opened");f.seekg(offset);Bytes bytes(n);
  if(n)f.read(reinterpret_cast<char*>(bytes.data()),n);
  d::require(bool(f),"source payload region truncated");
  d::require(d::sha(bytes)==block.at(sha_key).get<std::string>(),"source payload region SHA mismatch");return bytes;
}
ors::Block load_block(const Json& b) {
  ors::Block r;r.dtype=b.at("dtype");r.row_count=b.at("row_count");
  r.values=region(b,"values_path","values_offset","values_length","values_sha256");
  r.missing=region(b,"missing_bitmap_path","missing_bitmap_offset","missing_bitmap_length","missing_bitmap_sha256");
  return r;
}

// A fresh exec decodes each distinct measured frame. It receives no model,
// source path, source data, or encoder object. Expected hashes are checked only
// after decode_candidate has reconstructed the Block from the frame alone.
int verify_stream() {
  std::string line;std::uint64_t records=0;
  while(std::getline(std::cin,line)) {
    if(line.empty())continue;const Json h=Json::parse(line);
    const auto n=h.at("frame_bytes").get<std::uint64_t>();
    d::require(n<=512ULL*1024*1024,"verify frame bound");Bytes frame(n);
    if(n)std::cin.read(reinterpret_cast<char*>(frame.data()),n);
    d::require(bool(std::cin),"verify frame truncated");
    d::require(d::sha(frame)==h.at("frame_sha256").get<std::string>(),"verify frame SHA mismatch");
    auto decoded=ors::decode_candidate(frame);
    d::require(decoded.dtype==h.at("dtype").get<std::string>() && decoded.row_count==h.at("row_count") &&
      d::sha(decoded.values)==h.at("values_sha256").get<std::string>() && d::sha(decoded.missing)==h.at("missing_sha256").get<std::string>(),"fresh exec bit-exact mismatch");
    std::cout<<Json({{"verified",true},{"record_id",h.at("record_id")},{"frame_sha256",h.at("frame_sha256")},
      {"decode_pid",getpid()},{"frame_only_decode",true},{"record",++records}}).dump()<<'\n'<<std::flush;
  }return 0;
}
class Verifier {
  int input_=-1;FILE* output_=nullptr;pid_t pid_=-1;bool closed_=false;
  void send(std::span<const std::uint8_t> bytes) {
    std::size_t done=0;while(done<bytes.size()) {auto n=::write(input_,bytes.data()+done,bytes.size()-done);
      if(n<0 && errno==EINTR)continue;d::require(n>0,"fresh verifier pipe closed");done+=n;}
  }
 public:
  explicit Verifier(const Path& executable) {
    int in[2],out[2];d::require(pipe(in)==0 && pipe(out)==0,"verifier pipe failed");
    pid_=fork();d::require(pid_>=0,"verifier fork failed");
    if(pid_==0) {
      rlimit mem{2ULL<<30,2ULL<<30},core{0,0},cpu{179,180};
      if(setrlimit(RLIMIT_AS,&mem)||setrlimit(RLIMIT_CORE,&core)||setrlimit(RLIMIT_CPU,&cpu))_exit(125);
      dup2(in[0],STDIN_FILENO);dup2(out[1],STDOUT_FILENO);
      close(in[0]);close(in[1]);close(out[0]);close(out[1]);
      execl(executable.c_str(),executable.c_str(),"--verify-stream",nullptr);_exit(127);
    }
    close(in[0]);close(out[1]);input_=in[1];output_=fdopen(out[0],"r");d::require(output_!=nullptr,"verifier fdopen failed");
  }
  Json verify(const std::string& record,const Bytes& frame,const ors::Block& source) {
    Json header{{"record_id",record},{"frame_bytes",frame.size()},{"frame_sha256",d::sha(frame)},
      {"dtype",source.dtype},{"row_count",source.row_count},{"values_sha256",d::sha(source.values)},
      {"missing_sha256",d::sha(source.missing)}};
    auto line=header.dump()+"\n";send(std::span(reinterpret_cast<const std::uint8_t*>(line.data()),line.size()));send(frame);
    std::string response;int c;while((c=fgetc(output_))!=EOF && c!='\n')response.push_back(char(c));
    d::require(!response.empty(),"fresh verifier returned no receipt");auto result=Json::parse(response);
    d::require(result.at("verified")==true && result.at("decode_pid")==pid_ && result.at("decode_pid")!=getpid() &&
      result.at("record_id")==record && result.at("frame_sha256")==header.at("frame_sha256"),"fresh verifier binding mismatch");
    return result;
  }
  void finish() {
    if(closed_)return;closed_=true;close(input_);input_=-1;fclose(output_);output_=nullptr;
    int status=0;while(waitpid(pid_,&status,0)<0)if(errno!=EINTR)throw std::runtime_error("wait verifier failed");
    d::require(WIFEXITED(status)&&WEXITSTATUS(status)==0,"fresh verifier exited unsuccessfully");
  }
  ~Verifier() {
    if(!closed_) {if(input_>=0)close(input_);if(output_)fclose(output_);if(pid_>0){kill(pid_,SIGTERM);int s;waitpid(pid_,&s,0);}}
  }
};

struct LoadedModel {
  Json original, binding;ors::PreparedModel prepared;Time preparation;
  LoadedModel(Json model,Json b):original(std::move(model)),binding(std::move(b)),prepared(original) {}
};

Json measure_column(const Json& col,const std::string& key,LoadedModel& model,const ors::PreparedRegistry& registry,const Path& executable) {
  const Path mp=col.at("manifest_path").get<std::string>();
  d::require(d::file_sha(mp)==col.at("manifest_sha256").get<std::string>(),"column manifest SHA mismatch");
  const auto manifest=d::read_json(mp);const auto& blocks=manifest.at("blocks");
  d::require(!blocks.empty(),"column has no source blocks");
  d::require(manifest.at("schema")=="native-baseline86-scaled-column-segment-manifest-v1","unexpected source manifest schema");
  d::require(mp.stem().string()==col.at("column_id").get<std::string>(),"manifest filename/column ID mismatch");
  d::require(manifest.at("dataset_id")==col.at("dataset_id"),"manifest dataset mismatch");
  d::require(manifest.at("column_identity").at("dtype")==col.at("dtype") &&
    manifest.at("column_identity").at("column")==col.at("column"),"manifest column identity/dtype mismatch");
  for(const auto& b:blocks)d::require(b.at("dataset_id")==col.at("dataset_id") && b.at("dtype")==col.at("dtype") &&
    b.at("column")==col.at("column"),"source block column identity mismatch");
  std::size_t representative=0;
  while(representative+1<blocks.size() && blocks[representative].at("valid_count")==0)++representative;
  auto first=load_block(blocks.at(representative));auto first_selection=select(first,registry,model.prepared);
  auto reference=ors::predict(model.original,first_selection.features,first_selection.eligible);
  compare_predictions(reference,first_selection.prediction);
  auto plan_start=now();
  Json plan{{"schema","operator-prepared135-column-plan-v1"},{"model_sha256",model.binding.at("sha256")},
    {"catalog_sha256",model.prepared.candidate_catalog_sha256()},{"dtype",col.at("dtype")},
    {"source_manifest_sha256",col.at("manifest_sha256")},{"representative_block_index",representative},
    {"selected_ids",first_selection.ids},{"empty_domain_adaptation","zstd3"}};
  const auto plan_bytes=plan.dump().size();const auto plan_time=elapsed(plan_start);
  Json result{{"schema","operator-prepared135-column-v1"},{"status","running"},{"column",col},{"model_key",key},
    {"model_sha256",model.binding.at("sha256")},{"model_storage_bytes_charged_per_column",model.binding.at("model_json_bytes")},
    {"plan",plan},{"reuse_plan_bytes",plan_bytes},{"initial_plan_construction",timing(plan_time)},
    {"initial_selection",selection_receipt(first_selection)},
    {"initial_full_prediction",first_selection.prediction},{"reference_prepared_full_ranking_equal",true},
    {"blocks",Json::array()},{"encoder_pid",getpid()}};
  Verifier verifier(executable);std::uint64_t total_bytes=0,total_valid=0,verified=0;
  constexpr int repeats=3;
  for(std::size_t bi=0;bi<blocks.size();++bi) {
    const auto& meta=blocks.at(bi);auto source=bi==representative?std::move(first):load_block(meta);
    const auto eligibility=registry.prepare(source);
    d::require(eligibility.block.value_count==meta.at("valid_count"),"source valid_count mismatch");
    d::require(source.values.size()==meta.at("original_numeric_bytes"),"source original byte count mismatch");
    total_bytes+=source.values.size();total_valid+=eligibility.block.value_count;
    Json reused=first_selection.ids;Json adaptations=Json::array();
    for(auto t:targets) {
      auto id=reused.at(t).get<std::string>();
      if(std::find(eligibility.candidate_ids.begin(),eligibility.candidate_ids.end(),id)==eligibility.candidate_ids.end()) {
        d::require(eligibility.block.value_count==0,"nonempty cached plan domain changed");reused[t]="zstd3";adaptations.push_back(t);
      }
    }
    std::map<std::string,Bytes> frames;std::map<std::string,std::vector<Time>> enc,dec;
    std::vector<Time> per_block_total,reuse_total,selection_samples,guard_samples;
    std::map<std::string,std::vector<Time>> goal_e2e_block,goal_e2e_reuse;
    Selection last;Json block_choice;
    // Warm all selected candidates and the two fixed baselines; no trial result
    // influences selection. Candidate frames are deduplicated only for audit.
    auto warm=select(source,registry,model.prepared);block_choice=warm.ids;
    std::set<std::string> ids{"lz4frame","zstd3"};
    for(auto t:targets){ids.insert(block_choice.at(t));ids.insert(reused.at(t));}
    for(const auto& id:ids) {auto f=ors::encode_candidate(id,source);d::require(ors::decode_candidate(f)==source,"warm bit-exact mismatch");frames[id]=std::move(f);}
    for(int repeat=0;repeat<repeats;++repeat) {
      for(int order=0;order<2;++order) {
        const bool reuse=((repeat+order)%2)==1;auto start=now();Time selection_time;
        Json selected;
        if(reuse) {
          auto gt=now();auto check=registry.prepare(source);
          selected=reused;
          for(auto t:targets)d::require(std::find(check.candidate_ids.begin(),check.candidate_ids.end(),selected.at(t).get<std::string>())!=check.candidate_ids.end(),"reuse plan is not eligible");
          selection_time=elapsed(gt);guard_samples.push_back(selection_time);
        } else {
          last=select(source,registry,model.prepared);selected=last.ids;selection_time=last.total;selection_samples.push_back(selection_time);
          d::require(selected==block_choice,"nondeterministic per-block selection");
        }
        std::array<Bytes,4> produced;
        std::array<Time,4> encode_times;
        std::array<std::string,4> produced_ids;
        for(std::size_t ti=0;ti<targets.size();++ti) {
          produced_ids[ti]=selected.at(targets[ti]).get<std::string>();
          auto t=now();produced[ti]=ors::encode_candidate(produced_ids[ti],source);encode_times[ti]=elapsed(t);
        }
        const auto batch_time=elapsed(start);
        (reuse?reuse_total:per_block_total).push_back(batch_time);
        for(std::size_t ti=0;ti<targets.size();++ti) {
          const auto& id=produced_ids[ti];const auto et=encode_times[ti];
          d::require(produced[ti]==frames.at(id),"measured frame differs from frozen selected frame");
          if(!reuse)enc[id].push_back(et);
          else if(std::find(block_choice.begin(),block_choice.end(),Json(id))==block_choice.end())enc[id].push_back(et);
          Time component{et.cpu+selection_time.cpu,et.wall+selection_time.wall};
          (reuse?goal_e2e_reuse:goal_e2e_block)[targets[ti]].push_back(component);
        }
      }
      for(const char* fixed:{"lz4frame","zstd3"}) {
        auto t=now();auto f=ors::encode_candidate(fixed,source);auto et=elapsed(t);
        d::require(f==frames.at(fixed),"fixed frame nondeterministic");enc[std::string("fixed:")+fixed].push_back(et);
      }
      std::vector<std::string> order(ids.begin(),ids.end());if(repeat%2)std::reverse(order.begin(),order.end());
      for(const auto& id:order) {auto t=now();auto decoded=ors::decode_candidate(frames.at(id));auto dt=elapsed(t);
        d::require(decoded==source,"timed decode bit-exact mismatch");dec[id].push_back(dt);}
    }
    Json measured=Json::object();
    for(const auto& [id,frame]:frames) {
      auto witness=verifier.verify(std::to_string(bi)+":"+id,frame,source);++verified;
      Json m{{"frame_bytes",frame.size()},{"frame_sha256",d::sha(frame)},
        {"decode",timing(middle(dec.at(id)))},{"fresh_decode",witness}};
      if(enc.contains(id))m["encode"]=timing(middle(enc.at(id)));
      if(enc.contains("fixed:"+id))m["fixed_encode"]=timing(middle(enc.at("fixed:"+id)));
      measured[id]=std::move(m);
    }
    Json goal_block=Json::object(),goal_reuse=Json::object();
    for(auto t:targets){goal_block[t]=timing(middle(goal_e2e_block[t]));goal_reuse[t]=timing(middle(goal_e2e_reuse[t]));}
    Json per_block_plan=plan;per_block_plan["selected_ids"]=block_choice;
    result["blocks"].push_back({{"block_index",bi},{"block_id",meta.at("block_id")},{"entity_id",meta.at("entity_id")},
      {"source_values_sha256",meta.at("values_sha256")},{"source_missing_sha256",meta.at("missing_bitmap_sha256")},
      {"row_count",source.row_count},{"valid_count",eligibility.block.value_count},{"original_numeric_bytes",source.values.size()},
      {"per_block_selected_ids",block_choice},{"reuse_selected_ids",reused},{"empty_domain_adaptations",adaptations},
      {"per_block_plan_bytes",per_block_plan.dump().size()},{"selection",timing(middle(selection_samples))},
      {"reuse_guard",timing(middle(guard_samples))},{"four_goal_batch_per_block_e2e",timing(middle(per_block_total))},
      {"four_goal_batch_reuse_e2e",timing(middle(reuse_total))},{"per_goal_per_block_component_sum",goal_block},
      {"per_goal_reuse_component_sum",goal_reuse},{"candidates",measured}});
  }
  verifier.finish();
  d::require(total_bytes==col.at("original_numeric_bytes") && total_valid==col.at("valid_value_count"),"whole-column source denominator mismatch");
  result["status"]="strict_verified";result["source_bytes_verified"]=total_bytes;
  result["source_valid_values_verified"]=total_valid;result["independent_frame_decodes"]=verified;
  Json aggregate=Json::object();
  for(const auto* policy:{"per_block","reuse"}) {
    const bool reuse=std::string(policy)=="reuse";
    const auto setup_cpu=reuse?first_selection.total.cpu+plan_time.cpu:0;
    const auto setup_wall=reuse?first_selection.total.wall+plan_time.wall:0;
    std::uint64_t selection_cpu=setup_cpu,selection_wall=setup_wall,batch_cpu=setup_cpu,batch_wall=setup_wall;
    std::uint64_t charged_plans=reuse?plan_bytes:0;
    Json goals=Json::object();
    for(auto t:targets)goals[t]={{"complete_frame_bytes",0ULL},{"encode_component_sum_cpu_ns",setup_cpu},
      {"encode_component_sum_wall_ns",setup_wall},{"decode_cpu_ns",0ULL},{"decode_wall_ns",0ULL}};
    for(const auto& b:result.at("blocks")) {
      const auto& cost=b.at(reuse?"reuse_guard":"selection");
      selection_cpu+=cost.at("cpu_ns").get<std::uint64_t>();selection_wall+=cost.at("wall_ns").get<std::uint64_t>();
      const auto& batch=b.at(reuse?"four_goal_batch_reuse_e2e":"four_goal_batch_per_block_e2e");
      batch_cpu+=batch.at("cpu_ns").get<std::uint64_t>();batch_wall+=batch.at("wall_ns").get<std::uint64_t>();
      if(!reuse)charged_plans+=b.at("per_block_plan_bytes").get<std::uint64_t>();
      for(auto t:targets) {
        const auto id=b.at(reuse?"reuse_selected_ids":"per_block_selected_ids").at(t).get<std::string>();
        const auto& m=b.at("candidates").at(id);const auto& ec=b.at(reuse?"per_goal_reuse_component_sum":"per_goal_per_block_component_sum").at(t);
        auto& g=goals[t];
        for(const auto& [field,value]:std::array<std::pair<const char*,std::uint64_t>,5>{{
          {"complete_frame_bytes",m.at("frame_bytes").get<std::uint64_t>()},
          {"encode_component_sum_cpu_ns",ec.at("cpu_ns").get<std::uint64_t>()},
          {"encode_component_sum_wall_ns",ec.at("wall_ns").get<std::uint64_t>()},
          {"decode_cpu_ns",m.at("decode").at("cpu_ns").get<std::uint64_t>()},
          {"decode_wall_ns",m.at("decode").at("wall_ns").get<std::uint64_t>()}}})g[field]=g.at(field).get<std::uint64_t>()+value;
      }
    }
    const auto storage=model.binding.at("model_json_bytes").get<std::uint64_t>()+charged_plans;
    std::uint64_t containers_cpu=0,containers_wall=0;
    for(auto t:targets) {
      auto packaging_start=now();
      Json container{{"schema","operator-prepared135-column-container-v1"},{"dataset_id",col.at("dataset_id")},
        {"column_id",col.at("column_id")},{"column",col.at("column")},{"dtype",col.at("dtype")},
        {"source_manifest_sha256",col.at("manifest_sha256")},{"policy",policy},{"objective",t},
        {"ordered_frames",Json::array()}};
      for(const auto& b:result.at("blocks")) {
        const auto id=b.at(reuse?"reuse_selected_ids":"per_block_selected_ids").at(t).get<std::string>();
        const auto& frame=b.at("candidates").at(id);
        container["ordered_frames"].push_back({{"block_index",b.at("block_index")},{"block_id",b.at("block_id")},
          {"entity_id",b.at("entity_id")},{"row_count",b.at("row_count")},{"valid_count",b.at("valid_count")},
          {"candidate_id",id},{"frame_bytes",frame.at("frame_bytes")},{"frame_sha256",frame.at("frame_sha256")}});
      }
      auto serialized=container.dump();const auto packaging=elapsed(packaging_start);
      auto& g=goals[t];g["column_container_bytes"]=serialized.size();g["column_container_sha256"]=d::sha(serialized);
      g["column_container"]=std::move(container);g["column_container_construction"]=timing(packaging);
      g["encode_component_sum_cpu_ns"]=g.at("encode_component_sum_cpu_ns").get<std::uint64_t>()+packaging.cpu;
      g["encode_component_sum_wall_ns"]=g.at("encode_component_sum_wall_ns").get<std::uint64_t>()+packaging.wall;
      g["bytes_with_column_container"]=g.at("complete_frame_bytes").get<std::uint64_t>()+serialized.size();
      g["bytes_with_conservative_model_plan_charge"]=g.at("bytes_with_column_container").get<std::uint64_t>()+storage;
      containers_cpu+=packaging.cpu;containers_wall+=packaging.wall;
    }
    aggregate[policy]={{"goals",goals},{"selection_cpu_ns",selection_cpu},{"selection_wall_ns",selection_wall},
      {"four_goal_batch_e2e_cpu_ns",batch_cpu+containers_cpu},{"four_goal_batch_e2e_wall_ns",batch_wall+containers_wall},
      {"initial_selection_and_plan_cpu_ns",setup_cpu},{"initial_selection_and_plan_wall_ns",setup_wall},
      {"plan_storage_bytes",charged_plans},{"model_storage_bytes",model.binding.at("model_json_bytes")}};
  }
  result["whole_column"]=std::move(aggregate);
  result["timing_contract"]={{"warmups",1},{"repeats",repeats},{"all_four_targets_computed",true},
    {"selection_validation_passes",1},{"encoder_and_decoder_keep_their_own_safety_validation",true},
    {"source_load_sha_fsync_verification_excluded",true},{"per_goal_cost","component sum, not a separately timed single-target call"},
    {"four_goal_batch_cost","outer clock around decision/guard and four complete encode calls; includes local frame retention; frame comparisons are outside clock"},
    {"fresh_exec_scope","one new decoder process per column, multiple independent frame-only decode calls"},
    {"reuse_initial_selection_charged_once_per_column","whole_column.reuse includes initial_selection + initial_plan_construction once"},
    {"whole_column_packaging","goal-specific complete column/entity/block/frame index serialized and charged once; its measured construction is added to component sums and whole-column batch totals"}};
  return result;
}

void run(const Path& columns_path,const Path& split_path,const Path& models_path,const Path& output,const Path& executable,std::size_t limit) {
  const auto columns=d::read_json(columns_path),split=d::read_json(split_path),models=d::read_json(models_path);
  d::require(columns.at("schema")=="operator-prepared135-source-columns-v1" &&
    columns.at("oracle_reference_jsonl_sha256")==oracle_sha256,"source columns are not bound to frozen 135/3519 oracle");
  const auto protocol=d::read_json(columns_path.parent_path()/"PROTOCOL.json");
  const bool witness=columns.value("scope",std::string())=="witness";
  d::require(d::file_sha(columns_path)==protocol.at(witness?"witness_columns_sha256":"columns_sha256").get<std::string>(),"source column metadata SHA does not match frozen protocol");
  std::set<std::string> column_keys,dataset_ids;
  for(const auto& c:columns.at("columns")) {
    const auto ds=c.at("dataset_id").get<std::string>(),id=c.at("column_id").get<std::string>();
    d::require(column_keys.insert(ds+":"+id).second,"duplicate source column");dataset_ids.insert(ds);
  }
  d::require(witness?(column_keys.size()==3 && dataset_ids.size()>=2):(column_keys.size()==3519 && dataset_ids.size()==135),"source column coverage differs from declared scope");
  d::require(models.at("split_sha256")==d::file_sha(split_path),"model split SHA mismatch");
  d::require(models.at("confirm_labels_in_fit")==false && models.at("hyperparameters_changed")==false,"formal model training contract rejected");
  std::set<std::string> model_keys;for(const auto& [k,v]:models.at("models").items()){(void)v;model_keys.insert(k);}
  d::require(model_keys==std::set<std::string>({"dev0","dev1","dev2","dev3","dev4","confirm"}),"formal model set is incomplete");
  const auto& assignments=split.at("dataset_assignments");ors::PreparedRegistry registry;
  d::require(models.at("catalog_sha256")==d::sha(registry.catalog().dump()),"model/runtime catalog SHA mismatch");
  Json bindings{{"columns_sha256",d::file_sha(columns_path)},{"split_sha256",d::file_sha(split_path)},
    {"models_sha256",d::file_sha(models_path)},{"binary_sha256",d::file_sha(executable)},
    {"catalog_sha256",models.at("catalog_sha256")},{"NNCP","OFF"},{"excluded_action_ids",{106,107,108,109,110}}};
  std::map<std::string,std::unique_ptr<LoadedModel>> loaded;
  Json preparations=Json::object();
  for(const auto& [key,b]:models.at("models").items()) {
    d::require(d::file_sha(b.at("path").get<std::string>())==b.at("sha256").get<std::string>(),"frozen model bytes changed");
    auto t=now();auto original=d::read_json(b.at("path").get<std::string>());
    const auto& formal=original.at("formal_protocol");
    d::require(formal.at("confirm_labels_in_fit")==false && formal.at("model_key")==key && formal.at("split_sha256")==models.at("split_sha256"),"formal model/index protocol mismatch");
    for(const auto& ds:formal.at("training_dataset_ids")) {
      const auto& assignment=assignments.at(ds.get<std::string>());
      d::require(assignment.at("split")=="development" && (key=="confirm" || assignment.at("dev_fold")!=std::stoi(key.substr(3))),"confirm or held-out development fold entered model fit");
    }
    loaded[key]=std::make_unique<LoadedModel>(std::move(original),b);auto cost=elapsed(t);
    preparations[key]={{"load_parse_prepare",timing(cost)},{"model_sha256",b.at("sha256")}};
    d::require(loaded.at(key)->prepared.candidate_catalog_sha256()==models.at("catalog_sha256"),"prepared model catalog mismatch");
  }
  std::filesystem::create_directories(output);
  if(std::filesystem::exists(output/"BINDINGS.json"))d::require(d::read_json(output/"BINDINGS.json")==bindings,"resume bindings mismatch");
  else d::write_json(output/"BINDINGS.json",bindings);
  d::write_json(output/"MODEL_PREPARATION.json",preparations);
  std::vector<Json> ordered=columns.at("columns").get<std::vector<Json>>();
  std::stable_sort(ordered.begin(),ordered.end(),[&](const Json& a,const Json& b){
    return assignments.at(a.at("dataset_id").get<std::string>()).at("split").get<std::string>()=="development" &&
      assignments.at(b.at("dataset_id").get<std::string>()).at("split").get<std::string>()=="confirm";});
  Json index=Json::array();std::size_t completed=0,failed=0;
  for(const auto& col:ordered) {
    if(limit && completed+failed>=limit)break;
    const auto ds=col.at("dataset_id").get<std::string>(),id=col.at("column_id").get<std::string>();
    const auto& a=assignments.at(ds);const auto key=a.at("split")=="confirm"?"confirm":"dev"+std::to_string(a.at("dev_fold").get<int>());
    const Path dir=output/"columns"/ds/id,file=dir/"RESULT.json";
    try {
      Json r;
      if(std::filesystem::exists(file)) {
        r=d::read_json(file);d::require(r.at("bindings")==bindings && r.at("column")==col && r.at("status")=="strict_verified","invalid existing column receipt");
      } else {
        r=measure_column(col,key,*loaded.at(key),registry,executable);r["bindings"]=bindings;
        r["split"]=a.at("split");r["source_identity_sha256"]=a.at("source_identity_sha256");r["dev_fold"]=a.at("dev_fold");
        d::write_json(file,r);
      }
      ++completed;index.push_back({{"dataset_id",ds},{"column_id",id},{"split",a.at("split")},{"path",file.string()},
        {"sha256",d::file_sha(file)},{"status","strict_verified"},{"independent_frame_decodes",r.at("independent_frame_decodes")}});
    }catch(const std::exception& e) {
      ++failed;Json error{{"dataset_id",ds},{"column_id",id},{"status","failed"},{"error",e.what()},{"bindings",bindings}};
      d::write_json(dir/"FAILED.json",error);index.push_back(error);
    }
    std::ofstream progress(output/"PROGRESS.json");progress<<Json({{"completed",completed},{"failed",failed},{"total",ordered.size()},
      {"last_dataset",ds},{"last_column",id},{"last_split",a.at("split")}}).dump()<<'\n';
    if((completed+failed)%25==0)std::cout<<Json({{"completed",completed},{"failed",failed},{"total",ordered.size()}}).dump()<<std::endl;
  }
  d::write_json(output/"INDEX.json",index);
  d::write_json(output/"COMPLETE.json",{{"schema","operator-prepared135-columns-complete-v1"},{"completed",completed},{"failed",failed},
    {"expected",ordered.size()},{"limited",limit!=0},{"bindings",bindings},{"index_sha256",d::file_sha(output/"INDEX.json")},
    {"status",failed?"failed":(completed==ordered.size()?"complete":"limited_witness")}});
  d::require(failed==0,"one or more whole-column cases failed");
}
int main(int argc,char** argv) {
  signal(SIGPIPE,SIG_IGN);
  try {
    if(argc==2 && std::string(argv[1])=="--verify-stream")return verify_stream();
    d::require(argc>=2,"fit ROWS SPLIT OUTPUT | run COLUMNS SPLIT MODELS OUTPUT [LIMIT]");
    if(std::string(argv[1])=="fit" && argc==5)fit(argv[2],argv[3],argv[4]);
    else if(std::string(argv[1])=="run" && (argc==6||argc==7))
      run(argv[2],argv[3],argv[4],argv[5],std::filesystem::canonical("/proc/self/exe"),argc==7?std::stoull(argv[6]):0);
    else throw std::runtime_error("invalid prepared_columns arguments");
    return 0;
  }catch(const std::exception& e){std::cerr<<Json({{"status","failed"},{"error",e.what()}}).dump()<<std::endl;return 1;}
}
