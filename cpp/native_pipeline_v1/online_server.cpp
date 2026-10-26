// Diagnostic JSONL driver. Only OnlinePipeline::compress's native outer clock
// is compression time; JSON, model loading and verification are outside it.
#include "online.hpp"
#include "stage_dispatch.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace p = wc::pipeline_v1;
namespace g = native_graph_v1;
namespace pol = weighted_grpo::native_policy_v1;
using J = nlohmann::json;
using Clock = std::chrono::steady_clock;
void check(bool v, const char* message) { if (!v) throw std::invalid_argument(message); }
std::uint64_t ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}
g::Bytes unhex(const std::string& s) {
  check(s.size()%2==0 && s.size()<=512ULL*1024*1024,"invalid hex size");
  auto digit=[](char c)->unsigned { if(c>='0'&&c<='9')return c-'0';
    if(c>='a'&&c<='f')return c-'a'+10; throw std::invalid_argument("invalid lowercase hex"); };
  g::Bytes b; b.reserve(s.size()/2);
  for(std::size_t i=0;i<s.size();i+=2)b.push_back((digit(s[i])<<4)|digit(s[i+1])); return b;
}
std::string hex(std::span<const std::uint8_t> b) {
  constexpr char h[]="0123456789abcdef"; std::string s; s.reserve(b.size()*2);
  for(auto v:b){s.push_back(h[v>>4]);s.push_back(h[v&15]);}return s;
}
p::OnlineRequest request(const J& doc) {
  p::OnlineRequest r; r.weights=doc.at("weights").get<std::array<double,3>>();
  r.max_policy_steps=doc.value("max_policy_steps",10U);
  check(doc.at("roots").is_array() && !doc.at("roots").empty() && doc.at("roots").size()<=32,"invalid roots");
  for(const auto& item:doc.at("roots")) {
    const auto k=item.at("kind").get<std::string>();
    check(k=="numeric"||k=="serial"||k=="struct","invalid root kind");
    r.roots.push_back({{k=="numeric"?g::Kind::Numeric:k=="serial"?g::Kind::Serial:g::Kind::Struct,
      item.at("width").get<std::uint32_t>(),unhex(item.at("data_hex"))},
      item.at("original_numeric_payload").get<bool>(),item.at("logical_view").get<std::string>()});
  }
  return r;
}
void verify(const g::Decoded& decoded,const p::OnlineRequest& r) {
  check(decoded.roots.size()==r.roots.size(),"decoded root count changed");
  std::uint64_t raw=0;
  for(std::size_t i=0;i<r.roots.size();++i) {
    const auto& a=decoded.roots[i];const auto& b=r.roots[i];
    check(a.stream.kind==b.stream.kind && a.stream.width==b.stream.width &&
      a.stream.data==b.stream.data && a.logical_view==b.logical_view &&
      a.original_numeric_payload==b.original_numeric_payload,"bit-exact root roundtrip failed");
    if(b.original_numeric_payload)raw+=b.stream.data.size();
  }
  check(decoded.original_numeric_bytes==raw,"decoded numeric denominator changed");
}
J timing(const p::OnlineResult& r) {
  const auto& t=r.timing;
  check(t.total_ns>=t.stage_ns+t.framing_ns,"timing intervals overlap");
  return {{"feature_ns",t.feature_ns},{"mask_ns",t.mask_ns},{"policy_ns",t.policy_ns},
    {"stage_ns",t.stage_ns},{"framing_ns",t.framing_ns},{"total_ns",t.total_ns},
    {"decision_overhead_ns",t.total_ns-t.stage_ns-t.framing_ns},
    {"recipe_ready_ns",t.recipe_ready_ns}};
}
J benchmark(const p::OnlinePipeline& pipeline, const g::StageDispatcher& dispatcher,const J& doc) {
  const auto r=request(doc);const auto warmups=doc.value("warmups",1U),repeats=doc.value("repeats",5U);
  check(warmups<=31 && repeats>0 && repeats<=1001,"invalid repetition count");
  std::vector<p::OnlineResult> samples;std::vector<std::uint64_t> decode_times;g::Bytes expected;
  for(unsigned i=0;i<warmups;++i) {
    auto sample=pipeline.compress(r,dispatcher);
    verify(g::decode(sample.frame,dispatcher,sample.effective_graph_limits),r);
  }
  // Match baseline86: consecutive compression samples, then consecutive decode
  // samples. No decoder changes encoder cache state between measured samples.
  for(unsigned i=0;i<repeats;++i) {
    auto sample=pipeline.compress(r,dispatcher);
    if(i==0)expected=sample.frame;else check(expected==sample.frame,"nondeterministic online frame");
    samples.push_back(std::move(sample));
  }
  for(const auto& sample:samples) {
    const auto begin=Clock::now();auto restored=g::decode(sample.frame,dispatcher,sample.effective_graph_limits);
    decode_times.push_back(ns(begin,Clock::now()));verify(restored,r);
  }
  std::vector<std::size_t> order(samples.size());std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return samples[a].timing.total_ns<samples[b].timing.total_ns;});
  const auto selected=order[order.size()/2];const auto& best=samples[selected];
  auto decode_order=decode_times;std::sort(decode_order.begin(),decode_order.end());
  const auto raw=best.original_numeric_bytes;
  J rows=J::array();for(std::size_t i=0;i<samples.size();++i) {
    auto row=timing(samples[i]);row["decode_ns"]=decode_times[i];rows.push_back(row);
  }
  // Same selected recipe, freshly encoded only as an OFFLINE diagnostic.
  // Never consulted by the online selector, never subtracted from E_total.
  g::Recipe fixed;fixed.policy_registry=p::online_registry_fingerprint;fixed.steps=best.native_steps;
  if(fixed.steps.empty()||fixed.steps.back().kind!=g::StepKind::Stop) {
    g::Step stop;stop.kind=g::StepKind::Stop;stop.policy_action_id=-1;fixed.steps.push_back(stop);
  }
  const auto fixed_result=g::benchmark(r.roots,fixed,dispatcher,warmups,repeats,best.effective_graph_limits);
  check(fixed_result.frame==best.frame,"incremental/fixed-recipe wire mismatch");
  const auto& c=best.counters;
  return {{"schema","online-policy-latency-flow-stop-v2"},
    {"runtime_schema",p::online_runtime_schema},
    {"semantic_mask_contract",p::online_semantic_mask_contract},
    {"diagnostic_only",best.diagnostic_only},
    {"control_schema",p::online_control_schema},{"registry_fingerprint",p::online_registry_fingerprint},
    {"action_count",p::online_action_count},
    {"decode_limits",{{"max_nodes",best.effective_graph_limits.max_nodes},
      {"max_active_streams",best.effective_graph_limits.max_active_streams},
      {"max_ordinary_depth",best.effective_graph_limits.max_ordinary_depth}}},
    {"matched_training",best.matched_training},
    {"matched_max_policy_steps",best.matched_max_policy_steps},
    {"model_sha256",best.model_sha256},
    {"source_checkpoint_sha256",best.source_checkpoint_sha256},
    {"matched_credential_sha256",best.matched_credential_sha256},
    {"complete_online_timing",best.complete_online_timing},
    {"warmups",warmups},{"repeats",repeats},
    {"max_policy_steps",r.max_policy_steps},{"termination",best.termination},
    {"recipe_action_ids",best.recipe_action_ids},{"original_numeric_bytes",raw},
    {"frame_bytes",best.frame.size()},{"frame_hex",hex(best.frame)},
    {"frame_sha256",hex(pol::sha256(best.frame))},{"selected_sample_index",selected},
    {"timing",timing(best)},{"samples",rows},{"decode_ns",decode_order[decode_order.size()/2]},
    {"compression_ratio",raw?J(double(best.frame.size())/raw):J(nullptr)},
    {"compression_ns_per_byte",raw?J(double(best.timing.total_ns)/raw):J(nullptr)},
    {"decompression_ns_per_byte",raw?J(double(decode_order[decode_order.size()/2])/raw):J(nullptr)},
    {"fixed_recipe_encode_ns",fixed_result.encode_nanoseconds},
    {"policy_calls",c.policy_calls},{"selected_policy_actions",c.selected_policy_actions},
    {"semantic_mask_flow_scans",c.semantic_mask_flow_scans},
    {"semantic_mask_value_visits",c.semantic_mask_value_visits},
    {"forward_stage_calls",c.forward_stage_calls},{"candidate_evaluations",c.candidate_evaluations},
    {"replayed_prefix_stages",c.replayed_prefix_stages},
    {"closed_flow_count",c.closed_flow_count},{"forced_flow_stops",c.forced_flow_stops}};
}
int main(int argc,char** argv) try {
  check(argc==4||argc==5,"usage: native_online_policy_server MODEL CATALOG EXPERIMENT_PROFILE [MATCHED_CREDENTIAL]");
  const auto credential=argc==5
      ? std::optional<std::filesystem::path>(argv[4]) : std::nullopt;
  const auto pipeline=p::OnlinePipeline::load(argv[1],argv[2],argv[3],credential);
  const auto dispatcher=g::make_default_dispatcher();std::string line;
  while(std::getline(std::cin,line)) {if(line.empty())continue;J id=nullptr;
    try{auto doc=J::parse(line);id=doc.value("request_id",J(nullptr));
      auto out=benchmark(pipeline,*dispatcher,doc);out["ok"]=true;out["request_id"]=id;
      std::cout<<out.dump()<<'\n'<<std::flush;
    }catch(const std::exception& error){std::cout<<J{{"ok",false},{"request_id",id},{"error",error.what()},
      {"diagnostic_only",pipeline.admission().diagnostic_only},
      {"matched_training",pipeline.admission().matched_training}}.dump()<<'\n'<<std::flush;}
  }return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
