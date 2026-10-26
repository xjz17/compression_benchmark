// JSON is only the untimed driver protocol. All graph execution and timing is
// native and each sample creates/parses a complete NGF1 representation.
#include "stage_dispatch.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace native_graph_v1;
using J=nlohmann::json;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
Bytes unhex(const std::string& s) {
  require(s.size()%2==0 && s.size()<=2ULL*(1<<28),"invalid/oversized hex");
  auto digit=[](char c)->unsigned {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::invalid_argument("noncanonical hex");};
  Bytes b;b.reserve(s.size()/2);for(std::size_t i=0;i<s.size();i+=2)b.push_back((digit(s[i])<<4)|digit(s[i+1]));return b;
}
std::string hex(std::span<const std::uint8_t> b){static const char* h="0123456789abcdef";std::string s;s.reserve(b.size()*2);for(auto c:b){s.push_back(h[c>>4]);s.push_back(h[c&15]);}return s;}
Kind kind(const std::string& s){if(s=="serial")return Kind::Serial;if(s=="numeric")return Kind::Numeric;if(s=="struct")return Kind::Struct;throw std::invalid_argument("unknown flow kind");}
const char* kind(Kind k){return k==Kind::Serial?"serial":k==Kind::Numeric?"numeric":"struct";}
Limits decode_limits(const J& req) {
 Limits out;
 if(!req.contains("limits"))return out;
 const auto& v=req.at("limits");require(v.is_object(),"decode limits must be an object");
 out.max_nodes=v.value("max_nodes",out.max_nodes);
 out.max_active_streams=v.value("max_active_streams",out.max_active_streams);
 out.max_ordinary_depth=v.value("max_ordinary_depth",out.max_ordinary_depth);
 out.max_stream_bytes=v.value("max_stream_bytes",out.max_stream_bytes);
 out.max_working_bytes=v.value("max_working_bytes",out.max_working_bytes);
 out.max_frame_bytes=v.value("max_frame_bytes",out.max_frame_bytes);
 out.max_metadata_bytes=v.value("max_metadata_bytes",out.max_metadata_bytes);
 return out;
}
std::vector<RootInput> roots(const J& j){std::vector<RootInput> out;require(j.is_array()&&!j.empty()&&j.size()<=32,"invalid roots");for(const auto& row:j)out.push_back({{kind(row.at("kind").get<std::string>()),row.at("width"),unhex(row.at("data_hex"))},row.at("original_numeric_payload"),row.at("logical_view")});return out;}
Recipe recipe(const J& j){Recipe r;r.policy_registry=j.at("policy_registry");const auto& steps=j.at("steps");require(steps.is_array()&&steps.size()<=65,"invalid recipe size");for(const auto& row:steps){Step s;const auto name=row.at("kind").get<std::string>();if(name=="stage")s.kind=StepKind::Stage;else if(name=="split_segments")s.kind=StepKind::SplitSegments;else if(name=="concat")s.kind=StepKind::Concat;else if(name=="stop")s.kind=StepKind::Stop;else throw std::invalid_argument("unknown step kind");s.inputs=row.value("inputs",std::vector<StreamId>{});s.segment_sizes=row.value("segment_sizes",std::vector<std::uint64_t>{});s.policy_action_id=row.value("policy_action_id",std::int64_t(-1));if(row.contains("stage")){s.stage.action=row.at("stage").at("action");s.stage.profile=row.at("stage").value("profile",std::string{});s.stage.parameters=row.at("stage").value("parameters",std::map<std::string,std::string>{});}r.steps.push_back(std::move(s));}return r;}
}
int main(int argc,char** argv){try{
 require(argc==3,"usage: native_graph_request_cli encode|benchmark|decode request.json");
 std::ifstream in(argv[2],std::ios::binary|std::ios::ate);require(bool(in),"request read failed");auto size=in.tellg();require(size>=0&&size<=std::streamoff(1ULL<<30),"request too large");in.seekg(0);J req;in>>req;require(req.at("schema")=="native-graph-request-v1","wrong request schema");
 const auto dispatcher=make_default_dispatcher();const auto mode=std::string(argv[1]);J out={{"schema","native-graph-response-v1"},{"dispatcher",dispatcher->identity()}};
 if(mode=="decode") {const auto f=unhex(req.at("frame_hex"));auto restored=decode(f,*dispatcher,decode_limits(req));out["original_numeric_bytes"]=restored.original_numeric_bytes;out["roots"]=J::array();for(const auto& root:restored.roots)out["roots"].push_back({{"kind",kind(root.stream.kind)},{"width",root.stream.width},{"logical_view",root.logical_view},{"original_numeric_payload",root.original_numeric_payload},{"data_hex",hex(root.stream.data)}});}
 else {require(mode=="encode"||mode=="benchmark","unknown mode");const auto inputs=roots(req.at("roots"));const auto plan=recipe(req.at("recipe"));Bytes frame;std::uint64_t raw=0;for(const auto& root:inputs)if(root.original_numeric_payload)raw+=root.stream.data.size();
 if(mode=="benchmark") {auto b=benchmark(inputs,plan,*dispatcher,req.at("warmups"),req.at("repeats"));require(b.original_numeric_bytes==raw,"denominator mismatch");frame=std::move(b.frame);out["encode_ns"]=b.encode_nanoseconds;out["decode_ns"]=b.decode_nanoseconds;out["encode_ns_per_original_byte"]=J::array();out["decode_ns_per_original_byte"]=J::array();for(auto t:b.encode_nanoseconds)out["encode_ns_per_original_byte"].push_back(raw?J(double(t)/raw):J(nullptr));for(auto t:b.decode_nanoseconds)out["decode_ns_per_original_byte"].push_back(raw?J(double(t)/raw):J(nullptr));out["timing_boundary"]="C++ fixed-recipe graph scheduling + complete NGF1 framing/parse + all forward/inverse stages; excludes driver I/O and policy/search";}
 else frame=encode(inputs,plan,*dispatcher);
 out["original_numeric_bytes"]=raw;out["frame_bytes"]=frame.size();out["compression_ratio"]=raw?J(double(frame.size())/raw):J(nullptr);out["frame_hex"]=hex(frame);}
 std::cout<<out.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
