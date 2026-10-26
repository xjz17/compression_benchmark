#include "graph_internal.hpp"
#include "stage_dispatch.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
using namespace native_graph_v1;
namespace {
unsigned checks = 0;
void check(bool ok, const char* msg) { ++checks; if (!ok) throw std::runtime_error(msg); }
template<class F> void rejects(F f) { bool rejected=false; try { f(); } catch(const std::exception&) { rejected=true; } check(rejected,"expected rejection"); }
Bytes words(std::initializer_list<std::uint64_t> values) { Bytes b; for(auto v:values) for(unsigned i=0;i<8;++i)b.push_back(v>>(8*i)); return b; }
std::vector<RootInput> roots() {
  return {{{Kind::Numeric,8,words({0,0x8000000000000000ULL,0x7ff8000000000001ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL,1,0xffffffffffffffffULL})},true,"float64"},
          {{Kind::Struct,3,{1,2,3,4,5,6,7,8,9}},true,"bytes"},
          {{Kind::Serial,1,{0,1,0,0,1,0,0}},false,"bytes"}};
}
Step stage(StreamId input, std::string name, std::string profile="bytes", std::map<std::string,std::string> params={}) {
  return {StepKind::Stage,{input},{std::move(name),std::move(profile),std::move(params)},{},-1};
}
Recipe recipe() {
  Recipe r; r.policy_registry="contract-test-only";
  // Keep physical Struct(3), empty continuous segments and separate output ports.
  r.steps.push_back({StepKind::SplitSegments,{1},{},{1,0,2},-1}); // 3,4,5
  r.steps.push_back({StepKind::Concat,{5,4,3},{},{},-1}); // 6; reverse explicit order
  r.steps.push_back(stage(6,"zlib.level6")); // 7
  r.steps.push_back(stage(0,"bitsplit","numeric",{{"mode","top8"}})); // 8,9
  r.steps.push_back(stage(8,"zlib.level6")); // 10
  r.steps.push_back(stage(9,"op.run_length.byte_optimal.cpp_v1","bytes",{{"symbol_width","1"}})); // 11
  r.steps.push_back(stage(2,"zlib.level6")); // 12, auxiliary mask is encoded too
  r.steps.push_back({StepKind::Stop});
  return r;
}
void verify(const Decoded& d) {
  const auto expected=roots(); check(d.original_numeric_bytes==65,"B_raw includes auxiliary mask or changed width");
  check(d.roots.size()==expected.size(),"root count");
  for(std::size_t i=0;i<expected.size();++i) {
    check(d.roots[i].stream==expected[i].stream,"original bit patterns not restored");
    check(d.roots[i].logical_view==expected[i].logical_view,"logical view not restored");
    check(d.roots[i].original_numeric_payload==expected[i].original_numeric_payload,"numeric flag not restored");
  }
}
void resign(Bytes& frame) {
  auto hash=detail::sha256(std::span(frame).first(frame.size()-32)); std::copy(hash.begin(),hash.end(),frame.end()-32);
}
void write64(Bytes& b,std::size_t at,std::uint64_t n) { for(unsigned i=0;i<8;++i)b.at(at+i)=n>>(8*i); }
std::size_t skiptext(const Bytes& b,std::size_t p) { std::size_t n=0;for(unsigned i=0;i<4;++i)n|=std::size_t(b.at(p+i))<<(i*8); return p+4+n; }
void run() {
  const auto d=make_default_dispatcher(); const auto data=roots(); const auto r=recipe();
  auto frame=encode(data,r,*d); verify(decode(frame,*d));
  auto parsed=detail::parse_frame(frame,{}); check(parsed.nodes.size()==7,"graph nodes");
  check(parsed.leaves.size()==4,"independent leaf flows lost");
  check(parsed.roots[1].spec.width==3,"Struct(3) silently widened");
  check(frame.size()>12,"complete frame missing");
  auto b=benchmark(data,r,*d,1,3); check(b.original_numeric_bytes==65,"timing denominator");
  check(b.encode_nanoseconds.size()==3 && b.decode_nanoseconds.size()==3,"timing sample count"); verify(decode(b.frame,*d));
  // Same physical words may carry different semantic views. Their common
  // output is bytes; each original view and bit pattern remains in the DAG.
  std::vector<RootInput> mixed={{{Kind::Numeric,8,words({0x8000000000000000ULL,1})},true,"int64"},
                               {{Kind::Numeric,8,words({0x7ff800000000002aULL,0})},true,"float64"}};
  Recipe mixed_plan;mixed_plan.steps={{StepKind::Concat,{1,0},{},{},-1},stage(2,"zlib.level6"),{StepKind::Stop}};
  auto mixed_frame=encode(mixed,mixed_plan,*d);auto mixed_graph=detail::parse_frame(mixed_frame,{});
  check(mixed_graph.nodes[0].output_logical_views==std::vector<std::string>{"bytes"},"mixed Concat must use bytes view");
  const auto mixed_decoded=decode(mixed_frame,*d);
  for(std::size_t i=0;i<mixed.size();++i) {
    check(mixed_decoded.roots[i].stream==mixed[i].stream,"mixed Concat changed root bits");
    check(mixed_decoded.roots[i].logical_view==mixed[i].logical_view,"mixed Concat lost root logical view");
  }
  mixed_graph.nodes[0].output_logical_views[0]="float64";
  rejects([&]{detail::serialize_frame(mixed_graph,{});});
  // Every truncation and every one-byte payload corruption must fail.
  for(std::size_t n=0;n<frame.size();++n)rejects([&]{decode(std::span(frame).first(n),*d);});
  for(std::size_t n=0;n<frame.size();++n){auto bad=frame;bad[n]^=1;rejects([&]{decode(bad,*d);});}
  auto bad=frame; bad.push_back(0); rejects([&]{decode(bad,*d);});
  // These altered complete frames have valid checksums, exercising semantic checks.
  bad=frame; const auto denominator=skiptext(frame,skiptext(frame,16)); write64(bad,denominator,66);resign(bad);rejects([&]{decode(bad,*d);});
  bad=frame; bad[denominator+8+4+13]=2;resign(bad);rejects([&]{decode(bad,*d);}); // root numeric flag
  auto corrupt=parsed; corrupt.leaves.erase(corrupt.leaves.begin()); rejects([&]{detail::serialize_frame(corrupt,{});});
  corrupt=parsed; corrupt.nodes[0].step.inputs={1,1}; rejects([&]{detail::serialize_frame(corrupt,{});});
  corrupt=parsed; corrupt.nodes[1].step.inputs={5,5,3}; rejects([&]{detail::serialize_frame(corrupt,{});});
  corrupt=parsed; corrupt.nodes[0].outputs[1]=corrupt.nodes[0].outputs[0];rejects([&]{detail::serialize_frame(corrupt,{});});
  corrupt=parsed; corrupt.nodes[0].step.segment_sizes[0]++;rejects([&]{detail::serialize_frame(corrupt,{});});
  corrupt=parsed; corrupt.nodes[0].output_logical_views[0]="float64";rejects([&]{detail::serialize_frame(corrupt,{});});
  corrupt=parsed; corrupt.dispatcher="other-version";bad=detail::serialize_frame(corrupt,{});rejects([&]{decode(bad,*d);});
  corrupt=parsed; corrupt.roots[0].digest[0]^=1;bad=detail::serialize_frame(corrupt,{});rejects([&]{decode(bad,*d);});
  // State rejection must leave all active inputs intact.
  GraphSession graph(data,*d); const auto before=graph.active_streams();
  rejects([&]{graph.apply({StepKind::SplitSegments,{1},{},{1,1},-1});});check(graph.active_streams()==before,"failed Split consumed input");
  rejects([&]{graph.apply({StepKind::Concat,{0,1},{},{},-1});});check(graph.active_streams()==before,"failed Concat consumed input");
  rejects([&]{graph.apply(stage(0,"unknown.operator"));});check(graph.active_streams()==before,"failed stage consumed input");
  rejects([&]{graph.apply({StepKind::Concat,{0,0},{},{},-1});});
  rejects([&]{graph.apply({StepKind::Stop});});
  auto stopped=graph.stop();verify(decode(stopped,*d));rejects([&]{graph.stop();});rejects([&]{graph.apply(stage(0,"zlib.level6"));});

  // A per-flow STOP hides one terminal leaf from subsequent policy actions
  // while preserving it in the complete frame. The other branch can keep
  // growing, and closed-input rejection happens before stage dispatch.
  struct CountingDispatcher final : StageDispatcher {
    std::unique_ptr<StageDispatcher> inner=make_default_dispatcher();
    mutable std::uint32_t forward_calls=0;
    std::string identity() const override { return inner->identity(); }
    StageResult forward(const StageRequest& request,const std::vector<Stream>& inputs) const override {
      ++forward_calls;return inner->forward(request,inputs);
    }
    std::vector<Stream> inverse(const StageRequest& request,const std::vector<Stream>& outputs,
                                std::span<const std::uint8_t> metadata) const override {
      return inner->inverse(request,outputs,metadata);
    }
  } counting;
  GraphSession branch_graph(mixed,counting);
  check(branch_graph.open_stream_ids()==std::vector<StreamId>({0,1}),"initial open flow IDs");
  check(branch_graph.open_stream_count()==2 && !branch_graph.is_closed(0),"initial open flow state");
  branch_graph.close_flow(0);
  check(branch_graph.is_closed(0) && branch_graph.open_stream_ids()==std::vector<StreamId>({1}),
        "closed flow remained policy-visible");
  check(branch_graph.active_streams().contains(0),"closed flow disappeared from complete leaves");
  rejects([&]{branch_graph.close_flow(0);});
  const auto calls_before_closed_rejection=counting.forward_calls;
  rejects([&]{branch_graph.apply(stage(0,"zlib.level6"));});
  check(counting.forward_calls==calls_before_closed_rejection,
        "closed stage input reached dispatcher forward");
  rejects([&]{branch_graph.apply({StepKind::SplitSegments,{0},{},{1,1},-1});});
  rejects([&]{branch_graph.apply({StepKind::Concat,{0,1},{},{},-1});});
  const auto extended=branch_graph.apply(stage(1,"zlib.level6"));
  check(extended.size()==1 && branch_graph.open_stream_ids()==extended,
        "open branch could not extend after sibling STOP");
  branch_graph.close_flow(extended.front());
  check(branch_graph.open_stream_count()==0 && branch_graph.active_streams().size()==2,
        "closed branches were not retained as final leaves");
  const auto branch_frame=branch_graph.stop();
  const auto branch_parsed=detail::parse_frame(branch_frame,{});
  check(branch_parsed.leaves.size()==2,"per-flow STOP lost a final leaf from wire frame");
  const auto branch_decoded=decode(branch_frame,counting);
  check(branch_decoded.roots.size()==mixed.size(),"per-flow STOP changed decoded root count");
  for(std::size_t i=0;i<mixed.size();++i)
    check(branch_decoded.roots[i].stream==mixed[i].stream &&
          branch_decoded.roots[i].logical_view==mixed[i].logical_view &&
          branch_decoded.roots[i].original_numeric_payload==mixed[i].original_numeric_payload,
          "per-flow STOP changed cold-decoded root");
  rejects([&]{branch_graph.close_flow(0);});

  // Legacy callers may still globally stop a session with open flows.
  GraphSession legacy_stop(mixed,*d);
  check(legacy_stop.open_stream_count()==2,"legacy global STOP fixture unexpectedly closed");
  const auto legacy_frame=legacy_stop.stop();
  check(decode(legacy_frame,*d).roots.size()==mixed.size(),"legacy global STOP behavior changed");
  Recipe incomplete; rejects([&]{encode(data,incomplete,*d);});incomplete.steps={stage(0,"zlib.level6")};rejects([&]{encode(data,incomplete,*d);});
  incomplete.steps={{StepKind::Stop},stage(0,"zlib.level6"),{StepKind::Stop}};rejects([&]{encode(data,incomplete,*d);});
  Limits limited; limited.max_nodes=1;rejects([&]{encode(data,r,*d,limited);});
  limited=Limits{};limited.max_active_streams=3;rejects([&]{encode(data,r,*d,limited);});
  limited=Limits{};limited.max_ordinary_depth=1;rejects([&]{encode(data,r,*d,limited);});
  limited=Limits{};limited.max_frame_bytes=64;rejects([&]{encode(data,r,*d,limited);});
  rejects([&]{benchmark(data,r,*d,0,0);});
  // All-null input: denominator remains exactly zero; complete mask still survives.
  std::vector<RootInput> empty={{{Kind::Numeric,4,{}},true,"float32"},{{Kind::Serial,1,{1,1,1}},false,"bytes"}};
  Recipe stop;stop.steps={{StepKind::Stop}};auto e=benchmark(empty,stop,*d,0,1);check(e.original_numeric_bytes==0,"empty denominator");check(decode(e.frame,*d).roots[1].stream.data==empty[1].stream.data,"all-null mask lost");
  std::cout<<"graph contract checks passed: "<<checks<<"\n";
}
}
int main(int argc,char** argv) {
  try {
    if(argc==1){run();return 0;}
    check(argc==3,"expected encode|decode path");auto d=make_default_dispatcher();
    if(std::string(argv[1])=="encode") { auto b=encode(roots(),recipe(),*d); std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<const char*>(b.data()),b.size());check(bool(out),"fixture write failed"); }
    else if(std::string(argv[1])=="decode") {std::ifstream in(argv[2],std::ios::binary);check(bool(in),"fixture read failed");Bytes b((std::istreambuf_iterator<char>(in)),{});verify(decode(b,*d));std::cout<<"independent process cold decode passed\n";}
    else throw std::runtime_error("unknown mode");
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
