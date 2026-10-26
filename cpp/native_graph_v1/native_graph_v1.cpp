#include "native_graph_v1.h"
#include "stage_dispatch.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace native_graph_v1;
struct wc_native_graph_v1_result { Bytes data; std::vector<uint64_t> enc,dec; uint64_t raw=0; std::string error; };
namespace {
template<class F> wc_native_graph_v1_result* protect(F&& fn) noexcept { try { auto p=std::make_unique<wc_native_graph_v1_result>(); fn(*p); return p.release(); } catch(const std::exception& e) { auto p=std::make_unique<wc_native_graph_v1_result>(); p->error=e.what(); return p.release(); } catch(...) { auto p=std::make_unique<wc_native_graph_v1_result>(); p->error="unknown native graph failure"; return p.release(); } }
std::vector<RootInput> root(const uint8_t* data,size_t size) { if(size && !data) throw std::invalid_argument("null input"); Bytes b; if(size) b.assign(data,data+size); return {{{Kind::Serial,1,std::move(b)},true,"bytes"}}; }
Recipe recipe() { Recipe r; r.policy_registry="cpp-canonical-127-v4"; Step s; s.kind=StepKind::Stage; s.inputs={0}; s.stage.action="zlib.level6"; s.stage.profile="bytes"; s.policy_action_id=-1; r.steps.push_back(std::move(s)); Step stop; stop.kind=StepKind::Stop; r.steps.push_back(std::move(stop)); return r; }
}
extern "C" {
uint32_t wc_native_graph_v1_abi_version(){return 1;}
wc_native_graph_v1_result* wc_native_graph_v1_encode_zlib(const uint8_t* d,size_t n){return protect([&](auto& out){auto dsp=make_default_dispatcher();out.data=encode(root(d,n),recipe(),*dsp);out.raw=n;});}
wc_native_graph_v1_result* wc_native_graph_v1_decode(const uint8_t* d,size_t n){return protect([&](auto& out){if(n&&!d)throw std::invalid_argument("null frame");auto dsp=make_default_dispatcher();auto restored=decode({d,n},*dsp);if(restored.roots.size()!=1)throw std::invalid_argument("expected one root");out.data=std::move(restored.roots[0].stream.data);out.raw=restored.original_numeric_bytes;});}
wc_native_graph_v1_result* wc_native_graph_v1_benchmark_zlib(const uint8_t* d,size_t n,uint32_t w,uint32_t r){return protect([&](auto& out){auto dsp=make_default_dispatcher();auto b=benchmark(root(d,n),recipe(),*dsp,w,r);out.data=std::move(b.frame);out.enc=std::move(b.encode_nanoseconds);out.dec=std::move(b.decode_nanoseconds);out.raw=b.original_numeric_bytes;});}
const uint8_t* wc_native_graph_v1_result_data(const wc_native_graph_v1_result* p){return p&&p->error.empty()&&!p->data.empty()?p->data.data():nullptr;}
size_t wc_native_graph_v1_result_size(const wc_native_graph_v1_result* p){return p&&p->error.empty()?p->data.size():0;}
const char* wc_native_graph_v1_result_error(const wc_native_graph_v1_result* p){return p?p->error.c_str():"null result";}
const uint64_t* wc_native_graph_v1_encode_ns(const wc_native_graph_v1_result* p){return p&&p->error.empty()&&!p->enc.empty()?p->enc.data():nullptr;}
size_t wc_native_graph_v1_encode_ns_count(const wc_native_graph_v1_result* p){return p?p->enc.size():0;}
const uint64_t* wc_native_graph_v1_decode_ns(const wc_native_graph_v1_result* p){return p&&p->error.empty()&&!p->dec.empty()?p->dec.data():nullptr;}
size_t wc_native_graph_v1_decode_ns_count(const wc_native_graph_v1_result* p){return p?p->dec.size():0;}
uint64_t wc_native_graph_v1_original_numeric_bytes(const wc_native_graph_v1_result* p){return p?p->raw:0;}
void wc_native_graph_v1_result_free(wc_native_graph_v1_result* p){delete p;}
}
