#include "stage_dispatch.hpp"
#include "reference_operator_actions.hpp"
#include "native_lz_action_stages.hpp"
#include "legacy_numeric_action_stages.hpp"
#include "general_codec_reference_stages.hpp"
#include "huffman_bytes.hpp"
#include "arithmetic_bytes.hpp"
#include "pongo_exact_stages.hpp"
#include "elf_float_codec.hpp"
#include "elf_double_codec.hpp"
#include "camel_exact_stages.hpp"
#include "buff_native_exact_stages.hpp"
#include "fpc_exact_stages.hpp"
#include "pfpc_exact_stages.hpp"
#include "dbp_exact_stages.hpp"
#include "ts2diff_exact_stages.hpp"
#include "gorilla_exact_stages.hpp"
#include "chimp128_exact_stages.hpp"
#include "spdp_exact_stages.hpp"
#include "optimal_rle.hpp"
#include "minimum_for.hpp"
#include "max_value_bitpack.hpp"
#include "unsigned_leb128.hpp"
#include "zigzag.hpp"
#include "dictionary_recode.hpp"
#include "exception_separation.hpp"
#include "adjacent_residual.hpp"
#include "residual_window_pack.hpp"
#include "bos_exact_stages.hpp"
#include "svb_exact_stages.hpp"
#include "simple8b_exact_stages.hpp"
#include "fastpfor_exact_stages.hpp"
#include "p04_exact_stages.hpp"
#include "rle_exact_stages.hpp"
#include "learning_codec_exact_stages.hpp"
#include "lz4_exact_stages.hpp"
#include "snappy_exact_stages.hpp"
#include "fpzip_exact_stages.hpp"
#include "prediction_probability_model.hpp"
#include "chimp_history_xor_v2.hpp"
#ifdef NATIVE_GRAPH_V1_HAVE_NNCP
#include "nncp_probability.h"
#endif
#include <nlohmann/json.hpp>
#include <zlib.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#if defined(__linux__)
#include <sys/mman.h>
#endif

// Existing source-level API from corad_lossless_cpp.cpp.  That translation
// unit predates a public header; its CLI main is renamed in stage_dispatch.cmake.
namespace corad_lossless_cpp {
std::vector<std::uint8_t> normalize_forward(const std::vector<std::uint8_t>&,
                                            const std::string&);
std::vector<std::uint8_t> normalize_inverse(const std::vector<std::uint8_t>&);
std::vector<std::uint8_t> tricklet_forward(const std::vector<std::uint8_t>&);
std::vector<std::uint8_t> tricklet_inverse(const std::vector<std::uint8_t>&);
}

namespace native_graph_v1 {
namespace {
using J = nlohmann::json;
using U = std::uint64_t;
constexpr U kMaxBytes = 256ULL << 20;
static_assert(std::endian::native == std::endian::little, "existing typed stage APIs require little-endian host");
void require(bool v, const std::string& m) { if (!v) throw std::invalid_argument(m); }
U number(const std::string& v) {
  U n = 0; const auto [end, ec] = std::from_chars(v.data(), v.data()+v.size(), n);
  require(!v.empty() && ec == std::errc{} && end == v.data()+v.size(), "invalid unsigned stage parameter"); return n;
}
U param(const StageRequest& r, const std::string& k, U fallback) {
  auto i = r.parameters.find(k); return i == r.parameters.end() ? fallback : number(i->second);
}
std::string string_param(const StageRequest& r, const std::string& k, std::string fallback) {
  auto i=r.parameters.find(k); return i==r.parameters.end()?fallback:i->second;
}
bool bool_param(const StageRequest& r, const std::string& k, bool fallback) {
  auto i=r.parameters.find(k);if(i==r.parameters.end())return fallback;
  require(i->second=="true"||i->second=="false"||i->second=="1"||i->second=="0",
          "invalid boolean stage parameter: "+k);
  return i->second=="true"||i->second=="1";
}
void parameters(const StageRequest& r, std::initializer_list<const char*> names) {
  std::set<std::string> allowed; for(auto n:names) allowed.insert(n);
  for(const auto& [k,v]:r.parameters) require(allowed.contains(k), "unsupported stage parameter: "+k);
}
void parameters(const StageRequest& r, const std::vector<std::string>& names) {
  const std::set<std::string> allowed(names.begin(),names.end());
  for(const auto& [k,v]:r.parameters) require(allowed.contains(k), "unsupported stage parameter: "+k);
}
void shape(const Stream& s) {
  require(s.data.size() <= kMaxBytes, "stage input exceeds 256 MiB limit");
  require((s.kind==Kind::Serial && s.width==1) ||
          (s.kind==Kind::Numeric && (s.width==1 || s.width==2 || s.width==4 || s.width==8)) ||
          (s.kind==Kind::Struct && s.width>0), "invalid stage data flow kind/width");
  require(s.data.size()%s.width==0, "stage data flow is not lane aligned");
}
void word64(const Stream& s) { require(s.data.size()%8==0 && (s.kind==Kind::Serial || s.width==8), "stage requires 64-bit lanes"); }
void profile(const StageRequest& r, std::initializer_list<const char*> accepted) {
  for(auto p:accepted) if(r.profile==p) return;
  throw std::invalid_argument("unsupported stage profile: "+r.profile);
}
template<class T> std::vector<T> typed(const Bytes& b) {
  require(b.size()%sizeof(T)==0,"typed stage input is not aligned");
  std::vector<T> v(b.size()/sizeof(T)); if(!b.empty()) std::memcpy(v.data(),b.data(),b.size()); return v;
}
template<class T> Bytes raw(const std::vector<T>& v) {
  require(v.size()<=kMaxBytes/sizeof(T),"decoded stage exceeds byte limit");
  Bytes b(v.size()*sizeof(T)); if(!b.empty()) std::memcpy(b.data(),v.data(),b.size()); return b;
}
J spec(const Stream& s) { return J{{"kind",static_cast<unsigned>(s.kind)},{"width",s.width},{"bytes",s.data.size()}}; }
Stream restore_shape(const J& s, Bytes data) {
  require(s.at("bytes").get<U>()==data.size(),"inverse stage length mismatch");
  Stream out{static_cast<Kind>(s.at("kind").get<unsigned>()),s.at("width").get<std::uint32_t>(),std::move(data)};shape(out);return out;
}
Stream serial(Bytes b) { require(b.size()<=kMaxBytes,"encoded stage exceeds byte limit"); return {Kind::Serial,1,std::move(b)}; }
Bytes zlib_encode(const Bytes& in) {
  uLongf size=compressBound(in.size());Bytes out(size);
  require(compress2(out.data(),&size,in.empty()?reinterpret_cast<const Bytef*>(""):in.data(),in.size(),6)==Z_OK,"zlib6 encode failed");out.resize(size);return out;
}
Bytes zlib_decode(const Bytes& in,U expected) {
  require(expected<=kMaxBytes && in.size()<=UINT_MAX,"zlib decode bound exceeded");
  z_stream s{};require(inflateInit(&s)==Z_OK,"inflate init failed");
  Bytes out(static_cast<std::size_t>(expected)+1);s.next_in=const_cast<Bytef*>(in.data());s.avail_in=in.size();s.next_out=out.data();s.avail_out=out.size();
  const auto code=inflate(&s,Z_FINISH);const uLong bytes=s.total_out; const uInt remaining=s.avail_in;inflateEnd(&s);
  require(code==Z_STREAM_END && remaining==0 && bytes==expected,"invalid/truncated/trailing zlib stream");out.resize(expected);return out;
}

struct ProbabilityBitWriter {
  Bytes bytes; unsigned used = 0;
  void put(unsigned bit) {
    if (used == 0) bytes.push_back(0);
    if (bit) bytes.back() |= static_cast<std::uint8_t>(1U << (7U - used));
    used = (used + 1U) & 7U;
  }
};
struct ProbabilityBitReader {
  std::span<const std::uint8_t> bytes; std::size_t bit = 0;
  unsigned get() {
    require(bit < bytes.size() * 8U + 32U, "probability entropy padding exhausted");
    const auto index = bit++;
    return index < bytes.size() * 8U
               ? (bytes[index / 8U] >> (7U - index % 8U)) & 1U
               : 0U;
  }
};
prediction_probability::Options probability_options(const std::string& action,
                                                     const StageRequest& request) {
  prediction_probability::Options options;
  if (action=="op.prediction_probability.ppm.provider_v1") {
    options.provider="ppmd";
    options.ppmd_order=param(request,"ppmd_order",6);
    options.ppmd_memory_mib=param(request,"ppmd_memory_mib",16);
  }
  else if (action=="op.prediction_probability.dmc.provider_v1") options.provider="dmc";
  else if (action=="op.prediction_probability.paq.provider_v1") {
    options.provider="paq";options.paq_level=param(request,"paq_level",0);
  }
  else if (action=="op.prediction_probability.cmix.provider_v1") {
    options.provider="cmix";
    options.allow_large_memory=bool_param(request,"allow_large_memory",false);
  }
  else throw std::invalid_argument("unsupported in-process probability provider");
  prediction_probability::validate_options(options);
  return options;
}
std::uint32_t probability_one(double value) {
  require(std::isfinite(value) && value>=0.0 && value<=1.0,
          "invalid probability model output");
  const auto scaled=1U+static_cast<std::uint32_t>(value*65534.0);
  return std::min<std::uint32_t>(scaled,65535U);
}
Bytes probability_entropy_encode(const Bytes& raw,const std::string& action,
                                 const StageRequest& request) {
  require(raw.size()<=16ULL*1024*1024,"probability entropy input exceeds 16 MiB");
  prediction_probability::ProbabilityModel model(probability_options(action,request));
  constexpr std::uint32_t half=0x80000000U,quarter=0x40000000U,three=0xc0000000U,total=65536U;
  std::uint32_t low=0,high=0xffffffffU;U pending=0;ProbabilityBitWriter writer;
  auto emit=[&](unsigned bit){writer.put(bit);while(pending){writer.put(bit^1U);--pending;}};
  for(auto byte:raw)for(int shift=7;shift>=0;--shift){
    const auto bit=(byte>>shift)&1U,one=probability_one(model.predict()),zero=total-one;
    const U range=U{high}-low+1;const auto split=low+static_cast<std::uint32_t>(range*zero/total);
    require(split!=low&&split<=high,"probability interval underflow");
    if(bit)low=split;else high=split-1;
    for(;;){
      if(high<half)emit(0);
      else if(low>=half){emit(1);low-=half;high-=half;}
      else if(low>=quarter&&high<three){++pending;low-=quarter;high-=quarter;}
      else break;
      low<<=1;high=(high<<1)|1U;
    }
    model.observe(bit);
  }
  ++pending;emit(low>=quarter?1U:0U);return writer.bytes;
}
Bytes probability_entropy_decode(const Bytes& body,U length,const std::string& action,
                                 const StageRequest& request) {
  require(length<=16ULL*1024*1024 && !body.empty(),"invalid probability entropy shape");
  prediction_probability::ProbabilityModel model(probability_options(action,request));
  constexpr std::uint32_t half=0x80000000U,quarter=0x40000000U,three=0xc0000000U,total=65536U;
  std::uint32_t low=0,high=0xffffffffU,code=0;ProbabilityBitReader reader{body};
  for(unsigned i=0;i<32;++i)code=(code<<1)|reader.get();
  Bytes raw;raw.reserve(length);
  for(U index=0;index<length;++index){
    std::uint8_t byte=0;
    for(unsigned j=0;j<8;++j){
      require(code>=low&&code<=high,"probability arithmetic code outside interval");
      const auto one=probability_one(model.predict()),zero=total-one;const U range=U{high}-low+1;
      const auto split=low+static_cast<std::uint32_t>(range*zero/total);
      require(split!=low&&split<=high,"probability interval underflow");
      const auto bit=code>=split?1U:0U;if(bit)low=split;else high=split-1;
      for(;;){
        if(high<half){}
        else if(low>=half){low-=half;high-=half;code-=half;}
        else if(low>=quarter&&high<three){low-=quarter;high-=quarter;code-=quarter;}
        else break;
        low<<=1;high=(high<<1)|1U;code=(code<<1)|reader.get();
      }
      byte=static_cast<std::uint8_t>((byte<<1)|bit);model.observe(bit);
    }
    raw.push_back(byte);
  }
  require(probability_entropy_encode(raw,action,request)==body,
          "noncanonical/trailing probability entropy body");
  return raw;
}

class PaqIsolatedPlugin {
 public:
  using Encode = int (*)(const std::uint8_t*,std::size_t,unsigned,
                         std::uint8_t**,std::size_t*,char*,std::size_t);
  using Decode = int (*)(const std::uint8_t*,std::size_t,std::size_t,unsigned,
                         std::uint8_t**,std::size_t*,char*,std::size_t);
  using Free = void (*)(void*);

  explicit PaqIsolatedPlugin(bool raw_block_header) {
    handle_=dlopen(NATIVE_GRAPH_V1_PAQ_PLUGIN_PATH,RTLD_NOW|RTLD_LOCAL);
    const char* load_error=handle_?nullptr:dlerror();
    require(handle_!=nullptr,std::string("PAQ isolated backend load failed: ")+
                                (load_error?load_error:"unknown dlopen error"));
    try {
      encode_=symbol<Encode>(raw_block_header?"native_graph_paq_encode_v2":"native_graph_paq_encode");
      decode_=symbol<Decode>(raw_block_header?"native_graph_paq_decode_v2":"native_graph_paq_decode");
      free_=symbol<Free>("native_graph_paq_free");
    } catch (...) {dlclose(handle_);handle_=nullptr;throw;}
  }
  ~PaqIsolatedPlugin(){if(handle_)dlclose(handle_);}
  PaqIsolatedPlugin(const PaqIsolatedPlugin&)=delete;
  PaqIsolatedPlugin& operator=(const PaqIsolatedPlugin&)=delete;

  Bytes encode(const Bytes& input,unsigned level) const {
    std::uint8_t* output=nullptr;std::size_t size=0;std::array<char,512> error{};
    require(encode_(input.data(),input.size(),level,&output,&size,error.data(),error.size())==0,
            std::string("PAQ isolated encode failed: ")+error.data());
    require(size<=kMaxBytes && output!=nullptr,"PAQ isolated encode shape invalid");
    Bytes result(output,output+size);free_(output);return result;
  }
  Bytes decode(const Bytes& input,U raw_size,unsigned level) const {
    require(raw_size<=kMaxBytes,"PAQ decoded size exceeds bound");
    std::uint8_t* output=nullptr;std::size_t size=0;std::array<char,512> error{};
    require(decode_(input.data(),input.size(),raw_size,level,&output,&size,error.data(),error.size())==0,
            std::string("PAQ isolated decode failed: ")+error.data());
    require(size==raw_size && output!=nullptr,"PAQ isolated decode shape invalid");
    Bytes result(output,output+size);free_(output);return result;
  }

 private:
  template<class Function> Function symbol(const char* name) {
    dlerror();void* address=dlsym(handle_,name);const char* error=dlerror();
    require(error==nullptr&&address!=nullptr,std::string("PAQ isolated symbol missing: ")+name);
    return reinterpret_cast<Function>(address);
  }
  void* handle_=nullptr;Encode encode_=nullptr;Decode decode_=nullptr;Free free_=nullptr;
};

class CmixIsolatedPlugin {
 public:
  using Encode = int (*)(const std::uint8_t*,std::size_t,
                         std::uint8_t**,std::size_t*,char*,std::size_t);
  using Decode = int (*)(const std::uint8_t*,std::size_t,std::size_t,
                         std::uint8_t**,std::size_t*,char*,std::size_t);
  using Free = void (*)(void*);
  using ArenaSnapshot = void (*)(void**,std::size_t*,std::size_t*);

  struct Arena {
    void* base=nullptr;
    std::size_t reserved_bytes=0;
    std::size_t used_bytes=0;
  };

  explicit CmixIsolatedPlugin(Arena& arena): arena_(arena) {
    handle_=dlopen(NATIVE_GRAPH_V1_CMIX_PLUGIN_PATH,RTLD_NOW|RTLD_LOCAL);
    const char* load_error=handle_?nullptr:dlerror();
    require(handle_!=nullptr,std::string("cmix isolated backend load failed: ")+
                                (load_error?load_error:"unknown dlopen error"));
    try {
      encode_=symbol<Encode>("native_graph_cmix_encode");
      decode_=symbol<Decode>("native_graph_cmix_decode");
      free_=symbol<Free>("native_graph_cmix_free");
      arena_snapshot_=symbol<ArenaSnapshot>("native_graph_cmix_arena_snapshot");
    } catch (...) {dlclose(handle_);handle_=nullptr;throw;}
  }
  ~CmixIsolatedPlugin(){close();}
  CmixIsolatedPlugin(const CmixIsolatedPlugin&)=delete;
  CmixIsolatedPlugin& operator=(const CmixIsolatedPlugin&)=delete;

  Bytes encode(const Bytes& input) const {
    std::uint8_t* output=nullptr;std::size_t size=0;std::array<char,512> error{};
    require(encode_(input.data(),input.size(),&output,&size,error.data(),error.size())==0,
            std::string("cmix isolated encode failed: ")+error.data());
    require(size<=kMaxBytes && output!=nullptr,"cmix isolated encode shape invalid");
    Bytes result(output,output+size);free_(output);return result;
  }
  Bytes decode(const Bytes& input,U raw_size) const {
    require(raw_size<=kMaxBytes,"cmix decoded size exceeds bound");
    std::uint8_t* output=nullptr;std::size_t size=0;std::array<char,512> error{};
    require(decode_(input.data(),input.size(),raw_size,&output,&size,error.data(),error.size())==0,
            std::string("cmix isolated decode failed: ")+error.data());
    require(size==raw_size && output!=nullptr,"cmix isolated decode shape invalid");
    Bytes result(output,output+size);free_(output);return result;
  }

 private:
  void close() noexcept {
    if (!handle_) return;
    if (arena_snapshot_)
      arena_snapshot_(&arena_.base,&arena_.reserved_bytes,&arena_.used_bytes);
    dlclose(handle_);
    handle_=nullptr;
  }
  template<class Function> Function symbol(const char* name) {
    dlerror();void* address=dlsym(handle_,name);const char* error=dlerror();
    require(error==nullptr&&address!=nullptr,std::string("cmix isolated symbol missing: ")+name);
    return reinterpret_cast<Function>(address);
  }
  Arena& arena_;
  void* handle_=nullptr;Encode encode_=nullptr;Decode decode_=nullptr;Free free_=nullptr;
  ArenaSnapshot arena_snapshot_=nullptr;
};

class CmixArenaRelease {
 public:
  explicit CmixArenaRelease(CmixIsolatedPlugin::Arena& arena):arena_(arena){}
  ~CmixArenaRelease() { release(); }
  void release() noexcept {
#if defined(__linux__)
    if (arena_.base && arena_.reserved_bytes) {
      munmap(arena_.base,arena_.reserved_bytes);
      arena_.base=nullptr;arena_.reserved_bytes=0;arena_.used_bytes=0;
    }
#endif
  }
 private:
  CmixIsolatedPlugin::Arena& arena_;
};

void require_plugin_unloaded(const char* path,const char* label) {
#ifdef RTLD_NOLOAD
  void* resident=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
  if(resident){dlclose(resident);throw std::runtime_error(std::string(label)+
      " isolated backend remained resident after dlclose");}
#else
  (void)path;(void)label;
#endif
}

std::mutex paq_plugin_mutex;
Bytes paq_isolated_encode_unlocked(const Bytes& raw,unsigned level,bool raw_block_header) {
  Bytes result;
  {PaqIsolatedPlugin plugin(raw_block_header);result=plugin.encode(raw,level);}
  require_plugin_unloaded(NATIVE_GRAPH_V1_PAQ_PLUGIN_PATH,"PAQ");
  return result;
}
Bytes paq_isolated_decode(const Bytes& body,U raw_size,unsigned level,bool raw_block_header) {
  // dlopen may return the same handle to concurrent callers; PAQ8's upstream
  // globals are process-wide within that DSO.  Serialize the complete
  // load/call/unload sequence so one process is safe even if a caller has
  // multiple threads.  Multi-worker experiments use separate processes.
  const std::lock_guard<std::mutex> lock(paq_plugin_mutex);
  Bytes raw;
  {PaqIsolatedPlugin plugin(raw_block_header);raw=plugin.decode(body,raw_size,level);}
  require_plugin_unloaded(NATIVE_GRAPH_V1_PAQ_PLUGIN_PATH,"PAQ");
  const auto canonical=paq_isolated_encode_unlocked(raw,level,raw_block_header);
  require(canonical==body,"noncanonical/trailing PAQ entropy body");
  return raw;
}
Bytes paq_isolated_encode(const Bytes& raw,unsigned level) {
  const std::lock_guard<std::mutex> lock(paq_plugin_mutex);
  return paq_isolated_encode_unlocked(raw,level,true);
}
std::mutex cmix_plugin_mutex;
Bytes cmix_isolated_encode_unlocked(const Bytes& raw) {
  CmixIsolatedPlugin::Arena arena;CmixArenaRelease release(arena);
  Bytes result;
  {CmixIsolatedPlugin plugin(arena);result=plugin.encode(raw);}
  require_plugin_unloaded(NATIVE_GRAPH_V1_CMIX_PLUGIN_PATH,"cmix");
  return result;
}
Bytes cmix_isolated_encode(const Bytes& raw) {
  const std::lock_guard<std::mutex> lock(cmix_plugin_mutex);
  return cmix_isolated_encode_unlocked(raw);
}
Bytes cmix_isolated_decode(const Bytes& body,U raw_size) {
  const std::lock_guard<std::mutex> lock(cmix_plugin_mutex);
  CmixIsolatedPlugin::Arena decode_arena;CmixArenaRelease decode_release(decode_arena);
  Bytes raw;
  {CmixIsolatedPlugin plugin(decode_arena);raw=plugin.decode(body,raw_size);}
  require_plugin_unloaded(NATIVE_GRAPH_V1_CMIX_PLUGIN_PATH,"cmix");
  // Canonical validation loads an independent fresh predictor.  Reclaim the
  // decoder arena first so two 48 GiB virtual mappings never overlap inside
  // the worker's enforced 64 GiB address-space limit.
  decode_release.release();
  const auto canonical=cmix_isolated_encode_unlocked(raw);
  require(canonical==body,"noncanonical/trailing cmix entropy body");
  return raw;
}
#ifdef NATIVE_GRAPH_V1_HAVE_NNCP
Bytes nncp_file_bytes(FILE* file) {
  require(std::fflush(file)==0 && std::fseek(file,0,SEEK_END)==0,
          "NNCP temporary output seek failed");
  const auto end=std::ftell(file);require(end>=0 && static_cast<U>(end)<=kMaxBytes,
                                    "NNCP temporary output size invalid");
  require(std::fseek(file,0,SEEK_SET)==0,"NNCP temporary output rewind failed");
  Bytes out(static_cast<std::size_t>(end));
  require(out.empty()||std::fread(out.data(),1,out.size(),file)==out.size(),
          "NNCP temporary output read failed");
  return out;
}
Bytes nncp_entropy(const Bytes& input,U count,bool encode) {
  require(count<=NNCP_PROBABILITY_SYMBOL_LIMIT,
          "NNCP bounded symbol limit exceeded");
  std::unique_ptr<FILE,decltype(&std::fclose)> source(std::tmpfile(),std::fclose);
  std::unique_ptr<FILE,decltype(&std::fclose)> sink(std::tmpfile(),std::fclose);
  require(source&&sink,"NNCP temporary file allocation failed");
  require(input.empty()||std::fwrite(input.data(),1,input.size(),source.get())==input.size(),
          "NNCP temporary input write failed");
  require(std::fseek(source.get(),0,SEEK_SET)==0,"NNCP temporary input rewind failed");
  const auto status=encode?nncp_probability_encode(source.get(),sink.get(),count)
                          :nncp_probability_decode(source.get(),sink.get(),count);
  require(status==0,encode?"NNCP native encode failed":"NNCP native decode failed");
  return nncp_file_bytes(sink.get());
}
#endif
using Transform=std::function<Bytes(const Bytes&,const StageRequest&,J&)>;
struct PacketEntry { StageCapability info; Transform encode,decode; std::vector<std::string> parameter_names; bool lanes64=false; };
std::vector<PacketEntry> build_packets() {
  std::vector<PacketEntry> entries;
  auto pair=[&](std::string name,std::string source,auto enc,auto dec,std::string semantics="existing exact stage packet",bool lanes=false) {
    entries.push_back({{std::move(name),std::move(source),std::move(semantics),false},
      [enc](const Bytes& b,const StageRequest&,J&){return enc(b);},
      [dec](const Bytes& b,const StageRequest&,J&){return dec(b);},{},lanes});
  };
  auto custom=[&](std::string name,std::string source,std::string semantics,
                  Transform enc,Transform dec,bool lanes=false) {
    entries.push_back({{std::move(name),std::move(source),std::move(semantics),false},
                       std::move(enc),std::move(dec),{},lanes});
  };
#define PAIR(NAME,NS,FWD,INV,SOURCE) pair(NAME,SOURCE,[](const Bytes& b){return NS::FWD(std::span<const std::uint8_t>(b));},[](const Bytes& b){return NS::INV(std::span<const std::uint8_t>(b));})
  PAIR("op.eraser.pongo.cpp_v1",exact_actions::pongo,eraser_forward,eraser_inverse,"cpp/src/pongo_codec.cpp");entries.back().lanes64=true;
  PAIR("op.xor.pongo.cpp_v1",exact_actions::pongo,xor_forward,xor_inverse,"cpp/src/pongo_codec.cpp");
  PAIR("op.trailing_zero_pack.pongo.cpp_v1",exact_actions::pongo,trailing_pack_forward,trailing_pack_inverse,"cpp/src/pongo_codec.cpp");
  PAIR("op.eraser.elf64.cpp_v1",elf_double,eraser64Forward,eraser64Inverse,"cpp/src/elf_double_codec.cpp");entries.back().lanes64=true;
  pair("op.eraser.elf.cpp_v1","cpp/src/elf_float_codec.cpp",[](const Bytes& b){return exact_actions::elf::eraser_forward_double_bits(typed<std::int64_t>(b));},[](const Bytes& b){return raw(exact_actions::elf::eraser_inverse_double_bits(b));},"existing lossless float64-to-float32 gated Elf32 profile",true);
  PAIR("op.xor.elf.cpp_v1",exact_actions::elf,elf_xor_forward,elf_xor_inverse,"cpp/src/elf_float_codec.cpp");
  PAIR("op.xor.elf_plus.cpp_v1",exact_actions::elf,elf_plus_xor_forward,elf_plus_xor_inverse,"cpp/src/elf_float_codec.cpp");
  PAIR("op.xor.camel.cpp_v1",exact_actions::camel,xor_forward,xor_inverse,"cpp/src/camel_exact_stages.cpp");
  PAIR("op.bitstream_pack.camel.cpp_v1",exact_actions::camel,bitstream_pack_forward,bitstream_pack_inverse,"cpp/src/camel_exact_stages.cpp");
  PAIR("op.block_transform.buff_native_word_order.cpp_v1",exact_actions::buff_native,word_order_forward,word_order_inverse,"cpp/src/buff_native_exact_stages.cpp");entries.back().lanes64=true;
  PAIR("op.container_frame.buff_native_raw64.cpp_v1",exact_actions::buff_native,frame_forward,frame_inverse,"cpp/src/buff_native_exact_stages.cpp");
  PAIR("op.trailing_zero_pack.fpc.cpp_v1",exact_actions::fpc,trailing_pack_forward,trailing_pack_inverse,"cpp/src/fpc_exact_stages.cpp");
  PAIR("op.trailing_zero_pack.pfpc.cpp_v1",exact_actions::pfpc,trailing_pack_forward,trailing_pack_inverse,"cpp/src/pfpc_exact_stages.cpp");
  PAIR("op.delta.spdp.cpp_v1",exact_actions::spdp,delta_forward,delta_inverse,"cpp/src/spdp_exact_stages.cpp");entries.back().lanes64=true;
  PAIR("op.block_transform.spdp.cpp_v1",exact_actions::spdp,block_transform_forward,block_transform_inverse,"cpp/src/spdp_exact_stages.cpp");
  PAIR("op.frame_of_reference.ts_2diff.cpp_v1",exact_actions::ts2diff,frame_of_reference_forward,frame_of_reference_inverse,"cpp/src/ts2diff_exact_stages.cpp");
  PAIR("op.zigzag.dbp_residual.cpp_v2",exact_actions::dbp,zigzag_forward,zigzag_inverse,"cpp/src/dbp_exact_stages.cpp");
  PAIR("op.integer_code.uleb128.dbp_residuals.cpp_v2",exact_actions::dbp,varint_forward,varint_inverse,"cpp/src/dbp_exact_stages.cpp");

  custom("op.delta.bos.cpp_v1","cpp/src/bos_exact_stages.cpp","BOSA modular delta packet",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64"});return exact_actions::bos::delta_forward(typed<std::int64_t>(b));},
    [](const Bytes& b,const StageRequest&,J&){return raw(exact_actions::bos::delta_inverse(b));},true);
  pair("op.frame_of_reference.bos.cpp_v1","cpp/src/bos_exact_stages.cpp",exact_actions::bos::first_for_forward,exact_actions::bos::first_for_inverse);
  pair("op.outlier_separation.bos.cpp_v1","cpp/src/bos_exact_stages.cpp",exact_actions::bos::outlier_forward,exact_actions::bos::outlier_inverse);
  custom("op.delta.stream_vbyte.cpp_v1","cpp/src/svb_exact_stages.cpp","modular delta-1 over int64 limbs",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64","bytes"});return r.profile=="int64"?exact_actions::svb::delta_forward(typed<std::int64_t>(b)):exact_actions::svb::delta_forward_scaled(b);},
    [](const Bytes& b,const StageRequest& r,J&){return r.profile=="int64"?raw(exact_actions::svb::delta_inverse(b)):exact_actions::svb::delta_inverse_scaled(b);});
  pair("op.integer_code.stream_vbyte.cpp_v1","cpp/src/svb_exact_stages.cpp",exact_actions::svb::integer_code_forward,exact_actions::svb::integer_code_inverse);
  custom("op.bitpack_int.simple8b.cpp_v1","cpp/src/simple8b_exact_stages.cpp","Simple8b exact typed stage",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64","bytes"});return r.profile=="int64"?exact_actions::simple8b::bitpack_forward_int64(typed<std::int64_t>(b)):exact_actions::simple8b::bitpack_forward_scaled(b);},
    [](const Bytes& b,const StageRequest& r,J&){return r.profile=="int64"?raw(exact_actions::simple8b::bitpack_inverse_int64(b)):exact_actions::simple8b::bitpack_inverse_scaled(b);});
  custom("op.bitpack_int.fastpfor_pfor.cpp_v1","cpp/src/fastpfor_exact_stages.cpp","FastPFor<8> prefix stage",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64","bytes"});return r.profile=="int64"?exact_actions::fastpfor::pfor_stage_forward_int(typed<std::int64_t>(b)):exact_actions::fastpfor::pfor_stage_forward_scaled(b);},
    [](const Bytes& b,const StageRequest& r,J&){return r.profile=="int64"?raw(exact_actions::fastpfor::pfor_stage_inverse_int(b)):exact_actions::fastpfor::pfor_stage_inverse_scaled(b);});
  custom("op.integer_code.fastpfor_variablebyte.cpp_v1","cpp/src/fastpfor_exact_stages.cpp","Lemire VariableByte tail stage",
    [](const Bytes& b,const StageRequest&,J& side){require(b.size()>=17&&std::memcmp(b.data(),"PFS1",4)==0,"invalid FastPFor stage input");U count=0;for(unsigned i=0;i<8;++i)count|=U{b[9+i]}<<(8*i);side["count"]=count;return exact_actions::fastpfor::variable_byte_forward(b);},
    [](const Bytes& b,const StageRequest&,J& side){return exact_actions::fastpfor::variable_byte_inverse(b,side.at("count").get<U>());});
  custom("op.delta.sprintz.cpp_v1","cpp/src/p04_exact_stages.cpp","Sprintz predictor residual stage",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64","float64"});auto v=typed<std::int64_t>(b);return r.profile=="float64"?exact_actions::p04::sprintz_predictor_forward_double_bits(v):exact_actions::p04::sprintz_predictor_forward(v);},
    [](const Bytes& b,const StageRequest&,J&){return raw(exact_actions::p04::sprintz_predictor_inverse(b));},true);
  custom("op.delta.fpzip.cpp_v1","cpp/src/fpzip_exact_stages.cpp","FPZIP 1D Lorenzo predictor",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"float64"});return exact_actions::fpzip::predictor_forward(b);},
    [](const Bytes& b,const StageRequest&,J&){return exact_actions::fpzip::predictor_inverse(b);},true);
  pair("op.entropy_range.fpzip.cpp_v1","cpp/src/fpzip_exact_stages.cpp",exact_actions::fpzip::range_forward,exact_actions::fpzip::range_inverse);
  custom("op.scaling.tsfile_rle.cpp_v1","cpp/src/rle_exact_stages.cpp","bitwise-reversible TsFile-RLE scaling",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"float64"});auto n=param(r,"max_point",4);require(n<=18,"max_point exceeds 18");return exact_actions::rle::scaling_forward(typed<double>(b),n);},
    [](const Bytes& b,const StageRequest&,J&){return raw(exact_actions::rle::scaling_inverse(b));},true);
  entries.back().parameter_names={"max_point"};
  custom("op.scaling.alp.cpp_v1","cpp/src/learning_codec_exact_stages.cpp","ALP exact scaling/raw-bit fallback",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"float64"});return exact_actions::learning::alp_scaling_forward(b);},
    [](const Bytes& b,const StageRequest&,J&){return exact_actions::learning::alp_scaling_inverse(b);},true);
  custom("op.scaling.buff.cpp_v1","cpp/src/learning_codec_exact_stages.cpp","BUFF exact scaling/raw-bit fallback",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"float64"});return exact_actions::learning::buff_scaling_forward(b);},
    [](const Bytes& b,const StageRequest&,J&){return exact_actions::learning::buff_scaling_inverse(b);},true);
  pair("op.frame_of_reference.buff.cpp_v2","cpp/src/learning_codec_exact_stages.cpp",exact_actions::learning::buff_for_forward,exact_actions::learning::buff_for_inverse);
  custom("op.block_transform.buff_byte_plane_split.cpp_v2","cpp/src/learning_codec_exact_stages.cpp","BUFF byte planes at exact required width",
    [](const Bytes& b,const StageRequest&,J& side){const auto width=exact_actions::learning::buff_byte_plane_width(b);side["bit_width"]=width;return exact_actions::learning::buff_byte_plane_forward(b,width);},
    [](const Bytes& b,const StageRequest&,J&){return exact_actions::learning::buff_byte_plane_inverse(b);});
  custom("op.regression_predictor.leco_linear.cpp_v2","cpp/src/learning_codec_exact_stages.cpp","LECO signed linear residuals",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64"});return exact_actions::learning::leco_regression_forward(b);},
    [](const Bytes& b,const StageRequest&,J&){return exact_actions::learning::leco_regression_inverse(b);},true);
  pair("op.block_state_control.lz4_matchfinder_residual_core.cpp_v1","cpp/src/lz4_exact_stages.cpp",exact_actions::lz4::lz_match_forward,exact_actions::lz4::lz_match_inverse,"vendored LZ4 frame matcher and block decisions");
  pair("op.match_token_format.lz4.cpp_v1","cpp/src/lz4_exact_stages.cpp",exact_actions::lz4::token_format_forward,exact_actions::lz4::token_format_inverse);
  pair("op.integer_code.lz4.cpp_v1","cpp/src/lz4_exact_stages.cpp",exact_actions::lz4::integer_code_forward,exact_actions::lz4::integer_code_inverse);
  pair("op.block_state_control.snappy_matchfinder_residual_core.cpp_v1","cpp/src/snappy_exact_stages.cpp",exact_actions::snappy::snap111_match_forward,exact_actions::snappy::lz_match_inverse,"vendored snap-1.1.1 matcher and semantic records");
  pair("op.lz_match.snappy_snap_1_1_1.cpp_v1","cpp/src/snappy_exact_stages.cpp",exact_actions::snappy::snap111_match_forward,exact_actions::snappy::lz_match_inverse,"vendored snap-1.1.1 matcher and semantic records");
  pair("op.match_token_format.snappy.cpp_v1","cpp/src/snappy_exact_stages.cpp",exact_actions::snappy::token_format_forward,exact_actions::snappy::token_format_inverse);
  custom("op.scaling.corad_normalize_exact.cpp_v2","cpp/src/corad_lossless_cpp.cpp","CORAD normalization with exact repair words",
    [](const Bytes& b,const StageRequest& r,J&){profile(r,{"int64","float64"});return corad_lossless_cpp::normalize_forward(b,r.profile);},
    [](const Bytes& b,const StageRequest&,J&){return corad_lossless_cpp::normalize_inverse(b);},true);
  pair("op.dictionary.corad_tricklet_xor.cpp_v2","cpp/src/corad_lossless_cpp.cpp",corad_lossless_cpp::tricklet_forward,corad_lossless_cpp::tricklet_inverse,"CORAD 40-lane tricklet XOR dictionary");
  for(const auto& probability_action:{"op.prediction_probability.ppm.provider_v1","op.prediction_probability.dmc.provider_v1"}) {
    custom(probability_action,"cpp/src/prediction_probability_model.cpp","actual causal predictor plus in-process binary arithmetic entropy edge",
      [probability_action](const Bytes& b,const StageRequest& r,J& side){side["raw_bytes"]=b.size();return probability_entropy_encode(b,probability_action,r);},
      [probability_action](const Bytes& b,const StageRequest& r,J& side){return probability_entropy_decode(b,side.at("raw_bytes").get<U>(),probability_action,r);});
    if(std::string(probability_action).find("ppm.")!=std::string::npos)
      entries.back().parameter_names={"ppmd_order","ppmd_memory_mib"};
  }
  custom("op.prediction_probability.cmix.provider_v1","cpp/native_graph_v1/cmix_isolated_plugin.cpp",
    "full upstream cmix predictor with per-call timed DSO load/unload state isolation plus in-process arithmetic entropy edge",
    [](const Bytes& b,const StageRequest& r,J& side){require(bool_param(r,"allow_large_memory",false),"cmix requires allow_large_memory=true");side["raw_bytes"]=b.size();return cmix_isolated_encode(b);},
    [](const Bytes& b,const StageRequest& r,J& side){require(bool_param(r,"allow_large_memory",false),"cmix requires allow_large_memory=true");return cmix_isolated_decode(b,side.at("raw_bytes").get<U>());});
  entries.back().parameter_names={"allow_large_memory"};
  custom("op.prediction_probability.paq.provider_v1","cpp/native_graph_v1/paq_isolated_plugin.cpp",
    "upstream PAQ predictor with explicit DEFAULT raw-block initialization, timed DSO state isolation and in-process arithmetic edge",
    [](const Bytes& b,const StageRequest& r,J& side){side["raw_bytes"]=b.size();side["raw_block_header_version"]=2;return paq_isolated_encode(b,param(r,"paq_level",0));},
    [](const Bytes& b,const StageRequest& r,J& side){const auto version=side.value("raw_block_header_version",1);require(version==1||version==2,"unknown PAQ raw block initialization version");return paq_isolated_decode(b,side.at("raw_bytes").get<U>(),param(r,"paq_level",0),version==2);});
  entries.back().parameter_names={"paq_level"};
#ifdef NATIVE_GRAPH_V1_HAVE_NNCP
  custom("op.prediction_probability.nncp.provider_v1","cpp/src/nncp_probability_bridge.c",
    "upstream NNCP Transformer plus matched native arithmetic coder",
    [](const Bytes& b,const StageRequest&,J& side){side["raw_bytes"]=b.size();return nncp_entropy(b,b.size(),true);},
    [](const Bytes& b,const StageRequest&,J& side){return nncp_entropy(b,side.at("raw_bytes").get<U>(),false);});
#endif
  PAIR("op.dictionary.brotli_static_small.cpp_reference_v1",reference_actions::general,dictionary_forward,dictionary_inverse,"cpp/src/general_codec_reference_stages.cpp");
  entries.back().info.semantics="Brotli-shaped reference dictionary; not upstream Brotli wire";
  PAIR("op.match_token_format.blosclz.ctrl8.cpp_reference_v1",reference_actions::general,token_forward,token_inverse,"cpp/src/general_codec_reference_stages.cpp");entries.back().info.semantics="existing reference LZS1 token adapter, not upstream BloscLZ";
  PAIR("op.match_parse_policy.generic_packet.cpp_reference_v1",reference_actions::general,parse_policy_forward,parse_policy_inverse,"cpp/src/general_codec_reference_stages.cpp");
  PAIR("op.block_transform.bzip2_bwt.cpp_reference_v1",reference_actions::general,bwt_forward,bwt_inverse,"cpp/src/general_codec_reference_stages.cpp");
  PAIR("op.symbol_recode.bzip2_mtf.cpp_reference_v1",reference_actions::general,mtf_forward,mtf_inverse,"cpp/src/general_codec_reference_stages.cpp");
  PAIR("op.entropy_huffman.project_bytes.cpp_v1",huffman_bytes,encode,decode,"cpp/src/huffman_bytes.cpp");
  PAIR("op.entropy_range.project_bytes.cpp_v1",arithmetic_bytes,encode,decode,"cpp/src/arithmetic_bytes.cpp");
  PAIR("op.entropy_fse.zstd_rans.cpp_reference_v1",reference_actions::general,rans_forward,rans_inverse,"cpp/src/general_codec_reference_stages.cpp");entries.back().info.semantics="project rANS reference; not upstream Zstd FSE";
  for(const auto& a:reference_operator_actions::all_actions()) {
    pair(a,"cpp/src/reference_operator_actions.cpp",[a](const Bytes& b){return reference_operator_actions::encode(b,{a});},[](const Bytes& b){return reference_operator_actions::decode(b);},"frozen reference/proxy OPA1; never relabeled as production algorithm",true);
  }
  for(auto width:{1,4,8}) {
    pair("op.block_transform.blosc_shuffle.typesize"+std::to_string(width)+".cpp_reference_v1","cpp/src/general_codec_reference_stages.cpp",[width](const Bytes& b){return reference_actions::general::shuffle_forward(b,width);},[](const Bytes& b){return reference_actions::general::shuffle_inverse(b);},"existing reference byte shuffle packet");
  }
  struct LZ {const char* name;unsigned window,minimum,maximum;bool lazy;};
  for(const auto& a:std::vector<LZ>{{"brotli_lazy65535",65535,4,258,true},{"zstd_lazy65535",65535,3,258,true},{"deflate_lazy32768",32768,3,258,true},{"xz_lazy16777216",16777216,3,273,true},{"blosclz.greedy",65535,4,258,false}}) {
    pair(std::string("op.lz_match.")+a.name+".cpp_reference_v1","cpp/src/general_codec_reference_stages.cpp",[a](const Bytes& b){return reference_actions::general::lz_forward(b,a.window,a.minimum,a.maximum,a.lazy);},[](const Bytes& b){return reference_actions::general::lz_inverse(b);},"existing parameterized reference LZ search; no upstream wire claim");
  }
#undef PAIR
  return entries;
}
const std::vector<PacketEntry>& packets(){ static const auto p=build_packets();return p; }
const PacketEntry* packet(const std::string& name){for(const auto& p:packets())if(p.info.action==name)return &p;return nullptr;}
const std::vector<std::string> legacy_names{
 "op.scaling.legacy_exact_decimal.cpp_v1","op.scaling.alp_legacy_fe_exceptions.cpp_v1",
 "op.block_transform.int64_lo_hi32.cpp_v1","op.lz_match.native_lz_limb_batch.aligned8_greedy.cpp_v1",
 "op.match_token_format.native_lz_limb_batch.ctrl8.cpp_v1","op.container_frame.legacy_numeric_limbs.cpp_v1","op.container_frame.alp2_legacy.cpp_v1"};
bool legacy(const std::string& name){return std::find(legacy_names.begin(),legacy_names.end(),name)!=legacy_names.end();}
std::string legacy_wire_name(std::string name){const auto p=name.rfind(".cpp_v1");require(p!=std::string::npos,"invalid legacy action");name.replace(p,7,".python_v1");return name;}
const std::vector<std::string> native_lz_names{
 "op.match_window_model.native_lz.window4096","op.match_window_model.native_lz.window65535",
 "op.lz_match.native_lz.aligned8.min3.max18","op.match_parse_policy.native_lz.aligned8_greedy",
 "op.match_token_format.native_lz.ctrl8_offset16_len8.bias3","op.match_token_format.native_lz.ctrl8_offset16_len8.bias4"};
bool native_lz(const std::string& n){return std::find(native_lz_names.begin(),native_lz_names.end(),n)!=native_lz_names.end();}
void put(Bytes& b,U n,unsigned width) {for(unsigned i=0;i<width;++i)b.push_back(n>>(8*i));}
U get(std::span<const std::uint8_t> b,std::size_t& p,unsigned width) {
  require(p<=b.size() && width<=b.size()-p,"truncated stage metadata");U n=0;for(unsigned i=0;i<width;++i)n|=U{b[p++]}<<(8*i);return n;
}
bool reference(const std::string& name) {const auto& a=reference_operator_actions::all_actions();return std::find(a.begin(),a.end(),name)!=a.end();}
std::vector<Stream> reference_forward(const Stream& input,const std::string& name,J& side) {
  word64(input);const auto inspection=reference_operator_actions::inspect(input.data,{name});std::size_t p=0;
  const auto meta_bytes=get(inspection,p,8),value_bytes=get(inspection,p,8);
  require(meta_bytes<=inspection.size()-p && value_bytes<=inspection.size()-p-meta_bytes,"invalid native reference inspection");
  const Bytes meta(inspection.begin()+p,inspection.begin()+p+meta_bytes);p+=meta_bytes;
  std::vector<Stream> out{{Kind::Numeric,8,Bytes(inspection.begin()+p,inspection.begin()+p+value_bytes)}};
  require(meta.size()>=18,"reference metadata too short");p=12;const auto arrays=get(meta,p,2),extra=get(meta,p,4);
  side["record_header"]=Bytes(meta.begin(),meta.begin()+18);side["array_headers"]=J::array();
  require(extra<=meta.size()-p,"invalid reference extra length");
  if(extra)out.push_back(serial(Bytes(meta.begin()+p,meta.begin()+p+extra)));p+=extra;
  for(U i=0;i<arrays;++i){const auto begin=p;const auto count=get(meta,p,4),size=get(meta,p,4);require(size==count*8 && size<=meta.size()-p,"invalid reference array");side["array_headers"].push_back(Bytes(meta.begin()+begin,meta.begin()+p));out.push_back({Kind::Numeric,8,Bytes(meta.begin()+p,meta.begin()+p+size)});p+=size;}
  require(p==meta.size(),"trailing reference metadata");return out;
}
Bytes reference_inverse(const std::vector<Stream>& out,const std::string& name,const J& side) {
  Bytes state=side.at("record_header").get<Bytes>();require(state.size()==18,"invalid reference header metadata");std::size_t p=12;const auto arrays=get(state,p,2),extra=get(state,p,4);
  require(out.size()==1+arrays+(extra?1:0),"missing reference output port");std::size_t index=1;
  if(extra){require(out[index].data.size()==extra,"reference extra port mismatch");state.insert(state.end(),out[index].data.begin(),out[index].data.end());++index;}
  require(side.at("array_headers").size()==arrays,"reference array header mismatch");
  for(const auto& h:side.at("array_headers")){auto header=h.get<Bytes>();require(header.size()==8,"invalid reference array header");std::size_t q=0;auto count=get(header,q,4),size=get(header,q,4);require(size==count*8 && out[index].data.size()==size,"reference array port mismatch");state.insert(state.end(),header.begin(),header.end());state.insert(state.end(),out[index].data.begin(),out[index].data.end());++index;}
  const auto compact=reference_operator_actions::compact_encode(out.at(0).data);state.insert(state.end(),compact.begin(),compact.end());return reference_operator_actions::inverse_serialized(state,{name});
}
std::vector<unsigned> list(const std::string& text) {
  std::vector<unsigned> v;std::size_t p=0;while(p<text.size()){const auto end=text.find(',',p);const auto n=number(text.substr(p,end==std::string::npos?end:end-p));require(n<=UINT_MAX,"list item overflow");v.push_back(n);if(end==std::string::npos)break;p=end+1;require(p<text.size(),"trailing comma");}return v;
}
bool layout(const std::string& a){return a=="columnize" || a=="float_deconstruct" || a=="bitsplit" || a=="bitunpack";}
std::vector<Stream> layout_forward(const StageRequest& r,const Stream& s,J& m) {
  openzl_layout::SplitResult result;
  if(r.action=="columnize") {
    parameters(r,{"field_widths","mode","width"});
    if(r.parameters.contains("field_widths")){
      require(!r.parameters.contains("mode")&&!r.parameters.contains("width"),
              "columnize field_widths cannot be combined with transpose parameters");
      auto fields=list(r.parameters.at("field_widths"));std::vector<std::uint32_t>w(fields.begin(),fields.end());result=openzl_layout::split_struct(s.data,w);
    }else{
      const auto has_mode=r.parameters.contains("mode"),has_width=r.parameters.contains("width");
      require(has_mode==has_width,"columnize transpose mode and width must be paired");
      if(has_mode){require(string_param(r,"mode","")=="transpose","unsupported columnize mode");require(param(r,"width",0)==s.width,"columnize width does not match input width");}
      result=openzl_layout::transpose(s);
    }
  } else if(r.action=="float_deconstruct") {
    parameters(r,{"format"});result=openzl_layout::float_deconstruct(s,string_param(r,"format",r.profile=="float32"?"fp32":"fp16"));
  } else if(r.action=="bitsplit") {
    parameters(r,{"mode","widths","width"});const auto mode=string_param(r,"mode","top8");
    if(r.parameters.contains("width")){
      require(mode=="top8","bitsplit width parameter is valid only for top8 mode");
      require(param(r,"width",0)==s.width,"bitsplit width does not match input width");
    }
    if(mode=="top8")result=openzl_layout::bitsplit_top8(s);
    else if(mode=="fp")result=openzl_layout::bitsplit_fp(s);
    else if(mode=="bf16")result=openzl_layout::bitsplit_bf16(s);
    else if(mode=="widths"){auto fields=list(string_param(r,"widths",""));std::vector<std::uint8_t>w;for(auto x:fields){require(x<=64,"bitsplit field width too large");w.push_back(x);}result=openzl_layout::bitsplit(s,w);}
    else throw std::invalid_argument("unsupported bitsplit mode");
  } else {parameters(r,{"bits"});const auto bits=param(r,"bits",0);require(bits>=1 && bits<=8,"bitunpack requires bits in [1,8]");result=openzl_layout::bitunpack(s.data,bits);}
  m={{"operation",static_cast<unsigned>(result.operation)},{"input_kind",static_cast<unsigned>(result.input_kind)},{"input_width",result.input_width},{"input_bytes",result.input_bytes},{"side",result.side}};
  return result.streams;
}
Bytes layout_inverse(const std::vector<Stream>& out,const J& m){
  openzl_layout::SplitResult r{static_cast<openzl_layout::Operation>(m.at("operation").get<unsigned>()),static_cast<Kind>(m.at("input_kind").get<unsigned>()),m.at("input_width").get<std::uint32_t>(),m.at("input_bytes").get<U>(),m.at("side").get<Bytes>(),out};
  auto values=openzl_layout::inverse(r);require(values.size()==1,"layout inverse must reconstruct one input");return values[0].data;
}
struct Bits {
  Bytes bytes;std::size_t p=0;
  void write(U v,int n){require(n>=0&&n<=64,"invalid window width");for(int i=n;i-->0;++p){if(p%8==0)bytes.push_back(0);bytes.back()|=((v>>i)&1U)<<(7-p%8);}}
  U read(int n){require(n>=0&&n<=64&&p<=bytes.size()*8&&std::size_t(n)<=bytes.size()*8-p,"truncated window payload");U v=0;for(int i=0;i<n;++i,++p)v=(v<<1)|((bytes[p/8]>>(7-p%8))&1U);return v;}
};
Bytes residual_window(const Bytes& data,unsigned width,bool inverse,U count) {
  require(width==4||width==8,"Elf+ residual width must be 4 or 8");const int bits=width*8;
  constexpr std::array<int,8> b32{0,6,10,12,14,16,18,20},b64{0,8,12,16,18,20,22,24};const auto& buckets=width==4?b32:b64;
  exact_actions::residual_window_pack::ElfPlusProfile profile{bits,width==4?3:4,width==4?5:6};int sl=bits+1,st=bits+1;Bits packet;Bytes out;
  if(!inverse){require(data.size()%width==0,"unaligned residual input");for(std::size_t p=0;p<data.size();){const auto v=get(data,p,width);const int leading=v==0?bits:std::countl_zero(v)-(64-bits),trailing=v==0?bits:std::countr_zero(v);unsigned bucket=0;while(bucket+1<buckets.size()&&buckets[bucket+1]<=leading)++bucket;exact_actions::residual_window_pack::elf_plus([&](U x,int n){packet.write(x,n);},v,buckets[bucket],trailing,bucket,sl,st,profile);}return packet.bytes;}
  require(count<=kMaxBytes/width && count<=data.size()*4,"impossible residual count");packet.bytes=data;
  for(U i=0;i<count;++i){const auto v=exact_actions::residual_window_pack::elf_plus_inverse([&](int n){return packet.read(n);},buckets,sl,st,profile);put(out,v,width);}
  require((packet.p+7)/8==data.size(),"trailing residual window bytes");while(packet.p%8)require(packet.read(1)==0,"nonzero residual window padding");return out;
}
std::vector<Stream> forward_impl(const StageRequest& r,const Stream& s,J& side) {
  const auto& a=r.action;const auto& b=s.data;
  if(layout(a))return layout_forward(r,s,side);
  if(reference(a)){parameters(r,{});return reference_forward(s,a,side);}
  if(a=="zlib.level6"){parameters(r,{});return {serial(zlib_encode(b))};}
  if(a=="op.outlier_separation.high_bits.uint32.cpp_v1"){
    parameters(r,{"width"});require(s.width==4&&s.kind==Kind::Numeric,"A111 requires uint32 numeric flow");const auto width=param(r,"width",33);require(width<=32,"A111 width must be [0,32]");auto p=exact_actions::exception_separation::separate(exact_actions::exception_separation::words(b),width);side["width"]=width;return {{Kind::Numeric,4,raw(p.low)},{Kind::Numeric,4,raw(p.positions)},{Kind::Numeric,4,raw(p.high)}};
  }
  if(a=="op.frame_of_reference.minimum.int64_uint64_le.cpp_v1" || a=="op.frame_of_reference.minimum.cpp_v1"){
    parameters(r,{});word64(s);profile(r,{"int64","uint64","bytes"});
    if(r.profile=="int64"){auto p=minimum_for::encode<std::int64_t>(typed<std::int64_t>(b));return {{Kind::Numeric,8,raw(std::vector<std::int64_t>{p.minimum})},{Kind::Numeric,8,raw(p.offsets)}};}
    auto p=minimum_for::encode<U>(typed<U>(b));return {{Kind::Numeric,8,raw(std::vector<U>{p.minimum})},{Kind::Numeric,8,raw(p.offsets)}};
  }
  if(a=="op.dictionary.project_words64.cpp_v1"){
    parameters(r,{"dictionary_order"});word64(s);const auto order=string_param(r,"dictionary_order","first_seen");using O=exact_actions::dictionary_recode::Order;require(order=="first_seen"||order=="value"||order=="frequency","unknown dictionary order");const auto p=exact_actions::dictionary_recode::encode<std::int64_t>(typed<std::int64_t>(b),order=="first_seen"?O::first_seen:order=="value"?O::value:O::frequency);return {{Kind::Numeric,8,raw(p.dictionary)},{Kind::Numeric,8,raw(p.ids)}};
  }
  if(a=="op.zigzag.signed64.raw.cpp_v2"){
    parameters(r,{});word64(s);std::vector<U>v;for(auto x:typed<std::int64_t>(b))v.push_back(compression_operators::zigzag::encode64(x));return {{Kind::Numeric,8,raw(v)}};
  }
  if(a=="op.integer_code.uleb128.reference_values.cpp_v2" || a=="op.integer_code.uleb128_unsigned.cpp_v2"){
    parameters(r,{});word64(s);return {serial(compression_operators::unsigned_leb128::encode(typed<U>(b)))};
  }
  if(a=="op.bitpack_int.max_value_width.cpp_v1"){
    parameters(r,{"bit_order","alignment_bytes","minimum_width","wide_cutoff"});word64(s);auto order=string_param(r,"bit_order","msb");require(order=="msb"||order=="lsb","invalid bit order");auto alignment=param(r,"alignment_bytes",1),min=param(r,"minimum_width",0),cutoff=param(r,"wide_cutoff",64);require((alignment==1||alignment==8)&&min<=64&&cutoff<=64,"invalid bitpack profile");using namespace exact_actions::max_value_bitpack;auto v=typed<U>(b);auto packed=pack_fixed(v,allowed_width(v,min,cutoff),{order=="msb"?BitOrder::msb_first:BitOrder::lsb_first,alignment});side["bit_width"]=packed.width;return {serial(std::move(packed.payload))};
  }
  if(a=="op.xor.previous_word.semantic_ports.cpp_v1"){
    parameters(r,{});word64(s);auto v=typed<U>(b);std::vector<U>first,remaining;if(!v.empty())first.push_back(v[0]);for(std::size_t i=1;i<v.size();++i)remaining.push_back(exact_actions::adjacent_residual::xor_word(v[i],v[i-1]));return {{Kind::Numeric,8,raw(first)},{Kind::Numeric,8,raw(remaining)}};
  }
  if(a=="op.xor.previous_word.cpp_v2"){
    parameters(r,{});profile(r,{"bytes","int64","float64"});
    if(r.profile=="bytes")require(s.kind==Kind::Serial&&s.width==1,"byte previous-word XOR requires a serial byte stream");
    else word64(s);
    Bytes out(b.size());
    if(s.width==1){if(!b.empty())out[0]=b[0];for(std::size_t i=1;i<b.size();++i)out[i]=b[i]^b[i-1];}
    else {auto v=typed<U>(b);std::vector<U>x(v.size());if(!v.empty())x[0]=v[0];for(std::size_t i=1;i<v.size();++i)x[i]=exact_actions::adjacent_residual::xor_word(v[i],v[i-1]);out=raw(x);}
    return {serial(std::move(out))};
  }
  if(a=="op.xor.chimp_history128.cpp_v2"){
    parameters(r,{});profile(r,{"int64","float64"});word64(s);
    namespace chimp=native_graph_v1::chimp_history_xor_v2;
    const auto logical=r.profile=="int64"?chimp::Profile::Int64:chimp::Profile::Float64;
    auto encoded=chimp::encode_bytes(b,logical);side["chimp_history_metadata"]=encoded.metadata;
    return {{Kind::Serial,1,std::move(encoded.reference_slots)},
            {Kind::Numeric,8,std::move(encoded.xor_residuals_le)}};
  }
  if(a=="op.trailing_zero_pack.elf_plus_residual.cpp_v1"){
    parameters(r,{});require(s.kind==Kind::Numeric&&(s.width==4||s.width==8),"Elf+ requires residual numeric words");return {serial(residual_window(b,s.width,false,0))};
  }
  if(a=="op.run_length.byte_optimal.cpp_v1"){
    parameters(r,{"symbol_width"});auto w=param(r,"symbol_width",1);require(w>=1&&w<=8,"invalid RLE symbol width");return {serial(unified_rle::encode(b,w))};
  }
  if(a=="op.eraser.camel.cpp_v1"){
    parameters(r,{});word64(s);profile(r,{"int64","float64","bytes"});auto v=typed<std::int64_t>(b);return {serial(r.profile=="int64"?exact_actions::camel::eraser_forward_int64_numeric(v):exact_actions::camel::eraser_forward(v))};
  }
  if(a=="op.xor.fpc.cpp_v1" || a=="op.xor.pfpc.cpp_v1"){
    parameters(r,{"predictor_lg2"});word64(s);auto n=param(r,"predictor_lg2",16);require(n>=1&&n<=20,"predictor table exceeds bounded profile");return {serial(a=="op.xor.fpc.cpp_v1"?exact_actions::fpc::predictor_xor_forward(b,n):exact_actions::pfpc::predictor_xor_forward(b,n))};
  }
  if(a=="op.scaling.exact_decimal.cpp_v2" || a=="op.scaling.delta_binary_packed.cpp_v1"){
    parameters(r,a=="op.scaling.exact_decimal.cpp_v2"
                   ? std::vector<std::string>{"max_point","io_variant"}
                   : std::vector<std::string>{"max_point"});
    word64(s);profile(r,{"float64","bytes"});auto n=param(r,"max_point",4);
    require(n<=18,"max_point exceeds 18");auto v=typed<double>(b);
    if(a=="op.scaling.delta_binary_packed.cpp_v1")
      return {serial(exact_actions::dbp::scaling_forward(v,n))};
    const auto variant=string_param(r,"io_variant","op.scaling.ts_2diff.cpp_v1");
    require(variant=="op.scaling.ts_2diff.cpp_v1" ||
            variant=="op.scaling.simple8b.cpp_v1",
            "unsupported exact-decimal I/O variant");
    return {serial(variant=="op.scaling.simple8b.cpp_v1"
                       ? exact_actions::simple8b::scaling_forward(v,n)
                       : exact_actions::ts2diff::scaling_forward(v,n))};
  }
  if(a=="op.delta.ts_2diff.cpp_v1"){
    parameters(r,{});if(r.profile=="int64")word64(s);else profile(r,{"scaled_packet"});
    return {serial(r.profile=="int64"?exact_actions::ts2diff::delta_forward_int64(typed<std::int64_t>(b)):exact_actions::ts2diff::delta_forward_scaled(b))};
  }
  if(a=="op.xor.gorilla.cpp_v1" || a=="op.xor.chimp128.cpp_v1"){
    parameters(r,{});word64(s);profile(r,{"int64","float64"});auto v=typed<std::int64_t>(b);
    if(a=="op.xor.gorilla.cpp_v1"){
      // The current Gorilla wire uses one raw 64-bit word as its terminal
      // marker.  Match the complete-method adapter's admissible domain rather
      // than accepting a value that a cold inverse would treat as end-of-data.
      const U marker=r.profile=="int64"?U{0x8000000000000000ULL}
                                           :U{0x7ff8000000000000ULL};
      require(std::none_of(v.begin(),v.end(),[&](std::int64_t value){
        return static_cast<U>(value)==marker;
      }),"Gorilla input contains reserved end-marker bit pattern");
      return {serial(r.profile=="int64"?exact_actions::gorilla::xor_forward_int64(v):exact_actions::gorilla::xor_forward_double_bits(v))};
    }
    return {serial(r.profile=="int64"?exact_actions::chimp128::xor_forward_int64(v):exact_actions::chimp128::xor_forward_double_bits(v))};
  }
  if(a=="op.trailing_zero_pack.gorilla.cpp_v1" || a=="op.trailing_zero_pack.chimp128.cpp_v1"){
    parameters(r,{});profile(r,{"bytes","int64","float64"});
    require(b.size()>=4,"truncated typed XOR packet");
    const std::array<std::uint8_t,4> magic{b[0],b[1],b[2],b[3]};
    const auto is_magic=[&](char c0,char c1,char c2,char c3){
      return magic==std::array<std::uint8_t,4>{static_cast<std::uint8_t>(c0),static_cast<std::uint8_t>(c1),static_cast<std::uint8_t>(c2),static_cast<std::uint8_t>(c3)};
    };
    std::string source_profile;
    if(a=="op.trailing_zero_pack.gorilla.cpp_v1") {
      if(is_magic('G','X','I','1'))source_profile="int64";
      else if(is_magic('G','X','F','1'))source_profile="float64";
      else throw std::invalid_argument("wrong Gorilla XOR packet type");
    } else {
      if(is_magic('C','X','I','1'))source_profile="int64";
      else if(is_magic('C','X','F','1'))source_profile="float64";
      else throw std::invalid_argument("wrong Chimp128 XOR packet type");
    }
    // The frozen policy intentionally exposes ordinary predecessor outputs as
    // logical `bytes`.  Preserve the typed packet discriminator in the stage
    // metadata so cold inverse does not guess from the policy-visible view.
    if(r.profile!="bytes")require(r.profile==source_profile,"typed XOR packet/profile mismatch");
    side["source_profile"]=source_profile;
    return {serial(a=="op.trailing_zero_pack.gorilla.cpp_v1"?exact_actions::gorilla::trailing_pack_forward(b):exact_actions::chimp128::trailing_pack_forward(b))};
  }
  if(a=="op.entropy_special.spdp.cpp_v1"){
    parameters(r,{"level"});require(b.size()>=12,"truncated SPDP block packet");const auto n=param(r,"level",5);require(n<=9,"invalid SPDP level");side["original_size"]=b.size()-12;return {serial(exact_actions::spdp::entropy_forward(b,n))};
  }
  if(legacy(a)){
    parameters(r,{"max_point"});if(a.starts_with("op.scaling.")){word64(s);profile(r,{"int64","float64"});}
    std::optional<int> point;if(r.parameters.contains("max_point")){auto p=param(r,"max_point",0);require(p<=18,"invalid legacy decimal point");point=p;}
    if(a=="op.container_frame.legacy_numeric_limbs.cpp_v1"){require(b.size()>=4,"truncated LNS1");auto packet=J::parse(b.begin()+4,b.end());side["value_count"]=packet.at("blocks").at(0).at("count");}
    return {serial(legacy_numeric_action::forward(legacy_wire_name(a),b,r.profile.empty()?"bytes":r.profile,point))};
  }
  if(native_lz(a)){
    parameters(r,{});if(a.starts_with("op.match_token_format.")){auto c=native_lz_action::context(b);side["lz_context"]={c.role,c.window,c.minimum,c.maximum,c.raw_size};}
    return {serial(native_lz_action::forward(a,b))};
  }
  if(const auto* p=packet(a)){parameters(r,p->parameter_names);if(p->lanes64)word64(s);return {serial(p->encode(b,r,side))};}
  throw std::invalid_argument("UNSUPPORTED_NATIVE_STAGE: "+a);
}


} // namespace (stage helpers)
namespace {
J metadata(const StageRequest& request, const Stream& input, const std::vector<Stream>& outputs, const J& side) {
  J out; out["schema"]="native-stage-metadata-v1";out["action"]=request.action;out["profile"]=request.profile;out["parameters"]=request.parameters;out["input"]=spec(input);out["outputs"]=J::array();for(const auto& x:outputs)out["outputs"].push_back(spec(x));out["side"]=side;return out;
}
Stream stream_from_spec(const J& s, Bytes b) {return restore_shape(s,std::move(b));}

}
const std::vector<StageCapability>& native_stage_capabilities() {
  static const auto caps=[] {std::vector<StageCapability> v;for(const auto& p:packets())v.push_back(p.info);v.push_back({"zlib.level6","cpp zlib","complete zlib level6 stream",false});v.push_back({"columnize","cpp/native_graph_v1","OpenZL layout transpose/split",true});v.push_back({"float_deconstruct","cpp/native_graph_v1","OpenZL float bit split",true});v.push_back({"bitsplit","cpp/native_graph_v1","OpenZL bit split",true});v.push_back({"bitunpack","cpp/native_graph_v1","OpenZL bit unpack",true});v.push_back({"op.frame_of_reference.minimum.int64_uint64_le.cpp_v1","cpp/native_graph_v1","minimum plus uint64 offsets",true});v.push_back({"op.outlier_separation.high_bits.uint32.cpp_v1","cpp/native_graph_v1","low/positions/high uint32 ports",true});v.push_back({"op.dictionary.project_words64.cpp_v1","cpp/native_graph_v1","dictionary plus ids ports",true});v.push_back({"op.zigzag.signed64.raw.cpp_v2","cpp/native_graph_v1","signed int64 zigzag",false});v.push_back({"op.integer_code.uleb128.reference_values.cpp_v2","cpp/native_graph_v1","unsigned ULEB128",false});v.push_back({"op.bitpack_int.max_value_width.cpp_v1","cpp/native_graph_v1","max width bit packing",false});
  for(const char* name : {"op.eraser.camel.cpp_v1","op.xor.fpc.cpp_v1","op.xor.pfpc.cpp_v1","op.scaling.exact_decimal.cpp_v2","op.scaling.delta_binary_packed.cpp_v1","op.delta.ts_2diff.cpp_v1","op.xor.gorilla.cpp_v1","op.xor.chimp128.cpp_v1","op.trailing_zero_pack.gorilla.cpp_v1","op.trailing_zero_pack.chimp128.cpp_v1","op.entropy_special.spdp.cpp_v1","op.run_length.byte_optimal.cpp_v1"}) v.push_back({name,"cpp/native_graph_v1","direct exact typed stage",false});
  for(const auto& name:legacy_names) v.push_back({name,"cpp/src/legacy_numeric_action_stages.cpp","frozen legacy adapter, not canonical ALP/FastPFor",false});
  for(const auto& name:native_lz_names) v.push_back({name,"cpp/src/native_lz_action_stages.cpp","existing aligned-eight parameterized native-LZ",false});
  v.push_back({"op.xor.previous_word.cpp_v2","cpp/src/adjacent_residual.hpp","single-stream adjacent-XOR carrier over byte or 64-bit physical lanes",false});
  v.push_back({"op.xor.previous_word.semantic_ports.cpp_v1","cpp/src/adjacent_residual.hpp","separate first-word and XOR-residual physical ports",true});
  v.push_back({"op.xor.chimp_history128.cpp_v2","cpp/native_graph_v1/chimp_history_xor_v2.cpp","count-driven Chimp history128 low14-key XOR ports without sentinel",true});
  v.push_back({"op.trailing_zero_pack.elf_plus_residual.cpp_v1","cpp/src/residual_window_pack.hpp","exact residual-window packet",false});
  for(auto& c:v)if(reference(c.action)){c.multiple_outputs=true;c.semantics="frozen reference/proxy transform with independent numeric/metadata ports; no hidden zlib";}
  return v;}();return caps;
}
std::string NativeStageDispatcher::identity() const {return "native_graph_v1.stage_dispatch.cpp-in-process";}
StageResult NativeStageDispatcher::forward(const StageRequest& r,const std::vector<Stream>& inputs) const {
  require(std::any_of(native_stage_capabilities().begin(),native_stage_capabilities().end(),[&](const auto& c){return c.action==r.action;}),"UNSUPPORTED_NATIVE_STAGE: "+r.action);
  require(inputs.size()==1,"stage dispatcher requires exactly one input; Concat is graph built-in");shape(inputs[0]);J side=J::object();auto outputs=forward_impl(r,inputs[0],side);for(const auto& x:outputs)shape(x);auto meta=metadata(r,inputs[0],outputs,side); const auto dumped=meta.dump();
  // Frozen policy view contract from _apply_graph_operator: physical layout
  // remains in Stream.kind/width while ordinary policy outputs default to bytes.
  std::vector<std::string> views(outputs.size(), "bytes");
  if(r.action=="op.xor.previous_word.semantic_ports.cpp_v1")
    views={"uint64", "xor_residual64"};
  if(r.action=="op.outlier_separation.high_bits.uint32.cpp_v1")
    views.assign(outputs.size(), "uint32");
  return {std::move(outputs),Bytes(dumped.begin(),dumped.end()),std::move(views)};
}
std::vector<Stream> NativeStageDispatcher::inverse(const StageRequest& r,const std::vector<Stream>& outputs,std::span<const std::uint8_t> metadata_bytes) const {
  require(std::any_of(native_stage_capabilities().begin(),native_stage_capabilities().end(),[&](const auto& c){return c.action==r.action;}),"UNSUPPORTED_NATIVE_STAGE: "+r.action);
  require(metadata_bytes.size()<=16*1024*1024,"stage metadata exceeds bound"); J m=J::parse(metadata_bytes.begin(),metadata_bytes.end()); require(m.at("action")==r.action && m.at("profile")==r.profile && m.at("parameters")==J(r.parameters),"stage request/metadata mismatch");require(m.at("schema")=="native-stage-metadata-v1","invalid native stage metadata");require(m.at("outputs").size()==outputs.size(),"stage output port count mismatch");for(std::size_t i=0;i<outputs.size();++i){shape(outputs[i]);require(spec(outputs[i])==m.at("outputs")[i],"stage output shape changed");}
  const auto& a=r.action; require(!outputs.empty(),"stage outputs are missing");
  if(layout(a)) return {stream_from_spec(m.at("input"),layout_inverse(outputs,m.at("side")))};
  if(reference(a)) return {stream_from_spec(m.at("input"),reference_inverse(outputs,a,m.at("side")))};
  if(a=="zlib.level6"){require(outputs.size()==1,"zlib output count");return {stream_from_spec(m.at("input"),zlib_decode(outputs[0].data,m.at("input").at("bytes")))};}
  if(a=="op.frame_of_reference.minimum.int64_uint64_le.cpp_v1" || a=="op.frame_of_reference.minimum.cpp_v1") {require(outputs.size()==2,"FOR output ports");auto in=m.at("input");if(r.profile=="int64"){auto base=typed<std::int64_t>(outputs[0].data);auto off=typed<std::uint64_t>(outputs[1].data);require(base.size()==1,"FOR base port");return {stream_from_spec(in,raw(minimum_for::decode(base[0],off)))};}auto base=typed<U>(outputs[0].data);auto off=typed<U>(outputs[1].data);require(base.size()==1,"FOR base port");return {stream_from_spec(in,raw(minimum_for::decode(base[0],off)))};}
  if(a=="op.outlier_separation.high_bits.uint32.cpp_v1"){require(outputs.size()==3,"A111 output ports");auto low=exact_actions::exception_separation::words(outputs[0].data),pos=exact_actions::exception_separation::words(outputs[1].data),high=exact_actions::exception_separation::words(outputs[2].data);exact_actions::exception_separation::Parts p{m.at("side").at("width").get<unsigned>(),std::move(low),std::move(pos),std::move(high)};return {stream_from_spec(m.at("input"),exact_actions::exception_separation::raw_bytes(exact_actions::exception_separation::restore(p)))};}
  if(a=="op.dictionary.project_words64.cpp_v1"){require(outputs.size()==2,"dictionary output ports");auto dict=typed<std::int64_t>(outputs[0].data); auto ids=typed<U>(outputs[1].data); return {stream_from_spec(m.at("input"),raw(exact_actions::dictionary_recode::decode(std::span<const std::int64_t>(dict),std::span<const U>(ids))))};}
  if(a=="op.zigzag.signed64.raw.cpp_v2"){auto v=typed<U>(outputs[0].data);std::vector<std::int64_t>out;for(auto x:v)out.push_back(compression_operators::zigzag::decode64(x));return {stream_from_spec(m.at("input"),raw(out))};}
  if(a=="op.integer_code.uleb128.reference_values.cpp_v2" || a=="op.integer_code.uleb128_unsigned.cpp_v2"){auto v=compression_operators::unsigned_leb128::decode(outputs[0].data);return {stream_from_spec(m.at("input"),raw(v))};}
  if(a=="op.bitpack_int.max_value_width.cpp_v1"){auto in=m.at("input");const auto n=in.at("bytes").get<U>()/8;auto pbits=m.at("side").at("bit_width").get<unsigned>();auto order=string_param(r,"bit_order","msb");auto alignment=param(r,"alignment_bytes",1);auto vals=exact_actions::max_value_bitpack::unpack(outputs[0].data,n,pbits,{order=="msb"?exact_actions::max_value_bitpack::BitOrder::msb_first:exact_actions::max_value_bitpack::BitOrder::lsb_first,alignment});return {stream_from_spec(in,raw(vals))};}
  if(a=="op.run_length.byte_optimal.cpp_v1"){return {stream_from_spec(m.at("input"),unified_rle::decode(outputs[0].data))};}
  if(a=="op.eraser.camel.cpp_v1"){auto b=outputs[0].data;return {stream_from_spec(m.at("input"),r.profile=="int64"?raw(exact_actions::camel::eraser_inverse_int64_numeric(b)):raw(exact_actions::camel::eraser_inverse(b)))};}
  if(a=="op.xor.fpc.cpp_v1"||a=="op.xor.pfpc.cpp_v1"){return {stream_from_spec(m.at("input"),a=="op.xor.fpc.cpp_v1"?exact_actions::fpc::predictor_xor_inverse(outputs[0].data):exact_actions::pfpc::predictor_xor_inverse(outputs[0].data))};}
  if(a=="op.scaling.exact_decimal.cpp_v2"||a=="op.scaling.delta_binary_packed.cpp_v1"){
    Bytes v;
    if(a=="op.scaling.delta_binary_packed.cpp_v1")
      v=raw(exact_actions::dbp::scaling_inverse(outputs[0].data));
    else {
      const auto variant=string_param(r,"io_variant","op.scaling.ts_2diff.cpp_v1");
      require(variant=="op.scaling.ts_2diff.cpp_v1" ||
              variant=="op.scaling.simple8b.cpp_v1",
              "unsupported exact-decimal I/O variant");
      v=variant=="op.scaling.simple8b.cpp_v1"
            ? raw(exact_actions::simple8b::scaling_inverse(outputs[0].data))
            : raw(exact_actions::ts2diff::scaling_inverse(outputs[0].data));
    }
    return {stream_from_spec(m.at("input"),v)};
  }
  if(a=="op.delta.ts_2diff.cpp_v1"){Bytes b = r.profile=="int64" ? raw(exact_actions::ts2diff::delta_inverse_int64(outputs[0].data)) : exact_actions::ts2diff::delta_inverse_scaled(outputs[0].data); return {stream_from_spec(m.at("input"),std::move(b))};}
  if(a=="op.xor.gorilla.cpp_v1"||a=="op.xor.chimp128.cpp_v1"){auto b=a=="op.xor.gorilla.cpp_v1"?raw(exact_actions::gorilla::xor_inverse(outputs[0].data)):raw(exact_actions::chimp128::xor_inverse(outputs[0].data));return {stream_from_spec(m.at("input"),b)};}
  if(a=="op.trailing_zero_pack.gorilla.cpp_v1" || a=="op.trailing_zero_pack.chimp128.cpp_v1") {
    const auto source_profile=m.at("side").at("source_profile").get<std::string>();
    require(source_profile=="int64"||source_profile=="float64","invalid typed XOR source profile");
    if(r.profile!="bytes")require(r.profile==source_profile,"typed XOR packet/profile mismatch");
    Bytes b;if(a=="op.trailing_zero_pack.gorilla.cpp_v1") b=source_profile=="int64"?exact_actions::gorilla::trailing_pack_inverse_int64(outputs[0].data):exact_actions::gorilla::trailing_pack_inverse_double(outputs[0].data);
    else b=source_profile=="int64"?exact_actions::chimp128::trailing_pack_inverse_int64(outputs[0].data):exact_actions::chimp128::trailing_pack_inverse_double(outputs[0].data);
    return {stream_from_spec(m.at("input"),std::move(b))};
  }
  if(a=="op.trailing_zero_pack.elf_plus_residual.cpp_v1")return {stream_from_spec(m.at("input"),residual_window(outputs[0].data,m.at("input").at("width").get<unsigned>(),true,m.at("input").at("bytes").get<U>()/m.at("input").at("width").get<U>()))};
  if(a=="op.xor.previous_word.semantic_ports.cpp_v1") {
    require(outputs.size()==2,"adjacent XOR ports missing");auto first=typed<U>(outputs[0].data),tail=typed<U>(outputs[1].data);require(first.size()<=1 && (!first.empty()||tail.empty()),"invalid XOR first-word port");auto v=first;for(auto residual:tail)v.push_back(residual^v.back());return {stream_from_spec(m.at("input"),raw(v))};
  }
  if(a=="op.xor.previous_word.cpp_v2") {
    require(outputs.size()==1,"previous-word XOR requires one carrier");
    const auto input=m.at("input");const auto width=input.at("width").get<unsigned>();const auto& b=outputs[0].data;Bytes restored(b.size());
    if(width==1){if(!b.empty())restored[0]=b[0];for(std::size_t i=1;i<b.size();++i)restored[i]=b[i]^restored[i-1];}
    else {require(width==8,"previous-word XOR metadata width invalid");auto x=typed<U>(b);std::vector<U>v(x.size());if(!x.empty())v[0]=x[0];for(std::size_t i=1;i<x.size();++i)v[i]=x[i]^v[i-1];restored=raw(v);}
    return {stream_from_spec(input,std::move(restored))};
  }
  if(a=="op.xor.chimp_history128.cpp_v2") {
    require(outputs.size()==2,"Chimp history128 requires two ports");
    namespace chimp=native_graph_v1::chimp_history_xor_v2;
    const auto logical=r.profile=="int64"?chimp::Profile::Int64:chimp::Profile::Float64;
    chimp::StagePayload payload{m.at("side").at("chimp_history_metadata").get<Bytes>(),
                                outputs[0].data,outputs[1].data};
    return {stream_from_spec(m.at("input"),chimp::decode_bytes(payload,logical))};
  }
  if(a=="op.entropy_special.spdp.cpp_v1"){auto b=exact_actions::spdp::entropy_inverse(outputs[0].data,m.at("side").at("original_size").get<U>(),param(r,"level",5));return {stream_from_spec(m.at("input"),b)};}
  if(legacy(a)) {return {stream_from_spec(m.at("input"),legacy_numeric_action::inverse(legacy_wire_name(a),outputs[0].data,m.at("side").value("value_count",U{0})))};}
  if(native_lz(a)){auto c=m.at("side").value("lz_context",J::array());native_lz_action::Context ctx{};if(!c.empty()){require(c.size()==5,"LZ inverse context size");ctx={c[0],c[1],c[2],c[3],c[4]};}return {stream_from_spec(m.at("input"),native_lz_action::inverse(a,outputs[0].data,c.empty()?nullptr:&ctx))};}
  if(const auto* p=packet(a))return {stream_from_spec(m.at("input"),p->decode(outputs[0].data,r,const_cast<J&>(m["side"])))};
  throw std::invalid_argument("UNSUPPORTED_NATIVE_STAGE: "+a);
}
std::unique_ptr<StageDispatcher> make_default_dispatcher(){return std::make_unique<NativeStageDispatcher>();}
} // namespace native_graph_v1
