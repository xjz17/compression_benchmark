#include "baseline.hpp"
#include "decimal_scaling.hpp"
#include "graph_internal.hpp"
#include "chimp128_exact_stages.hpp"
#include "gorilla_exact_stages.hpp"
#include "pongo_exact_stages.hpp"
#include "ts2diff_exact_stages.hpp"
#include "fpc_exact_stages.hpp"
#include "pfpc_exact_stages.hpp"
#include "spdp_exact_stages.hpp"
#include "buff_native_exact_stages.hpp"
#include "camel_exact_stages.hpp"
#include "elf_double_codec.hpp"
#include "dbp_exact_stages.hpp"
#include "optimal_rle.hpp"
#include "dictionary_recode.hpp"
#include "ans_rans.hpp"
#include "huffman_bytes.hpp"
#include "arithmetic_bytes.hpp"
#include "bos_exact_stages.hpp"
#include "fastpfor_exact_stages.hpp"
#include "fpzip_exact_stages.hpp"
#include "learning_codec_exact_stages.hpp"
#include "p04_exact_stages.hpp"
#include "simple8b_exact_stages.hpp"
#include "svb_exact_stages.hpp"
#include "vendored_lightweight_codecs_v1.hpp"
#include "upstream_general_a.hpp"
#include "upstream_general_b.hpp"
#include "upstream_legacy.hpp"
#include "upstream_specialists.hpp"
#ifdef BASELINE86_BLOSC1
#include "blosc_codec.hpp"
#include <blosc.h>
#endif
#ifdef BASELINE86_BLOSC2
#include "blosc2_codec.hpp"
#include <blosc2.h>
#endif
#include <zlib.h>
#ifdef BASELINE86_ZSTD
#include <zstd.h>
#endif
#ifdef BASELINE86_BROTLI
#include <brotli/encode.h>
#include <brotli/decode.h>
#endif
#ifdef BASELINE86_LZ4
#include <lz4frame.h>
#include <lz4.h>
#include <lz4hc.h>
#endif
#ifdef BASELINE86_SNAPPY
#include <snappy.h>
#endif
#ifdef BASELINE86_LZMA
#include <lzma.h>
#endif
#ifdef BASELINE86_BZIP2
#include <bzlib.h>
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

// The current CORAD lossless translation unit predates a public header.  Its
// functions are linked by native_graph_v1_stages; these declarations expose
// only the stable whole-chain pieces used by this component.
namespace corad_lossless_cpp {
using Bytes = std::vector<std::uint8_t>;
Bytes normalize_forward(const Bytes&, const std::string&);
Bytes normalize_inverse(const Bytes&);
Bytes tricklet_forward(const Bytes&);
Bytes tricklet_inverse(const Bytes&);
Bytes entropy_forward(const Bytes&, int);
Bytes entropy_inverse(const Bytes&);
Bytes frame_forward(const Bytes&);
Bytes frame_inverse(const Bytes&);
}
namespace native_baseline86_v1 {
namespace {
using J=nlohmann::json;
using namespace native_graph_v1;
constexpr std::size_t maximum=256ULL<<20;
void require(bool ok,const std::string& message){if(!ok)throw std::invalid_argument(message);}
static_assert(std::endian::native==std::endian::little,"existing typed kernels require little-endian host");
const std::vector<std::string> all_dtypes={"int8","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64"};
std::vector<Record> make_inventory(){
  const std::vector<std::string> labels={
    "Chimp","Chimp128","Pongo","Gorilla","2DIFF","BOS","REGER","SVB","FastPFor","Simple8b","RLE","RLBE","Sprintz","Dict.","DBP","ALP","LECO","Camel","Elf","Elf+","FPC","PFPC","SPDP","CORAD","BUFF","FPZIP","FPCmp-R","FPCmp-S",
    "gzip","Brotli","Zstd","LZ4","XZ","Density","Gipfeli","GLZA","LZMAT","LZO1","LZO2A","LZRW","LZSSE","LZVN","Pithy","Shrinker","NRV","WFLZ","XPACK","YALZ77","Yappy","Snappy","ANS","FSE","Huffman","Huff0","Arithmetic","Blosc","Blosc2","BriefLZ","BSC","Bzip2","Bzip3","Crush","CSC","FastLZ","FastLZMA2","Kanzi","Libdeflate","Lizard","LZ4 Fast","LZ4HC","LZAV","LZF","LZFSE","LZG","LZHAM","LZJB","Lzlib","PPMD8","QuickLZ","SLZ","Tornado","Zlib","Zlib-ng","Zling","ZXC","TAMP"};
  require(labels.size()==86 && std::set(labels.begin(),labels.end()).size()==86,"baseline catalog changed");
  std::vector<Record> r;
  for(const auto& label:labels)r.push_back({label,"unconnected", "", "not_admitted", "No verified in-process whole-method adapter in this component; historical CLI/proxy/best-level registration is not admission.",{},false,false,"unsupported",false});
  auto add=[&](std::string name,std::string impl,std::string source,std::string config,std::vector<std::string> types,bool shared,std::string classification="direct_current",bool primary=true,std::string limit="No full-cohort or performance claim; data-dependent failures remain explicit."){
    auto it=std::find_if(r.begin(),r.end(),[&](const Record& v){return v.label==name;});require(it!=r.end(),"unknown catalog label");
    *it={name,impl,source,config,limit,types,true,shared,classification,primary};
  };
  const std::vector<std::string> lanes={"int64","float64"};
  add("Chimp128","project_tsfile_chimp128","cpp/src/chimp128_exact_stages.cpp","tsfile_chimp128_compose_default",lanes,true,"direct_current",true,"The current wire reserves INT64_MIN and float64 bits 0x7ff8000000000000 as end markers, so blocks containing those values are rejected.");
  add("Pongo","project_exact_pongo","cpp/src/pongo_codec.cpp","compose_default",lanes,false);
  add("Gorilla","project_tsfile_gorilla","cpp/src/gorilla_exact_stages.cpp","compose_default",lanes,false,"direct_current",true,"The current wire reserves INT64_MIN and float64 bits 0x7ff8000000000000 as end markers, so blocks containing those values are rejected.");
  add("2DIFF","project_tsfile_2diff","cpp/src/ts2diff_exact_stages.cpp","compose_int64_default",{"int64"},false,"direct_current",true,"Only int64; no decimal scaling or dtype widening. A block is rejected when an adjacent signed delta cannot be represented in int64.");
  add("BOS","project_bos64_adapter","cpp/src/bos_exact_stages.cpp","adapter_compose_int64",{"int64"},false);
  add("REGER","project_reger_exact","cpp/src/learning_codec_exact_stages.cpp","reorder_delta_zigzag_for_bitpack_residual_core",{"int64"},false);
  add("SVB","project_streamvbyte_int64","cpp/src/svb_exact_stages.cpp","compose_int64_default",{"int64"},false);
  add("FastPFor","lemire_fastpfor_composite","cpp/src/fastpfor_exact_stages.cpp + external/compression/FastPFor","FastPFor8_prefix_then_variablebyte_tail",{"int64"},false);
  add("Simple8b","lemire_simple8b_int64","cpp/src/simple8b_exact_stages.cpp + external/compression/FastPFor","bitpack_int64_default",{"int64"},false);
  add("DBP","project_dbp","cpp/src/dbp_exact_stages.cpp","compose_int64_default",{"int64"},false,"direct_current",true,"Only int64; no decimal scaling or dtype widening.");
  add("RLBE","project_tsfile_rlbe","cpp/src/p04_exact_stages.cpp","rlbe_compose_int64",{"int64"},false);
  add("Sprintz","project_tsfile_sprintz","cpp/src/p04_exact_stages.cpp","sprintz_current_int64_or_float64_bits",lanes,false);
  add("ALP","project_alp_exact","cpp/src/learning_codec_exact_stages.cpp","scaling_for_bitpack_residual_core",{"float64"},false);
  add("LECO","project_leco_exact","cpp/src/learning_codec_exact_stages.cpp","regression_zigzag_bitpack",{"int64"},false);
  add("Camel","project_exact_camel_float64","cpp/src/camel_exact_stages.cpp","compose_float64_bits_default_with_literal_escape",{"float64"},false,"direct_current",true,"Float64 only. Non-finite values and decimal integer jumps outside the 16-bit Camel delta field use the bit-exact literal escape.");
  for(auto label:{"Elf","Elf+"})add(label,"project_elf64","cpp/src/elf_double_codec.cpp",label==std::string("Elf")?"EL64_plus_false":"EL64_plus_true",{"float64"},false,"direct_current",true,"NaN payloads are unsupported by the existing EL64 sentinel wire; rejection required.");
  add("FPC","project_fpc","cpp/src/fpc_exact_stages.cpp","predictor_lg2_16",{"float64"},false);
  add("PFPC","project_pfpc","cpp/src/pfpc_exact_stages.cpp","predictor_lg2_16_single_thread",{"float64"},false);
  add("SPDP","project_spdp","cpp/src/spdp_exact_stages.cpp","level_5",{"float64"},false);
  add("FPZIP","project_fpzip_lossless_1d","cpp/src/fpzip_exact_stages.cpp + external/compression/fpzip","lossless_1d_float64",{"float64"},false);
  for(auto label:{"Zlib","gzip"})add(label,"zlib_library_"+std::string(zlibVersion()),"zlib.h / linked zlib C library",label==std::string("Zlib")?"level6_window15_mem8_strategy0":"level6_window31_mem8_strategy0",all_dtypes,false,"direct_current",true,"New predeclared direct library profile; historical composed reference and best-level results are not reused.");
#ifdef BASELINE86_ZSTD
  add("Zstd","libzstd_"+std::string(ZSTD_versionString()),"zstd.h / linked Zstd C library","level3_single_thread",all_dtypes,false);
#endif
#ifdef BASELINE86_BROTLI
  add("Brotli","libbrotli_"+std::to_string(BrotliEncoderVersion()),"brotli/encode.h / linked Brotli C library","quality5_lgwin22_generic",all_dtypes,false);
#endif
#ifdef BASELINE86_LZ4
  add("LZ4","liblz4_"+std::string(LZ4_versionString()),"lz4frame.h / linked LZ4 C library","frame_level0_default_single_thread",all_dtypes,false);
  add("LZ4 Fast","liblz4_"+std::string(LZ4_versionString()),"lz4.h / linked LZ4 C library","raw_acceleration1_single_thread",all_dtypes,false,"direct_current",true,"Fresh fixed library default; the historical post-hoc lz4fast_17 key is not reused.");
  add("LZ4HC","liblz4_"+std::string(LZ4_versionString()),"lz4hc.h / linked LZ4 C library","raw_HC_level9_single_thread",all_dtypes,false,"direct_current",true,"Fresh fixed library default; the historical post-hoc lz4hc_12 key is not reused.");
#endif
#ifdef BASELINE86_SNAPPY
  add("Snappy","libsnappy_raw","snappy.h / linked Snappy C++ library","raw_default",all_dtypes,false);
#endif
#ifdef BASELINE86_LZMA
  add("XZ","liblzma_"+std::string(lzma_version_string()),"lzma.h / linked liblzma C library","preset6_crc64_single_thread",all_dtypes,false);
#endif
#ifdef BASELINE86_BZIP2
  add("Bzip2","libbz2_"+std::string(BZ2_bzlibVersion()),"bzlib.h / linked Bzip2 C library","block_size100k_9_work_factor30",all_dtypes,false);
#endif
#ifdef BASELINE86_BLOSC1
  add("Blosc","c_blosc_"+std::string(BLOSC_VERSION_STRING),"cpp/src/blosc_codec.cpp + external/compression/c-blosc","blosclz_clevel5_shuffle_single_thread",all_dtypes,false);
#endif
#ifdef BASELINE86_BLOSC2
  add("Blosc2","c_blosc2_"+std::string(BLOSC2_VERSION_STRING),"cpp/src/blosc2_codec.cpp + external/compression/c-blosc2","blosclz_clevel5_shuffle_single_thread",all_dtypes,false);
#endif
  const std::string fixed_fallback =
      "_adapter_tagged_raw_fallback_when_upstream_not_smaller";
  add("Libdeflate","libdeflate_1.25_deflate",
      "external/compression/lzbench/lz/libdeflate",
      "raw_deflate_level6"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed upstream profile; historical post-hoc levels are not reused. The adapter stores verbatim bytes when the upstream result is not smaller.");
  add("FastLZ","fastlz_0.5.0",
      "external/compression/lzbench/lz/fastlz",
      "level1"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Upstream requires at least 16 input bytes; shorter inputs use the declared verbatim adapter branch.");
  add("LZF","liblzf_3.6_very",
      "external/compression/lzbench/lz/lzf",
      "compress_very"+fixed_fallback,all_dtypes,false);
  add("LZFSE","apple_lzfse_2017-03-08",
      "external/compression/lzbench/lz/lzfse",
      "buffer_api_default"+fixed_fallback,all_dtypes,false);
  add("LZVN","apple_lzvn_2017-03-08",
      "external/compression/lzbench/lz/lzfse",
      "buffer_api_default"+fixed_fallback,all_dtypes,false);
  add("BriefLZ","brieflz_1.3.0",
      "external/compression/lzbench/lz/brieflz",
      "level1"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed fastest upstream level; no historical corpus-selected level is reused.");
  add("LZAV","lzav_5.7",
      "external/compression/lzbench/lz/lzav/lzav.h",
      "compress_default"+fixed_fallback,all_dtypes,false);
  add("LZG","liblzg_1.0.10",
      "external/compression/lzbench/lz/liblzg",
      "library_default_level5_fast"+fixed_fallback,all_dtypes,false);
  add("LZJB","opensolaris_lzjb_2010",
      "external/compression/lzbench/lz/lzjb",
      "lzjb2010_n0"+fixed_fallback,all_dtypes,false);
  add("LZMAT","lzmat_1.01",
      "external/compression/lzbench/lz/lzmat",
      "ansi_c_default"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed upstream profile; no historical corpus-selected parameters are reused. lzbench flags its generic wrapper as buggy, so this adapter uses exact uint32 bounds and remains subject to explicit per-block failure handling.");
  add("LZRW","lzrw1_1991",
      "external/compression/lzbench/lz/lzrw/lzrw1.c",
      "lzrw1_default"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed LZRW1 profile. Inputs shorter than 32 bytes use the declared verbatim branch.");
  add("WFLZ","wflz_fast",
      "external/compression/lzbench/lz/wflz/wfLZ.c",
      "compress_fast_same_endian"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Same-endian wfLZ Fast wire. lzbench flags its generic wrapper as compiler-sensitive; this adapter validates its 16-byte header and exact sizes, and data-dependent failures remain explicit.");
  add("Yappy","yappy_default",
      "external/compression/lzbench/lz/yappy/yappy.cpp",
      "default_level10"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "The upstream vectorized kernel reads/writes full lanes; this adapter supplies private 32-byte input/output padding and stores only the logical stream.");
  add("YALZ77","yalz77_default",
      "external/compression/lzbench/lz/yalz77/lz77.h",
      "searchlen12_block64KiB"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "The decoder is constructed with the exact expected-size ceiling and rejects incomplete/trailing streams; lzbench flags its generic wrapper as buggy, so data-dependent failures remain explicit.");
  add("QuickLZ","quicklz_1.5.1b7",
      "external/compression/lzbench/lz/quicklz/quicklz151b7.c",
      "level1_no_streaming"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed upstream profile. Inputs shorter than 16 bytes use the declared verbatim branch.");
  add("SLZ","slz_1.2.1",
      "external/compression/lzbench/lz/slz/src/slz.c",
      "gzip_level1_window32KiB"+fixed_fallback,all_dtypes,false);
  add("TAMP","tamp_2.1.0",
      "external/compression/lzbench/lz/tamp",
      "window10_literal8_single_stream"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed 1 KiB window and 8-bit literal profile; no historical corpus-selected window is reused.");
  add("Bzip3","bzip3_1.5.2",
      "external/compression/lzbench/bwt/bzip3",
      "high_level_block_size_1MiB"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Fresh fixed minimum documented lzbench block profile; high-level API remains single-threaded and no historical corpus-selected level is reused.");
  add("FSE","finite_state_entropy_0.9.0",
      "external/compression/lzbench/lz/lizard/entropy",
      "simple_default_128KiB_chunks"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Baseline-private prefixed build of the vendored FSE source; fixed chunking handles arbitrary admitted block sizes and does not reuse historical parameters.");
  add("Huff0","huff0_fse_0.9.0",
      "external/compression/lzbench/lz/lizard/entropy",
      "simple_default_128KiB_chunks"+fixed_fallback,all_dtypes,false,"direct_current",true,
      "Baseline-private prefixed build of the vendored Huff0 source; fixed chunking handles arbitrary admitted block sizes and does not reuse historical parameters.");
  auto explain_unsupported = [&](const std::string& name,
                                 const std::string& source,
                                 const std::string& reason) {
    auto it=std::find_if(r.begin(),r.end(),[&](const Record& v){return v.label==name;});
    require(it!=r.end(),"unknown unsupported catalog label");
    require(!it->linked,"unsupported explanation applied to linked method");
    it->source=source;
    it->limitations=reason;
  };
  // v3 source migrations replace the historical unsupported/proxy records.
  // Availability is a build property; bit-exact applicability remains checked
  // per input and must never be replaced with a different codec on failure.
  for (const auto& d : upstream_general_a::inventory()) {
    if (d.available)
      add(d.label,d.implementation,d.source,d.configuration,all_dtypes,false,
          "direct_current",true,d.limitations);
    else
      explain_unsupported(d.label,d.source,d.limitations);
  }
  for (const auto& d : upstream_general_b::descriptors())
    add(std::string(d.label),std::string(d.implementation),std::string(d.source),
        std::string(d.configuration),all_dtypes,false,"direct_current",true,
        std::string(d.limitations));
  for (const auto& d : upstream_specialists::profiles())
    add(d.label,d.implementation,d.source,d.configuration,d.dtypes,
        d.admission==upstream_specialists::Admission::direct_project_generic,
        d.admission==upstream_specialists::Admission::direct_exact_subset
          ? "direct_exact_subset"
          : (d.admission==upstream_specialists::Admission::direct_project_generic
             ? "direct_generic" : "direct_current"),true,d.limitations);
  const std::string legacy="external/operator_ablation_sources/lzbench_legacy/upstream/lz/";
  add("Pithy","pithy_lzbench_legacy_direct",legacy+"pithy",
      "level6_tagged_raw_when_not_smaller",all_dtypes,false,"direct_current",true,
      "Historical direct source restored from 192; ablation mode NONE, one thread, complete stream. Private input/output padding is inside timing.");
  add("Shrinker","shrinker_lzbench_legacy_direct",legacy+"shrinker",
      "default_zero_initialized_hash_tagged_raw_when_not_smaller",all_dtypes,false,"direct_current",true,
      "Historical direct source restored from 192; hash table initialization fixed. Inputs below 32 bytes or at least 128 MiB use explicit verbatim representation; bounded stream validation precedes the upstream inverse.");
  add("XPACK","xpack_lzbench_legacy_direct",legacy+"xpack",
      "level6_tagged_raw_when_not_smaller",all_dtypes,false,"direct_current",true,
      "Historical direct source restored from 192; fixed level 6, ablation mode NONE, one thread, exact-size decode.");
  return r;
}
std::uint64_t natural(const std::string& s){std::uint64_t n=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);require(!s.empty()&&e==std::errc{}&&p==s.data()+s.size(),"invalid unsigned field");return n;}
template<class T>std::vector<T> words(const Bytes& b){require(b.size()%sizeof(T)==0,"unaligned typed values");std::vector<T>v(b.size()/sizeof(T));if(!b.empty())std::memcpy(v.data(),b.data(),b.size());return v;}
template<class T>Bytes raw(const std::vector<T>&v){require(v.size()<=maximum/sizeof(T),"decoded bytes exceed bound");Bytes b(v.size()*sizeof(T));if(!b.empty())std::memcpy(b.data(),v.data(),b.size());return b;}
bool contains_word(const Bytes& bytes, std::uint64_t needle) {
  require(bytes.size() % sizeof(std::uint64_t) == 0, "unaligned 64-bit values");
  for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(needle)) {
    std::uint64_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if (word == needle) return true;
  }
  return false;
}
void reject_reserved_end_marker(const Record& record, const Block& block) {
  const auto marker = block.dtype == "int64" ? UINT64_C(0x8000000000000000)
                                               : UINT64_C(0x7ff8000000000000);
  if (contains_word(block.values, marker))
    throw Unsupported(record.label + ": input contains the reserved end-marker bit pattern");
}
void put(Bytes& b,std::uint64_t n,unsigned width){for(unsigned i=0;i<width;++i)b.push_back(std::uint8_t(n>>(8*i)));}
std::uint64_t get(std::span<const std::uint8_t>b,std::size_t&p,unsigned width){require(p<=b.size()&&width<=b.size()-p,"truncated dictionary");std::uint64_t n=0;for(unsigned i=0;i<width;++i)n|=std::uint64_t(b[p++])<<(8*i);return n;}
Bytes dictionary_encode(const Bytes& b){auto v=words<std::uint64_t>(b);auto e=exact_actions::dictionary_recode::encode<std::uint64_t>(v,exact_actions::dictionary_recode::Order::first_seen);Bytes out;put(out,e.dictionary.size(),4);for(auto x:e.dictionary)put(out,x,8);put(out,e.ids.size(),4);for(auto x:e.ids)put(out,x,8);return out;}
Bytes dictionary_decode(const Bytes&b,std::size_t expected){std::size_t p=0;auto count=get(b,p,4);require(count<=(b.size()-p)/8,"oversized dictionary");std::vector<std::uint64_t> d;for(std::uint64_t i=0;i<count;++i)d.push_back(get(b,p,8));count=get(b,p,4);require(count==expected/8 && expected%8==0 && count==(b.size()-p)/8 && (b.size()-p)%8==0,"dictionary indices mismatch");std::vector<std::uint64_t>ids;for(std::uint64_t i=0;i<count;++i)ids.push_back(get(b,p,8));return raw(exact_actions::dictionary_recode::decode<std::uint64_t>(d,ids));}
Bytes zlib_encode(const Bytes&b,bool gzip){
  z_stream z{};require(deflateInit2(&z,6,Z_DEFLATED,gzip?31:15,8,Z_DEFAULT_STRATEGY)==Z_OK,"deflate init");
  Bytes out(deflateBound(&z,b.size()));z.next_in=const_cast<Bytef*>(b.empty()?reinterpret_cast<const Bytef*>(""):b.data());z.avail_in=b.size();z.next_out=out.data();z.avail_out=out.size();const auto status=deflate(&z,Z_FINISH);auto n=z.total_out;deflateEnd(&z);require(status==Z_STREAM_END,"deflate failed");out.resize(n);return out;
}
Bytes zlib_decode(const Bytes&b,std::size_t expected,bool gzip){z_stream z{};require(inflateInit2(&z,gzip?31:15)==Z_OK,"inflate init");Bytes out(expected+1);z.next_in=const_cast<Bytef*>(b.data());z.avail_in=b.size();z.next_out=out.data();z.avail_out=out.size();auto status=inflate(&z,Z_FINISH);auto n=z.total_out;auto left=z.avail_in;inflateEnd(&z);require(status==Z_STREAM_END&&n==expected&&left==0,"invalid zlib/gzip wire");out.resize(n);return out;}
void eligible(const Record&r,const std::string&dtype){if(!r.linked)throw Unsupported(r.label+": in-process backend not connected");if(std::find(r.dtypes.begin(),r.dtypes.end(),dtype)==r.dtypes.end())throw Unsupported(r.label+": dtype "+dtype+" not admitted");}
Bytes compress(const Record&r,const Block&b){
  const auto&n=r.label;const auto&v=b.values;
  // A valid all-missing block has no numeric payload.  Some upstream methods
  // reject empty input, so the adapter uses one canonical empty method payload;
  // NGF1 still carries the selected method, row count and complete bitmap.
  if(v.empty())return {};
  for (const auto& d : upstream_general_a::inventory())
    if (n==d.label) return upstream_general_a::encode(n,v);
  if(upstream_general_b::supports(n))return upstream_general_b::encode(n,v);
  if(upstream_legacy::supports(n))return upstream_legacy::encode(n,v);
  for (const auto& d : upstream_specialists::profiles())
    if (n==d.label) {
      try {return upstream_specialists::encode(n,v);}
      catch(const upstream_specialists::NotApplicable& e){throw Unsupported(e.what());}
    }
  if(n=="Chimp128"){reject_reserved_end_marker(r,b);return b.dtype=="int64"?exact_actions::chimp128::compose_int64(words<std::int64_t>(v)):exact_actions::chimp128::compose_double_bits(words<std::int64_t>(v));}
  if(n=="Gorilla"){reject_reserved_end_marker(r,b);return b.dtype=="int64"?exact_actions::gorilla::compose_int64(words<std::int64_t>(v)):exact_actions::gorilla::compose_double_bits(words<std::int64_t>(v));}
  if(n=="Pongo")return exact_actions::pongo::compose(v);
  if(n=="2DIFF")return exact_actions::ts2diff::compose_int64(words<std::int64_t>(v));
  if(n=="BOS")return exact_actions::bos::adapter_compose(words<std::int64_t>(v));
  if(n=="REGER")return exact_actions::learning::reger_residual_core_forward(exact_actions::learning::reger_bitpack_forward(exact_actions::learning::reger_for_forward(exact_actions::learning::reger_zigzag_forward(exact_actions::learning::reger_delta_forward(exact_actions::learning::reger_reorder_forward(v))))));
  if(n=="SVB")return exact_actions::svb::compose(words<std::int64_t>(v));
  if(n=="FastPFor")return exact_actions::fastpfor::adapter_forward_int(words<std::int64_t>(v));
  if(n=="Simple8b")return exact_actions::simple8b::bitpack_forward_int64(words<std::int64_t>(v));
  if(n=="DBP")return exact_actions::dbp::compose_int64(words<std::int64_t>(v));
  if(n=="RLBE")return exact_actions::p04::rlbe_compose(words<std::int64_t>(v));
  if(n=="Sprintz")return b.dtype=="int64"?exact_actions::p04::sprintz_compose(words<std::int64_t>(v)):exact_actions::p04::sprintz_compose_double_bits(words<std::int64_t>(v));
  if(n=="Camel")return exact_actions::camel::compose(words<std::int64_t>(v));
  if(n=="ALP")return exact_actions::learning::alp_residual_core_forward(exact_actions::learning::alp_bitpack_forward(exact_actions::learning::alp_for_forward(exact_actions::learning::alp_scaling_forward(v))));
  if(n=="LECO")return exact_actions::learning::leco_bitpack_forward(exact_actions::learning::leco_zigzag_forward(exact_actions::learning::leco_regression_forward(v)));
  if(n=="Elf"||n=="Elf+")return elf_double::encodeDoubleColumnExact(v,n=="Elf+");
  if(n=="FPC")return exact_actions::fpc::compose(v,16);
  if(n=="PFPC")return exact_actions::pfpc::compose(v,16);
  if(n=="SPDP")return exact_actions::spdp::compose(v,5);
  if(n=="FPZIP")return exact_actions::fpzip::compose(v);
  if(n=="Libdeflate"||n=="FastLZ"||n=="LZF"||n=="LZFSE"||
     n=="LZVN"||n=="BriefLZ"||n=="LZAV"||n=="LZG"||
     n=="LZJB"||n=="LZMAT"||n=="LZRW"||n=="WFLZ"||
     n=="Yappy"||n=="YALZ77"||n=="QuickLZ"||n=="SLZ"||
     n=="TAMP"||n=="Bzip3"||n=="FSE"||n=="Huff0")
    return vendored_lightweight::encode(n,v);
#ifdef BASELINE86_BLOSC1
  if(n=="Blosc")return wc_blosc1_bench::encodeBufferExact(v,width(b.dtype));
#endif
#ifdef BASELINE86_BLOSC2
  if(n=="Blosc2")return wc_blosc2_bench::encodeBufferExact(v,width(b.dtype));
#endif
  if(n=="Zlib"||n=="gzip")return zlib_encode(v,n=="gzip");
#ifdef BASELINE86_ZSTD
  if(n=="Zstd"){Bytes out(ZSTD_compressBound(v.size()));auto size=ZSTD_compress(out.data(),out.size(),v.data(),v.size(),3);require(!ZSTD_isError(size),"Zstd encode");out.resize(size);return out;}
#endif
#ifdef BASELINE86_BROTLI
  if(n=="Brotli"){Bytes out(std::max<std::size_t>(1,BrotliEncoderMaxCompressedSize(v.size())));std::size_t size=out.size();require(BrotliEncoderCompress(5,22,BROTLI_MODE_GENERIC,v.size(),v.data(),&size,out.data()),"Brotli encode");out.resize(size);return out;}
#endif
#ifdef BASELINE86_LZ4
  if(n=="LZ4"){LZ4F_preferences_t prefs{};prefs.compressionLevel=0;Bytes out(LZ4F_compressFrameBound(v.size(),&prefs));auto size=LZ4F_compressFrame(out.data(),out.size(),v.data(),v.size(),&prefs);require(!LZ4F_isError(size),"LZ4 frame encode");out.resize(size);return out;}
  if(n=="LZ4 Fast"||n=="LZ4HC"){require(v.size()<=std::size_t(INT_MAX),"LZ4 input exceeds int range");Bytes out(LZ4_compressBound(static_cast<int>(v.size())));const int size=n=="LZ4 Fast"?LZ4_compress_fast(reinterpret_cast<const char*>(v.data()),reinterpret_cast<char*>(out.data()),v.size(),out.size(),1):LZ4_compress_HC(reinterpret_cast<const char*>(v.data()),reinterpret_cast<char*>(out.data()),v.size(),out.size(),9);require(size>0,"LZ4 raw encode");out.resize(size);return out;}
#endif
#ifdef BASELINE86_SNAPPY
  if(n=="Snappy"){std::string out;snappy::Compress(reinterpret_cast<const char*>(v.data()),v.size(),&out);return Bytes(out.begin(),out.end());}
#endif
#ifdef BASELINE86_LZMA
  if(n=="XZ"){Bytes out(lzma_stream_buffer_bound(v.size()));std::size_t size=0;require(lzma_easy_buffer_encode(6,LZMA_CHECK_CRC64,nullptr,v.data(),v.size(),out.data(),&size,out.size())==LZMA_OK,"XZ encode");out.resize(size);return out;}
#endif
#ifdef BASELINE86_BZIP2
  if(n=="Bzip2"){Bytes out(v.size()+v.size()/100+601);unsigned size=out.size();require(BZ2_bzBuffToBuffCompress(reinterpret_cast<char*>(out.data()),&size,const_cast<char*>(reinterpret_cast<const char*>(v.empty()?reinterpret_cast<const std::uint8_t*>(""):v.data())),v.size(),9,0,30)==BZ_OK,"Bzip2 encode");out.resize(size);return out;}
#endif
  throw Unsupported(n+": no encode adapter");
}
Bytes decompress(const Record&r,const Bytes&v,const std::string&dtype,std::size_t expected){
  const auto&n=r.label;
  if(expected==0){require(v.empty(),"non-empty method payload for an all-missing block");return {};}
  for (const auto& d : upstream_general_a::inventory())
    if (n==d.label) return upstream_general_a::decode(n,v,expected);
  if(upstream_general_b::supports(n))return upstream_general_b::decode(n,v,expected);
  if(upstream_legacy::supports(n))return upstream_legacy::decode(n,v,expected);
  for (const auto& d : upstream_specialists::profiles())
    if (n==d.label)return upstream_specialists::decode(n,v,expected);
  if(n=="Chimp128")return raw(dtype=="int64"?exact_actions::chimp128::decompose_int64(v):exact_actions::chimp128::decompose_double_bits(v));
  if(n=="Gorilla")return raw(dtype=="int64"?exact_actions::gorilla::decompose_int64(v):exact_actions::gorilla::decompose_double_bits(v));
  if(n=="Pongo")return exact_actions::pongo::decompose(v);
  if(n=="2DIFF")return raw(exact_actions::ts2diff::decompose_int64(v));
  if(n=="BOS")return raw(exact_actions::bos::adapter_decompose(v));
  if(n=="REGER")return exact_actions::learning::reger_reorder_inverse(exact_actions::learning::reger_delta_inverse(exact_actions::learning::reger_zigzag_inverse(exact_actions::learning::reger_for_inverse(exact_actions::learning::reger_bitpack_inverse(exact_actions::learning::reger_residual_core_inverse(v))))));
  if(n=="SVB")return raw(exact_actions::svb::decompose(v));
  if(n=="FastPFor")return raw(exact_actions::fastpfor::adapter_inverse_int(v,expected/8));
  if(n=="Simple8b")return raw(exact_actions::simple8b::bitpack_inverse_int64(v));
  if(n=="DBP")return raw(exact_actions::dbp::decompose_int64(v));
  if(n=="RLBE")return raw(exact_actions::p04::rlbe_decompose(v));
  if(n=="Sprintz")return raw(dtype=="int64"?exact_actions::p04::sprintz_decompose(v):exact_actions::p04::sprintz_decompose_double_bits(v));
  if(n=="Camel")return raw(exact_actions::camel::decompose(v));
  if(n=="ALP")return exact_actions::learning::alp_scaling_inverse(exact_actions::learning::alp_for_inverse(exact_actions::learning::alp_bitpack_inverse(exact_actions::learning::alp_residual_core_inverse(v))));
  if(n=="LECO")return exact_actions::learning::leco_regression_inverse(exact_actions::learning::leco_zigzag_inverse(exact_actions::learning::leco_bitpack_inverse(v)));
  if(n=="Elf"||n=="Elf+")return elf_double::decodeDoubleColumnExact(v,n=="Elf+");
  if(n=="FPC")return exact_actions::fpc::decompose(v);
  if(n=="PFPC")return exact_actions::pfpc::decompose(v);
  if(n=="SPDP")return exact_actions::spdp::decompose(v,expected,5);
  if(n=="FPZIP")return exact_actions::fpzip::decompose(v);
  if(n=="Libdeflate"||n=="FastLZ"||n=="LZF"||n=="LZFSE"||
     n=="LZVN"||n=="BriefLZ"||n=="LZAV"||n=="LZG"||
     n=="LZJB"||n=="LZMAT"||n=="LZRW"||n=="WFLZ"||
     n=="Yappy"||n=="YALZ77"||n=="QuickLZ"||n=="SLZ"||
     n=="TAMP"||n=="Bzip3"||n=="FSE"||n=="Huff0")
    return vendored_lightweight::decode(n,v,expected);
#ifdef BASELINE86_BLOSC1
  if(n=="Blosc")return wc_blosc1_bench::decodeBufferExact(v);
#endif
#ifdef BASELINE86_BLOSC2
  if(n=="Blosc2")return wc_blosc2_bench::decodeBufferExact(v,width(dtype));
#endif
  if(n=="Zlib"||n=="gzip")return zlib_decode(v,expected,n=="gzip");
#ifdef BASELINE86_ZSTD
  if(n=="Zstd"){auto size=ZSTD_findFrameCompressedSize(v.data(),v.size());require(!ZSTD_isError(size)&&size==v.size(),"Zstd trailing or truncated frame");Bytes out(expected+1);size=ZSTD_decompress(out.data(),out.size(),v.data(),v.size());require(!ZSTD_isError(size)&&size==expected,"Zstd decode size");out.resize(size);return out;}
#endif
#ifdef BASELINE86_BROTLI
  if(n=="Brotli"){auto*state=BrotliDecoderCreateInstance(nullptr,nullptr,nullptr);require(state,"Brotli decoder allocation");Bytes out(expected+1);std::size_t available_in=v.size(),available_out=out.size(),total=0;const std::uint8_t*next_in=v.data();std::uint8_t*next_out=out.data();const auto status=BrotliDecoderDecompressStream(state,&available_in,&next_in,&available_out,&next_out,&total);BrotliDecoderDestroyInstance(state);require(status==BROTLI_DECODER_RESULT_SUCCESS&&available_in==0&&total==expected,"invalid Brotli stream");out.resize(expected);return out;}
#endif
#ifdef BASELINE86_LZ4
  if(n=="LZ4"){LZ4F_dctx*state=nullptr;require(!LZ4F_isError(LZ4F_createDecompressionContext(&state,LZ4F_VERSION)),"LZ4 decoder init");Bytes out(expected+1);std::size_t ip=0,op=0,code=1;while(code&&ip<v.size()){auto src=v.size()-ip,dst=out.size()-op;code=LZ4F_decompress(state,out.data()+op,&dst,v.data()+ip,&src,nullptr);if(LZ4F_isError(code)||(!src&&!dst)){LZ4F_freeDecompressionContext(state);throw std::invalid_argument("invalid LZ4 frame");}ip+=src;op+=dst;}LZ4F_freeDecompressionContext(state);require(code==0&&ip==v.size()&&op==expected,"LZ4 frame size or trailing data");out.resize(expected);return out;}
  if(n=="LZ4 Fast"||n=="LZ4HC"){require(v.size()<=std::size_t(INT_MAX)&&expected<=std::size_t(INT_MAX),"LZ4 raw size exceeds int range");Bytes out(expected+1);const int size=LZ4_decompress_safe(reinterpret_cast<const char*>(v.data()),reinterpret_cast<char*>(out.data()),v.size(),expected);require(size>=0&&std::size_t(size)==expected,"LZ4 raw decode size");out.resize(expected);return out;}
#endif
#ifdef BASELINE86_SNAPPY
  if(n=="Snappy"){std::size_t size=0;require(snappy::GetUncompressedLength(reinterpret_cast<const char*>(v.data()),v.size(),&size)&&size==expected,"Snappy length");Bytes out(expected+1);require(snappy::RawUncompress(reinterpret_cast<const char*>(v.data()),v.size(),reinterpret_cast<char*>(out.data())),"Snappy decode");out.resize(expected);return out;}
#endif
#ifdef BASELINE86_LZMA
  if(n=="XZ"){Bytes out(expected+1);std::uint64_t memory=256ULL<<20;std::size_t ip=0,op=0;require(lzma_stream_buffer_decode(&memory,0,nullptr,v.data(),&ip,v.size(),out.data(),&op,out.size())==LZMA_OK&&ip==v.size()&&op==expected,"XZ decode size or trailing frame");out.resize(expected);return out;}
#endif
#ifdef BASELINE86_BZIP2
  if(n=="Bzip2"){bz_stream stream{};require(BZ2_bzDecompressInit(&stream,0,0)==BZ_OK,"Bzip2 decoder init");Bytes out(expected+1);stream.next_in=const_cast<char*>(reinterpret_cast<const char*>(v.data()));stream.avail_in=v.size();stream.next_out=reinterpret_cast<char*>(out.data());stream.avail_out=out.size();const auto status=BZ2_bzDecompress(&stream);const auto remain=stream.avail_in;const auto decoded=out.size()-stream.avail_out;BZ2_bzDecompressEnd(&stream);require(status==BZ_STREAM_END&&remain==0&&decoded==expected,"Bzip2 decode size or trailing frame");out.resize(expected);return out;}
#endif
  throw Unsupported(n+": no decode adapter");
}
std::uint64_t valid_values(std::uint64_t rows,const Bytes&bitmap){
  require(rows<=maximum*8 && bitmap.size()==(rows+7)/8,"bitmap row count/size mismatch");std::uint64_t missing=0;for(auto b:bitmap)missing+=std::popcount(b);if(rows%8&&!bitmap.empty())require((bitmap.back()&(~((1U<<(rows%8))-1)&255))==0,"nonzero bitmap tail padding");require(missing<=rows,"invalid missing count");return rows-missing;
}
class Dispatcher final:public StageDispatcher {
 public:
  explicit Dispatcher(ScalingStats*stats=nullptr):stats_(stats){}
  std::string identity()const override{return "native-baseline86-full-frame-dispatch-v1";}
  static bool scaled(const StageRequest&r){return r.action.starts_with("baseline86/scaled-int64-v1/");}
  const Record& request(const StageRequest&r)const{
    if(scaled(r)){
      const auto&record=method(r.action.substr(std::string("baseline86/scaled-int64-v1/").size()));
      require(supports_decimal_scaling(record.label)&&record.linked,"method has no decimal scaling adapter");
      require(r.profile=="float32"||r.profile=="float64","decimal scaling requires original float dtype");
      require(r.parameters.size()==4&&r.parameters.contains("configuration")&&r.parameters.at("configuration")==record.configuration&&r.parameters.contains("scaling_configuration")&&r.parameters.at("scaling_configuration")==scaling_configuration&&r.parameters.contains("row_count")&&r.parameters.contains("decimal_places"),"decimal scaling fixed configuration mismatch");
      require(natural(r.parameters.at("decimal_places"))<=decimal_scaling::maximum_decimal_places,"decimal precision exceeds 324");
      return record;
    }
    const std::string prefix="baseline86/";require(r.action.starts_with(prefix),"unexpected baseline action");auto&record=method(r.action.substr(prefix.size()));eligible(record,r.profile);require(r.parameters.size()==2&&r.parameters.contains("configuration")&&r.parameters.at("configuration")==record.configuration&&r.parameters.contains("row_count"),"baseline fixed configuration mismatch");return record;
  }
  StageResult forward(const StageRequest&r,const std::vector<Stream>&in)const override{
    const auto&record=request(r);require(in.size()==2&&in[0].kind==Kind::Numeric&&in[0].width==width(r.profile)&&in[1].kind==Kind::Serial&&in[1].width==1,"baseline root shape");Block b{r.profile,natural(r.parameters.at("row_count")),in[0].data,in[1].data};validate(b);
    if(scaled(r)){
      auto converted=decimal_scaling::forward(b.values,b.dtype,static_cast<std::uint16_t>(natural(r.parameters.at("decimal_places"))));
      if(!b.values.empty()&&converted.values.empty())throw Unsupported("no value fits the frozen decimal int64 scale; raw-only result is not an integer-codec measurement");
      Block integers{"int64",converted.values.size(),raw(converted.values),Bytes((converted.values.size()+7)/8,0)};
      Bytes result;
      try{result=compress(record,integers);}catch(const std::overflow_error&error){
        if(record.label=="2DIFF"&&std::string_view(error.what())=="TS_2DIFF delta exceeds int64")
          throw Unsupported("scaled int64 adjacent delta exceeds the 2DIFF support domain");
        throw;
      }
      require(result.size()<=maximum,"scaled compressed payload exceeds bound");
      if(stats_)*stats_={converted.values.size(),converted.exception_count,
                         converted.metadata.size(),converted.exception_count*width(b.dtype)};
      return {{{Kind::Serial,1,std::move(result)},in[1]},std::move(converted.metadata),{"bytes","bytes"}};
    }
    auto result=compress(record,b);require(result.size()<=maximum,"compressed payload exceeds bound");
    return {{{Kind::Serial,1,std::move(result)},in[1]}, {}, {"bytes","bytes"}};
  }
  std::vector<Stream> inverse(const StageRequest&r,const std::vector<Stream>&out,std::span<const std::uint8_t>metadata)const override{
    if(scaled(r)){
      const auto&record=request(r);
      require(out.size()==2&&out[0].kind==Kind::Serial&&out[1].kind==Kind::Serial&&out[0].width==1&&out[1].width==1,"scaled inverse shape");
      const auto count=valid_values(natural(r.parameters.at("row_count")),out[1].data);
      const auto info=decimal_scaling::inspect(metadata,r.profile);
      require(info.original_count==count&&info.decimal_places==natural(r.parameters.at("decimal_places")),"scaled metadata differs from original rows or frozen precision");
      require(count==0||info.scaled_count>0,"raw-only integer codec frame");
      auto integers=decompress(record,out[0].data,"int64",info.scaled_count*8);
      require(integers.size()==info.scaled_count*8,"scaled inverse integer length mismatch");
      auto restored=decimal_scaling::inverse(words<std::int64_t>(integers),metadata,r.profile);
      require(restored.size()==count*width(r.profile),"scaled inverse original length mismatch");
      return {{Kind::Numeric,width(r.profile),std::move(restored)},out[1]};
    }
    const auto&record=request(r);require(metadata.empty()&&out.size()==2&&out[0].kind==Kind::Serial&&out[1].kind==Kind::Serial&&out[0].width==1&&out[1].width==1,"baseline inverse shape/metadata");const auto count=valid_values(natural(r.parameters.at("row_count")),out[1].data);const auto w=width(r.profile);require(count<=maximum/w,"decoded allocation bound");const auto size=count*w;auto restored=decompress(record,out[0].data,r.profile,size);require(restored.size()==size,"inverse original length mismatch");return {{Kind::Numeric,w,std::move(restored)},out[1]};
  }
 private:
  ScalingStats*stats_;
};
} // namespace
const std::vector<Record>&inventory(){static const auto result=make_inventory();return result;}
const Record&method(const std::string&name){for(const auto&r:inventory())if(r.label==name)return r;throw Unsupported("unknown method label: "+name);}
bool supports_decimal_scaling(const std::string&label){
  static const std::set<std::string> labels={"2DIFF","BOS","REGER","SVB","FastPFor","Simple8b","DBP","RLBE","LECO"};
  return labels.contains(label);
}
std::vector<Record> scaling_inventory(){
  std::vector<Record> result;
  for(const auto&original:inventory())if(supports_decimal_scaling(original.label)){
    auto record=original;record.dtypes={"float32","float64"};
    record.implementation="exact_decimal_scaling("+original.implementation+")";
    record.source=original.source+"; cpp/native_baseline86_v1/decimal_scaling.cpp";
    record.configuration=std::string(scaling_configuration)+";base="+original.configuration;
    record.limitations="Lossless original-dtype decimal scaling with frozen whole-column precision; raw-bit exceptions and complete metadata are counted. Integer kernel domain limits still apply.";
    result.push_back(std::move(record));
  }
  return result;
}
std::uint32_t width(const std::string&dtype){if(dtype=="int8"||dtype=="uint8")return 1;if(dtype=="int16"||dtype=="uint16")return 2;if(dtype=="int32"||dtype=="uint32"||dtype=="float32")return 4;if(dtype=="int64"||dtype=="uint64"||dtype=="float64")return 8;throw Unsupported("unknown original numeric dtype: "+dtype);}
void validate(const Block&b){const auto w=width(b.dtype);require(b.values.size()<=maximum&&b.missing.size()<=maximum,"prepared block byte cap exceeded");auto count=valid_values(b.row_count,b.missing);require(count<=maximum/w&&b.values.size()==count*w,"valid numeric bytes differ from bitmap/native width");}
Bytes encode(const std::string&label,const Block&b){
  validate(b);const auto&r=method(label);eligible(r,b.dtype);Dispatcher dispatcher;
  Recipe recipe;recipe.policy_registry=profile_version;
  Step s;s.kind=StepKind::Stage;s.inputs={0,1};s.stage={"baseline86/"+label,b.dtype,{{"configuration",r.configuration},{"row_count",std::to_string(b.row_count)}}};recipe.steps.push_back(std::move(s));Step stop;stop.kind=StepKind::Stop;recipe.steps.push_back(stop);
  return native_graph_v1::encode({{{Kind::Numeric,width(b.dtype),b.values},true,b.dtype},{{Kind::Serial,1,b.missing},false,"missing_bitmap_lsb_first:rows="+std::to_string(b.row_count)}},recipe,dispatcher);
}
Bytes encode_scaled(const std::string&label,const Block&b,std::uint16_t p,ScalingStats*stats){
  validate(b);const auto&r=method(label);
  if(!supports_decimal_scaling(label)||!r.linked)throw Unsupported("method has no decimal scaling adapter: "+label);
  if(b.dtype!="float32"&&b.dtype!="float64")throw Unsupported("decimal scaling requires float32 or float64");
  require(p<=decimal_scaling::maximum_decimal_places,"decimal precision exceeds 324");
  Dispatcher dispatcher(stats);Recipe recipe;recipe.policy_registry=scaled_profile_version;
  Step s;s.kind=StepKind::Stage;s.inputs={0,1};
  s.stage={"baseline86/scaled-int64-v1/"+label,b.dtype,{{"configuration",r.configuration},{"scaling_configuration",scaling_configuration},{"row_count",std::to_string(b.row_count)},{"decimal_places",std::to_string(p)}}};
  recipe.steps.push_back(std::move(s));Step stop;stop.kind=StepKind::Stop;recipe.steps.push_back(stop);
  return native_graph_v1::encode({{{Kind::Numeric,width(b.dtype),b.values},true,b.dtype},{{Kind::Serial,1,b.missing},false,"missing_bitmap_lsb_first:rows="+std::to_string(b.row_count)}},recipe,dispatcher);
}
ScalingFrameInfo inspect_scaled(std::span<const std::uint8_t>bytes){
  const auto frame=native_graph_v1::detail::parse_frame(bytes,{});
  require(frame.policy_registry==scaled_profile_version&&frame.nodes.size()==1,"not a single-stage decimal-scaled frame");
  const auto&node=frame.nodes.front();Dispatcher dispatcher;
  require(frame.dispatcher==dispatcher.identity()&&node.step.kind==StepKind::Stage&&Dispatcher::scaled(node.step.stage),"scaled frame dispatcher or stage mismatch");
  const auto&record=dispatcher.request(node.step.stage);
  const auto info=decimal_scaling::inspect(node.metadata,node.step.stage.profile);
  require(info.decimal_places==natural(node.step.stage.parameters.at("decimal_places")),"scaled frame precision mismatch");
  return {record.label,node.step.stage.profile,natural(node.step.stage.parameters.at("row_count")),info.decimal_places,
          {info.scaled_count,info.exception_count,node.metadata.size(),info.exception_count*width(node.step.stage.profile)}};
}
Block decode(std::span<const std::uint8_t>frame){Dispatcher dispatcher;auto decoded=native_graph_v1::decode(frame,dispatcher);require(decoded.roots.size()==2&&decoded.roots[0].original_numeric_payload&&!decoded.roots[1].original_numeric_payload,"baseline decoded root flags");auto&numeric=decoded.roots[0];auto&bitmap=decoded.roots[1];const std::string prefix="missing_bitmap_lsb_first:rows=";require(bitmap.logical_view.starts_with(prefix),"missing bitmap row metadata");Block b{numeric.logical_view,natural(bitmap.logical_view.substr(prefix.size())),std::move(numeric.stream.data),std::move(bitmap.stream.data)};require(numeric.stream.kind==Kind::Numeric&&numeric.stream.width==width(b.dtype)&&bitmap.stream.kind==Kind::Serial&&bitmap.stream.width==1,"baseline decoded root type");validate(b);require(decoded.original_numeric_bytes==b.values.size(),"numeric denominator mismatch");return b;}
Measurement measure(const std::string&label,const Block&b,unsigned warmups,unsigned repeats){
  require(warmups<=100&&repeats>0&&repeats<=1000,"invalid measurement repeat count");Measurement result;result.original_numeric_bytes=b.values.size();
  for(unsigned i=0;i<warmups;++i){auto f=encode(label,b);require(decode(f)==b,"warmup bit-exact roundtrip failed");}
  for(unsigned i=0;i<repeats;++i){auto start=std::chrono::steady_clock::now();auto frame=encode(label,b);auto end=std::chrono::steady_clock::now();result.compression_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());if(i)require(frame==result.frame,"nondeterministic fixed encoding");else result.frame=std::move(frame);}
  for(unsigned i=0;i<repeats;++i){auto start=std::chrono::steady_clock::now();auto restored=decode(result.frame);auto end=std::chrono::steady_clock::now();result.decompression_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());require(restored==b,"measured bit-exact roundtrip failed");}
  return result;
}
Measurement measure_scaled(const std::string&label,const Block&b,std::uint16_t p,unsigned warmups,unsigned repeats){
  require(warmups<=100&&repeats>0&&repeats<=1000,"invalid measurement repeat count");Measurement result;result.original_numeric_bytes=b.values.size();
  for(unsigned i=0;i<warmups;++i){auto f=encode_scaled(label,b,p);require(decode(f)==b,"scaled warmup bit-exact roundtrip failed");}
  for(unsigned i=0;i<repeats;++i){ScalingStats stats;auto start=std::chrono::steady_clock::now();auto frame=encode_scaled(label,b,p,&stats);auto end=std::chrono::steady_clock::now();result.compression_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());if(i)require(frame==result.frame,"nondeterministic scaled encoding");else{result.frame=std::move(frame);result.scaling_stats=stats;}}
  for(unsigned i=0;i<repeats;++i){auto start=std::chrono::steady_clock::now();auto restored=decode(result.frame);auto end=std::chrono::steady_clock::now();result.decompression_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());require(restored==b,"scaled measured bit-exact roundtrip failed");}
  return result;
}
} // namespace native_baseline86_v1
