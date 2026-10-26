#include "unified_rle_numeric.hpp"
#include "tsfile_encoding/tsfile_encode_p02.hpp"
#include "tsfile_encoding/tsfile_decode_p02.hpp"
#include "tsfile_encoding/tsfile_encode_p04.hpp"
#include "tsfile_encoding/tsfile_decode_p04.hpp"

#include <iostream>
#include <iterator>
#include <string>

namespace {
using unified_rle_numeric::Bytes;
void write(const Bytes& raw) {
  std::cout.write(reinterpret_cast<const char*>(raw.data()),raw.size());
  if(!std::cout)throw std::runtime_error("failed to write numeric codec output");
}
unsigned width_of(const std::string& kind) {
  if(kind=="i32"||kind=="f32")return 4;
  if(kind=="i64"||kind=="f64")return 8;
  if(kind.size()==2&&kind[0]=='w'&&kind[1]>='1'&&kind[1]<='8')return kind[1]-'0';
  throw std::invalid_argument("unsupported numeric codec type");
}
}

int main(int argc,char** argv) try {
  if(argc<4||argc>5)throw std::invalid_argument("usage: unified_rle_numeric_cli encode|decode rle|rle_scaled|rlbe|rlbe_deltas|sprintz|sprintz_reference i32|i64|f32|f64|w1..w8 [decode_max_bytes or encode_rle_scaled_max_point]");
  const std::string op=argv[1],codec=argv[2],kind=argv[3];
  const bool encode=op=="encode";
  if(!encode&&op!="decode")throw std::invalid_argument("unknown codec operation");
  const unsigned width=width_of(kind);
  std::size_t limit=256u*1024u*1024u;
  if(argc==5) {
    std::size_t used=0;const std::string value=argv[4];
    if(value.empty()||value.front()=='-')throw std::invalid_argument("invalid output limit");
    limit=std::stoull(value,&used);
    if(used!=value.size())throw std::invalid_argument("invalid output limit");
  }
  Bytes raw{std::istreambuf_iterator<char>(std::cin),{}};
  using namespace unified_rle_numeric;
  if(encode&&raw.size()%width)throw std::invalid_argument("unaligned numeric codec input");
  if(codec=="rle_scaled") {
    if(kind!="f32"&&kind!="f64")throw std::invalid_argument("scaled RLE needs floating input");
    if(encode) {
      const auto scale=argc==5?limit:2;
      if(scale>18)throw std::invalid_argument("RLE max_point must be in [0,18]");
      write(encode_rle_scaled(raw,width,static_cast<int>(scale)));return 0;
    }
  }
  if((codec=="rle"||codec=="rle_scaled") && !encode && is_scaled_rle_frame(raw)) {
    if((kind!="f32"&&kind!="f64")||scaled_rle_symbol_width(raw)!=width)throw std::runtime_error("scaled RLE type mismatch");
    write(decode_rle_scaled(raw,limit));return 0;
  }
  if(codec=="rle") {
    if(encode){write(encode_rle(raw,width));return 0;}
    if(is_rle_frame(raw)) {
      if(rle_symbol_width(raw)!=width)throw std::runtime_error("RLE wire type mismatch");
      write(decode_rle(raw,limit));return 0;
    }
  }
  if(codec=="rlbe"||codec=="rlbe_deltas") {
    const bool delta=codec=="rlbe";
    if(encode) {
      write(kind=="f32"&&delta?encode_rlbe_float32_arithmetic(raw):encode_rlbe(raw,width,delta));return 0;
    }
    if(is_rlbe_frame(raw)) {
      const auto mode=rlbe_delta_mode(raw);
      if(rlbe_symbol_width(raw)!=width || (!delta&&mode!=0) ||
          (delta&&kind=="f32"&&mode!=0&&mode!=2) ||
          (delta&&kind!="f32"&&mode!=1))throw std::runtime_error("RLBE wire type or delta mode mismatch");
      write(decode_rlbe(raw,limit));return 0;
    }
    if(!delta) {
      if(width!=8)throw std::invalid_argument("old RLBE delta wire requires int64");
      std::vector<std::int64_t> out;
      if(!tsfile::p04::decode_rlbe_int64_ablation(raw,"delta",out))throw std::runtime_error("invalid legacy RLBE delta wire");
      if(out.size()>limit/8)throw std::runtime_error("legacy decode exceeds output limit");
      write(raw_bytes<std::int64_t>(out));return 0;
    }
  }
#define ROUTE(C,K,T,NS,ENC,DEC) \
  if(codec==C&&kind==K) { \
    if(encode)write(NS::ENC(typed_values<T>(raw))); \
    else {std::vector<T> out;if(!NS::DEC(raw,out))throw std::runtime_error("invalid legacy numeric wire"); \
      if(out.size()>limit/sizeof(T))throw std::runtime_error("decoded output exceeds explicit limit");write(raw_bytes<T>(out));} \
    return 0;}
  ROUTE("rle","i32",std::int32_t,tsfile::p02,encode_rle_int32,decode_rle_int32)
  ROUTE("rle","i64",std::int64_t,tsfile::p02,encode_rle_int64,decode_rle_int64)
  ROUTE("rle","f32",float,tsfile::p02,encode_rle_float,decode_rle_float)
  ROUTE("rle","f64",double,tsfile::p02,encode_rle_double,decode_rle_double)
  ROUTE("rle_scaled","f32",float,tsfile::p02,encode_rle_float,decode_rle_float)
  ROUTE("rle_scaled","f64",double,tsfile::p02,encode_rle_double,decode_rle_double)
  ROUTE("rlbe","i32",std::int32_t,tsfile::p04,encode_rlbe_int32,decode_rlbe_int32)
  ROUTE("rlbe","i64",std::int64_t,tsfile::p04,encode_rlbe_int64,decode_rlbe_int64)
  ROUTE("rlbe","f32",float,tsfile::p04,encode_rlbe_float,decode_rlbe_float)
  ROUTE("rlbe","f64",double,tsfile::p04,encode_rlbe_double,decode_rlbe_double)
  ROUTE("sprintz","i32",std::int32_t,tsfile::p04,encode_sprintz_int32,decode_sprintz_int32)
  ROUTE("sprintz","f32",float,tsfile::p04,encode_sprintz_float,decode_sprintz_float)
  ROUTE("sprintz","i64",std::int64_t,tsfile::p04,encode_sprintz_int64,decode_sprintz_int64)
  ROUTE("sprintz","f64",double,tsfile::p04,encode_sprintz_double,decode_sprintz_double)
  ROUTE("sprintz_reference","i64",std::int64_t,tsfile::p04,encode_sprintz_int64_reference,decode_sprintz_int64_reference)
  ROUTE("sprintz_reference","f64",double,tsfile::p04,encode_sprintz_double_reference,decode_sprintz_double_reference)
#undef ROUTE
  throw std::invalid_argument("unsupported codec/type combination");
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
