#include "upstream_legacy.hpp"
extern "C" {
#include "lz/pithy/pithy.h"
#include "lz/shrinker/shrinker.h"
#include "lz/xpack/lib/libxpack.h"
}
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
namespace native_baseline86_v1::upstream_legacy {
namespace {
constexpr std::size_t limit=256ULL<<20;
std::mutex global_modes;
void need(bool good,const char* message){if(!good)throw std::runtime_error(message);}
bool ablation_magic(std::span<const std::uint8_t> b){return b.size()>=4&&!std::memcmp(b.data(),"SHT1",4);}
// The upstream Shrinker decoder has no input-length argument. Check every
// token before calling it, and give its eight-byte copies private padding.
void check_shrinker(std::span<const std::uint8_t> b,std::size_t expected){
  need(!ablation_magic(b),"Shrinker ablation wire is outside the fixed profile");
  std::size_t at=0,out=0;
  auto byte=[&](){need(at<b.size(),"truncated Shrinker token");return b[at++];};
  auto extension=[&](std::size_t n){unsigned x;do{x=byte();need(n<=expected&&x<=expected-n,"Shrinker length overflow");n+=x;}while(x==255);return n;};
  for(;;){
    auto flag=byte();std::size_t literal=flag>>5,match=flag&15;
    if(literal==7)literal=extension(literal);
    if(match==15)match=extension(match);
    unsigned distance=byte();if(flag&16)distance|=unsigned(byte())<<8;
    need(literal<=b.size()-at&&literal<=expected-out,"Shrinker literal overrun");
    at+=literal;out+=literal;
    if((flag&16)&&distance==65535){need(at==b.size()&&out==expected,"Shrinker terminal size/trailing input");return;}
    need(distance+1<=out&&match<=expected-out&&4<=expected-out-match,"Shrinker invalid match");
    out+=match+4;
  }
}
Bytes verbatim(std::span<const std::uint8_t> b){Bytes out(b.size()+1,0);if(!b.empty())std::memcpy(out.data()+1,b.data(),b.size());return out;}
}
bool supports(const std::string& n){return n=="Pithy"||n=="Shrinker"||n=="XPACK";}
Bytes encode(const std::string& n,std::span<const std::uint8_t> b){
  need(supports(n)&&b.size()<=limit,"unknown legacy codec or oversized input");
  const std::lock_guard<std::mutex> lock(global_modes);
  if(b.empty()||(n=="Shrinker"&&(b.size()<32||b.size()>=(1ULL<<27))))return verbatim(b);
  // Original implementations use word-wide lookahead. Padding is transient,
  // included in native encode time, and never part of the input denominator.
  Bytes padded(b.begin(),b.end());padded.resize(b.size()+64,0);
  Bytes out(b.size()*2+65536+64);std::size_t used=0;
  if(n=="Pithy"){
    pithy_SetAblationMode(PITHY_ABLATION_NONE);
    used=pithy_Compress(reinterpret_cast<const char*>(padded.data()),b.size(),reinterpret_cast<char*>(out.data()),out.size()-64,6);
  }else if(n=="Shrinker"){
    shrinker_set_ablation_mode(SHRINKER_ABLATION_NONE);
    const int result=shrinker_compress_with_capacity(padded.data(),out.data(),out.size()-64,static_cast<int>(b.size()));
    if(result>0)used=static_cast<std::size_t>(result);
  }else{
    std::unique_ptr<xpack_compressor,decltype(&xpack_free_compressor)> context(xpack_alloc_compressor(b.size(),6),xpack_free_compressor);
    need(bool(context),"XPACK compressor allocation failed");
    xpack_set_ablation_mode(context.get(),XPACK_ABLATION_NONE);
    used=xpack_compress(context.get(),padded.data(),b.size(),out.data(),out.size()-64);
  }
  need(used<=out.size()-64,"legacy encoder output overflow");
  if(!used||used>=b.size()||(n=="Shrinker"&&ablation_magic({out.data(),used})))return verbatim(b);
  if(n=="Shrinker"){
    // The legacy compressor may emit a syntactically valid stream whose
    // restored bytes differ.  Qualify the exact candidate while the global
    // mode lock is still held; a rejected candidate is represented by the
    // existing tagged verbatim branch and therefore remains lossless.
    try{
      const std::span<const std::uint8_t> body(out.data(),used);
      check_shrinker(body,b.size());
      Bytes encoded_input(body.begin(),body.end());encoded_input.resize(used+64,0);
      Bytes restored(b.size()+64,0);
      const int restored_size=shrinker_decompress(encoded_input.data(),restored.data(),static_cast<int>(b.size()));
      if(restored_size!=static_cast<int>(b.size())||
          !std::equal(b.begin(),b.end(),restored.begin()))return verbatim(b);
    }catch(const std::exception&){return verbatim(b);}
  }
  out.resize(used);out.insert(out.begin(),1);return out;
}
Bytes decode(const std::string& n,std::span<const std::uint8_t> packet,std::size_t expected){
  need(supports(n)&&expected<=limit&&!packet.empty(),"invalid legacy decode request");
  if(packet[0]==0){need(packet.size()-1==expected,"legacy verbatim length mismatch");return {packet.begin()+1,packet.end()};}
  need(packet[0]==1&&expected>0,"invalid legacy adapter marker");
  auto body=packet.subspan(1);need(!body.empty()&&body.size()<=limit,"invalid legacy compressed length");
  const std::lock_guard<std::mutex> lock(global_modes);
  if(n=="Shrinker")check_shrinker(body,expected);
  Bytes input(body.begin(),body.end());input.resize(body.size()+64,0);
  Bytes out(expected+64,0);
  if(n=="Pithy"){
    std::size_t declared=0;
    need(pithy_GetDecompressedLength(reinterpret_cast<const char*>(input.data()),body.size(),&declared)&&declared==expected,"Pithy declared length mismatch");
    need(pithy_Decompress(reinterpret_cast<const char*>(input.data()),body.size(),reinterpret_cast<char*>(out.data()),expected),"Pithy decompression failed");
  }else if(n=="Shrinker"){
    need(expected<(1ULL<<27),"Shrinker output limit");
    need(shrinker_decompress(input.data(),out.data(),static_cast<int>(expected))==static_cast<int>(expected),"Shrinker decoded length mismatch");
  }else{
    std::unique_ptr<xpack_decompressor,decltype(&xpack_free_decompressor)> context(xpack_alloc_decompressor(),xpack_free_decompressor);
    need(bool(context),"XPACK decompressor allocation failed");
    need(xpack_decompress(context.get(),input.data(),body.size(),out.data(),expected,nullptr)==DECOMPRESS_SUCCESS,"XPACK decompression failed");
  }
  out.resize(expected);return out;
}
}
