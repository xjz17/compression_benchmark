#include "openzl_layout_stages.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef OPENZL_LAYOUT_UPSTREAM_ORACLE
// Optional direct differential checks against unmodified upstream C kernels.
// Build those .c files separately and link their objects; ordinary tests have
// no source-tree or OpenZL-library dependency.
extern "C" {
void FLTDECON_float16_deconstruct_encode(const std::uint16_t*,std::uint8_t*,std::uint8_t*,std::size_t);
void FLTDECON_bfloat16_deconstruct_encode(const std::uint16_t*,std::uint8_t*,std::uint8_t*,std::size_t);
void FLTDECON_float32_deconstruct_encode(const std::uint32_t*,std::uint8_t*,std::uint8_t*,std::size_t);
void ZL_bitSplitEncode(void* const*,const std::size_t*,std::size_t,const void*,std::size_t,const std::uint8_t*,std::size_t);
void ZS_splitTransposeEncode(std::uint8_t**,const void*,std::size_t,std::size_t);
}
#endif

using namespace openzl_layout;
namespace {
std::size_t checks=0;
void check(bool condition,const char* message) { ++checks; if(!condition) throw std::runtime_error(message); }
template<class F> void rejects(F&& f) {
  ++checks;
  try { f(); } catch(const std::exception&) { return; }
  throw std::runtime_error("malformed input was accepted");
}
void put(Bytes& out,std::uint64_t v,unsigned width) {
  for(unsigned i=0;i<width;++i) out.push_back(static_cast<std::uint8_t>(v>>(8*i)));
}
Stream numeric(unsigned width,std::initializer_list<std::uint64_t> values) {
  Stream out{Kind::Numeric,width,{}}; for(auto v:values) put(out.data,v,width); return out;
}
void roundtrip(const SplitResult& result,const std::vector<Stream>& expected) {
  check(inverse(result)==expected,"direct inverse mismatch");
  const auto wire=serialize(result);
  check(inverse(deserialize(wire))==expected,"wire roundtrip mismatch");
  // Every split result contains only transformed substreams; replacing a stream
  // with a restored downstream codec result remains independently reversible.
  auto replacement=deserialize(wire);
  for(auto& stream:replacement.streams) {
    Bytes reversible=stream.data;
    for(auto& byte:reversible) byte^=0xA5;
    for(auto& byte:reversible) byte^=0xA5;
    stream.data=std::move(reversible);
  }
  check(inverse(replacement)==expected,"stream replacement roundtrip mismatch");
#ifdef OPENZL_LAYOUT_UPSTREAM_ORACLE
  if(expected.size()!=1) return;
  const auto& input=expected.front();
  if(input.data.empty()) return;
  if(result.operation==Operation::FloatDeconstruct) {
    Bytes sf(result.streams[0].data.size()),exponent(result.streams[1].data.size());
    const auto n=input.data.size()/input.width;
    if(result.side[0]==0) FLTDECON_float32_deconstruct_encode(reinterpret_cast<const std::uint32_t*>(input.data.data()),exponent.data(),sf.data(),n);
    else if(result.side[0]==1) FLTDECON_bfloat16_deconstruct_encode(reinterpret_cast<const std::uint16_t*>(input.data.data()),exponent.data(),sf.data(),n);
    else FLTDECON_float16_deconstruct_encode(reinterpret_cast<const std::uint16_t*>(input.data.data()),exponent.data(),sf.data(),n);
    check(sf==result.streams[0].data && exponent==result.streams[1].data,"upstream float component mismatch");
  } else if(result.operation==Operation::BitSplit) {
    std::vector<Bytes> buffers; std::vector<void*> pointers; std::vector<std::size_t> widths;
    for(const auto& s:result.streams) { buffers.emplace_back(s.data.size()); widths.push_back(s.width); }
    for(auto& b:buffers) pointers.push_back(b.data());
    ZL_bitSplitEncode(pointers.data(),widths.data(),input.data.size()/input.width,input.data.data(),input.width,result.side.data()+1,result.streams.size());
    for(std::size_t i=0;i<buffers.size();++i) check(buffers[i]==result.streams[i].data,"upstream bit split component mismatch");
  } else if(result.operation==Operation::Transpose) {
    std::vector<Bytes> buffers; std::vector<std::uint8_t*> pointers;
    for(const auto& s:result.streams) buffers.emplace_back(s.data.size());
    for(auto& b:buffers) pointers.push_back(b.data());
    ZS_splitTransposeEncode(pointers.data(),input.data.data(),input.data.size()/input.width,input.width);
    for(std::size_t i=0;i<buffers.size();++i) check(buffers[i]==result.streams[i].data,"upstream transpose component mismatch");
  }
#endif
}
void float_tests() {
  Stream all16{Kind::Numeric,2,{}};
  for(unsigned i=0;i<65536;++i) put(all16.data,i,2);
  for(const std::string format:{"fp16","bf16"}) {
    const auto result=float_deconstruct(all16,format); roundtrip(result,{all16});
    check(result.side.size()==1,"float metadata stores more than format");
    check(result.streams[0].width==(format=="fp16" ? 2u:1u),"incorrect signfrac width");
  }
  const auto fp16=numeric(2,{0xBE00,0xBF00});
  const auto example=float_deconstruct(fp16,"fp16");
  check(example.streams[0].data==Bytes({1,4,1,6}),"FP16 example signfrac mismatch");
  check(example.streams[1].data==Bytes({15,15}),"FP16 example exponent mismatch");
  auto invalid=example; invalid.streams[0].data[1]|=0x80; rejects([&]{inverse(invalid);});
  invalid=example; invalid.streams[1].data[0]=32; rejects([&]{inverse(invalid);});
  Stream fp32=numeric(4,{0,0x80000000,1,0x007FFFFF,0x00800000,0x7F800000,0xFF800000,0x7FC01234,0x7FA00001,0xFFC0ABCD,0x3FC00000,0xBFC00000});
  std::mt19937_64 gen(9182);
  for(unsigned i=0;i<50000;++i) put(fp32.data,gen(),4);
  const auto result=float_deconstruct(fp32,"fp32"); roundtrip(result,{fp32});
  check(result.streams[0].width==3 && result.streams[0].data.size()+result.streams[1].data.size()==fp32.data.size(),"FP32 signfrac not tightly stored");
  roundtrip(float_deconstruct({Kind::Numeric,4,{ }},"fp32"),{{Kind::Numeric,4,{}}});
  rejects([&]{float_deconstruct({Kind::Numeric,8,{ }},"fp64");});
}
void bit_tests() {
  std::mt19937_64 gen(444);
  for(unsigned width:{1u,2u,4u,8u}) {
    Stream source{Kind::Numeric,width,{}};
    for(unsigned i=0;i<1024;++i) put(source.data,gen(),width);
    std::vector<std::uint8_t> one_bits(width*8,1);
    roundtrip(bitsplit(source,one_bits),{source});
    std::vector<std::uint8_t> whole{static_cast<std::uint8_t>(width*8)};
    roundtrip(bitsplit(source,whole),{source});
    roundtrip(bitsplit_top8(source),{source});
    if(width>1) roundtrip(bitsplit_fp(source),{source});
    if(width==2) roundtrip(bitsplit_bf16(source),{source});
    const auto zero=numeric(width,{0,0,0}); roundtrip(bitsplit_top8(zero),{zero});
    roundtrip(bitsplit_top8({Kind::Numeric,width,{}}),{{Kind::Numeric,width,{}}});
  }
  const auto source=numeric(2,{512,513}); const auto r=bitsplit_top8(source);
  check(r.side==Bytes({2,2,8}),"top8 did not split effective 10 bits into 2+8");
  check(r.streams[0].data==Bytes({0,1}) && r.streams[1].data==Bytes({128,128}),"top8 expected columns mismatch");
  auto invalid=r; invalid.streams[0].data[0]=4; rejects([&]{inverse(invalid);});
  const std::uint8_t too_small[]={8}; rejects([&]{bitsplit(source,too_small);});
  const std::uint8_t too_large[]={64,1}; rejects([&]{bitsplit(numeric(8,{0}),too_large);});
  const std::uint8_t zero[]={0}; rejects([&]{bitsplit(source,zero);});
  const auto example=bitunpack(Bytes({0xE4}),2);
  check(example.streams[0].data==Bytes({0,1,2,3}),"bitunpack example wrong order");
  auto tail=bitunpack(Bytes({0xFF}),3); check(tail.side==Bytes({3,3}),"bitunpack nonzero remainder missing"); roundtrip(tail,{{Kind::Serial,1,{0xFF}}});
  invalid=tail; invalid.side[1]=4; rejects([&]{inverse(invalid);});
  invalid=tail; invalid.streams[0].data[0]=8; rejects([&]{inverse(invalid);});
  for(unsigned bits=1;bits<=64;++bits) {
    roundtrip(bitunpack({},bits),{{Kind::Serial,1,{}}});
    for(std::size_t size=1;size<=70;++size) {
      Bytes bytes(size); for(auto& b:bytes) b=static_cast<std::uint8_t>(gen());
      const auto used=size*8/bits*bits;
      if((used+7)/8==size) roundtrip(bitunpack(bytes,bits),{{Kind::Serial,1,bytes}});
      else rejects([&]{bitunpack(bytes,bits);});
    }
  }
}
void layout_tests() {
  const Stream rows{Kind::Struct,3,{1,0,0,2,0,0}};
  const auto r=transpose(rows); check(r.streams.size()==3 && r.streams[0].data==Bytes({1,2}),"transpose column mismatch"); roundtrip(r,{rows});
  const std::uint32_t widths[]={1,2}; const auto fields=split_struct(rows.data,widths);
  check(fields.streams[0]==Stream{Kind::Struct,1,{1,2}} && fields.streams[1]==Stream{Kind::Struct,2,{0,0,0,0}},"split struct fields mismatch");
  roundtrip(fields,{{Kind::Serial,1,rows.data}});
  roundtrip(transpose({Kind::Struct,4,{}}),{{Kind::Struct,4,{}}});
  roundtrip(split_struct({},widths),{{Kind::Serial,1,{}}});
  rejects([&]{split_struct(Bytes({1,2}),widths);});
  const Stream nums=numeric(2,{10,11,90,91,92}); const std::uint64_t sizes[]={2,0};
  const auto split=split_segments(nums,sizes); roundtrip(split,{nums});
  check(split.streams[0]==numeric(2,{10,11}) && split.streams[1]==numeric(2,{90,91,92}),"continuous split shape mismatch");
  roundtrip(concat(split.streams),split.streams);
  const auto concatenated=concat(split.streams); check(concatenated.streams[0].data==Bytes({2,0,0,0,3,0,0,0}),"concat uint32 lengths mismatch");
  roundtrip(split_segments({Kind::Serial,1,{}},{}),{{Kind::Serial,1,{}}});
  roundtrip(concat({{Kind::Struct,3,{}},{Kind::Struct,3,{}}}),{{Kind::Struct,3,{}},{Kind::Struct,3,{}}});
  const std::uint64_t excess[]={2,4}; rejects([&]{split_segments(nums,excess);});
  const std::uint64_t missing[]={2}; rejects([&]{split_segments(nums,missing);});
  rejects([&]{concat({numeric(1,{1}),numeric(2,{2})});});
  auto invalid=concatenated; invalid.streams[0].data[0]=99; rejects([&]{inverse(invalid);});
  invalid=split; invalid.input_bytes+=2; rejects([&]{inverse(invalid);});
}
void range_tests() {
  Stream source{Kind::Numeric,4,{}};
  for(unsigned i=0;i<500;++i) put(source.data,100+i%11,4);
  for(unsigned i=0;i<500;++i) put(source.data,500+i%11,4);
  auto r=split_by_range(source); roundtrip(r,{source});
  check(r.streams.size()==2 && r.streams[0].data.size()==2000 && r.streams[1].data.size()==2000,"range split failed 500/500 boundary");
  for(unsigned width:{1u,2u,4u,8u}) {
    Stream values{Kind::Numeric,width,{}};
    for(unsigned segment=0;segment<3;++segment)
      for(unsigned i=0;i<512;++i) put(values.data,segment*80+i%11,width);
    r=split_by_range(values); roundtrip(r,{values});
    check(r.streams.size()==3,"range split failed three ranges");
    for(const auto& s:r.streams) check(s.data.size()==512*width,"range split nonexact boundary");
    const auto small=numeric(width,{1,200}); r=split_by_range(small); roundtrip(r,{small});
    check(r.streams.size()==1,"default minimum ignored");
    roundtrip(split_by_range({Kind::Numeric,width,{}}),{{Kind::Numeric,width,{}}});
  }
  Stream ramp{Kind::Numeric,4,{}}; for(unsigned i=0;i<1000;++i) put(ramp.data,i,4);
  r=split_by_range(ramp); roundtrip(r,{ramp}); check(r.streams.size()<=2,"monotonic ramp oversegmented");
  Stream extreme{Kind::Numeric,8,{}};
  for(unsigned i=0;i<500;++i) put(extreme.data,i%3,8);
  for(unsigned i=0;i<500;++i) put(extreme.data,UINT64_MAX-i%3,8);
  roundtrip(split_by_range(extreme),{extreme});
}
void wire_tests() {
  const auto result=transpose({Kind::Struct,2,{1,2,3,4}}); const auto wire=serialize(result);
  for(std::size_t i=0;i<wire.size();++i) rejects([&]{deserialize(std::span(wire).first(i));});
  auto invalid=wire; invalid.push_back(0); rejects([&]{deserialize(invalid);});
  invalid=wire; invalid[4]=255; rejects([&]{deserialize(invalid);});
  invalid=wire; invalid[5]=0; rejects([&]{deserialize(invalid);});
  invalid=wire; invalid[22]=255; rejects([&]{deserialize(invalid);});
  auto bad=result; bad.side={0}; rejects([&]{inverse(bad);});
  bad=result; bad.streams[0].data.pop_back(); rejects([&]{inverse(bad);});
  const auto raw=raw_bundle({numeric(2,{1,2}),{Kind::Struct,3,{1,2,3}}}); roundtrip(raw,raw.streams);
}
} // namespace
int main() try {
  float_tests(); bit_tests(); layout_tests(); range_tests(); wire_tests();
  std::cout<<"openzl_layout_stages: "<<checks<<" checks passed; fp16/bf16 exhaustively round-tripped all 65536 bit patterns each\n";
  return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
