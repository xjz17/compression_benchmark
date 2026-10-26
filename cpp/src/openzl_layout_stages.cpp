/*
 * Source-informed port of OpenZL scalar layout transformations and SplitByRange.
 * Source: https://github.com/facebook/openzl/tree/423e1c8f2d3abb53709f197bfa880c587db8c9ca/src/openzl/codecs
 * Corresponding codec sources are unchanged in efba48d8614f5b6c0c7fcbc7b8ebacb560308a19.
 * Transport, validation and C++ interfaces are local, not OpenZL-wire-compatible.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name Facebook, nor Meta, nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "openzl_layout_stages.hpp"
#include "operator_layout_primitives.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace openzl_layout {
namespace {
constexpr std::size_t kMaxStreams = 65536;
[[noreturn]] void fail(const char* message) { throw std::invalid_argument(message); }
void require(bool condition, const char* message) { if (!condition) fail(message); }
bool numeric_width(std::uint32_t width) { return width == 1 || width == 2 || width == 4 || width == 8; }
void type_width(Kind kind, std::uint32_t width) {
  require(width > 0 && width <= kMaxStreams, "invalid element width");
  if (kind == Kind::Serial) require(width == 1, "serial width must be one");
  else if (kind == Kind::Numeric) require(numeric_width(width), "invalid numeric width");
  else require(kind == Kind::Struct, "unknown stream kind");
}
void validate(const Stream& s) {
  type_width(s.kind, s.width);
  require(s.data.size() % s.width == 0, "stream byte count is not width aligned");
}
std::size_t checked_size(std::uint64_t n) {
  require(n <= std::numeric_limits<std::size_t>::max(), "size overflows size_t");
  return static_cast<std::size_t>(n);
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
  require(b <= UINT64_MAX - a, "size addition overflow"); return a + b;
}
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
  require(a == 0 || b <= UINT64_MAX / a, "size multiplication overflow"); return a * b;
}
void put(Bytes& bytes, std::uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8*i)));
}
std::uint64_t read(std::span<const std::uint8_t> data, std::size_t position, unsigned width) {
  require(width <= 8 && position <= data.size() && width <= data.size()-position, "truncated value");
  std::uint64_t value = 0;
  for (unsigned i=0; i<width; ++i) value |= std::uint64_t(data[position+i]) << (8*i);
  return value;
}
struct Reader {
  std::span<const std::uint8_t> bytes;
  std::size_t position = 0;
  std::uint64_t get(unsigned width) { auto n = read(bytes, position, width); position += width; return n; }
  Bytes take(std::uint64_t n) {
    require(n <= bytes.size()-position, "truncated byte region");
    auto end = position + checked_size(n);
    Bytes out(bytes.begin()+position, bytes.begin()+end); position=end; return out;
  }
  void end() const { require(position==bytes.size(), "unexpected trailing bytes"); }
};
std::uint64_t mask(unsigned bits) { return bits == 64 ? UINT64_MAX : (std::uint64_t(1)<<bits)-1; }
unsigned storage_width(unsigned bits) { return bits<=8 ? 1 : bits<=16 ? 2 : bits<=32 ? 4 : 8; }
std::size_t count(const Stream& s) { return s.data.size()/s.width; }
SplitResult begin(Operation op, const Stream& input) {
  validate(input); return {op, input.kind, input.width, input.data.size(), {}, {}};
}
void append(Bytes& dst, std::span<const std::uint8_t> src) { dst.insert(dst.end(), src.begin(), src.end()); }
void expect_stream(const Stream& s, Kind kind, std::uint32_t width, std::uint64_t n) {
  validate(s);
  require(s.kind==kind && s.width==width && s.data.size()==multiply(n,width), "output stream shape mismatch");
}
void common(const SplitResult& r) {
  require(static_cast<unsigned>(r.operation)<=8, "unknown layout operation");
  type_width(r.input_kind,r.input_width);
  require(r.input_bytes%r.input_width==0, "original byte count is not width aligned");
  require(r.streams.size()<=kMaxStreams, "too many streams");
  require(r.side.size()<=UINT32_MAX, "side information too large");
  for (const auto& s:r.streams) validate(s);
}

// OpenZL splitN/encode_split_byrange_binding.c: block statistics, confirmed
// non-overlapping ranges, and prefix/suffix refinement. Scalar LE input loads
// replace native typed-pointer loads; overflow-prone threshold products are
// widened without altering the mathematical predicate.
struct Range { std::uint64_t lo=UINT64_MAX, hi=0; };
void include(Range& r,std::uint64_t v) { r.lo=std::min(r.lo,v); r.hi=std::max(r.hi,v); }
void include(Range& r,Range v) { r.lo=std::min(r.lo,v.lo); r.hi=std::max(r.hi,v.hi); }
std::uint64_t gap(Range l,Range r) { return l.hi<r.lo ? r.lo-l.hi : r.hi<l.lo ? l.lo-r.hi : 0; }
bool stationary(Range window,std::uint64_t max_amplitude) {
  return max_amplitude==0 || static_cast<unsigned __int128>(window.hi-window.lo)*10 <= static_cast<unsigned __int128>(max_amplitude)*13;
}
std::vector<std::uint64_t> detect_ranges(const Stream& input,std::size_t minimum) {
  const auto n=count(input);
  if (!n) return {};
  if (minimum>n/2) return {n};
  const auto block=std::max<std::size_t>(minimum/2,1);
  const auto blocks=n/block+(n%block!=0);
  if (blocks<3) return {n};
  std::vector<Range> ranges(blocks);
  for(std::size_t b=0;b<blocks;++b)
    for(std::size_t i=b*block;i<std::min(n,(b+1)*block);++i)
      include(ranges[b],read(input.data,i*input.width,input.width));
  const auto m=std::max<std::size_t>(1,std::min<std::size_t>((blocks-1)/2,7));
  const auto minimum_left=m>=2 ? 2u:1u;
  std::vector<std::size_t> splits;
  std::size_t left_bound=0,b=m-1;
  while(b+m+1<blocks) {
    const auto left_start=std::max(b-m+1,left_bound);
    if(b-left_start+1<minimum_left) { ++b; continue; }
    Range left,right;
    std::uint64_t left_amp=0,right_amp=0;
    for(auto i=left_start;i<=b;++i) { include(left,ranges[i]); left_amp=std::max(left_amp,ranges[i].hi-ranges[i].lo); }
    for(auto i=b+2;i<=b+m+1;++i) { include(right,ranges[i]); right_amp=std::max(right_amp,ranges[i].hi-ranges[i].lo); }
    const auto separation=gap(left,right);
    const auto amplitude=std::max(left_amp,right_amp);
    const auto transition=ranges[b+1].hi-ranges[b+1].lo;
    const bool significant=amplitude==0 || static_cast<unsigned __int128>(separation)>static_cast<unsigned __int128>(amplitude)*2 ||
      static_cast<unsigned __int128>(transition)>static_cast<unsigned __int128>(amplitude)*2 || (stationary(left,left_amp)&&stationary(right,right_amp));
    if(separation && significant) { splits.push_back(b); left_bound=b+2; b+=2; }
    else ++b;
  }
  std::vector<std::uint64_t> sizes;
  std::size_t position=0;
  for(std::size_t s=0;s<splits.size();++s) {
    const auto sb=splits[s];
    const auto end_block=s+1<splits.size() ? splits[s+1]+1 : blocks;
    const auto start=std::max(position,sb>0 ? (sb-1)*block : 0);
    const auto end=std::min(n,(sb+3)*block);
    if(end<=position+1) continue;
    std::vector<Range> suffix(end-start);
    for(std::size_t i=end;i>start;--i) {
      if(i<end) suffix[i-1-start]=suffix[i-start];
      include(suffix[i-1-start],read(input.data,(i-1)*input.width,input.width));
    }
    const auto context_start=end/block+(end%block!=0);
    if(context_start<end_block) {
      Range context;
      for(auto i=context_start;i<end_block;++i) include(context,ranges[i]);
      for(auto& r:suffix) include(r,context);
    }
    Range prefix;
    include(prefix,read(input.data,position*input.width,input.width));
    auto best=(sb+1)*block;
    std::uint64_t best_gap=0;
    for(auto i=position+1;i<end;++i) {
      if(i>=start) {
        const auto g=gap(prefix,suffix[i-start]);
        if(g>best_gap) { best_gap=g; best=i; }
      }
      include(prefix,read(input.data,i*input.width,input.width));
    }
    best=std::max(best,position+minimum);
    if(best>n-minimum || best<=position) continue;
    sizes.push_back(best-position); position=best;
  }
  sizes.push_back(n-position);
  return sizes;
}
} // namespace

Bytes serialize(const SplitResult& result) {
  common(result);
  Bytes out={'O','L','S','1'};
  put(out,static_cast<unsigned>(result.operation),1);
  put(out,static_cast<unsigned>(result.input_kind),1);
  put(out,result.input_width,4); put(out,result.input_bytes,8);
  put(out,result.side.size(),4); put(out,result.streams.size(),4);
  append(out,result.side);
  for(const auto& s:result.streams) {
    put(out,static_cast<unsigned>(s.kind),1); put(out,s.width,4);
    put(out,s.data.size(),8); append(out,s.data);
  }
  return out;
}
SplitResult deserialize(std::span<const std::uint8_t> bytes) {
  Reader r{bytes};
  require(r.take(4)==Bytes({'O','L','S','1'}),"invalid OLS1 magic/version");
  SplitResult out;
  out.operation=static_cast<Operation>(r.get(1)); out.input_kind=static_cast<Kind>(r.get(1));
  out.input_width=static_cast<std::uint32_t>(r.get(4)); out.input_bytes=r.get(8);
  const auto side_size=r.get(4), streams=r.get(4);
  out.side=r.take(side_size);
  require(streams<=kMaxStreams && streams<=(bytes.size()-r.position)/13,"invalid stream count");
  for(std::uint64_t i=0;i<streams;++i) {
    Stream s; s.kind=static_cast<Kind>(r.get(1)); s.width=static_cast<std::uint32_t>(r.get(4));
    s.data=r.take(r.get(8)); out.streams.push_back(std::move(s));
  }
  r.end(); common(out); return out;
}
SplitResult raw_bundle(std::vector<Stream> streams) {
  SplitResult out; out.streams=std::move(streams);
  for(const auto& s:out.streams) { validate(s); out.input_bytes=add(out.input_bytes,s.data.size()); }
  common(out); return out;
}
SplitResult transpose(const Stream& input) {
  require(input.kind==Kind::Struct,"transpose requires Struct input");
  auto r=begin(Operation::Transpose,input); r.streams.resize(input.width);
  const auto columns=exact_actions::operator_layout::columnize(input.data,input.width);
  for(std::size_t j=0;j<input.width;++j) {
    r.streams[j].data=Bytes(columns.begin()+j*count(input),columns.begin()+(j+1)*count(input));
  }
  return r;
}
SplitResult split_struct(std::span<const std::uint8_t> bytes,std::span<const std::uint32_t> widths) {
  require(!widths.empty() && widths.size()<=kMaxStreams,"invalid field count");
  std::uint64_t row_width=0;
  for(auto w:widths) { type_width(Kind::Struct,w); row_width=add(row_width,w); }
  require(bytes.size()%row_width==0,"partial struct record");
  Stream input{Kind::Serial,1,Bytes(bytes.begin(),bytes.end())};
  auto r=begin(Operation::SplitStruct,input); put(r.side,widths.size(),4);
  const auto n=bytes.size()/row_width;
  std::size_t offset=0;
  for(auto w:widths) {
    put(r.side,w,4); Stream out{Kind::Struct,w,{}};
    out.data.reserve(n*w);
    for(std::size_t i=0;i<n;++i) append(out.data,bytes.subspan(i*row_width+offset,w));
    r.streams.push_back(std::move(out)); offset+=w;
  }
  return r;
}
SplitResult float_deconstruct(const Stream& input,const std::string& format) {
  require(input.kind==Kind::Numeric,"float deconstruction requires Numeric bit patterns");
  unsigned id,frac,exp,width;
  if(format=="fp32") { id=0; frac=23; exp=8; width=4; }
  else if(format=="bf16") { id=1; frac=7; exp=8; width=2; }
  else if(format=="fp16") { id=2; frac=10; exp=5; width=2; }
  else fail("unsupported float deconstruction format");
  require(input.width==width,"float format width mismatch");
  auto r=begin(Operation::FloatDeconstruct,input); put(r.side,id,1);
  r.streams={{Kind::Struct,(frac+1+7)/8,{}},{Kind::Serial,1,{}}};
  for(std::size_t i=0;i<count(input);++i) {
    const auto value=read(input.data,i*width,width);
    put(r.streams[0].data,((value&mask(frac))<<1)|(value>>(width*8-1)),r.streams[0].width);
    put(r.streams[1].data,(value>>frac)&mask(exp),1);
  }
  return r;
}
SplitResult bitsplit(const Stream& input,std::span<const std::uint8_t> widths) {
  require(input.kind==Kind::Numeric,"bitsplit requires Numeric input");
  auto r=begin(Operation::BitSplit,input);
  require(!widths.empty() && widths.size()<=64,"invalid bit field count");
  unsigned total=0;
  for(auto w:widths) { require(w>0 && w<=64,"invalid bit field width"); total+=w; }
  require(total<=input.width*8,"bit fields exceed input width");
  put(r.side,widths.size(),1);
  for(auto w:widths) { put(r.side,w,1); r.streams.push_back({Kind::Numeric,storage_width(w),{}}); }
  const exact_actions::operator_layout::BitFields layout(widths,input.width*8);
  for(std::size_t i=0;i<count(input);++i) {
    const auto value=read(input.data,i*input.width,input.width);
    require(total==64 || (value>>total)==0,"omitted top bits are nonzero");
    for(std::size_t j=0;j<widths.size();++j) {
      put(r.streams[j].data,layout.field(value,j),r.streams[j].width);
    }
  }
  return r;
}
SplitResult bitsplit_top8(const Stream& input) {
  validate(input); require(input.kind==Kind::Numeric,"top8 requires Numeric input");
  std::uint64_t maximum=0;
  for(std::size_t i=0;i<count(input);++i) maximum=std::max(maximum,read(input.data,i*input.width,input.width));
  unsigned bits=0; while(maximum) { ++bits; maximum>>=1; }
  if(bits<=8) { const std::uint8_t b=static_cast<std::uint8_t>(std::max(bits,1u)); return bitsplit(input,{&b,1}); }
  const std::uint8_t b[]={static_cast<std::uint8_t>(bits-8),8}; return bitsplit(input,b);
}
SplitResult bitsplit_fp(const Stream& input) {
  if(input.width==2) { const std::uint8_t w[]={10,5,1}; return bitsplit(input,w); }
  if(input.width==4) { const std::uint8_t w[]={23,8,1}; return bitsplit(input,w); }
  if(input.width==8) { const std::uint8_t w[]={52,11,1}; return bitsplit(input,w); }
  fail("IEEE bitsplit requires width 2, 4 or 8");
}
SplitResult bitsplit_bf16(const Stream& input) {
  require(input.width==2,"BF16 requires width 2"); const std::uint8_t w[]={7,8,1}; return bitsplit(input,w);
}
SplitResult bitunpack(std::span<const std::uint8_t> bytes,unsigned bits) {
  require(bits>=1 && bits<=64,"invalid bitunpack width");
  const auto total=multiply(bytes.size(),8), n=total/bits, used=multiply(n,bits);
  require((used+7)/8==bytes.size(),"bitunpack cannot preserve an unused whole byte");
  Stream input{Kind::Serial,1,Bytes(bytes.begin(),bytes.end())};
  auto r=begin(Operation::Bitunpack,input);
  const auto remainder=static_cast<unsigned>(total-used);
  const auto remainder_value=remainder ? bytes.back()>>(8-remainder) : 0;
  r.side={static_cast<std::uint8_t>(bits),static_cast<std::uint8_t>(remainder_value)};
  r.streams={{Kind::Numeric,storage_width(bits),{}}};
  auto& output=r.streams[0].data; output.reserve(checked_size(multiply(n,r.streams[0].width)));
  for(std::uint64_t i=0;i<n;++i) {
    std::uint64_t v=0;
    for(unsigned j=0;j<bits;++j) { const auto bit=i*bits+j; v|=std::uint64_t((bytes[bit/8]>>(bit%8))&1)<<j; }
    put(output,v,r.streams[0].width);
  }
  return r;
}
SplitResult concat(const std::vector<Stream>& inputs) {
  require(!inputs.empty() && inputs.size()<=kMaxStreams,"concat requires 1..65536 inputs");
  auto r=begin(Operation::Concat,inputs.front()); r.input_bytes=0;
  r.streams={{Kind::Numeric,4,{}},{inputs.front().kind,inputs.front().width,{}}};
  for(const auto& input:inputs) {
    validate(input); require(input.kind==r.input_kind && input.width==r.input_width,"concat input types differ");
    require(count(input)<UINT32_MAX,"concat input element count too large");
    r.input_bytes=add(r.input_bytes,input.data.size()); put(r.streams[0].data,count(input),4); append(r.streams[1].data,input.data);
  }
  return r;
}
SplitResult split_segments(const Stream& input,std::span<const std::uint64_t> sizes) {
  auto r=begin(Operation::SplitSegments,input);
  require(sizes.size()<=kMaxStreams,"too many segments");
  std::uint64_t position=0;
  for(std::size_t i=0;i<sizes.size();++i) {
    const auto n=(i+1==sizes.size() && sizes[i]==0) ? count(input)-position : sizes[i];
    require(n<=count(input)-position,"segment exceeds input");
    const auto bytes=std::span(input.data).subspan(checked_size(position*input.width),checked_size(n*input.width));
    r.streams.push_back({input.kind,input.width,Bytes(bytes.begin(),bytes.end())}); position+=n;
  }
  require(position==count(input),"segments do not cover input"); return r;
}
SplitResult split_by_range(const Stream& input,std::uint64_t minimum) {
  validate(input); require(input.kind==Kind::Numeric,"range splitting requires Numeric input");
  if(!minimum) minimum=input.width==1 ? 128 : input.width==2 ? 64 : 32;
  auto sizes=detect_ranges(input,checked_size(minimum));
  auto r=split_segments(input,sizes); r.operation=Operation::SplitByRange; return r;
}

std::vector<Stream> inverse(const SplitResult& r) {
  common(r); const auto n=r.input_bytes/r.input_width;
  Stream out{r.input_kind,r.input_width,{}};
  Reader side{r.side};
  switch(r.operation) {
  case Operation::Raw: {
    require(r.input_kind==Kind::Serial && r.input_width==1,"noncanonical raw descriptor"); side.end();
    std::uint64_t bytes=0; for(const auto& s:r.streams) bytes=add(bytes,s.data.size());
    require(bytes==r.input_bytes,"raw bundle size mismatch"); return r.streams;
  }
  case Operation::Transpose: {
    side.end(); require(r.input_kind==Kind::Struct && r.streams.size()==r.input_width,"transpose metadata mismatch");
    for(const auto& s:r.streams) expect_stream(s,Kind::Serial,1,n);
    Bytes columns; columns.reserve(checked_size(r.input_bytes));
    for(const auto& s:r.streams) append(columns,s.data);
    out.data=exact_actions::operator_layout::uncolumnize(columns,r.input_width);
    break;
  }
  case Operation::SplitStruct: {
    require(r.input_kind==Kind::Serial && r.input_width==1,"split struct input type mismatch");
    const auto fields=side.get(4); require(fields>0 && fields==r.streams.size(),"field count mismatch");
    std::uint64_t row_width=0; std::vector<std::uint32_t> widths;
    for(std::uint64_t i=0;i<fields;++i) { auto w=static_cast<std::uint32_t>(side.get(4)); type_width(Kind::Struct,w); row_width=add(row_width,w); widths.push_back(w); }
    side.end(); require(r.input_bytes%row_width==0,"partial original struct record");
    const auto rows=r.input_bytes/row_width;
    for(std::size_t i=0;i<fields;++i) expect_stream(r.streams[i],Kind::Struct,widths[i],rows);
    out.data.reserve(checked_size(r.input_bytes));
    for(std::uint64_t i=0;i<rows;++i) for(const auto& s:r.streams) append(out.data,std::span(s.data).subspan(i*s.width,s.width));
    break;
  }
  case Operation::FloatDeconstruct: {
    const auto format=side.get(1); side.end(); require(format<=2,"unknown float format");
    const unsigned frac=format==0 ? 23 : format==1 ? 7 : 10;
    const unsigned exponent=format==2 ? 5 : 8, width=format==0 ? 4:2;
    require(r.input_kind==Kind::Numeric && r.input_width==width && r.streams.size()==2,"float metadata mismatch");
    expect_stream(r.streams[0],Kind::Struct,(frac+8)/8,n); expect_stream(r.streams[1],Kind::Serial,1,n);
    out.data.reserve(checked_size(r.input_bytes));
    for(std::uint64_t i=0;i<n;++i) {
      const auto sf=read(r.streams[0].data,i*r.streams[0].width,r.streams[0].width);
      const auto e=r.streams[1].data[i];
      require(sf<=mask(frac+1) && e<=mask(exponent),"float component has nonzero padding bits");
      put(out.data,((sf&1)<<(width*8-1))|(std::uint64_t(e)<<frac)|(sf>>1),width);
    }
    break;
  }
  case Operation::BitSplit: {
    require(r.input_kind==Kind::Numeric,"bitsplit original kind mismatch");
    const auto fields=side.get(1); require(fields>0 && fields<=64 && fields==r.streams.size(),"bit field count mismatch");
    std::vector<std::uint8_t> widths; unsigned total=0;
    for(std::size_t i=0;i<fields;++i) { auto w=static_cast<unsigned>(side.get(1)); require(w>=1 && w<=64,"invalid bit field width"); total+=w; widths.push_back(w); expect_stream(r.streams[i],Kind::Numeric,storage_width(w),n); }
    side.end(); require(total<=r.input_width*8,"bit fields exceed original width");
    const exact_actions::operator_layout::BitFields layout(widths,r.input_width*8);
    out.data.reserve(checked_size(r.input_bytes));
    std::vector<std::uint64_t> values(fields);
    for(std::uint64_t i=0;i<n;++i) {
      for(std::size_t j=0;j<fields;++j) { const auto& s=r.streams[j]; values[j]=read(s.data,i*s.width,s.width); }
      put(out.data,layout.join(values),r.input_width);
    }
    break;
  }
  case Operation::Bitunpack: {
    require(r.input_kind==Kind::Serial && r.input_width==1 && r.streams.size()==1,"bitunpack metadata mismatch");
    const auto bits=static_cast<unsigned>(side.get(1)), rem=static_cast<unsigned>(side.get(1)); side.end();
    require(bits>=1 && bits<=64,"invalid bitunpack width"); const auto total=multiply(r.input_bytes,8), elements=total/bits, used=elements*bits;
    require((used+7)/8==r.input_bytes,"bitunpack original size cannot be represented");
    const auto rem_bits=static_cast<unsigned>(total-used);
    require(rem<=mask(rem_bits),"bitunpack remainder exceeds available bits");
    expect_stream(r.streams[0],Kind::Numeric,storage_width(bits),elements);
    out.data.resize(checked_size(r.input_bytes),0);
    for(std::uint64_t i=0;i<elements;++i) {
      const auto value=read(r.streams[0].data,i*r.streams[0].width,r.streams[0].width); require(value<=mask(bits),"unpacked value exceeds width");
      for(unsigned j=0;j<bits;++j) { const auto bit=i*bits+j; out.data[bit/8]|=static_cast<std::uint8_t>(((value>>j)&1)<<(bit%8)); }
    }
    if(rem_bits) out.data.back()|=static_cast<std::uint8_t>(rem<<(8-rem_bits));
    break;
  }
  case Operation::Concat: {
    side.end(); require(r.streams.size()==2,"concat requires sizes and data streams");
    const auto& sizes=r.streams[0]; require(sizes.kind==Kind::Numeric && sizes.width==4 && count(sizes)>0 && count(sizes)<=kMaxStreams,"invalid concat lengths stream");
    expect_stream(r.streams[1],r.input_kind,r.input_width,n);
    std::vector<Stream> outputs; std::uint64_t position=0;
    for(std::size_t i=0;i<count(sizes);++i) {
      const auto length=read(sizes.data,4*i,4); require(length<UINT32_MAX && length<=n-position,"concat length exceeds input");
      const auto bytes=std::span(r.streams[1].data).subspan(checked_size(position*r.input_width),checked_size(length*r.input_width));
      outputs.push_back({r.input_kind,r.input_width,Bytes(bytes.begin(),bytes.end())}); position+=length;
    }
    require(position==n,"concat lengths do not cover data"); return outputs;
  }
  case Operation::SplitSegments:
  case Operation::SplitByRange: {
    side.end(); if(r.operation==Operation::SplitByRange) require(r.input_kind==Kind::Numeric,"range split original type mismatch");
    std::uint64_t bytes=0;
    for(const auto& s:r.streams) { require(s.kind==r.input_kind && s.width==r.input_width,"segment type mismatch"); bytes=add(bytes,s.data.size()); }
    require(bytes==r.input_bytes,"segments do not cover original bytes");
    out.data.reserve(checked_size(bytes)); for(const auto& s:r.streams) append(out.data,s.data);
    break;
  }
  }
  require(out.data.size()==r.input_bytes,"reconstruction byte count mismatch"); return {std::move(out)};
}
} // namespace openzl_layout
