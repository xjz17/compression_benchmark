#include "unified_rle_numeric.hpp"
#include "optimal_rle.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <limits>

namespace unified_rle_numeric {
namespace {
constexpr std::size_t kBlock=10000;
constexpr std::array<std::uint8_t,4> kMagic{'U','R','B','1'};
void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}
void put(Bytes& out,std::uint64_t value,unsigned width) {
  for(unsigned i=0;i<width;++i)out.push_back(static_cast<std::uint8_t>(value>>(i*8)));
}
std::uint64_t get(std::span<const std::uint8_t> raw,std::size_t& pos,unsigned width) {
  require(pos<=raw.size() && width<=raw.size()-pos,"truncated URB1 field");
  std::uint64_t value=0;
  for(unsigned i=0;i<width;++i)value|=std::uint64_t(raw[pos++])<<(i*8);
  return value;
}
std::uint64_t mask(unsigned width) {
  return width==8?std::numeric_limits<std::uint64_t>::max():(std::uint64_t{1}<<(width*8))-1;
}
struct BitWriter {
  Bytes raw;
  unsigned used=0;
  void value(std::uint64_t value,unsigned width) {
    for(unsigned i=width;i>0;--i){
      if(!used)raw.push_back(0);
      raw.back()|=static_cast<std::uint8_t>(((value>>(i-1))&1)<<(7-used));
      used=(used+1)%8;
    }
  }
};
struct BitReader {
  std::span<const std::uint8_t> raw;
  std::size_t bit=0;
  std::uint64_t value(unsigned width) {
    require(width<=64 && bit<=raw.size()*8 && width<=raw.size()*8-bit,"truncated URB1 residual");
    std::uint64_t value=0;
    for(unsigned i=0;i<width;++i,++bit)value=(value<<1)|((raw[bit/8]>>(7-bit%8))&1);
    return value;
  }
  void finish() {
    require((bit+7)/8==raw.size(),"extra URB1 residual bytes");
    while(bit<raw.size()*8)require(value(1)==0,"nonzero URB1 padding");
  }
};
Bytes encode_impl(std::span<const std::uint8_t> raw,unsigned width,unsigned mode) {
  require(width>=1 && width<=8 && raw.size()%width==0,"invalid URB1 symbol width");
  require(mode<=2 && (mode!=2 || width==4),"invalid URB1 delta mode");
  const std::size_t count=raw.size()/width;
  Bytes out(kMagic.begin(),kMagic.end());
  out.push_back(static_cast<std::uint8_t>(width));out.push_back(static_cast<std::uint8_t>(mode));
  put(out,0,2);put(out,count,8);put(out,kBlock,4);
  std::size_t pos=0;
  for(std::size_t begin=0;begin<count;begin+=kBlock) {
    const auto n=std::min(kBlock,count-begin);
    Bytes widths; widths.reserve(n);
    BitWriter payload;
    std::uint64_t previous=0;
    for(std::size_t i=0;i<n;++i) {
      const auto current=get(raw,pos,width);
      std::uint64_t delta=current;
      if(i && mode==1)delta=(current-previous)&mask(width);
      else if(i && mode==2) {
        const float a=std::bit_cast<float>(static_cast<std::uint32_t>(current));
        const float b=std::bit_cast<float>(static_cast<std::uint32_t>(previous));
        const float diff=a-b;
        // The old float32 RLBE operation is arithmetic delta. Preserve that
        // operation and reject values it cannot reconstruct bit for bit.
        const float restored=b+diff;
        require(std::bit_cast<std::uint32_t>(restored)==current,"float32 RLBE arithmetic delta is not bitwise reversible");
        delta=std::bit_cast<std::uint32_t>(diff);
      }
      const unsigned bits=std::max(1u,static_cast<unsigned>(std::bit_width(delta)));
      widths.push_back(static_cast<std::uint8_t>(bits));payload.value(delta,bits);
      previous=current;
    }
    const auto width_frame=unified_rle::encode(widths,1);
    require(width_frame.size()<=UINT32_MAX && payload.raw.size()<=UINT32_MAX,"URB1 block too large");
    put(out,width_frame.size(),4);put(out,payload.raw.size(),4);
    out.insert(out.end(),width_frame.begin(),width_frame.end());
    out.insert(out.end(),payload.raw.begin(),payload.raw.end());
  }
  return out;
}
}  // namespace

Bytes encode_rle(std::span<const std::uint8_t> raw,unsigned width){
  require(width>=1&&width<=8,"invalid RLE symbol width");
  return unified_rle::encode(raw,static_cast<std::uint8_t>(width));
}
Bytes decode_rle(std::span<const std::uint8_t> frame,std::size_t max_output_bytes){return unified_rle::decode(frame,max_output_bytes);}
bool is_rle_frame(std::span<const std::uint8_t> frame){return unified_rle::is_frame(frame);}
unsigned rle_symbol_width(std::span<const std::uint8_t> frame){return unified_rle::symbol_width(frame);}
bool is_rlbe_frame(std::span<const std::uint8_t> frame){return frame.size()>=4 && std::equal(kMagic.begin(),kMagic.end(),frame.begin());}
unsigned rlbe_symbol_width(std::span<const std::uint8_t> frame){require(is_rlbe_frame(frame)&&frame.size()>=20,"invalid URB1 header");return frame[4];}
unsigned rlbe_delta_mode(std::span<const std::uint8_t> frame){require(is_rlbe_frame(frame)&&frame.size()>=20,"invalid URB1 header");return frame[5];}
Bytes encode_rlbe(std::span<const std::uint8_t> raw,unsigned width,bool delta_enabled){return encode_impl(raw,width,delta_enabled?1:0);}
Bytes encode_rlbe_float32_arithmetic(std::span<const std::uint8_t> raw){
  require(raw.size()%4==0,"unaligned float32 RLBE input");
  std::size_t pos=0;
  std::uint32_t previous=0;
  for(std::size_t i=0;i<raw.size()/4;++i) {
    const auto current=static_cast<std::uint32_t>(get(raw,pos,4));
    if(i%kBlock) {
      const float a=std::bit_cast<float>(current),b=std::bit_cast<float>(previous);
      const float delta=a-b;
      if(std::bit_cast<std::uint32_t>(b+delta)!=current)
        return encode_impl(raw,4,0);
    }
    previous=current;
  }
  return encode_impl(raw,4,2);
}
Bytes decode_rlbe(std::span<const std::uint8_t> frame,std::size_t max_output_bytes) {
  const unsigned width=rlbe_symbol_width(frame),mode=rlbe_delta_mode(frame);
  require(width>=1 && width<=8 && mode<=2 && (mode!=2||width==4),"invalid URB1 width or delta mode");
  std::size_t pos=6;
  require(get(frame,pos,2)==0,"unsupported URB1 flags");
  const auto count=get(frame,pos,8);
  require(get(frame,pos,4)==kBlock,"unsupported URB1 block size");
  require(count<=std::numeric_limits<std::size_t>::max()/width,"URB1 count overflow");
  require(count<=max_output_bytes/width,"URB1 decoded size exceeds explicit limit");
  // Every nonempty block requires two lengths and a nonempty residual. Check
  // the physical lower bound before allocating any decoded output.
  const auto blocks=count/kBlock+(count%kBlock!=0);
  require(blocks<=(frame.size()-pos)/9,"URB1 count exceeds framed blocks");
  Bytes out;
  for(std::uint64_t begin=0;begin<count;begin+=kBlock) {
    const auto n=static_cast<std::size_t>(std::min<std::uint64_t>(kBlock,count-begin));
    const auto wl=get(frame,pos,4),pl=get(frame,pos,4);
    require(wl<=frame.size()-pos,"truncated URB1 width frame");
    const auto width_frame=frame.subspan(pos,static_cast<std::size_t>(wl));pos+=wl;
    require(unified_rle::symbol_width(width_frame)==1,"URB1 widths are uint8 symbols");
    const auto widths=unified_rle::decode(width_frame,kBlock);
    require(widths.size()==n,"URB1 width count mismatch");
    require(pl<=frame.size()-pos,"truncated URB1 payload");
    BitReader payload{frame.subspan(pos,static_cast<std::size_t>(pl))};pos+=pl;
    std::uint64_t previous=0;
    for(std::size_t i=0;i<n;++i) {
      require(widths[i]>=1 && widths[i]<=width*8,"invalid URB1 residual bit width");
      const auto delta=payload.value(widths[i]);
      require(std::max(1u,static_cast<unsigned>(std::bit_width(delta)))==widths[i],"noncanonical URB1 residual width");
      std::uint64_t current=delta;
      if(i && mode==1)current=(previous+delta)&mask(width);
      else if(i && mode==2) {
        const float p=std::bit_cast<float>(static_cast<std::uint32_t>(previous));
        const float d=std::bit_cast<float>(static_cast<std::uint32_t>(delta));
        current=std::bit_cast<std::uint32_t>(p+d);
      }
      put(out,current,width);previous=current;
    }
    payload.finish();
  }
  require(pos==frame.size(),"extra URB1 bytes");
  return out;
}
namespace {
constexpr std::array<std::uint8_t,4> kFloatRleMagic{'U','R','F','1'};
template<class Float,class Integer,class Bits>
Bytes scale_float_input(std::span<const std::uint8_t> raw,int requested) {
  const auto values=typed_values<Float>(Bytes(raw.begin(),raw.end()));
  int effective=requested;
  bool reversible=false;
  std::vector<Integer> scaled(values.size());
  for(;effective>=0;--effective) {
    const double power=std::pow(10.0,effective);
    bool fits=true;
    for(std::size_t i=0;i<values.size();++i) {
      const double product=static_cast<double>(values[i])*power;
      const double rounded=std::round(product);
      // The upper endpoint is exclusive: on macOS ARM long double is double,
      // and converting INT64_MAX to either rounds it up to 2^63.
      const double exclusive_max=std::ldexp(1.0,sizeof(Integer)*8-1);
      if(!std::isfinite(product) || rounded < -exclusive_max || rounded >= exclusive_max) {fits=false;break;}
      scaled[i]=static_cast<Integer>(rounded);
      if constexpr(sizeof(Integer)==8) {
        if(i) {
          const __int128 diff=static_cast<__int128>(scaled[i])-scaled[i-1];
          if(diff<std::numeric_limits<Integer>::min() || diff>std::numeric_limits<Integer>::max()) {fits=false;break;}
        }
      }
    }
    if(!fits) {
      if constexpr(sizeof(Integer)==8) continue;
      else break;
    }
    reversible=true;
    for(std::size_t i=0;i<values.size();++i) {
      const Float recovered=static_cast<Float>(static_cast<double>(scaled[i])/power);
      if(std::bit_cast<Bits>(recovered)!=std::bit_cast<Bits>(values[i])) {reversible=false;break;}
    }
    break;
  }
  Bytes out(kFloatRleMagic.begin(),kFloatRleMagic.end());
  out.push_back(sizeof(Float));out.push_back(reversible?1:0);
  out.push_back(static_cast<std::uint8_t>(requested));
  out.push_back(static_cast<std::uint8_t>(reversible?effective:0));
  const auto payload=reversible?raw_bytes<Integer>(scaled):Bytes(raw.begin(),raw.end());
  const auto encoded=unified_rle::encode(payload,sizeof(Float));
  out.insert(out.end(),encoded.begin(),encoded.end());return out;
}
template<class Float,class Integer>
Bytes unscale_float_input(Bytes raw,unsigned scale) {
  const auto values=typed_values<Integer>(std::move(raw));
  const double power=std::pow(10.0,scale);
  std::vector<Float> out;out.reserve(values.size());
  for(const auto value:values)out.push_back(static_cast<Float>(static_cast<double>(value)/power));
  return raw_bytes<Float>(out);
}
}
bool is_scaled_rle_frame(std::span<const std::uint8_t> frame) {
  return frame.size()>=4 && std::equal(kFloatRleMagic.begin(),kFloatRleMagic.end(),frame.begin());
}
unsigned scaled_rle_symbol_width(std::span<const std::uint8_t> frame) {
  require(is_scaled_rle_frame(frame)&&frame.size()>=8,"invalid URF1 header");
  require((frame[4]==4||frame[4]==8)&&frame[5]<=1&&frame[6]<=18&&frame[7]<=frame[6]&&
          (frame[5]!=0||frame[7]==0),"invalid URF1 width/mode/scale");
  return frame[4];
}
Bytes encode_rle_scaled(std::span<const std::uint8_t> raw,unsigned width,int requested_scale) {
  require(requested_scale>=0&&requested_scale<=18,"RLE max_point must be in [0,18]");
  require((width==4||width==8)&&raw.size()%width==0,"scaled RLE requires float32/64 raw input");
  return width==4?scale_float_input<float,std::int32_t,std::uint32_t>(raw,requested_scale)
                 :scale_float_input<double,std::int64_t,std::uint64_t>(raw,requested_scale);
}
Bytes decode_rle_scaled(std::span<const std::uint8_t> frame,std::size_t limit) {
  const auto width=scaled_rle_symbol_width(frame);
  require(unified_rle::symbol_width(frame.subspan(8))==width,"URF1 payload width mismatch");
  auto raw=unified_rle::decode(frame.subspan(8),limit);
  if(!frame[5])return raw;
  return width==4?unscale_float_input<float,std::int32_t>(std::move(raw),frame[7])
                 :unscale_float_input<double,std::int64_t>(std::move(raw),frame[7]);
}
}  // namespace unified_rle_numeric
