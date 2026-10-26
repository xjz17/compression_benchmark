#include "decimal_scaling_v2.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace native_baseline86_v1::decimal_scaling_v2 {
namespace {

using U128 = unsigned __int128;

constexpr std::array<std::uint8_t, 4> magic{'D', 'S', 'C', '3'};
constexpr std::uint8_t version = 1;
constexpr std::size_t header_bytes = 56;
constexpr std::uint64_t maximum_buffer_bytes = 256ULL << 20;

enum class Class : std::uint8_t {
  common_scaled = 0,
  finite_factored = 1,
  ieee_special = 2,
};

struct Type {
  std::uint8_t width;
};

Type type(std::string_view dtype) {
  if (dtype == "float32") return {4};
  if (dtype == "float64") return {8};
  throw std::invalid_argument("DSC2 decimal scaling requires float32 or float64");
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t checked_size(std::uint64_t value, const char* message) {
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::length_error(message);
  return static_cast<std::size_t>(value);
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char* message) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::length_error(message);
  return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right,
                          const char* message) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::length_error(message);
  return left * right;
}

std::uint64_t classmap_size(std::uint64_t count) {
  return checked_add(count, 3, "DSC2 class count overflow") / 4;
}

void put_u16(Bytes& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}

void put_u64(Bytes& out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint16_t get_u16(std::span<const std::uint8_t> input, std::size_t& pos) {
  require(input.size() - pos >= 2, "truncated DSC2 uint16");
  const auto value = static_cast<std::uint16_t>(input[pos]) |
      static_cast<std::uint16_t>(input[pos + 1]) << 8;
  pos += 2;
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t& pos) {
  require(input.size() - pos >= 8, "truncated DSC2 uint64");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8)
    value |= static_cast<std::uint64_t>(input[pos++]) << shift;
  return value;
}

template <class UInt>
UInt read_little(std::span<const std::uint8_t> input, std::size_t index) {
  const auto offset = index * sizeof(UInt);
  UInt value = 0;
  for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
    value |= static_cast<UInt>(input[offset + byte]) << (byte * 8);
  return value;
}

template <class UInt>
void append_little(Bytes& output, UInt value) {
  for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

struct Decimal {
  bool negative = false;
  std::uint64_t coefficient = 0;
  int exponent10 = 0;
  std::uint16_t required_places = 0;
};

template <class Float, class UInt, int FractionBits, int ExponentBits>
std::optional<Decimal> decimal_from_bits(UInt raw) {
  static_assert(std::numeric_limits<Float>::is_iec559);
  constexpr UInt exponent_mask = (UInt{1} << ExponentBits) - 1;
  constexpr UInt sign_mask = UInt{1} << (FractionBits + ExponentBits);
  const UInt encoded_exponent = (raw >> FractionBits) & exponent_mask;
  // NaN, infinity, and negative zero retain their exact IEEE bits through
  // algorithm-compressed uint32 words instead of decimal conversion.
  if (encoded_exponent == exponent_mask || raw == sign_mask)
    return std::nullopt;

  const Float value = std::bit_cast<Float>(raw);
  std::array<char, 128> buffer{};
  const auto rendered = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                      value, std::chars_format::general);
  require(rendered.ec == std::errc{}, "floating to_chars failed");
  const std::string_view text(
      buffer.data(), static_cast<std::size_t>(rendered.ptr - buffer.data()));
  std::size_t pos = 0;
  bool negative = false;
  if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
    negative = text[pos] == '-';
    ++pos;
  }
  const auto exponent_pos = text.find_first_of("eE", pos);
  const auto mantissa_end = exponent_pos == std::string_view::npos
      ? text.size() : exponent_pos;
  int explicit_exponent = 0;
  if (exponent_pos != std::string_view::npos) {
    const char* begin = text.data() + exponent_pos + 1;
    if (begin != text.data() + text.size() && *begin == '+') ++begin;
    const auto parsed = std::from_chars(begin, text.data() + text.size(),
                                        explicit_exponent);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
            "invalid exponent emitted by floating to_chars");
  }

  std::uint64_t coefficient = 0;
  bool after_point = false;
  bool saw_digit = false;
  int digits_after_point = 0;
  for (; pos < mantissa_end; ++pos) {
    const char ch = text[pos];
    if (ch == '.') {
      require(!after_point, "multiple points emitted by floating to_chars");
      after_point = true;
      continue;
    }
    require(ch >= '0' && ch <= '9',
            "nondigit emitted by floating to_chars");
    saw_digit = true;
    const auto digit = static_cast<unsigned>(ch - '0');
    require(coefficient <=
                (std::numeric_limits<std::uint64_t>::max() - digit) / 10,
            "shortest decimal coefficient exceeds uint64");
    coefficient = coefficient * 10 + digit;
    if (after_point) ++digits_after_point;
  }
  require(saw_digit, "floating to_chars emitted no mantissa digit");
  const int exponent10 = explicit_exponent - digits_after_point;
  const int places = std::max(0, -exponent10);
  require(places <= maximum_decimal_places,
          "shortest binary32/64 decimal requires more than 324 places");
  // Binary32/64 shortest round-trip decimal strings use at most max_digits10
  // significant digits (9/17), so their coefficient always fits int64.
  require(coefficient <=
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
          "shortest decimal coefficient exceeds int64");
  return Decimal{negative, coefficient, exponent10,
                 static_cast<std::uint16_t>(places)};
}

template <class UInt>
std::optional<Decimal> decimal(UInt raw);

template <>
std::optional<Decimal> decimal(std::uint32_t raw) {
  return decimal_from_bits<float, std::uint32_t, 23, 8>(raw);
}

template <>
std::optional<Decimal> decimal(std::uint64_t raw) {
  return decimal_from_bits<double, std::uint64_t, 52, 11>(raw);
}

std::int64_t signed_coefficient(const Decimal& value) {
  const auto magnitude = static_cast<std::int64_t>(value.coefficient);
  return value.negative ? -magnitude : magnitude;
}

std::optional<std::int64_t> integer_at(const Decimal& value,
                                       std::uint16_t places) {
  if (places < value.required_places) return std::nullopt;
  if (value.coefficient == 0) return std::int64_t{0};
  const int zeros = value.exponent10 + static_cast<int>(places);
  require(zeros >= 0, "DSC2 decimal places do not yield an integer");
  const U128 limit = value.negative ? (U128{1} << 63)
                                    : static_cast<U128>(
                                          std::numeric_limits<std::int64_t>::max());
  // Nonzero coefficient * 10^19 already exceeds either signed limit.
  // A bounded integer multiply replaces the identical per-decimal-zero loop.
  static constexpr std::array<std::uint64_t, 19> powers10{
      1ULL,10ULL,100ULL,1000ULL,10000ULL,100000ULL,1000000ULL,
      10000000ULL,100000000ULL,1000000000ULL,10000000000ULL,
      100000000000ULL,1000000000000ULL,10000000000000ULL,
      100000000000000ULL,1000000000000000ULL,10000000000000000ULL,
      100000000000000000ULL,1000000000000000000ULL};
  if (zeros >= static_cast<int>(powers10.size())) return std::nullopt;
  const U128 magnitude = static_cast<U128>(value.coefficient) * powers10[zeros];
  if (magnitude > limit) return std::nullopt;
  if (!value.negative) return static_cast<std::int64_t>(magnitude);
  if (magnitude == (U128{1} << 63))
    return std::numeric_limits<std::int64_t>::min();
  return -static_cast<std::int64_t>(magnitude);
}

template <class Float>
std::optional<Float> decode_decimal(std::int64_t coefficient, int exponent10) {
  std::array<char, 128> buffer{};
  auto out = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                           coefficient);
  if (out.ec != std::errc{} || out.ptr == buffer.data() + buffer.size())
    return std::nullopt;
  *out.ptr++ = 'e';
  const auto exponent = std::to_chars(out.ptr, buffer.data() + buffer.size(),
                                      exponent10);
  if (exponent.ec != std::errc{}) return std::nullopt;
  out.ptr = exponent.ptr;
  Float decoded{};
  const auto parsed = std::from_chars(buffer.data(), out.ptr, decoded,
                                      std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != out.ptr) return std::nullopt;
  return decoded;
}

template <class UInt, class Float>
bool common_scaled(UInt raw, const Decimal& value, std::uint16_t places,
                   std::int64_t* encoded) {
  const auto integer = integer_at(value, places);
  if (!integer) return false;
  const auto restored = decode_decimal<Float>(*integer,
                                               -static_cast<int>(places));
  if (!restored || std::bit_cast<UInt>(*restored) != raw) return false;
  if (encoded) *encoded = *integer;
  return true;
}

// DSC3 preserves the frozen shortest-decimal precision definition. Its wire
// stores independent coefficient/exponent planes and compact recoverable
// metadata. Every plane is sent through the selected integer codec.

void put_varint(Bytes& out, std::uint64_t v) {
  do {auto b=static_cast<std::uint8_t>(v&127);v>>=7;out.push_back(b|(v?128:0));} while(v);
}
std::uint64_t get_varint(std::span<const std::uint8_t> data,std::size_t& p) {
  std::uint64_t v=0;
  for(unsigned n=0;n<10;++n){
    require(p<data.size(),"truncated DSC3 varint");auto b=data[p++];
    if(n==9)require((b&254)==0,"DSC3 varint overflow");
    v|=static_cast<std::uint64_t>(b&127)<<(7*n);
    if(!(b&128)){require(n==0||b!=0,"noncanonical DSC3 varint");return v;}
  }
  throw std::runtime_error("DSC3 varint overflow");
}
void set_class3(Bytes& out,std::size_t i,unsigned cls){out[i/4]|=static_cast<std::uint8_t>(cls<<(2*(i%4)));}
unsigned map_class(std::span<const std::uint8_t> map,std::size_t i){return (map[i/4]>>(2*(i%4)))&3;}
struct Header3 {
  std::uint8_t width=0,mode=0;
  std::uint16_t p=0;
  std::uint64_t n=0,common=0,factored=0,special=0,words=0;
  std::size_t offset=0;
};
unsigned class_at(std::span<const std::uint8_t> map,const Header3& h,std::size_t i){return h.mode==3?map_class(map,i):h.mode;}
Header3 parse3(std::span<const std::uint8_t> meta){
  require(meta.size()>=7&&std::equal(magic.begin(),magic.end(),meta.begin()),"invalid DSC3 header");
  std::size_t pos=4;auto flag=meta[pos++];require((flag&248)==0,"DSC3 reserved flags");
  Header3 h;h.width=(flag&1)?8:4;h.mode=(flag>>1)&3;
  auto p=get_varint(meta,pos);require(p<=maximum_decimal_places,"DSC3 precision exceeds 324");h.p=static_cast<std::uint16_t>(p);
  h.n=get_varint(meta,pos);h.offset=pos;
  require(h.n<=maximum_buffer_bytes/16,"DSC3 count exceeds word bound");
  const auto mapsize=h.mode==3?(h.n+3)/4:0;
  require(mapsize<=meta.size()-pos&&pos+mapsize==meta.size(),"DSC3 metadata length mismatch");
  auto map=meta.subspan(pos);
  if(h.mode==3&&h.n%4)require((map.back()>>(2*(h.n%4)))==0,"DSC3 nonzero classmap padding");
  if(h.mode==0)h.common=h.n;
  else if(h.mode==1)h.factored=h.n;
  else if(h.mode==2)h.special=h.n;
  else for(std::size_t i=0;i<h.n;++i){
    auto cls=map_class(map,i);require(cls!=3,"DSC3 reserved class");
    if(cls==0)++h.common;else if(cls==1)++h.factored;else ++h.special;
  }
  h.words=h.common+2*h.factored+(h.width/4)*h.special;
  require(h.words<=maximum_buffer_bytes/8,"DSC3 integer plane overflow");
  return h;
}
template<class UInt,class Float>
Scaled forward3(std::span<const std::uint8_t> input,std::uint16_t p){
  const std::size_t n=input.size()/sizeof(UInt);
  Bytes classes((n+3)/4,0);
  std::vector<std::int64_t> common,coeff,exponents,special;
  common.reserve(n);coeff.reserve(n/16);exponents.reserve(n/16);
  UInt previous_raw=0;bool previous_valid=false;
  std::optional<Decimal> previous_decimal;
  std::optional<std::int64_t> previous_integer;
  for(std::size_t i=0;i<n;++i){
    const UInt raw=read_little<UInt>(input,i);
    if(!previous_valid || raw!=previous_raw){
      previous_decimal=decimal(raw);
      previous_integer=previous_decimal?integer_at(*previous_decimal,p):std::nullopt;
      previous_raw=raw;previous_valid=true;
    }
    const auto& parsed=previous_decimal;
    if(parsed){
      require(parsed->required_places<=p,"DSC3 precision below input requirement");
      // to_chars(general) is a shortest round-trip representation. Appending
      // decimal zeros changes neither its exact rational value nor its rounded
      // binary value, so an additional from_chars trial is redundant here.
      if(previous_integer){common.push_back(*previous_integer);continue;}
      set_class3(classes,i,1);coeff.push_back(signed_coefficient(*parsed));
      exponents.push_back(static_cast<std::int64_t>(p)+parsed->exponent10);
    }else{
      set_class3(classes,i,2);special.push_back(static_cast<std::uint32_t>(raw));
      if constexpr(sizeof(UInt)==8)special.push_back(static_cast<std::uint32_t>(raw>>32));
    }
  }
  const std::size_t specials=special.size()/(sizeof(UInt)/4);
  unsigned mode=3;
  if(common.size()==n)mode=0;else if(coeff.size()==n)mode=1;else if(specials==n)mode=2;
  Bytes metadata;metadata.reserve(16+(mode==3?classes.size():0));
  metadata.insert(metadata.end(),magic.begin(),magic.end());
  metadata.push_back(static_cast<std::uint8_t>((sizeof(UInt)==8?1:0)|(mode<<1)));
  put_varint(metadata,p);put_varint(metadata,n);
  if(mode==3)metadata.insert(metadata.end(),classes.begin(),classes.end());
  std::vector<std::int64_t> words;words.reserve(common.size()+2*coeff.size()+special.size());
  words.insert(words.end(),common.begin(),common.end());
  words.insert(words.end(),coeff.begin(),coeff.end());
  words.insert(words.end(),exponents.begin(),exponents.end());
  words.insert(words.end(),special.begin(),special.end());
  return {std::move(words),std::move(metadata),p,common.size(),coeff.size(),specials};
}
template<class UInt,class Float>
Bytes inverse3(std::span<const std::int64_t> words,std::span<const std::uint8_t> meta,const Header3& h){
  require(words.size()==h.words,"DSC3 integer word count mismatch");
  const auto map=meta.subspan(h.offset);Bytes output;output.reserve(h.n*sizeof(UInt));
  std::size_t ci=0,fi=h.common,ei=h.common+h.factored,si=h.common+2*h.factored;
  for(std::size_t i=0;i<h.n;++i){
    const auto cls=class_at(map,h,i);UInt raw=0;
    if(cls<=1){
      std::int64_t c=0;int exponent=0;
      if(cls==0){c=words[ci++];exponent=-static_cast<int>(h.p);}
      else {
        c=words[fi++];const auto k=words[ei++];
        require(k>=0&&k<=maximum_decimal_places+400,"DSC3 exponent out of range");
        // A factored value is legal only when its common-scale integer does
        // not fit int64. Check this directly in the integer domain rather
        // than rendering and reparsing the recovered floating value.
        U128 magnitude=c<0?static_cast<std::uint64_t>(-(c+1))+U128{1}:static_cast<U128>(c);
        const U128 limit=c<0?(U128{1}<<63):static_cast<U128>(std::numeric_limits<std::int64_t>::max());
        bool overflow=false;
        if(magnitude!=0)for(std::int64_t j=0;j<k;++j){if(magnitude>limit/10){overflow=true;break;}magnitude*=10;}
        require(overflow,"DSC3 factored value fits common class");
        exponent=static_cast<int>(k)-h.p;
      }
      const auto decoded=decode_decimal<Float>(c,exponent);
      require(decoded.has_value(),"DSC3 coefficient cannot be decoded");
      raw=std::bit_cast<UInt>(*decoded);
      // Decimal payload cannot encode IEEE specials; these have exact raw
      // planes. We do not repeat the encoder's decimal rendering in decode.
      constexpr unsigned mant=sizeof(UInt)==4?23:52,expbits=sizeof(UInt)==4?8:11;
      constexpr UInt em=(UInt{1}<<expbits)-1;
      require(((raw>>mant)&em)!=em,"DSC3 decimal produced nonfinite value");
    }else{
      const auto lo=words[si++];require(lo>=0&&static_cast<std::uint64_t>(lo)<=UINT32_MAX,"DSC3 raw low word range");
      raw=static_cast<UInt>(lo);
      if constexpr(sizeof(UInt)==8){const auto hi=words[si++];require(hi>=0&&static_cast<std::uint64_t>(hi)<=UINT32_MAX,"DSC3 raw high word range");raw|=static_cast<UInt>(hi)<<32;}
      constexpr unsigned mant=sizeof(UInt)==4?23:52,expbits=sizeof(UInt)==4?8:11;
      constexpr UInt em=(UInt{1}<<expbits)-1,negative_zero=UInt{1}<<(mant+expbits);
      require(((raw>>mant)&em)==em||raw==negative_zero,"DSC3 raw class is not special");
    }
    append_little(output,raw);
  }
  require(ci==h.common&&fi==h.common+h.factored&&ei==h.common+2*h.factored&&si==words.size(),"DSC3 plane counts mismatch");
  return output;
}
bool old_metadata(std::span<const std::uint8_t> bytes){return bytes.size()>=4&&bytes[0]=='D'&&bytes[1]=='S'&&bytes[2]=='C'&&bytes[3]=='2';}
} // namespace

namespace legacy {
MetadataInfo inspect(std::span<const std::uint8_t>,std::string_view);
Bytes inverse(std::span<const std::int64_t>,std::span<const std::uint8_t>,std::string_view);
}
std::uint16_t required_decimal_places(std::span<const std::uint8_t> input,std::string_view dtype){return decimal_scaling::required_decimal_places(input,dtype);}
Scaled forward(std::span<const std::uint8_t> input,std::string_view dtype,std::uint16_t p){
  auto t=type(dtype);require(input.size()%t.width==0,"DSC3 input width mismatch");
  require(input.size()/t.width<=maximum_buffer_bytes/16,"DSC3 word size bound");
  require(p<=maximum_decimal_places,"DSC3 precision exceeds 324");
  return t.width==4?forward3<std::uint32_t,float>(input,p):forward3<std::uint64_t,double>(input,p);
}
MetadataInfo inspect(std::span<const std::uint8_t> metadata,std::string_view dtype){
  if(old_metadata(metadata))return legacy::inspect(metadata,dtype);
  const auto h=parse3(metadata);require(h.width==type(dtype).width,"DSC3 dtype mismatch");
  return {h.n,h.words,h.common,h.factored,h.special,h.p};
}
Bytes inverse(std::span<const std::int64_t> words,std::span<const std::uint8_t> metadata,std::string_view dtype){
  if(old_metadata(metadata))return legacy::inverse(words,metadata,dtype);
  const auto h=parse3(metadata);require(h.width==type(dtype).width,"DSC3 dtype mismatch");
  return h.width==4?inverse3<std::uint32_t,float>(words,metadata,h):inverse3<std::uint64_t,double>(words,metadata,h);
}
} // namespace native_baseline86_v1::decimal_scaling_v2
