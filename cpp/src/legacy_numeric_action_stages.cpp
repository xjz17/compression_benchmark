#include "legacy_numeric_action_stages.hpp"
#include "operator_layout_primitives.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

// No fast-math: Python explicitly rounds every double operation. In particular
// ALP's magic addition/subtraction must not be reassociated or contracted.
#pragma STDC FP_CONTRACT OFF

namespace legacy_numeric_action {
namespace {
using J = nlohmann::json;
constexpr const char* kScale = "op.scaling.legacy_exact_decimal.python_v1";
constexpr const char* kAlpScale = "op.scaling.alp_legacy_fe_exceptions.python_v1";
constexpr const char* kLimbs = "op.block_transform.int64_lo_hi32.python_v1";
constexpr const char* kMatch = "op.lz_match.native_lz_limb_batch.aligned8_greedy.python_v1";
constexpr const char* kPack = "op.match_token_format.native_lz_limb_batch.ctrl8.python_v1";
constexpr const char* kFrame = "op.container_frame.legacy_numeric_limbs.python_v1";
constexpr const char* kAlpFrame = "op.container_frame.alp2_legacy.python_v1";
constexpr std::uint32_t kWindow = 32768, kMinimum = 5, kMaximum = 175;
constexpr double kHigh = 9223372036854774784.0, kMagic = 6755399441055744.0;
constexpr double kFraction[] = {1.,.1,.01,.001,.0001,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10};
constexpr double kExponent[] = {1.,10.,100.,1000.,10000.,100000.,1000000.,10000000.,100000000.,1000000000.,10000000000.};
constexpr std::int64_t kFactor[] = {1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000};
void require(bool ok, const char* text) { if (!ok) throw std::invalid_argument(text); }
void put(Bytes& out, std::uint64_t n, unsigned width) {
  for (unsigned i = 0; i < width; ++i) out.push_back(static_cast<std::uint8_t>(n>>(8*i)));
}
std::uint64_t get(const Bytes& raw, std::size_t& pos, unsigned width) {
  require(pos <= raw.size() && width <= raw.size()-pos,"truncated legacy numeric data");
  std::uint64_t n = 0;
  for (unsigned i = 0; i < width; ++i) n |= std::uint64_t(raw[pos++])<<(8*i);
  return n;
}
std::uint64_t bits(double x) { std::uint64_t u; std::memcpy(&u,&x,8); return u; }
double real(std::uint64_t u) { double x; std::memcpy(&x,&u,8); return x; }
std::int64_t signed_word(std::uint64_t u) { std::int64_t x; std::memcpy(&x,&u,8); return x; }
std::vector<double> doubles(const Bytes& raw) {
  require(raw.size()%8 == 0,"numeric input must contain complete 64-bit values");
  std::vector<double> out;
  for (std::size_t pos = 0; pos < raw.size();) out.push_back(real(get(raw,pos,8)));
  return out;
}
std::string hex(const Bytes& raw) {
  constexpr char digits[] = "0123456789abcdef";
  std::string out; out.reserve(raw.size()*2);
  for (auto b : raw) { out.push_back(digits[b>>4]); out.push_back(digits[b&15]); }
  return out;
}
Bytes unhex(const std::string& text) {
  require(text.size()%2 == 0,"odd-length legacy hex body");
  auto nibble = [](char c) -> unsigned {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'a' && c <= 'f') return c-'a'+10;
    if (c >= 'A' && c <= 'F') return c-'A'+10;
    throw std::invalid_argument("invalid legacy hex digit");
  };
  Bytes out;
  for (std::size_t p = 0; p < text.size(); p += 2) out.push_back((nibble(text[p])<<4)|nibble(text[p+1]));
  return out;
}
Bytes packet(const J& obj) {
  // std::map-backed nlohmann::json sorts object keys, matching sort_keys=True.
  // All packet numbers are integers; uint64 exception bits remain unsigned.
  const std::string text = obj.dump(-1,' ',true);
  Bytes out{'L','N','S','1'}; out.insert(out.end(),text.begin(),text.end()); return out;
}
J read(const Bytes& raw, const char* stage) {
  require(raw.size() >= 4 && raw[0]=='L' && raw[1]=='N' && raw[2]=='S' && raw[3]=='1',"expected LNS1 packet");
  J obj = J::parse(raw.begin()+4,raw.end());
  require(obj.at("stage") == stage,"legacy numeric stage mismatch"); return obj;
}
J pop(J& obj, const char* key) { auto value = obj.at(key); obj.erase(key); return value; }
std::int64_t i64(const J& value) {
  if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= std::uint64_t(INT64_MAX),"legacy signed code overflow");
  require(value.is_number_integer(),"legacy code is not integer"); return value.get<std::int64_t>();
}
bool rounded_code(double value, std::int64_t& out) {
  if (!std::isfinite(value)) throw std::invalid_argument("cannot round nonfinite legacy scaling value");
  // Python round(x), without ndigits, uses round-to-nearest/ties-to-even.
  // nearbyint under a caller-modified rounding mode would be incorrect.
  double rounded = std::floor(value), fraction = value-rounded;
  if (fraction > .5 || (fraction == .5 && std::fmod(rounded,2.) != 0.)) rounded += 1.;
  if (rounded < -9223372036854775808.0 || rounded >= 9223372036854775808.0) return false;
  out = static_cast<std::int64_t>(rounded); return true;
}
bool codes_at(const std::vector<double>& values, int scale, std::vector<std::int64_t>& codes) {
  const double power = std::pow(10.,scale); codes.clear();
  for (double value : values) {
    std::int64_t code;
    if (!rounded_code(value*power,code)) return false;
    codes.push_back(code);
  }
  return true;
}
Bytes scaled_values(const std::vector<std::int64_t>& codes, int scale) {
  const double power = std::pow(10.,scale); Bytes out;
  for (auto code : codes) put(out,bits(static_cast<double>(code)/power),8);
  return out;
}
J scale(const Bytes& raw, const std::string& kind, std::optional<int> max_point) {
  require(raw.size()%8 == 0,"numeric input must contain complete 64-bit values");
  std::vector<std::int64_t> codes;
  if (kind == "int64") {
    for (std::size_t pos = 0; pos < raw.size();) codes.push_back(signed_word(get(raw,pos,8)));
    return J{{"format","scaled"},{"kind",kind},{"scale",0},{"stage","codes"},{"blocks",J::array({J{{"codes",codes}}})}};
  }
  require(kind == "float64" && max_point.has_value(),"legacy scaling requires float64 and max_point");
  const auto values = doubles(raw);
  require(!values.empty(),"legacy codec cannot scale empty floating column");
  // The frozen helper first chooses the largest bounded scale that fits int64,
  // even if it is not reversible. Exactness is enforced by the outer stage.
  bool found = false;
  for (int s = std::max(0,*max_point); s >= 0; --s) {
    if (codes_at(values,s,codes)) { found = true; break; }
  }
  if (!found) {
    int maximum = 0;
    for (double value : values) {
      require(std::isfinite(value),"legacy decimal scaling rejects nonfinite values");
      if (value == 0.) continue;
      bool exact = false;
      for (int s = 0; s <= 15; ++s) {
        std::int64_t code; const double power = std::pow(10.,s);
        if (!rounded_code(value*power,code)) break;
        if (bits(static_cast<double>(code)/power) == bits(value)) { maximum = std::max(maximum,s); exact = true; break; }
      }
      require(exact,"legacy codec has no exact decimal scale");
    }
    require(codes_at(values,maximum,codes),"legacy scaled code overflow");
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (values[i] == 0.) codes[i] = signed_word(bits(values[i]));
      else require(bits(static_cast<double>(codes[i])/std::pow(10.,maximum)) == bits(values[i]),"legacy inexact decimal code");
    }
  }
  std::vector<std::int64_t> other;
  for (int s = std::max(18,*max_point); s >= 0; --s) {
    if (!codes_at(values,s,other) || other != codes) continue;
    if (scaled_values(codes,s) == raw)
      return J{{"format","scaled"},{"kind",kind},{"scale",s},{"stage","codes"},{"blocks",J::array({J{{"codes",codes}}})}};
  }
  throw std::invalid_argument("legacy codec has no bit-exact decimal inverse");
}
bool impossible(double x) { return !std::isfinite(x) || x > kHigh || x < -kHigh; }
__int128 alp_encode(double value, int fac, int exp) {
  double tmp = value*kExponent[exp]; tmp = tmp*kFraction[fac];
  if (impossible(tmp)) return INT64_MAX;
  // Volatile intermediates also preserve Python's operation boundaries under
  // ordinary optimization and protect the rounding trick from reassociation.
  volatile double added = tmp+kMagic;
  volatile double rounded = added-kMagic;
  return static_cast<__int128>(rounded);
}
double alp_decode(__int128 code, int fac, int exp) {
  require(fac >= 0 && fac <= 9 && exp >= 0 && exp <= 10,"ALP factor/exponent outside table");
  // Python does exact integer multiplication first, then converts to double.
  const __int128 product = code*kFactor[fac];
  return static_cast<double>(product)*kFraction[exp];
}
std::optional<std::pair<int,int>> best_fe(const std::vector<double>& values) {
  const auto n = values.size();
  if (n == 1) {
    if (impossible(values[0])) return std::nullopt;
    for (int exp = 10; exp >= 0; --exp)
      for (int fac = std::min(exp,9); fac >= 0; --fac)
        if (bits(alp_decode(alp_encode(values[0],fac,exp),fac,exp)) == bits(values[0])) return {{fac,exp}};
    return std::nullopt;
  }
  std::uint64_t best_cost = std::uint64_t(1)<<62;
  std::optional<std::pair<int,int>> chosen;
  for (int exp = 10; exp >= 0; --exp) {
    for (int fac = std::min(exp,9); fac >= 0; --fac) {
      std::size_t nonexc = 0; std::uint64_t minimum = UINT64_MAX, maximum = 0; bool bad = false;
      for (double value : values) {
        if (impossible(value)) { bad = true; break; }
        const auto code = alp_encode(value,fac,exp);
        if (bits(alp_decode(code,fac,exp)) == bits(value)) {
          ++nonexc; const auto u = static_cast<std::uint64_t>(code);
          minimum = std::min(minimum,u); maximum = std::max(maximum,u);
        }
      }
      if (bad || nonexc < 2) continue;
      unsigned width = 0;
      if (maximum > minimum) for (auto delta = maximum-minimum; delta; delta >>= 1) ++width;
      const std::uint64_t cost = n*width+(n-nonexc)*80;
      if (cost < best_cost) { best_cost = cost; chosen = {{fac,exp}}; }
    }
  }
  return chosen;
}
J alp_scale(const Bytes& raw) {
  const auto values = doubles(raw); J blocks = J::array();
  for (std::size_t start = 0; start < values.size(); start += 1024) {
    const auto end = std::min(start+1024,values.size());
    const std::vector<double> block_values(values.begin()+start,values.begin()+end);
    const auto choice = best_fe(block_values);
    auto raw_block = [&]() { blocks.push_back(J{{"raw",hex(Bytes(raw.begin()+start*8,raw.begin()+end*8))},{"count",end-start}}); };
    if (!choice) { raw_block(); continue; }
    const auto [fac,exp] = *choice;
    std::vector<std::int64_t> codes; J exceptions = J::array(); std::optional<std::int64_t> placeholder;
    for (std::size_t i = 0; i < block_values.size(); ++i) {
      const double value = block_values[i]; const auto code = alp_encode(value,fac,exp);
      if (bits(alp_decode(code,fac,exp)) == bits(value)) {
        require(code >= INT64_MIN && code <= INT64_MAX,"ALP exact code outside signed int64");
        codes.push_back(static_cast<std::int64_t>(code));
        if (!placeholder) placeholder = codes.back();
      } else { codes.push_back(0); exceptions.push_back(J::array({i,bits(value)})); }
    }
    if (!placeholder) { raw_block(); continue; }
    for (const auto& exception : exceptions) codes.at(exception.at(0).get<std::size_t>()) = *placeholder;
    blocks.push_back(J{{"count",end-start},{"fac",fac},{"exp",exp},{"exceptions",exceptions},{"codes",codes}});
  }
  return J{{"format","alp"},{"stage","codes"},{"count",values.size()},{"blocks",blocks}};
}
J tokens_json(const std::vector<native_lz_action::Token>& tokens) {
  J out = J::array();
  for (const auto& t : tokens) out.push_back(J::array({t.literal,t.offset,t.length}));
  return out;
}
std::vector<native_lz_action::Token> tokens_cpp(const J& array) {
  std::vector<native_lz_action::Token> tokens;
  for (const auto& t : array) {
    require(t.is_array() && t.size() == 3,"invalid legacy token triple");
    tokens.push_back({t.at(0).get<int>(),t.at(1).get<std::uint32_t>(),t.at(2).get<std::uint32_t>()});
  }
  return tokens;
}
Bytes limb_body(const J& block) {
  const auto lo = unhex(block.at("wires").at(0).get<std::string>()), hi = unhex(block.at("wires").at(1).get<std::string>());
  require(lo.size() <= UINT32_MAX,"legacy low limb too large");
  Bytes body; put(body,lo.size(),4); body.insert(body.end(),lo.begin(),lo.end()); body.insert(body.end(),hi.begin(),hi.end()); return body;
}
J limb_wires(const Bytes& body) {
  std::size_t pos = 0; const auto size = get(body,pos,4);
  require(size <= body.size()-pos,"truncated low limb");
  return J::array({hex(Bytes(body.begin()+pos,body.begin()+pos+size)),hex(Bytes(body.begin()+pos+size,body.end()))});
}
Bytes slice(const Bytes& raw, std::size_t& pos, std::uint64_t length) {
  require(pos <= raw.size() && length <= raw.size()-pos,"truncated legacy frame body");
  Bytes out(raw.begin()+pos,raw.begin()+pos+length); pos += length; return out;
}
}  // namespace

Bytes forward(const std::string& action, const Bytes& raw, const std::string& kind, std::optional<int> max_point) {
  if (action == kScale) return packet(scale(raw,kind,max_point));
  if (action == kAlpScale) { require(kind == "float64","ALP scaling requires float64"); return packet(alp_scale(raw)); }
  J obj;
  if (action == kLimbs) {
    const exact_actions::operator_layout::BitFields layout{32,32};
    obj = read(raw,"codes");
    for (auto& block : obj.at("blocks")) {
      if (block.contains("raw")) continue;
      const auto codes = pop(block,"codes"); block["count"] = codes.size(); J limbs = J::array();
      for (unsigned field : {0,1}) {
        Bytes body;
        for (const auto& code : codes) put(body,layout.field(static_cast<std::uint64_t>(i64(code)),field),4);
        limbs.push_back(hex(body));
      }
      block["limbs"] = limbs;
    }
    obj["stage"] = "limbs";
  } else if (action == kMatch) {
    obj = read(raw,"limbs");
    for (auto& block : obj.at("blocks")) {
      if (block.contains("raw")) continue;
      const auto limbs = pop(block,"limbs"); J token_sets = J::array();
      for (const auto& limb : limbs) token_sets.push_back(tokens_json(native_lz_action::parse(unhex(limb.get<std::string>()),kWindow,kMinimum,kMaximum)));
      block["tokens"] = token_sets;
    }
    obj["stage"] = "tokens";
  } else if (action == kPack) {
    obj = read(raw,"tokens");
    for (auto& block : obj.at("blocks")) {
      if (block.contains("raw")) continue;
      const auto token_sets = pop(block,"tokens"); J wires = J::array();
      for (const auto& tokens : token_sets) wires.push_back(hex(native_lz_action::pack(tokens_cpp(tokens),kMinimum)));
      block["wires"] = wires;
    }
    obj["stage"] = "wires";
  } else if (action == kFrame || action == kAlpFrame) {
    obj = read(raw,"wires"); Bytes out;
    if (action == kFrame) {
      require(obj.at("format") == "scaled" && obj.at("blocks").size() == 1,"scaled frame needs one scaled block");
      if (obj.at("kind") == "int64") out.push_back('I');
      else { out.push_back('F'); put(out,static_cast<std::uint32_t>(obj.at("scale").get<std::int32_t>()),4); }
      const auto body = limb_body(obj.at("blocks").at(0)); out.insert(out.end(),body.begin(),body.end()); return out;
    }
    require(obj.at("format") == "alp","ALP frame needs ALP blocks");
    out = {'A','L','P','2'}; put(out,obj.at("count").get<std::uint32_t>(),4);
    for (const auto& block : obj.at("blocks")) {
      const auto count = block.at("count").get<std::uint16_t>();
      if (block.contains("raw")) {
        const auto body = unhex(block.at("raw").get<std::string>());
        put(out,0,1); put(out,count,2); put(out,body.size(),4); out.insert(out.end(),body.begin(),body.end());
      } else {
        const auto body = limb_body(block); const auto& exceptions = block.at("exceptions");
        put(out,1,1); put(out,count,2); put(out,block.at("fac").get<int>(),1); put(out,block.at("exp").get<int>(),1);
        put(out,body.size(),4); put(out,exceptions.size(),2); out.insert(out.end(),body.begin(),body.end());
        for (const auto& exception : exceptions) { put(out,exception.at(0).get<std::uint16_t>(),2); put(out,exception.at(1).get<std::uint64_t>(),8); }
      }
    }
    return out;
  } else throw std::invalid_argument("unknown legacy numeric action");
  return packet(obj);
}

Bytes inverse(const std::string& action, const Bytes& raw, std::uint64_t count) {
  if (action == kScale || action == kAlpScale) {
    const auto obj = read(raw,"codes"); Bytes out;
    if (action == kScale) {
      std::vector<std::int64_t> codes;
      for (const auto& code : obj.at("blocks").at(0).at("codes")) codes.push_back(i64(code));
      if (obj.at("kind") != "int64") return scaled_values(codes,obj.at("scale").get<int>());
      for (auto code : codes) put(out,static_cast<std::uint64_t>(code),8);
      return out;
    }
    for (const auto& block : obj.at("blocks")) {
      if (block.contains("raw")) {
        const auto body = unhex(block.at("raw").get<std::string>()); out.insert(out.end(),body.begin(),body.end());
      } else {
        Bytes restored;
        for (const auto& code : block.at("codes")) put(restored,bits(alp_decode(i64(code),block.at("fac").get<int>(),block.at("exp").get<int>())),8);
        require(restored.size() == block.at("count").get<std::uint64_t>()*8,"ALP code count mismatch");
        for (const auto& exception : block.at("exceptions")) {
          const auto index = exception.at(0).get<std::uint64_t>(), value = exception.at(1).get<std::uint64_t>();
          require(index < restored.size()/8,"ALP exception index out of range");
          for (unsigned i = 0; i < 8; ++i) restored[index*8+i] = value>>(8*i);
        }
        out.insert(out.end(),restored.begin(),restored.end());
      }
    }
    return out;
  }
  J obj;
  if (action == kLimbs) {
    const exact_actions::operator_layout::BitFields layout{32,32};
    obj = read(raw,"limbs");
    for (auto& block : obj.at("blocks")) {
      if (block.contains("raw")) continue;
      const auto n = block.at("count").get<std::uint64_t>(); const auto limbs = pop(block,"limbs");
      const auto lo = unhex(limbs.at(0).get<std::string>()), hi = unhex(limbs.at(1).get<std::string>());
      require(lo.size() == n*4 && hi.size() == n*4,"legacy limb size mismatch");
      std::vector<std::int64_t> codes;
      for (std::size_t lpos = 0, hpos = 0; lpos < lo.size();) {
        const std::uint64_t fields[]{get(lo,lpos,4),get(hi,hpos,4)};
        codes.push_back(signed_word(layout.join(fields)));
      }
      block["codes"] = codes; if (obj.at("format") == "scaled") block.erase("count");
    }
    obj["stage"] = "codes";
  } else if (action == kMatch) {
    obj = read(raw,"tokens");
    for (auto& block : obj.at("blocks")) {
      if (block.contains("raw")) continue;
      const auto token_sets = pop(block,"tokens"); J limbs = J::array();
      for (const auto& tokens : token_sets) limbs.push_back(hex(native_lz_action::expand(tokens_cpp(tokens),block.at("count").get<std::uint64_t>()*4)));
      block["limbs"] = limbs;
    }
    obj["stage"] = "limbs";
  } else if (action == kPack) {
    obj = read(raw,"wires");
    for (auto& block : obj.at("blocks")) {
      if (block.contains("raw")) continue;
      const auto wires = pop(block,"wires"); J token_sets = J::array();
      for (const auto& wire : wires) token_sets.push_back(tokens_json(native_lz_action::unpack(unhex(wire.get<std::string>()),kMinimum,block.at("count").get<std::uint64_t>()*4)));
      block["tokens"] = token_sets;
    }
    obj["stage"] = "tokens";
  } else if (action == kFrame) {
    require(!raw.empty(),"invalid scaled numeric frame"); std::size_t pos = 1; std::string kind; std::int32_t scale_value = 0;
    if (raw[0] == 'I') kind = "int64";
    else if (raw[0] == 'F') { kind = "float64"; const auto u = static_cast<std::uint32_t>(get(raw,pos,4)); std::memcpy(&scale_value,&u,4); }
    else throw std::invalid_argument("invalid scaled numeric frame");
    obj = J{{"format","scaled"},{"kind",kind},{"scale",scale_value},{"stage","wires"},
            {"blocks",J::array({J{{"count",count},{"wires",limb_wires(Bytes(raw.begin()+pos,raw.end()))}}})}};
  } else if (action == kAlpFrame) {
    require(raw.size() >= 8 && raw[0]=='A' && raw[1]=='L' && raw[2]=='P' && raw[3]=='2',"invalid ALP2 header");
    std::size_t pos = 4; const auto n = get(raw,pos,4); std::uint64_t produced = 0; J blocks = J::array();
    while (produced < n) {
      const auto mode = get(raw,pos,1), length = get(raw,pos,2); J block;
      require(length > 0 && length <= 1024 && length <= n-produced,"invalid ALP block extent");
      if (mode == 0) {
        const auto size = get(raw,pos,4); require(size == length*8,"invalid ALP raw block size");
        block = J{{"count",length},{"raw",hex(slice(raw,pos,size))}};
      } else if (mode == 1) {
        const int fac = static_cast<std::int8_t>(get(raw,pos,1)), exp = static_cast<std::int8_t>(get(raw,pos,1));
        const auto size = get(raw,pos,4), nexc = get(raw,pos,2);
        block = J{{"count",length},{"fac",fac},{"exp",exp},{"wires",limb_wires(slice(raw,pos,size))},{"exceptions",J::array()}};
        for (std::uint64_t i = 0; i < nexc; ++i) {
          const auto index = get(raw,pos,2), value = get(raw,pos,8);
          block["exceptions"].push_back(J::array({index,value}));
        }
      } else throw std::invalid_argument("invalid ALP2 block mode");
      produced += length; blocks.push_back(block);
    }
    require(pos == raw.size(),"trailing ALP2 bytes");
    obj = J{{"format","alp"},{"stage","wires"},{"count",n},{"blocks",blocks}};
  } else throw std::invalid_argument("unknown legacy numeric action");
  return packet(obj);
}
}  // namespace legacy_numeric_action
