#include "native_lz_action_stages.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace native_lz_action {
namespace {
[[noreturn]] void fail(const char* message) { throw std::invalid_argument(message); }
void require(bool test, const char* message) { if (!test) fail(message); }
void put(Bytes& out, std::uint64_t n, unsigned width) {
  for (unsigned i = 0; i < width; ++i) out.push_back(static_cast<std::uint8_t>(n >> (8*i)));
}
std::uint64_t get(const Bytes& raw, std::size_t& pos, unsigned width) {
  require(pos <= raw.size() && width <= raw.size()-pos, "truncated native-LZ packet");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < width; ++i) value |= std::uint64_t(raw[pos++]) << (8*i);
  return value;
}
std::uint64_t key_at(const Bytes& raw, std::size_t pos) {
  std::uint64_t key = 0;
  for (unsigned i = 0; i < 8; ++i) key |= std::uint64_t(raw[pos+i]) << (8*i);
  return key;
}
constexpr std::string_view kWindow = "op.match_window_model.native_lz.window";
constexpr std::string_view kMatch = "op.lz_match.native_lz.aligned8.min";
constexpr std::string_view kParse = "op.match_parse_policy.native_lz.aligned8_greedy";
constexpr std::string_view kPack = "op.match_token_format.native_lz.ctrl8_offset16_len8.bias";
struct Action { unsigned role, minimum = 0, maximum = 0, window = 0; };
std::uint32_t number(std::string_view s) {
  require(!s.empty(), "missing native-LZ action parameter");
  std::uint64_t n = 0;
  for (char c : s) {
    require(c >= '0' && c <= '9', "invalid native-LZ action parameter");
    n = n*10 + (c-'0');
    require(n <= std::numeric_limits<std::uint32_t>::max(), "native-LZ parameter overflow");
  }
  return static_cast<std::uint32_t>(n);
}
Action action_spec(std::string_view name) {
  if (name.substr(0,kWindow.size()) == kWindow)
    return {1,0,0,number(name.substr(kWindow.size()))};
  if (name.substr(0,kMatch.size()) == kMatch) {
    const auto rest = name.substr(kMatch.size());
    const auto split = rest.find(".max");
    require(split != std::string_view::npos, "invalid native-LZ match action");
    return {2,number(rest.substr(0,split)),number(rest.substr(split+4)),0};
  }
  if (name == kParse) return {3};
  if (name.substr(0,kPack.size()) == kPack)
    return {4,number(name.substr(kPack.size()))};
  fail("unknown native-LZ stage action");
}
Bytes packet(const Context& ctx, const Bytes& body) {
  Bytes out{'N','L','S','1',ctx.role};
  put(out,ctx.window,4); put(out,ctx.minimum,4); put(out,ctx.maximum,4); put(out,ctx.raw_size,8);
  out.insert(out.end(),body.begin(),body.end());
  return out;
}
void append_token(Bytes& out, const Token& t, std::uint64_t expected) {
  if (t.literal >= 0) {
    require(t.literal <= 255, "literal out of byte range");
    out.push_back(static_cast<std::uint8_t>(t.literal));
  } else {
    require(t.offset > 0 && t.offset <= out.size() && t.length > 0, "invalid native-LZ match");
    require(out.size() <= expected && t.length <= expected-out.size(), "native-LZ expansion exceeds expected size");
    for (std::uint32_t i = 0; i < t.length; ++i) out.push_back(out[out.size()-t.offset]);
  }
  require(out.size() <= expected, "native-LZ expansion exceeds expected size");
}
}  // namespace

std::vector<Token> parse(const Bytes& raw, std::uint32_t window,
                         std::uint32_t minimum, std::uint32_t maximum) {
  require(minimum > 0 && maximum >= minimum && window > 0 && window <= 65535,
          "invalid native-LZ search parameters");
  // Searching eight-byte keys makes min=3/4/5 equally restrictive. Keep exact
  // configured bounds in NLS1 while searching their effective aligned range;
  // token serialization selects its own independent length bias.
  const auto effective_max = maximum / 8 * 8;
  const auto effective_min = std::max<std::uint64_t>(8, (std::uint64_t(minimum)+7)/8*8);
  std::unordered_map<std::uint64_t,std::size_t> previous;
  std::vector<Token> tokens;
  auto remember = [&](std::size_t pos) {
    if (pos <= raw.size() && raw.size()-pos >= 8) previous[key_at(raw,pos)] = pos;
  };
  std::size_t pos = 0;
  while (pos < raw.size()) {
    unsigned bit = 0;
    while (bit < 8 && pos < raw.size()) {
      std::size_t length = 0, offset = 0;
      if (raw.size()-pos >= 8) {
        const auto it = previous.find(key_at(raw,pos));
        if (it != previous.end() && it->second < pos && pos-it->second <= window) {
          offset = pos-it->second;
          const auto bound = std::min<std::size_t>(effective_max,raw.size()-pos);
          if (bound >= 8) {
            length = 8;
            while (length < bound && raw[it->second+length] == raw[pos+length]) ++length;
            length = length/8*8;
            if (length < effective_min) length = 0;
          }
        }
      }
      if (length) {
        tokens.push_back({-1,static_cast<std::uint32_t>(offset),static_cast<std::uint32_t>(length)});
        remember(pos);
        if (length > 8) remember(pos+length-8);
        pos += length; ++bit;
      } else if (bit == 0 && raw.size()-pos >= 8) {
        remember(pos);
        for (unsigned i = 0; i < 8; ++i) tokens.push_back({raw[pos+i],0,0});
        pos += 8; bit = 8;
      } else {
        tokens.push_back({raw[pos],0,0}); remember(pos); ++pos; ++bit;
      }
    }
  }
  return tokens;
}

std::vector<Token> parse(const Bytes& raw, const SearchConfiguration& configuration) {
  return parse(raw, configuration.window, configuration.minimum, configuration.maximum);
}

Bytes configured_parse_packet(const Bytes& raw, const SearchConfiguration& configuration) {
  const auto tokens = parse(raw, configuration);
  return packet({3,configuration.window,configuration.minimum,configuration.maximum,raw.size()},
                serialize_tokens(tokens));
}

Bytes attach_window(const Bytes& raw, std::uint32_t window) {
  return packet({1,window,0,0,raw.size()},raw);
}

Bytes attach_match_limits(const Bytes& raw, std::uint32_t minimum, std::uint32_t maximum) {
  auto c = context(raw); const Bytes body(raw.begin()+25,raw.end());
  require(c.role == 1 && body.size() == c.raw_size,"match requires window packet");
  c.role = 2; c.minimum = minimum; c.maximum = maximum;
  return packet(c,body);
}

std::vector<Token> parse_chain(const Bytes& raw, std::uint32_t window,
                               std::uint32_t minimum, std::uint32_t maximum,
                               std::uint32_t candidates, bool lazy) {
  require(minimum >= 3 && maximum >= minimum && maximum-minimum <= 255 &&
          window > 0 && window <= 65535 && candidates > 0 && candidates <= 1000000,
          "invalid native-LZ chain parameters");
  const auto cap = std::max<std::size_t>(4,std::size_t(candidates)*8);
  std::unordered_map<std::uint32_t,std::vector<std::size_t>> chains;
  auto key = [&](std::size_t pos) { return (std::uint32_t(raw[pos])<<16)|(std::uint32_t(raw[pos+1])<<8)|raw[pos+2]; };
  auto remember = [&](std::size_t pos) {
    if (raw.size()-pos < 3) return;
    auto& bucket = chains[key(pos)]; bucket.push_back(pos);
    if (bucket.size() > cap*2) bucket.erase(bucket.begin(),bucket.end()-cap);
  };
  auto find = [&](std::size_t pos) -> std::pair<std::size_t,std::size_t> {
    if (raw.size()-pos < minimum || raw.size()-pos < 3) return {0,0};
    const auto found = chains.find(key(pos));
    if (found == chains.end()) return {0,0};
    const auto& bucket = found->second;
    const auto bound = std::min<std::size_t>(maximum,raw.size()-pos);
    std::size_t best_length = 0, best_offset = 0, checked = 0;
    const auto first = bucket.size() > cap ? bucket.size()-cap : 0;
    for (std::size_t i = bucket.size(); i > first;) {
      const auto prev = bucket[--i];
      if (prev >= pos || pos-prev > window) continue;
      const auto offset = pos-prev; ++checked;
      std::size_t length = 3;
      while (length < bound && raw[prev+length] == raw[pos+length]) ++length;
      if (length >= minimum && (length > best_length ||
          (length == best_length && (best_offset == 0 || offset < best_offset)))) {
        best_length = length; best_offset = offset;
        if (best_length == bound) break;
      }
      if (checked >= candidates) break;
    }
    return {best_length,best_offset};
  };
  std::vector<Token> tokens; std::size_t pos = 0;
  while (pos < raw.size()) {
    const auto [length,offset] = find(pos);
    if (lazy && length >= minimum && pos+1 < raw.size()) {
      remember(pos);
      const auto next = find(pos+1);
      if (next.first > length) { tokens.push_back({raw[pos],0,0}); ++pos; continue; }
    }
    if (length >= minimum) {
      tokens.push_back({-1,static_cast<std::uint32_t>(offset),static_cast<std::uint32_t>(length)});
      for (std::size_t i = pos; i < pos+length; ++i) remember(i);
      pos += length;
    } else { tokens.push_back({raw[pos],0,0}); remember(pos); ++pos; }
  }
  return tokens;
}
Bytes literal(const Bytes& raw) {
  Bytes out;
  for (std::size_t pos = 0; pos < raw.size(); pos += 8) {
    out.push_back(0); const auto end = std::min(pos+8,raw.size());
    out.insert(out.end(),raw.begin()+pos,raw.begin()+end);
  }
  return out;
}

Bytes serialize_tokens(const std::vector<Token>& tokens) {
  Bytes out; put(out,tokens.size(),8);
  for (const auto& t : tokens) {
    if (t.literal >= 0) {
      require(t.literal <= 255,"literal out of byte range"); out.push_back(0); out.push_back(t.literal);
    } else {
      require(t.offset > 0 && t.offset <= 65535 && t.length > 0 && t.length <= 65535,"token outside NLS1 range");
      out.push_back(1); put(out,t.offset,2); put(out,t.length,2);
    }
  }
  return out;
}
std::vector<Token> deserialize_tokens(const Bytes& raw) {
  std::size_t pos = 0; const auto count = get(raw,pos,8);
  require(count <= (raw.size()-pos)/2,"truncated native-LZ token list");
  std::vector<Token> tokens;
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto tag = get(raw,pos,1);
    if (tag == 0) tokens.push_back({static_cast<int>(get(raw,pos,1)),0,0});
    else if (tag == 1) {
      const auto off = get(raw,pos,2), len = get(raw,pos,2);
      require(off > 0 && len > 0,"zero native-LZ match field");
      tokens.push_back({-1,static_cast<std::uint32_t>(off),static_cast<std::uint32_t>(len)});
    } else fail("invalid native-LZ token tag");
  }
  require(pos == raw.size(),"trailing native-LZ token bytes");
  return tokens;
}
Bytes expand(const std::vector<Token>& tokens, std::uint64_t expected) {
  Bytes out;
  for (const auto& t : tokens) append_token(out,t,expected);
  require(out.size() == expected,"native-LZ expanded size mismatch");
  return out;
}
Bytes pack(const std::vector<Token>& tokens, std::uint32_t minimum) {
  Bytes out;
  for (std::size_t start = 0; start < tokens.size(); start += 8) {
    const auto ctrl_pos = out.size(); out.push_back(0); std::uint8_t ctrl = 0;
    for (unsigned bit = 0; bit < 8 && start+bit < tokens.size(); ++bit) {
      const auto& t = tokens[start+bit];
      if (t.literal >= 0) { require(t.literal <= 255,"invalid ctrl8 literal"); out.push_back(t.literal); }
      else {
        require(t.offset > 0 && t.offset <= 65535 && t.length >= minimum && t.length-minimum <= 255,"match outside ctrl8 wire range");
        ctrl |= std::uint8_t(1u<<bit); put(out,t.offset,2); put(out,t.length-minimum,1);
      }
    }
    out[ctrl_pos] = ctrl;
  }
  return out;
}
std::vector<Token> unpack(const Bytes& raw, std::uint32_t minimum, std::uint64_t expected) {
  Bytes expanded; std::vector<Token> tokens; std::size_t pos = 0;
  while (expanded.size() < expected) {
    const auto ctrl = get(raw,pos,1);
    for (unsigned bit = 0; bit < 8 && expanded.size() < expected; ++bit) {
      Token t;
      if (ctrl & (1u<<bit)) {
        t.offset = get(raw,pos,2); t.length = get(raw,pos,1)+minimum;
      } else t.literal = get(raw,pos,1);
      append_token(expanded,t,expected); tokens.push_back(t);
    }
  }
  require(pos == raw.size(),"trailing native-LZ ctrl8 bytes");
  return tokens;
}
Context context(const Bytes& raw) {
  require(raw.size() >= 25 && raw[0]=='N' && raw[1]=='L' && raw[2]=='S' && raw[3]=='1',"invalid NLS1 header");
  std::size_t pos = 4; Context ctx;
  ctx.role = get(raw,pos,1); require(ctx.role >= 1 && ctx.role <= 3,"invalid NLS1 role");
  ctx.window = get(raw,pos,4); ctx.minimum = get(raw,pos,4); ctx.maximum = get(raw,pos,4); ctx.raw_size = get(raw,pos,8);
  return ctx;
}
Bytes forward(const std::string& action, const Bytes& raw) {
  const auto a = action_spec(action);
  if (a.role == 1) return attach_window(raw,a.window);
  if (a.role == 2) return attach_match_limits(raw,a.minimum,a.maximum);
  auto c = context(raw); const Bytes body(raw.begin()+25,raw.end());
  if (a.role == 3) {
    require(c.role == 2 && body.size() == c.raw_size,"parse requires match packet");
    return configured_parse_packet(body,{c.window,c.minimum,c.maximum});
  }
  // The search threshold is independent of the token format's length bias.
  // pack() validates every actual match against that wire representation.
  require(c.role == 3,"token format requires parse packet");
  return pack(deserialize_tokens(body),a.minimum);
}
Bytes inverse(const std::string& action, const Bytes& raw, const Context* recorded) {
  const auto a = action_spec(action);
  if (a.role == 4) {
    require(recorded != nullptr && recorded->role == 3,"token inverse needs parse context");
    auto c = *recorded; c.role = 3;
    return packet(c,serialize_tokens(unpack(raw,a.minimum,c.raw_size)));
  }
  auto c = context(raw); const Bytes body(raw.begin()+25,raw.end());
  require(c.role == a.role,"stage role mismatch");
  if (a.role == 1) {
    require(c.window == a.window && body.size() == c.raw_size,"window packet mismatch"); return body;
  }
  if (a.role == 2) {
    require(c.minimum == a.minimum && c.maximum == a.maximum && body.size() == c.raw_size,"match packet mismatch");
    c.role = 1; c.minimum = c.maximum = 0; return packet(c,body);
  }
  c.role = 2; return packet(c,expand(deserialize_tokens(body),c.raw_size));
}
}  // namespace native_lz_action
