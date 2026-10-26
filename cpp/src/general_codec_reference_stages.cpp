#include "general_codec_reference_stages.hpp"
#include "operator_layout_primitives.hpp"

#include "ans_rans.hpp"
#include "arithmetic_bytes.hpp"
#include "huffman_bytes.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace reference_actions::general {
namespace {

using Bytes = std::vector<std::uint8_t>;

void put_u32(Bytes& out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void put_u64(Bytes& out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 4 > input.size()) throw std::runtime_error("truncated uint32");
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) value |= std::uint32_t(input[pos++]) << (8 * i);
  return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t& pos) {
  if (pos + 8 > input.size()) throw std::runtime_error("truncated uint64");
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= std::uint64_t(input[pos++]) << (8 * i);
  return value;
}

void require_magic(std::span<const std::uint8_t> input, std::string_view magic) {
  if (magic.size() != 4 || input.size() < 4 ||
      std::memcmp(input.data(), magic.data(), 4) != 0)
    throw std::runtime_error("unexpected reference-stage packet");
}

Bytes wrap(std::string_view magic, std::span<const std::uint8_t> payload) {
  Bytes out(magic.begin(), magic.end());
  put_u64(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

Bytes unwrap(std::string_view magic, std::span<const std::uint8_t> packet) {
  require_magic(packet, magic);
  std::size_t pos = 4;
  const auto size = get_u64(packet, pos);
  if (size != packet.size() - pos) throw std::runtime_error("packet size mismatch");
  return Bytes(packet.begin() + static_cast<std::ptrdiff_t>(pos), packet.end());
}

struct Token {
  bool match = false;
  std::uint8_t literal = 0;
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
};

struct Semantic {
  std::uint64_t raw_size = 0;
  std::vector<Token> tokens;
};

Bytes serialize_semantic(const Semantic& value) {
  Bytes out{'L', 'Z', 'S', '1'};
  put_u64(out, value.raw_size);
  put_u32(out, static_cast<std::uint32_t>(value.tokens.size()));
  for (const auto& token : value.tokens) {
    out.push_back(token.match ? 1 : 0);
    if (!token.match) out.push_back(token.literal);
    else {
      put_u32(out, token.offset);
      put_u32(out, token.length);
    }
  }
  return out;
}

Semantic parse_semantic(std::span<const std::uint8_t> packet) {
  require_magic(packet, "LZS1");
  std::size_t pos = 4;
  Semantic value;
  value.raw_size = get_u64(packet, pos);
  const auto count = get_u32(packet, pos);
  value.tokens.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (pos >= packet.size()) throw std::runtime_error("truncated LZ token tag");
    Token token;
    token.match = packet[pos++] != 0;
    if (!token.match) {
      if (pos >= packet.size()) throw std::runtime_error("truncated LZ literal");
      token.literal = packet[pos++];
    } else {
      token.offset = get_u32(packet, pos);
      token.length = get_u32(packet, pos);
    }
    value.tokens.push_back(token);
  }
  if (pos != packet.size()) throw std::runtime_error("trailing LZ semantic bytes");
  return value;
}

Bytes expand(const Semantic& value) {
  Bytes out;
  out.reserve(static_cast<std::size_t>(value.raw_size));
  for (const auto& token : value.tokens) {
    if (!token.match) out.push_back(token.literal);
    else {
      if (token.offset == 0 || token.offset > out.size() || token.length == 0)
        throw std::runtime_error("invalid reference LZ match");
      for (std::uint32_t i = 0; i < token.length; ++i)
        out.push_back(out[out.size() - token.offset]);
    }
    if (out.size() > value.raw_size) throw std::runtime_error("LZ expansion overflow");
  }
  if (out.size() != value.raw_size) throw std::runtime_error("LZ expansion size mismatch");
  return out;
}

std::uint32_t key3(std::span<const std::uint8_t> input, std::size_t pos) {
  return (std::uint32_t(input[pos]) << 16) |
         (std::uint32_t(input[pos + 1]) << 8) | input[pos + 2];
}

void remember(std::span<const std::uint8_t> input, std::size_t begin, std::size_t end,
              std::unordered_map<std::uint32_t, std::vector<std::size_t>>& history) {
  for (std::size_t pos = begin; pos < end && pos + 2 < input.size(); ++pos) {
    auto& bucket = history[key3(input, pos)];
    bucket.push_back(pos);
    if (bucket.size() > 128) bucket.erase(bucket.begin(), bucket.begin() + 64);
  }
}

std::pair<std::uint32_t, std::uint32_t> best_match(
    std::span<const std::uint8_t> input, std::size_t pos,
    const std::unordered_map<std::uint32_t, std::vector<std::size_t>>& history,
    std::uint32_t window, std::uint32_t min_match, std::uint32_t max_match) {
  if (pos + min_match > input.size() || pos + 2 >= input.size()) return {0, 0};
  const auto found = history.find(key3(input, pos));
  if (found == history.end()) return {0, 0};
  std::uint32_t best_len = 0, best_off = 0;
  std::size_t checked = 0;
  for (auto it = found->second.rbegin(); it != found->second.rend() && checked < 64;
       ++it, ++checked) {
    const std::size_t previous = *it;
    const std::size_t offset = pos - previous;
    if (offset == 0 || offset > window) continue;
    const std::size_t limit = std::min<std::size_t>(max_match, input.size() - pos);
    std::size_t length = 0;
    while (length < limit && input[previous + length] == input[pos + length]) ++length;
    if (length >= min_match && length > best_len) {
      best_len = static_cast<std::uint32_t>(length);
      best_off = static_cast<std::uint32_t>(offset);
    }
  }
  return {best_len, best_off};
}

}  // namespace

Bytes dictionary_forward(std::span<const std::uint8_t> input) {
  static constexpr std::array<std::string_view, 12> words = {
      "compression", "temperature", "timestamp", "sensor", "value", "column",
      "dictionary", "integer", "floating", "the", "and", "http"};
  Bytes out{'B', 'R', 'D', '1'};
  put_u64(out, input.size());
  for (std::size_t pos = 0; pos < input.size();) {
    std::size_t chosen = words.size();
    for (std::size_t i = 0; i < words.size(); ++i) {
      if (words[i].size() <= input.size() - pos &&
          std::equal(words[i].begin(), words[i].end(),
                     input.begin() + static_cast<std::ptrdiff_t>(pos))) {
        chosen = i;
        break;
      }
    }
    if (chosen != words.size()) {
      out.push_back(1);
      out.push_back(static_cast<std::uint8_t>(chosen));
      pos += words[chosen].size();
    } else {
      out.push_back(0);
      out.push_back(input[pos++]);
    }
  }
  return out;
}

Bytes dictionary_inverse(std::span<const std::uint8_t> packet) {
  static constexpr std::array<std::string_view, 12> words = {
      "compression", "temperature", "timestamp", "sensor", "value", "column",
      "dictionary", "integer", "floating", "the", "and", "http"};
  require_magic(packet, "BRD1");
  std::size_t pos = 4;
  const auto raw_size = get_u64(packet, pos);
  Bytes out;
  out.reserve(raw_size);
  while (pos < packet.size() && out.size() < raw_size) {
    if (pos + 2 > packet.size()) throw std::runtime_error("truncated dictionary token");
    const auto tag = packet[pos++];
    const auto value = packet[pos++];
    if (tag == 0) out.push_back(value);
    else if (tag == 1 && value < words.size())
      out.insert(out.end(), words[value].begin(), words[value].end());
    else throw std::runtime_error("invalid dictionary token");
  }
  if (pos != packet.size() || out.size() != raw_size)
    throw std::runtime_error("dictionary size mismatch");
  return out;
}

Bytes lz_forward(std::span<const std::uint8_t> input, std::uint32_t window,
                 std::uint32_t min_match, std::uint32_t max_match, bool lazy) {
  if (window == 0 || min_match < 3 || max_match < min_match)
    throw std::invalid_argument("invalid reference LZ parameters");
  Semantic semantic;
  semantic.raw_size = input.size();
  std::unordered_map<std::uint32_t, std::vector<std::size_t>> history;
  for (std::size_t pos = 0; pos < input.size();) {
    auto [length, offset] = best_match(input, pos, history, window, min_match, max_match);
    if (lazy && length >= min_match && pos + 1 < input.size()) {
      auto future_history = history;
      remember(input, pos, pos + 1, future_history);
      const auto next = best_match(input, pos + 1, future_history, window, min_match, max_match);
      if (next.first > length + 1) length = 0;
    }
    if (length >= min_match) {
      semantic.tokens.push_back(Token{true, 0, offset, length});
      remember(input, pos, pos + length, history);
      pos += length;
    } else {
      semantic.tokens.push_back(Token{false, input[pos], 0, 0});
      remember(input, pos, pos + 1, history);
      ++pos;
    }
  }
  return serialize_semantic(semantic);
}

Bytes lz_inverse(std::span<const std::uint8_t> packet) {
  return expand(parse_semantic(packet));
}

Bytes token_forward(std::span<const std::uint8_t> packet) {
  const auto semantic = parse_semantic(packet);
  Bytes out{'L', 'Z', 'T', '1'};
  put_u64(out, semantic.raw_size);
  put_u32(out, static_cast<std::uint32_t>(semantic.tokens.size()));
  for (std::size_t begin = 0; begin < semantic.tokens.size(); begin += 8) {
    const std::size_t end = std::min(begin + 8, semantic.tokens.size());
    const auto ctrl_pos = out.size();
    out.push_back(0);
    for (std::size_t i = begin; i < end; ++i) {
      const auto& token = semantic.tokens[i];
      if (!token.match) out.push_back(token.literal);
      else {
        out[ctrl_pos] |= static_cast<std::uint8_t>(1u << (i - begin));
        put_u32(out, token.offset);
        put_u32(out, token.length);
      }
    }
  }
  return out;
}

Bytes token_inverse(std::span<const std::uint8_t> packet) {
  require_magic(packet, "LZT1");
  std::size_t pos = 4;
  Semantic semantic;
  semantic.raw_size = get_u64(packet, pos);
  const auto count = get_u32(packet, pos);
  semantic.tokens.reserve(count);
  while (semantic.tokens.size() < count) {
    if (pos >= packet.size()) throw std::runtime_error("truncated token control byte");
    const auto ctrl = packet[pos++];
    for (int bit = 0; bit < 8 && semantic.tokens.size() < count; ++bit) {
      Token token;
      token.match = (ctrl & (1u << bit)) != 0;
      if (!token.match) {
        if (pos >= packet.size()) throw std::runtime_error("truncated packed literal");
        token.literal = packet[pos++];
      } else {
        token.offset = get_u32(packet, pos);
        token.length = get_u32(packet, pos);
      }
      semantic.tokens.push_back(token);
    }
  }
  if (pos != packet.size()) throw std::runtime_error("trailing token bytes");
  return serialize_semantic(semantic);
}

Bytes wrap_lz_semantic_packet(std::span<const std::uint8_t> packet) {
  (void)parse_semantic(packet);
  return wrap("ZPP1", packet);
}

Bytes unwrap_lz_semantic_packet(std::span<const std::uint8_t> packet) {
  auto semantic = unwrap("ZPP1", packet);
  (void)parse_semantic(semantic);
  return semantic;
}

Bytes parse_policy_forward(std::span<const std::uint8_t> packet) {
  return wrap_lz_semantic_packet(packet);
}

Bytes parse_policy_inverse(std::span<const std::uint8_t> packet) {
  return unwrap_lz_semantic_packet(packet);
}

Bytes huffman_forward(std::span<const std::uint8_t> input) {
  return huffman_bytes::encode(input);
}

Bytes huffman_inverse(std::span<const std::uint8_t> packet) {
  return huffman_bytes::decode(packet);
}

Bytes rans_forward(std::span<const std::uint8_t> input) {
  return ans_rans::encode(input);
}

Bytes rans_inverse(std::span<const std::uint8_t> packet) {
  return ans_rans::decode(packet);
}

Bytes range_forward(std::span<const std::uint8_t> input) {
  return arithmetic_bytes::encode(input);
}

Bytes range_inverse(std::span<const std::uint8_t> packet) {
  return arithmetic_bytes::decode(packet);
}

Bytes shuffle_forward(std::span<const std::uint8_t> input, std::size_t type_size) {
  if (type_size == 0 || type_size > 255)
    throw std::invalid_argument("shuffle type size must be in [1,255]");
  Bytes out{'S', 'H', 'F', '1'};
  out.push_back(static_cast<std::uint8_t>(type_size));
  put_u64(out, input.size());
  const std::size_t elements = input.size() / type_size;
  const auto columns=exact_actions::operator_layout::columnize(input.first(elements*type_size),type_size);
  out.insert(out.end(),columns.begin(),columns.end());
  out.insert(out.end(),
             input.begin() + static_cast<std::ptrdiff_t>(elements * type_size),
             input.end());
  return out;
}

Bytes shuffle_inverse(std::span<const std::uint8_t> packet) {
  require_magic(packet, "SHF1");
  std::size_t pos = 4;
  if (pos >= packet.size()) throw std::runtime_error("truncated shuffle type size");
  const std::size_t type_size = packet[pos++];
  const std::size_t raw_size = get_u64(packet, pos);
  if (type_size == 0 || packet.size() - pos != raw_size)
    throw std::runtime_error("shuffle packet size mismatch");
  Bytes out(raw_size);
  const std::size_t elements = raw_size / type_size;
  const auto records=exact_actions::operator_layout::uncolumnize(packet.subspan(pos,elements*type_size),type_size);
  std::copy(records.begin(),records.end(),out.begin()); pos+=records.size();
  std::copy(packet.begin() + static_cast<std::ptrdiff_t>(pos), packet.end(),
            out.begin() + static_cast<std::ptrdiff_t>(elements * type_size));
  return out;
}

Bytes frame_forward(std::span<const std::uint8_t> input, bool blosc2) {
  return wrap(blosc2 ? "B2F1" : "B1F1", input);
}

Bytes frame_inverse(std::span<const std::uint8_t> packet, bool blosc2) {
  return unwrap(blosc2 ? "B2F1" : "B1F1", packet);
}

Bytes gzip_frame_forward(std::span<const std::uint8_t> input) {
  return wrap("GZF1", input);
}
Bytes gzip_frame_inverse(std::span<const std::uint8_t> packet) {
  return unwrap("GZF1", packet);
}
Bytes zlib_frame_forward(std::span<const std::uint8_t> input) {
  return wrap("ZLF1", input);
}
Bytes zlib_frame_inverse(std::span<const std::uint8_t> packet) {
  return unwrap("ZLF1", packet);
}
Bytes xz_frame_forward(std::span<const std::uint8_t> input) {
  return wrap("XZF1", input);
}
Bytes xz_frame_inverse(std::span<const std::uint8_t> packet) {
  return unwrap("XZF1", packet);
}

Bytes bwt_forward(std::span<const std::uint8_t> input) {
  const std::size_t n = input.size();
  Bytes out{'B', 'W', 'T', '1'};
  put_u64(out, n);
  if (n == 0) {
    put_u64(out, 0);
    return out;
  }
  std::vector<std::size_t> order(n), classes(n), shifted(n), next_classes(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return input[a] < input[b];
  });
  for (std::size_t i = 1; i < n; ++i)
    classes[order[i]] = classes[order[i - 1]] +
                        static_cast<std::size_t>(input[order[i]] != input[order[i - 1]]);
  for (std::size_t length = 1; length < n;) {
    for (std::size_t i = 0; i < n; ++i) shifted[i] = (order[i] + n - length) % n;
    std::vector<std::size_t> count(classes[order.back()] + 1, 0), position;
    for (auto value : shifted) ++count[classes[value]];
    position.resize(count.size(), 0);
    for (std::size_t i = 1; i < count.size(); ++i)
      position[i] = position[i - 1] + count[i - 1];
    for (auto value : shifted) order[position[classes[value]]++] = value;
    next_classes[order[0]] = 0;
    for (std::size_t i = 1; i < n; ++i) {
      const auto current = std::pair(classes[order[i]], classes[(order[i] + length) % n]);
      const auto previous =
          std::pair(classes[order[i - 1]], classes[(order[i - 1] + length) % n]);
      next_classes[order[i]] = next_classes[order[i - 1]] +
                               static_cast<std::size_t>(current != previous);
    }
    classes.swap(next_classes);
    if (length > n / 2) break;
    length <<= 1;
  }
  std::size_t primary = 0;
  for (std::size_t row = 0; row < n; ++row)
    if (order[row] == 0) primary = row;
  put_u64(out, primary);
  for (auto start : order) out.push_back(input[(start + n - 1) % n]);
  return out;
}

Bytes bwt_inverse(std::span<const std::uint8_t> packet) {
  require_magic(packet, "BWT1");
  std::size_t pos = 4;
  const std::size_t n = get_u64(packet, pos);
  const std::size_t primary = get_u64(packet, pos);
  if (packet.size() - pos != n || (n != 0 && primary >= n))
    throw std::runtime_error("BWT packet size mismatch");
  if (n == 0) return {};
  const auto last = packet.subspan(pos);
  std::array<std::size_t, 256> count{}, start{}, seen{};
  for (auto byte : last) ++count[byte];
  for (std::size_t i = 1; i < 256; ++i) start[i] = start[i - 1] + count[i - 1];
  std::vector<std::size_t> next(n);
  for (std::size_t i = 0; i < n; ++i) next[start[last[i]] + seen[last[i]]++] = i;
  Bytes out(n);
  std::size_t row = primary;
  for (std::size_t i = 0; i < n; ++i) {
    row = next[row];
    out[i] = last[row];
  }
  return out;
}

Bytes mtf_forward(std::span<const std::uint8_t> input) {
  std::array<std::uint8_t, 256> table{};
  std::iota(table.begin(), table.end(), 0);
  Bytes body;
  body.reserve(input.size());
  for (auto value : input) {
    const auto found = std::find(table.begin(), table.end(), value);
    const auto index = static_cast<std::size_t>(found - table.begin());
    body.push_back(static_cast<std::uint8_t>(index));
    std::move_backward(table.begin(), found, found + 1);
    table[0] = value;
  }
  return wrap("MTF1", body);
}

Bytes mtf_inverse(std::span<const std::uint8_t> packet) {
  const auto body = unwrap("MTF1", packet);
  std::array<std::uint8_t, 256> table{};
  std::iota(table.begin(), table.end(), 0);
  Bytes out;
  out.reserve(body.size());
  for (auto index : body) {
    const auto value = table[index];
    out.push_back(value);
    std::move_backward(table.begin(), table.begin() + index,
                       table.begin() + index + 1);
    table[0] = value;
  }
  return out;
}

Bytes rle_forward(std::span<const std::uint8_t> input) {
  Bytes out{'R', 'L', 'B', '1'};
  put_u64(out, input.size());
  for (std::size_t pos = 0; pos < input.size();) {
    std::size_t end = pos + 1;
    while (end < input.size() && input[end] == input[pos] &&
           end - pos < std::numeric_limits<std::uint32_t>::max())
      ++end;
    out.push_back(input[pos]);
    put_u32(out, static_cast<std::uint32_t>(end - pos));
    pos = end;
  }
  return out;
}

Bytes rle_inverse(std::span<const std::uint8_t> packet) {
  require_magic(packet, "RLB1");
  std::size_t pos = 4;
  const std::size_t raw_size = get_u64(packet, pos);
  Bytes out;
  out.reserve(raw_size);
  while (pos < packet.size()) {
    const auto value = packet[pos++];
    const auto count = get_u32(packet, pos);
    if (count == 0 || count > raw_size - out.size())
      throw std::runtime_error("invalid RLE run");
    out.insert(out.end(), count, value);
  }
  if (out.size() != raw_size) throw std::runtime_error("RLE size mismatch");
  return out;
}

}  // namespace reference_actions::general
