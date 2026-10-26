#include "lzjb_operator_ablation.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace lzjb_operator_ablation {
namespace {

constexpr int kMatchBits = 6;
constexpr int kMatchMin = 3;
constexpr int kMatchMax = (1 << kMatchBits) + (kMatchMin - 1);
constexpr int kOffsetMask = (1 << (16 - kMatchBits)) - 1;
constexpr int kLempelSize = 1024;

struct Token {
  bool match = false;
  std::uint8_t literal = 0;
  std::uint16_t offset = 0;
  std::uint8_t length = 0;
};

std::vector<Token> parse_tokens(std::span<const std::uint8_t> input) {
  std::vector<Token> tokens;
  std::array<std::size_t, kLempelSize> lempel{};
  std::size_t source = 0;
  while (source < input.size()) {
    if (source + kMatchMax > input.size()) {
      tokens.push_back(Token{false, input[source++], 0, 0});
      continue;
    }
    const std::uint32_t hash =
        static_cast<std::uint32_t>(input[source]) << 16 |
        static_cast<std::uint32_t>(input[source + 1]) << 8 |
        input[source + 2];
    const std::uint32_t mixed = hash + (hash >> 9) + (hash >> 5);
    const std::size_t previous = lempel[mixed & (kLempelSize - 1)];
    const int offset = static_cast<int>((source - previous) & kOffsetMask);
    lempel[mixed & (kLempelSize - 1)] = source;
    if (offset > 0 && offset <= static_cast<int>(source)) {
      const std::size_t candidate = source - static_cast<std::size_t>(offset);
      if (input[source] == input[candidate] &&
          input[source + 1] == input[candidate + 1] &&
          input[source + 2] == input[candidate + 2]) {
        int length = kMatchMin;
        for (; length < kMatchMax && source + length < input.size(); ++length) {
          if (input[source + length] != input[candidate + length]) break;
        }
        tokens.push_back(Token{true, 0, static_cast<std::uint16_t>(offset),
                               static_cast<std::uint8_t>(length)});
        source += static_cast<std::size_t>(length);
        continue;
      }
    }
    tokens.push_back(Token{false, input[source++], 0, 0});
  }
  return tokens;
}

std::vector<std::uint8_t> encode_literal_only(std::span<const std::uint8_t> input) {
  std::vector<std::uint8_t> out;
  out.reserve(input.size() + (input.size() + 7) / 8);
  for (std::size_t start = 0; start < input.size(); start += 8) {
    out.push_back(0);
    const std::size_t end = std::min(start + 8, input.size());
    out.insert(out.end(), input.begin() + static_cast<std::ptrdiff_t>(start),
               input.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return out;
}

std::vector<std::uint8_t> encode_packed(const std::vector<Token>& tokens) {
  std::vector<std::uint8_t> out;
  out.reserve(tokens.size() * 2 + (tokens.size() + 7) / 8);
  for (std::size_t start = 0; start < tokens.size(); start += 8) {
    const std::size_t end = std::min(start + 8, tokens.size());
    out.push_back(0);
    const std::size_t map_pos = out.size() - 1;
    std::uint8_t mask = 0x80;
    for (std::size_t i = start; i < end; ++i, mask >>= 1) {
      const Token& token = tokens[i];
      if (!token.match) {
        out.push_back(token.literal);
        continue;
      }
      out[map_pos] |= mask;
      out.push_back(static_cast<std::uint8_t>(
          ((token.length - kMatchMin) << (8 - kMatchBits)) | (token.offset >> 8)));
      out.push_back(static_cast<std::uint8_t>(token.offset));
    }
  }
  return out;
}

std::vector<std::uint8_t> encode_unpacked(const std::vector<Token>& tokens) {
  std::vector<std::uint8_t> out;
  out.reserve(tokens.size() * 3 + 4);
  out.push_back('L');
  out.push_back('J');
  out.push_back('U');
  out.push_back('P');
  for (const Token& token : tokens) {
    out.push_back(token.match ? 1U : 0U);
    if (!token.match) {
      out.push_back(token.literal);
    } else {
      out.push_back(token.length);
      out.push_back(static_cast<std::uint8_t>(token.offset >> 8));
      out.push_back(static_cast<std::uint8_t>(token.offset));
    }
  }
  return out;
}

std::vector<std::uint8_t> decode_packed(std::span<const std::uint8_t> encoded,
                                        std::size_t original_size) {
  std::vector<std::uint8_t> out;
  out.reserve(original_size);
  std::size_t pos = 0;
  while (out.size() < original_size) {
    if (pos >= encoded.size()) throw std::runtime_error("LZJB control map truncated");
    const std::uint8_t map = encoded[pos++];
    for (std::uint8_t mask = 0x80; mask != 0 && out.size() < original_size; mask >>= 1) {
      if ((map & mask) == 0) {
        if (pos >= encoded.size()) throw std::runtime_error("LZJB literal truncated");
        out.push_back(encoded[pos++]);
        continue;
      }
      if (pos + 2 > encoded.size()) throw std::runtime_error("LZJB match truncated");
      const std::uint8_t first = encoded[pos++];
      const std::uint16_t offset = static_cast<std::uint16_t>(
          ((first & ((1U << (8 - kMatchBits)) - 1U)) << 8) | encoded[pos++]);
      const std::size_t length = (first >> (8 - kMatchBits)) + kMatchMin;
      if (offset == 0 || offset > out.size()) throw std::runtime_error("LZJB invalid offset");
      const std::size_t start = out.size() - offset;
      for (std::size_t i = 0; i < length && out.size() < original_size; ++i) {
        out.push_back(out[start + i]);
      }
    }
  }
  if (pos != encoded.size()) throw std::runtime_error("LZJB trailing bytes");
  return out;
}

std::vector<std::uint8_t> decode_unpacked(std::span<const std::uint8_t> encoded,
                                          std::size_t original_size) {
  if (encoded.size() < 4 || encoded[0] != 'L' || encoded[1] != 'J' ||
      encoded[2] != 'U' || encoded[3] != 'P') {
    throw std::runtime_error("LZJB unpacked stream magic mismatch");
  }
  std::vector<std::uint8_t> out;
  out.reserve(original_size);
  std::size_t pos = 4;
  while (out.size() < original_size) {
    if (pos >= encoded.size()) throw std::runtime_error("LZJB token truncated");
    const std::uint8_t tag = encoded[pos++];
    if (tag == 0) {
      if (pos >= encoded.size()) throw std::runtime_error("LZJB literal truncated");
      out.push_back(encoded[pos++]);
    } else if (tag == 1) {
      if (pos + 3 > encoded.size()) throw std::runtime_error("LZJB unpacked match truncated");
      const std::size_t length = encoded[pos++];
      const std::uint8_t high = encoded[pos++];
      const std::uint8_t low = encoded[pos++];
      const std::uint16_t offset = static_cast<std::uint16_t>(high << 8 | low);
      if (length < kMatchMin || length > kMatchMax || offset == 0 || offset > out.size()) {
        throw std::runtime_error("LZJB invalid unpacked match");
      }
      const std::size_t start = out.size() - offset;
      for (std::size_t i = 0; i < length && out.size() < original_size; ++i) {
        out.push_back(out[start + i]);
      }
    } else {
      throw std::runtime_error("LZJB invalid token tag");
    }
  }
  if (pos != encoded.size()) throw std::runtime_error("LZJB unpacked trailing bytes");
  return out;
}

}  // namespace

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> input,
                                 std::string_view omitted_operator) {
  if (!omitted_operator.empty() && omitted_operator != "lz_match" &&
      omitted_operator != "bitstream_pack") {
    throw std::invalid_argument("unsupported LZJB ablation operator");
  }
  if (omitted_operator == "lz_match") return encode_literal_only(input);
  const auto tokens = parse_tokens(input);
  return omitted_operator == "bitstream_pack" ? encode_unpacked(tokens)
                                                : encode_packed(tokens);
}

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded,
                                 std::size_t original_size,
                                 std::string_view omitted_operator) {
  if (omitted_operator == "bitstream_pack") {
    return decode_unpacked(encoded, original_size);
  }
  return decode_packed(encoded, original_size);
}

}  // namespace lzjb_operator_ablation
