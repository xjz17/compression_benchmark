#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

// A023: unsigned base-128 variable-byte encoding. A caller selects the values
// to encode; signed mapping, first values, scale and frame fields are external.
namespace compression_operators::unsigned_leb128 {

inline unsigned encoded_length(std::uint64_t value) noexcept {
  unsigned size = 1;
  while (value >= 128) { value >>= 7; ++size; }
  return size;
}

inline void append(std::vector<std::uint8_t>& output, std::uint64_t value) {
  while (value >= 128) {
    output.push_back(static_cast<std::uint8_t>((value & 127) | 128));
    value >>= 7;
  }
  output.push_back(static_cast<std::uint8_t>(value));
}

inline std::uint64_t read(std::span<const std::uint8_t> input,
                          std::size_t& position, unsigned word_bits = 64) {
  if (word_bits != 32 && word_bits != 64)
    throw std::invalid_argument("ULEB128 word_bits must be 32 or 64");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < word_bits; shift += 7) {
    if (position >= input.size()) throw std::runtime_error("truncated ULEB128");
    const auto byte = input[position++];
    const auto payload = static_cast<std::uint64_t>(byte & 127);
    const auto remaining = word_bits - shift;
    if (remaining < 7 && payload >= (std::uint64_t{1} << remaining))
      throw std::runtime_error("ULEB128 value exceeds input word width");
    value |= payload << shift;
    if ((byte & 128) == 0) return value;
  }
  throw std::runtime_error("oversized ULEB128");
}

inline std::vector<std::uint8_t> encode(std::span<const std::uint64_t> values) {
  std::vector<std::uint8_t> output;
  for (const auto value : values) append(output, value);
  return output;
}

inline std::vector<std::uint64_t> decode(std::span<const std::uint8_t> input) {
  std::vector<std::uint64_t> output;
  for (std::size_t position = 0; position < input.size();)
    output.push_back(read(input, position));
  return output;
}

}  // namespace compression_operators::unsigned_leb128
