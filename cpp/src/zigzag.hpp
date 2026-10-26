#pragma once

#include <bit>
#include <cstdint>

// A046: signed integer bits <-> unsigned ZigZag code, with no prediction,
// packing, or framing. Unsigned arithmetic covers the complete signed domain.
namespace compression_operators::zigzag {

constexpr std::uint64_t encode64(std::int64_t value) noexcept {
  const auto word = std::bit_cast<std::uint64_t>(value);
  return (word << 1) ^ (std::uint64_t{0} - (word >> 63));
}

constexpr std::int64_t decode64(std::uint64_t value) noexcept {
  return std::bit_cast<std::int64_t>(
      (value >> 1) ^ (std::uint64_t{0} - (value & 1)));
}

constexpr std::uint32_t encode32(std::int32_t value) noexcept {
  const auto word = std::bit_cast<std::uint32_t>(value);
  return (word << 1) ^ (std::uint32_t{0} - (word >> 31));
}

constexpr std::int32_t decode32(std::uint32_t value) noexcept {
  return std::bit_cast<std::int32_t>(
      (value >> 1) ^ (std::uint32_t{0} - (value & 1)));
}

}  // namespace compression_operators::zigzag
