#pragma once
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace exact_actions::adjacent_residual {

template<std::unsigned_integral Word>
constexpr Word xor_word(Word current, Word previous) { return current ^ previous; }

template<std::unsigned_integral Word>
constexpr Word delta_word(Word current, Word previous) { return current - previous; }

inline std::int64_t checked_delta(std::int64_t current, std::int64_t previous) {
  const __int128 value = static_cast<__int128>(current) - previous;
  if (value < std::numeric_limits<std::int64_t>::min() || value > std::numeric_limits<std::int64_t>::max())
    throw std::runtime_error("signed delta exceeds int64");
  return static_cast<std::int64_t>(value);
}

inline std::int64_t checked_restore(std::int64_t delta, std::int64_t previous) {
  const __int128 value = static_cast<__int128>(previous) + delta;
  if (value < std::numeric_limits<std::int64_t>::min() || value > std::numeric_limits<std::int64_t>::max())
    throw std::runtime_error("signed delta reconstruction exceeds int64");
  return static_cast<std::int64_t>(value);
}
}  // namespace exact_actions::adjacent_residual
