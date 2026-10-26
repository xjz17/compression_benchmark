#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Numerical FOR only: one minimum and one unsigned offset per input value.
// Codec block boundaries, exception masks, scales and wire headers stay outside.
namespace minimum_for {
template<class T>
concept Word = std::same_as<T, std::int64_t> || std::same_as<T, std::uint64_t> ||
               std::same_as<T, std::int32_t> || std::same_as<T, std::uint32_t>;

template<Word T> struct Result {
  T minimum{};
  std::vector<std::make_unsigned_t<T>> offsets;
};

template<Word T> inline std::make_unsigned_t<T> offset(T value, T base) {
  return static_cast<std::make_unsigned_t<T>>(value) - static_cast<std::make_unsigned_t<T>>(base);
}

// Modular arithmetic preserves the historical adapters' full-word semantics.
template<Word T> inline T restore(T base, std::make_unsigned_t<T> residual) {
  const auto word = static_cast<std::make_unsigned_t<T>>(base) + residual;
  if constexpr (std::is_signed_v<T>) return std::bit_cast<T>(word);
  else return word;
}

// For existing codecs whose caller combines a prediction before checking range.
template<Word T> inline __int128 restore_wide(T base, std::make_unsigned_t<T> residual) {
  return static_cast<__int128>(base) + static_cast<__int128>(residual);
}

template<Word T> Result<T> encode(std::span<const T> values) {
  Result<T> result;
  if (!values.empty()) result.minimum = *std::min_element(values.begin(), values.end());
  result.offsets.reserve(values.size());
  for (const auto value : values) result.offsets.push_back(offset(value, result.minimum));
  return result;
}

template<Word T> std::vector<T> decode(T minimum, std::span<const std::make_unsigned_t<T>> offsets) {
  std::vector<T> values;
  values.reserve(offsets.size());
  for (const auto residual : offsets) values.push_back(restore(minimum, residual));
  return values;
}

// Strict public minimum/offset contract. Legacy supplied-base adapters explicitly
// retain their pre-existing modulo behavior and therefore do not call this check.
template<Word T> void validate_canonical(T minimum, std::span<const std::make_unsigned_t<T>> offsets) {
  if (offsets.empty()) {
    if (minimum != 0) throw std::invalid_argument("empty FOR requires minimum zero");
    return;
  }
  if (*std::min_element(offsets.begin(), offsets.end()) != 0)
    throw std::invalid_argument("FOR offsets must contain zero");
  const auto maximum_offset = static_cast<std::uint64_t>(std::numeric_limits<T>::max()) -
                              static_cast<std::uint64_t>(minimum);
  for (const auto residual : offsets)
    if (residual > maximum_offset)
      throw std::invalid_argument("FOR inverse exceeds the declared integer domain");
}
}  // namespace minimum_for
