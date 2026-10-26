#pragma once

#include "decimal_scaling.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace native_baseline86_v1::decimal_scaling_v2 {

using Bytes = std::vector<std::uint8_t>;

inline constexpr std::uint16_t maximum_decimal_places =
    decimal_scaling::maximum_decimal_places;

struct Scaled {
  // Every entry is part of the integer-codec payload.  Normal finite values
  // contribute one common-scale word, factored finite overflows contribute
  // coefficient and exponent-adjustment words, and IEEE specials contribute
  // one uint32 word per four original bytes.
  std::vector<std::int64_t> values;
  Bytes metadata;
  std::uint16_t decimal_places = 0;
  std::uint64_t scaled_count = 0;
  std::uint64_t overflow_count = 0;
  std::uint64_t special_count = 0;
};

struct MetadataInfo {
  std::uint64_t original_count = 0;
  std::uint64_t integer_word_count = 0;
  std::uint64_t scaled_count = 0;
  std::uint64_t overflow_count = 0;
  std::uint64_t special_count = 0;
  std::uint16_t decimal_places = 0;
};

// Uses the unchanged V1 whole-column precision definition.  V2 changes only
// how values that overflow the common signed-int64 scale are represented.
std::uint16_t required_decimal_places(std::span<const std::uint8_t> input,
                                      std::string_view dtype);

// P must be the frozen whole-column maximum returned by
// required_decimal_places.  No value is rounded or clipped.  All recovery
// words are returned in values and must be passed through the selected integer
// codec; metadata contains classification only and never raw floating bytes.
Scaled forward(std::span<const std::uint8_t> input, std::string_view dtype,
               std::uint16_t p);

// Validates the complete DSC2 header and two-bit classification map.  Payload
// word canonicality is checked by inverse because the words are separately
// compressed by the caller.
MetadataInfo inspect(std::span<const std::uint8_t> metadata,
                     std::string_view dtype);

// Reconstructs the compact valid-value byte stream from only the decoded
// integer words and DSC2 metadata.  It rejects noncanonical classes, decimal
// factors, special-value words, counts, padding, and trailing bytes.
Bytes inverse(std::span<const std::int64_t> values,
              std::span<const std::uint8_t> metadata,
              std::string_view dtype);

}  // namespace native_baseline86_v1::decimal_scaling_v2
