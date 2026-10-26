#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace native_baseline86_v1::decimal_scaling {

using Bytes = std::vector<std::uint8_t>;

inline constexpr std::uint16_t maximum_decimal_places = 324;
inline constexpr std::uint64_t maximum_buffer_bytes = 256ULL << 20;

struct Scaled {
  std::vector<std::int64_t> values;
  std::vector<std::uint8_t> metadata;
  std::uint16_t decimal_places = 0;
  std::uint64_t exception_count = 0;
};

struct MetadataInfo {
  std::uint16_t decimal_places = 0;
  std::uint64_t original_count = 0;
  std::uint64_t scaled_count = 0;
  std::uint64_t exception_count = 0;
};

// Returns the largest fractional-place requirement in the canonical shortest
// round-trip decimal spellings of the compact valid-value stream.  NaN, Inf,
// and negative zero are raw-bit exceptions and do not affect the scale.
std::uint16_t required_decimal_places(std::span<const std::uint8_t> input,
                                      std::string_view dtype);

// p is explicit experiment metadata and must be at least
// required_decimal_places(input, dtype).  Scalable lanes are compacted into
// values; metadata (DSC1) carries their positions plus byte-exact exceptions.
Scaled forward(std::span<const std::uint8_t> input, std::string_view dtype,
               std::uint16_t p);

// Validates the complete DSC1 structure, bitmap padding/count, raw exception
// length and canonicality, dtype, trailing bytes, and allocation bounds.  It
// lets the outer frame validate scaled_count before invoking an integer codec.
MetadataInfo inspect(std::span<const std::uint8_t> metadata,
                     std::string_view dtype);

// Reconstructs the complete compact valid-value stream using only the decoded
// integer payload and DSC1 metadata.  It rejects malformed and noncanonical
// exception metadata and never consults the original input or a cache.
Bytes inverse(std::span<const std::int64_t> values,
              std::span<const std::uint8_t> metadata,
              std::string_view dtype);

}  // namespace native_baseline86_v1::decimal_scaling
