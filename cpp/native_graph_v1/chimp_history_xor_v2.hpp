#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace native_graph_v1::chimp_history_xor_v2 {

// Both profiles run the same history/XOR mechanism.  Float64 is deliberately
// represented by its raw IEEE-754 bits; the codec never performs FP arithmetic.
enum class Profile : std::uint8_t {
  Int64 = 1,
  Float64 = 2,
};

struct Encoded {
  Profile profile = Profile::Int64;
  std::uint64_t value_count = 0;
  std::uint64_t first_value = 0;
  std::vector<std::uint8_t> reference_slots;
  std::vector<std::uint64_t> xor_residuals;

  bool operator==(const Encoded&) const = default;
};

// Split stage representation used by the graph dispatcher.  The first value,
// count, profile and format version live in metadata.  The independently
// compressible ports contain one reference byte and one little-endian uint64
// XOR residual for each value after the first.
struct StagePayload {
  std::vector<std::uint8_t> metadata;
  std::vector<std::uint8_t> reference_slots;
  std::vector<std::uint8_t> xor_residuals_le;

  bool operator==(const StagePayload&) const = default;
};

Encoded encode_words(std::span<const std::uint64_t> words, Profile profile);
std::vector<std::uint64_t> decode_words(const Encoded& encoded);

StagePayload encode_bytes(std::span<const std::uint8_t> input, Profile profile);
std::vector<std::uint8_t> decode_bytes(
    const StagePayload& payload, Profile expected_profile);

// Exposed so the dispatcher can keep each port as a distinct graph stream.
std::vector<std::uint8_t> serialize_metadata(const Encoded& encoded);
Encoded parse_stage_payload(std::span<const std::uint8_t> metadata,
                            std::span<const std::uint8_t> reference_slots,
                            std::span<const std::uint8_t> xor_residuals_le,
                            Profile expected_profile);

}  // namespace native_graph_v1::chimp_history_xor_v2
