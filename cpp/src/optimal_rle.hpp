#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// URL1 is byte-optimal among this format's four candidate representations.
// It does not claim optimality among arbitrary compression algorithms.
namespace unified_rle {

inline constexpr std::size_t kDefaultMaxDecodedBytes = 256u * 1024u * 1024u;

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> raw,
                               std::uint8_t symbol_width);
std::vector<std::uint8_t> decode(std::span<const std::uint8_t> frame);
std::vector<std::uint8_t> decode(std::span<const std::uint8_t> frame,
                               std::size_t max_decoded_bytes);

// Recognition checks only the magic. A recognized but malformed URL1 must be
// rejected by decode, rather than retried as another format.
bool is_frame(std::span<const std::uint8_t> frame);
// Validates the common header and returns its fixed symbol width (1..8).
std::uint8_t symbol_width(std::span<const std::uint8_t> frame);

// Differential-test oracle for URL1 wire compatibility. Production graph and
// baseline paths call encode(); this entry preserves the frozen O(N*128)
// encoder so optimized implementations can be checked byte for byte.
namespace testing {
std::vector<std::uint8_t> encode_frozen_reference(
    std::span<const std::uint8_t> raw, std::uint8_t symbol_width);
}

}  // namespace unified_rle
