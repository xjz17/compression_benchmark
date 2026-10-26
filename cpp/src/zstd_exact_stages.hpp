#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::zstd {

// Zstd's public sequence API exposes match finding and parse selection as one
// boundary.  The packet carries literal bytes plus ZSTD_Sequence records.
std::vector<std::uint8_t> sequence_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> sequence_inverse(std::span<const std::uint8_t> packet);

// Consume exported sequences and materialize literals, FSE/Huffman tables and
// frame bytes using ZSTD_compressSequences().
std::vector<std::uint8_t> entropy_frame_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> entropy_frame_inverse(std::span<const std::uint8_t> wire);

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::zstd
