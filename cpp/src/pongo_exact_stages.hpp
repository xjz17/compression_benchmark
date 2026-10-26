#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::pongo {

// All typed inputs are carried as little-endian 64-bit lanes.  For float64
// those lanes are interpreted as their exact IEEE-754 bit patterns; int64
// inputs are accepted by the unified runtime as the same lossless bit domain.
std::vector<std::uint8_t> eraser_forward(std::span<const std::uint8_t> raw_lanes);
std::vector<std::uint8_t> eraser_inverse(std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> xor_forward(std::span<const std::uint8_t> eraser_packet);
std::vector<std::uint8_t> xor_inverse(std::span<const std::uint8_t> xor_packet);

std::vector<std::uint8_t> trailing_pack_forward(
    std::span<const std::uint8_t> xor_packet);
std::vector<std::uint8_t> trailing_pack_inverse(
    std::span<const std::uint8_t> public_wire);

std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw_lanes);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> public_wire);

}  // namespace exact_actions::pongo
