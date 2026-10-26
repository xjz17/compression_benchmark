#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::camel {

// Independent Camel stages: decimal erasure/literal selection, semantic XOR or
// integer-delta records, and public bitstream materialization. Float entry
// points use raw float64 bit patterns; numeric int64 entry points preserve the
// benchmark adapter's int64-to-double semantics.
std::vector<std::uint8_t> eraser_forward(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> eraser_forward_int64_numeric(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> eraser_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::int64_t> eraser_inverse_int64_numeric(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> xor_forward(
    std::span<const std::uint8_t> eraser_packet);
std::vector<std::uint8_t> xor_inverse(
    std::span<const std::uint8_t> xor_packet);
std::vector<std::uint8_t> bitstream_pack_forward(
    std::span<const std::uint8_t> xor_packet);
std::vector<std::uint8_t> bitstream_pack_inverse(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose(std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose_int64_numeric(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose_int64_numeric(
    std::span<const std::uint8_t> wire);

}  // namespace exact_actions::camel
