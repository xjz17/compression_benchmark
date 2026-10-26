#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::svb {

// Modular delta-1 over the low/high uint32 limbs of an int64 column.
std::vector<std::uint8_t> delta_forward(std::span<const std::int64_t> values);
std::vector<std::uint8_t> delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> delta_inverse(
    std::span<const std::uint8_t> delta_packet);
std::vector<std::uint8_t> delta_inverse_scaled(
    std::span<const std::uint8_t> delta_packet);

// Stream-VByte code bytes for the already-delta-coded limbs.  The emitted
// public wire is byte-identical to int64_column_codec::stream_vbyte_encode_int64.
std::vector<std::uint8_t> integer_code_forward(
    std::span<const std::uint8_t> delta_packet);
std::vector<std::uint8_t> integer_code_inverse(
    std::span<const std::uint8_t> public_wire);

std::vector<std::uint8_t> compose(std::span<const std::int64_t> values);
std::vector<std::uint8_t> compose_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> decompose(
    std::span<const std::uint8_t> public_wire);
std::vector<std::uint8_t> decompose_scaled(
    std::span<const std::uint8_t> public_wire);

}  // namespace exact_actions::svb
