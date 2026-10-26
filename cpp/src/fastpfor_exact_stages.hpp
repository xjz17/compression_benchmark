#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::fastpfor {

// The source CompositeCodec is split at its real block boundary: FastPFor<8>
// owns the floor(n / 256) * 256 prefix and VariableByte owns the tail.  The
// PFS1 packet keeps the encoded PFor prefix plus the still-raw tail so that
// both stages are independently reversible.
std::vector<std::uint8_t> pfor_stage_forward_int(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> pfor_stage_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> pfor_stage_inverse_int(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> pfor_stage_inverse_scaled(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> variable_byte_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> variable_byte_inverse(
    std::span<const std::uint8_t> wire, std::size_t expected_count);

// Whole CompositeCodec helpers are retained only as an excluded reference
// path. They are not registered as policy actions.
std::vector<std::uint8_t> adapter_forward_int(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> adapter_inverse_int(
    std::span<const std::uint8_t> wire, std::size_t expected_count);
std::vector<std::uint8_t> adapter_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::uint8_t> adapter_inverse_scaled(
    std::span<const std::uint8_t> wire, std::size_t expected_count);

}  // namespace exact_actions::fastpfor
