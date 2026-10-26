#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::bos {

// Full BOS operator-ablation implementation: delta -> FOR -> outlier split
// -> second FOR -> bit packing.  The terminal wire is byte-identical to
// bos_operator_ablation::encode(values, "").
std::vector<std::uint8_t> delta_forward(std::span<const std::int64_t> values);
std::vector<std::int64_t> delta_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> first_for_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> first_for_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> outlier_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> outlier_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> second_for_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> second_for_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> bitpack_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> bitpack_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose(std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose(std::span<const std::uint8_t> wire);

// The BOS64 adapter used by scripts/benchmark_86_combinations.py has a
// different, smaller pipeline.  Keep its stages separate so exact benchmark
// parity is not inferred from the richer BOSA implementation above.
std::vector<std::uint8_t> adapter_plain_delta_forward(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> adapter_plain_delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> adapter_plain_delta_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_plain_delta_inverse_scaled(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_zigzag_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_zigzag_forward_scaled(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_zigzag_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_zigzag_inverse_scaled(
    std::span<const std::uint8_t> packet);
// Compatibility wrappers for the former fused delta -> ZigZag stage.
std::vector<std::uint8_t> adapter_delta_forward(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> adapter_delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> adapter_delta_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_delta_inverse_scaled(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_bitpack_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> adapter_bitpack_inverse(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> adapter_wire_forward_int(
    std::span<const std::uint8_t> delta_packet);
std::vector<std::uint8_t> adapter_wire_forward_scaled(
    std::span<const std::uint8_t> delta_packet);
std::vector<std::uint8_t> adapter_wire_inverse_int(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> adapter_wire_inverse_scaled(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> adapter_compose(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> adapter_decompose(
    std::span<const std::uint8_t> wire);

}  // namespace exact_actions::bos
