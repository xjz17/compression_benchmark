#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::ts2diff {

// Each forward function performs one real TS_2DIFF data-flow stage.
// Intermediate packets are private, reversible wires; only bitpack_forward()
// emits the public TSFile TS_2DIFF wire.
std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int max_point);
std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> delta_forward_int64(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> delta_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> delta_inverse_int64(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> delta_inverse_scaled(
    std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> frame_of_reference_forward(
    std::span<const std::uint8_t> delta_packet);
std::vector<std::uint8_t> frame_of_reference_inverse(
    std::span<const std::uint8_t> for_packet);

std::vector<std::uint8_t> bitpack_forward(
    std::span<const std::uint8_t> for_packet);
std::vector<std::uint8_t> bitpack_inverse(
    std::span<const std::uint8_t> public_wire, bool floating_point,
    int max_point = 0);

std::vector<std::uint8_t> compose_int64(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int max_point);
std::vector<std::int64_t> decompose_int64(
    std::span<const std::uint8_t> public_wire);
std::vector<double> decompose_double(
    std::span<const std::uint8_t> public_wire, int max_point);

}  // namespace exact_actions::ts2diff
