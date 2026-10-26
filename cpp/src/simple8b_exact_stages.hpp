#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::simple8b {

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int max_point);
std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> bitpack_forward_int64(
    std::span<const std::int64_t> values);
std::vector<std::uint8_t> bitpack_forward_scaled(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::int64_t> bitpack_inverse_int64(
    std::span<const std::uint8_t> public_wire);
std::vector<std::uint8_t> bitpack_inverse_scaled(
    std::span<const std::uint8_t> public_wire);

std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int max_point);
std::vector<double> decompose_double(
    std::span<const std::uint8_t> public_wire);

}  // namespace exact_actions::simple8b
