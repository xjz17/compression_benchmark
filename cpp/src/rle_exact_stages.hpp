#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::rle {

std::vector<std::uint8_t> scaling_forward(std::span<const double> values,
                                          int requested_max_point);
std::vector<double> scaling_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> run_length_forward(
    std::span<const std::uint8_t> scaling_packet);
std::vector<std::uint8_t> run_length_inverse(
    std::span<const std::uint8_t> public_wire);
std::vector<std::uint8_t> compose_double(std::span<const double> values,
                                         int requested_max_point);
std::vector<double> decompose_double(std::span<const std::uint8_t> public_wire);

}  // namespace exact_actions::rle

