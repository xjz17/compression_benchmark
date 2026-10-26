#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::p04 {

std::vector<std::uint8_t> rlbe_delta_forward(std::span<const std::int64_t> values);
std::vector<std::int64_t> rlbe_delta_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> rlbe_run_length_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> rlbe_run_length_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> rlbe_compose(std::span<const std::int64_t> values);
std::vector<std::int64_t> rlbe_decompose(std::span<const std::uint8_t> wire);

std::vector<std::uint8_t> sprintz_predictor_forward(std::span<const std::int64_t> values);
std::vector<std::uint8_t> sprintz_predictor_forward_double_bits(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> sprintz_predictor_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> sprintz_zigzag_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> sprintz_zigzag_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> sprintz_bitpack_forward(std::span<const std::uint8_t> packet);
// Current profile: same Sprintz groups, integer tail is a versioned URL1 packet.
std::vector<std::uint8_t> sprintz_bitpack_forward_current(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> sprintz_bitpack_inverse_current(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> sprintz_bitpack_inverse_double_current(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> sprintz_bitpack_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> sprintz_bitpack_inverse_double(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> sprintz_compose(std::span<const std::int64_t> values);
std::vector<std::uint8_t> sprintz_compose_double_bits(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> sprintz_decompose(std::span<const std::uint8_t> wire);
std::vector<std::int64_t> sprintz_decompose_double_bits(
    std::span<const std::uint8_t> wire);

}  // namespace exact_actions::p04
