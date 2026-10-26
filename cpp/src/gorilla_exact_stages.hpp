#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::gorilla {

std::vector<std::uint8_t> xor_forward_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> xor_forward_double_bits(std::span<const std::int64_t> values);
std::vector<std::int64_t> xor_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> trailing_pack_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> trailing_pack_inverse_int64(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> trailing_pack_inverse_double(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose_int64(std::span<const std::int64_t> values);
std::vector<std::uint8_t> compose_double_bits(std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose_int64(std::span<const std::uint8_t> wire);
std::vector<std::int64_t> decompose_double_bits(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::gorilla
