#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::lz4 {

std::vector<std::uint8_t> original_encode(std::span<const std::uint8_t> input);

std::vector<std::uint8_t> lz_match_forward(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> lz_match_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> token_format_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> token_format_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> integer_code_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> integer_code_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> container_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> container_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose(std::span<const std::uint8_t> input);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::lz4
