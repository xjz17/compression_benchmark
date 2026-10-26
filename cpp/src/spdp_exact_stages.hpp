#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::spdp {

std::vector<std::uint8_t> delta_forward(std::span<const std::uint8_t> raw);
std::vector<std::uint8_t> delta_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> block_transform_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> block_transform_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> entropy_forward(
    std::span<const std::uint8_t> packet, unsigned level = 5);
std::vector<std::uint8_t> entropy_inverse(
    std::span<const std::uint8_t> wire, std::size_t original_size,
    unsigned level = 5);
std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw,
                                  unsigned level = 5);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire,
                                    std::size_t original_size,
                                    unsigned level = 5);

}  // namespace exact_actions::spdp
