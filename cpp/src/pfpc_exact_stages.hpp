#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::pfpc {

std::vector<std::uint8_t> predictor_xor_forward(
    std::span<const std::uint8_t> raw, unsigned predictor_lg2 = 16);
std::vector<std::uint8_t> predictor_xor_inverse(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> trailing_pack_forward(
    std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> trailing_pack_inverse(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw,
                                  unsigned predictor_lg2 = 16);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::pfpc
