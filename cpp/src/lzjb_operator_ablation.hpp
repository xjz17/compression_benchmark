#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace lzjb_operator_ablation {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> input,
                                 std::string_view omitted_operator);
std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded,
                                 std::size_t original_size,
                                 std::string_view omitted_operator);

}  // namespace lzjb_operator_ablation
