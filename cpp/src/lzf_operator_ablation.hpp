#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace lzf_operator_ablation {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> input, int level,
                                 std::string_view omitted_operator);

std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded,
                                 std::size_t expected_size, int level,
                                 std::string_view omitted_operator);

}  // namespace lzf_operator_ablation
