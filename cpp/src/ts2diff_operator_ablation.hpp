#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ts2diff_operator_ablation {

std::vector<std::uint8_t> encode(std::span<const std::int64_t> values,
                                 std::string_view omitted_operator);
std::vector<std::int64_t> decode(std::span<const std::uint8_t> payload);

}  // namespace ts2diff_operator_ablation
