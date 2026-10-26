#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace bos_operator_ablation {

std::vector<std::uint8_t> encode(std::span<const std::int64_t> values,
                                 std::string_view omitted_operator);
std::vector<std::int64_t> decode(std::span<const std::uint8_t> encoded);

}  // namespace bos_operator_ablation
