#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace native_baseline86_v1::vendored_entropy {
using Bytes = std::vector<std::uint8_t>;
Bytes encode(const std::string& label, std::span<const std::uint8_t> input);
Bytes decode(const std::string& label, std::span<const std::uint8_t> payload,
             std::size_t expected_size);
}  // namespace native_baseline86_v1::vendored_entropy
