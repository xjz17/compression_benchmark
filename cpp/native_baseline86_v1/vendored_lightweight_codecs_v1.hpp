#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace native_baseline86_v1::vendored_lightweight {

using Bytes = std::vector<std::uint8_t>;

// The adapter wire starts with one byte: 0 is a verbatim fallback and 1 is
// the upstream codec's byte stream.  The choice is fixed before the corpus is
// observed: keep upstream output only when it is smaller than the input.
Bytes encode(const std::string& label, std::span<const std::uint8_t> input);
Bytes decode(const std::string& label, std::span<const std::uint8_t> payload,
             std::size_t expected_size);

}  // namespace native_baseline86_v1::vendored_lightweight
