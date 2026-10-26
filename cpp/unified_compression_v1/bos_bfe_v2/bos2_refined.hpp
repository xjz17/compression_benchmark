#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace unified_compression_v1::bos_bfe_v2::integer {

// Version-isolated Project-150 BOS2 integer kernel.  Keeping this namespace
// separate lets the unified harness encode new BOS2 frames while retaining
// native_baseline86_v1's older BOS64 decoder for historical UCC1 frames.
std::vector<std::uint8_t> optimized_compose(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> optimized_decompose(
    std::span<const std::uint8_t> wire);

}  // namespace unified_compression_v1::bos_bfe_v2::integer
