#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace benchmark_wire {

inline constexpr std::size_t kIntBytes = 8;
inline constexpr std::size_t kFloatBytes = 8;

inline void append_int64_le(std::vector<std::uint8_t> &out, std::int64_t v) {
    for (int i = 0; i < 8; i++) {
        out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
    }
}

inline void append_double_le(std::vector<std::uint8_t> &out, double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    for (int i = 0; i < 8; i++) {
        out.push_back(static_cast<std::uint8_t>((bits >> (8 * i)) & 0xFF));
    }
}

inline std::size_t column_original_size(std::size_t n, bool is_float) {
    return n * (is_float ? kFloatBytes : kIntBytes);
}

} // namespace benchmark_wire
