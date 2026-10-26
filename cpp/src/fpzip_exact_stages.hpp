#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace exact_actions::fpzip {

// Lossless 1D double path used by the benchmark: the multidimensional Lorenzo
// predictor degenerates to previous-value prediction along x.
std::vector<std::uint8_t> predictor_forward(std::span<const std::uint8_t> raw_f64);
std::vector<std::uint8_t> predictor_inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> range_forward(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> range_inverse(std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose(std::span<const std::uint8_t> raw_f64);
std::vector<std::uint8_t> decompose(std::span<const std::uint8_t> wire);

}  // namespace exact_actions::fpzip
