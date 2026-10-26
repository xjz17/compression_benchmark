#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace arithmetic_bytes {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> data);
std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded);

}  // namespace arithmetic_bytes

/** Static order-0 range coding (carry-free); CSV row key ``arithmetic``. See ArithmeticBytes.java. */
void testArithmeticCompression(const std::vector<char> &data, std::vector<long long> &result,
                               LossAccum *loss = nullptr);
