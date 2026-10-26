#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace ans_rans {

std::vector<std::uint8_t> encode(std::span<const std::uint8_t> data);
std::vector<std::uint8_t> decode(std::span<const std::uint8_t> encoded);

}  // namespace ans_rans

void testAnsEntropyCompression(const std::vector<char> &data,
                               std::vector<long long> &result,
                               LossAccum *loss = nullptr);
