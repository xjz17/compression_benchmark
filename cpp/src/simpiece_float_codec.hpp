#pragma once

#include "benchmark_loss.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace simpiece_float {

/**
 * Sim-Piece lossy float column: {@code epsilon = (max-min)*rel_range_frac}, or 1.0 when range invalid
 * (same as {@link ExternalCompressionBench#benchSimPieceFloat}).
 */
std::vector<std::uint8_t> compressColumnRelRange(std::span<const float> col, double rel_range_frac);

/** Round-trip packed blob and check |restored[i]-original[i]| ≤ epsilon·1.0001 + 1e-9 (Java bench). */
bool verifyDecompressTolerance(std::span<const std::uint8_t> packed, std::span<const float> col,
                               double rel_range_frac);

void benchFloatColumnsRelRange(const std::vector<std::vector<float>> &columns, double rel_range_frac,
                               std::vector<long long> &result, LossAccum *loss = nullptr);
void benchDoubleColumnsRelRange(const std::vector<std::vector<double>> &columns, double rel_range_frac,
                                std::vector<long long> &result, LossAccum *loss = nullptr);
void benchInt64ColumnsRelRange(const std::vector<std::vector<std::int64_t>> &columns, double rel_range_frac,
                               std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace simpiece_float
