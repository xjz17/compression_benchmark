#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace leco_float {

/** Same CSV aggregation as {@code ExternalCompressionBench.benchLeCoFloatScaledIntegersFastPFor}. */
void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns, std::vector<long long> &result);

/** Max decimal scale {@code p} for the column (lossless), or {@code -1}. */
int maxLosslessDecimalScale(const std::vector<float> &col);

/** Quantize with fixed {@code p}; returns false if not bit-identical round-trip per value. */
bool quantizeScaledInts(const std::vector<float> &col, int p, std::vector<std::int32_t> &out);

} // namespace leco_float
