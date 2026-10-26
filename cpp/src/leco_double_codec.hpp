#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace leco_double {

void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                std::vector<long long> &result,
                                const std::vector<int> *max_point_per_column = nullptr,
                                LossAccum *loss = nullptr);

int maxLosslessDecimalScale(const std::vector<double> &col);

bool quantizeScaledInts(const std::vector<double> &col, int p, std::vector<std::int64_t> &out);

bool quantizeScaledIntsWithMaxPointFallback(const std::vector<double> &col,
                                            int max_point,
                                            std::vector<std::int64_t> &out,
                                            int *used_scale = nullptr);

} // namespace leco_double
