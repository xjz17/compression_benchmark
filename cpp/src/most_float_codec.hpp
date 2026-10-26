#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace most_float {

void benchFloatColumnsMostMinimal(const std::vector<std::vector<float>> &columns,
                                  std::vector<long long> &result, LossAccum *loss = nullptr);
void benchDoubleColumnsMostMinimal(const std::vector<std::vector<double>> &columns,
                                   std::vector<long long> &result, LossAccum *loss = nullptr);
void benchInt64ColumnsMostMinimal(const std::vector<std::vector<std::int64_t>> &columns,
                                  std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace most_float
