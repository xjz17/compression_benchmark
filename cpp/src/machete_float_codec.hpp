#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace machete_float {

void benchFloatColumnsMacheteRel(const std::vector<std::vector<float>> &columns, double relError,
                                 std::vector<long long> &result, LossAccum *loss = nullptr);
void benchDoubleColumnsMacheteRel(const std::vector<std::vector<double>> &columns, double relError,
                                  std::vector<long long> &result, LossAccum *loss = nullptr);
void benchInt64ColumnsMacheteRel(const std::vector<std::vector<std::int64_t>> &columns, double relError,
                                 std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace machete_float
