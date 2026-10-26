#pragma once

#include "benchmark_loss.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace simple8b_int {

/** Same CSV counters as Java {@code Simple8bIntColumn.benchSimple8bInt}. */
void benchIntColumnsSimple8b(const std::vector<std::vector<int>> &columns, std::vector<long long> &result,
                             LossAccum *loss = nullptr);

/** Float columns: decimal-scaled int32 values, then the same Simple8b codec as the int path. */
void benchFloatColumnsSimple8b(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                               LossAccum *loss = nullptr);

/** Double columns: decimal-scaled int64 values, then the same Simple8b codec as the int64 path. */
void benchDoubleColumnsSimple8b(const std::vector<std::vector<double>> &columns,
                                std::vector<long long> &result,
                                const std::vector<int> *max_point_per_column = nullptr,
                                LossAccum *loss = nullptr);
void benchInt64ColumnsSimple8b(const std::vector<std::vector<std::int64_t>> &columns,
                               std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace simple8b_int
