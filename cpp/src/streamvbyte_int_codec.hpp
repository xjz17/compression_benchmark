#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace streamvbyte_int {

/** Stream VByte (Lemire) with delta-1 on uint32-cast int32 columns (matches FastPFOR {@code svb_encode(...,1,1)}). */
void benchIntColumnsStreamVByteDelta1(const std::vector<std::vector<int>> &columns,
                                      std::vector<long long> &result, LossAccum *loss = nullptr);

void benchInt64ColumnsStreamVByteDelta1(const std::vector<std::vector<std::int64_t>> &columns,
                                        std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace streamvbyte_int
