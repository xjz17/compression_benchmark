#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace corad_tricklet_float {

/** Legacy C++ CORAD benchmark is disabled: its normalized wire is not lossless.
 * These production entrypoints return zero counters, so callers do not publish
 * a false lossless row. The versioned CRD2 baseline lives in the Python runtime.
 */
void benchFloatColumnsCoradTricklet(const std::vector<std::vector<float>> &columns,
                                    std::vector<long long> &result, LossAccum *loss = nullptr);
void benchDoubleColumnsCoradTricklet(const std::vector<std::vector<double>> &columns,
                                     std::vector<long long> &result, LossAccum *loss = nullptr);
void benchInt64ColumnsCoradTricklet(const std::vector<std::vector<std::int64_t>> &columns,
                                    std::vector<long long> &result, LossAccum *loss = nullptr);

/** For forensic tests only. Never use this lossy legacy result in a leaderboard. */
void benchDoubleColumnsCoradTrickletLegacyForensics(
    const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
    LossAccum *loss = nullptr);

} // namespace corad_tricklet_float
