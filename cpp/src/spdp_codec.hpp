#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace spdp_codec {

/** SPDP compression level range (upstream SPDP_11.c / Burtscher standalone). */
inline constexpr unsigned kMinLevel = 0;
inline constexpr unsigned kMaxLevel = 9;

/** Report row key, e.g. {@code spdp_level_5}. */
std::string rowName(unsigned level);

/** Levels to benchmark (always all 0..9; partial regen uses {@code --algorithms spdp_level_N}). */
std::vector<unsigned> levelsToBench();

/** Burtscher SPDP (DCC 2018) on raw float column bytes; {@code level} 0..9. */
void benchFloatColumnsSpdp(const std::vector<std::vector<float>> &columns, unsigned level,
                           std::vector<long long> &result, LossAccum *loss = nullptr);

/** Burtscher SPDP (DCC 2018) on raw double column bytes; {@code level} 0..9. */
void benchDoubleColumnsSpdp(const std::vector<std::vector<double>> &columns, unsigned level,
                            std::vector<long long> &result, LossAccum *loss = nullptr);
void benchInt64ColumnsSpdp(const std::vector<std::vector<std::int64_t>> &columns, unsigned level,
                           std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace spdp_codec
