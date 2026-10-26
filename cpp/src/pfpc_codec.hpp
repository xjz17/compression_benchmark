#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace pfpc_codec {

/** Report CSV row key. */
inline constexpr const char *kRowName = "pfpc";

/** Default FCM/dFCM table size exponent ({@code 1 << lg2} entries). */
inline constexpr unsigned kDefaultPredSizeLg2 = 20;
inline constexpr unsigned kDefaultThreads = 8;
inline constexpr int kDefaultChunkSize = 2048;

unsigned predSizeLg2ToBench();
unsigned threadsToBench();
int chunkSizeToBench();

void benchDoubleColumnsPfpc(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                            LossAccum *loss = nullptr);

void benchInt64ColumnsPfpc(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                           LossAccum *loss = nullptr);

} // namespace pfpc_codec
