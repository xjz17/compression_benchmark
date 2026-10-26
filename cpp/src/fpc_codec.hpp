#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fpc_codec {

/** Report CSV row key. */
inline constexpr const char *kRowName = "fpc";

/** Default FCM/dFCM table size exponent ({@code 1 << lg2} entries). Override with {@code WEB_COMPRESSION_FPC_PRED_SIZE_LG2}. */
inline constexpr unsigned kDefaultPredSizeLg2 = 16;

unsigned predSizeLg2ToBench();

/** Burtscher FPC on 64-bit double column bytes. */
void benchDoubleColumnsFpc(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                           LossAccum *loss = nullptr);

/** Burtscher FPC on 64-bit integer column bytes. */
void benchInt64ColumnsFpc(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                          LossAccum *loss = nullptr);

void benchDoubleColumnsFpcAblation(const std::vector<std::vector<double>> &columns,
                                   std::string_view omitted_operator,
                                   std::vector<long long> &result,
                                   LossAccum *loss = nullptr);
void benchInt64ColumnsFpcAblation(const std::vector<std::vector<std::int64_t>> &columns,
                                  std::string_view omitted_operator,
                                  std::vector<long long> &result,
                                  LossAccum *loss = nullptr);

} // namespace fpc_codec
