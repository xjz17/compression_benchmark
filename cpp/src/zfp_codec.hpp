#pragma once

#include "benchmark_loss.hpp"

#include <vector>

namespace zfp_codec {

inline constexpr const char *kRowReversible = "ext_zfp_zfpy";

/** Returns false when {@code WEB_COMPRESSION_ZFP_BENCH=0}. */
bool bench_enabled();

void benchReversibleDoubleColumns(const std::vector<std::vector<double>> &columns,
                                  std::vector<long long> &result, LossAccum *loss = nullptr);

void benchToleranceDoubleColumns(const std::vector<std::vector<double>> &columns, double tolerance,
                                 const char *row_name, std::vector<long long> &result,
                                 LossAccum *loss = nullptr);

} // namespace zfp_codec
