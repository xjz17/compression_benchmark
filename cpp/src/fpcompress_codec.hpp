#pragma once

#include "benchmark_loss.hpp"

#include <vector>

namespace fpcompress_codec {

/** True when FPCompress CPU exes are available (or {@code COMPRESSION_BENCHMARK_FPCOMPRESS_BENCH=1}). */
bool bench_enabled();

/** Lossless LC-framework CPU algorithms on float64 column bytes (DPratio / DPspeed). */
void benchDoubleColumnsDpratio(const std::vector<std::vector<double>> &columns,
                               std::vector<long long> &result, LossAccum *loss = nullptr);
void benchDoubleColumnsDpspeed(const std::vector<std::vector<double>> &columns,
                               std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace fpcompress_codec
