#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace fpzip_codec {

/** fpzip lossless compress/decompress; int columns use cast-to-double values (not int bits). */
void benchFloatColumns(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                       LossAccum *loss = nullptr);
void benchDoubleColumns(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                        LossAccum *loss = nullptr);
void benchInt64Columns(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                       LossAccum *loss = nullptr);

} // namespace fpzip_codec
