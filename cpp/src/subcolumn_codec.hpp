#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <vector>

namespace subcolumn_codec {

std::vector<std::uint8_t> encode_int64_column(const std::vector<std::int64_t> &values);
std::vector<std::int64_t> decode_int64_column(const std::uint8_t *data, std::size_t len);

void benchInt64ColumnsSubcolumn(const std::vector<std::vector<std::int64_t>> &columns,
                                std::vector<long long> &result, LossAccum *loss = nullptr);

void benchDoubleColumnsSubcolumn(const std::vector<std::vector<double>> &columns,
                                 std::vector<long long> &result,
                                 const std::vector<int> *max_point_per_column = nullptr,
                                 LossAccum *loss = nullptr);

} // namespace subcolumn_codec
