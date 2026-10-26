#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace ts2diff_bos_native {

std::vector<std::uint8_t> encode_int64_column_payload(const std::vector<std::int64_t> &values);
std::vector<std::uint8_t> encode_int64_column_payload_ablation(
    const std::vector<std::int64_t> &values, std::string_view omitted_operator);
std::vector<std::int64_t> decode_int64_column_payload(const std::uint8_t *data, std::size_t len);

void benchInt64ColumnsBos(const std::vector<std::vector<std::int64_t>> &columns,
                          std::vector<long long> &result, LossAccum *loss = nullptr);

void benchDoubleColumnsBos(const std::vector<std::vector<double>> &columns,
                           const std::vector<int> *max_point_per_column,
                           std::vector<long long> &result, LossAccum *loss = nullptr);

void benchInt64ColumnsBosLevel(const std::vector<std::vector<std::int64_t>> &columns,
                               int level,
                               std::vector<long long> &result,
                               LossAccum *loss = nullptr);

void benchDoubleColumnsBosLevel(const std::vector<std::vector<double>> &columns,
                                const std::vector<int> *max_point_per_column,
                                int level,
                                std::vector<long long> &result,
                                LossAccum *loss = nullptr);

} // namespace ts2diff_bos_native
