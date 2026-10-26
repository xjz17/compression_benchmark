#pragma once

#include "benchmark_loss.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace alp_float {

/** Same CSV aggregation as {@link org.example.AlpFloatBench#benchAlpFloat}. */
void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                               LossAccum *loss = nullptr);
void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                                LossAccum *loss = nullptr);
void benchInt64ColumnsLikeJava(const std::vector<std::vector<std::int64_t>> &columns,
                               std::vector<long long> &result, LossAccum *loss = nullptr);

/** Native ALP pipeline with one named operator bypassed and a matching decoder. */
void benchDoubleColumnsAblation(const std::vector<std::vector<double>> &columns,
                                std::string_view omitted_operator,
                                std::vector<long long> &result, LossAccum *loss = nullptr);
void benchInt64ColumnsAblation(const std::vector<std::vector<std::int64_t>> &columns,
                               std::string_view omitted_operator,
                               std::vector<long long> &result, LossAccum *loss = nullptr);

/** On-disk layout matching {@link AlpFloatGoldenExport}. */
std::vector<std::uint8_t> goldenSerializeColumn(const std::vector<float> &col);

/** Parse golden bytes into floats; returns false on corrupt input. */
bool goldenDeserializeColumn(std::span<const std::uint8_t> bytes, std::vector<float> &out);

/** Materialized lossless double-column wire used by benchmark_86_combinations.py. */
std::vector<std::uint8_t> encodeDoubleColumnExact(const std::vector<double> &values);
std::vector<double> decodeDoubleColumnExact(std::span<const std::uint8_t> wire);

} // namespace alp_float
