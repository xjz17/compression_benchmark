#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace reger_codec {

void setAblatedOperatorForTesting(std::string_view omitted_operator);

std::vector<std::uint8_t> encodeInt64Column(const std::vector<std::int64_t>& values);
std::vector<std::int64_t> decodeInt64Column(const std::vector<std::uint8_t>& payload);

std::vector<std::uint8_t> encodeDoubleColumn(const std::vector<double>& values);
std::vector<double> decodeDoubleColumn(const std::vector<std::uint8_t>& payload);

std::vector<std::uint8_t> encodeInt64Columns(
    const std::vector<std::vector<std::int64_t>>& columns);
std::vector<std::uint8_t> encodeInt64Columns(
    const std::vector<std::vector<std::int64_t>>& columns,
    const std::vector<std::int64_t>* times);
std::vector<std::vector<std::int64_t>> decodeInt64Columns(
    const std::vector<std::uint8_t>& payload);

std::vector<std::uint8_t> encodeDoubleColumns(const std::vector<std::vector<double>>& columns);
std::vector<std::uint8_t> encodeDoubleColumns(const std::vector<std::vector<double>>& columns,
                                              const std::vector<std::int64_t>* times);
std::vector<std::vector<double>> decodeDoubleColumns(const std::vector<std::uint8_t>& payload);

void benchInt64ColumnsReger(const std::vector<std::vector<std::int64_t>>& columns,
                            std::vector<long long>& result,
                            LossAccum* loss = nullptr);
void benchInt64ColumnsReger(const std::vector<std::vector<std::int64_t>>& columns,
                            std::vector<long long>& result,
                            LossAccum* loss,
                            const std::vector<std::int64_t>* times);

void benchDoubleColumnsReger(const std::vector<std::vector<double>>& columns,
                             std::vector<long long>& result,
                             LossAccum* loss = nullptr);
void benchDoubleColumnsReger(const std::vector<std::vector<double>>& columns,
                             std::vector<long long>& result,
                             LossAccum* loss,
                             const std::vector<std::int64_t>* times);

}  // namespace reger_codec
