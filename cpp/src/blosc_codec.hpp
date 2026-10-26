#pragma once

#include "benchmark_loss.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace wc_blosc1_bench {

inline constexpr const char *kRowBlosclz = "blosc_blosclz";

/** Returns false when {@code COMPRESSION_BENCHMARK_BLOSC_BENCH=0}. */
bool bench_enabled();

/** Canonical one-buffer Blosc1/BLOSCLZ wire used by the C++ benchmark. */
std::vector<std::uint8_t> encodeBufferExact(
    std::span<const std::uint8_t> input, int typesize);
std::vector<std::uint8_t> decodeBufferExact(
    std::span<const std::uint8_t> encoded);

void benchFloatColumns(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                       LossAccum *loss = nullptr);
void benchInt64Columns(const std::vector<std::vector<std::int64_t>> &columns, std::vector<long long> &result,
                       LossAccum *loss = nullptr);
void benchByteColumns(const std::vector<std::vector<char>> &columns, std::vector<long long> &result,
                      LossAccum *loss = nullptr);
void benchBufferOperatorAblation(const std::vector<char> &data, int typesize,
                                 std::string_view omitted_operator,
                                 std::vector<long long> &result,
                                 LossAccum *loss = nullptr);

} // namespace wc_blosc1_bench
