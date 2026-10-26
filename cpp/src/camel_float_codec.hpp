#pragma once

#include "benchmark_loss.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace camel_float {

enum class AblationMode {
    none = 0,
    xor_transform = 1,
    xor_bit_reduction = 2,
    scaling = 3,
};

void setAblationMode(AblationMode mode);

/** Concatenated 1000-point Camel chunks; bytes match Java {@code parityCompressedChunksCamel} payloads. */
std::vector<std::uint8_t> encodeCamelFloatColumn(std::span<const float> col);

bool decodeCamelFloatChunk(std::span<const std::uint8_t> compressed, std::vector<double> &out, int expectedLen);

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                               LossAccum *loss = nullptr);

void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                              std::vector<long long> &result, LossAccum *loss = nullptr,
                              std::size_t *raw_fallback_columns = nullptr);

} // namespace camel_float
