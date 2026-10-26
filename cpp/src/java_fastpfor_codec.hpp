#pragma once

#include "benchmark_loss.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace java_fastpfor {

/** Same stack as Java {@code new Composition(new FastPFOR(), new VariableByte())}. */
std::vector<std::uint32_t> compressIntBits(const std::int32_t *data, std::size_t n);

void uncompressIntBits(const std::uint32_t *compressed_words, std::size_t word_count,
                       std::int32_t *out, std::size_t n);

inline std::vector<std::int32_t> uncompressIntBits(const std::vector<std::uint32_t> &w,
                                                    std::size_t n) {
    std::vector<std::int32_t> out(n);
    uncompressIntBits(w.data(), w.size(), out.data(), n);
    return out;
}

/** Same CSV counters as Java `ExternalCompressionBench.benchJavaFastPFOR` (orig bytes, comp bytes, enc ns, dec ns). */
void benchIntColumnsLikeJava(const std::vector<std::vector<int>> &columns, std::vector<long long> &result,
                             LossAccum *loss = nullptr);

/** `floatToRawIntBits` per value, then same codec as the int path (`benchJavaFastPFORFloat`). */
void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns, std::vector<long long> &result,
                               LossAccum *loss = nullptr);
void benchDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns, std::vector<long long> &result,
                                LossAccum *loss = nullptr);
void benchInt64ColumnsLikeJava(const std::vector<std::vector<std::int64_t>> &columns,
                               std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace java_fastpfor
