#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace chimp32_paper {

/**
 * Paper Chimp32 over IEEE floats (`gr.aueb.delorean.chimp.Chimp32`); compressed length uses Java
 * `getSize()` bit count (see `ExternalCompressionBench.benchChimp32Paper`), not trailing flush bits.
 */
std::vector<std::uint8_t> encodeFloatBitsColumn(const float *values, std::size_t n);

/** Int column as Java bench: `Float.intBitsToFloat` per element before Chimp32. */
std::vector<std::uint8_t> encodeIntColumnAsFloatBits(const std::int32_t *ints, std::size_t n);

void decodeFloatBitsColumn(std::span<const std::uint8_t> compressed, float *out, std::size_t n);
void decodeIntColumnAsFloatBits(std::span<const std::uint8_t> compressed, std::int32_t *out,
                                  std::size_t n);

void benchFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                               std::vector<long long> &result);
void benchIntColumnsLikeJava(const std::vector<std::vector<int>> &columns, std::vector<long long> &result);

} // namespace chimp32_paper
