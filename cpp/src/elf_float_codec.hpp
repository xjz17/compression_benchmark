#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace elf_float {

std::vector<std::uint8_t> encodeElfOnChimpFloatColumn(std::span<const float> col);
std::vector<std::uint8_t> encodeElfPlusFloatColumn(std::span<const float> col);

/** One 2048-or-fewer float chunk (matches a single Java compressor session). */
bool decodeElfOnChimpFloatChunk(std::span<const std::uint8_t> bytes, std::vector<float> &out, int expected_n);
bool decodeElfPlusFloatChunk(std::span<const std::uint8_t> bytes, std::vector<float> &out, int expected_n);

void benchElfOnChimpFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                                         std::vector<long long> &result);
void benchElfPlusFloatColumnsLikeJava(const std::vector<std::vector<float>> &columns,
                                      std::vector<long long> &result);

} // namespace elf_float

namespace exact_actions::elf {

// Input/output values are raw float64 bit patterns.  The supported domain is
// the benchmark's lossless Elf32 domain: every value is finite and its float64
// bit pattern must survive the float32 cast and promotion bit-for-bit.
std::vector<std::uint8_t> eraser_forward_double_bits(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> eraser_inverse_double_bits(
    std::span<const std::uint8_t> packet);

std::vector<std::uint8_t> elf_xor_forward(
    std::span<const std::uint8_t> eraser_packet);
std::vector<std::uint8_t> elf_xor_inverse(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> elf_plus_xor_forward(
    std::span<const std::uint8_t> eraser_packet);
std::vector<std::uint8_t> elf_plus_xor_inverse(
    std::span<const std::uint8_t> wire);

std::vector<std::uint8_t> compose_elf(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose_elf(
    std::span<const std::uint8_t> wire);
std::vector<std::uint8_t> compose_elf_plus(
    std::span<const std::int64_t> values);
std::vector<std::int64_t> decompose_elf_plus(
    std::span<const std::uint8_t> wire);

}  // namespace exact_actions::elf
