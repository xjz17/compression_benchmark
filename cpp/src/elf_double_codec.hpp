#pragma once

#include "benchmark_loss.hpp"

#include <string_view>
#include <span>
#include <cstdint>
#include <vector>

namespace elf_double {

// EL64 uses the existing double eraser and distinct Chimp/ElfXor backends.
// NaNs are rejected because their canonical value is the original EOS marker.
std::vector<std::uint8_t> encodeDoubleColumnExact(std::span<const std::uint8_t> raw, bool plus);
std::vector<std::uint8_t> decodeDoubleColumnExact(std::span<const std::uint8_t> wire, bool plus);
std::vector<std::uint8_t> eraser64Forward(std::span<const std::uint8_t> raw);
std::vector<std::uint8_t> eraser64Inverse(std::span<const std::uint8_t> packet);
std::vector<std::uint8_t> xor64Forward(std::span<const std::uint8_t> packet, bool plus);
std::vector<std::uint8_t> xor64Inverse(std::span<const std::uint8_t> wire, bool plus);

void benchChimpDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                     std::vector<long long> &result,
                                     LossAccum *loss = nullptr);
void benchChimpDoubleColumnsAblation(
    const std::vector<std::vector<double>> &columns, std::string_view omitted_operator,
    std::vector<long long> &result, LossAccum *loss = nullptr);
void benchElfOnChimpDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                          std::vector<long long> &result,
                                          LossAccum *loss = nullptr);
void benchElfPlusDoubleColumnsLikeJava(const std::vector<std::vector<double>> &columns,
                                       std::vector<long long> &result,
                                       LossAccum *loss = nullptr);
void benchElfOnChimpDoubleColumnsAblation(
    const std::vector<std::vector<double>> &columns, std::string_view omitted_operator,
    std::vector<long long> &result, LossAccum *loss = nullptr);
void benchElfPlusDoubleColumnsAblation(
    const std::vector<std::vector<double>> &columns, std::string_view omitted_operator,
    std::vector<long long> &result, LossAccum *loss = nullptr);

} // namespace elf_double
